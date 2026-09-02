"""qwen3_moe block-fp8 checkpoints (Qwen3-30B-A3B-FP8 layout): config detection, the
dense loader (q|k|v fusion per kind), the resident expert banks and the offload hook."""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch

FP8 = torch.float8_e4m3fn
H, MOE_I, E, L = 256, 128, 2, 2
Q_DIM, KV_DIM = 2 * 128, 1 * 128


def _hf_config(*, fp8: bool = True):
    from transformers import Qwen3MoeConfig

    cfg = Qwen3MoeConfig(
        hidden_size=H,
        intermediate_size=512,
        moe_intermediate_size=MOE_I,
        num_experts=E,
        num_experts_per_tok=1,
        num_hidden_layers=L,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        vocab_size=64,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        norm_topk_prob=True,
        tie_word_embeddings=False,
        architectures=["Qwen3MoeForCausalLM"],
    )
    if fp8:
        cfg.quantization_config = {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["lm_head"],
        }
    return cfg


def _fp8_pair(out_f: int, in_f: int) -> tuple[torch.Tensor, torch.Tensor]:
    w = torch.randn(out_f, in_f).to(FP8)
    s = torch.rand(out_f // 128, in_f // 128, dtype=torch.bfloat16) + 0.5
    return w, s


def _fp8_checkpoint_tensors() -> dict[str, torch.Tensor]:
    """Real Qwen3-30B-A3B-FP8 key layout: fp8 attention + per-expert fp8 pairs, bf16 rest."""
    t = {
        "model.embed_tokens.weight": torch.randn(64, H, dtype=torch.bfloat16),
        "model.norm.weight": torch.randn(H, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(64, H, dtype=torch.bfloat16),
    }
    for li in range(L):
        lp = f"model.layers.{li}."
        for proj, (o, i) in {
            "q_proj": (Q_DIM, H), "k_proj": (KV_DIM, H), "v_proj": (KV_DIM, H), "o_proj": (H, Q_DIM),
        }.items():
            w, s = _fp8_pair(o, i)
            t[lp + f"self_attn.{proj}.weight"] = w
            t[lp + f"self_attn.{proj}.weight_scale_inv"] = s
        t[lp + "self_attn.q_norm.weight"] = torch.randn(128, dtype=torch.bfloat16)
        t[lp + "self_attn.k_norm.weight"] = torch.randn(128, dtype=torch.bfloat16)
        t[lp + "input_layernorm.weight"] = torch.randn(H, dtype=torch.bfloat16)
        t[lp + "post_attention_layernorm.weight"] = torch.randn(H, dtype=torch.bfloat16)
        t[lp + "mlp.gate.weight"] = torch.randn(E, H, dtype=torch.bfloat16)
        for e in range(E):
            for proj, (o, i) in {"gate_proj": (MOE_I, H), "up_proj": (MOE_I, H), "down_proj": (H, MOE_I)}.items():
                w, s = _fp8_pair(o, i)
                t[lp + f"mlp.experts.{e}.{proj}.weight"] = w
                t[lp + f"mlp.experts.{e}.{proj}.weight_scale_inv"] = s
    return t


def _write_checkpoint(tmp_path, tensors: dict, hf) -> str:
    import safetensors.torch

    shard = "model-00001-of-00001.safetensors"
    safetensors.torch.save_file(tensors, str(tmp_path / shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}})
    )
    (tmp_path / "config.json").write_text(json.dumps(hf.to_dict()))
    return str(tmp_path)


@pytest.fixture
def tp1():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


@pytest.fixture
def fp8_checkpoint(tmp_path, monkeypatch):
    hf = _hf_config()
    tensors = _fp8_checkpoint_tensors()
    path = _write_checkpoint(tmp_path, tensors, hf)
    import freetoken.models.qwen3_moe.weight as w

    monkeypatch.setattr(w, "cached_load_hf_config", lambda _p: hf)
    return path, hf, tensors


def test_parse_config_detects_block_fp8():
    from freetoken.models.qwen3_moe.config import parse_config

    cfg = parse_config(_hf_config())
    assert cfg.expert_quant == "fp8_block"
    assert cfg.weight_block_size == (128, 128)
    bf16 = parse_config(_hf_config(fp8=False))
    assert bf16.expert_quant == "none" and bf16.weight_block_size is None


def test_fp8_model_uses_block_fp8_linears(tp1):
    from freetoken.kernel.triton.fp8_block_linear import Fp8BlockColMerged, Fp8BlockLinear
    from freetoken.layers import LinearOProj, LinearQKVMerged
    from freetoken.models.qwen3_moe.config import parse_config
    from freetoken.models.qwen3_moe.model import Qwen3MoeForCausalLM

    attn = Qwen3MoeForCausalLM(parse_config(_hf_config())).model.layers.op_list[0].self_attn
    assert isinstance(attn.qkv_proj, Fp8BlockColMerged) and isinstance(attn.o_proj, Fp8BlockLinear)
    assert attn.qkv_proj.weight.shape == (Q_DIM + 2 * KV_DIM, H)
    bf16 = Qwen3MoeForCausalLM(parse_config(_hf_config(fp8=False))).model.layers.op_list[0].self_attn
    assert isinstance(bf16.qkv_proj, LinearQKVMerged) and isinstance(bf16.o_proj, LinearOProj)


def test_iter_weights_fp8_matches_resident_model_state_dict(tp1, fp8_checkpoint):
    """--moe-backend fused: dense weights + stacked fp8 experts, exactly the model's keys."""
    from freetoken.models.qwen3_moe.config import parse_config
    from freetoken.models.qwen3_moe.model import Qwen3MoeForCausalLM
    from freetoken.models.qwen3_moe.weight import iter_weights

    path, hf, tensors = fp8_checkpoint
    loaded = dict(iter_weights(path, torch.device("cpu"), include_moe_experts=True, include_non_moe=True))
    expected = Qwen3MoeForCausalLM(parse_config(hf)).state_dict()
    assert set(loaded) == set(expected)
    for k in expected:
        assert loaded[k].shape == expected[k].shape, k
        # fp8 weights and their bf16 block scales pass through verbatim (the engine's
        # load-time cast to the model buffer dtype is a no-op on them).
        if expected[k].dtype == FP8 or k.endswith("_scale_inv"):
            assert loaded[k].dtype == expected[k].dtype, k

    p = "model.layers.1.self_attn."
    for kind in ("weight", "weight_scale_inv"):
        fused = loaded[p + f"qkv_proj.{kind}"]
        parts = [tensors[p + f"{x}_proj.{kind}"] for x in ("q", "k", "v")]
        assert torch.equal(fused.view(torch.uint8), torch.cat(parts).view(torch.uint8))
    assert torch.equal(loaded[p + "o_proj.weight_scale_inv"], tensors[p + "o_proj.weight_scale_inv"])

    ep = "model.layers.1.mlp.experts."
    gate_up, gate_up_s = loaded[ep + "gate_up_proj"], loaded[ep + "gate_up_scale_inv"]
    down, down_s = loaded[ep + "down_proj"], loaded[ep + "down_scale_inv"]
    assert gate_up.shape == (E, 2 * MOE_I, H) and gate_up_s.shape == (E, 2 * MOE_I // 128, H // 128)
    assert down.shape == (E, H, MOE_I) and down_s.shape == (E, H // 128, MOE_I // 128)
    for e in range(E):
        def src(proj: str, kind: str, e: int = e) -> torch.Tensor:
            return tensors[ep + f"{e}.{proj}.{kind}"]

        assert torch.equal(gate_up[e, :MOE_I].view(torch.uint8), src("gate_proj", "weight").view(torch.uint8))
        assert torch.equal(gate_up[e, MOE_I:].view(torch.uint8), src("up_proj", "weight").view(torch.uint8))
        assert torch.equal(down[e].view(torch.uint8), src("down_proj", "weight").view(torch.uint8))
        assert torch.equal(gate_up_s[e, : MOE_I // 128], src("gate_proj", "weight_scale_inv"))
        assert torch.equal(gate_up_s[e, MOE_I // 128 :], src("up_proj", "weight_scale_inv"))
        assert torch.equal(down_s[e], src("down_proj", "weight_scale_inv"))


def test_iter_weights_fp8_offload_excludes_experts(tp1, fp8_checkpoint):
    """--moe-backend offload: the dense loader yields exactly the offload model's keys."""
    from freetoken.models.qwen3_moe.config import parse_config
    from freetoken.models.qwen3_moe.model import Qwen3MoeForCausalLM
    from freetoken.models.qwen3_moe.weight import iter_weights

    path, hf, _ = fp8_checkpoint
    cfg = dataclasses.replace(parse_config(hf), moe_backend="offload")
    loaded = dict(iter_weights(path, torch.device("cpu"), include_moe_experts=False, include_non_moe=True))
    expected = Qwen3MoeForCausalLM(cfg).state_dict()
    assert set(loaded) == set(expected)
    assert not any(".experts." in k for k in loaded)


@pytest.mark.parametrize("parallel", [False, True])
def test_setup_offload_expert_banks_fp8(tp1, fp8_checkpoint, parallel):
    from freetoken.models.qwen3_moe.config import parse_config
    from freetoken.models.qwen3_moe.weight import setup_offload_expert_banks

    path, hf, tensors = fp8_checkpoint
    banks = setup_offload_expert_banks(
        path, parse_config(hf), device=torch.device("cpu"), dtype=torch.bfloat16, parallel=parallel,
    )
    assert banks.quant_format == "fp8_block"
    src = banks.sources
    assert set(src) == {"gate_up", "gate_up_scale", "down", "down_scale"}
    assert len(src["gate_up"]) == L
    for li in range(L):
        for e in range(E):
            ep = f"model.layers.{li}.mlp.experts.{e}."
            assert torch.equal(
                src["gate_up"][li][e, MOE_I:].view(torch.uint8),
                tensors[ep + "up_proj.weight"].view(torch.uint8),
            )
            # scale rows are 16B-padded in the bank (fp8_block_scale_pad); compare the payload
            assert torch.equal(
                src["down_scale"][li][e, :, : MOE_I // 128], tensors[ep + "down_proj.weight_scale_inv"]
            )


def test_setup_offload_expert_banks_bf16_mode(tp1, fp8_checkpoint, monkeypatch):
    """FREETOKEN_FP8_EXPERTS=bf16 dequantizes at load onto the plain bf16 offload path."""
    from freetoken.kernel.triton.fp8_block_linear import dequant_block_fp8
    from freetoken.models.qwen3_moe.config import parse_config
    from freetoken.models.qwen3_moe.weight import setup_offload_expert_banks

    monkeypatch.setenv("FREETOKEN_FP8_EXPERTS", "bf16")
    path, hf, tensors = fp8_checkpoint
    banks = setup_offload_expert_banks(
        path, parse_config(hf), device=torch.device("cpu"), dtype=torch.bfloat16, parallel=False,
    )
    assert banks.quant_format == "bf16"
    ep = "model.layers.0.mlp.experts.1."
    ref = dequant_block_fp8(tensors[ep + "down_proj.weight"], tensors[ep + "down_proj.weight_scale_inv"])
    torch.testing.assert_close(banks.sources["down"][0][1], ref.to(torch.bfloat16))


def test_bf16_checkpoint_offload_hook_defers_to_bf16_provider(tp1, monkeypatch):
    """A plain bf16 qwen3_moe checkpoint must still take the generic bf16 provider."""
    import freetoken.moe.expert_banks as eb
    from freetoken.models.qwen3_moe.config import parse_config
    from freetoken.models.qwen3_moe.weight import setup_offload_expert_banks

    calls = []
    monkeypatch.setitem(eb._PROVIDERS, "none", lambda *a, **kw: calls.append((a, kw)) or "bf16-banks")
    out = setup_offload_expert_banks(
        "unused", parse_config(_hf_config(fp8=False)), device=torch.device("cpu"), dtype=torch.bfloat16,
        parallel=True, workers=3, chunk=1 << 20, decode_target="cpu",
    )
    assert out == "bf16-banks" and len(calls) == 1
    assert calls[0][1]["parallel"] is True and calls[0][1]["decode_target"] == "cpu"
