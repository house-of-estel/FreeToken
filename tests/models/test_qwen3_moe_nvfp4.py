"""qwen3_moe modelopt NVFP4 checkpoints (nvidia/Qwen3-30B-A3B-NVFP4 layout): config
detection, native W4A16 attention, the dense loader (q|k|v fused per kind) and the NVFP4
expert source banks for the offload cache."""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch

FP8 = torch.float8_e4m3fn
H, MOE_I, E, L = 256, 128, 2, 2
Q_DIM, KV_DIM = 2 * 128, 1 * 128


def _hf_config(*, nvfp4: bool = True):
    from transformers import Qwen3MoeConfig

    cfg = Qwen3MoeConfig(
        hidden_size=H, intermediate_size=512, moe_intermediate_size=MOE_I, num_experts=E,
        num_experts_per_tok=1, num_hidden_layers=L, num_attention_heads=2, num_key_value_heads=1,
        head_dim=128, vocab_size=64, max_position_embeddings=128, rope_theta=10000.0,
        rms_norm_eps=1e-6, hidden_act="silu", norm_topk_prob=True, tie_word_embeddings=False,
        architectures=["Qwen3MoeForCausalLM"],
    )
    if nvfp4:
        cfg.quantization_config = {
            "quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8", "group_size": 16,
            "exclude_modules": ["model.layers.0.mlp.gate", "lm_head"],
        }
    return cfg


def _nvfp4_tensors(base: str, out_f: int, in_f: int, t: dict) -> None:
    t[base + ".weight"] = torch.randint(0, 256, (out_f, in_f // 2), dtype=torch.uint8)
    t[base + ".weight_scale"] = (torch.rand(out_f, in_f // 16) + 0.5).to(FP8)
    t[base + ".weight_scale_2"] = torch.rand(()) + 0.5
    t[base + ".input_scale"] = torch.rand(()) + 0.5


def _checkpoint_tensors() -> dict[str, torch.Tensor]:
    t = {
        "model.embed_tokens.weight": torch.randn(64, H, dtype=torch.bfloat16),
        "model.norm.weight": torch.randn(H, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(64, H, dtype=torch.bfloat16),
    }
    for li in range(L):
        lp = f"model.layers.{li}."
        for proj, (o, i) in {"q_proj": (Q_DIM, H), "k_proj": (KV_DIM, H), "v_proj": (KV_DIM, H), "o_proj": (H, Q_DIM)}.items():
            _nvfp4_tensors(lp + f"self_attn.{proj}", o, i, t)
        t[lp + "self_attn.k_proj.k_scale"] = torch.rand(())
        t[lp + "self_attn.v_proj.v_scale"] = torch.rand(())
        for n in ("self_attn.q_norm", "self_attn.k_norm"):
            t[lp + n + ".weight"] = torch.randn(128, dtype=torch.bfloat16)
        for n in ("input_layernorm", "post_attention_layernorm"):
            t[lp + n + ".weight"] = torch.randn(H, dtype=torch.bfloat16)
        t[lp + "mlp.gate.weight"] = torch.randn(E, H, dtype=torch.bfloat16)
        for e in range(E):
            for proj, (o, i) in {"gate_proj": (MOE_I, H), "up_proj": (MOE_I, H), "down_proj": (H, MOE_I)}.items():
                _nvfp4_tensors(lp + f"mlp.experts.{e}.{proj}", o, i, t)
    return t


def _write_checkpoint(tmp_path, tensors, hf) -> str:
    import safetensors.torch

    shard = "model-00001-of-00001.safetensors"
    safetensors.torch.save_file(tensors, str(tmp_path / shard))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {n: shard for n in tensors}}))
    (tmp_path / "config.json").write_text(json.dumps(hf.to_dict()))
    return str(tmp_path)


@pytest.fixture
def tp1():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


@pytest.fixture
def nvfp4_checkpoint(tmp_path, monkeypatch):
    hf = _hf_config()
    tensors = _checkpoint_tensors()
    path = _write_checkpoint(tmp_path, tensors, hf)
    import freetoken.models.qwen3_moe.weight as w

    monkeypatch.setattr(w, "cached_load_hf_config", lambda _p: hf)
    return path, hf, tensors


def test_parse_config_detects_nvfp4():
    from freetoken.models.qwen3_moe.config import parse_config

    cfg = parse_config(_hf_config())
    assert cfg.expert_quant == "nvfp4" and cfg.attn_quant == "nvfp4" and cfg.weight_block_size is None
    bf16 = parse_config(_hf_config(nvfp4=False))
    assert bf16.expert_quant == "none" and bf16.attn_quant == "none"


def test_unsupported_quant_is_a_clear_error():
    from freetoken.models.qwen3_moe.config import parse_config

    hf = _hf_config(nvfp4=False)
    hf.quantization_config = {"quant_method": "awq", "bits": 4}
    with pytest.raises(NotImplementedError, match="unsupported quantization 'awq'"):
        parse_config(hf)


def test_nvfp4_model_uses_w4a16_attention(tp1):
    from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseColMerged, Nvfp4DenseLinear
    from freetoken.models.qwen3_moe.config import parse_config
    from freetoken.models.qwen3_moe.model import Qwen3MoeForCausalLM

    cfg = dataclasses.replace(parse_config(_hf_config()), moe_backend="offload")
    attn = Qwen3MoeForCausalLM(cfg).model.layers.op_list[0].self_attn
    assert isinstance(attn.qkv_proj, Nvfp4DenseColMerged) and isinstance(attn.o_proj, Nvfp4DenseLinear)
    assert attn.qkv_proj.weight.shape == (Q_DIM + 2 * KV_DIM, H // 2)
    assert attn.qkv_proj.weight_scale.shape == (Q_DIM + 2 * KV_DIM, H // 16)
    assert attn.qkv_proj.weight_global.shape == (Q_DIM + 2 * KV_DIM,)


def test_iter_weights_nvfp4_matches_offload_model_state_dict(tp1, nvfp4_checkpoint):
    from freetoken.models.qwen3_moe.config import parse_config
    from freetoken.models.qwen3_moe.model import Qwen3MoeForCausalLM
    from freetoken.models.qwen3_moe.weight import iter_weights

    path, hf, tensors = nvfp4_checkpoint
    cfg = dataclasses.replace(parse_config(hf), moe_backend="offload")
    loaded = dict(iter_weights(path, torch.device("cpu"), include_moe_experts=False, include_non_moe=True))
    expected = Qwen3MoeForCausalLM(cfg).state_dict()
    assert set(loaded) == set(expected)
    for k in expected:
        assert loaded[k].shape == expected[k].shape, k
        if expected[k].dtype in (torch.uint8, FP8, torch.float16):
            assert loaded[k].dtype == expected[k].dtype, k
    assert not any(".experts." in k for k in loaded)
    assert not any(k.endswith(("input_scale", "k_scale", "v_scale", "weight_scale_2")) for k in loaded)

    p = "model.layers.1.self_attn."
    q, k, v = (tensors[p + f"{x}_proj.weight"] for x in ("q", "k", "v"))
    assert torch.equal(loaded[p + "qkv_proj.weight"], torch.cat([q, k, v]))
    qs, ks, vs = (tensors[p + f"{x}_proj.weight_scale"].view(torch.uint8) for x in ("q", "k", "v"))
    assert torch.equal(loaded[p + "qkv_proj.weight_scale"].view(torch.uint8), torch.cat([qs, ks, vs]))
    glob = loaded[p + "qkv_proj.weight_global"]
    assert glob.dtype == torch.float16 and glob.shape == (Q_DIM + 2 * KV_DIM,)
    for x, sl in (("q", slice(0, Q_DIM)), ("k", slice(Q_DIM, Q_DIM + KV_DIM)), ("v", slice(Q_DIM + KV_DIM, None))):
        g2 = tensors[p + f"{x}_proj.weight_scale_2"].to(torch.float16)
        assert torch.all(glob[sl] == g2), x
    assert torch.equal(loaded[p + "o_proj.weight"], tensors[p + "o_proj.weight"])
    assert torch.equal(loaded["lm_head.weight"], tensors["lm_head.weight"])


def test_iter_weights_nvfp4_resident_experts_rejected(tp1, nvfp4_checkpoint):
    from freetoken.models.qwen3_moe.weight import iter_weights

    path, _, _ = nvfp4_checkpoint
    with pytest.raises(NotImplementedError, match="offload-only"):
        list(iter_weights(path, torch.device("cpu"), include_moe_experts=True, include_non_moe=True))


@pytest.mark.parametrize("parallel", [False, True])
def test_load_nvfp4_expert_sources(tp1, nvfp4_checkpoint, parallel):
    from freetoken.models.qwen3_moe.config import parse_config
    from freetoken.models.qwen3_moe.weight import load_nvfp4_expert_sources, load_nvfp4_expert_sources_parallel

    path, hf, tensors = nvfp4_checkpoint
    cfg = parse_config(hf)
    banks = (load_nvfp4_expert_sources_parallel if parallel else load_nvfp4_expert_sources)(path, cfg)
    assert set(banks) == {"gate_up_packed", "gate_up_scale", "gate_up_global", "down_packed", "down_scale", "down_global"}
    assert len(banks["gate_up_packed"]) == L
    assert banks["gate_up_packed"][0].shape == (E, 2 * MOE_I, H // 2)
    assert banks["gate_up_scale"][0].shape == (E, 2 * MOE_I, H // 16)
    assert banks["gate_up_global"][0].shape == (E, 2 * MOE_I)
    assert banks["down_packed"][0].shape == (E, H, MOE_I // 2)
    for li in range(L):
        for e in range(E):
            ep = f"model.layers.{li}.mlp.experts.{e}."
            assert torch.equal(banks["gate_up_packed"][li][e, :MOE_I], tensors[ep + "gate_proj.weight"])
            assert torch.equal(banks["gate_up_packed"][li][e, MOE_I:], tensors[ep + "up_proj.weight"])
            assert torch.equal(banks["down_packed"][li][e], tensors[ep + "down_proj.weight"])
            assert torch.equal(
                banks["down_scale"][li][e].view(torch.uint8), tensors[ep + "down_proj.weight_scale"].view(torch.uint8)
            )
            assert torch.all(banks["down_global"][li][e] == tensors[ep + "down_proj.weight_scale_2"].to(torch.float16))
            assert torch.all(banks["gate_up_global"][li][e, MOE_I:] == tensors[ep + "up_proj.weight_scale_2"].to(torch.float16))


def test_fp8_block_path_still_selected(tp1):
    """The block-fp8 detection (fix 4) wins over the generic quant detector."""
    from freetoken.models.qwen3_moe.config import parse_config

    hf = _hf_config(nvfp4=False)
    hf.quantization_config = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    cfg = parse_config(hf)
    assert cfg.expert_quant == "fp8_block" and cfg.attn_quant == "none"
