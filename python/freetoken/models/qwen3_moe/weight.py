from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.fp8_block_banks import (
    Fp8BlockExpertSpec,
    iter_fp8_resident_experts,
    setup_fp8_block_offload_banks,
)
from freetoken.models.loader import (
    MergeRule,
    drop_page_cache,
    iter_merged_tensors,
    iter_stacked_experts,
    iter_weight_files,
    nvfp4_parts_modelopt,
    shard_tensor,
)
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
    load_nvfp4_expert_source_banks_parallel,
)
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")
_MERGE_RULES = {
    ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
    ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
    ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
    ".gate_proj": MergeRule(".gate_up_proj", "gate", ("gate", "up")),
    ".up_proj": MergeRule(".gate_up_proj", "up", ("gate", "up")),
}

# Block-fp8 checkpoint (Qwen3-30B-A3B-FP8 / Qwen3-235B-A22B-FP8): every routed expert is
# stored un-fused as ``{gate,up,down}_proj.{weight,weight_scale_inv}``; the attention
# projections carry the same pair; embed/lm_head, norms and the router gate stay bf16.
_FP8_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate|up|down)_proj\.(?P<kind>weight|weight_scale_inv)$"
)
_FP8_EXPERT_SPEC = Fp8BlockExpertSpec(
    key_pattern=_FP8_EXPERT_RE, layer_prefix="model.layers", desc="Qwen3-MoE fp8 experts"
)
# q|k|v -> qkv_proj for both the fp8 ``.weight`` and the bf16 ``.weight_scale_inv``
# (concatenated along the output dim; each part's out dim is /128 so the scale rows align).
_FP8_QKV_PARTS = (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj")
_FP8_QKV_FUSED = ".self_attn.qkv_proj"
_FP8_KIND_SUFFIXES = (".weight_scale_inv", ".weight")


# modelopt NVFP4 checkpoint (nvidia/Qwen3-30B-A3B-NVFP4): every Linear is packed FP4 --
# ``weight`` uint8 [O, IN/2] + ``weight_scale`` fp8 [O, IN/16] + scalar ``weight_scale_2``;
# routed experts per-expert under ``model.layers.N.mlp.experts.E.{proj}``; ``input_scale`` /
# ``k_scale`` / ``v_scale`` (W4A4 / FP8-KV static scales) are unused and dropped.
_NVFP4_EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_NVFP4_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,  # every layer is MoE
    desc="Qwen3-MoE NVFP4 experts",
)
_NVFP4_DROP_SUFFIXES = (
    ".weight_scale", ".weight_scale_2", ".input_scale", ".k_scale", ".v_scale",
)
_NVFP4_QKV_KINDS = (".weight", ".weight_scale", ".weight_global")


def _split_kind(name: str) -> tuple[str, str]:
    """``name`` -> ``(base, kind_suffix)``; ``kind_suffix`` is "" for other keys."""
    for suf in _FP8_KIND_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)], suf
    return name, ""


def _iter_weights_fp8(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Block-fp8 weights, renamed + fused to the model buffers. fp8 weights (e4m3) and their
    bf16 ``weight_scale_inv`` pass through verbatim (the engine's load-time cast is a no-op
    against the fp8/bf16 model buffers); q/k/v -> qkv_proj per kind. Routed experts are
    skipped under offload (loaded by ``setup_offload_expert_banks``); the resident path
    (``include_moe_experts``) yields the per-layer stacked fp8 expert banks."""
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen3_moe block-fp8 weight loading supports TP=1 only")
    config = parse_config(cached_load_hf_config(model_path))
    if include_non_moe:
        fuse_buf: dict[str, dict[int, torch.Tensor]] = {}
        for file in tqdm(
            iter_weight_files(model_path),
            desc="Loading fp8 weights",
            disable=not get_tp_info().is_primary(),
        ):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                for raw_name in f.keys():
                    name = raw_name.removeprefix("language_model.")
                    if _EXPERT_PATTERN.match(name) is not None:
                        continue  # routed experts: resident banks below / offload hook
                    tensor = f.get_tensor(raw_name)
                    base, suf = _split_kind(name)
                    for idx, part in enumerate(_FP8_QKV_PARTS):
                        if base.endswith(part):
                            key = base[: -len(part)] + _FP8_QKV_FUSED + suf
                            slots = fuse_buf.setdefault(key, {})
                            slots[idx] = tensor
                            if len(slots) == len(_FP8_QKV_PARTS):
                                del fuse_buf[key]
                                yield key, torch.cat(
                                    [slots[i] for i in range(len(_FP8_QKV_PARTS))], dim=0
                                )
                            break
                    else:
                        yield name, tensor
        assert not fuse_buf, f"qwen3_moe: incomplete qkv fusions: {sorted(fuse_buf)}"
    if include_moe_experts:
        yield from iter_fp8_resident_experts(model_path, config, _FP8_EXPERT_SPEC)


def _iter_weights_nvfp4(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """modelopt NVFP4 dense weights kept native (W4A16): each quantized projection yields
    ``(.weight uint8, .weight_scale fp8, .weight_global fp16 per-row)`` for the
    ``Nvfp4Dense*`` buffers; q|k|v are concatenated per kind into ``qkv_proj``. bf16 tensors
    (embed, lm_head, norms, router gate) pass through. Routed experts are never yielded here:
    NVFP4 experts are served from the offload cache (``load_nvfp4_expert_sources``)."""
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen3_moe NVFP4 weight loading supports TP=1 only")
    if include_moe_experts:
        raise NotImplementedError(
            "qwen3_moe NVFP4 experts are offload-only (--moe-backend offload/cpu/hybrid)"
        )
    if not include_non_moe:
        return
    fuse_buf: dict[str, dict[int, tuple]] = {}
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading NVFP4 weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            keyset = set(f.keys())
            for raw_name in f.keys():
                name = raw_name.removeprefix("language_model.")
                if _EXPERT_PATTERN.match(name) is not None or name.endswith(_NVFP4_DROP_SUFFIXES):
                    continue  # routed experts (offload cache) / scales consumed with .weight
                if not name.endswith(".weight") or raw_name[: -len(".weight")] + ".weight_scale_2" not in keyset:
                    yield name, f.get_tensor(raw_name)  # bf16 pass-through
                    continue
                base = name[: -len(".weight")]
                parts = nvfp4_parts_modelopt(f, raw_name[: -len(".weight")])
                for idx, part in enumerate(_FP8_QKV_PARTS):
                    if base.endswith(part):
                        key = base[: -len(part)] + _FP8_QKV_FUSED
                        slots = fuse_buf.setdefault(key, {})
                        slots[idx] = parts
                        if len(slots) == len(_FP8_QKV_PARTS):
                            del fuse_buf[key]
                            for j, suf in enumerate(_NVFP4_QKV_KINDS):
                                yield key + suf, torch.cat(
                                    [slots[i][j] for i in range(len(_FP8_QKV_PARTS))], dim=0
                                )
                        break
                else:
                    for j, suf in enumerate(_NVFP4_QKV_KINDS):
                        yield base + suf, parts[j]
    assert not fuse_buf, f"qwen3_moe: incomplete NVFP4 qkv fusions: {sorted(fuse_buf)}"


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None):
    """CPU NVFP4 expert source banks for the offload cache (gate|up fused on the output-row
    axis, down separate; ``weight_scale_2`` carried as the per-row global scale)."""
    return load_nvfp4_expert_source_banks(
        model_path, config, _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache, primary=get_tp_info().is_primary(), layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None
):
    """Same banks via the common chunked multi-threaded O_DIRECT reader."""
    return load_nvfp4_expert_source_banks_parallel(
        model_path, config, _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache, primary=get_tp_info().is_primary(),
        workers=workers, chunk=chunk, layer_sink=layer_sink,
    )


def setup_offload_expert_banks(
    model_path: str, model_config, *, device: torch.device, dtype: torch.dtype,
    dummy: bool = False, parallel: bool = False, workers: int = 8, chunk: int = 8 << 20,
    decode_target: str = "gpu", layer_sink=None,
):
    """Routed-expert offload banks. Exported, so it intercepts every qwen3_moe offload load:
    block-fp8 checkpoints build the shared fp8 banks (``FREETOKEN_FP8_EXPERTS=bf16``
    dequantizes at load instead); anything else defers to the generic provider for that
    ``expert_quant`` (bf16, or nvfp4 via ``load_nvfp4_expert_sources`` and the marlin/b12x
    repack), exactly as before this hook existed."""
    eq = getattr(model_config, "expert_quant", "none")
    if eq != "fp8_block":
        from freetoken.moe.expert_banks import _PROVIDERS

        return _PROVIDERS[eq](model_path, model_config, device, dtype, dummy,
                              parallel=parallel, workers=workers, chunk=chunk,
                              decode_target=decode_target, layer_sink=layer_sink)
    return setup_fp8_block_offload_banks(
        model_path, model_config, _FP8_EXPERT_SPEC, device=device, dummy=dummy,
        parallel=parallel, workers=workers, chunk=chunk, layer_sink=layer_sink,
    )


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    config = parse_config(cached_load_hf_config(model_path))
    if config.expert_quant == "fp8_block":
        yield from _iter_weights_fp8(
            model_path, device,
            include_moe_experts=include_moe_experts, include_non_moe=include_non_moe,
        )
        return
    if config.expert_quant == "nvfp4":
        yield from _iter_weights_nvfp4(
            model_path, device,
            include_moe_experts=include_moe_experts, include_non_moe=include_non_moe,
        )
        return
    tp_info = get_tp_info()

    def sharded_tensors() -> Iterator[tuple[str, torch.Tensor]]:
        for file in tqdm(
            iter_weight_files(model_path),
            desc="Loading weights",
            disable=not tp_info.is_primary(),
        ):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                for raw_name in f.keys():
                    name = raw_name.removeprefix("language_model.")
                    is_expert = _EXPERT_PATTERN.match(name) is not None
                    if is_expert and not include_moe_experts:
                        continue
                    if not is_expert and not include_non_moe:
                        continue

                    raw = f.get_tensor(raw_name)
                    tensor = shard_tensor(
                        name,
                        raw,
                        rank=tp_info.rank,
                        world_size=tp_info.size,
                        num_kv_heads=config.num_kv_heads,
                    )
                    del raw
                    yield name, tensor

    merged = iter_merged_tensors(
        sharded_tensors(),
        _MERGE_RULES,
        model_name="qwen3_moe",
    )
    yield from iter_stacked_experts(
        merged,
        num_experts=config.num_experts,
        model_name="qwen3_moe",
        expert_pattern=_EXPERT_PATTERN,
    )


def iter_weights_parallel(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
) -> Iterator[tuple[str, torch.Tensor]]:
    """experts-only iter_weights: raw experts read via the common chunked multi-threaded
    O_DIRECT reader, then same merge+stack pipeline."""
    assert include_moe_experts and not include_non_moe, (
        "qwen3_moe parallel reader is experts-only (used by load_moe_expert_sources)"
    )
    from freetoken.models.weight import iter_expert_tensors_parallel

    config = parse_config(cached_load_hf_config(model_path))
    if config.expert_quant != "none":
        raise NotImplementedError(
            f"qwen3_moe {config.expert_quant} experts are loaded by setup_offload_expert_banks"
        )
    tp_info = get_tp_info()

    def _is_expert(raw_name: str) -> bool:
        return _EXPERT_PATTERN.match(raw_name.removeprefix("language_model.")) is not None

    def raw_experts() -> Iterator[tuple[str, torch.Tensor]]:
        for raw_name, raw in iter_expert_tensors_parallel(
            model_path, _is_expert, workers=workers, chunk=chunk
        ):
            name = raw_name.removeprefix("language_model.")
            tensor = shard_tensor(
                name, raw, rank=tp_info.rank, world_size=tp_info.size,
                num_kv_heads=config.num_kv_heads,
            )
            yield name, tensor

    merged = iter_merged_tensors(raw_experts(), _MERGE_RULES, model_name="qwen3_moe")
    yield from iter_stacked_experts(
        merged, num_experts=config.num_experts, model_name="qwen3_moe",
        expert_pattern=_EXPERT_PATTERN,
    )


__all__ = [
    "iter_weights",
    "iter_weights_parallel",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "setup_offload_expert_banks",
]
