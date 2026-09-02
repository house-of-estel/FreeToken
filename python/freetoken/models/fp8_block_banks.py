"""Block-fp8 (DeepSeek-V3-style 128x128) routed-expert banks, shared by the models that
serve such checkpoints (qwen3_5_moe, qwen3_moe).

A block-fp8 checkpoint stores each routed expert un-fused as ``{gate,up,down}_proj.weight``
(fp8-e4m3) plus a bf16 ``weight_scale_inv`` per 128x128 block. The routines here read those
per-expert tensors and stack them into the per-layer ``[E, ...]`` banks that the offload
cache (``ExpertBanks("fp8_block", ...)``) and the resident ``MoELayer(weight_format=
"fp8_block")`` buffers consume. Only the checkpoint key layout differs between models, so
each model passes an :class:`Fp8BlockExpertSpec`; the reading, stacking, pinning and
streaming live here once.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterator

import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import ShardReader
from tqdm import tqdm

_BLOCK = 128
_PROJS = ("gate", "up", "down")
_KINDS = ("weight", "weight_scale_inv")


@dataclass(frozen=True)
class Fp8BlockExpertSpec:
    """How a model's block-fp8 routed experts are keyed in the checkpoint.

    ``key_pattern`` matches the RAW safetensors key and must expose the groups ``layer``,
    ``expert``, ``proj`` (``gate|up|down``) and ``kind`` (``weight|weight_scale_inv``).
    ``layer_prefix`` is the raw key prefix up to (excluding) the layer index, used to name
    the keys for the serial shard reader:
    ``{layer_prefix}.{L}.mlp.experts.{E}.{proj}_proj.{kind}``.
    """

    key_pattern: re.Pattern[str]
    layer_prefix: str
    desc: str = "fp8 experts"

    def key(self, layer: int, expert: int, proj: str, kind: str) -> str:
        return f"{self.layer_prefix}.{layer}.mlp.experts.{expert}.{proj}_proj.{kind}"

    def is_expert(self, raw_name: str) -> bool:
        return self.key_pattern.match(raw_name) is not None


def moe_dims(model_config) -> tuple[int, int, int, int, int]:
    """``(num_moe_layers, num_experts, hidden, moe_intermediate, dense_prefix_layers)``."""
    L = model_config.num_moe_layers
    return (
        L, model_config.num_experts, model_config.hidden_size,
        model_config.moe_intermediate_size, model_config.num_layers - L,
    )


def build_fp8_expert_banks(
    model_path, config, spec: Fp8BlockExpertSpec, *, dummy: bool, parallel: bool | None = None,
    workers: int = 8, chunk: int = 8 << 20, pin: bool = True, layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """The single block-fp8 routed-expert reader, shared by offload (``pin=True`` -> per-layer
    :class:`HostBank` banks, pin-after-fill / streamable) and resident (``pin=False`` -> plain
    pageable tensors, never pinned or streamed). Reads via the common chunked multi-threaded
    O_DIRECT reader (drops page cache, no per-tensor serial overhead) when the experts are
    scattered per-tensor; else a serial shard fallback. Stacks gate|up -> gate_up into one
    ``[E, ...]`` tensor per layer per bank.

    Each expert contributes 6 ``place()`` writes per layer ({gate,up,down} x {weight,
    weight_scale_inv}), so a layer completes after ``E * 6`` writes. ``layer_sink=None``
    (serving) pins each layer's 4 banks as it completes via an internally-owned
    :class:`PinPipeline`; ``layer_sink`` given (converter) fires the completion tracker into it
    instead (nothing pinned; released banks -- caller owns that tradeoff). ``pin=False`` and the
    CUDA-less host stay on the plain materialize path (no pin, no stream)."""
    from freetoken.kernel.triton.fp8_block_linear import FP8
    from freetoken.models.weight import experts_scattered, iter_expert_tensors_parallel

    B = _BLOCK
    L, E, H, I, dense = moe_dims(config)
    assert H % B == 0 and I % B == 0, (
        f"block-fp8 experts need hidden/moe_intermediate multiples of {B}, got {H}/{I}"
    )

    # 16B-align the per-expert scale rows (Qwen3.8: down_scale is 20x5 bf16 = 200 B) so the
    # fused multi-bank copy engages; the GEMMs read scales through explicit strides, so the
    # padding is inert. Unconditional: one layout per format, shared with the byte formulas.
    from freetoken.moe.offload_cache import fp8_block_scale_pad as _pad_cols

    specs = {
        "gate_up": ((E, 2 * I, H), FP8),
        "gate_up_scale": ((E, 2 * I // B, _pad_cols(2 * I // B, H // B)), torch.bfloat16),
        "down": ((E, H, I), FP8),
        "down_scale": ((E, H // B, _pad_cols(H // B, I // B)), torch.bfloat16),
    }
    hb = None
    if pin:
        from freetoken.moe.host_banks import alloc_layer_banks

        hb = alloc_layer_banks(specs, L)  # lazy anon mmaps (unpinned)
        banks = {name: [b.tensor for b in hb[name]] for name in specs}
    else:  # resident dequant source: plain pageable tensors (never pinned / streamed)
        banks = {name: [torch.empty(shape, dtype=dt) for _ in range(L)] for name, (shape, dt) in specs.items()}
    gate_up, gate_up_scale, down, down_scale = (
        banks["gate_up"], banks["gate_up_scale"], banks["down"], banks["down_scale"]
    )
    if dummy:
        for li in range(L):
            gate_up[li].view(torch.uint8).random_(0, 16)  # small fp8 codes (avoid NaN/inf)
            down[li].view(torch.uint8).random_(0, 16)
            gate_up_scale[li].fill_(1.0)
            down_scale[li].fill_(1.0)
        if hb is not None and torch.cuda.is_available():
            from freetoken.moe.host_banks import pin_banks

            pin_banks(hb)  # pin-after-fill (match the other dummies)
        return banks

    def place(raw_name: str, t: torch.Tensor) -> int | None:
        m = spec.key_pattern.match(raw_name)
        if m is None:
            return None
        li, e = int(m["layer"]) - dense, int(m["expert"])
        proj, kind = m["proj"], m["kind"]
        if kind == "weight":
            (gate_up[li][e, :I] if proj == "gate" else
             gate_up[li][e, I:] if proj == "up" else down[li][e]).copy_(t)
        else:  # weight_scale_inv
            (gate_up_scale[li][e, : I // B, : H // B] if proj == "gate" else
             gate_up_scale[li][e, I // B :, : H // B] if proj == "up" else
             down_scale[li][e, :, : I // B]).copy_(t)
        return li

    if parallel is None:
        parallel = experts_scattered(model_path)

    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline

    def _load(sink) -> None:
        # {gate,up,down} x {weight, weight_scale_inv} per expert -> E*6 writes/layer.
        tracker = LayerCompletionTracker(E * 6, hb, sink) if sink is not None else None
        if parallel:
            for raw_name, t in iter_expert_tensors_parallel(
                model_path, spec.is_expert, workers=workers, chunk=chunk
            ):
                li = place(raw_name, t)
                if tracker is not None and li is not None:
                    tracker.note(li)
        else:
            reader = ShardReader(model_path, torch.device("cpu"))
            primary = get_tp_info().is_primary()
            try:
                for li in tqdm(range(L), desc=f"Loading {spec.desc} (serial)", disable=not primary):
                    layer = dense + li
                    for e in range(E):
                        for proj in _PROJS:
                            for kind in _KINDS:
                                key = spec.key(layer, e, proj, kind)
                                place(key, reader.get_tensor(key))
                                if tracker is not None:
                                    tracker.note(li)
            finally:
                reader.close()

    if not pin:
        assert layer_sink is None, "pin=False (resident source) cannot stream to a layer_sink"
        _load(None)
    elif layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)  # CUDA-less: mmap banks stay pageable, never pinned
    return banks


def setup_bf16_dequant_banks(
    model_path, model_config, spec: Fp8BlockExpertSpec, device, dummy: bool, *, layer_sink=None
):
    """``FREETOKEN_FP8_EXPERTS=bf16``: dequantize every routed expert to bf16 at load and hand
    the result to the plain bf16 offload path (~2x the host bytes of the fp8 banks)."""
    from freetoken.models.weight import dummy_moe_expert_sources
    from freetoken.moe.expert_banks import ExpertBanks

    if dummy:
        gate_up, down = dummy_moe_expert_sources(model_config, dtype=torch.bfloat16)
        return ExpertBanks("bf16", {"gate_up": gate_up, "down": down})

    from freetoken.kernel.triton.fp8_block_linear import dequant_block_fp8
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    L, E, H, I, dense = moe_dims(model_config)
    specs = {"gate_up": ((E, 2 * I, H), torch.bfloat16), "down": ((E, H, I), torch.bfloat16)}
    hb = alloc_layer_banks(specs, L)  # lazy anon mmaps (unpinned)
    banks = {name: [b.tensor for b in hb[name]] for name in specs}
    gate_up, down = banks["gate_up"], banks["down"]
    reader = ShardReader(model_path, device)
    primary = get_tp_info().is_primary()

    def _deq(layer: int, e: int, proj: str) -> torch.Tensor:
        return dequant_block_fp8(
            reader.get_tensor(spec.key(layer, e, proj, "weight")),
            reader.get_tensor(spec.key(layer, e, proj, "weight_scale_inv")),
        )

    def _load(sink) -> None:
        # Whole-layer gate_up + down copies -> 2 writes/layer.
        tracker = LayerCompletionTracker(2, hb, sink) if sink is not None else None
        try:
            for li in tqdm(range(L), desc=f"Loading {spec.desc} (dequant->bf16)", disable=not primary):
                layer = dense + li
                gu_rows = torch.empty(E, 2 * I, H, dtype=torch.bfloat16, device=device)
                dn_rows = torch.empty(E, H, I, dtype=torch.bfloat16, device=device)
                for e in range(E):
                    gu_rows[e, :I] = _deq(layer, e, "gate")
                    gu_rows[e, I:] = _deq(layer, e, "up")
                    dn_rows[e] = _deq(layer, e, "down")
                gate_up[li].copy_(gu_rows)
                if tracker is not None:
                    tracker.note(li)
                down[li].copy_(dn_rows)
                if tracker is not None:
                    tracker.note(li)
        finally:
            reader.close()

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)  # CUDA-less: mmap banks stay pageable, never pinned
    return ExpertBanks("bf16", banks, streamed=layer_sink is not None)


def setup_fp8_block_offload_banks(
    model_path, model_config, spec: Fp8BlockExpertSpec, *, device: torch.device,
    dummy: bool = False, parallel: bool = False, workers: int = 8, chunk: int = 8 << 20,
    layer_sink=None,
):
    """Build the block-fp8 routed-expert offload banks (the model's
    ``setup_offload_expert_banks`` body for ``expert_quant == "fp8_block"``).

    Default (``FREETOKEN_FP8_EXPERTS=fp8``): keep experts block-fp8 -- ``gate_up``/``down``
    fp8 banks + their bf16 ``weight_scale_inv`` banks (half the host/cache bytes; routed rows
    are dequantized on demand in ``_expert_gemm``). ``FREETOKEN_FP8_EXPERTS=bf16`` instead
    dequantizes every expert to bf16 at load (reuses the bf16 offload path; ~2x the memory).
    Both modes build per-layer :class:`HostBank` banks (pin-after-fill), so the converter's
    ``layer_sink`` streams each completed layer straight through."""
    if get_tp_info().size > 1:
        raise NotImplementedError("block-fp8 expert banks support TP=1 only")
    from freetoken.moe.expert_banks import ExpertBanks

    mode = os.environ.get("FREETOKEN_FP8_EXPERTS", "fp8").strip().lower()
    if mode not in ("fp8", "bf16"):
        raise ValueError(f"FREETOKEN_FP8_EXPERTS must be 'fp8' or 'bf16', got {mode!r}")
    sink = None if dummy else layer_sink
    if mode == "bf16":
        return setup_bf16_dequant_banks(model_path, model_config, spec, device, dummy, layer_sink=sink)
    banks = build_fp8_expert_banks(
        model_path, model_config, spec, dummy=dummy, parallel=parallel, workers=workers,
        chunk=chunk, pin=True, layer_sink=sink,
    )
    return ExpertBanks("fp8_block", banks, streamed=sink is not None)


def iter_fp8_resident_experts(
    model_path, model_config, spec: Fp8BlockExpertSpec,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Resident (non-offload) experts: build the stacked fp8 banks once (pageable host; the
    engine copies the per-layer slices to GPU), then yield the per-layer buffers of the
    ``MoELayer(weight_format="fp8_block")`` layout under ``model.layers.N.mlp.experts.*``."""
    if not model_config.is_moe:
        return  # dense checkpoint: no routed experts to build as resident banks
    L, _, H, I, dense = moe_dims(model_config)
    B = _BLOCK
    banks = build_fp8_expert_banks(model_path, model_config, spec, dummy=False, pin=False)
    for li in range(L):
        pre = f"model.layers.{dense + li}.mlp.experts"
        # The resident MoELayer buffers are unpadded: drop the 16B scale-row padding.
        yield f"{pre}.gate_up_proj", banks["gate_up"][li]
        yield f"{pre}.gate_up_scale_inv", banks["gate_up_scale"][li][:, :, : H // B]
        yield f"{pre}.down_proj", banks["down"][li]
        yield f"{pre}.down_scale_inv", banks["down_scale"][li][:, :, : I // B]


__all__ = [
    "Fp8BlockExpertSpec",
    "build_fp8_expert_banks",
    "iter_fp8_resident_experts",
    "moe_dims",
    "setup_bf16_dequant_banks",
    "setup_fp8_block_offload_banks",
]
