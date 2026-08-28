from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, LinearOProj, LinearQKVMerged, RMSNorm
from freetoken.layers.rotary import get_rope
from freetoken.utils import div_even, nvtx_annotate

if TYPE_CHECKING:
    import torch

    from freetoken.models.config import ModelConfig


class Qwen3MoeAttention(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        has_attn_bias: bool = False,
        has_qk_norm: bool = False,
    ):
        head_dim = config.head_dim
        self.layer_id = layer_id
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(config.num_qo_heads, tp_size)
        self.num_kv_heads = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.head_dim = head_dim
        # Quantized checkpoints keep q/k/v/o native: block-fp8 (fp8-e4m3 + 128x128
        # ``weight_scale_inv``) or modelopt NVFP4 (W4A16). Column-merged along the output dim
        # (q|k|v out dims are all /128). TP=1 only; bf16 keeps the TP-aware layers below.
        self.quant = (
            "fp8_block" if getattr(config, "expert_quant", "none") == "fp8_block"
            else "nvfp4" if getattr(config, "attn_quant", "none") == "nvfp4"
            else "none"
        )
        if self.quant != "none":
            from freetoken.models.quant_linear import make_col_merged

            assert tp_size == 1, f"{self.quant} attention supports TP=1 only"
            assert not has_attn_bias, f"{self.quant} attention has no bias"
            self.qkv_proj = make_col_merged(
                config, config.hidden_size, [self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim]
            )
        else:
            self.qkv_proj = LinearQKVMerged(
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_qo_heads=config.num_qo_heads,
                num_kv_heads=config.num_kv_heads,
                has_bias=has_attn_bias,
            )
        if has_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=(
                tuple(config.rotary_config.scaling.items())
                if config.rotary_config.scaling
                else None
            ),
        )
        if self.quant != "none":
            from freetoken.models.quant_linear import make_replicated

            self.o_proj = make_replicated(config, self.qo_attn_dim, config.hidden_size)
        else:
            self.o_proj = LinearOProj(
                head_dim * config.num_qo_heads,
                config.hidden_size,
                has_bias=False,
            )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        qkv = self.qkv_proj.forward(x)
        del x
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return self.o_proj.forward(o.view(-1, self.qo_attn_dim))


__all__ = ["Qwen3MoeAttention"]
