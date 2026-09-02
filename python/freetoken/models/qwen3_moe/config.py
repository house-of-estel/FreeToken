from __future__ import annotations

from typing import Any

from freetoken.models.config import (
    ModelConfig,
    RotaryConfig,
    detect_expert_quant,
    detect_fp8_block_quant,
)


def parse_config(hf_config: Any) -> ModelConfig:
    num_kv_heads = getattr(
        hf_config,
        "num_key_value_heads",
        hf_config.num_attention_heads,
    )
    head_dim = (
        getattr(hf_config, "head_dim", None)
        or hf_config.hidden_size // hf_config.num_attention_heads
    )
    rope_scaling = getattr(hf_config, "rope_scaling", None)
    rope_theta = getattr(hf_config, "rope_theta", None)
    if rope_theta is None and rope_scaling is not None:
        rope_theta = rope_scaling["rope_theta"]
    # Block-fp8 checkpoints (Qwen3-30B-A3B-FP8, Qwen3-235B-A22B-FP8, ...): routed experts and
    # the attention projections are fp8-e4m3 + 128x128 ``weight_scale_inv``; embed/lm_head,
    # norms and the router gate stay bf16.
    expert_quant, weight_block_size = detect_fp8_block_quant(hf_config)
    attn_quant = "none"
    if expert_quant == "none":
        # modelopt NVFP4 (nvidia/Qwen3-30B-A3B-NVFP4): every Linear is packed FP4 -- routed
        # experts (offload cache) and the attention projections (native W4A16); embed/lm_head,
        # norms and the router gate stay bf16.
        expert_quant = detect_expert_quant(hf_config)
        if expert_quant == "nvfp4":
            attn_quant = "nvfp4"
        elif expert_quant != "none":
            raise NotImplementedError(
                f"qwen3_moe: unsupported quantization {expert_quant!r} (supported: 128x128 "
                "block-fp8, modelopt NVFP4, bf16)"
            )

    return ModelConfig(
        num_layers=hf_config.num_hidden_layers,
        num_qo_heads=hf_config.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hf_config.hidden_size,
        vocab_size=hf_config.vocab_size,
        intermediate_size=hf_config.intermediate_size,
        hidden_act=hf_config.hidden_act,
        rms_norm_eps=hf_config.rms_norm_eps,
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        rotary_config=RotaryConfig(
            head_dim=head_dim,
            rotary_dim=head_dim,
            max_position=hf_config.max_position_embeddings,
            base=rope_theta,
            scaling=rope_scaling,
        ),
        num_experts=getattr(
            hf_config,
            "num_local_experts",
            getattr(hf_config, "num_experts", 0),
        ),
        num_experts_per_tok=getattr(hf_config, "num_experts_per_tok", 0),
        moe_intermediate_size=getattr(hf_config, "moe_intermediate_size", 0),
        norm_topk_prob=bool(getattr(hf_config, "norm_topk_prob", False)),
        model_type=getattr(hf_config, "model_type", "qwen3_moe"),
        expert_quant=expert_quant,
        weight_block_size=weight_block_size,
        attn_quant=attn_quant,
        architectures=getattr(hf_config, "architectures", ["Qwen3MoeForCausalLM"]),
    )


__all__ = ["parse_config"]
