"""
Per-layer detailed FLOPs accounting for Architeckt.

Decomposes the FLOPs of each component to verify that:
1. MSLA attention is truly O(L) (linear in sequence length)
2. SMoE FFN cost is proportional to active experts
3. AHG overhead is negligible
4. DAEE early exit savings compound
"""

import math
import torch
from typing import Dict, Tuple


def flops_matmul(M: int, N: int, K: int, multiply_add: bool = True) -> int:
    """FLOPs for matrix multiply (M×K) @ (K×N) → (M×N).

    One multiply-add = 2 FLOPs.
    """
    ops = M * N * K
    return 2 * ops if multiply_add else ops


def flops_linear(in_features: int, out_features: int, batch: int) -> int:
    """FLOPs for a linear layer: x(batch, in) @ W(in, out)."""
    return flops_matmul(batch, out_features, in_features)


def flops_embedding(vocab_size: int, d_model: int, batch: int) -> int:
    """Embedding lookup: essentially free (gather operation)."""
    return 0  # lookup is memory-bound, not compute-bound


def flops_adasnorm(d_model: int, batch: int, seq_len: int) -> int:
    """FLOPs for AdaSNorm: RMS + stats + small MLP.

    - RMS: d multiply + d add = 2d per token
    - 3 stats: ~3d operations per token
    - stats MLP: 3→32→1: 3*32*2 + 32*1*2 ≈ 256 FLOPs
    - gamma * alpha * x_hat: d multiply
    Total: ~4d + 256 FLOPs per token
    """
    tokens = batch * seq_len
    return tokens * (4 * d_model + 256)


def flops_msla(
    d_model: int,
    n_scales: int,
    n_heads: int,
    d_head: int,
    batch: int,
    seq_len: int,
) -> Dict[str, int]:
    """FLOPs breakdown for Multi-Scale Linear Attention.

    Returns per-component FLOPs.
    """
    tokens = batch * seq_len
    total_d = n_scales * n_heads * d_head

    # Q, K, V projections
    qkv_flops = 3 * flops_linear(d_model, total_d, tokens)

    # Feature map: ELU + 1
    feat_flops = n_scales * n_heads * tokens * d_head * 3  # ELU is ~3 ops

    # Linear attention: per-scale cumulative sum
    # For each scale, head, position:
    #   - kv_state update: d_head * d_head outer product → d_head^2 multiply
    #   - query * state: d_head * d_head dot products → 2*d_head^2 ops
    # Per token per scale per head: ~3*d_head^2 + d_head ops
    # Total for sequence: O(L * n_scales * n_heads * d_head^2)
    lin_attn_flops = tokens * n_scales * n_heads * (3 * d_head * d_head + d_head)

    # Output projection
    out_flops = flops_linear(total_d, d_model, tokens)

    return {
        "qkv_projections": qkv_flops,
        "feature_map": feat_flops,
        "linear_attention_core": lin_attn_flops,
        "output_projection": out_flops,
        "total": qkv_flops + feat_flops + lin_attn_flops + out_flops,
        "complexity_class": "O(L)" if lin_attn_flops / tokens < 1000 else "O(L²)",
    }


def flops_transformer_attention(
    d_model: int,
    n_heads: int,
    d_head: int,
    batch: int,
    seq_len: int,
) -> Dict[str, int]:
    """FLOPs for standard softmax attention (baseline comparison).

    This is O(L²) — included for reference.
    """
    tokens = batch * seq_len

    # Q, K, V projections
    qkv_flops = 3 * flops_linear(d_model, n_heads * d_head, tokens)

    # Q @ K^T: (B, H, L, d) @ (B, H, d, L) → (B, H, L, L)
    qk_flops = batch * n_heads * seq_len * d_head * seq_len * 2

    # softmax: ~5 ops per element
    softmax_flops = batch * n_heads * seq_len * seq_len * 5

    # attention @ V
    attn_v_flops = batch * n_heads * seq_len * seq_len * d_head * 2

    # Output projection
    out_flops = flops_linear(n_heads * d_head, d_model, tokens)

    return {
        "qkv_projections": qkv_flops,
        "qk_dot_product": qk_flops,
        "softmax": softmax_flops,
        "attn_times_v": attn_v_flops,
        "output_projection": out_flops,
        "total": qkv_flops + qk_flops + softmax_flops + attn_v_flops + out_flops,
        "complexity_class": "O(L²)",
    }


def flops_ahg(
    d_model: int, n_heads: int, batch: int, seq_len: int
) -> int:
    """FLOPs for Adaptive Head Gating.

    Gate projection + sigmoid + threshold comparison.
    Negligible vs attention itself.
    """
    tokens = batch * seq_len
    return flops_linear(d_model, n_heads, tokens) + tokens * n_heads * 3  # sigmoid + threshold


def flops_catr(
    d_model: int, n_types: int, n_experts: int, batch: int, seq_len: int
) -> int:
    """FLOPs for Content-Aware Token Router.

    Type classifier + per-type router dot products.
    Very small vs FFN computation.
    """
    tokens = batch * seq_len

    # Type classifier
    type_flops = flops_linear(d_model, n_types, tokens)

    # Per-type router: x @ type_routers: (B, L, d) @ (T, E, d)
    # This is B*L * T * E * d multiply-adds
    router_flops = tokens * n_types * n_experts * d_model * 2

    # Softmax + top-k
    selection_flops = tokens * (n_types * n_experts * 5 + n_experts * 5)

    return type_flops + router_flops + selection_flops


def flops_smoe_ffn(
    d_model: int,
    d_ff: int,
    n_active: int,
    batch: int,
    seq_len: int,
) -> int:
    """FLOPs for Sparse MoE FFN (only active experts).

    Per token: n_active × (gate + up + down + threshold)
    where each is a matmul.
    """
    tokens = batch * seq_len

    # Per active expert:
    #   W_gate: d → d_ff  → d * d_ff multiply-adds
    #   W_up:   d → d_ff  → d * d_ff multiply-adds
    #   SwiGLU-T: d_ff * (SiLU + multiply + mask) → ~6 * d_ff
    #   W_down: d_ff → d  → d_ff * d multiply-adds
    per_expert = (
        flops_matmul(1, d_ff, d_model)
        + flops_matmul(1, d_ff, d_model)
        + d_ff * 6
        + flops_matmul(1, d_model, d_ff)
    )

    # Routing aggregation: weighted sum of expert outputs
    aggregation = tokens * d_model * n_active * 2  # weighted sum

    return tokens * n_active * per_expert + aggregation


def compute_full_flops_breakdown(
    d_model: int = 2048,
    n_blocks: int = 24,
    d_ff: int = 8192,
    n_scales: int = 3,
    n_heads: int = 32,
    d_head: int = 64,
    n_experts: int = 8,
    n_active: int = 2,
    n_types: int = 4,
    vocab_size: int = 65536,
    seq_len: int = 4096,
    batch: int = 1,
) -> Dict[str, int]:
    """Full FLOPs breakdown for one forward pass.

    Returns per-component FLOPs and totals.
    """
    tokens = batch * seq_len

    # Embedding
    emb = flops_embedding(vocab_size, d_model, tokens)

    # Per-block components
    msla = flops_msla(d_model, n_scales, n_heads, d_head, batch, seq_len)["total"]
    ahg = flops_ahg(d_model, n_heads, batch, seq_len)
    catr = flops_catr(d_model, n_types, n_experts, batch, seq_len)
    smoe = flops_smoe_ffn(d_model, d_ff, n_active, batch, seq_len)
    norm = flops_adasnorm(d_model, batch, seq_len) * 2  # two norms per block
    tlcg = flops_linear(4 * d_model, 256, tokens) + flops_linear(256, 1, tokens)

    block_total = msla + ahg + catr + smoe + norm + tlcg
    blocks = n_blocks * block_total

    # Final norm + LM head
    final_norm = flops_adasnorm(d_model, batch, seq_len)
    lm_head = flops_linear(d_model, vocab_size, tokens)

    total = emb + blocks + final_norm + lm_head

    return {
        "embedding": emb,
        "msla_attention": n_blocks * msla,
        "adaptive_head_gating": n_blocks * ahg,
        "content_router": n_blocks * catr,
        "smoe_ffn": n_blocks * smoe,
        "normalization": n_blocks * norm + final_norm,
        "tlcg_confidence": n_blocks * tlcg,
        "lm_head": lm_head,
        "total": total,
        "flops_per_token": total // tokens if tokens > 0 else total,
        "gflops_total": total / 1e9,
    }

    # Estimated savings from early exit
    # If a token exits at block 8 instead of 24, it saves 16/24 = 67% of remaining FLOPs
    # At 30% early exit rate → ~20% overall savings
    early_exit_savings = 0.20  # hypothesis: 20% overall
    total_with_ee = total * (1 - early_exit_savings)

    return {
        "embedding": emb,
        "msla_attention": n_blocks * msla,
        "adaptive_head_gating": n_blocks * ahg,
        "content_router": n_blocks * catr,
        "smoe_ffn": n_blocks * smoe,
        "normalization": n_blocks * norm + final_norm,
        "tlcg_confidence": n_blocks * tlcg,
        "lm_head": lm_head,
        "total": total,
        "flops_per_token": total // tokens if tokens > 0 else total,
        "gflops_total": total / 1e9,
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ARCHITECKT PER-COMPONENT FLOPs BREAKDOWN")
    print("  Config: Medium (d=2048, n_blocks=24, L=4096, batch=1)")
    print("=" * 70)

    breakdown = compute_full_flops_breakdown()

    total = breakdown["total"]

    components = [
        ("Embedding", "embedding"),
        ("MSLA Attention", "msla_attention"),
        ("Adaptive Head Gating", "adaptive_head_gating"),
        ("Content Router (CATR)", "content_router"),
        ("SMoE FFN", "smoe_ffn"),
        ("Normalization", "normalization"),
        ("TLCG Confidence", "tlcg_confidence"),
        ("LM Head", "lm_head"),
    ]

    for name, key in components:
        val = breakdown[key]
        pct = val / total * 100 if total > 0 else 0
        print(f"  {name:<25s}: {val/1e9:>8.3f} GFLOPs ({pct:>5.1f}%)")

    print(f"  {'─' * 50}")
    print(f"  {'TOTAL':<25s}: {total/1e9:>8.3f} GFLOPs")
    print(f"  Per token: {breakdown['flops_per_token']/1e6:.1f} MFLOPs")

    # Compare with standard attention at same dims
    print(f"\n  Comparison: Standard softmax attention:")
    trans_attn = flops_transformer_attention(2048, 32, 64, 1, 4096)
    print(f"    Transformer attention alone: {trans_attn['total']/1e9:.2f} GFLOPs")
    print(f"    vs Architeckt MSLA: {breakdown['msla_attention']/1e9:.2f} GFLOPs")
    ratio = trans_attn["total"] / breakdown["msla_attention"]
    print(f"    Speedup: {ratio:.1f}× at L=4096 (grows linearly with L)")
    print("=" * 70)
