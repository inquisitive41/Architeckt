"""
Detailed memory analysis for Architeckt.

Tracks memory usage by component for:
1. Training (fp32 optimizer + bf16 model + activations)
2. Inference (bf16 model + KV state + activations)

Verifies:
- Training fits in 8×A100 80GB
- Inference fits in 24GB consumer GPU
- KV cache is truly O(1) per layer (not O(L) like Transformers)
"""

import torch
import math
from typing import Dict, Tuple
from dataclasses import dataclass


BYTES_PER_PARAM = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
}


@dataclass
class MemoryBudget:
    """Memory breakdown by component."""
    embedding_gb: float
    attention_gb: float
    ffn_gb: float
    routing_gb: float
    normalization_gb: float
    tlcg_gb: float
    lm_head_gb: float
    early_exit_gb: float
    total_params_gb: float


def compute_model_memory(
    d_model: int = 2048,
    n_blocks: int = 24,
    vocab_size: int = 65536,
    n_scales: int = 3,
    n_heads: int = 32,
    d_head: int = 64,
    n_experts: int = 8,
    d_ff: int = 8192,
    n_types: int = 4,
    exit_layers: Tuple[int, ...] = (8, 16, 24),
    dtype: str = "bf16",
) -> MemoryBudget:
    """Compute memory for model parameters only (no optimizer/activations)."""
    bpe = BYTES_PER_PARAM[dtype]

    # Embedding (tied with LM head, counted once)
    embedding = vocab_size * d_model * bpe / 1e9

    # Per-block components
    # MSLA: Q, K, V, O projections + scale weights
    msla_per_block = (
        4 * d_model * n_scales * n_heads * d_head * bpe  # projections
        + n_scales * bpe  # scale weights
    ) / 1e9

    # AHG: gate projection
    ahg_per_block = (d_model * n_heads + n_heads) * bpe / 1e9

    # CATR: type classifier + per-type routers
    catr_per_block = (
        d_model * n_types * bpe  # type classifier
        + n_types * n_experts * d_model * bpe  # per-type routers
    ) / 1e9

    # SMoE: gate, up, down per expert + thresholds
    smoe_per_block = (
        n_experts * 3 * d_model * d_ff * bpe  # W_gate, W_up, W_down
        + n_experts * d_ff * bpe  # thresholds
        + bpe  # log_beta
    ) / 1e9

    # AdaSNorm × 2: gamma × 2 + stats_net
    norm_per_block = (
        2 * d_model * bpe  # gamma × 2
        + 2 * (3 * 32 + 32 + 32 * 1 + 1) * bpe  # stats_net × 2
    ) / 1e9

    # TLCG: confidence_net
    tlcg_per_block = (
        (4 * d_model * 256 + 256 + 256 * 1 + 1) * bpe
    ) / 1e9

    block_total = (msla_per_block + ahg_per_block + catr_per_block
                   + smoe_per_block + norm_per_block + tlcg_per_block)

    # Final norm
    final_norm = d_model * bpe / 1e9

    # LM head (tied with embedding — 0 additional)
    lm_head = 0.0

    # Early exit heads
    early_exit = len(exit_layers) * (d_model * vocab_size * bpe) / 1e9

    total = embedding + n_blocks * block_total + final_norm + lm_head + early_exit

    return MemoryBudget(
        embedding_gb=round(embedding, 3),
        attention_gb=round(n_blocks * msla_per_block, 3),
        ffn_gb=round(n_blocks * smoe_per_block, 3),
        routing_gb=round(n_blocks * (ahg_per_block + catr_per_block), 3),
        normalization_gb=round(n_blocks * norm_per_block + final_norm, 3),
        tlcg_gb=round(n_blocks * tlcg_per_block, 3),
        lm_head_gb=round(lm_head, 3),
        early_exit_gb=round(early_exit, 3),
        total_params_gb=round(total, 3),
    )


def compute_training_memory(
    d_model: int = 2048,
    n_blocks: int = 24,
    vocab_size: int = 65536,
    n_experts: int = 8,
    d_ff: int = 8192,
    batch_size: int = 2,
    seq_len: int = 4096,
    activation_checkpointing: bool = True,
) -> Dict[str, float]:
    """Estimate training memory per GPU (A100 80GB target).

    Components:
    - Model weights (bf16): 2 bytes/param
    - Optimizer states (fp32): 12 bytes/param (AdamW: param + m + v)
    - Gradients (fp32): 4 bytes/param
    - Activations: depends on batch, seq_len, checkpointing

    For the target medium config (1.5B params):
        Weights:         ~3.0 GB
        Optimizer:       ~18.0 GB
        Gradients:       ~6.0 GB
        Total:           ~27.0 GB (minimum)

    Activations add:
        Without checkpointing: ~40-60 GB
        With checkpointing:    ~8-12 GB
        Total with CKPT:       ~35-39 GB → fits in 80 GB
    """
    model_mem = compute_model_memory(d_model, n_blocks, vocab_size)

    # Total params from the model memory (bf16 → bytes)
    n_params = int(model_mem.total_params_gb / 2 * 1e9)  # rough

    weights_gb = model_mem.total_params_gb

    # Optimizer: AdamW stores fp32 param + 2 momentum buffers = 12 bytes/param
    optimizer_gb = n_params * 12 / 1e9

    # Gradients in fp32
    gradients_gb = n_params * 4 / 1e9

    # Activations: depends on batch × seq × d × layers × overhead
    # Without checkpointing: each layer stores its full input
    # With checkpointing: only checkpoints every ~3 layers
    tokens = batch_size * seq_len
    layer_activation_bytes = (
        tokens * d_model * 2  # bf16
        + tokens * 32 * 64 * 3 * 2  # attention intermediates per scale
        + tokens * d_ff * 2  # FFN intermediate
        + tokens * d_model * 2  # norms
    )

    if activation_checkpointing:
        # Only store checkpoints (1/3 of total)
        checkpoint_overhead = 3
        activations_gb = layer_activation_bytes * n_blocks / checkpoint_overhead / 1e9 * 2
    else:
        activations_gb = layer_activation_bytes * n_blocks / 1e9 * 2

    total = weights_gb + optimizer_gb + gradients_gb + activations_gb + 2.0  # +2GB overhead

    return {
        "model_weights_gb": round(weights_gb, 2),
        "optimizer_states_gb": round(optimizer_gb, 2),
        "gradients_gb": round(gradients_gb, 2),
        "activations_gb": round(activations_gb, 2),
        "frameworks_overhead_gb": 2.0,
        "total_gb": round(total, 2),
        "fits_in_80gb": total <= 80,
        "remaining_gb": round(80 - total, 2),
    }


def compute_inference_memory(
    d_model: int = 2048,
    n_blocks: int = 24,
    vocab_size: int = 65536,
    n_scales: int = 3,
    n_heads: int = 32,
    d_head: int = 64,
    n_experts: int = 8,
    d_ff: int = 8192,
    seq_len: int = 4096,
    batch: int = 1,
    dtype: str = "bf16",
) -> Dict[str, float]:
    """Estimate inference memory (consumer GPU, 24GB target).

    During inference:
    - Model weights stay in bf16
    - No optimizer/gradient states
    - KV cache is O(1) per layer (linear attention)
    - Activations are single-token (auto-regressive)
    """
    model_mem = compute_model_memory(d_model, n_blocks, vocab_size, dtype=dtype)

    bpe = BYTES_PER_PARAM[dtype]
    weights_gb = model_mem.total_params_gb

    # KV cache: for linear attention, it's O(1) per layer
    # Each scale × head maintains a (d_head × d_head) state matrix
    # Plus normalizer vector of size d_head
    kv_cache_bytes = (
        n_blocks
        * n_scales
        * n_heads
        * (d_head * d_head + d_head)  # state matrix + normalizer
        * bpe
        * batch
    )
    kv_cache_gb = kv_cache_bytes / 1e9

    # Activations: per-token for auto-regressive
    # Even with full batch, only the current token + residual streams
    activations_bytes = (
        n_blocks
        * d_model              # hidden state
        * (3 + 10)              # intermediate tensors overhead factor
        * bpe
        * batch
    )
    activations_gb = activations_bytes / 1e9 * (seq_len / 4096)  # scale with seq_len

    # Early exit heads (in memory but small)
    exit_heads_gb = model_mem.early_exit_gb

    # Framework overhead (CUDA context, allocator, etc.)
    overhead_gb = 1.5 + batch * 0.2

    total = weights_gb + kv_cache_gb + activations_gb + exit_heads_gb + overhead_gb

    return {
        "model_weights_gb": round(weights_gb, 2),
        "kv_cache_gb": round(kv_cache_gb, 4),
        "activations_gb": round(activations_gb, 2),
        "early_exit_heads_gb": round(exit_heads_gb, 3),
        "framework_overhead_gb": round(overhead_gb, 2),
        "total_gb": round(total, 2),
        "fits_in_24gb": total <= 24,
        "remaining_gb": round(24 - total, 2),
    }


def kv_cache_comparison(
    n_blocks: int = 24,
    d_model: int = 2048,
    n_heads: int = 32,
    d_head: int = 64,
    n_scales: int = 3,
    batch: int = 1,
    max_seq_len: int = 32768,
    dtype: str = "bf16",
) -> Dict[int, Dict[str, float]]:
    """Compare KV cache memory between Architeckt and Transformer.

    Transformer: 2 × n_blocks × n_heads × d_head × seq_len bytes (grows with L)
    Architeckt: n_blocks × n_scales × n_heads × d_head² bytes (constant)

    Returns dict mapping seq_len → {architeckt_mb, transformer_mb, savings_ratio}
    """
    bpe = BYTES_PER_PARAM[dtype]
    results = {}

    for L in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        if L > max_seq_len:
            break

        arch_bytes = n_blocks * n_scales * n_heads * (d_head * d_head + d_head) * bpe * batch
        trans_bytes = 2 * n_blocks * n_heads * d_head * L * bpe * batch

        results[L] = {
            "architeckt_mb": round(arch_bytes / 1e6, 2),
            "transformer_mb": round(trans_bytes / 1e6, 2),
            "savings_ratio": round(trans_bytes / arch_bytes, 1) if arch_bytes > 0 else float("inf"),
        }

    return results


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ARCHITECKT MEMORY ANALYSIS")
    print("=" * 70)

    # Model parameter memory
    model_mem = compute_model_memory()
    print(f"\n  Model Parameters (bf16):")
    print(f"    Embedding:         {model_mem.embedding_gb:>6.2f} GB")
    print(f"    MSLA Attention:    {model_mem.attention_gb:>6.2f} GB")
    print(f"    SMoE FFN:          {model_mem.ffn_gb:>6.2f} GB")
    print(f"    Routing (AHG+CATR):{model_mem.routing_gb:>6.2f} GB")
    print(f"    Normalization:     {model_mem.normalization_gb:>6.2f} GB")
    print(f"    TLCG Confidence:   {model_mem.tlcg_gb:>6.2f} GB")
    print(f"    Early Exit Heads:  {model_mem.early_exit_gb:>6.2f} GB")
    print(f"    ──────────────────────────")
    print(f"    TOTAL:              {model_mem.total_params_gb:>6.2f} GB ({model_mem.total_params_gb/2*1e3:.0f}M params)")

    # Training memory
    train_mem = compute_training_memory(activation_checkpointing=True)
    print(f"\n  Training Memory (per A100 80GB, with activation checkpointing):")
    print(f"    Model weights:     {train_mem['model_weights_gb']:>6.2f} GB")
    print(f"    Optimizer states:  {train_mem['optimizer_states_gb']:>6.2f} GB")
    print(f"    Gradients:         {train_mem['gradients_gb']:>6.2f} GB")
    print(f"    Activations:       {train_mem['activations_gb']:>6.2f} GB")
    print(f"    Framework overhead:{train_mem['frameworks_overhead_gb']:>6.2f} GB")
    print(f"    ──────────────────────────")
    print(f"    TOTAL:              {train_mem['total_gb']:>6.2f} GB")
    print(f"    Fits in 80GB:       {'✓' if train_mem['fits_in_80gb'] else '✗'} ({train_mem['remaining_gb']:.1f} GB remaining)")

    # Without checkpointing
    train_no_ckpt = compute_training_memory(activation_checkpointing=False)
    print(f"\n    Without activation checkpointing: {train_no_ckpt['total_gb']:.1f} GB"
          f" {'✓' if train_no_ckpt['fits_in_80gb'] else '✗'}")

    # Inference memory
    inf_mem = compute_inference_memory(seq_len=4096)
    print(f"\n  Inference Memory (RTX 4090 24GB, batch=1, L=4096):")
    print(f"    Model weights:     {inf_mem['model_weights_gb']:>6.2f} GB")
    print(f"    KV cache:          {inf_mem['kv_cache_gb']:>6.4f} GB")
    print(f"    Activations:       {inf_mem['activations_gb']:>6.2f} GB")
    print(f"    Framework overhead:{inf_mem['framework_overhead_gb']:>6.2f} GB")
    print(f"    ──────────────────────────")
    print(f"    TOTAL:              {inf_mem['total_gb']:>6.2f} GB")
    print(f"    Fits in 24GB:       {'✓' if inf_mem['fits_in_24gb'] else '✗'} ({inf_mem['remaining_gb']:.1f} GB remaining)")

    # Inference at longer sequences
    for L in [8192, 16384]:
        inf_L = compute_inference_memory(seq_len=L)
        print(f"    At L={L}:            {inf_L['total_gb']:.2f} GB {'✓' if inf_L['fits_in_24gb'] else '✗'}")

    # KV cache comparison
    print(f"\n  KV Cache Comparison (Architeckt vs Transformer):")
    print(f"  {'Seq Len':<10s} {'Arch MB':<10s} {'Trans MB':<10s} {'Savings':<8s}")
    kv_comp = kv_cache_comparison(max_seq_len=16384)
    for L, data in kv_comp.items():
        print(f"  {L:<10d} {data['architeckt_mb']:>8.2f}   {data['transformer_mb']:>8.1f}   {data['savings_ratio']:>5.0f}×")

    print("=" * 70)
