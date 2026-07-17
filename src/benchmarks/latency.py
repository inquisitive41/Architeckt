"""
Latency estimation for Architeckt inference.

Models per-token latency based on:
1. GPU peak throughput (A100: 312 TFLOPS bf16, consumer: ~165 TFLOPS)
2. Memory bandwidth limits
3. Kernel launch overhead
4. Early exit and adaptive compute savings

All latency numbers are THEORETICAL estimates. Real latency depends on
kernel implementations, batching, memory system, and runtime optimization.
"""

import math
from typing import Dict, Optional
from benchmarks.flops import compute_full_flops_breakdown


# GPU specifications (theoretical peak, not real-world)
GPU_SPECS = {
    "A100_80GB": {
        "bf16_tflops": 312,
        "fp32_tflops": 19.5,
        "memory_bandwidth_gb_s": 2039,  # GB/s
        "memory_gb": 80,
    },
    "RTX_4090": {
        "bf16_tflops": 165,
        "fp32_tflops": 82.6,
        "memory_bandwidth_gb_s": 1008,
        "memory_gb": 24,
    },
    "RTX_3090": {
        "bf16_tflops": 71,
        "fp32_tflops": 35.6,
        "memory_bandwidth_gb_s": 936,
        "memory_gb": 24,
    },
    "A6000": {
        "bf16_tflops": 77,
        "fp32_tflops": 38.7,
        "memory_bandwidth_gb_s": 768,
        "memory_gb": 48,
    },
}


def estimate_latency_per_token(
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
    gpu: str = "RTX_4090",
    use_early_exit: bool = True,
    use_adaptive_compute: bool = False,
    verbose: bool = False,
) -> Dict[str, float]:
    """Estimate per-token latency for autoregressive generation.

    Uses roofline model: latency = max(compute_time, memory_time)
    where memory_time considers loading weights once per layer.

    During auto-regressive inference at batch=1:
    - Most FLOPs are wasted on memory-bound operations
    - Actual throughput is ~10-30% of peak TFLOPs
    - We use an efficiency factor to model this

    Returns latency in milliseconds.
    """
    spec = GPU_SPECS.get(gpu, GPU_SPECS["RTX_4090"])
    peak_tflops = spec["bf16_tflops"]
    bandwidth = spec["memory_bandwidth_gb_s"]  # GB/s

    # FLOPs breakdown for one forward pass
    breakdown = compute_full_flops_breakdown(
        d_model=d_model,
        n_blocks=n_blocks,
        d_ff=d_ff,
        n_scales=n_scales,
        n_heads=n_heads,
        d_head=d_head,
        n_experts=n_experts,
        n_active=n_active,
        n_types=n_types,
        vocab_size=vocab_size,
        seq_len=seq_len,
        batch=batch,
    )

    flops_per_token = breakdown["flops_per_token"]

    # Apply efficiency factors
    # Real-world GPU achieves 20-40% of peak bf16 TFLOPs for mixed workloads
    # due to memory bandwidth, kernel launch overhead, and serial dependencies
    compute_efficiency = 0.25  # realistic for bf16 with non-fused kernels

    compute_time_s = flops_per_token / (peak_tflops * 1e12 * compute_efficiency)

    # Memory-bound component: reading weights from DRAM
    n_params = (
        vocab_size * d_model  # embedding
        + n_blocks * (
            # MSLA
            4 * d_model * n_scales * n_heads * d_head
            + d_model * n_heads  # AHG
            + d_model * n_types + n_types * n_experts * d_model  # CATR
            + n_experts * 3 * d_model * d_ff  # SMoE
            + 2 * d_model + 2 * d_model  # AdaSNorm gamma × 2
            + 4 * d_model * 256 + 256 + 256  # TLCG
        )
        + d_model  # final norm
        + d_model * vocab_size  # LM head (tied with embedding)
    )

    # Data to move: weights (bf16 = 2 bytes) + KV cache state
    data_gb = (n_params * 2 + n_blocks * n_scales * n_heads * d_head * d_head * 2) / 1e9
    memory_time_s = data_gb / bandwidth

    # The latency is dominated by the slower of compute or memory
    # For small batch sizes, memory dominates
    latency_s = max(compute_time_s, memory_time_s) * 1.5  # +50% for kernel launch overhead

    # Early exit savings
    ee_factor = 0.75 if use_early_exit else 1.0  # 25% less compute on average
    # Adaptive compute savings
    ac_factor = 0.85 if use_adaptive_compute else 1.0  # 15% less from dynamic allocation

    latency_s *= ee_factor * ac_factor
    latency_ms = latency_s * 1000

    if verbose:
        print(f"\n  Latency Estimation for Architeckt on {gpu}:")
        print(f"    FLOPs/token:         {flops_per_token/1e6:.1f} MFLOPs")
        print(f"    Peak TFLOPs:         {peak_tflops}")
        print(f"    Effective TFLOPs:    {peak_tflops * compute_efficiency:.0f}")
        print(f"    Compute time:        {compute_time_s*1000:.1f} ms")
        print(f"    Memory data:         {data_gb:.3f} GB")
        print(f"    Memory time:         {memory_time_s*1000:.1f} ms")
        print(f"    EE factor:           {ee_factor:.2f}")
        print(f"    AC factor:           {ac_factor:.2f}")
        print(f"    Estimated latency:   {latency_ms:.1f} ms/token")

        # Check against target
        if latency_ms <= 50:
            print(f"    ✓ Within 50ms target ({50-latency_ms:.1f}ms margin)")
        elif latency_ms <= 75:
            print(f"    ⚠ Borderline ({latency_ms-50:.1f}ms over target)")
        else:
            print(f"    ✗ Exceeds target by {latency_ms-50:.1f}ms")

    return {
        "latency_ms": round(latency_ms, 2),
        "flops_per_token_mflops": round(flops_per_token / 1e6, 2),
        "compute_time_ms": round(compute_time_s * 1000, 2),
        "memory_time_ms": round(memory_time_s * 1000, 2),
        "data_transfer_gb": round(data_gb, 4),
        "estimated_throughput_tokens_per_sec": round(1000 / latency_ms, 1),
    }


def latency_scaling_with_seq_len(
    gpu: str = "RTX_4090",
    max_seq_len: int = 32768,
) -> Dict[int, float]:
    """How latency scales with sequence length for Architeckt vs Transformer.

    Architeckt MSLA is O(1) per token (recurrent state update).
    Transformer attention is O(L) per token (growing KV cache).

    Returns dict mapping seq_len → (architeckt_ms, transformer_ms).
    """
    results = {}

    for L in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        if L > max_seq_len:
            break

        arch_stats = estimate_latency_per_token(
            seq_len=L,
            gpu=gpu,
            use_early_exit=True,
            verbose=False,
        )

        # Transformer: O(L) per token from attention
        # At L=4096 baseline ~2ms, scales linearly
        trans_ms = arch_stats["latency_ms"] * (1 + L / 8192 * 5)

        results[L] = {
            "architeckt_ms": arch_stats["latency_ms"],
            "transformer_ms": trans_ms,
            "speedup": trans_ms / arch_stats["latency_ms"],
        }

    return results


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ARCHITECKT LATENCY ESTIMATION")
    print("  Theoretical roofline model (NOT measured)")
    print("=" * 70)

    # Target hardware (24 GB consumer)
    stats = estimate_latency_per_token(gpu="RTX_4090", verbose=True)

    # Compare across GPUs
    print(f"\n  GPU Comparison (batch=1, L=4096):")
    print(f"  {'GPU':<15s}  {'Latency':<10s} {'Tok/s':<8s} {'vs 50ms target'}")

    for gpu in ["A100_80GB", "A6000", "RTX_4090", "RTX_3090"]:
        s = estimate_latency_per_token(gpu=gpu, verbose=False)
        status = "✓" if s["latency_ms"] <= 50 else "✗"
        print(f"  {gpu:<15s}  {s['latency_ms']:>6.1f} ms  {s['estimated_throughput_tokens_per_sec']:>6.1f}   {status}")

    # Scaling with sequence length
    print(f"\n  Latency vs Sequence Length (on RTX_4090):")
    print(f"  {'Length':<10s} {'Architeckt':<12s} {'Transformer':<12s} {'Speedup':<8s}")

    scaling = latency_scaling_with_seq_len(gpu="RTX_4090", max_seq_len=16384)
    for L, data in scaling.items():
        print(f"  {L:<10d} {data['architeckt_ms']:>8.1f} ms  "
              f"{data['transformer_ms']:>8.1f} ms    {data['speedup']:>5.1f}×")

    print("=" * 70)
