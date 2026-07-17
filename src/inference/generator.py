"""
Architeckt Inference Engine

Optimized inference with:
- KV caching for linear attention (recurrent state, not growing matrix)
- Early exit for computational savings
- Confidence-based error detection
- Batching support
- Memory-efficient decoding

Latency target: ≤50 ms/token at batch=1 on 24GB consumer GPU.
Memory target: ≤7.5 GB total for 1.5B parameter model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import time


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    use_early_exit: bool = True
    use_confidence_verification: bool = True
    confidence_threshold: float = 0.5
    stop_tokens: List[int] = None

    def __post_init__(self):
        if self.stop_tokens is None:
            self.stop_tokens = [0]  # EOS token


class ArchitecktInference:
    """Efficient inference wrapper for Architeckt models.

    Features:
    - Temperature + top-p + top-k sampling
    - Repetition penalty
    - Early exit with dynamic entropy threshold
    - Confidence-based error resilience
    - KV caching for linear attention
    - Performance monitoring
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[InferenceConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config or InferenceConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Performance tracking
        self.latency_history: List[float] = []
        self.total_tokens_generated: int = 0
        self.early_exit_count: int = 0

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tokenizer,  # callable: str -> List[int] and List[int] -> str
        verbose: bool = False,
    ) -> Tuple[str, Dict[str, float]]:
        """Generate text from a prompt string.

        Args:
            prompt: input text
            tokenizer: tokenizer object with encode/decode methods
            verbose: print generation details

        Returns:
            generated_text: the full text including prompt
            stats: dict with generation statistics
        """
        # Tokenize
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate
        start_time = time.time()
        output_ids, gen_stats = self.model.generate(
            input_ids=input_tensor,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            use_early_exit=self.config.use_early_exit,
            use_confidence_verification=self.config.use_confidence_verification,
            confidence_threshold=self.config.confidence_threshold,
            verbose=verbose,
        )
        end_time = time.time()

        # Decode
        generated_text = tokenizer.decode(output_ids[0].tolist())

        # Record metrics
        total_time = end_time - start_time
        tokens_generated = gen_stats["tokens_generated"]
        latency_per_token = total_time / tokens_generated * 1000 if tokens_generated > 0 else 0

        self.latency_history.append(latency_per_token)
        self.total_tokens_generated += tokens_generated
        self.early_exit_count += gen_stats.get("early_exit_rate", 0) * tokens_generated

        stats = {
            "tokens_generated": tokens_generated,
            "total_time_s": total_time,
            "latency_ms_per_token": latency_per_token,
            "tokens_per_second": tokens_generated / total_time if total_time > 0 else 0,
            "avg_confidence": gen_stats["avg_confidence"],
            "early_exit_rate": gen_stats.get("early_exit_rate", 0.0),
            "early_exit_threshold": gen_stats.get("early_exit_threshold", 0.0),
        }

        if verbose:
            print(f"\n--- Generation Stats ---")
            print(f"  Tokens: {tokens_generated}")
            print(f"  Time: {total_time:.2f}s ({latency_per_token:.1f} ms/token)")
            print(f"  Speed: {tokens_generated/total_time:.1f} tok/s" if total_time > 0 else "")
            print(f"  Confidence: {gen_stats['avg_confidence']:.3f}")
            print(f"  Early exits: {gen_stats.get('early_exit_rate', 0)*100:.1f}%")

        return generated_text, stats

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        tokenizer,
        verbose: bool = False,
    ) -> Tuple[List[str], Dict[str, float]]:
        """Batch generation for multiple prompts.

        Pads shorter sequences and generates simultaneously.

        Args:
            prompts: list of input texts
            tokenizer: tokenizer with encode/decode

        Returns:
            generated_texts: list of generated texts
            stats: aggregate generation statistics
        """
        batch_size = len(prompts)

        # Tokenize all prompts
        encoded = [tokenizer.encode(p) for p in prompts]
        max_len = max(len(e) for e in encoded)

        # Pad to equal length
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        for i, e in enumerate(encoded):
            input_ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)

        # Generate
        start_time = time.time()
        output_ids, gen_stats = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            use_early_exit=self.config.use_early_exit,
            use_confidence_verification=self.config.use_confidence_verification,
            confidence_threshold=self.config.confidence_threshold,
            verbose=verbose,
        )
        end_time = time.time()

        # Decode each sequence
        texts = [tokenizer.decode(ids.tolist()) for ids in output_ids]

        total_time = end_time - start_time
        tokens_generated = gen_stats["tokens_generated"]
        # Note: gen_stats["tokens_generated"] counts total tokens generated per batch member

        stats = {
            "batch_size": batch_size,
            "tokens_generated_per_seq": tokens_generated,
            "total_tokens": tokens_generated * batch_size,
            "total_time_s": total_time,
            "latency_ms_per_token": total_time / (tokens_generated * batch_size) * 1000,
            "avg_confidence": gen_stats["avg_confidence"],
            "early_exit_rate": gen_stats.get("early_exit_rate", 0.0),
        }

        return texts, stats

    def estimate_memory_usage(self, batch_size: int = 1, seq_len: int = 2048) -> Dict[str, float]:
        """Estimate GPU memory usage for inference.

        Returns estimated memory in GB for:
        - Model weights
        - KV cache (linear attention state)
        - Activations
        - Total
        """
        n_params = self.model.get_num_params()

        # Weights in bf16
        weights_gb = n_params * 2 / 1e9

        # Linear attention KV state (not a matrix!)
        # For each scale: (d_head × d_head) state matrix
        # This is O(1) per layer, NOT O(L)
        cfg = self.model.config
        kv_cache_gb = (
            cfg.n_blocks
            * cfg.n_scales
            * cfg.max_heads
            * cfg.d_head * cfg.d_head
            * 2  # bytes
            * batch_size
            / 1e9
        )

        # Activations (rough estimate)
        # For auto-regressive: one token at a time, plus transient intermediates
        activations_gb = (
            cfg.n_blocks
            * batch_size * seq_len
            * cfg.d_model
            * 10  # overhead factor for intermediate tensors
            * 2
            / 1e9
        )

        total_gb = weights_gb + kv_cache_gb + activations_gb

        return {
            "weights_gb": round(weights_gb, 2),
            "kv_cache_gb": round(kv_cache_gb, 4),
            "activations_gb": round(activations_gb, 2),
            "total_gb": round(total_gb, 2),
            "headroom_24gb": round(24 - total_gb, 2),
        }

    def get_performance_report(self) -> Dict[str, float]:
        """Get cumulative performance statistics."""
        if not self.latency_history:
            return {"error": "No generations performed yet"}

        latencies = self.latency_history
        return {
            "mean_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": sorted(latencies)[len(latencies)//2],
            "p95_latency_ms": sorted(latencies)[int(len(latencies)*0.95)],
            "p99_latency_ms": sorted(latencies)[int(len(latencies)*0.99)],
            "total_tokens": self.total_tokens_generated,
            "early_exit_fraction": self.early_exit_count / self.total_tokens_generated
                if self.total_tokens_generated > 0 else 0,
        }
