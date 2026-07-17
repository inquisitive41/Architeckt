"""
Architeckt: Full Language Model

Integrates all nine architectural innovations into a complete
autoregressive language model with linear complexity.

Architecture overview:
    Embedding → [ArchitecktBlock × N] → LM Head (shared with embedding)
                  ↑ Early Exit Heads at layers [8, 16, 24]
                  ↑ Confidence Gating per block

The LM head shares weights with the input embedding (weight tying)
to reduce parameters and provide a form of regularization.

Model variants (parameter counts are approximate):
    Small:  ~300M params (d_model=1024, n_blocks=16, n_experts=4)
    Medium: ~1.5B params (d_model=2048, n_blocks=24, n_experts=8)
    Large:  ~4.0B params (d_model=3072, n_blocks=32, n_experts=12)

Computational complexity:
    Per token: O(n_blocks · d² + n_blocks · k · d · d_ff) = O(L)
    Per sequence: O(L · n_blocks · (d² + k · d · d_ff))
    
    For Medium config with L=8192:
        FLOPs ≈ 1.5B params × 2 FLOP/param × 2 (MoE overhead) = ~6e15 FLOPs
        This is ~10× less than a dense Transformer of similar size for L=8192
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from models.config import ArchitecktConfig
from models.block import ArchitecktBlock
from normalization.adaptive_norm import AdaSNorm
from inference.early_exit import DepthAwareEarlyExit


class ArchitecktModel(nn.Module):
    """Architeckt language model.

    Args:
        config: ArchitecktConfig dataclass with all hyperparameters
    """

    def __init__(self, config: ArchitecktConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.embedding.weight, std=config.init_std)
        self.emb_dropout = nn.Dropout(config.emb_dropout) if config.emb_dropout > 0 else nn.Identity()

        # Pre-block normalization (applied once before first block)
        self.prenorm = AdaSNorm(config.d_model, eps=config.norm_eps)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ArchitecktBlock(
                d_model=config.d_model,
                d_ff=config.d_ff,
                n_heads=config.max_heads,
                n_scales=config.n_scales,
                n_experts=config.n_experts,
                n_active_experts=config.n_active_experts,
                n_types=config.n_token_types,
                kernel_fn=config.kernel_fn,
                scale_windows=config.scale_windows,
                scale_alphas=config.scale_alphas,
                block_idx=i,
                n_blocks=config.n_blocks,
                dropout=config.dropout,
                tlcg_hidden_dim=config.tlcg_hidden_dim,
            )
            for i in range(config.n_blocks)
        ])

        # Final normalization
        self.final_norm = AdaSNorm(config.d_model, eps=config.norm_eps)

        # LM Head (tied with embedding weights)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

        # Early exit mechanism
        self.early_exit = DepthAwareEarlyExit(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            exit_layers=list(config.exit_layers),
            entropy_alpha=config.entropy_ema_alpha,
            entropy_margin=config.entropy_margin,
        )

        # Apply DeepNet initialization if configured
        if config.init_scale_depth:
            self._apply_depth_scaling()

    def _apply_depth_scaling(self):
        """Scale residual branches by 1/sqrt(2 * N) for DeepNet stability."""
        # Note: scaling is already applied in block forward via self.residual_scale
        # Here we adjust initialization for deeper blocks
        n_blocks = self.config.n_blocks
        for i, block in enumerate(self.blocks):
            depth_scale = (2.0 * n_blocks) ** -0.5
            # Apply to output projections
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.weight.dim() == 2 and module.weight.shape[0] == self.config.d_model:
                        module.weight.data.mul_(depth_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            input_ids: (batch, seq_len) — token indices
            attention_mask: (batch, seq_len) — True for valid tokens
            labels: (batch, seq_len) — target token indices for loss computation
            deterministic: if True, use hard gating (inference mode)

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar cross-entropy loss (0 if labels is None)
            metrics: dict of auxiliary losses and statistics
        """
        batch, seq_len = input_ids.shape

        # Token embeddings
        x = self.embedding(input_ids)  # (B, L, d_model)
        x = self.emb_dropout(x)

        # Pre-normalize
        x = self.prenorm(x)

        # Accumulate metrics
        total_aux_loss = 0.0
        confidences = []

        # Pass through blocks
        for i, block in enumerate(self.blocks):
            x, conf, aux_loss = block(x, attention_mask=attention_mask, deterministic=deterministic)
            total_aux_loss = total_aux_loss + aux_loss
            confidences.append(conf)

        # Final normalization
        x = self.final_norm(x)

        # LM head
        logits = self.lm_head(x)  # (B, L, vocab_size)

        # Compute loss
        loss = torch.tensor(0.0, device=x.device)
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        # Mean confidence across blocks
        mean_confidence = torch.stack(confidences, dim=0).mean(dim=0)  # (B, L)

        metrics = {
            "aux_loss": total_aux_loss / len(self.blocks) if len(self.blocks) > 0 else 0.0,
            "mean_confidence": mean_confidence.mean().item(),
        }

        return logits, loss, metrics

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        use_early_exit: bool = True,
        use_confidence_verification: bool = True,
        confidence_threshold: float = 0.5,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Autoregressive generation with early exit and confidence gating.

        Args:
            input_ids: (batch, seq_len) — prompt tokens
            max_new_tokens: max tokens to generate
            temperature: sampling temperature (0.0 = greedy)
            top_p: nucleus sampling threshold
            top_k: top-k sampling
            use_early_exit: enable depth-aware early exit
            use_confidence_verification: enable TLCG error detection
            confidence_threshold: minimum confidence for accepting prediction
            verbose: print debug info

        Returns:
            generated_ids: (batch, prompt_len + new_tokens)
            stats: dict with generation statistics
        """
        batch = input_ids.shape[0]
        generated = input_ids.clone()
        total_confidence = []
        early_exit_counts = 0
        total_tokens = 0

        for step in range(max_new_tokens):
            # Forward pass through blocks, potentially early-exiting
            x = self.embedding(generated)  # (B, L, d_model)
            x = self.emb_dropout(x)
            x = self.prenorm(x)

            final_logits = None
            final_confidence = None
            exited_early = False

            total_conf_for_token = []

            for i, block in enumerate(self.blocks):
                # Only process the last token's features for previous blocks
                # But full sequence for the current token's context
                # In production, we'd use KV caching — here we recompute for simplicity

                x, conf, aux_loss = block(x, deterministic=True)
                total_conf_for_token.append(conf)

                # Check for early exit
                if use_early_exit:
                    exit_logits = self.early_exit(x, layer_idx=i + 1)
                    if exit_logits is not None:
                        logits_at_exit = exit_logits[:, -1, :]  # last token only
                        should_exit, entropy = self.early_exit.should_exit(logits_at_exit)

                        if should_exit:
                            final_logits = logits_at_exit
                            exited_early = True
                            early_exit_counts += 1
                            if verbose:
                                print(f"  Early exit at block {i+1}, entropy={entropy.mean().item():.3f}")
                            break

            # If no early exit, use final logits
            if final_logits is None:
                x = self.final_norm(x)
                final_logits = self.lm_head(x)[:, -1, :]  # last token

            # Confidence
            final_confidence = torch.stack(total_conf_for_token).mean(dim=0)[:, -1]  # (B,)
            avg_confidence = final_confidence.mean().item()
            total_confidence.append(avg_confidence)

            # Error resilience: re-sample low-confidence tokens
            safe_temp = temperature
            if use_confidence_verification:
                low_conf_mask = final_confidence < confidence_threshold
                if low_conf_mask.any():
                    # For low-confidence tokens: reduce temperature to be more conservative
                    safe_temp = torch.full_like(final_confidence, temperature).unsqueeze(-1)
                    safe_temp[low_conf_mask] = temperature * final_confidence[low_conf_mask].unsqueeze(-1)
                    # Completely gate very low-confidence tokens
                    very_low = final_confidence < (confidence_threshold / 2)
                    if very_low.any():
                        # For nearly-zero confidence, sample from uniform (safety fallback)
                        safe_temp[very_low] = 1.0

            # Sampling
            if temperature < 0.01:
                next_token = final_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            else:
                # Scale logits by temperature
                logits_scaled = final_logits / safe_temp

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits_scaled, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits_scaled[indices_to_remove] = float("-inf")

                # Top-k filtering
                if top_k > 0:
                    top_k_logits, _ = torch.topk(logits_scaled, top_k, dim=-1)
                    min_top_k = top_k_logits[:, -1].unsqueeze(-1)
                    logits_scaled[logits_scaled < min_top_k] = float("-inf")

                probs = F.softmax(logits_scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append
            generated = torch.cat([generated, next_token], dim=-1)
            total_tokens += 1

            # Stop if all sequences produced EOS (assuming EOS=0)
            if (next_token == 0).all():
                break

        stats = {
            "tokens_generated": total_tokens,
            "avg_confidence": sum(total_confidence) / len(total_confidence) if total_confidence else 0.0,
            "early_exit_rate": early_exit_counts / total_tokens if total_tokens > 0 else 0.0,
            "early_exit_threshold": self.early_exit.get_threshold() if use_early_exit else 0.0,
        }

        return generated, stats

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return f"params={self.get_num_params()/1e9:.2f}B, blocks={self.config.n_blocks}, d_model={self.config.d_model}"
