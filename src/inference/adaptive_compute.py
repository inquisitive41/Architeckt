"""
Adaptive compute allocation during inference.

Dynamically adjusts computation per token based on:
- Input complexity (entropy, perplexity)
- Position in sequence (later tokens tend to be harder)
- Model's own confidence

This goes beyond DAEE by also adjusting:
- Number of active attention heads (AHG threshold tuning)
- Number of active experts (SMoE top-k adjustment)
- Whether to skip FFN entirely for very simple tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class AdaptiveComputeController(nn.Module):
    """Dynamically allocates compute resources during inference.

    For each token, decides:
    1. How many attention heads to activate (via AHG threshold)
    2. How many experts to route to (via SMoE top-k)
    3. Whether to skip FFN entirely

    The controller observes the running statistics of previous tokens
    to make predictions about the current token's difficulty.

    Args:
        d_model: feature dimension
        min_active_heads: minimum attention heads (floor)
        max_active_heads: maximum attention heads (ceiling)
        min_active_experts: minimum SMoE experts
        max_active_experts: maximum SMoE experts
        skip_ffn_threshold: confidence above which FFN can be skipped
    """

    def __init__(
        self,
        d_model: int,
        min_active_heads: int = 4,
        max_active_heads: int = 32,
        min_active_experts: int = 1,
        max_active_experts: int = 8,
        skip_ffn_threshold: float = 0.95,
    ):
        super().__init__()
        self.d_model = d_model
        self.min_active_heads = min_active_heads
        self.max_active_heads = max_active_heads
        self.min_active_experts = min_active_experts
        self.max_active_experts = max_active_experts
        self.skip_ffn_threshold = skip_ffn_threshold

        # Small network to predict token difficulty from statistics
        self.difficulty_net = nn.Sequential(
            nn.Linear(4, 32, bias=True),     # 4 input stats
            nn.SiLU(),
            nn.Linear(32, 3, bias=True),     # 3 outputs: heads_frac, experts_frac, skip_prob
        )

        nn.init.normal_(self.difficulty_net[0].weight, std=0.02)
        nn.init.zeros_(self.difficulty_net[0].bias)
        nn.init.normal_(self.difficulty_net[2].weight, std=0.02)
        nn.init.zeros_(self.difficulty_net[2].bias)

    def _compute_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-token statistics as difficulty indicators.

        Returns:
            (batch, seq_len, 4) with: activation_norm, activation_range,
            attention_output_norm, last_layer_entropy_est
        """
        with torch.no_grad():
            norm = x.norm(p=2, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
            range_ = (x.max(dim=-1).values - x.min(dim=-1).values).unsqueeze(-1)
            skew = x.abs().mean(dim=-1, keepdim=True) / (x.std(dim=-1, keepdim=True) + 1e-6)
            peak = x.abs().max(dim=-1).values.unsqueeze(-1)
        return torch.cat([norm, range_, skew, peak], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        current_confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model) — token representations
            current_confidence: (batch, seq_len) — TLCG confidence

        Returns:
            compute_budget: dict with:
                - n_active_heads: (batch, seq_len) — how many heads to use
                - n_active_experts: (batch, seq_len) — how many experts to route
                - skip_ffn: (batch, seq_len) — bool whether to skip FFN
                - compute_ratio: (batch, seq_len) — fraction of full compute
        """
        batch, seq_len, _ = x.shape

        # Compute difficulty indicators
        stats = self._compute_stats(x)  # (B, L, 4)
        predictions = self.difficulty_net(stats)  # (B, L, 3)

        # heads_frac: fraction of max heads to use
        heads_frac = torch.sigmoid(predictions[..., 0])  # (B, L) in [0, 1]
        n_heads = (heads_frac * (self.max_active_heads - self.min_active_heads)
                   + self.min_active_heads).round().long()

        # experts_frac: fraction of max experts to use
        experts_frac = torch.sigmoid(predictions[..., 1])
        n_experts = (experts_frac * (self.max_active_experts - self.min_active_experts)
                     + self.min_active_experts).round().long()

        # skip_ffn_prob: probability of skipping FFN
        skip_logit = predictions[..., 2]
        skip_ffn = torch.sigmoid(skip_logit) > 0.5  # (B, L) bool

        # Override skip_ffn if confidence is known and too low
        if current_confidence is not None:
            must_use_ffn = current_confidence < self.skip_ffn_threshold
            skip_ffn = skip_ffn & ~must_use_ffn

        # Overall compute ratio (0.0 = minimal, 1.0 = full)
        heads_ratio = (n_heads.float() - self.min_active_heads) / (
            self.max_active_heads - self.min_active_heads + 1e-8
        )
        experts_ratio = (n_experts.float() - self.min_active_experts) / (
            self.max_active_experts - self.min_active_experts + 1e-8
        )
        ffn_ratio = (~skip_ffn).float()
        compute_ratio = (heads_ratio + experts_ratio + ffn_ratio) / 3.0

        return {
            "n_active_heads": n_heads,
            "n_active_experts": n_experts,
            "skip_ffn": skip_ffn,
            "compute_ratio": compute_ratio,
        }

    def extra_repr(self) -> str:
        return (
            f"heads=[{self.min_active_heads}, {self.max_active_heads}], "
            f"experts=[{self.min_active_experts}, {self.max_active_experts}]"
        )


def test_adaptive_compute():
    controller = AdaptiveComputeController(d_model=512)
    x = torch.randn(2, 16, 512)
    budget = controller(x)
    assert budget["n_active_heads"].shape == (2, 16)
    assert budget["n_active_experts"].shape == (2, 16)
    assert budget["skip_ffn"].shape == (2, 16)
    assert (budget["compute_ratio"] >= 0).all() and (budget["compute_ratio"] <= 1).all()

    # With high confidence, more tokens should have skip_ffn
    high_conf = torch.full((2, 16), 0.99)
    budget_hc = controller(x, current_confidence=high_conf)
    skip_rate_high = budget_hc["skip_ffn"].float().mean()

    # With low confidence, fewer tokens should skip
    low_conf = torch.full((2, 16), 0.1)
    budget_lc = controller(x, current_confidence=low_conf)
    skip_rate_low = budget_lc["skip_ffn"].float().mean()

    print(f"Skip-FFN rate: high-conf={skip_rate_high:.2f}, low-conf={skip_rate_low:.2f}")
    assert skip_rate_low <= skip_rate_high + 0.1, "Low confidence should reduce FFN skipping"
    print("AdaptiveComputeController test passed.")


if __name__ == "__main__":
    test_adaptive_compute()
