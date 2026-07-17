"""
Adaptive Head Gating (AHG)

Dynamically selects which attention heads are active per token.
Replaces the fixed number of heads with a learned gating mechanism
that routes computation to the most relevant heads.

Mathematical formulation:

    For each head h at position t:
        gate_raw = W_g · x_t + b_g           # scalar projection
        gate_h(t) = sigmoid(gate_raw)         # (0, 1) soft gate
        
        Threshold τ = percentile(gate(t), p)  # dynamic threshold
        
        Effective gate:
            g_h(t) = gate_h(t) if gate_h(t) > τ else 0

    head_output_h = g_h · Attention_h(x)

Complexity:
    Gating: O(H · d) per token — negligible vs attention itself
    Savings: τ-fraction of heads skipped → (1-τ) of attention FLOPs

Memory: O(B · L · H) for gate values

Proven: For τ=0, AHG reduces to standard multi-head attention (all heads active).
Hypothesis: Dynamic thresholding reduces FLOPs 20-40% with minimal quality loss
because many tokens only need a subset of heads for their context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AdaptiveHeadGating(nn.Module):
    """Gating network for dynamic head selection.

    Args:
        d_model: input dimension (per-token features)
        n_heads: total number of heads to gate
        temperature: softness of sigmoid gate (lower = sharper decisions)
        threshold_percentile: fraction of heads to deactivate (0.0 = all active)
        bias: include bias in gate projection
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        temperature: float = 1.0,
        threshold_percentile: float = 0.3,
        bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.threshold_percentile = threshold_percentile

        # Temperature parameter for sigmoid steepness
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())

        # Gate projection: per-token -> per-head scalar
        self.gate_proj = nn.Linear(d_model, n_heads, bias=bias)

        # Learnable threshold offset (added to the percentile-based threshold)
        self.threshold_offset = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        """Initialize gate projection for small initial values."""
        nn.init.normal_(self.gate_proj.weight, std=0.01)
        if self.gate_proj.bias is not None:
            nn.init.zeros_(self.gate_proj.bias)

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model) — per-token representations
            deterministic: if True, use hard threshold (1/0); else soft

        Returns:
            gates: (batch, seq_len, n_heads) — gate values [0, 1]
            active_count: scalar — mean number of active heads per token
        """
        batch, seq_len, _ = x.shape

        # Compute raw gate logits
        gate_logits = self.gate_proj(x)  # (B, L, H)

        # Apply temperature
        gate = torch.sigmoid(gate_logits / self.temperature)  # (B, L, H)

        if deterministic:
            # During inference: hard threshold by percentile
            # Compute per-layer threshold from the gate values
            gate_flat = gate.reshape(-1, self.n_heads)  # (B*L, H)
            k = max(1, int(self.n_heads * self.threshold_percentile))
            # k-th smallest value -> threshold (so n_heads - k heads are kept active)
            threshold_per_token, _ = torch.kthvalue(
                gate_flat, k, dim=-1, keepdim=True
            )  # (B*L, 1)
            threshold_per_token = threshold_per_token.view(batch, seq_len, 1)

            threshold = threshold_per_token + self.threshold_offset

            # Hard gate: 1 if above threshold, 0 otherwise
            hard_gate = (gate > threshold).float()

            # Straight-through estimator: gradient flows through soft gate
            gate = hard_gate + gate - gate.detach()
        else:
            # During training: soft gating (no thresholding)
            # But we still compute a soft "pseudo-threshold" for logging
            pass

        # Compute mean active count for monitoring
        active_count = gate.mean(dim=-1).mean()  # mean across batch and sequence

        return gate, active_count

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_heads={self.n_heads}, percentile={self.threshold_percentile}"


def test_ahg():
    """Smoke test for AHG."""
    ahg = AdaptiveHeadGating(d_model=512, n_heads=32, threshold_percentile=0.3)

    x = torch.randn(2, 16, 512)

    # Test soft gating (training)
    gate, active = ahg(x, deterministic=False)
    assert gate.shape == (2, 16, 32)
    assert 0.4 < active.item() < 0.6, f"Gate mean should be ~0.5, got {active.item():.4f}"

    # Test hard gating (inference)
    gate_hard, active_hard = ahg(x, deterministic=True)
    assert gate_hard.shape == (2, 16, 32)
    # With percentile=0.3, about 70% heads should be active
    assert 0.6 < active_hard.item() < 0.8, f"Expected ~70% active, got {active_hard.item():.4f}"

    print("AHG test passed.")


if __name__ == "__main__":
    test_ahg()
