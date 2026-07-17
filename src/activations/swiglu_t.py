"""
SwiGLU-Thresholded Activation (SwiGLU-T)

Replaces GELU. SwiGLU with a learnable threshold that induces
natural sparsity in the gating signal.

Mathematical formulation:
    gate = SiLU(x @ W_gate)                     # standard SwiGLU gate
    up   = x @ W_up                             # value projection
    gated = gate * up                           # element-wise gating
    
    # Thresholding: suppress small magnitudes
    mask = sigmoid(beta * (|up| - threshold))
    output = (gated * mask) @ W_down

Where:
    - threshold: learnable per-channel parameter
    - beta: temperature controlling softness of threshold
    - SiLU(x) = x * sigmoid(x)

Complexity: O(d * d_ff) — identical to SwiGLU (threshold is O(d_ff) element-wise)
Memory: O(B * L * d_ff) for intermediate activations

Expected sparsity: 10-30% of activations become exactly zero or negligible
at convergence, reducing effective FLOPs in the down-projection.
This is a HYPOTHESIS requiring empirical validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU_T(nn.Module):
    """SwiGLU with learnable threshold for activation sparsity.

    Args:
        d_model: input/output dimension
        d_ff: intermediate (hidden) dimension
        threshold_init: initial value for learnable threshold
        beta_init: initial temperature for threshold sharpness
        bias: include bias in linear projections
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        threshold_init: float = 0.0,
        beta_init: float = 5.0,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.W_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.W_up = nn.Linear(d_model, d_ff, bias=bias)
        self.W_down = nn.Linear(d_ff, d_model, bias=bias)

        # Per-channel learnable threshold
        self.threshold = nn.Parameter(torch.full((d_ff,), threshold_init))
        # Temperature — larger = sharper threshold
        self.log_beta = nn.Parameter(torch.full((1,), torch.tensor(beta_init).log()))

    @property
    def beta(self) -> torch.Tensor:
        """Ensure beta stays positive via exponential parameterization."""
        return torch.exp(self.log_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        gate = F.silu(self.W_gate(x))  # (B, L, d_ff)
        up = self.W_up(x)  # (B, L, d_ff)

        gated = gate * up  # element-wise gating

        # Threshold mask: sigmoid(beta * (|up| - threshold))
        mask = torch.sigmoid(self.beta * (up.abs() - self.threshold))

        # Apply mask and project
        output = self.W_down(gated * mask)  # (B, L, d_model)

        return output

    def sparsity(self, x: torch.Tensor, hard: bool = False) -> float:
        """Estimate sparsity of the gated-up projection.

        Args:
            x: input tensor
            hard: if True, use hard threshold; if False, count values where mask < 0.01

        Returns:
            fraction of near-zero activations
        """
        with torch.no_grad():
            up = self.W_up(x)
            if hard:
                mask = (up.abs() >= self.threshold).float()
            else:
                mask = torch.sigmoid(self.beta * (up.abs() - self.threshold))
            sparsity = (mask < 0.01).float().mean().item()
        return sparsity

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}, threshold_range=[{self.threshold.min().item():.3f}, {self.threshold.max().item():.3f}]"


def test_swiglu_t():
    """Quick smoke test."""
    layer = SwiGLU_T(512, 2048)
    x = torch.randn(2, 16, 512)
    y = layer(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype

    # Test with zero threshold (should match SwiGLU approximately)
    layer.threshold.data.fill_(-10.0)
    y_zero = layer(x)

    # Test sparsity measurement
    sp = layer.sparsity(x)
    assert 0.0 <= sp <= 1.0, f"Sparsity out of range: {sp}"
    print(f"SwiGLU-T test passed. Sparsity (threshold=0): {sp:.4f}")


if __name__ == "__main__":
    test_swiglu_t()
