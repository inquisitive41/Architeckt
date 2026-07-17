"""
Individual expert FFN implementations for SMoE.

Each expert is a standalone feed-forward network with SwiGLU-T
activation. Experts can optionally be weight-tied (for parameter
efficiency) or serve as independent specialists.

Mathematical formulation (per expert):

    gate = SiLU(x · W_gate)
    up   = x · W_up
    gated = gate * up
    mask = σ(β · (|up| - τ))                    # SwiGLU-T threshold
    output = (gated * mask) · W_down

Complexity: O(n_tokens · d_model · d_ff) per expert
Memory: O(E · d_model · d_ff) for expert weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ExpertFFN(nn.Module):
    """A single expert FFN with SwiGLU-T activation.

    This is the building block for SparseMoE. Each expert learns
    a specialized transformation of the input tokens it receives.

    Args:
        d_model: token dimension (input and output)
        d_ff: intermediate hidden dimension
        threshold_init: initial threshold for activation sparsity
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

        self.W_gate = nn.Parameter(torch.empty(d_model, d_ff))
        self.W_up = nn.Parameter(torch.empty(d_model, d_ff))
        self.W_down = nn.Parameter(torch.empty(d_ff, d_model))

        self.threshold = nn.Parameter(torch.full((d_ff,), threshold_init))
        self.log_beta = nn.Parameter(torch.full((1,), torch.tensor(beta_init).log()))

        self._init_weights()

    def _init_weights(self):
        std = 0.02 / (2 * self.d_ff) ** 0.5
        for param in [self.W_gate, self.W_up, self.W_down]:
            nn.init.normal_(param, std=std)

    @property
    def beta(self) -> torch.Tensor:
        return torch.exp(self.log_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (n_tokens, d_model) — tokens routed to this expert

        Returns:
            output: (n_tokens, d_model)
        """
        gate = F.silu(torch.mm(x, self.W_gate))  # (n, d_ff)
        up = torch.mm(x, self.W_up)               # (n, d_ff)

        gated = gate * up

        # SwiGLU-T threshold mask
        mask = torch.sigmoid(self.beta * (up.abs() - self.threshold))

        return torch.mm(gated * mask, self.W_down)


class SharedExpertFFN(ExpertFFN):
    """A shared expert used by all tokens, plus per-domain specialists.

    Useful for Mixture of Experts with a common expert that captures
    shared knowledge, reducing total parameters compared to
    fully-independent experts.

    In Architeckt, this is an optional variant of the SMoE block.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_specialists: int = 4,
        **kwargs,
    ):
        super().__init__(d_model, d_ff, **kwargs)
        self.n_specialists = n_specialists

        # Per-specialist supplementary weights (shared expert + specialist)
        self.specialist_W_gate = nn.Parameter(torch.empty(n_specialists, d_model, d_ff))
        self.specialist_W_up = nn.Parameter(torch.empty(n_specialists, d_model, d_ff))
        self.specialist_W_down = nn.Parameter(torch.empty(n_specialists, d_ff, d_model))

        std = 0.02 / (2 * self.d_ff) ** 0.5
        for param in [self.specialist_W_gate, self.specialist_W_up, self.specialist_W_down]:
            nn.init.normal_(param, std=std)

    def forward_specialist(
        self, x: torch.Tensor, specialist_idx: int
    ) -> torch.Tensor:
        """Forward through shared expert + one specialist.

        Args:
            x: (n_tokens, d_model)
            specialist_idx: which specialist to apply

        Returns:
            output: (n_tokens, d_model)
        """
        gate = F.silu(torch.mm(x, self.W_gate + self.specialist_W_gate[specialist_idx]))
        up = torch.mm(x, self.W_up + self.specialist_W_up[specialist_idx])

        gated = gate * up
        mask = torch.sigmoid(self.beta * (up.abs() - self.threshold))

        return torch.mm(gated * mask, self.W_down + self.specialist_W_down[specialist_idx])


def test_expert_ffn():
    expert = ExpertFFN(d_model=512, d_ff=2048)
    x = torch.randn(32, 512)
    y = expert(x)
    assert y.shape == (32, 512)
    assert y.dtype == x.dtype
    print("ExpertFFN test passed.")


def test_shared_expert():
    shared = SharedExpertFFN(d_model=512, d_ff=2048, n_specialists=4)
    x = torch.randn(16, 512)
    # Test shared expert alone
    y_shared = shared(x)
    assert y_shared.shape == (16, 512)
    # Test with specialist
    y_spec = shared.forward_specialist(x, specialist_idx=2)
    assert y_spec.shape == (16, 512)
    print("SharedExpertFFN test passed.")


if __name__ == "__main__":
    test_expert_ffn()
    test_shared_expert()
