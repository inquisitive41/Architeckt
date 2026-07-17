"""
Custom linear layers for Architeckt.

Provides:
- ScaledLinear: linear with scaled initialization for depth stability
- GatedLinear: linear with a learned output gate for dynamic scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ScaledLinear(nn.Module):
    """Linear layer with depth-aware initialization.

    Multiplies output by 1/√(2N) for DeepNet-style scaling.
    Used as a convenience wrapper when the residual scaling
    isn't already applied at the block level.

    Args:
        in_features: input dimension
        out_features: output dimension
        depth_scale: scaling factor (e.g., (2*N)**-0.5)
        bias: include bias term
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        depth_scale: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.depth_scale = depth_scale

        nn.init.normal_(self.linear.weight, std=0.02 * depth_scale)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.depth_scale

    def extra_repr(self) -> str:
        return f"in={self.linear.in_features}, out={self.linear.out_features}, scale={self.depth_scale:.4f}"


class GatedLinear(nn.Module):
    """Linear projection with a learned scalar output gate.

    Can learn to suppress outputs for certain input patterns,
    providing an additional degree of freedom for the model
    to control information flow.

    y = sigmoid(W_gate · x + b_gate) · (W_proj · x + b_proj)

    Args:
        in_features: input dimension
        out_features: output dimension
        bias: include bias in both gate and projection
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=bias)
        self.gate_proj = nn.Linear(in_features, out_features, bias=bias)

        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.normal_(self.gate_proj.weight, std=0.01)
        if bias:
            nn.init.zeros_(self.proj.bias)
            nn.init.zeros_(self.gate_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(x))
        return gate * self.proj(x)

    def extra_repr(self) -> str:
        return f"in={self.proj.in_features}, out={self.proj.out_features}"


def test_scaled_linear():
    sl = ScaledLinear(512, 256, depth_scale=0.1)
    x = torch.randn(4, 8, 512)
    y = sl(x)
    assert y.shape == (4, 8, 256)
    print("ScaledLinear test passed.")


def test_gated_linear():
    gl = GatedLinear(512, 256)
    x = torch.randn(4, 8, 512)
    y = gl(x)
    assert y.shape == (4, 8, 256)
    print("GatedLinear test passed.")


if __name__ == "__main__":
    test_scaled_linear()
    test_gated_linear()
