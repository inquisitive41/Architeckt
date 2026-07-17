"""
Residual connection variants for Architeckt.

Provides:
- DeepNetResidual: scaling by α = (2N)^(-0.5) for training stability
- PreNormResidual: pre-normalization + residual + optional stochastic depth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DeepNetResidual(nn.Module):
    """DeepNet-style residual with learnable scaling.

    y = x + α · f(x)

    where α = β · (2N)^(-0.5), β is learnable (initialized to 1.0).

    This formulation stabilizes training in very deep (>100 layers)
    networks by preventing residual variance explosion. For 24 blocks,
    the default scaling is 1/√48 ≈ 0.144.

    Args:
        n_blocks: total number of blocks in the network
        learn_scale: if True, make the residual scale learnable
        init_scale: override initial scaling (default: (2*n_blocks)**-0.5)
    """

    def __init__(
        self,
        n_blocks: int,
        learn_scale: bool = False,
        init_scale: Optional[float] = None,
    ):
        super().__init__()
        self.n_blocks = n_blocks

        base_scale = init_scale if init_scale is not None else (2.0 * n_blocks) ** -0.5
        self.register_buffer("base_scale", torch.tensor(base_scale))

        if learn_scale:
            self.log_scale = nn.Parameter(torch.zeros(1))
        else:
            self.log_scale = None

    @property
    def scale(self) -> torch.Tensor:
        if self.log_scale is not None:
            return torch.exp(self.log_scale) * self.base_scale
        return self.base_scale

    def forward(
        self,
        x: torch.Tensor,
        f: torch.Tensor,
        dropout: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: residual input (batch, ..., dim)
            f: transform output, same shape as x
            dropout: optional dropout module to apply to f

        Returns:
            x + scale * dropout(f)
        """
        if dropout is not None:
            f = dropout(f)
        return x + self.scale * f

    def extra_repr(self) -> str:
        return f"scale={self.scale.item():.4f}, learnable={self.log_scale is not None}"


class PreNormResidual(nn.Module):
    """Pre-normalization residual block.

    Wraps a sublayer: norm(x) → sublayer → residual.

    This is the standard Transformer block pattern used in
    Architeckt with AdaSNorm normalization.

    Args:
        norm: normalization module (e.g., AdaSNorm)
        sublayer: the sublayer to apply (attention or FFN)
        residual_scale: DeepNet scaling factor
        dropout_rate: dropout applied after the sublayer
    """

    def __init__(
        self,
        norm: nn.Module,
        sublayer: nn.Module,
        residual_scale: float = 1.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.norm = norm
        self.sublayer = sublayer
        self.residual_scale = residual_scale
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x + self.residual_scale * self.dropout(self.sublayer(self.norm(x), **kwargs))

    def extra_repr(self) -> str:
        return f"scale={self.residual_scale:.4f}, norm={type(self.norm).__name__}"


def test_deepnet_residual():
    res = DeepNetResidual(n_blocks=24)
    x = torch.randn(4, 16, 512)
    f = torch.randn(4, 16, 512) * 0.1
    y = res(x, f)
    assert y.shape == x.shape
    assert not torch.allclose(y, x)
    print("DeepNetResidual test passed.")


def test_prenorm_residual():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from normalization.adaptive_norm import AdaSNorm
    norm = AdaSNorm(512)
    sublayer = nn.Linear(512, 512)
    block = PreNormResidual(norm, sublayer, residual_scale=0.1)
    x = torch.randn(4, 16, 512)
    y = block(x)
    assert y.shape == x.shape
    print("PreNormResidual test passed.")


if __name__ == "__main__":
    test_deepnet_residual()
    test_prenorm_residual()
