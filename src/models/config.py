"""
Architeckt Model Configuration
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ArchitecktConfig:
    """Configuration for the Architeckt language model.

    Hardware targets: 7 days × 8×A100 80GB training, 24GB VRAM inference.
    Default config targets ~1.5B parameters.
    """

    # Model dimensions
    d_model: int = 2048
    n_blocks: int = 24
    vocab_size: int = 65536
    max_seq_len: int = 8192

    # Multi-Scale Linearized Attention (MSLA)
    n_scales: int = 3
    d_head: int = 64
    max_heads: int = 32  # maximum heads; actual is adaptive
    kernel_fn: str = "elu"  # "elu", "relu", "softplus"
    scale_windows: tuple = (128, 512, 8192)  # window sizes per scale
    scale_alphas: tuple = (1.0, 1.0, 1.0)  # decay rates per scale

    # Adaptive Head Gating (AHG)
    head_gate_temperature: float = 1.0
    head_gate_threshold_percentile: float = 0.3  # top 30% heads active by default

    # Sparse Mixture of Experts (SMoE)
    n_experts: int = 8
    n_active_experts: int = 2
    d_ff: int = 8192  # per-expert FFN dimension
    router_temperature: float = 1.0
    load_balance_coef: float = 0.01
    router_z_loss_coef: float = 0.001

    # Content-Aware Token Routing (CATR)
    n_token_types: int = 4  # semantic categories for routing
    catr_temperature: float = 1.0

    # Adaptive Scale Normalization (AdaSNorm)
    norm_eps: float = 1e-6
    norm_adaptive_dim: int = 32  # hidden dim for dynamic modulation

    # SwiGLU-Thresholded activation
    swiglu_t_threshold_init: float = 0.0
    swiglu_t_beta_init: float = 5.0

    # Rotary Position Embedding
    rope_theta: float = 10000.0
    rope_partial_factor: float = 1.0  # fraction of head dim to apply RoPE

    # Token-Level Confidence Gating (TLCG)
    confidence_threshold: float = 0.5
    tlcg_hidden_dim: int = 256

    # Depth-Aware Early Exit (DAEE)
    exit_layers: tuple = (8, 16, 24)
    entropy_threshold_base: float = 1.0
    entropy_ema_alpha: float = 0.9
    entropy_margin: float = 0.5

    # Regularization
    dropout: float = 0.0
    emb_dropout: float = 0.0
    weight_decay: float = 0.1

    # Initialization
    init_std: float = 0.02
    init_scale_depth: bool = True  # scale init by 1/sqrt(2*n_blocks)

    # Training (derived from compute budget)
    @property
    def total_params(self) -> int:
        """Estimate total parameters."""
        # Rough estimate: vocab * d + n_blocks * (4*d^2 for attention + E*d*d_ff for FFN + overhead)
        vocab_params = self.vocab_size * self.d_model
        attn_params = self.n_blocks * 4 * self.d_model * self.d_model
        ffn_params = self.n_blocks * self.n_experts * 3 * self.d_model * self.d_ff
        norm_params = self.n_blocks * 4 * self.d_model  # norms and gates
        router_params = self.n_blocks * (self.n_experts + self.n_token_types) * self.d_model
        exit_params = len(self.exit_layers) * self.vocab_size * self.d_model
        return vocab_params + attn_params + ffn_params + norm_params + router_params + exit_params


@dataclass
class ArchitecktSmallConfig(ArchitecktConfig):
    """Small variant: ~300M parameters for fast iteration."""
    d_model: int = 1024
    n_blocks: int = 16
    n_experts: int = 4
    n_active_experts: int = 1
    d_ff: int = 4096
    max_heads: int = 16
    d_head: int = 64


@dataclass
class ArchitecktMediumConfig(ArchitecktConfig):
    """Medium variant: ~1.5B parameters — primary research target."""
    pass  # все значения как в базовом конфиге


@dataclass
class ArchitecktLargeConfig(ArchitecktConfig):
    """Large variant: ~4B parameters for scaling studies (requires more compute)."""
    d_model: int = 3072
    n_blocks: int = 32
    n_experts: int = 12
    n_active_experts: int = 3
    d_ff: int = 12288
    max_heads: int = 40
