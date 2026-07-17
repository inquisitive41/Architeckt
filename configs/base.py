"""
Config factory: Base ArchitecktConfig (re-export from models.config).

The canonical config class lives in models/config.py. This module
provides convenience constructors with pre-filled defaults for
standard model variants.
"""

from models.config import ArchitecktConfig


def ArchitecktBaseConfig(**overrides) -> ArchitecktConfig:
    """Default ~1.5B parameter config for research exploration.

    Override any field with keyword arguments:
        cfg = ArchitecktBaseConfig(d_model=4096, n_blocks=32)
    """
    defaults = dict(
        d_model=2048,
        n_blocks=24,
        vocab_size=65536,
        max_seq_len=8192,
        n_scales=3,
        d_head=64,
        max_heads=32,
        kernel_fn="elu",
        scale_windows=(128, 512, 8192),
        scale_alphas=(1.0, 1.0, 1.0),
        head_gate_temperature=1.0,
        head_gate_threshold_percentile=0.3,
        n_experts=8,
        n_active_experts=2,
        d_ff=8192,
        router_temperature=1.0,
        load_balance_coef=0.01,
        router_z_loss_coef=0.001,
        n_token_types=4,
        catr_temperature=1.0,
        norm_eps=1e-6,
        norm_adaptive_dim=32,
        swiglu_t_threshold_init=0.0,
        swiglu_t_beta_init=5.0,
        rope_theta=10000.0,
        rope_partial_factor=1.0,
        confidence_threshold=0.5,
        tlcg_hidden_dim=256,
        exit_layers=(8, 16, 24),
        entropy_threshold_base=1.0,
        entropy_ema_alpha=0.9,
        entropy_margin=0.5,
        dropout=0.0,
        emb_dropout=0.0,
        weight_decay=0.1,
        init_std=0.02,
        init_scale_depth=True,
    )
    defaults.update(overrides)
    return ArchitecktConfig(**defaults)
