"""
Small Architeckt variant: ~300M parameters.

Suitable for fast iteration, ablation studies, and single-GPU
experiments. 4× smaller than the Medium config with proportionally
reduced compute requirements.

Estimated compute budget:
    ~4 days on 8×A100 or ~5 days on 4×A100 for 300B tokens.
"""

from configs.base import ArchitecktBaseConfig


def ArchitecktSmallConfig(**overrides):
    """~300M parameter small variant.

    Changes from base:
        d_model: 2048 → 1024
        n_blocks: 24 → 16
        n_experts: 8 → 4
        n_active_experts: 2 → 1
        d_ff: 8192 → 4096
        max_heads: 32 → 16
    """
    defaults = dict(
        d_model=1024,
        n_blocks=16,
        n_experts=4,
        n_active_experts=1,
        d_ff=4096,
        max_heads=16,
        d_head=64,
        exit_layers=(6, 11, 16),
    )
    defaults.update(overrides)
    return ArchitecktBaseConfig(**defaults)
