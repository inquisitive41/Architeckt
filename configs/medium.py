"""
Medium Architeckt variant: ~1.5B parameters — primary research target.

This is the canonical Architeckt config optimized for the 7-day
× 8×A100 compute budget. All theoretical analyses and benchmarks
are performed on this variant.

Target metrics (hypotheses requiring experimental validation):
    C4 PPL: ≤ 8
    MMLU (5-shot): ≥ 60%
    HumanEval pass@1: ≥ 50%
    Inference latency: ≤ 50 ms/token on 24GB GPU
"""

from configs.base import ArchitecktBaseConfig


def ArchitecktMediumConfig(**overrides):
    """~1.5B parameter medium variant — primary research target.

    Uses default base config without modifications. All values
    are the ArchitecktConfig defaults.
    """
    return ArchitecktBaseConfig(**overrides)
