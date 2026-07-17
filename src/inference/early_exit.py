"""
Depth-Aware Early Exit (DAEE)

Dynamically adjusts computation depth during inference without
changing model weights. Tokens exit early when prediction confidence
is high.

Mathematical formulation:

    For each exit layer l in exit_layers:
        Compute logits_l = LMHead_l · x_l
        Compute entropy H_l = -Σ p(y) * log(p(y))
        
        If H_l < H_threshold(t):
            Return logits_l (early exit)
    
    H_threshold evolves via EMA:
        H_threshold(t) = α · H_threshold(t-1) + (1-α) · mean(H_over_last_K) + β

Where:
    - α: smoothing factor
    - β: margin to prevent over-eager exit
    - K: recent history window

Complexity: O(vocab_size · d_model) per exit head
Memory: O(n_exits · vocab_size · d_model) for exit head weights
Savings: Up to (n_blocks - exit_layer) / n_blocks of FLOPs for early-exiting tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class EarlyExitHead(nn.Module):
    """LM head for intermediate exit points."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) -> logits: (batch, seq_len, vocab_size)"""
        return self.proj(x)


class DepthAwareEarlyExit(nn.Module):
    """Manages multiple early exit heads and dynamic threshold.

    Args:
        d_model: model dimension
        vocab_size: output vocabulary size
        exit_layers: list of block indices where exits are placed
        entropy_alpha: EMA smoothing factor
        entropy_margin: β — prevents premature exit
        window_size: K — recent tokens for computing running entropy
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        exit_layers: List[int],
        entropy_alpha: float = 0.9,
        entropy_margin: float = 0.5,
        window_size: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.exit_layers = sorted(exit_layers)
        self.entropy_alpha = entropy_alpha
        self.entropy_margin = entropy_margin
        self.window_size = window_size

        # Exit heads — one per exit layer
        self.exit_heads = nn.ModuleDict({
            str(layer): EarlyExitHead(d_model, vocab_size)
            for layer in exit_layers
        })

        # Running statistics for adaptive threshold
        self.register_buffer("ema_entropy", torch.tensor(2.0))  # initialize to moderately high
        self.register_buffer("entropy_buffer", torch.zeros(window_size))
        self.register_buffer("buffer_pos", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _update_threshold(self, entropy: float):
        """Update running entropy EMA and compute adaptive threshold.

        Called at each token step during inference.

        Args:
            entropy: entropy of the predicted distribution at the final layer
        """
        # Update rolling buffer
        pos = self.buffer_pos.item()
        self.entropy_buffer[pos] = entropy
        self.buffer_pos[0] = (pos + 1) % self.window_size

        # Update EMA
        recent_mean = self.entropy_buffer.mean()
        self.ema_entropy = (
            self.entropy_alpha * self.ema_entropy
            + (1 - self.entropy_alpha) * recent_mean
        )

    def get_threshold(self) -> float:
        """Current adaptive entropy threshold.

        Returns:
            threshold: if entropy < threshold, exit early
        """
        return max(0.01, self.ema_entropy.item() - self.entropy_margin)

    def should_exit(self, logits: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        """Check if early exit should happen based on entropy.

        Args:
            logits: (batch, vocab_size) — prediction at current exit

        Returns:
            should_exit: True if entropy below threshold
            entropy: computed entropy value
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch,)

        threshold = self.get_threshold()
        should_exit = (entropy < threshold).all()

        return should_exit, entropy

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        """Get exit logits if this is an exit layer.

        Args:
            hidden_states: (batch, seq_len, d_model)
            layer_idx: current block index

        Returns:
            logits if layer_idx is an exit layer, else None
        """
        layer_key = str(layer_idx)
        if layer_key in self.exit_heads:
            return self.exit_heads[layer_key](hidden_states)
        return None

    def register_entropy(self, entropy: float):
        """Record the final entropy for threshold adaptation."""
        self._update_threshold(entropy)

    def extra_repr(self) -> str:
        return (
            f"exit_layers={self.exit_layers}, "
            f"ema_entropy={self.ema_entropy.item():.3f}, "
            f"threshold={self.get_threshold():.3f}"
        )


def test_daee():
    """Smoke test for DAEE."""
    daee = DepthAwareEarlyExit(
        d_model=512,
        vocab_size=1000,
        exit_layers=[8, 16, 24],
    )

    x = torch.randn(2, 1, 512)

    # Test exit head
    logits = daee(x, layer_idx=8)
    assert logits is not None
    assert logits.shape == (2, 1, 1000)

    # Test non-exit layer
    assert daee(x, layer_idx=5) is None

    # Test threshold adaptation
    daee.register_entropy(1.5)
    daee.register_entropy(1.0)
    daee.register_entropy(0.8)
    print(f"Threshold: {daee.get_threshold():.3f}")

    # Test should_exit
    logits_last = daee(x, layer_idx=24)
    should, ent = daee.should_exit(logits_last.squeeze(1))
    print(f"Should exit: {should}, entropy: {ent.mean().item():.3f}")

    print("DAEE test passed.")


if __name__ == "__main__":
    test_daee()
