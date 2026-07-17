"""
Rotary Position Embedding (RoPE)

Implements built-in position encoding without separate absolute
positional embeddings. RoPE modifies the query-key dot product
by rotating each dimension pair by an angle proportional to position.

Mathematical formulation:
    For positions m, n and feature index i:
    theta_i = base^(-2i/d_head)
    q_rot_i(m) = q_i * cos(m*theta_i) + rotate_pair(q_i) * sin(m*theta_i)

Where rotate_pair swaps and negates adjacent dimensions.

The dot product becomes:
    q_rot(m)^T * k_rot(n) = q^T * R_{n-m} * k

i.e. depends only on relative position (n-m), not absolute positions.

Complexity: O(L * d_head) per layer — linear, with no learnable parameters
Memory: O(L * d_head) for pre-computed cos/sin tables

Proven: RoPE provides relative position encoding without additional embeddings.
Experiments show it matches or exceeds learned absolute embeddings while
enabling length extrapolation.
"""

import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    """Rotary position embedding for attention QK computation.

    Applies to a fraction of the head dimension (typically 1/4 to 1/2),
    leaving the rest as content-only features.

    Args:
        d_head: head dimension
        max_seq_len: pre-computed maximum sequence length
        theta: base for geometric progression of frequencies
        partial_factor: fraction of d_head to apply RoPE to (0.0 to 1.0)
    """

    def __init__(
        self,
        d_head: int = 64,
        max_seq_len: int = 8192,
        theta: float = 10000.0,
        partial_factor: float = 1.0,
    ):
        super().__init__()
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.partial_factor = partial_factor

        self.rope_dim = int(d_head * partial_factor)
        assert self.rope_dim % 2 == 0, f"RoPE dim must be even, got {self.rope_dim}"

        # Pre-compute frequency bands
        freqs = 1.0 / (
            theta ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32) / self.rope_dim)
        )
        self.register_buffer("freqs", freqs, persistent=False)

        # Pre-compute cos/sin for all positions up to max_seq_len
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, freqs)  # (max_seq_len, rope_dim // 2)

        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Swap pairs: [x1, x2, x3, x4, ...] -> [-x2, x1, -x4, x3, ...]"""
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to queries and keys.

        Args:
            q: (batch, n_heads, seq_len, d_head)
            k: (batch, n_heads, seq_len, d_head)
            seq_len: actual sequence length (may be < max_seq_len)
            offset: position offset (for KV cache during inference)

        Returns:
            q_rotated: (batch, n_heads, seq_len, d_head)
            k_rotated: (batch, n_heads, seq_len, d_head)
        """
        # Get cached cos/sin for this sequence
        cos = self.cos_cached[offset : offset + seq_len]  # (seq_len, rope_dim//2)
        sin = self.sin_cached[offset : offset + seq_len]

        # Repeat each value for pair: (L, d/2) -> (L, d)
        cos = cos.repeat_interleave(2, dim=-1)  # (seq_len, rope_dim)
        sin = sin.repeat_interleave(2, dim=-1)

        # Reshape for broadcasting: (1, 1, L, rope_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply RoPE to the first rope_dim dimensions of q, k
        q_rope = q[..., :self.rope_dim]
        q_content = q[..., self.rope_dim:]

        k_rope = k[..., :self.rope_dim]
        k_content = k[..., self.rope_dim:]

        # Rotate: x_rot = x * cos + rotate_half(x) * sin
        q_rope_rotated = q_rope * cos + self._rotate_half(q_rope) * sin
        k_rope_rotated = k_rope * cos + self._rotate_half(k_rope) * sin

        # Concatenate with content-only dimensions
        if self.rope_dim < self.d_head:
            q_rotated = torch.cat([q_rope_rotated, q_content], dim=-1)
            k_rotated = torch.cat([k_rope_rotated, k_content], dim=-1)
        else:
            q_rotated = q_rope_rotated
            k_rotated = k_rope_rotated

        return q_rotated, k_rotated

    def extra_repr(self) -> str:
        return f"d_head={self.d_head}, rope_dim={self.rope_dim}, max_seq_len={self.max_seq_len}, theta={self.theta}"


def test_rope():
    """Quick smoke test for RoPE."""
    rope = RotaryPositionEmbedding(d_head=64, max_seq_len=128, partial_factor=0.5)
    q = torch.randn(2, 4, 32, 64)
    k = torch.randn(2, 4, 32, 64)
    qr, kr = rope(q, k, seq_len=32, offset=0)
    assert qr.shape == q.shape
    assert kr.shape == k.shape
    # Check that content part is unchanged
    assert torch.allclose(qr[..., rope.rope_dim:], q[..., rope.rope_dim:])
    # Check that RoPE part was modified
    assert not torch.allclose(qr[..., :rope.rope_dim], q[..., :rope.rope_dim])
    print("RoPE test passed.")


if __name__ == "__main__":
    test_rope()
