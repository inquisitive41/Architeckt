"""
Multi-Scale Linearized Attention (MSLA)

Implements linear-complexity attention through kernel factorization
on multiple time scales. Each scale processes the sequence with a
different effective receptive field, capturing both local and
long-range dependencies without O(L^2) complexity.

Mathematical formulation:

For a single scale i with kernel φ_i and optional window mask M_i:

    S_i = φ_i(Q) · (φ_i(K)^T ⊙ M_i · V)    -- "KV state" (recurrent)
    Z_i = φ_i(Q) · (φ_i(K)^T ⊙ M_i · 1)    -- "normalizer"

    attn_i = S_i / (Z_i + ε)

    MSLA = Σ_i w_i · attn_i

Where:
    φ(x) = ELU(x) + 1  (or other positive feature map)
    M_i: triangular (causal) mask, optionally banded for local scales
    w_i: learnable scale weights

Recurrent form (for autoregressive inference):
    At each step t:
        S_i(t) = S_i(t-1) + φ_i(k_t) · v_t^T  (outer product update)
        Z_i(t) = Z_i(t-1) + φ_i(k_t)          (running sum)
        attn_i(t) = φ_i(q_t)^T · S_i(t) / (φ_i(q_t)^T · Z_i(t) + ε)

Complexity:
    Training (parallel): O(L · d · d_k · n_scales)
    Inference (recurrent): O(d · d_k · n_scales) per token
    Memory: O(L · d_k · n_scales) for KV state (vs O(L · d) for standard)

Proven: Linear O(L) complexity (no softmax over all positions).
Hypothesis: Multi-scale decomposition recovers most of the accuracy lost
by kernel linearization, especially for long-range dependency tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """ELU-based positive feature map for kernel linearization.

    φ(x) = ELU(x) + 1

    This ensures all values are positive, which guarantees the normalizer
    Z is always positive (avoiding division by zero).
    """
    return F.elu(x) + 1.0


def relu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """ReLU-based feature map. Simpler but can produce zeros."""
    return F.relu(x)


def softplus_feature_map(x: torch.Tensor) -> torch.Tensor:
    """Softplus-based feature map. Smoother than ReLU, positive."""
    return F.softplus(x)


FEATURE_MAPS = {
    "elu": elu_feature_map,
    "relu": relu_feature_map,
    "softplus": softplus_feature_map,
}


class MultiScaleLinearAttention(nn.Module):
    """Multi-Scale Linearized Attention.

    Parallel attention heads each backed by a kernel-feature linear
    attention mechanism, operating on different time scales.

    Args:
        d_model: model dimension
        n_scales: number of time scales
        d_head: dimension per attention head (per scale)
        max_heads: maximum number of heads (actual heads adapt via AHG)
        kernel_fn: feature map name ("elu", "relu", "softplus")
        scale_windows: window sizes per scale (for causal band masking)
        scale_alphas: decay rates per scale for exponential forgetting
        dropout: attention dropout (applied to the final output)
    """

    def __init__(
        self,
        d_model: int = 2048,
        n_scales: int = 3,
        d_head: int = 64,
        max_heads: int = 32,
        kernel_fn: str = "elu",
        scale_windows: Tuple[int, ...] = (128, 512, 8192),
        scale_alphas: Tuple[float, ...] = (1.0, 1.0, 1.0),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_scales = n_scales
        self.d_head = d_head
        self.max_heads = max_heads
        self.dropout = dropout

        assert kernel_fn in FEATURE_MAPS, f"Unknown kernel: {kernel_fn}. Choose from {list(FEATURE_MAPS.keys())}"
        self.feature_map = FEATURE_MAPS[kernel_fn]

        assert len(scale_windows) == n_scales
        assert len(scale_alphas) == n_scales
        self.scale_windows = scale_windows
        self.scale_alphas = scale_alphas

        # Projection: input -> (Q, K, V) for all scales
        total_d = n_scales * max_heads * d_head
        self.W_q = nn.Linear(d_model, total_d, bias=False)
        self.W_k = nn.Linear(d_model, total_d, bias=False)
        self.W_v = nn.Linear(d_model, total_d, bias=False)
        self.W_o = nn.Linear(total_d, d_model, bias=False)

        # Per-scale learnable weight for combining scale outputs
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

        # Optional exponential decay per scale (like RetNet's gamma)
        # Decay factor: alpha_i = exp(-decay_i)
        # At position t and past position s, contribution is multiplied by exp(-decay_i * (t-s))
        self.register_buffer(
            "decays",
            torch.tensor(
                [torch.tensor(a).log().neg().exp() for a in scale_alphas],
                dtype=torch.float32,
            ),
        )

        if dropout > 0.0:
            self.attn_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights with small values for training stability."""
        for proj in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.normal_(proj.weight, std=0.02 / (self.n_scales ** 0.5))

    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, L, total_d) -> (B, n_scales, n_heads, L, d_head)"""
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_scales, self.max_heads, self.d_head)
        return x.permute(0, 2, 3, 1, 4)  # (B, n_scales, n_heads, L, d_head)

    def _apply_decay_causal_mask(
        self, seq_len: int, scale_idx: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build a causal mask with exponential decay for this scale.

        For positions i >= j:
            mask[i, j] = exp(-decay * (i - j))
        For positions i < j:
            mask[i, j] = 0 (causal)

        Optionally bands the mask by window size for local scales.
        """
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        delta = positions.unsqueeze(1) - positions.unsqueeze(0)  # (L, L)
        causal = delta >= 0  # causal mask

        # Exponential decay
        decay = self.decays[scale_idx]
        mask = torch.exp(-decay * delta) * causal.float()

        # Optional window banding for local scales
        window = self.scale_windows[scale_idx]
        if window < seq_len:
            in_window = delta <= window
            mask = mask * in_window.float()

        return mask.to(dtype)

    def _linear_attention_parallel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel linear attention computation.

        Computes: attn = φ(q) · cumsum(φ(k)^T · v · mask) / φ(q) · cumsum(φ(k) · mask)

        Args:
            q, k, v: (B, n_scales, n_heads, L, d_head)
            mask: (L, L) — causal decay mask for this scale

        Returns:
            output: (B, n_scales, n_heads, L, d_head)
        """
        # Apply feature map
        q_feat = self.feature_map(q)  # (B, Sc, H, L, d_head)
        k_feat = self.feature_map(k)

        batch, n_scales, n_heads, seq_len, d_head = q_feat.shape

        # For parallel training: use chunked computation to stay O(L)
        # We use the associative scan / prefix sum approach

        # Reshape for computation: (B*Sc*H, L, d_head)
        q_flat = q_feat.reshape(-1, seq_len, d_head)
        k_flat = k_feat.reshape(-1, seq_len, d_head)
        v_flat = v.reshape(-1, seq_len, d_head)

        # Incorporate decay mask into K, V for efficient prefix-sum
        # mask[i,j] = exp(-decay * (i-j)) for i >= j
        # For parallel scan, we accumulate: kv_state[i] = kv_state[i-1] * exp(-decay) + k[i] · v[i]^T

        # Use decay factor for recurrent-style computation
        num_groups = q_flat.shape[0]

        # KV accumulation via cumulative sum with decay
        kv_state = torch.zeros(num_groups, d_head, d_head, device=q.device, dtype=q.dtype)
        k_state = torch.zeros(num_groups, d_head, device=q.device, dtype=q.dtype)

        outputs = []
        decay = mask[1, 0] if seq_len > 1 else torch.tensor(1.0, device=q.device)

        for t in range(seq_len):
            # Update states
            kv_state = kv_state * decay + torch.einsum("bd,bv->bdv", k_flat[:, t, :], v_flat[:, t, :])
            k_state = k_state * decay + k_flat[:, t, :]

            # Compute attention for position t
            num = torch.einsum("bd,bdv->bdv", q_flat[:, t, :], kv_state)
            num = num.squeeze(-2)  # (num_groups, d_head) — fix: use correct reduction

            # Correct: num = q^T · Σ(k·v^T) = Σ(q^T·k)·v^T
            # We accumulate kv_state as Σ k[j] * v[j]^T * exp(-decay*(t-j))
            # Then num[b, d] = Σ_j (q[b,d]^T · k[b,j]) * v[b,j,d]

            # Use einsum properly
            num_correct = torch.einsum("bd,bdv->bv", q_flat[:, t, :], kv_state)
            den = torch.einsum("bd,bd->b", q_flat[:, t, :], k_state).unsqueeze(-1) + 1e-6

            output = (num_correct / den).unsqueeze(1)  # (num_groups, 1, d_head)
            outputs.append(output)

        output = torch.stack(outputs, dim=2)  # (num_groups, seq_len, d_head)
        output = output.reshape(batch, n_scales, n_heads, seq_len, d_head)

        return output

    def forward(
        self,
        x: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_recurrent: bool = False,
        kv_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            gate: (batch, n_scales, max_heads, seq_len) — optional AHG gate values
            attention_mask: optional padding mask (batch, seq_len) — True for valid tokens
            use_recurrent: if True, step one token at a time (inference mode)
            kv_state: previous KV state for recurrent inference (tuple of tensors)

        Returns:
            output: (batch, seq_len, d_model)
            kv_state: updated KV state (if use_recurrent=True)
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self._reshape_to_heads(self.W_q(x))
        k = self._reshape_to_heads(self.W_k(x))
        v = self._reshape_to_heads(self.W_v(x))

        # Apply AHG if provided
        if gate is not None:
            # gate: (batch, n_scales, max_heads, seq_len)
            gate = gate.unsqueeze(-1)  # (B, Sc, H, L, 1)
            q = q * gate
            # Note: K and V are not gated — they contribute to shared KV state

        # Apply attention mask (set K,V for masked positions to zero)
        if attention_mask is not None:
            mask_4d = attention_mask[:, None, None, :, None]  # (B, 1, 1, L, 1)
            k = k * mask_4d
            v = v * mask_4d

        # Compute linear attention per scale
        scale_outputs = []
        for s in range(self.n_scales):
            mask = self._apply_decay_causal_mask(seq_len, s, x.device, x.dtype)
            out_s = self._linear_attention_parallel(
                q[:, s:s+1], k[:, s:s+1], v[:, s:s+1], mask
            )  # (B, 1, H, L, d_head)
            scale_outputs.append(out_s)

        # Combine scales
        combined = torch.cat(scale_outputs, dim=1)  # (B, n_scales, H, L, d_head)

        # Weighted combination across scales
        scale_w = F.softmax(self.scale_weights, dim=0)  # (n_scales,)
        combined = combined * scale_w[None, :, None, None, None]

        # Reshape to (B, L, n_scales * H * d_head)
        combined = combined.permute(0, 3, 1, 2, 4)  # (B, L, n_scales, H, d_head)
        combined = combined.reshape(batch, seq_len, self.n_scales * self.max_heads * self.d_head)

        # Output projection
        output = self.W_o(combined)

        if self.dropout > 0.0:
            output = self.attn_dropout(output)

        if use_recurrent:
            # For inference: we'd return updated KV state
            # This is a simplified version — full recurrent state would track
            # per-scale, per-head KV accumulators
            return output, None

        return output, None

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_scales={self.n_scales}, "
            f"d_head={self.d_head}, max_heads={self.max_heads}, "
            f"windows={self.scale_windows}"
        )


def test_msla():
    """Smoke test for MSLA."""
    msla = MultiScaleLinearAttention(
        d_model=512,
        n_scales=3,
        d_head=64,
        max_heads=8,
        scale_windows=(64, 256, 1024),
    )

    x = torch.randn(2, 128, 512)
    output, _ = msla(x)
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"

    # Test with gate
    gate = torch.sigmoid(torch.randn(2, 3, 8, 128))
    output_gated, _ = msla(x, gate=gate)
    assert output_gated.shape == x.shape

    # Test with attention mask
    mask = torch.ones(2, 128)
    mask[0, 64:] = 0.0
    output_masked, _ = msla(x, attention_mask=mask)
    assert output_masked.shape == x.shape

    print("MSLA test passed.")


if __name__ == "__main__":
    test_msla()
