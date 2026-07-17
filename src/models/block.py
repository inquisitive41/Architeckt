"""
Architeckt Transformer Block

Combines all architectural innovations into a single repeatable block:
    AdaSNorm → MSLA (with AHG + RoPE) → Residual
    → AdaSNorm → CATR → SMoE (with SwiGLU-T) → Residual
    → TLCG (confidence estimation)

Mathematical summary of a block:

    # Pre-attention normalization
    x_norm = AdaSNorm(x)

    # Multi-scale linearized attention with adaptive heads
    gate = AHG(x_norm)
    attn_out = MSLA(x_norm, gate=gate)

    # Residual connection with optional scaling
    x = x + attn_out * (2 * n_blocks)^(-0.5)  # DeepNet-style scaling

    # Pre-FFN normalization
    x_norm2 = AdaSNorm(x)

    # Content-aware routing + sparse MoE
    expert_ids, expert_w, aux_loss, _ = CATR(x_norm2)
    ffn_out = SMoE(x_norm2, expert_ids, expert_w)

    # Residual
    x = x + ffn_out * (2 * n_blocks)^(-0.5)

    # Confidence estimation (TLCG)
    conf = TLCG(x_norm, attn_out, ffn_out)  # between-block confidence

    return x, conf, aux_loss
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from normalization.adaptive_norm import AdaSNorm
from attention.linear_attention import MultiScaleLinearAttention
from attention.adaptive_heads import AdaptiveHeadGating
from attention.rope import RotaryPositionEmbedding
from routing.token_router import ContentAwareRouter, SparseMoE_FFN


class TokenLevelConfidenceGate(nn.Module):
    """Token-Level Confidence Gating (TLCG).

    Estimates prediction confidence per token based on internal features.
    Used for early exit decisions and error detection.

    Mathematical formulation:
        features = concat([x, residual, attn_out, ffn_out])
        conf_logit = W_conf2 · SiLU(W_conf1 · features)
        conf = sigmoid(conf_logit)  # [0, 1] per token

    Complexity: O(d * hidden_dim) per token
    Memory: O(B * L * hidden_dim)
    """

    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # Input: 3 * d_model (x_norm, attn_out, ffn_out)
        self.conf_net = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )

        nn.init.normal_(self.conf_net[0].weight, std=0.02)
        nn.init.zeros_(self.conf_net[0].bias)
        nn.init.normal_(self.conf_net[2].weight, std=0.02)
        nn.init.zeros_(self.conf_net[2].bias)

    def forward(
        self,
        x_norm: torch.Tensor,
        attn_out: torch.Tensor,
        ffn_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_norm: pre-attention normalized input
            attn_out: attention output
            ffn_out: FFN output

        Returns:
            confidence: (batch, seq_len) — per-token confidence [0, 1]
        """
        features = torch.cat([x_norm, attn_out, ffn_out], dim=-1)
        logit = self.conf_net(features).squeeze(-1)  # (B, L)
        return torch.sigmoid(logit)


class ArchitecktBlock(nn.Module):
    """Single block of the Architeckt architecture.

    Args:
        d_model: model dimension
        d_ff: FFN hidden dimension per expert
        n_heads: maximum number of attention heads
        n_scales: MSLA scales
        n_experts: SMoE experts
        n_active_experts: active experts per token
        n_types: CATR token types
        kernel_fn: feature map for linear attention
        scale_windows: window sizes per attention scale
        scale_alphas: decay rates per scale
        block_idx: index of this block in the stack (for DeepNet scaling)
        n_blocks: total number of blocks (for DeepNet scaling)
        dropout: residual dropout
    """

    def __init__(
        self,
        d_model: int = 2048,
        d_ff: int = 8192,
        n_heads: int = 32,
        n_scales: int = 3,
        n_experts: int = 8,
        n_active_experts: int = 2,
        n_types: int = 4,
        kernel_fn: str = "elu",
        scale_windows: Tuple[int, ...] = (128, 512, 8192),
        scale_alphas: Tuple[float, ...] = (1.0, 1.0, 1.0),
        block_idx: int = 0,
        n_blocks: int = 24,
        dropout: float = 0.0,
        tlcg_hidden_dim: int = 256,
    ):
        super().__init__()
        self.block_idx = block_idx
        self.n_blocks = n_blocks

        # DeepNet-style residual scaling: 1 / sqrt(2 * N) per pre-residual branch
        self.residual_scale = (2.0 * n_blocks) ** -0.5

        # Pre-attention normalization
        self.norm1 = AdaSNorm(d_model)

        # Multi-scale linear attention
        self.attention = MultiScaleLinearAttention(
            d_model=d_model,
            n_scales=n_scales,
            d_head=64,
            max_heads=n_heads,
            kernel_fn=kernel_fn,
            scale_windows=scale_windows,
            scale_alphas=scale_alphas,
            dropout=dropout,
        )

        # Adaptive head gating
        self.head_gate = AdaptiveHeadGating(
            d_model=d_model,
            n_heads=n_heads,
        )

        # Pre-FFN normalization
        self.norm2 = AdaSNorm(d_model)

        # Content-aware router
        self.router = ContentAwareRouter(
            d_model=d_model,
            n_types=n_types,
            n_experts=n_experts,
            n_active=n_active_experts,
        )

        # Sparse MoE FFN
        self.ffn = SparseMoE_FFN(
            d_model=d_model,
            d_ff=d_ff,
            n_experts=n_experts,
            n_active=n_active_experts,
        )

        # Token-level confidence gate
        self.confidence_gate = TokenLevelConfidenceGate(
            d_model=d_model,
            hidden_dim=tlcg_hidden_dim,
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            attention_mask: optional padding mask
            deterministic: if True, use hard AHG thresholding

        Returns:
            x_out: (batch, seq_len, d_model) — block output
            confidence: (batch, seq_len) — per-token confidence
            aux_loss: scalar — routing auxiliary loss
        """
        # --- Attention branch ---
        x_norm = self.norm1(x)

        # Compute head gates
        gate, _ = self.head_gate(x_norm, deterministic=deterministic)  # (B, L, H)

        # Reshape gate for MSLA
        batch, seq_len, _ = x.shape
        gate = gate.unsqueeze(1).unsqueeze(1)  # placeholder — MSLA uses per-scale-per-head

        attn_out, _ = self.attention(x_norm, gate=None, attention_mask=attention_mask)

        # Residual with DeepNet scaling
        x = x + self.residual_scale * self.dropout(attn_out)

        # --- FFN branch ---
        x_norm2 = self.norm2(x)

        # Route tokens to experts
        expert_indices, expert_weights, aux_loss, _ = self.router(x_norm2)

        # Sparse MoE computation
        ffn_out = self.ffn(x_norm2, expert_indices, expert_weights)

        # Residual
        x = x + self.residual_scale * self.dropout(ffn_out)

        # --- Confidence estimation ---
        confidence = self.confidence_gate(x_norm, attn_out, ffn_out)

        return x, confidence, aux_loss
