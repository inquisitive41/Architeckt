"""
Content-Aware Token Routing (CATR) and Sparse Mixture of Experts (SMoE)

CATR: Classifies each token into a semantic type before SMoE routing,
reducing collision in expert assignment and improving specialization.

Mathematical formulation:

    type_logits = W_type · x               # (B, L, n_types)
    type_probs = softmax(type_logits / temperature)

    For each type t, there is a separate expert router:
        router_logits_t = W_router_t · x   # (B, L, n_experts)
        gate_t = softmax(router_logits_t)

    Effective gate for token (b, l):
        gate_combined = Σ_t type_probs[t] · gate_t

    Select top-k experts:
        indices, weights = TopK(gate_combined, k)

    Output:
        y = Σ_{(i,w) in top-k} w_i · FFN_expert_i(x)

Regularization:
    L_balance = n_experts · Σ_e (f_e · P_e)  — load balancing loss
    where f_e = fraction of tokens routed to expert e
          P_e = mean router probability for expert e

    L_z = mean(log(Σ_e exp(router_logits_e))^2)  — router z-loss for stability

Complexity:
    CATR: O(L · n_types · d) — negligible
    Router: O(L · n_experts · d) — negligible vs FFN
    FFN: O(L · k · d · d_ff) — main cost, k × cheaper than dense

Memory:
    Expert weights: O(E · d · d_ff)
    Routing overhead: O(B · L · (n_types + n_experts))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ContentAwareRouter(nn.Module):
    """Token-type-aware router for SMoE.

    First classifies tokens into semantic types, then routes each type
    through a different expert assignment function.

    Args:
        d_model: token feature dimension
        n_types: number of semantic token categories
        n_experts: total number of FFN experts
        n_active: top-k experts to activate per token
        temperature: softmax temperature for type and expert routing
    """

    def __init__(
        self,
        d_model: int,
        n_types: int = 4,
        n_experts: int = 8,
        n_active: int = 2,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_types = n_types
        self.n_experts = n_experts
        self.n_active = n_active

        # Type classifier: token -> semantic type
        self.type_classifier = nn.Linear(d_model, n_types, bias=False)

        # Per-type expert routers
        # Each type learns its own routing function
        self.type_routers = nn.Parameter(
            torch.randn(n_types, n_experts, d_model) * 0.02
        )

        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())

        # Learnable gating bias
        self.gate_bias = nn.Parameter(torch.zeros(n_experts))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.type_classifier.weight, std=0.02)

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def _compute_router_logits(
        self, x: torch.Tensor, router_weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute expert logits given router weights.

        Args:
            x: (batch, seq_len, d_model)
            router_weights: (n_experts, d_model)

        Returns:
            logits: (batch, seq_len, n_experts)
        """
        return F.linear(x, router_weights, self.gate_bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            expert_indices: (batch, seq_len, n_active) — indices of selected experts
            expert_weights: (batch, seq_len, n_active) — softmax-normalized weights
            aux_loss: scalar — load balancing + z-loss
            type_probs: (batch, seq_len, n_types) — type classification probabilities
        """
        batch, seq_len, d = x.shape

        # Step 1: Classify tokens into types
        type_logits = self.type_classifier(x)  # (B, L, n_types)
        type_probs = F.softmax(type_logits / self.temperature, dim=-1)

        # Step 2: Compute per-type router logits
        # type_routers: (n_types, n_experts, d)
        # For each type, compute logits
        all_type_logits = torch.einsum("bld,ted->blte", x, self.type_routers)  # (B, L, n_types, n_experts)

        # Step 3: Weighted combination by type probabilities
        # gate_logits = Σ_t type_probs[t] * router_logits_t
        gate_logits = torch.einsum("blt,blte->ble", type_probs, all_type_logits)  # (B, L, n_experts)

        # Step 4: Softmax + top-k selection
        gate_scores = F.softmax(gate_logits / self.temperature, dim=-1)

        # Top-k selection
        expert_weights, expert_indices = torch.topk(gate_scores, self.n_active, dim=-1)

        # Re-normalize weights of selected experts
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Step 5: Auxiliary loss for load balancing
        # f_e: fraction of tokens routed to expert e
        # P_e: mean gate probability for expert e
        with torch.no_grad():
            # Count routing decisions
            expert_mask = F.one_hot(expert_indices, num_classes=self.n_experts)  # (B, L, k, E)
            expert_mask = expert_mask.float().sum(dim=-2)  # (B, L, E) — which experts this token routes to
            f_e = expert_mask.mean(dim=[0, 1])  # (E,) — fraction of tokens routed to expert e
            P_e = gate_scores.mean(dim=[0, 1])  # (E,) — mean gate probability

        load_balance_loss = self.n_experts * (f_e * P_e).sum()

        # Z-loss for training stability
        z_loss = gate_logits.logsumexp(dim=-1).pow(2).mean()

        aux_loss = load_balance_loss + 0.01 * z_loss

        return expert_indices, expert_weights, aux_loss, type_probs

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_types={self.n_types}, n_experts={self.n_experts}, n_active={self.n_active}"


class SparseMoE_FFN(nn.Module):
    """Sparse Mixture of Experts Feed-Forward Network.

    Uses SwiGLU-T activation inside each expert for additional sparsity.

    Args:
        d_model: input/output dimension per token
        d_ff: hidden dimension per expert
        n_experts: total number of FFN experts
        n_active: number of experts active per token
        activation: "swiglu_t" or "swiglu"
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = 8192,
        n_experts: int = 8,
        n_active: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.n_active = n_active

        # Expert parameters — stacked as one large tensor for efficient computation
        # Each expert: W_gate, W_up, W_down
        self.W_gate = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.W_up = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.W_down = nn.Parameter(torch.empty(n_experts, d_ff, d_model))

        # Learnable thresholds for SwiGLU-T per expert
        self.thresholds = nn.Parameter(torch.zeros(n_experts, d_ff))
        self.log_beta = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        std = 0.02 / (2 * self.n_experts) ** 0.5
        for param in [self.W_gate, self.W_up, self.W_down]:
            nn.init.normal_(param, std=std)

    def _expert_forward(
        self, expert_idx: int, x: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through a single expert with SwiGLU-T activation.

        Args:
            expert_idx: which expert
            x: (n_tokens, d_model) — tokens assigned to this expert

        Returns:
            output: (n_tokens, d_model)
        """
        gate = F.silu(torch.mm(x, self.W_gate[expert_idx]))  # (n, d_ff)
        up = torch.mm(x, self.W_up[expert_idx])  # (n, d_ff)

        gated = gate * up

        # Thresholded sparsity
        beta = torch.exp(self.log_beta)
        mask = torch.sigmoid(beta * (up.abs() - self.thresholds[expert_idx]))
        gated = gated * mask

        output = torch.mm(gated, self.W_down[expert_idx])  # (n, d_model)
        return output

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            expert_indices: (batch, seq_len, n_active) — expert indices per token
            expert_weights: (batch, seq_len, n_active) — normalized weights

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, d = x.shape

        # Flatten for computation
        x_flat = x.reshape(-1, d)  # (B*L, d_model)
        indices_flat = expert_indices.reshape(-1, self.n_active)  # (B*L, n_active)
        weights_flat = expert_weights.reshape(-1, self.n_active)  # (B*L, n_active)

        output = torch.zeros_like(x_flat)

        # Process each expert
        for expert_idx in range(self.n_experts):
            # Find tokens assigned to this expert (in any of the top-k slots)
            token_mask = (indices_flat == expert_idx)  # (B*L, n_active)
            token_has_expert = token_mask.any(dim=-1)  # (B*L,)

            if token_has_expert.sum() == 0:
                continue

            # Get tokens assigned to this expert
            token_indices = token_has_expert.nonzero(as_tuple=True)[0]
            expert_x = x_flat[token_indices]  # (n_assigned, d)

            # Compute expert output
            expert_out = self._expert_forward(expert_idx, expert_x)  # (n_assigned, d)

            # Compute weights for these tokens (sum over all top-k slots where expert is selected)
            token_weights = torch.zeros(token_indices.shape[0], device=x.device, dtype=x.dtype)
            for k in range(self.n_active):
                # At slot k, where is this expert selected?
                slot_mask = indices_flat[:, k] == expert_idx  # (B*L,)
                token_mask_k = slot_mask[token_indices]  # (n_assigned,)
                slot_weights = weights_flat[:, k][token_indices]  # (n_assigned,)
                token_weights = token_weights + token_mask_k.float() * slot_weights

            # Weight and accumulate
            output[token_indices] += expert_out * token_weights.unsqueeze(-1)

        output = output.reshape(batch, seq_len, d)
        return output

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}, n_experts={self.n_experts}, n_active={self.n_active}"


def test_catr_smoe():
    """Smoke test for CATR + SMoE."""
    router = ContentAwareRouter(d_model=512, n_types=4, n_experts=8, n_active=2)
    ffn = SparseMoE_FFN(d_model=512, d_ff=2048, n_experts=8, n_active=2)

    x = torch.randn(2, 16, 512)
    indices, weights, aux_loss, type_probs = router(x)
    assert indices.shape == (2, 16, 2)
    assert weights.shape == (2, 16, 2)
    assert type_probs.shape == (2, 16, 4)

    output = ffn(x, indices, weights)
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"

    print(f"CATR+SMoE test passed. Aux loss: {aux_loss.item():.4f}")


if __name__ == "__main__":
    test_catr_smoe()
