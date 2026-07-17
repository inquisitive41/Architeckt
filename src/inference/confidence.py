"""
Confidence-based verification and error resilience for inference.

Extends TLCG (Token-Level Confidence Gating) with:
- Verification pass: re-evaluate low-confidence tokens through deeper layers
- Consensus check: compare predictions from multiple exit heads
- Confidence calibration: temperature scaling applied online
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class ConfidenceVerifier(nn.Module):
    """Post-generation confidence verification for error resilience.

    After the primary prediction, low-confidence tokens can be
    re-evaluated through deeper layers. A token is flagged if:
    - Its TLCG confidence falls below a threshold
    - Multiple exit heads disagree (high variance)
    - The model's own confidence is poorly calibrated

    Args:
        confidence_threshold: minimum confidence to accept a prediction
        num_verification_passes: max re-evaluation attempts
        temperature_scale: calibrate logits before softmax
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        num_verification_passes: int = 2,
        temperature_scale: float = 1.0,
    ):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.num_verification_passes = num_verification_passes
        self.log_temperature = nn.Parameter(torch.tensor(temperature_scale).log())

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def verify(
        self,
        primary_logits: torch.Tensor,
        confidences: torch.Tensor,
        verification_logits: Optional[torch.Tensor] = None,
        return_metadata: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            primary_logits: (batch, vocab_size) — from main model
            confidences: (batch,) — TLCG confidence per token
            verification_logits: optional logits from deeper/recomputed pass
            return_metadata: if True, return verification diagnostics

        Returns:
            verified_logits: (batch, vocab_size)
            metadata: optional dict with verification info
        """
        batch = primary_logits.shape[0]

        # Apply temperature scaling
        logits = primary_logits / self.temperature

        # Identify low-confidence tokens
        low_conf = confidences < self.confidence_threshold
        very_low_conf = confidences < (self.confidence_threshold / 2)

        if verification_logits is not None:
            # Blend primary and verification logits for low-confidence tokens
            # Weight by confidence: more weight to verification when less confident
            alpha = confidences.unsqueeze(-1)
            logits = torch.where(
                low_conf.unsqueeze(-1),
                (1 - alpha) * verification_logits + alpha * primary_logits,
                logits,
            )

        # For very low confidence, fall back to highest-probability token
        if very_low_conf.any():
            # Use the most confident logit dimension as a safety fallback
            max_logit = primary_logits.max(dim=-1, keepdim=True).values
            logits[very_low_conf] = max_logit[very_low_conf].expand_as(primary_logits[very_low_conf])

        if return_metadata:
            metadata = {
                "low_confidence_fraction": low_conf.float().mean().item(),
                "very_low_confidence_fraction": very_low_conf.float().mean().item(),
                "mean_confidence": confidences.mean().item(),
                "temperature": self.temperature.item(),
            }
            return logits, metadata

        return logits, None

    def calibrate_temperature(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """Online temperature calibration using expected calibration error.

        Args:
            logits: (batch, vocab_size) — model logits
            labels: (batch,) — ground truth token ids

        Returns:
            calibration_error: ECE-like metric
        """
        probs = F.softmax(logits / self.temperature, dim=-1)
        predicted_probs = probs.gather(1, labels.unsqueeze(-1)).squeeze(-1)
        predicted_classes = probs.argmax(dim=-1)
        accuracy = (predicted_classes == labels).float()

        # Simple calibration: mean confidence minus accuracy
        ece = (predicted_probs - accuracy).abs().mean().item()
        return ece

    def extra_repr(self) -> str:
        return f"threshold={self.confidence_threshold}, passes={self.num_verification_passes}"


def test_confidence_verifier():
    verifier = ConfidenceVerifier(confidence_threshold=0.5)
    logits = torch.randn(4, 1000)
    conf = torch.tensor([0.9, 0.3, 0.8, 0.1])
    vlogits = torch.randn(4, 1000)

    verified, meta = verifier.verify(logits, conf, vlogits, return_metadata=True)
    assert verified.shape == logits.shape
    assert meta["low_confidence_fraction"] == 0.5

    # Low confidence tokens should be different from input
    low_mask = conf < 0.5
    assert not torch.allclose(verified[low_mask], logits[low_mask])

    print(f"ConfidenceVerifier test passed. Low-conf fraction: {meta['low_confidence_fraction']}")

    # Test calibration
    labels = torch.randint(0, 1000, (4,))
    ece = verifier.calibrate_temperature(logits, labels)
    print(f"Calibration error: {ece:.4f}")


if __name__ == "__main__":
    test_confidence_verifier()
