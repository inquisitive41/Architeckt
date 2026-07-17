# API Reference

This document provides a high-level overview of the programmable interfaces and core classes of the Architeckt codebase. It is designed for developers building on top of Architeckt or writing custom training loops.

## Core Modules

### `src.attention.linear_attention.MSLA`
**Multi-Scale Linearized Attention**
The core attention mechanism replacing Softmax quadratic attention.

```python
class MSLA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, n_scales: int):
        ...
```
- **Inputs:** `x` Tensor of shape `(batch, seq_len, d_model)`.
- **Outputs:** `out` Tensor of shape `(batch, seq_len, d_model)`.
- **Description:** Computes linear attention using cumulative sums with exponential decay. Contains $O(1)$ memory characteristics for KV caching.

### `src.attention.adaptive_heads.AdaptiveHeadGating`
**AHG Gating Router**
Drops low-confidence attention heads during inference.

```python
class AdaptiveHeadGating(nn.Module):
    def __init__(self, d_model: int, n_heads: int, threshold_percentile: float = 0.3):
        ...
```
- **Inputs:** `x` Tensor `(batch, seq_len, d_model)`.
- **Outputs:** `gate` Tensor `(batch, seq_len, n_heads, 1, 1)`.
- **Usage:** Multiply the output of the attention heads by `gate`. If in `eval()` mode, the lowest 30% of heads (by default) will be strictly zeroed out.

### `src.activations.swiglu_t.SwiGLU_T`
**Thresholded SwiGLU**
Induces sparsity in FFNs.

```python
class SwiGLU_T(nn.Module):
    def __init__(self, dim: int):
        ...
```
- **Description:** Applies `SiLU` multiplied by a soft/hard mask parameterized by a learnable threshold. 

## Training Utilities

### `src.training.trainer.ArchitecktTrainer`
**Main Training Loop**
Handles standard PyTorch training, Mixed Precision (bf16), Gradient Accumulation, and Distributed FSDP training.

```python
class ArchitecktTrainer:
    def __init__(self, model: nn.Module, config: TrainingConfig, device: Optional[torch.device] = None):
        ...
```
- **Methods:**
  - `train()`: Starts the main epoch/step loop.
  - `save_checkpoint(name: str)`: Saves weights to disk.
  - `load_checkpoint(...)`: Restores states.

### `src.training.trainer.TrainingConfig`
Dataclass configuration for hyperparameters.
- **Attributes:** `learning_rate`, `use_fsdp`, `activation_checkpointing`, `max_steps`, etc.

## Inference Utilities

### `src.inference.inference_utils.load_for_inference`
**VRAM-friendly model loader.**

```python
def load_for_inference(model: nn.Module, checkpoint_path: str, use_8bit: bool = True) -> nn.Module
```
- **Description:** Parses a saved PyTorch checkpoint and dynamically replaces `nn.Linear` layers with `bitsandbytes.nn.Linear8bitLt`.
- **Warning:** Fails gracefully to standard precision if `bitsandbytes` is missing.
