# Frequently Asked Questions (FAQ)

## General Questions

### What makes Architeckt different from LLaMA or GPT?
Traditional models use Softmax Attention, which scales quadratically ($O(L^2)$) with sequence length. Architeckt uses **Multi-Scale Linearized Attention (MSLA)**, which scales linearly ($O(L)$). This means Architeckt can read infinitely long contexts without crashing. Additionally, it uses Sparse Mixture of Experts (SMoE) and Adaptive Head Gating (AHG) to drastically reduce the number of active parameters per token.

### What is the context limit of Architeckt?
Theoretically, there is **no hard context limit**. The KV-cache does not grow with the sequence length. In practice, the model is limited only by its trained positional embeddings (RoPE), but these can be dynamically extrapolated via methods like YaRN or Position Interpolation.

---

## Technical Troubleshooting

### "CUDA Out Of Memory" during Inference
**Issue:** You get an OOM error when loading the model on a 24GB GPU.
**Solution:** The 11B parameter dense model weighs ~23GB in bf16. To fit it inside a 24GB GPU, you must use 8-bit quantization. Ensure `bitsandbytes` is installed and set `use_8bit=True` in `load_for_inference`.

### "ModuleNotFoundError: No module named 'bitsandbytes'" on Windows
**Issue:** `bitsandbytes` fails to install or import on Windows.
**Solution:** `bitsandbytes` officially supports Linux. On Windows, you have two options:
1. Run your code inside **WSL2** (Windows Subsystem for Linux).
2. Install the unofficial Windows binary: `python -m pip install bitsandbytes-windows`.

### Model Loss Spikes / NaN during Training
**Issue:** Loss suddenly spikes to infinity (NaN) during mixed-precision training.
**Solution:** 
- Ensure you are using `bfloat16` instead of `float16`. Standard `fp16` lacks the dynamic range required for Linearized Attention accumulations.
- Check that your gradient clipping is enabled (`max_grad_norm=1.0`).

### FSDP crashes with "ProcessGroupNCCL" error
**Issue:** `ArchitecktTrainer` throws an NCCL error upon initialization.
**Solution:** This usually happens if you are running the script on a machine without NVLink or in an unsupported Windows environment. PyTorch FSDP requires a proper distributed environment (Linux, NCCL backend). If you are testing locally on a single GPU, ensure `use_fsdp=False` in your `TrainingConfig`.
