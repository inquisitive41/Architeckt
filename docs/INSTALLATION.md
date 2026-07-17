# Architeckt Installation Guide

This document outlines the system requirements and setup instructions for both local inference environments and distributed training clusters.

## 💻 Hardware Requirements

### Inference (Minimum)
- **GPU:** 1x NVIDIA GPU with at least 24GB VRAM (e.g., RTX 3090, RTX 4090, A10g).
- **RAM:** 32 GB System RAM.
- **Storage:** 50 GB of free space (SSD/NVMe strongly recommended).

> [!NOTE]
> Running the 11B parameter Architeckt model on 24GB VRAM strictly requires 8-bit quantization (`bitsandbytes`). For fp16/bf16 inference, you need at least 32GB VRAM.

### Training (Recommended)
- **GPU:** 8x NVIDIA A100 (80GB) or H100 nodes.
- **Interconnect:** NVLink or NVSwitch (to handle FSDP tensor sharding).
- **Storage:** High-performance network-attached storage (for large pre-tokenized corpora).

---

## 🛠 Software Requirements

Architeckt is a pure PyTorch implementation but relies on the latest CUDA features.

- **OS:** Linux (Ubuntu 22.04+) or Windows 11 (with WSL2).
- **Python:** `3.10` or higher.
- **PyTorch:** `2.1` or higher (with CUDA 11.8 / 12.1+ support).

---

## 📦 Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/Architeckt.git
cd Architeckt
```

### Step 2: Create a Virtual Environment

Using `conda` (recommended):
```bash
conda create -n architeckt python=3.10
conda activate architeckt
```

Or using `venv`:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Core Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pyyaml tqdm
```

### Step 4: Install Optional Utilities

To enable 8-bit quantization for local inference on consumer GPUs:
```bash
pip install bitsandbytes
```

To run the benchmarking suite (latency, memory profiling):
```bash
pip install matplotlib pandas
```

> [!WARNING]
> If `bitsandbytes` throws a CUDA error on Windows, make sure you are either using WSL2 or the unofficial Windows binary `bitsandbytes-windows`.

---

## 🚦 Verifying the Installation

After installing, run the component smoke tests to ensure your CUDA graphs and kernel configurations are functional.

```bash
export PYTHONPATH="src"
python src/attention/adaptive_heads.py
python src/attention/linear_attention.py
python src/activations/swiglu_t.py
```

If all tests output `test passed`, you are ready to start training or generating text!
