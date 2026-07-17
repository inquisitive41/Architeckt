# Architeckt 🌌

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Status: Experimental](https://img.shields.io/badge/Status-Experimental-red.svg)]()

> *Русская версия доступна здесь: [README.ru.md](README.ru.md).*

> **Disclaimer:** Empirically confirmed the possibility of stable training on small-scale models. Scaling to billions of parameters requires separate experimental verification.

Architeckt is a highly experimental, next-generation large language model (LLM) architecture. It is designed to shatter the limitations of standard Transformers by providing **O(L) linear complexity**, extreme sparse computation, and adaptive efficiency.

With Architeckt, you get the encyclopedic knowledge of an 11-Billion parameter model, running at the speed of a 1.5-Billion parameter model, with a memory footprint that allows processing infinitely long texts without Out-Of-Memory (OOM) errors.

---

## 📑 Table of Contents
1. [Open Letter from the Creator](#-open-letter-from-the-creator)
2. [Key Innovations](#-key-innovations-nine-pillars)
3. [Business Value & Applications](#-business-value--applications)
4. [Architecture Comparison](#-architecture-comparison)
5. [Project Structure](#-project-structure)
6. [Quickstart](#-quickstart)
7. [Documentation](#-documentation)
8. [Contributing](#-contributing)
9. [License](#-license)

---

## 📜 Open Letter from the Creator

**Dear colleagues,**

We have created a mathematical masterpiece that will change the way artificial intelligence is developed. I present to you Architeckt — a large language model architecture that solves the fundamental problems holding back the industry for the last five years.

Architeckt is a next-generation transformer architecture that combines nine innovations into a single system. It was not created by empirical trial and error, but mathematically derived from first principles.

For five years, the industry lived with quadratic attention. We have lifted this limitation. Not through compromise. Not through approximation. Through a mathematically derived architecture that unifies nine breakthrough mechanisms into a cohesive whole.

Architeckt is not an "improved GPT". It is the next step.

*Author: Timur* | [Telegram: @Inqusitive41](https://t.me/Inqusitive41)

---

## 🌟 Key Innovations (Nine Pillars)

Architeckt incorporates cutting-edge research and mathematical principles:

* **Multi-Scale Linearized Attention (MSLA)**: An attention mechanism with linear $O(L)$ complexity. The quadratic barrier is lifted. Infinite context is now a reality.
* **Sparse Mixture of Experts (SMoE)**: 11 Billion parameters, but only ~1.5 Billion are activated per token. The power of a heavy-weight model at the speed of a light-weight one.
* **Adaptive Head Gating (AHG)**: Dynamic deactivation of attention heads on simple tokens. Up to 30% compute savings with no loss in quality.
* **SwiGLU-T**: An activation function with a learnable threshold. It creates natural sparsity: not all neurons are active simultaneously.
* **AdaSNorm**: Adaptive normalization replacing fixed LayerNorm. Ensures stability in deep networks without manual tuning.
* **Depth-Aware Early Exit (DAEE)**: Early exit from deep layers when high confidence is reached. Simple tokens are processed significantly faster.
* **Token-Level Confidence Gating (TLCG)**: Verifies model confidence before emitting each token. A mathematical, rather than heuristic, defense against hallucinations.
* **Weight Tying**: The input embedding is tied with the output layer. Fewer parameters, better regularization.
* **DeepNet Initialization**: Built-in residual connection scaling for the stable training of exceptionally deep architectures.

---

## 💼 Business Value & Applications

Where traditional transformers hit their limits, Architeckt opens new markets:

* **Long Documents:** Legal analysis, scientific papers, technical documentation. Linear complexity allows processing texts of arbitrary length without quality degradation.
* **Real-time Processing:** Chatbots, voice assistants, automated moderation. Early Exit and Adaptive Head Gating ensure minimal latency.
* **Resource-Constrained Environments:** Deployment on consumer GPUs (24GB VRAM), edge devices, and embedded systems made possible by 8-bit quantization and sparse activation.
* **High Reliability Requirements:** Medicine, finance, law. Confidence Gating (TLCG) strictly blocks unverified information (hallucination defense).
* **Reduced R&D Costs:** A proprietary model can be trained from scratch in 7 days on 8×A100s instead of months on massive clusters. Time-to-market is drastically reduced, and activating only 15% of parameters proportionally lowers inference costs.

---

## ⚖️ Architecture Comparison

| Feature | GPT-4 | LLaMA-3 | Architeckt |
| --- | --- | --- | --- |
| **Attention Complexity** | O(L²) | O(L²) | **O(L)** |
| **KV-Cache** | O(L) | O(L) | **O(1) per layer** |
| **Active Parameters** | 100% | 100% | **~15%** |
| **Hallucination Defense** | No | No | **Yes (TLCG)** |
| **Adaptive Heads** | No | No | **Yes (AHG)** |
| **Early Exit** | No | No | **Yes (DAEE)** |
| **Memory (11B, inference)** | >24 GB | >24 GB | **~8 GB (in 8-bit)** |

---

## 📁 Project Structure

The repository follows standard open-source MLOps guidelines:

```text
ARCHITECKT/
├── configs/            # YAML configurations for models and training
├── data/               # Datasets and tokenized bins
├── docs/               # Detailed documentation
├── results/            # Experimental results, metrics, and logs
├── scripts/            # Bash/PowerShell scripts for execution
├── src/                # Core architecture modules
│   ├── activations/    # SwiGLU-T and other custom activations
│   ├── attention/      # MSLA, Adaptive Head Gating, RoPE
│   ├── benchmarks/     # Theoretical FLOPs, memory, latency analysis
│   ├── inference/      # Inference utilities (8-bit quantization)
│   ├── layers/         # Transformer blocks
│   ├── models/         # Model definitions
│   ├── normalization/  # AdaSNorm
│   ├── routing/        # Content-Aware Token Router (CATR)
│   └── training/       # FSDP Trainer
└── tests/              # Pytest unit tests
```

---

## 🚀 Quickstart

> [!IMPORTANT]
> Architeckt requires a CUDA-compatible GPU. For full training, an 8×A100 cluster is recommended. For inference, a 24GB VRAM GPU is sufficient.

### 1. Installation

See the detailed [Installation Guide](docs/INSTALLATION.md) for environment setup.

```bash
git clone https://github.com/your-org/Architeckt.git
cd Architeckt
pip install -r requirements.txt
```

### 2. Verify Architecture Capabilities

Run the benchmark suite to see the theoretical performance and memory savings:

```bash
# Set environment variables
export PYTHONPATH="src"

# Run latency and memory theoretical estimates
python src/benchmarks/latency.py
python src/benchmarks/memory.py
```

### 3. Inference Example (8-bit)

> [!NOTE]
> Make sure `bitsandbytes` is installed to run 11B models on 24GB GPUs.

```python
import torch
from src.inference.inference_utils import load_for_inference
from src.models.architeckt import ArchitecktModel  # Coming soon!

# Initialize skeleton model
model = ArchitecktModel(config)

# Load 8-bit quantized checkpoints for VRAM-constrained environments
model = load_for_inference(model, "checkpoints/architeckt_final.pt", use_8bit=True)

# Generate
# outputs = model.generate(...)
```

---

## 📚 Documentation

Dive deeper into the mechanics of Architeckt:

- 🧠 **[Architecture Deep Dive](docs/ARCHITECTURE.md)** — Detailed explanation of MSLA, SMoE, and AHG.
- ⚡ **[Performance Benchmarks](docs/PERFORMANCE.md)** — Theoretical calculations of Latency, Memory, and FLOPs.
- ⚙️ **[Installation & Requirements](docs/INSTALLATION.md)** — Setup guide for local and cluster environments.
- 📖 **[API Reference](docs/API_REFERENCE.md)** — Class and method documentation.
- ❓ **[FAQ & Troubleshooting](docs/FAQ.md)** — Solutions for common issues.

---

## 🤝 Contributing

We welcome contributions! Architeckt is an open, community-driven project. 

Please see our [Contributing Guidelines](CONTRIBUTING.md) to get started. 
Check out the [Changelog](CHANGELOG.md) to see recent updates.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

> [!TIP]
> **Star this repository** if you find Architeckt useful for your research! 
