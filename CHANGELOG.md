# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multi-Scale Linearized Attention (MSLA) prototype.
- Sparse Mixture of Experts (SMoE) layers.
- Adaptive Head Gating (AHG) implementation.
- SwiGLU-T activation functions.
- FSDP support in the training loop.
- 8-bit quantization loading utils via `bitsandbytes`.
- FLOPs, Latency, and Memory benchmarks.

### Fixed
- Fixed the hard-gating threshold computation in `adaptive_heads.py` to correctly drop specific percentages of heads during inference.

## [0.1.0] - 2026-07-17

### Added
- Initial project structure created.
- Core architecture drafted in documentation.
