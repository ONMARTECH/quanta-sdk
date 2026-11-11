# Installation

## Quick Install

```bash
pip install quanta-sdk
```

## With MCP Server

```bash
pip install "quanta-sdk[mcp]"
```

## Development

```bash
git clone https://github.com/ONMARTECH/quanta-sdk.git
cd quanta-sdk
pip install -e ".[dev]"
```

## Requirements

- Python 3.10+
- NumPy ≥ 1.24

## Optional Dependencies

| Extra | Packages | Purpose |
|-------|----------|---------|
| `mcp` | `fastmcp>=3.0` | MCP server with 25 quantum tools for AI assistants |
| `metal` | `mlx>=0.20, mlx-metal>=0.20` | Apple Silicon Metal GPU native quantum simulation |
| `torch` | `torch>=2.0` | Hybrid QML (`QuantumLayer`, `ContinuousResonantLayer`) |
| `qml` | `scikit-learn>=1.0` | Quantum classifiers, regressors & QSVM kernels |
| `docs` | `mkdocs-material>=9.5, mkdocstrings[python]>=0.24` | Documentation site & API reference generation |
| `dev` | `pytest, ruff, mypy, hypothesis` | Comprehensive development & formal testing |
| `gpu` | `jax[cuda12], cupy-cuda12x, cuquantum-python-cu12` | NVIDIA CUDA GPU-accelerated statevector simulation |
