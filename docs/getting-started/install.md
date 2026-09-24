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
| `mcp` | FastMCP ≥ 3.0 | MCP server with 23 quantum tools for AI assistants |
| `metal` | MLX ≥ 0.20 | Apple Silicon Metal GPU native quantum simulation |
| `torch` | PyTorch ≥ 2.0 | Hybrid QML (`QuantumLayer`, `ContinuousResonantLayer`) |
| `qml` | Scikit-learn ≥ 1.0 | Quantum classifiers, regressors & QSVM kernels |
| `dev` | pytest, ruff, mypy, hypothesis | Comprehensive development & formal testing |
| `gpu` | JAX, CuPy, cuQuantum | NVIDIA CUDA GPU-accelerated statevector simulation |
