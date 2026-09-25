# Quanta SDK

An AI-native, modular, and high-performance quantum computing SDK for Python. **v1.2.0-production** — [PyPI](https://pypi.org/project/quanta-sdk/) · [Documentation](https://quanta.onmartech.com/) · [Zenodo DOI](https://doi.org/10.5281/zenodo.22952779)

## Overview

Quanta delivers a standalone quantum software architecture engineered for 2026 quantum computing frontiers, AI agents (via MCP), deep learning researchers, and fault-tolerant quantum computing (FTQC) workloads. The framework is grounded in 5 core scientific paradigms:

1. **Local-First, Zero-Dependency First-Principles Core**: Full quantum execution engine operating in pure Python/NumPy without heavy C++/LLVM build chains or CUDA prerequisites, featuring 31 native gates with direct parity across IBM Heron, Google Sycamore, and IonQ architectures.
2. **Apple Silicon Metal / MLX Zero-Copy GPU Acceleration**: High-performance tensor contraction engine eliminating CPU-GPU memory copy overhead on Apple Unified Memory, alongside a SIMD-vectorized >3.13M gates/s Clifford tableau simulator.
3. **Continuous Hilbert Gradients & Daleckii-Krein Autograd (`quanta.torch`)**: Analytical closed-form Fréchet matrix derivative autograd maintaining machine precision ($10^{-15}$ error), dynamical Lie algebra ($\dim(\mathfrak{g})$) barren plateau diagnosis, and biomorphic continuous quantum brain dynamics (`ContinuousResonator`, `BiomorphicResonantBrain`).
4. **2026 Dual-Track Fault-Tolerance Engine (FTQC)**:
   - **Track A (2D Topological)**: Edmonds Blossom MWPM decoding eliminating greedy heuristic failure modes, with Google Willow-compliant 3D spacetime syndrome extraction.
   - **Track B (High-Rate qLDPC)**: Canonical Gross $[[144, 12, 12]]$ Bivariate Bicycle qLDPC code achieving $12\times$ physical qubit reduction over surface codes, paired with a native Normalized Min-Sum BP-OSD-0 decoder.
   - **Non-Clifford Universality**: 15-to-1 Bravyi-Kitaev magic state distillation factory ($\epsilon_{\text{out}} \le 35 p^3$) and planar lattice surgery.
5. **Agentic MCP Integration Layer**: **23 Model Context Protocol (MCP) tools** empowering autonomous AI agents (Claude, Gemini, GPT) to design, transpile, simulate, and benchmark quantum systems.
6. **Falsifiable Empiricism & Certified Rigor**: 2,076 fully automated regression-free tests ensuring exact unitarity ($\|U^\dagger U - I\| < 10^{-14}$) and CPTP trace preservation.

---

## Quick Start

### 1. Quantum Circuit DSL

```python
from quanta import circuit, H, CX, measure, run

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell, shots=1024)
print(result.summary())
```

### 2. PyTorch Differentiable `QuantumLayer`

```python
import torch
import torch.nn as nn
from quanta.torch import QuantumLayer

model = nn.Sequential(
    nn.Linear(4, 4),
    QuantumLayer(n_qubits=4, ansatz="hardware_efficient", n_layers=2),
    nn.Linear(4, 2)
)

x = torch.randn(8, 4, requires_grad=True)
out = model(x)
loss = out.sum()
loss.backward()  # Exact analytical parameter-shift gradients
```

### 3. Biomorphic Resonant Brain (`BiomorphicResonantBrain`)

```python
from quanta.torch import BiomorphicResonantBrain

# Dual-hemisphere quantum brain with 4 neuromodulators (DA, ACh, 5-HT, NE)
brain = BiomorphicResonantBrain(n_qubits=4)

x = torch.randn(1, 4)
output = brain(x)

# REM sleep continual learning (prevents catastrophic forgetting)
stats = brain.consolidate_rem_sleep(replay_cycles=3)
print("Consolidation Fidelity:", stats["retained_fidelity"])
```

---

## Hardware Acceleration (Apple Silicon Metal)

Achieve up to 404x speedup on Apple Silicon (M-series) hardware:

```python
from quanta import run

result = run(large_circuit, backend="mlx")
```

---

## Installation

```bash
# Standard release
pip install quanta-sdk

# With PyTorch and Deep Learning support
pip install "quanta-sdk[torch]"

# With Apple Silicon Metal (MLX) support
pip install "quanta-sdk[metal]"

# Development mode
git clone https://github.com/ONMARTECH/quanta-sdk.git
cd quanta-sdk
pip install -e ".[dev]"
pytest
```

---

## Documentation

Explore detailed documentation in the official portal:

- [Official Documentation Portal](https://quanta.onmartech.com/)
- [Quanta Architecture Whitepaper](papers/quanta_framework_paper.md)
- [Architecture](ARCHITECTURE_EN.md)
- [Features & Gate Set](FEATURES_EN.md)
- [Comparison with Other SDKs](COMPARISON_EN.md)
- [Installation Guide](INSTALL_TR.md)
- [Theoretical Foundations & Monographs](theory/quantum_brain_frontiers.md)

---

## Authorship & Identity Metadata

- **Lead Author & Principal Architect**: Abdullah Enes SARI (ORCID: [0000-0002-8827-0587](https://orcid.org/0000-0002-8827-0587))
- **Affiliation**: ONMARTECH Quantum Computing Initiative (`info@onmartech.com`)
- **Website**: [onmartech.com](https://onmartech.com) · [quanta.onmartech.com](https://quanta.onmartech.com)
- **Permanent Software DOI**: [10.5281/zenodo.22952779](https://doi.org/10.5281/zenodo.22952779)

## License

Apache License 2.0

