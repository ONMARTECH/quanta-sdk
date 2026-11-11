# Quanta SDK

An AI-native, modular, and high-performance quantum computing SDK for Python. **v1.2.0-production** — [PyPI](https://pypi.org/project/quanta-sdk/) · [Documentation](https://quanta.onmartech.com/) · [GitHub Release](https://github.com/ONMARTECH/quanta-sdk/releases/tag/v1.2.0)

## Overview

Quanta delivers a standalone quantum software architecture engineered for 2026 quantum computing frontiers, AI agents (via MCP), deep learning researchers, and fault-tolerant quantum computing (FTQC) workloads. The framework is grounded in 5 core scientific paradigms:

1. **Local-First First-Principles Core**: Core circuit/compiler/statevector require only NumPy (SciPy & NetworkX power Layer-3 solvers and QEC decoders) without heavy C++/LLVM build chains or CUDA prerequisites, featuring **34 native gates**, universal **1Q ZYZ / Heron / 2Q Cartan KAK $SU(4)$ unitary synthesis**, and **127Q/156Q IBM Heavy-Hex routing** across IBM Heron, Google Sycamore, and IonQ architectures.
2. **Apple Silicon Metal / MLX Zero-Copy GPU Acceleration**: High-performance tensor contraction engine eliminating CPU-GPU memory copy overhead on Apple Unified Memory (up to 58x–404x speedup depending on gate depth and qubit count), alongside a SIMD-vectorized >3.13M gates/s Clifford tableau simulator.
3. **Continuous Hilbert Gradients & Daleckii-Krein Autograd (`quanta.torch`)**: Analytical closed-form Fréchet matrix derivative autograd maintaining machine precision ($10^{-15}$ error), dynamical Lie algebra ($\dim(\mathfrak{g})$) barren plateau diagnosis, and quantum- and neurobiology-inspired differentiable continuous-time Hamiltonian PyTorch architectures (`ContinuousResonantLayer`, `BiomorphicResonantBrain`).
4. **2026 Dual-Track Fault-Tolerance Engine (FTQC)**:
   - **Track A (2D Topological)**: Edmonds Blossom MWPM and Delfosse-Nickerson Union-Find decoding, Kubica-Delfosse $A_2$ lattice + ILP Color Code decoding, and Google Willow-compliant 3D spacetime syndrome extraction.
   - **Track B (High-Rate qLDPC)**: Canonical Gross $[[144, 12, 12]]$ Bivariate Bicycle qLDPC code achieving $12\times$ physical qubit reduction over surface codes, paired with a native Normalized Min-Sum BP-OSD decoder.
   - **Non-Clifford Universality**: 15-to-1 Bravyi-Kitaev magic state distillation factory ($\epsilon_{\text{out}} \le 35 p^3$) and planar lattice surgery.
5. **Agentic MCP Integration Layer**: **25 Model Context Protocol (MCP) tools** empowering autonomous AI agents (Claude, Gemini, GPT) to design, transpile, simulate, and benchmark quantum systems.
6. **Falsifiable Empiricism & Certified Rigor**: **2,826+ automated tests** ensuring exact unitarity ($\|U^\dagger U - I\| < 10^{-14}$) and CPTP trace preservation.

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
    QuantumLayer(num_qubits=4, circuit_fn="hardware_efficient", num_layers=2),
    nn.Linear(4, 2)
)

x = torch.randn(8, 4, requires_grad=True)
out = model(x)
loss = out.sum()
loss.backward()  # Exact analytical parameter-shift gradients
```

### 3. Biomorphic Resonant Brain (`BiomorphicResonantBrain`)

`BiomorphicResonantBrain` is a **quantum- and neurobiology-inspired differentiable continuous-time Hamiltonian PyTorch architecture** (a classical deep-learning representation model inspired by open quantum dynamics and neuromodulation, not a claim of biological wetware quantum advantage):

```python
import torch
from quanta.torch.brain import BiomorphicResonantBrain, QuantumREMSleep

# Dual-hemisphere differentiable Hamiltonian PyTorch architecture
brain = BiomorphicResonantBrain(in_features=16, num_left_qubits=2, num_right_qubits=2)

x = torch.randn(2, 16)
output = brain(x)

# Multi-timescale Hamiltonian phase-dispersion replay for continual learning
rem = QuantumREMSleep(brain_module=brain, sleep_cycles=3)
rem.register_memory_state(output["state"])
stats = rem.sleep()
print("Retention Estimate:", stats["retention_estimate"])
```

---

## Hardware Acceleration (Apple Silicon Metal)

Achieve up to 58x–404x speedup depending on gate depth and qubit count on Apple Silicon Metal GPU:

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
- **GitHub Release**: [GitHub Release v1.2.0](https://github.com/ONMARTECH/quanta-sdk/releases/tag/v1.2.0)

## License

Apache License 2.0

