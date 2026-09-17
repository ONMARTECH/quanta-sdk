# Quanta SDK

An AI-native, modular, and high-performance quantum computing SDK for Python. **v1.1.0** — [PyPI](https://pypi.org/project/quanta-sdk/)

## Overview

Quanta delivers a modern, multi-tier quantum computing runtime designed for AI agents (via MCP), deep learning researchers, and production workloads:

- **Deep Learning Layer (`quanta.torch`)**: PyTorch `nn.Module` integration with analytical parameter-shift autograd (`QuantumLayer`) and continuous-time quantum resonance (`ContinuousResonator`, `BiomorphicResonantBrain`).
- **Layer 3 (Declarative)**: `search()`, `optimize()`, `vqe()`, `factor()`, `resolve()`, `MultiAgentSystem` — high-level algorithms without manual gate synthesis.
- **Layer 2 (Circuit DSL)**: `@circuit`, 31 built-in gates (H, CX, RZ, MS, ECR, etc.), parametric rotations, and measurement handling.
- **Layer 1 (Physical & Hardware Acceleration)**:
  - **Apple Silicon Metal/MLX**: Native tensor contractions on Apple Unified Memory delivering up to **404x speedup**.
  - **NVIDIA cuStateVec**: Large-scale GPU statevector simulation.
  - **Multi-Cloud Hardware**: Live verified execution on IonQ Cloud REST API v0.3, IBM Quantum Heron r3 (156 qubits), and Google Cirq Sycamore.
- **Agentic MCP Layer**: **23 Model Context Protocol (MCP) tools** empowering AI assistants (Claude, Gemini, GPT) to author, transpile, simulate, and optimize quantum workflows.

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

Explore detailed documentation in the `docs/` directory:

- [Architecture](ARCHITECTURE_EN.md)
- [Features & Gate Set](FEATURES_EN.md)
- [Comparison with Other SDKs](COMPARISON_EN.md)
- [Installation Guide](INSTALL_TR.md)
- [Academic Whitepapers & Proofs](theory/quantum_brain_frontiers.md)

---

## Author & Contact

**Abdullah Enes SARI** — ONMARTECH  
Email: info@onmartech.com  
Website: [onmartech.com](https://onmartech.com)

## License

Apache License 2.0
