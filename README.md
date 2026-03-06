<p align="center">
  <h1 align="center">Quanta SDK</h1>
  <p align="center">
    <strong>Multi-paradigm quantum computing SDK for Python</strong>
  </p>
  <p align="center">
    <a href="https://github.com/ONMARTECH/quanta-sdk/releases"><img src="https://img.shields.io/badge/version-0.4.0-blue.svg" alt="Version"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-brightgreen.svg" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-orange.svg" alt="License"></a>
    <a href="#quality-benchmark"><img src="https://img.shields.io/badge/tests-150%2B%20passed-success.svg" alt="Tests"></a>
    <a href="#quality-benchmark"><img src="https://img.shields.io/badge/benchmark-8%2F8-gold.svg" alt="Benchmark"></a>
    <a href="#qasmbench"><img src="https://img.shields.io/badge/QASMBench-10%2F10-success.svg" alt="QASMBench"></a>
    <a href="https://github.com/ONMARTECH/quanta-sdk"><img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg" alt="Platform"></a>
  </p>
</p>

---

Quanta is a clean, modular quantum computing SDK designed for researchers, engineers and developers. It provides a 3-layer abstraction — from high-level declarative APIs (`search()`, `factor()`) to low-level DAG manipulation and QASM export — so you can work at the level that fits your problem.

**Key highlights:**
- Shor, VQE, QAOA, QSVM, Grover — production-grade quantum algorithms
- DAG-based IR with 3-pass compiler and topology-aware qubit routing
- Statevector simulator up to 27 qubits with optional JAX/CuPy GPU acceleration
- Surface code QEC, BB84 QKD, and error correction primitives
- Real-world demo: [quantum entity resolution](#11-entity-resolution) for customer deduplication

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Features](#features)
- [Algorithms](#algorithms)
- [Examples & Use Cases](#examples--use-cases)
- [Quality Benchmark](#quality-benchmark)
- [Installation](#installation)
- [Documentation](#documentation)
- [Author](#author)
- [License](#license)

## Quick Start

```python
from quanta import circuit, H, CX, measure, run

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell, shots=1024)
print(result)
```

```
╔══════════════════════════════════════════════════╗
║  Quanta Result: bell                            ║
╠──────────────────────────────────────────────────╣
║  |00>  ████████████████████  50.5%              ║
║  |11>  ███████████████████   49.5%              ║
╠──────────────────────────────────────────────────╣
║  0.71|00> + 0.71|11>                            ║
╚══════════════════════════════════════════════════╝
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Layer 3 — Declarative API                               │
│  search() · optimize() · vqe() · factor() · qsvm()      │
│  portfolio_optimize() · resolve() · multi_agent()        │
├──────────────────────────────────────────────────────────┤
│  Layer 2 — Circuit API                                   │
│  @circuit · H · CX · RZ · measure · run · sweep          │
│  custom_gate() · 17 built-in gates                       │
├──────────────────────────────────────────────────────────┤
│  Layer 1 — Physical Layer                                │
│  DAG IR · 3-pass compiler · qubit routing · QASM I/O     │
│  statevector · density matrix · JAX/CuPy acceleration    │
└──────────────────────────────────────────────────────────┘
```

## Features

### Core
- **17 built-in gates** — H, X, Y, Z, CX, CCX, SWAP, RX, RY, RZ, S, T, and more
- **`custom_gate(name, matrix)`** — define your own unitary gates
- **`@circuit` decorator** — write quantum circuits as Python functions
- **`sweep(circuit, params)`** — parameter scans for variational algorithms

### Compiler & IR
- **DAG-based intermediate representation** — directed acyclic graph for circuit analysis
- **3-pass optimizer** — gate cancellation, gate merging, basis translation
- **Topology-aware routing** — SWAP insertion for linear, ring, and grid topologies
- **QASM 2.0/3.0** — import external circuits and export for cross-SDK interop

### Simulators
- **Statevector** — tensor contraction engine, up to **27 qubits** (100s, 2GB)
- **Density matrix** — mixed states + Kraus noise channels, up to 13 qubits
- **Accelerated backend** — auto-detects JAX-GPU / CuPy; falls back to NumPy on CPU

### Error Correction
- Bit-flip, Phase-flip, **Steane [[7,1,3]]** codes
- **Surface code [[d²,1,d]]** — logical qubits with configurable distance, threshold ~1%

### Security
- **BB84 QKD** — quantum key distribution with eavesdropper detection ([Example →](#08-qkd-bb84))

## Algorithms

| Algorithm | Module | Use Case | Example |
|-----------|--------|----------|---------|
| **Grover** | `layer3.search` | Unstructured search (√N speedup) | [05 →](#05-grover-search) |
| **QAOA** | `layer3.optimize` | Combinatorial optimization | [07 →](#07-portfolio-optimization) |
| **VQE** | `layer3.vqe` | Molecular ground-state energy | [06 →](#06-molecular-energy) |
| **Shor** | `layer3.shor` | Integer factoring (RSA) | [10 →](#10-quantum-benchmark) |
| **QSVM** | `layer3.qsvm` | Quantum kernel classification | [10 →](#10-quantum-benchmark) |
| **Multi-Agent** | `layer3.agent` | Quantum agent-based modeling | [09 →](#09-full-demo) |
| **Portfolio** | `layer3.finance` | Financial portfolio optimization | [07 →](#07-portfolio-optimization) |
| **Hamiltonian** | `layer3.hamiltonian` | Molecular simulation (H₂, LiH, HeH⁺) | [06 →](#06-molecular-energy) |
| **Entity Resolution** | `layer3.entity_resolution` | Customer deduplication (QAOA) | [11 →](#11-entity-resolution) |

## Examples & Use Cases

Run any example with `python -m quanta.examples.<name>`:

### 01 Bell State
EPR pair — the simplest entanglement demonstration.
```bash
python -m quanta.examples.01_bell_state
```

### 02 GHZ State
Multi-qubit entanglement: all-or-nothing correlations.
```bash
python -m quanta.examples.02_ghz_state
```

### 03 Quantum Teleportation
Transfer an unknown quantum state using entanglement + classical bits.
```bash
python -m quanta.examples.03_teleportation
```

### 04 Deutsch-Jozsa
Determine if a function is constant or balanced in one query.
```bash
python -m quanta.examples.04_deutsch_jozsa
```

### 05 Grover Search
Quadratic speedup for unstructured search — finds target with 99.9% probability.
```bash
python -m quanta.examples.05_grover
```

### 06 Molecular Energy
H₂ and HeH⁺ ground state via VQE + Hamiltonian time evolution.
```bash
python -m quanta.examples.06_molecule_energy
```

### 07 Portfolio Optimization
Quantum-optimized stock portfolios — tech vs crypto, conservative vs aggressive.
```bash
python -m quanta.examples.07_portfolio_optimization
```

### 08 QKD BB84
Quantum key distribution — detect eavesdroppers via ~25% error rate.
```bash
python -m quanta.examples.08_qkd_bb84
```

### 09 Full Demo
All SDK features in one script — circuits, custom gates, VQE, Grover, noise, routing, QASM.
```bash
python -m quanta.examples.09_full_demo
```

### 10 Quantum Benchmark
8-test quality litmus test — Bell fidelity, CHSH, teleportation, Grover, VQE, Shor, QSVM, surface code.
```bash
python -m quanta.examples.10_quantum_benchmark
```

### 11 Entity Resolution
**Real-world use case:** OTA customer deduplication with QAOA vs classical greedy.  
25 records, 8 columns, Turkish name handling, 3-layer blocking pipeline.  
**Result:** QAOA 86% accuracy vs Greedy 64%.
```bash
python -m quanta.examples.11_entity_resolution
```

## Quality Benchmark

### Turnusol Test — 8/8 🏆

| # | Test | Result | Metric |
|---|------|--------|--------|
| 1 | Bell State Fidelity | ✅ | F = 1.0000 |
| 2 | CHSH Inequality | ✅ | S = 2.8284 (Tsirelson bound) |
| 3 | Quantum Teleportation | ✅ | Unitarity preserved |
| 4 | Grover Amplification | ✅ | 99.9% target probability |
| 5 | VQE Convergence (H₂) | ✅ | 0.000054 Ha error |
| 6 | Shor Factoring | ✅ | 15 = 3 × 5 |
| 7 | QSVM Classification | ✅ | 100% accuracy |
| 8 | Surface Code QEC | ✅ | 0% logical error rate |

### QASMBench — 10/10

All standard QASMBench circuits import, compile, and simulate correctly:  
`bell` · `ghz` · `qft` · `teleportation` · `deutsch-jozsa` · `grover` · `adder` · `vqe_ansatz` · `swap_test` · `random`

Large circuit support: GHZ-20 (710ms), QFT-20 (3.5s), Random-24 (12s).

### Benchpress Compatible

Includes `QuantaBenchpressBackend` adapter for cross-SDK benchmarking alongside Qiskit, Cirq, and Braket using the [Benchpress](https://arxiv.org/abs/2406.14155) framework.

## Installation

```bash
# Clone and install
git clone https://github.com/ONMARTECH/quanta-sdk.git
cd quanta-sdk
pip install -e ".[dev]"

# Run tests
pytest

# Run benchmark
python -m quanta.examples.10_quantum_benchmark
```

**Optional GPU acceleration:**
```bash
pip install jax jaxlib   # JAX GPU backend
pip install cupy          # NVIDIA CUDA backend
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture (EN)](docs/ARCHITECTURE_EN.md) | System design, DAG IR, compiler pipeline |
| [Architecture (TR)](docs/ARCHITECTURE_TR.md) | Türkçe mimari dokümanı |
| [Features (EN)](docs/FEATURES_EN.md) | Complete feature list |
| [Comparison (EN)](docs/COMPARISON_EN.md) | vs Qiskit, Cirq, Braket |
| [CHANGELOG](CHANGELOG.md) | Version history |

## Project Stats

```
Files:       86          Languages:   Python
Lines:       11,770      Tests:       150+
Algorithms:  9           Examples:    11
Simulators:  3           QEC Codes:   4
QASM:        2.0 + 3.0   Max Qubits:  27
```

## Author

**Abdullah Enes SARI** — [ONMARTECH](https://onmartech.com)

**info@onmartech.com**

## Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check [issues page](https://github.com/ONMARTECH/quanta-sdk/issues).

## License

[Apache License 2.0](LICENSE) — same license used by [Google Cirq](https://github.com/quantumlib/Cirq) and [IBM Qiskit](https://github.com/Qiskit/qiskit).

---

<p align="center">
  <sub>Built for the quantum computing community</sub>
</p>
