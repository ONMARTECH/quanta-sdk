<p align="center">
  <h1 align="center">Quanta SDK</h1>
  <p align="center">
    <strong>AI-native quantum computing SDK for Python</strong><br>
    <em>The quantum runtime built for AI agents, researchers, and production workloads</em>
  </p>
  <p align="center">
    <a href="https://github.com/ONMARTECH/quanta-sdk/actions/workflows/tests.yml"><img src="https://github.com/ONMARTECH/quanta-sdk/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
    <a href="#quality-benchmark"><img src="https://img.shields.io/badge/coverage-88%25-brightgreen.svg" alt="Coverage"></a>
    <a href="https://pypi.org/project/quanta-sdk/"><img src="https://img.shields.io/badge/version-0.8.1-blue.svg" alt="Version"></a>
    <a href="https://pypi.org/project/quanta-sdk/"><img src="https://img.shields.io/pypi/v/quanta-sdk.svg" alt="PyPI"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-brightgreen.svg" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-orange.svg" alt="License"></a>
    <a href="#quality-benchmark"><img src="https://img.shields.io/badge/tests-586%20passed-success.svg" alt="Tests"></a>
    <a href="#quality-benchmark"><img src="https://img.shields.io/badge/benchmark-8%2F8-gold.svg" alt="Benchmark"></a>
    <a href="#ibm-quantum-integration"><img src="https://img.shields.io/badge/IBM%20Quantum-Heron%20r3-purple.svg" alt="IBM"></a>
    <a href="#mcp-ai-integration"><img src="https://img.shields.io/badge/MCP-16%20tools-teal.svg" alt="MCP"></a>
  </p>
</p>

---

Quanta is an **AI-native quantum computing SDK** — designed to be called by AI agents (via [MCP](https://modelcontextprotocol.io)), used by researchers, and deployed in production. It provides a 3-layer abstraction — from high-level declarative APIs (`search()`, `factor()`) to low-level DAG manipulation and QASM export — with **16 MCP tools** that let Claude, GPT, and other AI assistants run quantum computations directly.

### 🚀 What's New in v0.8.1

- **IBM Quantum Integration** — Run circuits on real quantum hardware (ibm_torino 156q, ibm_fez 156q) via direct REST API. No Qiskit needed.
- **25 Quantum Gates** — Full IBM parity: I, SDG, TDG, P, SX, SXdg, U(θ,φ,λ), RXX, RZZ, RCCX, RC3X
- **ISA Transpilation** — Automatic decomposition to Heron native gates (rz/sx/x/cz)
- **SVG Circuit Visualization** — Publication-quality HTML/SVG diagrams with color-coded gates
- **MCP Server** — 16 AI tools + 4 guided prompts for Claude, GPT, and other AI assistants
- **Quantum Machine Learning** — Variational classifier, quantum kernel, 3 feature maps
- **Multi-Backend** — IBM Quantum + IonQ + Google Quantum + local simulator
- **Quantum Monte Carlo** — Amplitude estimation for option pricing
- **Quantum Clustering** — Swap-test based distance computation + k-means

## Table of Contents

- [Quick Start](#quick-start)
- [IBM Quantum Integration](#ibm-quantum-integration)
- [MCP AI Integration](#mcp-ai-integration)
- [Architecture](#architecture)
- [Features](#features)
- [Algorithms](#algorithms)
- [Qubit Limits](#qubit-limits)
- [Examples & Use Cases](#examples--use-cases)
- [Quality Benchmark](#quality-benchmark)
- [Installation](#installation)
- [Documentation](#documentation)
- [Author](#author)

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

## IBM Quantum Integration

Run circuits on **real IBM quantum computers** — up to 156 qubits on Heron r3 processors. No Qiskit installation required.

```python
from quanta.backends.ibm_rest import IBMRestBackend

# Connect to IBM Quantum (set IBM_API_KEY and IBM_INSTANCE_CRN env vars)
backend = IBMRestBackend(region="us", backend_name="ibm_torino")

# List available backends
backends = backend.list_backends()
# ibm_fez: 156 qubits (Heron r2), ibm_torino: 156 qubits (Heron r3)

# Submit a Bell state to real hardware
result = backend.run(bell, shots=4096)
# Real quantum noise: |00⟩=47.5%, |11⟩=39.5%, fidelity=87%
```

### Available IBM Backends

| Backend | Qubits | Processor | 2Q Error | Use Case |
|---------|--------|-----------|----------|----------|
| **ibm_torino** | 156 | Heron r3 | 0.25% | General purpose |
| **ibm_fez** | 156 | Heron r2 | 0.28% | Large circuits |
| **ibm_marrakesh** | 156 | Heron r2 | 0.23% | Low error |

### ISA Transpilation

All circuits automatically transpile to Heron's native gate set:

| Your Gate | → ISA Decomposition |
|-----------|-------------------|
| H | rz(π/2) · sx · rz(π/2) |
| CX | H(target) · CZ · H(target) |
| RX, RY | rz + sx combinations |
| CZ, RZ, SX, X | Native (no change) |

### Free Tier Limits (Open Plan)

| Resource | Limit |
|----------|-------|
| QPU time | 10 min/month |
| Qubits | Up to 156 (Heron r3) |
| Shots | Up to 100,000 per job |
| Sessions | Supported |

## MCP AI Integration

Quanta exposes **16 MCP tools** for AI assistants (Claude, GPT, etc.):

```bash
# Install as MCP server
fastmcp install quanta/mcp_server.py --name "Quanta Quantum SDK"
```

### Available Tools

| Tool | Description |
|------|-------------|
| `run_circuit` | Execute quantum circuit code |
| `create_bell_state` | Quick Bell state |Φ+⟩ |
| `grover_search` | Grover's search algorithm |
| `shor_factor` | Shor's factoring algorithm |
| `simulate_noise` | Run with noise model |
| `draw_circuit` | **SVG circuit diagram** |
| `list_gates` | All 25 quantum gates |
| `explain_result` | Interpret measurements |
| `monte_carlo_price` | Quantum option pricing |
| `qaoa_optimize` | QAOA optimization |
| `cluster_data` | Quantum clustering |
| `run_on_ibm` | Run on IBM hardware |
| `ibm_backends` | List IBM quantum computers |
| `ibm_job_result` | Poll job status & fetch results |
| `surface_code_simulate` | Surface code QEC simulation |
| `compare_decoders` | Compare MWPM vs Union-Find decoders |

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Layer 3 — Declarative API                               │
│  search() · optimize() · vqe() · factor() · qsvm()      │
│  monte_carlo() · cluster() · resolve()                   │
├──────────────────────────────────────────────────────────┤
│  Layer 2 — Circuit API                                   │
│  @circuit · 25 gates · measure · run · sweep             │
│  SVG visualization · QASM 3.0 export                     │
├──────────────────────────────────────────────────────────┤
│  Layer 1 — Physical Layer                                │
│  DAG IR · 6-pass compiler · qubit routing · ISA transpile│
│  statevector · density matrix · Pauli frame · IBM REST   │
└──────────────────────────────────────────────────────────┘
```

## Features

### 25 Quantum Gates (Full IBM Parity)

| Category | Gates |
|----------|-------|
| **Pauli** | X, Y, Z |
| **Hadamard** | H |
| **Phase** | S, T, SDG (S†), TDG (T†), P(θ) |
| **Root** | SX (√X), SXdg (√X†) — Heron native |
| **Rotation** | RX(θ), RY(θ), RZ(θ) |
| **Universal** | U(θ, φ, λ) |
| **2-Qubit** | CX, CY, CZ, SWAP, RXX(θ), RZZ(θ) |
| **Multi** | CCX (Toffoli), RCCX, RC3X |
| **Other** | I (Identity), Measure |

### Compiler & IR
- **DAG-based intermediate representation** — directed acyclic graph for circuit analysis
- **6-pass compiler** — gate cancellation, merging, basis translation, routing
- **Topology-aware routing** — SWAP insertion for linear, ring, and grid topologies
- **QASM 3.0** — export for IBM and cross-SDK interop

### Circuit Visualization
- **ASCII** — `draw(circuit)` → terminal-friendly text diagrams
- **SVG/HTML** — `to_html(circuit)` → publication-quality visual diagrams
  - Color-coded gates by category
  - Control dots, target circles, measurement meters
  - Responsive layout with legend

### Noise Simulation
7 channels integrated into `run()`:
```python
result = run(circ, noise=NoiseModel().add(Depolarizing(0.01)))
```
Depolarizing · BitFlip · PhaseFlip · AmplitudeDamping · T2Relaxation · Crosstalk · ReadoutError

### Error Correction
- Bit-flip [[3,1,3]], Phase-flip [[3,1,3]], **Steane [[7,1,3]]** codes
- **Surface code [[d²,1,d]]** — stabilizer-based, threshold ~1%
- **Color code** — triangular lattice, transversal Clifford gates
- **Decoders** — MWPM (greedy) + Union-Find (near-linear O(n·α(n)))

## Algorithms

| Algorithm | Module | Use Case |
|-----------|--------|----------|
| **Grover** | `layer3.search` | Unstructured search (√N speedup) |
| **QAOA** | `layer3.optimize` | Combinatorial optimization |
| **VQE** | `layer3.vqe` | Molecular ground-state energy |
| **Shor** | `layer3.shor` | Integer factoring (RSA) |
| **QSVM** | `layer3.qsvm` | Quantum kernel classification |
| **Monte Carlo** | `layer3.monte_carlo` | Amplitude estimation + pricing |
| **Clustering** | `layer3.clustering` | Swap-test quantum distances |
| **QML Classifier** | `layer3.qml` | Variational quantum classification |
| **Entity Resolution** | `layer3.entity_resolution` | Customer deduplication |
| **Portfolio** | `layer3.finance` | Financial optimization |

## Qubit Limits

| Simulator | Max Qubits | Memory | Speed |
|-----------|-----------|--------|-------|
| **Statevector** | 27 | ~2 GB | Full state simulation |
| **Density Matrix** | 13 | ~1 GB | Mixed states + noise |
| **Pauli Frame** | 1,000+ | O(n²) | Clifford-only circuits |
| **IBM Heron** | 156 | Cloud | Real quantum hardware |
| **IonQ Forte** | 36 | Cloud | Trapped-ion QPU |
| **Google Sycamore** | 72 | Cloud | Superconducting QPU |

> **Note:** Statevector memory doubles per qubit (2^n). 20 qubits = 8 MB, 25 = 256 MB, 27 = 1 GB.

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

### IBM Hardware Validation

Bell state on **ibm_torino** (156 qubits, Heron r3):
```
4096 shots: |00⟩=47.5%, |11⟩=39.5%, entanglement fidelity=87%
```

### QASMBench — 10/10

All standard QASMBench circuits import, compile, and simulate correctly.

## Installation

```bash
# From PyPI (recommended)
pip install quanta-sdk

# From source (development)
git clone https://github.com/ONMARTECH/quanta-sdk.git
cd quanta-sdk
pip install -e ".[dev]"

# Run tests
pytest

# Run benchmark
python -m quanta.examples.10_quantum_benchmark
```

**Optional IBM Quantum:**
```bash
export IBM_API_KEY="your-api-key"
export IBM_INSTANCE_CRN="your-crn"
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
Version:     0.8.1        Gates:       25 (full IBM parity)
Files:       73+          Tests:       530
Algorithms:  10           Examples:    11
Simulators:  4            QEC Codes:   6
MCP Tools:   14           Max Qubits:  156 (IBM Heron r3)
Noise:       7 channels   Backends:    IBM + IonQ + Google + local
QASM:        3.0          Decoders:    2 (MWPM + UF)
```

## Author

**Abdullah Enes SARI** — [ONMARTECH](https://onmartech.com)

**info@onmartech.com**

## Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check [issues page](https://github.com/ONMARTECH/quanta-sdk/issues).

## License

[Apache License 2.0](LICENSE)

---

<p align="center">
  <sub>Built for the quantum computing community — now running on real IBM quantum hardware</sub>
</p>
