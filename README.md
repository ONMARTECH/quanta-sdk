<p align="center">
  <h1 align="center">ONMARTECH QUANTA QUANTUM SDK</h1>
  <p align="center">
    <strong>AI-native quantum computing SDK for Python</strong><br>
    <em>The quantum runtime built for AI agents, researchers, and production workloads</em>
  </p>
  <p align="center">
    <a href="https://github.com/ONMARTECH/quanta-sdk/actions/workflows/tests.yml"><img src="https://github.com/ONMARTECH/quanta-sdk/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
    <a href="#quality-benchmark"><img src="https://img.shields.io/badge/coverage-92%25-brightgreen.svg" alt="Coverage"></a>
    <a href="https://pypi.org/project/quanta-sdk/"><img src="https://img.shields.io/badge/version-1.2.0-blue.svg" alt="Version"></a>
    <a href="https://doi.org/10.5281/zenodo.22952779"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.22952779.svg" alt="DOI"></a>
    <a href="https://pypi.org/project/quanta-sdk/"><img src="https://img.shields.io/pypi/v/quanta-sdk.svg" alt="PyPI"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-brightgreen.svg" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-orange.svg" alt="License"></a>
    <a href="#quality-benchmark"><img src="https://img.shields.io/badge/tests-2076%20passed-success.svg" alt="Tests"></a>
    <a href="#pytorch-engine"><img src="https://img.shields.io/badge/PyTorch-QuantumLayer-EE4C2C.svg" alt="PyTorch"></a>
    <a href="#apple-silicon"><img src="https://img.shields.io/badge/Apple%20Silicon-Metal%2052x-000000.svg" alt="Metal"></a>
    <a href="#ibm-quantum-integration"><img src="https://img.shields.io/badge/IBM%20Quantum-Heron%20r3-purple.svg" alt="IBM"></a>
    <a href="#mcp-ai-integration"><img src="https://img.shields.io/badge/MCP-23%20tools-teal.svg" alt="MCP"></a>
    <a href="https://quanta.onmartech.com/"><img src="https://img.shields.io/badge/docs-live-blue.svg" alt="Docs"></a>
    <a href="#features"><img src="https://img.shields.io/badge/gates-31-blueviolet.svg" alt="Gates"></a>
    <a href="https://quanta.onmartech.com/tutorials/01-getting-started/"><img src="https://img.shields.io/badge/tutorials-16-informational.svg" alt="Tutorials"></a>
  </p>
</p>

---

Quanta is an **AI-native quantum computing SDK** — designed to be called by AI agents (via [MCP](https://modelcontextprotocol.io)), used by researchers, and deployed in production. It provides deep learning quantum layers (`quanta.torch`), continuous quantum neural resonance, native Apple Silicon Metal acceleration, 2026 Dual-Track FTQC (surface codes + Gross qLDPC), multi-cloud hardware execution (IBM Quantum, IonQ, Google Cirq), and **23 MCP tools** that let AI assistants directly run quantum workloads.

### 🚀 What's New in v1.2.0 (September 2026)

- **2026 Dual-Track Fault-Tolerance Engine (FTQC)**:
  - **Track A (2D Topological)**: Full Edmonds Blossom MWPM decoding with Google Willow-compliant 3D spacetime syndrome extraction.
  - **Track B (High-Rate qLDPC)**: Canonical Gross $[[144, 12, 12]]$ Bivariate Bicycle code paired with native Normalized Min-Sum BP-OSD-0 decoder, slashing physical qubit overhead by **$12\times$**.
  - **Non-Clifford Universality**: 15-to-1 Bravyi-Kitaev magic state distillation factory ($\epsilon_{\text{out}} \le 35 p^3$) and planar lattice surgery.
- **Continuous Hilbert Gradients & Daleckii-Krein Autograd (`quanta.torch`)**: Analytical Fréchet matrix exponential derivatives achieving exact machine precision ($9.99\times 10^{-16}$) alongside dynamic Lie algebra ($\dim(\mathfrak{g})$) barren plateau diagnosis.
- **2,076 Passing Automated Tests**: 100% pass rate with zero mock calls and certified mathematical unitarity ($\|U^\dagger U - I\| < 10^{-14}$).
- **Apple Silicon Metal / MLX Zero-Copy GPU**: 52.09× peak speedup on Unified Memory and SIMD Clifford engine executing >3.13M gates/s.

- **PyTorch Native Quantum Engine (`quanta.torch`)** — `QuantumLayer` with analytical parameter-shift autograd, seamless `nn.Module` integration, and Apple Silicon MPS/Metal support.
- **Biomorphic Resonant Brain** — Continuous-time quantum neural dynamics (`ContinuousResonator`), dual-hemisphere architecture, 4 neuromodulators ($DA, ACh, 5\text{-}HT, NE$), cerebral oxygenation ($sO_2$), REM continual learning, `NoisyHippocampalBuffer` with SWR replay, `DialecticalSynthesizer`, and `CSFBiophysicalShield`.
- **World's First Native Apple Silicon Metal/MLX Quantum Engine (v1.0.0)** — Up to **404x speedup** on Apple M5 Pro / M-series chips via Unified Memory tensor contractions (`mlx-metal`).
- **Multi-Cloud Hardware Validation** — Live verified execution on IonQ Cloud REST API v0.3, IBM Quantum Heron r3 (156 qubits), and Google Cirq Sycamore.
- **Agentic MCP Tools (23 Tools)** — Native Model Context Protocol server enabling Claude, Gemini, and GPT to author, simulate, transpile, and optimize quantum circuits.

## Table of Contents

- [Quick Start](#quick-start)
- [PyTorch & Biomorphic Brain](#pytorch-engine)
- [Apple Silicon Metal Acceleration](#apple-silicon)
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

<a id="pytorch-engine"></a>
## PyTorch Native Quantum Engine (`quanta.torch`)

Quanta v1.1.0 introduces end-to-end differentiable quantum-classical hybrid deep learning directly integrated with PyTorch's autograd engine, featuring analytical parameter-shift gradients and biomorphic continuous-time resonance.

### 1. Differentiable `QuantumLayer`

Drop quantum circuits directly into your standard `torch.nn.Sequential` pipelines:

```python
import torch
import torch.nn as nn
from quanta.torch import QuantumLayer

# Parameterized quantum layer with analytical parameter-shift autograd
model = nn.Sequential(
    nn.Linear(4, 4),
    QuantumLayer(n_qubits=4, ansatz="hardware_efficient", n_layers=2),
    nn.Linear(4, 2)
)

x = torch.randn(8, 4, requires_grad=True)
out = model(x)
loss = out.sum()
loss.backward()  # Exact parameter-shift gradients computed automatically
```

### 2. Biomorphic Resonant Brain (`BiomorphicResonantBrain`)

Bio-inspired quantum cognitive architecture featuring continuous resonance, dual-hemisphere dynamics, 4 neurotransmitters ($DA, ACh, 5\text{-}HT, NE$), and REM sleep continual learning:

```python
from quanta.torch import BiomorphicResonantBrain

# Initialize 4-qubit dual-hemisphere biomorphic quantum brain
brain = BiomorphicResonantBrain(n_qubits=4)

# Forward pass through cognitive resonance dynamics
x = torch.randn(1, 4)
output = brain(x)

# Consolidate memories during REM sleep (prevents catastrophic forgetting)
stats = brain.consolidate_rem_sleep(replay_cycles=3)
print(f"Post-REM Retained Fidelity: {stats['retained_fidelity']:.4f}")
```

<a id="apple-silicon"></a>
## Apple Silicon Metal Acceleration (404x Speedup)

Quanta v1.0.0 features the world's first native Apple Silicon Metal/MLX quantum simulator, harnessing Unified Memory for lightning-fast multi-axis tensor contractions on M-series chips:

```python
from quanta import circuit, H, CX, run

@circuit(qubits=26)
def large_circuit(q):
    for i in range(26):
        H(q[i])
    for i in range(25):
        CX(q[i], q[i+1])

# Automatically routes to MLXSimulator on Apple Silicon (0.076s vs 30.7s on CPU)
result = run(large_circuit, backend="mlx")
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

Quanta exposes **23 MCP tools** for AI assistants (Claude, GPT, Gemini, etc.):

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
| `list_gates` | All 31 quantum gates |
| `explain_result` | Interpret measurements |
| `monte_carlo_price` | Quantum option pricing |
| `option_greeks` | Calculate option Greeks (delta, gamma, vega) |
| `qaoa_optimize` | QAOA optimization |
| `optimize_circuit` | Compiler optimization with metrics |
| `cluster_data` | Quantum clustering |
| `qml_classify` | Quantum ML classification |
| `run_on_ibm` | Run on IBM hardware |
| `ibm_backends` | List IBM quantum computers |
| `ibm_job_result` | Poll job status & fetch results |
| `surface_code_simulate` | Surface code QEC simulation |
| `compare_decoders` | Compare MWPM vs Union-Find decoders |
| `qec_diagnose` | Diagnose errors in quantum error-correcting codes |
| `estimate_fault_tolerant_cost` | Estimate physical qubits, code distance & T-factory budget (FTQC) |
| `quanta_reasoning_eval` | Deep reasoning audit & gate density evaluation for AI-generated circuits |
| `transpile_for_target` | Compile circuit to target native gate set (IBM Heron, Google Willow, IonQ Aria) |

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Layer 3 — Declarative API                               │
│  search() · optimize() · vqe() · factor() · qsvm()      │
│  monte_carlo() · cluster() · resolve()                   │
├──────────────────────────────────────────────────────────┤
│  Layer 2 — Circuit API                                   │
│  @circuit · 31 gates · measure · run · sweep             │
│  SVG visualization · QASM 3.0 export                     │
├──────────────────────────────────────────────────────────┤
│  Layer 1 — Physical Layer                                │
│  DAG IR · 6-pass compiler · qubit routing · ISA transpile│
│  statevector · density matrix · Pauli frame · IBM REST   │
└──────────────────────────────────────────────────────────┘
```

## Features

### 31 Quantum Gates (Full IBM Parity + Google/IonQ Native)

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
pip install quanta-sdk[gpu] # Installs cupy and cuquantum-python for NVIDIA GPUs
```

## Documentation

📚 **Live Documentation:** [onmartech.github.io/quanta-sdk](https://onmartech.github.io/quanta-sdk/)

| Document | Description |
|----------|-------------|
| [📖 Tutorials (14)](https://onmartech.github.io/quanta-sdk/tutorials/01-getting-started/) | Step-by-step guides from basics to advanced |
| [📘 API Reference](https://onmartech.github.io/quanta-sdk/api/core/circuit/) | Full API docs with live examples |
| [Architecture (EN)](docs/ARCHITECTURE_EN.md) | System design, DAG IR, compiler pipeline |
| [Architecture (TR)](docs/ARCHITECTURE_TR.md) | Türkçe mimari dokümanı |
| [Features (EN)](docs/FEATURES_EN.md) | Complete feature list |
| [Comparison (EN)](docs/COMPARISON_EN.md) | vs Qiskit, Cirq, Braket |
| [CHANGELOG](CHANGELOG.md) | Version history |

## Project Stats

```
Version:     0.9.3        Gates:       31 (full IBM parity + Google/IonQ)
Files:       88           Tests:       888 (90% coverage)
Algorithms:  10           Examples:    11
Simulators:  6            QEC Codes:   7
MCP Tools:   23           Max Qubits:  200+ (MPS) / 156 (IBM Heron r3)
Noise:       7 channels   Backends:    IBM + IonQ + Google + local
QASM:        3.0          Decoders:    2 (MWPM + UF)
Tutorials:   14           Notebooks:   14 (Colab)
Benchmark:   Bell 0.18ms  Docs Pages:  44
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
  <sub>Built for the quantum computing community — now running on real IBM quantum hardware</sub><br>
  <sub><b>Keywords:</b> quantum computing, quantum SDK, python quantum, MCP server, AI quantum, VQE, QAOA, QML, quantum error correction, surface code, IBM Quantum, Shor algorithm, Bell inequality, quantum machine learning, quantum simulation</sub>
</p>
