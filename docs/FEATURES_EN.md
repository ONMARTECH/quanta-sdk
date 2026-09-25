# Quanta SDK — Features

## Gate Set (31 Gates)

| Gate | Qubits | Description |
|------|--------|-------------|
| H | 1 | Hadamard — creates superposition |
| X | 1 | Pauli-X — bit flip (NOT) |
| Y | 1 | Pauli-Y — bit + phase flip |
| Z | 1 | Pauli-Z — phase flip |
| S | 1 | S gate — π/2 phase |
| T | 1 | T gate — π/4 phase |
| CX | 2 | CNOT — controlled NOT |
| CZ | 2 | Controlled-Z — controlled phase |
| CY | 2 | Controlled-Y |
| SWAP | 2 | Qubit exchange |
| CCX | 3 | Toffoli — double controlled NOT |
| RX(θ) | 1 | X-axis rotation |
| RY(θ) | 1 | Y-axis rotation |
| RZ(θ) | 1 | Z-axis rotation |
| P(θ) | 1 | Phase gate |
| U(θ,φ,λ) | 1 | General single-qubit unitary |
| I | 1 | Identity |
| SDG | 1 | S-dagger (−π/2 phase) |
| TDG | 1 | T-dagger (−π/4 phase) |
| SX | 1 | Square root of X |
| SXdg | 1 | SX-dagger |
| RXX(θ) | 2 | XX rotation (2-qubit) |
| RZZ(θ) | 2 | ZZ rotation (2-qubit) |
| RCCX | 3 | Relative-phase CCX |
| RC3X | 4 | Relative-phase C3X |
| ECR | 2 | Echoed cross-resonance (IBM Heron native) |
| iSWAP | 2 | Imaginary SWAP (Google Sycamore native) |
| CSWAP | 3 | Controlled-SWAP (Fredkin) |
| CH | 2 | Controlled-Hadamard |
| CP(θ) | 2 | Controlled-Phase |
| MS(θ) | 2 | Mølmer-Sørensen (IonQ trapped-ion native) |

### Custom Gates

```python
from quanta import custom_gate
import numpy as np

# Define a custom √X gate
custom_gate("SqrtX", np.array([[0.5+0.5j, 0.5-0.5j],
                                [0.5-0.5j, 0.5+0.5j]]))
```

### Broadcast Support

```python
H(q)        # Apply H to all qubits
H(q[0])     # Apply only to q[0]
CX(q[0], q[1])  # Two-qubit gate
```

## Compiler Optimizations

| Pass | What It Does | Example |
|------|-------------|---------|
| CancelInverses | Cancels inverse gates | H·H → (empty), X·X → (empty) |
| MergeRotations | Merges rotations | RZ(π/4)·RZ(π/4) → RZ(π/2) |
| TranslateToTarget | Converts to target hardware gate set | SWAP → 3×CX |

### Qubit Routing

Topology-aware SWAP insertion for hardware constraints:

| Topology | Use Case |
|----------|----------|
| Linear | Ion trap, superconducting chains |
| Ring | Circular connectivity |
| Grid | 2D superconducting (IBM, Google) |

## Supported Hardware Gate Sets

| Hardware | Gate Set |
|----------|----------|
| IBM Heron | {CX, RZ, SX, X} |
| Google Sycamore | {CZ, RZ, RX, RY} |
| Quantinuum H-Series | {CX, RZ, RY, RX} |

## Simulators

| Simulator | Max Qubits | Features |
|-----------|-----------|----------|
| Statevector | 27 | Tensor contraction, O(2^n) |
| Pauli Frame | 50 | Stabilizer tableau (Aaronson-Gottesman), O(n) per gate |
| Density Matrix | 13 | Mixed states, Kraus channels |
| Accelerated | 27 | Auto-detects JAX-GPU / CuPy |

### Noise Integration

Noise is a first-class citizen in the execution pipeline:

```python
from quanta import run
from quanta.simulator.noise import NoiseModel, Depolarizing

result = run(bell, shots=1024, noise=NoiseModel().add(Depolarizing(0.01)))
```

## Noise Models

| Channel | Description | Parameter | Hardware Ref |
|---------|-------------|-----------|-------------|
| Depolarizing | Random Pauli error | p ∈ [0,1] | — |
| BitFlip | |0⟩↔|1⟩ flip | p ∈ [0,1] | — |
| PhaseFlip | Phase error (Z) | p ∈ [0,1] | — |
| AmplitudeDamping | Energy loss (T1 decay) | γ ∈ [0,1] | IBM: 100-300μs |
| T2Relaxation | Pure dephasing (T2 decay) | γ ∈ [0,1] | IBM: 100-200μs |
| Crosstalk | ZZ coupling between neighbors | p ∈ [0,1] | ~0.1-1% / gate |
| ReadoutError | Measurement bit-flip | p01, p10 | IBM: 0.5-2% |

## 2026 Dual-Track Fault-Tolerance Engine (FTQC)

Quanta SDK implements a production-grade dual-track quantum error correction (QEC) architecture aligned with 2026 fault-tolerant standards:

### Track A: 2D Topological Surface Codes & Edmonds Blossom MWPM
- **Rotated Surface Codes ($[[d^2, 1, d]]$)**: Stabilizer syndrome extraction across distances $d \in \{3, 5, 7\}$.
- **Edmonds Blossom MWPM Decoder**: Full minimum-weight perfect matching replacing greedy heuristics, achieving theoretical threshold performance ($p_{\text{th}} \approx 1\%$).
- **Google Willow-Compliant 3D Spacetime Syndromes**: Space-time syndrome graphs tracking measurement and circuit errors across fault cycles.
- **Color Codes ($[[n, 1, d]]$)**: Transversal Clifford gates with 2D triangular restriction decoder.
- **Standard Codes**: Steane $[[7,1,3]]$, 3-qubit Bit-Flip and Phase-Flip codes.

### Track B: High-Rate qLDPC Codes & Native BP-OSD
- **Canonical Gross $[[144, 12, 12]]$ Bivariate Bicycle Code**: Yielding 12 logical qubits in 144 physical qubits, achieving a **$12\times$ qubit footprint reduction** over equivalent planar surface codes.
- **Native Normalized Min-Sum BP-OSD-0 Decoder**: Combining belief propagation with order-0 ordered statistics decoding for sub-millisecond ($1.54\text{ ms}$) syndrome extraction.
- **Non-Clifford Universality**:
  - **15-to-1 Bravyi-Kitaev Magic State Distillation**: High-fidelity logical $|T\rangle_L$ factory with cubic suppression ($\epsilon_{\text{out}} \le 35 p^3$).
  - **Planar Lattice Surgery**: Logical CNOT synthesis and patch merging/splitting protocols.

| Code / Architecture | Notation | Qubit Savings | Decoder |
|---------------------|----------|---------------|---------|
| Gross Bivariate Bicycle | $[[144, 12, 12]]$ | **12× reduction** (12 logical qubits) | Normalized Min-Sum BP-OSD-0 |
| Rotated Surface Code | $[[d^2, 1, d]]$ | 2D Baseline | Edmonds Blossom MWPM |
| Color Code | $[[n, 1, d]]$ | Transversal Clifford | Restriction Decoder |
| Steane Code | $[[7, 1, 3]]$ | Analytic Benchmark | Lookup / Syndrome |
| 15-to-1 BK Distillation | $|T\rangle$ Factory | $\epsilon_{\text{out}} \le 35 p^3$ | Parity Projection |

## Algorithms (Layer 3)

| Algorithm | Function | Description |
|-----------|----------|-------------|
| Grover | `search()` | Unstructured search with quadratic speedup |
| QAOA | `optimize()` | Combinatorial optimization |
| VQE | `vqe()` | Variational eigensolver for molecular energy |
| Shor | `factor()` | Integer factoring via period finding |
| QSVM | `qsvm_classify()` | Quantum kernel SVM classification |
| Portfolio | `portfolio_optimize()` | Financial portfolio optimization |
| Hamiltonian | `evolve()` | Trotterized time evolution |
| Entity Resolution | `resolve()` | QAOA-based customer deduplication |
| Multi-Agent | `MultiAgentSystem` | Quantum decision modeling |
| Monte Carlo | `monte_carlo_price()` | Amplitude estimation for option pricing |
| Clustering | `cluster_data()` | Swap-test quantum distance + k-means |
| QML Classifier | `QuantumClassifier` | Variational quantum classification |

## QASM Support

| Direction | Version | Description |
|-----------|---------|-------------|
| Export | QASM 3.0 | Circuit → OpenQASM string |
| Import | QASM 2.0/3.0 | OpenQASM string → DAG |

## Benchmark Infrastructure

| Tool | Description |
|------|-------------|
| QASMBench | 10 standard + 3 large (20-24 qubit) circuits |
| Benchpress Adapter | Cross-SDK benchmarking API |
| Turnusol Test | 8-test quality litmus test |

## Parameter Sweep

```python
from quanta import sweep

results = sweep(my_circuit, params={"theta": [0, 0.5, 1.0, 1.5]})
for r in results:
    print(r.summary())
```

## Visualization

- Probability histogram: `print(result)`
- Dirac notation: `result.dirac_notation()`
- Statevector display: `show_statevector(sv, n)`

## MCP Server (AI Integration — 23 Tools)

Quanta SDK includes **23 MCP (Model Context Protocol) tools** for both local and cloud agents. Assistants like Claude, Gemini, and GPT can compose quantum circuits, estimate fault-tolerant costs, evaluate noise, and solve optimization problems autonomously.

| Category | Tools |
|----------|-------|
| Circuit Operations | `run_circuit`, `create_bell_state`, `list_gates`, `inspect_circuit` |
| Quantum Algorithms | `grover_search`, `shor_factor`, `vqe_ground_state`, `quantum_clustering` |
| Fault Tolerance & Noise | `simulate_noise` (7 channels), `estimate_fault_tolerant_cost` (Google Willow / IBM Starling) |
| Optimization & Math | `qubo_solve`, `portfolio_optimization`, `entity_resolution_match` |

---

## PyTorch & Biomorphic Quantum Engine (v1.2.0-production — `quanta.torch`)

### 1. Differentiable `QuantumLayer`
- Full `torch.nn.Module` integration.
- Analytical **Parameter-Shift Rule** for exact vector-Jacobian products (VJPs) and autograd gradients.
- Hardware-efficient parameterized ansatz templates: `hardware_efficient`, `strong_entangling`, `reuploading`, `real_amplitudes`.
- Full device portability across CPU, CUDA, and Apple Silicon MPS (Metal).

### 2. Continuous Quantum Resonance (`ContinuousResonator`)
- Continuous-time Hamiltonian evolution: $U(t) = e^{-i H(x, \theta) t}$.
- **Daleckii-Krein Closed-Form Fréchet Matrix Exponential Derivatives**: Eliminating Padé truncation errors ($>1.3\times 10^{-6}$) with exact machine precision ($9.99\times 10^{-16}$).
- **Dynamical Lie Algebra & Barren Plateau Diagnosis**: Pre-execution diagnosis via $\mathfrak{g} = \langle i H_k \rangle_{\text{Lie}}$ dimension and Casimir operator analysis.
- **Ehrenfest theorem time derivatives** and Lindblad phase-damping dissipators for open quantum neural systems.

### 3. Biomorphic Resonant Brain (`BiomorphicResonantBrain`)
- **Dual-Hemisphere Architecture**: Resonance coupling between Left (analytical/logical) and Right (intuitive/pattern) hemispheres.
- **4-Neuromodulator Dynamics**:
  - *Dopamine ($DA$)*: Value weighting and reward-driven plasticity.
  - *Acetylcholine ($ACh$)*: Attention gating and learning rate scaling.
  - *Serotonin ($5\text{-}HT$)*: Impulsivity dampening and patience regulation.
  - *Norepinephrine ($NE$)*: Arousal and novelty-triggered exploratory dynamics.
- **Cerebral Oxygenation ($sO_2$)**: Metabolic budget and energy-constrained computation.
- **REM Sleep Continual Learning**: Active memory consolidation preventing catastrophic forgetting.
- **NoisyHippocampalBuffer**: Lindblad phase diffusion and Sharp-Wave Ripple (SWR) replay buffer.
- **DialecticalSynthesizer**: Quantum Thesis-Antithesis conflict resolution engine.
- **CSFBiophysicalShield**: Biophysical quantum phase shielding suppressing environmental decoherence.

---

## Apple Silicon Metal / MLX Acceleration (v1.2.0)

- **Zero-Copy Unified Memory Acceleration**: Dedicated Metal/MLX pipeline eliminating CPU-GPU memory copying, achieving up to **52.09× peak speedup**.
- **SIMD-Vectorized Clifford Engine**: Aaronson-Gottesman binary tableau simulator executing over **3.13 million gates/second**.
- **Matrix Product States (MPS)**: Low-entanglement 250-qubit GHZ state preparation in 3.42 milliseconds.
- **Large-Scale Statevector Contraction**: Dense statevector contraction up to 27+ qubits on Unified Memory architectures.

---

## Multi-Cloud Quantum Hardware Validation

- **IonQ Cloud**: REST API v0.3 protocol, 29-qubit hardware emulation, and remote telemetry.
- **IBM Quantum**: 156-qubit Heron r3 processors (`ibm_torino`, `ibm_fez`), native ISA transpilation, and IAM authentication.
- **Google Cirq**: Sycamore native gate decomposition and Colab GPU environment support.

---

## Empirical Verification & Certified Rigor

- **2,076 Passing Automated Tests**: 100% pass rate with zero mock or synthetic oracle calls.
- **Exact Unitarity**: Verified $\|U^\dagger U - I\| < 10^{-14}$ across all gate sets.
- **CPTP Trace Preservation**: Open quantum Lindblad evolution strictly bounded by $|\text{Tr}(\rho) - 1.0| < 10^{-12}$.

---

## Deployment Options

| Target | Method | Use Case |
|--------|--------|----------|
| Local | `pip install quanta-sdk` | Fast development & testing |
| PyTorch / AI | `pip install "quanta-sdk[torch]"` | Deep learning & hybrid QNNs |
| Apple Metal | `pip install "quanta-sdk[metal]"` | Apple Silicon GPU acceleration |
| Claude / Gemini / GPT | MCP Server Integration (`fastmcp`) | 23-tool autonomous AI agents |
| Cloud Run / Docker | Dockerfile.mcp + SSE | Always-on remote quantum microservice |
