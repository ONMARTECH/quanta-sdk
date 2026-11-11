# Quanta SDK — Features

## Gate Set (34 Gates)

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
| CRX(θ) | 2 | Controlled X-axis rotation |
| CRY(θ) | 2 | Controlled Y-axis rotation |
| CRZ(θ) | 2 | Controlled Z-axis rotation |
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

## Compiler Optimizations & Unitary Synthesis

| Pass / Module | What It Does | Example |
|---------------|-------------|---------|
| CancelInverses | Cancels inverse gates | H·H → (empty), X·X → (empty) |
| MergeRotations | Merges rotations | RZ(π/4)·RZ(π/4) → RZ(π/2) |
| TranslateToTarget | Converts to target hardware gate set | SWAP → 3×CX |
| `synthesis.py` | Universal 1Q ZYZ, IBM Heron `RZ-SX-RZ-SX-RZ`, and 2Q Cartan/KAK $SU(4)$ synthesis | Arbitrary $4\times 4$ unitary → $\le 3$ CNOTs |

### Qubit Routing

Topology-aware SWAP insertion with dynamic SWAP layout tracking for hardware constraints:

| Topology | Use Case |
|----------|----------|
| Linear | Ion trap, superconducting chains |
| Ring | Circular connectivity |
| Grid | 2D superconducting (Google Sycamore/Willow) |
| Heavy-Hex (127Q / 156Q) | IBM Eagle / Heron r2 & r3 superconducting processors |

## Supported Hardware Gate Sets

| Hardware | Gate Set |
|----------|----------|
| IBM Heron | {CX / ECR, RZ, SX, X} |
| Google Sycamore | {CZ, RZ, RX, RY} |
| Quantinuum H-Series | {CX, RZ, RY, RX} |

## Simulators

| Simulator | Max Qubits | Features |
|-----------|-----------|----------|
| `StateVectorSimulator` | 27 | Dense tensor contraction, $O(2^n)$ |
| `MLXSimulator` | 30 | Apple Silicon Metal GPU zero-copy Unified Memory |
| `SparseSimulator` | 50 | Dict-based sparse statevector, $O(k)$ active basis states |
| `MPSSimulator` | 200+ | Mixed-canonical SVD Matrix Product State, Born sampling & non-adjacent SWAP routing |
| `PauliFrame` | 50+ | Full $(2n+1)\times(2n+1)$ Aaronson-Gottesman stabilizer/destabilizer tableau with phase vector |
| `DensityMatrixSimulator` | 13 | Mixed states via `np.tensordot` CPTP Kraus channels + exact `p/15` 2Q depolarizing |
| `LindbladMasterEquation` | 12 | Open-system Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) master equation solver |
| `AcceleratedSimulator` | 27 | Auto-detects JAX-GPU / CuPy with NumPy fallback |

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

### Track A: 2D Topological Surface Codes, Edmonds Blossom MWPM & Union-Find
- **Rotated Surface Codes ($[[d^2, 1, d]]$)**: Exact $\mathbb{F}_2$ homology stabilizer syndrome extraction across distances $d \in \{3, 5, 7\}$.
- **Edmonds Blossom MWPM & Delfosse-Nickerson `UnionFindDecoder`**: Full minimum-weight perfect matching alongside near-linear-time Union-Find cluster growth/peeling decoding.
- **Google Willow-Compliant 3D Spacetime Syndromes**: Space-time syndrome graphs tracking measurement and circuit errors across fault cycles.
- **Color Codes ($[[n, 1, d]]$)**: Transversal Clifford gates (`H`, `S`, `CNOT`) with syndrome-only Kubica-Delfosse $A_2$ lattice projection + integer linear programming (ILP) decoder.
- **Standard Codes**: Steane $[[7,1,3]]$, 3-qubit Bit-Flip and Phase-Flip codes.

### Track B: High-Rate qLDPC Codes & Native BP-OSD
- **Canonical Gross $[[144, 12, 12]]$ Bivariate Bicycle Code**: Yielding 12 logical qubits in 144 physical qubits, achieving a **$12\times$ qubit footprint reduction** over equivalent planar surface codes.
- **Native Normalized Min-Sum BP-OSD Decoder**: Combining belief propagation with ordered statistics decoding (`BPOSDDecoder`).
- **Non-Clifford Universality**:
  - **15-to-1 Bravyi-Kitaev Magic State Distillation**: High-fidelity logical $|T\rangle_L$ factory with cubic suppression ($\epsilon_{\text{out}} \le 35 p^3$).
  - **Planar Lattice Surgery**: Logical CNOT synthesis and patch merging/splitting protocols.

| Code / Architecture | Notation | Qubit Savings | Decoder |
|---------------------|----------|---------------|---------|
| Gross Bivariate Bicycle | $[[144, 12, 12]]$ | **12× reduction** (12 logical qubits) | Normalized Min-Sum BP-OSD |
| Rotated Surface Code | $[[d^2, 1, d]]$ | 2D Baseline ($\mathbb{F}_2$ homology) | Edmonds Blossom MWPM & `UnionFindDecoder` |
| Color Code | $[[n, 1, d]]$ | Transversal Clifford | Kubica-Delfosse $A_2$ Lattice + ILP |
| Steane Code | $[[7, 1, 3]]$ | Analytic Benchmark | Lookup / Syndrome |
| 15-to-1 BK Distillation | $|T\rangle$ Factory | $\epsilon_{\text{out}} \le 35 p^3$ | Parity Projection |

## Algorithms (Layer 3)

| Algorithm | Function | Description |
|-----------|----------|-------------|
| Grover | `search()` | Unstructured search with quadratic speedup |
| QAOA | `optimize()` | Combinatorial optimization |
| VQE | `vqe()` | Variational eigensolver for molecular energy |
| Shor | `factor()` | Griffiths-Niu / Beauregard semiclassical QPE integer factoring |
| QSVM | `qsvm_classify()` | Cortes-Vapnik Dual QP quantum kernel SVM classification |
| Portfolio | `portfolio_optimize()` | Markowitz + QAOA financial portfolio optimization |
| Hamiltonian | `evolve()` | Trotterized time evolution |
| Entity Resolution | `resolve()` | QAOA-based customer deduplication |
| Multi-Agent | `MultiAgentSystem` | Quantum decision modeling |
| Monte Carlo & Greeks | `quantum_monte_carlo()`, `compute_greeks()` | Grinko IQAE option pricing + Common Random Numbers (CRN) Black-Scholes Greeks |
| Clustering | `quantum_cluster()` | Buhrman Swap Test quantum distance + k-means |
| QML Classifier | `QuantumClassifier` | Variational quantum classification |

## Analytical Gradients (`quanta.gradients`)

- **Parameter-Shift Rule (`parameter_shift`)**: Exact 2-term shift for 1-qubit Pauli generators and **4-term Anselmetti shift** for 2-qubit entangling gates (`CRX`, `CRY`, `CRZ`, `CP`, `RXX`, `RZZ`, `MS`).
- **Adjoint Differentiation (`adjoint_gradient`)**: Exact **$O(G)$ statevector reverse sweep** independent of parameter count.
- **Quantum Natural Gradient (`natural_gradient`)**: **Mari-Bromley-Killoran gauge-invariant Quantum Fisher Information Matrix (QFIM)** for Fubini-Study preconditioned optimization.

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

## MCP Server (AI Integration — 25 Tools)

Quanta SDK includes **25 MCP (Model Context Protocol) tools** in `quanta/mcp_server.py` for both local and cloud agents. Assistants like Claude, Gemini, and GPT can compose quantum circuits, estimate fault-tolerant costs, evaluate noise, and solve optimization problems autonomously.

| Category | Exact `@mcp.tool` Names |
|----------|-------------------------|
| Circuit, Compilation & Simulation (8) | `run_circuit`, `mps_circuit_simulate`, `create_bell_state`, `list_gates`, `draw_circuit`, `explain_result`, `optimize_circuit`, `transpile_for_target` |
| Algorithms, Optimization & QML (5) | `grover_search`, `shor_factor`, `qaoa_optimize`, `cluster_data`, `qml_classify` |
| Quantitative Finance (3) | `monte_carlo_price`, `option_greeks`, `portfolio_optimize` |
| Noise, QEC & Fault Tolerance (5) | `simulate_noise`, `surface_code_simulate`, `compare_decoders`, `qec_diagnose`, `estimate_fault_tolerant_cost` |
| Cloud Hardware & Cognitive Eval (4) | `run_on_ibm`, `ibm_backends`, `ibm_job_result`, `quanta_reasoning_eval` |

---

## PyTorch & Biomorphic-Inspired Quantum Engine (v1.2.0-production — `quanta.torch`)

### 1. Differentiable `QuantumLayer`
- Full `torch.nn.Module` integration.
- Analytical **Parameter-Shift Rule** (including 4-term Anselmetti 2Q shift) and $O(G)$ `adjoint_gradient` support for exact vector-Jacobian products (VJPs).
- Hardware-efficient parameterized ansatz templates: `hardware_efficient`, `strong_entangling`, `reuploading`, `real_amplitudes`.
- Full device portability across CPU, CUDA, and Apple Silicon MPS (Metal).

### 2. Continuous Quantum Resonance (`ContinuousResonantLayer`)
- Continuous-time Hamiltonian evolution: $U(t) = e^{-i H(x, \theta) t}$.
- **Daleckii-Krein Closed-Form Fréchet Matrix Exponential Derivatives**: Eliminating Padé truncation errors ($>1.3\times 10^{-6}$) with exact machine precision ($9.99\times 10^{-16}$).
- **Dynamical Lie Algebra & Barren Plateau Diagnosis**: Pre-execution diagnosis via $\mathfrak{g} = \langle i H_k \rangle_{\text{Lie}}$ dimension and Casimir operator analysis.
- **Ehrenfest theorem time derivatives** and Lindblad phase-damping dissipators for open quantum neural systems.

### 3. Biomorphic Resonant Brain (`BiomorphicResonantBrain`)
> **Scientific Demarcation**: All modules in `quanta.torch.brain` and `quanta.cognitive` are **quantum- and neurobiology-inspired differentiable PyTorch / classical heuristic architectures**, not wetware quantum hardware claims.

- **Dual-Hemisphere Architecture**: Differentiable Hamiltonian coupling between Left (structured/logical) and Right (associative/pattern) hemispheres.
- **4-Neuromodulator Dynamics**:
  - *Dopamine ($DA$)*: Value weighting and reward-driven plasticity.
  - *Acetylcholine ($ACh$)*: Attention gating and learning rate scaling.
  - *Serotonin ($5\text{-}HT$)*: Impulsivity dampening and patience regulation.
  - *Norepinephrine ($NE$)*: Arousal and novelty-triggered exploratory dynamics.
- **Cerebral Oxygenation ($sO_2$)**: Metabolic budget and energy-constrained computation.
- **`QuantumREMSleep`**: Multi-timescale Hamiltonian phase-dispersion consolidation mitigating catastrophic forgetting.
- **`NoisyHippocampalBuffer`**: Lindblad phase diffusion and Sharp-Wave Ripple (SWR) replay buffer.
- **`QuantumZenoAttention` & `QuantumDecisionArbiter`**: Quantum Zeno-inspired frequent-projection attention gating (`quanta.torch.brain`) and classical heuristic decision arbitration (`quanta.cognitive`).
- **`CSFShieldedEnvironment` & `CSFShieldedResonantLayer`**: Differentiable PyTorch open-system Lindblad dephasing attenuation layer parameterized by dielectric screening and viscosity analogies.

---

## Apple Silicon Metal / MLX Acceleration (v1.2.0)

- **Zero-Copy Unified Memory Acceleration**: Dedicated Metal/MLX pipeline eliminating CPU-GPU memory copying, achieving up to **58x–404x speedup** depending on gate depth and qubit count.
- **SIMD-Vectorized Clifford Engine**: Full $(2n+1)\times(2n+1)$ Aaronson-Gottesman binary tableau simulator executing over **3.13 million gates/second**.
- **Matrix Product States (MPS)**: Mixed-canonical 200+ qubit tensor network simulator with Born sampling and non-adjacent SWAP routing (250-qubit GHZ in 3.42 ms).
- **Large-Scale Statevector Contraction**: Dense statevector contraction up to 30 qubits on Apple Silicon Unified Memory architectures.

---

## Multi-Cloud Quantum Hardware Validation

- **IonQ Cloud**: REST API v0.3 protocol, 29-qubit simulator/hardware emulation, and remote telemetry.
- **IBM Quantum**: 127Q/156Q Eagle & Heron processors (`ibm_torino`, `ibm_fez`), native Heavy-Hex ISA transpilation, and IAM authentication.
- **Google Cirq**: Sycamore native gate decomposition and Colab GPU environment support.

---

## Empirical Verification & Certified Rigor

- **2,826+ Passing Automated Tests**: 100% pass rate with zero mock or synthetic oracle calls.
- **Exact Unitarity**: Verified $\|U^\dagger U - I\| < 10^{-14}$ across all 34 gates and Cartan/KAK synthesized unitaries.
- **CPTP Trace Preservation**: Open quantum Lindblad and Kraus evolution strictly bounded by $|\text{Tr}(\rho) - 1.0| < 10^{-12}$.

---

## Deployment Options

| Target | Method | Use Case |
|--------|--------|----------|
| Local | `pip install quanta-sdk` | Fast development & testing |
| PyTorch / AI | `pip install "quanta-sdk[torch]"` | Deep learning & hybrid QNNs |
| Apple Metal | `pip install "quanta-sdk[metal]"` | Apple Silicon GPU acceleration |
| Claude / Gemini / GPT | MCP Server Integration (`fastmcp`) | 25-tool autonomous AI agents |
| Cloud Run / Docker | Dockerfile.mcp + SSE | Always-on remote quantum microservice |
