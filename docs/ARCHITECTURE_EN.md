# Quanta SDK — Architecture (v1.2.0-production)

## Overview

Quanta SDK is a standalone quantum software architecture engineered for 2026 quantum computing frontiers, structured across **5 Core Scientific Paradigms**. Core circuit, compiler, and statevector engines require only NumPy (while SciPy & NetworkX power Layer-3 solvers and QEC decoders), paired with Apple Silicon Metal GPU acceleration, analytical Daleckii-Krein Hilbert gradients, and dual-track fault-tolerant quantum error correction (FTQC).

## Layered Architecture Diagram

```
+---------------------------------------------------------------------------------+
|                       LAYER 4: AGENTIC & MCP ORCHESTRATION                      |
|  25 Model Context Protocol (MCP) Tools | Claude, Gemini, GPT Autonomous Agents  |
|  "Governing end-to-end quantum workflows via natural language and agentic loops"|
+---------------------------------------------------------------------------------+
|               LAYER 3: DEEP LEARNING & DECLARATIVE COGNITIVE API                |
|  quanta.torch: QuantumLayer | Daleckii-Krein Autograd | Lie Algebra Barren Tool |
|  BiomorphicResonantBrain | SWR Replay | ContinuousResonantLayer (Schrödinger)   |
|  search() | optimize() | vqe() | factor() | portfolio_optimize() | resolve()    |
|  "What to solve?" -- high-level algorithmic execution without manual gate design|
+---------------------------------------------------------------------------------+
|                       LAYER 2: ALGORITHMIC CIRCUIT DSL                          |
|  @circuit | 34 Native Gates (IBM Heron, Google Sycamore, IonQ native parity)    |
|  Parametric Rotations (RX, RY, RZ, CRX, CRY, CRZ, P, U) | measure() | sweep()   |
|  "How to construct the quantum circuit?"                                        |
+---------------------------------------------------------------------------------+
|             LAYER 1: 2026 DUAL-TRACK FAULT-TOLERANCE (FTQC ENGINE)              |
|  Track A: Blossom MWPM & Union-Find | ColorCode A2+ILP | Willow 3D Syndromes    |
|  Track B: Gross [[144, 12, 12]] qLDPC | Native Normalized Min-Sum BP-OSD        |
|  Non-Clifford: 15-to-1 Bravyi-Kitaev Magic State Distillation | Lattice Surgery  |
+---------------------------------------------------------------------------------+
|                 LAYER 0: PHYSICAL COMPUTATION & HARDWARE BACKENDS               |
|  DAG Circuit (Kahn) | Synthesis (ZYZ, Heron, Cartan/KAK SU(4)) | 127/156Q Route |
|  Metal/MLX Zero-Copy GPU | SIMD Clifford ((2n+1)x(2n+1)) | MPS (200+Q)          |
|  GKSL Lindblad & 7-Channel Kraus Noise | Multi-Cloud (IBM REST, Cirq, IonQ)     |
|  "How to execute on physical hardware and high-performance simulators?"         |
+---------------------------------------------------------------------------------+
```

## Dependency Graph

```
mcp_server.py ──┐
                ▼
      quanta.torch / layer3/ ───────► simulator/ ───────► core/
                 │                        │                 ▲
                 ▼                        ▼                 │
             qec/ (FTQC) ────────────► dag/ ────────────────┘
                 │                        ▲
                 ▼                        │
            compiler/ ────────────────────┘
                 │
                 ▼
         backends/ & export/
```

**Rule**: Dependencies always flow downward and inward toward the standalone core (`core/`). No lower layer depends on an upper layer.

## Module Details

### core/ -- Building Blocks

| File | Responsibility |
|------|----------------|
| `types.py` | QubitRef, Instruction, QubitRegister |
| `gates.py` | 34 native gates (`CRX`, `CRY`, `CRZ` + exact `iSWAP`, `ECR`, `CSWAP`, `RCCX`, `RC3X`) + broadcast |
| `circuit.py` | @circuit decorator, CircuitBuilder |
| `measure.py` | Flexible measurement (full, partial) |
| `equivalence.py` | Unitary comparison, fidelity |
| `custom_gate.py` | User-defined unitary gates |

### dag/ -- Directed Acyclic Graph

| File | Responsibility |
|------|----------------|
| `node.py` | InputNode, OpNode, OutputNode (immutable) |
| `dag_circuit.py` | Topological sort (Kahn's), depth, parallel layers |

### compiler/ -- Optimization & Unitary Synthesis Pipeline

| File | Responsibility |
|------|----------------|
| `pipeline.py` | CompilerPass Protocol, chaining, statistics |
| `synthesis.py` | Universal 1Q ZYZ Euler decomposition, IBM Heron `RZ-SX-RZ-SX-RZ`, and 2Q Cartan/KAK $SU(4)$ synthesis |
| `passes/optimize.py` | CancelInverses (H.H=I), MergeRotations |
| `passes/translate.py` | IBM/Google/Quantinuum gate set transpilation |
| `passes/routing.py` | Topology-aware SWAP insertion (`linear`, `ring`, `grid`, 127Q/156Q IBM Heavy-Hex) with dynamic SWAP layout tracking |

### simulator/ -- Simulation Engines

| File | Responsibility |
|------|----------------|
| `base.py` | `SimulatorBackend` ABC — abstract interface for all simulators |
| `statevector.py` | Dense tensor contraction, up to 27 qubits (exact), `apply_phase()` + `apply_noise()` |
| `mlx.py` | Apple Silicon Metal GPU zero-copy Unified Memory tensor contraction engine (up to 30 qubits) |
| `sparse.py` | Dict-based sparse statevector, up to 50 qubits, O(k) memory |
| `mps.py` | Matrix Product State (SVD), 200+ qubits, mixed-canonical Born sampling & non-adjacent SWAP routing |
| `factory.py` | `create_simulator()` — auto-selects best backend |
| `router.py` | Circuit-aware routing (Clifford detection, qubit count) |
| `density_matrix.py` | Mixed states up to 13 qubits via `np.tensordot` CPTP Kraus channels + exact `p/15` 2Q depolarizing |
| `lindblad.py` | Open-system Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) master equation solver |
| `pauli_frame.py` | Full $(2n+1)\times(2n+1)$ Aaronson-Gottesman stabilizer/destabilizer tableau with phase vector |
| `noise.py` | Gate-by-gate stochastic trajectories + `ReadoutError` across 7 channels (Depolarizing, BitFlip, PhaseFlip, AmplitudeDamping, T2Relaxation, Crosstalk, ReadoutError) |
| `accelerated.py` | JAX-GPU / CuPy auto-detection, NumPy fallback |

### layer3/ -- Declarative API

| File | Responsibility |
|------|----------------|
| `search.py` | Auto Grover search |
| `optimize.py` | QAOA optimization |
| `agent.py` | Multi-agent decision modeling |
| `vqe.py` | Variational Quantum Eigensolver |
| `shor.py` | Griffiths-Niu / Beauregard semiclassical QPE integer factoring (`shor.factor`) |
| `qsvm.py` | Cortes-Vapnik Dual QP quantum kernel SVM classification |
| `finance.py` | Portfolio optimization (Markowitz + QAOA) |
| `hamiltonian.py` | Trotterized time evolution, molecular Hamiltonians |
| `entity_resolution.py` | QAOA-based customer deduplication |
| `monte_carlo.py` | Grinko Iterative Quantum Amplitude Estimation (IQAE) + Common Random Numbers (CRN) Black-Scholes Greeks |
| `clustering.py` | Buhrman Swap Test quantum distances + k-means clustering |
| `qml.py` | Quantum ML: variational classifier, quantum kernel, feature maps |

### export/ -- QASM I/O

| File | Responsibility |
|------|----------------|
| `qasm.py` | OpenQASM 3.0 export |
| `qasm_import.py` | QASM 2.0/3.0 import to DAG |

### qec/ -- 2026 Dual-Track Fault-Tolerance (FTQC)

| File | Responsibility |
|------|----------------|
| `codes.py` | BitFlip [[3,1,3]], PhaseFlip [[3,1,3]], Steane [[7,1,3]] |
| `surface_code.py` | Rotated Surface Code [[d^2,1,d]] with exact $\mathbb{F}_2$ homology & 3D spacetime syndrome extraction |
| `color_code.py` | 2D Triangular Color Code with transversal Clifford gates & syndrome-only Kubica-Delfosse $A_2$ lattice + ILP decoder |
| `decoder.py` | Edmonds Blossom MWPM (full-weight perfect matching) and genuine Delfosse-Nickerson `UnionFindDecoder` |
| `qldpc.py` | Gross [[144, 12, 12]] Bivariate Bicycle code, Normalized Min-Sum BP-OSD |
| `distillation.py` | 15-to-1 Bravyi-Kitaev magic state distillation factory, lattice surgery |

### quanta.torch / cognitive/ -- Deep Learning & Biomorphic Inspired Engine

| Module / File | Responsibility |
|---------------|----------------|
| `quanta.torch.QuantumLayer` | PyTorch `nn.Module` layer with analytical parameter-shift autograd VJP |
| `quanta.torch.ContinuousResonantLayer` | Continuous-time Schrödinger evolution, Daleckii-Krein Fréchet derivatives |
| `quanta.torch.BiomorphicResonantBrain` | Quantum- and neurobiology-inspired differentiable dual-hemisphere PyTorch architecture |
| `quanta.lie_algebra` | Dynamical Lie algebra $\dim(\mathfrak{g})$ dimension and analytical barren plateau diagnosis |
| `quanta.cognitive` | Classical/PyTorch biomorphic heuristics: SWR memory consolidation, `QuantumZenoAttention`, `QuantumDecisionArbiter` |

### benchmark/ -- Quality Benchmarking & Empirical Verification

| File | Responsibility |
|------|----------------|
| `qasmbench.py` | 10 standard + 3 large QASMBench circuits |
| `benchpress_adapter.py` | Cross-SDK benchmarking API (Nation et al.) |
| `run_paper_benchmarks.py` | Peer-reviewed empirical microsecond benchmarks |

### Support Modules

| File | Responsibility |
|------|----------------|
| `gradients.py` | Mari-Bromley-Killoran gauge-invariant QFIM, 4-term Anselmetti 2Q parameter-shift, and $O(G)$ `adjoint_gradient` |
| `runner.py` | 6-stage orchestrator: build > DAG > compile > sim > noise > sample > result |
| `result.py` | Measurement results, probabilities, Dirac notation, statevector |
| `visualize.py` | ASCII and SVG circuit diagrams |
| `visualize_state.py` | Probability histogram, Bloch sphere, phase diagram |
| `mcp_server.py` | MCP server — **25 quantum tools** for autonomous AI assistants (SSE + stdio) |

## Data Flow

```
User Code / Autonomous AI Agent (MCP)
          │
          ▼
   @circuit / layer3 / quanta.torch
          │
          ▼
   DAGCircuit (Kahn topological sort)
          │
          ▼
   CompilerPipeline (CancelInverses, MergeRotations, Routing)
          │
          ▼
   QEC Fault-Tolerance Shield (Edmonds Blossom MWPM / Gross qLDPC BP-OSD)
          │
          ▼
   Execution Backend (Metal/MLX, StateVector, Clifford SIMD, MPS, or IBM/Google/IonQ Hardware)
          │
          ▼
   Result (Measurement counts, statevector, analytical gradients, syndrome logs)
```

## Architectural Design Principles

1. **Standalone First-Principles Core**: Core circuit/compiler/statevector engines run in pure Python/NumPy without heavy C++/LLVM toolchains or CUDA runtimes (SciPy & NetworkX power Layer-3 solvers and QEC decoders).
2. **Apple Silicon Zero-Copy GPU Acceleration**: Metal Performance Shaders / MLX zero-copy Unified Memory tensor pipeline yielding up to 58x–404x speedup depending on gate depth and qubit count.
3. **Continuous & Adjoint Hilbert Gradients**: Analytical closed-form Daleckii-Krein Fréchet derivatives ($9.99\times 10^{-16}$), $O(G)$ `adjoint_gradient`, 4-term Anselmetti 2Q shift, and Mari-Bromley-Killoran gauge-invariant QFIM.
4. **2026 Dual-Track FTQC**: Exact $\mathbb{F}_2$ homology Surface Codes with Edmonds Blossom MWPM & Delfosse-Nickerson Union-Find, Kubica-Delfosse $A_2$+ILP Color Codes, and high-rate canonical Gross $[[144, 12, 12]]$ qLDPC with native BP-OSD ($12\times$ physical qubit reduction).
5. **AI-Native MCP Orchestration**: **25 native Model Context Protocol tools** enabling Claude, GPT, and Gemini agents to autonomously design, simulate, transpile, and verify quantum workloads.
6. **Falsifiable Empiricism & Certified Rigor**: **2,826+ automated tests** with zero mock or synthetic oracles, certifying mathematical unitarity and CPTP trace preservation.

---

## Authorship & Identity Metadata

- **Lead Author & Principal Architect**: Abdullah Enes SARI (ORCID: [0000-0002-8827-0587](https://orcid.org/0000-0002-8827-0587))
- **Affiliation**: ONMARTECH Quantum Computing Initiative (`info@onmartech.com`)
- **GitHub Release**: [GitHub Release v1.2.0](https://github.com/ONMARTECH/quanta-sdk/releases/tag/v1.2.0)

