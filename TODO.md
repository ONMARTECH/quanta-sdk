# Quanta SDK — Engineering Status & Future Milestones

> Last updated: 2026-11-11 | Current version: **v1.3.0 (Production)** | Test Suite: **2,826+ passing tests** | Gates: **34** | MCP Tools: **25**

---

## ✅ Completed Milestones (v1.0.0 – v1.3.0)

### Core, Compiler & Unitary Synthesis
- [x] **34 Built-in Quantum Gates** synchronized across `quanta`, `quanta.core.gates`, `quanta.mcp_server`, and QASM 2.0/3.0 export/import (`CRX`, `CRY`, `CRZ`, `ECR`, `iSWAP`, `CSWAP`, `CH`, `CP`, `MS`, IBM Heron & Google Willow native gates).
- [x] **Exact Cartan / KAK $\text{SU}(4)$ Synthesis** (`quanta/compiler/synthesis.py`) decomposing arbitrary 2-qubit unitaries into $\le 3$ `CX` gates + single-qubit Euler rotations (`UnitarySynthesisPass`).
- [x] **Topology-Aware Routing** (`RouteToTopology`) supporting linear, ring, grid, custom graphs, and IBM Heavy-Hex connectivity.

### Simulators, Gradients & PyTorch Hybrid QML
- [x] **Multi-Backend Simulator Suite**: `StateVectorSimulator` ($O(2^n)$, up to 30 qubits on 48 GB RAM), `MLXSimulator` (`~20x–34x` verified Apple Silicon Metal GPU speedup), `MPSSimulator` (200+ qubits 1D tensor networks), `PauliFrameSimulator` (1,000+ Clifford qubits), `SparseSimulator`, `DensityMatrixSimulator`, and `LindbladSimulator`.
- [x] **Analytical Autodiff (`quanta.gradients` & `quanta.torch`)**: 2-term & 4-term parameter-shift rules, $O(P)$ reverse-mode `adjoint_gradient`, Mari–Bromley–Killoran gauge-invariant Quantum Fisher Information Matrix (`quantum_fisher_information`), `QuantumLayer`, and `ContinuousResonantLayer` (Daleckii–Krein Fréchet autograd).

### Fault-Tolerant Quantum Error Correction (QEC)
- [x] **Surface Code & Dynamic Spacetime Decoding**: Exact $\text{GF}(2)$ check-matrix homology (`H_X @ H_Z^T == 0 mod 2`), final noiseless data-readout boundary round in `simulate_dynamic`, Edmonds Blossom `MWPMDecoder`, and Delfosse–Nickerson peeling `UnionFindDecoder`.
- [x] **6.6.6 Triangular Color Code**: Commuting plaquette stabilizers ($d \in \{3, 5, 7\}$) with syndrome-only Chromobius-style `restriction_decode`.
- [x] **qLDPC & Magic-State Factories**: Gross `[[144, 12, 12]]` Bivariate Bicycle code with higher-order `BP-OSD` combination sweep, `ShorCode` (all 27 single-Pauli recoveries), `SteaneCode`, 15-to-1 Reed–Muller distillation, and exact 7-$T$ `CCZFactory`.

### Gate-Level Layer-3 Solvers, MCP Server & Scientific Integrity
- [x] **Genuine Gate-Level Algorithms**: Griffiths–Niu / Beauregard Semiclassical Iterative QPE Shor's factoring (`N < 65536` on $n+1$ qubits, `allow_classical_fallback=False`), zero-ancilla gate-decomposed Grover + BBHT blind search, Walsh–Hadamard QAOA, Ising Markowitz `portfolio_optimize`, shot-sampled IQAE, Black–Scholes CRN Greeks (`< 2%` error), and Cortes–Vapnik dual QP `qsvm_classify`.
- [x] **25 AI-Native MCP Tools** (`quanta-mcp` & `quanta` CLI entrypoints) with restricted AST-validated execution sandboxes.
- [x] **Scientific Integrity & Packaging Gates**: 3-arm adversarial CVP ablations, symmetric zero-leakage Zeno ablations, automated confidentiality & citation guard (`tests/test_confidentiality_guard.py`), `mkdocs build --strict`, CPU-optimized CI, and `twine check`.

---

## 🚀 Open Engineering Milestones (v1.4.0+)

### 1. Native C++/Rust QEC Kernel Acceleration
- [ ] Optional C++/Rust extension for `BP-OSD` Gaussian elimination and syndrome peeling at block lengths $n > 500$.
- [ ] Stim detector-error-model (`.dem`) import/export bridge for cross-benchmarking hardware syndrome traces.

### 2. Distributed & 2D Tensor-Network Simulation
- [ ] Projected Entangled Pair States (PEPS) contraction heuristics for shallow 2D non-Clifford lattices.
- [ ] Multi-node MPI / GPU-sharded statevector slicing beyond single-host unified RAM (`> 32` dense qubits).

### 3. Industrial QUBO Decomposition & Hardware FinOps
- [ ] Subgraph decomposition (qbsolv / Kernighan–Lin hybrid) for 1,000+ variable combinatorial QUBO instances.
- [ ] Live queue-time, shot-cost, and 2-qubit fidelity telemetry router across IBM Heron, IonQ Forte/Aria, and local simulators.
