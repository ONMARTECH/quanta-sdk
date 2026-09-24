# Project: Quanta SDK Comprehensive Scientific & Engineering Audit

## Architecture
Quanta SDK is a high-performance, standalone, zero-external-lock-in quantum computing framework for Apple Silicon and Python/NumPy.
- **Core Layer**: Analytical unitary gates, circuit DAGs, machine-precision unitarity verification.
- **Simulation Layer**: Statevector (NumPy CPU & MLX Metal GPU), Matrix Product States (MPS tensor networks), Stabilizer / Pauli frame (Aaronson-Gottesman), Density Matrix & Lindblad open system master equation.
- **QEC / FTQC Layer**: Planar Surface Code, Color Code, Bivariate Bicycle qLDPC ([[144, 12, 12]]), Edmonds Blossom MWPM / Union-Find decoders, Willow 3D spacetime syndrome extraction, Magic State Distillation.
- **Continuous / QML Layer**: `quanta.torch` Daleckii-Krein matrix exponential autograd, Dynamical Lie Algebras ($\mathfrak{g} = \langle i H_k \rangle_{\text{Lie}}$), Barren Plateau analytical bounds.
- **Dynamic Circuits / Export**: OpenQASM 3.0 parsing with mid-circuit measurement and classical feedforward conditional branches.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Exact Hamiltonian Evolution | Fix `_matrix_exp` Hermitian projection bug in `quanta/layer3/hamiltonian.py`; implement exact spectral decomposition $V e^{-i \Lambda t} V^\dagger$ and Suzuki-Trotter 2nd/4th order integrators | M1 | Survey R1 |
| 2 | Machine-Precision Unitarity | Enforce two-sided $\|U^\dagger U - I\|_\infty < 10^{-12}$ in `custom_gate.py` and Hilbert-Schmidt fidelity with $|phase|=1$ check in `equivalence.py` | M1 | Survey R1 |
| 3 | Open Quantum Systems & CPTP | Enforce Kraus completeness $\sum K_k^\dagger K_k = I$, trace preservation $\text{Tr}(\rho)=1$, and positive semi-definiteness $\rho \ge 0$ in `quanta/simulator/density_matrix.py` | M1 | Survey R1 |
| 4 | Lindblad Master Equation Solver | Implement differential Lindblad master equation solver ($\dot{\rho} = -i[H,\rho] + \sum \mathcal{D}[L_k]\rho$) with Liouvillian superoperator in `quanta/simulator/lindblad.py` | M1 | Survey R1 |
| 5 | Daleckii-Krein Precision Fix | Standardize `quanta/torch/ops.py` on `complex128` continuous evolution and fix `test_ops_unitary_evolution_norm` | M1 | Survey R1 |
| 6 | Dynamical Lie Algebras & Barren Plateaus | Implement DLA closure engine $\mathfrak{g} = \langle i H_k \rangle_{\text{Lie}}$ and analytical barren plateau mapping in `quanta/qml/lie_algebra.py` | M1 | Survey R1 |
| 7 | Edmonds Blossom MWPM Decoder | Replace greedy matching in `quanta/qec/decoder.py` with standalone Edmonds Blossom MWPM (`networkx.min_weight_matching`) and fix boundary node replication on even defect counts | M2 | Survey R2 |
| 8 | Physical Data Qubit Correction Chains | Reconstruct shortest-path Pauli correction chains on primal/dual lattices in `quanta/qec/decoder.py` rather than returning syndrome indices | M2 | Survey R2 |
| 9 | Surface Code Decoder Integration | Refactor `quanta/qec/surface_code.py` to invoke decoders and verify homology cancellation $H \cdot (e \oplus c) = 0$ without ground-truth cheating | M2 | Survey R2 |
| 10 | Willow Spacetime 3D Syndrome Decoding | Implement 3D spacetime defect graph with phenomenological measurement error noise and eliminate all synthetic mock objects in `DynamicSurfaceCodeResult` | M2 | Survey R2 |
| 11 | qLDPC Bivariate Bicycle Codes | Implement Gross [[144, 12, 12]] bivariate bicycle codes and BP-OSD decoding in `quanta/qec/qldpc.py` | M2 | Survey R2 |
| 12 | Magic State Distillation & Surgery | Implement executable 15-to-1 Bravyi-Kitaev and CCZ distillation circuits and lattice surgery patch models in `quanta/qec/distillation.py` | M2 | Survey R2 |
| 13 | MPS Singular Value Renormalization | Fix MPS state norm collapse on SVD truncation in `quanta/simulator/mps.py` and implement `entanglement_entropy(cut)` | M3 | Survey R3 |
| 14 | Apple Silicon MLX GPU Optimization | Eliminate synchronous host-device copies in `apply_phase()` / `apply_noise()` and optimize tensor transposition memory bandwidth in `quanta/simulator/mlx.py` | M3 | Survey R3 |
| 15 | Stabilizer / Clifford Fast Engine | Vectorize Aaronson-Gottesman tableau operations and implement `SimulatorBackend` interface `.apply()` in `quanta/simulator/pauli_frame.py` | M3 | Survey R3 |
| 16 | Dynamic Circuits & OpenQASM 3.0 | Add mid-circuit measurement and classical feedforward condition handling in `Instruction`, `DAGCircuit`, `quanta/export/qasm_import.py`, and `quanta/runner.py` | M3 | Survey R3 |
| 17 | Comprehensive Scientific Audit Report | Compile authoritative September 2026 academic and engineering audit report in `docs/scientific_audit_september_2026.md` | M4 | Survey R4 |
| 18 | Sept 2026 Ecosystem Matrix & Roadmap | Compile competitive positioning matrix vs Willow/Heron/QuEra and actionable roadmap in `docs/quantum_roadmap_2026.md` | M4 | Survey R4 |
| 19 | Dual Track Opaque-Box E2E Test Suite | Design and verify independent 4-tier E2E test suite in `tests/e2e/` covering all 18 features with `TEST_READY.md` | E2E | Survey Track |
| 20 | Zero Regression Final Pass | Execute complete test suite with 100% pass rate on all new tests and 0 regressions on existing 1611+ tests | M5 | Acceptance |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Theoretical Physics & Math Rigor | Features 1, 2, 3, 4, 5, 6 | None | DONE |
| M2 | Real-Time QEC & 2026 FTQC Standards | Features 7, 8, 9, 10, 11, 12 | None | DONE |
| M3 | Hardware Acceleration & Simulators | Features 13, 14, 15, 16 | None | DONE |
| M4 | Gap Analysis Report & 2026 Roadmap | Features 17, 18 | M1, M2, M3 | DONE |
| M5 | Final E2E Integration & Verification | Feature 20 | M1, M2, M3, M4, E2E | DONE |
| E2E | Opaque-Box E2E Testing Track | Feature 19 (Tiers 1-4 tests) | Parallel to M1-M3 | DONE |

## Interface Contracts

### M1: Hamiltonian Evolution & Open Systems
- `_matrix_exp(A: np.ndarray) -> np.ndarray`: For $A = -i H t$ with Hermitian $H$, $U = V \exp(-i \Lambda t) V^\dagger$. Unitarity $\|U^\dagger U - I\| < 10^{-12}$.
- `quanta/simulator/lindblad.py`:
  - `LindbladMasterEquation(H: np.ndarray, jump_ops: list[np.ndarray])`
  - `evolve(rho_0: np.ndarray, t_span: tuple[float, float], steps: int = 100) -> tuple[np.ndarray, list[np.ndarray]]`
  - Superoperator Liouvillian $\mathcal{L} = -i(I \otimes H - H^T \otimes I) + \sum_k \left( \overline{L_k} \otimes L_k - \frac{1}{2} I \otimes L_k^\dagger L_k - \frac{1}{2} L_k^T \overline{L_k} \otimes I \right)$.
  - Trace preservation $|\text{Tr}(\rho(t)) - 1.0| < 10^{-12}$ and CPTP $\sum_k K_k^\dagger K_k = I$.
- `quanta/qml/lie_algebra.py`:
  - `dynamical_lie_algebra(generators: list[np.ndarray], tol: float = 1e-10) -> list[np.ndarray]`: Computes orthonormal Lie basis $\mathfrak{g} = \langle i H_k \rangle_{\text{Lie}}$ using commutator closure $[A, B] = AB - BA$.
  - `barren_plateau_bound(dla_dim: int, n_qubits: int) -> float`: Returns gradient variance bound $\text{Var}[\partial_\theta \langle O \rangle] \sim O(1/\dim(\mathfrak{g}))$.

### M2: QEC, Decoders & FTQC
- `quanta/qec/decoder.py`:
  - `MWPMDecoder.decode(syndrome: np.ndarray, code_distance: int) -> CorrectionResult`:
    Uses `networkx.min_weight_matching` for minimum-weight perfect matching on all defect nodes with virtual boundary node pairing on both even and odd defect counts. Returns physical data qubit Pauli correction operators $C \in \{I, X, Y, Z\}^{\otimes n}$.
- `quanta/qec/surface_code.py`:
  - `SurfaceCode.simulate_error_correction(...) -> SurfaceCodeResult`: Passes extracted syndrome $s = H_Z e_X$ to `MWPMDecoder`, applies correction $c$, verifies homology $H_Z (e_X \oplus c_X) = 0$.
  - `SurfaceCode.simulate_dynamic(...) -> DynamicSurfaceCodeResult`: Genuine 3D spacetime decoding with time-like edge weights $\ln((1-p_m)/p_m)$. Zero mock objects in result.
- `quanta/qec/qldpc.py`:
  - `BivariateBicycleCode(l: int, m: int, A_poly: list, B_poly: list)`: Implements Gross $[[144, 12, 12]]$ code over $\mathbb{F}_2[x,y]/\langle x^\ell-1, y^m-1\rangle$.
  - `BPOSDDecoder(parity_check_matrix: np.ndarray, max_bp_iter: int = 30, osd_order: int = 10)`: Native BP-OSD decoder.
- `quanta/qec/distillation.py`:
  - `BravyiKitaev15to1Factory()`: Produces $|T\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$ with output error rate $\epsilon_{out} \le 35 p^3$.
  - `CCZFactory()`: Produces $|CCZ\rangle$ states for transversal non-Clifford gate synthesis.

### M3: MPS, MLX GPU & OpenQASM 3.0
- `quanta/simulator/mps.py`:
  - `MPSSimulator.entanglement_entropy(bipartition_cut: int) -> float`: Calculates von Neumann entanglement entropy $S = -\sum_k S_k^2 \ln(S_k^2)$.
  - Truncation re-normalization: $S_{\text{kept}} \leftarrow S_{\text{kept}} / \sqrt{\sum S_{\text{kept}}^2}$ ensuring $\|\psi\| \equiv 1.0$.
- `quanta/simulator/pauli_frame.py`:
  - `PauliFrameSimulator(num_qubits: int)` implements `apply(gate_name: str, qubits: tuple[int, ...]) -> None` conforming to `SimulatorBackend`.
- `quanta/export/qasm_import.py` & `quanta/runner.py`:
  - Supports OpenQASM 3.0 mid-circuit measurement `c[0] = measure q[0]` and conditional feedforward `if (c[0] == 1) { ... }`.

## Code Layout
- `quanta/core/`: Analytical gate matrices, custom gates, circuit equivalence.
- `quanta/layer3/`: Hamiltonian evolution, matrix exponential, time evolution integrators.
- `quanta/simulator/`:
  - `mps.py`: Matrix Product State simulator.
  - `mlx.py`: Apple Silicon Metal/MLX simulator.
  - `pauli_frame.py`: Stabilizer/Clifford simulator.
  - `density_matrix.py`: Density matrix simulator.
  - `lindblad.py`: Lindblad master equation solver.
- `quanta/qec/`:
  - `decoder.py`: Edmonds Blossom MWPM and Union-Find decoders.
  - `surface_code.py`: Surface code and Willow 3D dynamic syndrome extraction.
  - `qldpc.py`: Bivariate Bicycle codes and BP-OSD.
  - `distillation.py`: Magic state distillation factories.
- `quanta/qml/`:
  - `lie_algebra.py`: Dynamical Lie Algebras and Barren Plateau bounds.
- `quanta/torch/`:
  - `ops.py`: Daleckii-Krein matrix exponential autograd.
- `quanta/export/`:
  - `qasm_import.py`: OpenQASM 3.0 importer.
- `docs/`:
  - `scientific_audit_september_2026.md`: Comprehensive Scientific Audit Report.
  - `quantum_roadmap_2026.md`: Competitive Ecosystem Matrix & Roadmap.
- `tests/`:
  - `test_theoretical_physics_m1.py`: M1 unit and analytical tests.
  - `test_qec_ftqc_m2.py`: M2 unit and benchmark tests.
  - `test_hardware_simulators_m3.py`: M3 unit and performance tests.
  - `e2e/`: Opaque-box E2E test suite (Tiers 1-4).
