# Changelog

All notable changes to Quanta SDK.

Format: [Semantic Versioning](https://semver.org/)

---

## [v0.9.0] - 2026-03-29

### Added — Primitives (IBM V2 Compatible)
- **Estimator**: `Estimator().run(circuit, observables=[("ZZ", 1.0)])` — exact ⟨ψ|O|ψ⟩
  - Pauli string → tensor product matrix construction
  - Variance computation: Var(O) = ⟨O²⟩ - ⟨O⟩²
  - Single circuit broadcast to multiple observables
- **Sampler**: `Sampler().run(circuit, shots=4096)` — measurement sampling
  - Batch execution: pass list of circuits
  - Quasi-probability distributions + raw counts
- Both support `run_async()` for parallel batch execution

### Added — @quantum Decorator (PennyLane @qml.qnode Equivalent)
- `@quantum(qubits=2, observable=[("ZZ", 1.0)])` — auto-differentiable circuits
- `circuit.expectation()` — exact expectation values
- `circuit.gradient()` — parameter-shift rule gradients
  - Correct analytical gradients: d/dθ cos(θ) verified
- `await circuit.run_async()` — async execution
- Parametric circuit support: `build(**kwargs)` infrastructure

### Added — Async Execution
- `run_async(circuits, shots=N)` — top-level async batch runner
- Thread-pool parallelism via `asyncio.run_in_executor()`

### Added — Benchmark Suite
- `scripts/benchmark.py` — 10 benchmarks, auto-generates `docs/BENCHMARK.md`
  - Bell (0.45ms), GHZ-10 (0.69ms), GHZ-20 (83ms)
  - Grover-4 (0.38ms), Grover-8 (0.49ms), VQE-H₂ (301ms)
  - Estimator (0.09ms), Gradient (0.51ms), Sampler batch (4.16ms)

### Added — Property-Based Testing
- `tests/test_property.py` — Hypothesis-based invariant tests
  - Gate unitarity ∀ θ ∈ [-4π, 4π]: U†U = I
  - Pauli anticommutation: {σ_i, σ_j} = 2δ_ij
  - Probability normalization: Σp = 1
  - Circuit determinism: same seed → same result
  - Estimator bounds: ⟨Z⟩ ∈ [-1, 1]

### Changed
- `CircuitDefinition.build()` now accepts `**kwargs` for parametric circuits
- Version bump: 0.8.1 → 0.9.0

### Tests
- **669 tests**, 89% coverage
- 28 primitives tests + 21 property-based tests

---

## [v0.8.1] - 2026-03-15

### Fixed
- **Perplexity rebuttal**: Proof tests addressing all claimed SDK issues
- **Post-check workflow**: Automated validation script for SDK health

### Added — Documentation
- 14 tutorials (01–10c): Getting Started through QEC Threshold
- Migration guides: from-qiskit, from-pennylane, from-cirq
- 3 cookbook recipes: bell-state, option-pricing, quantum-classification
- MkDocs documentation site with API reference
- Domain use-case packs: Finance + Marketing/CRM

### Tests
- **620 tests**, 88% coverage

---

## [v0.8.0] - 2026-03-10

### Added
- **QML module** (`layer3/qml.py`): Variational quantum classifier with ZZFeatureMap
- **Quantum Monte Carlo precision**: Golden-section MLE refinement for amplitude estimation
- **IBM Quantum MCP tools**: `run_on_ibm`, `ibm_backends`, `ibm_job_result` — real hardware via MCP
- **surface_code_simulate** + **density_matrix_sim** MCP tools
- **16 MCP tools** total (was 10)

### Fixed
- 5 critical bugs from Claude MCP stress test
- Noise fidelity: Kraus density matrix simulation
- Readout error model + QMC precision scaling
- QML test suite + CI pipeline stability

### Tests
- **515 tests**, 87% coverage

---

## [v0.7.1] - 2026-03-09

### Added
- **IBM Quantum REST backend** (`backends/ibm_rest.py`): Direct REST API, no Qiskit dependency
  - ISA transpilation for IBM Heron r3 processors
  - Tested on real hardware (ibm_torino)
- **SVG circuit visualizer** (`visualize_svg.py`): Professional IBM-inspired circuit diagrams
- **25 quantum gates**: Full IBM Quantum gate parity
- **MCP server**: 10 tools for AI agent integration

### Tests
- **488 tests**, verified on real IBM Quantum hardware

---

## [v0.7.0] - 2026-03-08

### Added — 3 New Quantum Algorithms
- **Quantum Monte Carlo** (`layer3/monte_carlo.py`): Amplitude estimation for option pricing
- **QAOA Optimizer** (`layer3/optimize.py`): Combinatorial optimization (MaxCut, TSP)
- **Quantum Clustering** (`layer3/clustering.py`): Swap-test based data clustering

### Added — MCP Server
- **FastMCP AI integration** (`mcp_server.py`): 10 tools for Claude/GPT
  - `run_circuit`, `grover_search`, `shor_factor`, `simulate_noise`
  - `draw_circuit`, `list_gates`, `explain_result`
  - `monte_carlo_price`, `qaoa_optimize`, `cluster_data`

### Fixed
- Shor overflow for large modular exponentiation
- Grover parameter validation for odd-sized search spaces
- Recursive factoring edge cases

### Tests
- **476 tests** (was 457)

---

## [v0.6.1] - 2026-03-08

### Fixed — Structural Architecture Improvements

- **NoiseModel integration**: `run(circ, noise=NoiseModel())` — noise is now a first-class citizen in the execution pipeline
- **Shor QFT via DAG**: Real H/RZ/SWAP gates through DAG pipeline; modular exponentiation stays classical (documented trade-off)
- **Grover encapsulation**: `sim._state` → public `apply_phase()` + `state` setter
- **Surface code**: Stabilizer-based syndrome extraction + deterministic BFS logical error check (replaces probabilistic model)
- **Color code RNG**: Reproducible randomness via `rng` parameter propagation

### Fixed — Security

- **MCP server**: `exec()` sandboxed with restricted `__builtins__` whitelist
- **QASM import**: `eval()` replaced with safe arithmetic parser (`_safe_parse_param`)

### Fixed — Encapsulation

- All `sim._state` external access eliminated — public API only (`state`, `apply_phase`, `apply_noise`)
- `StateVectorSimulator.apply_noise()` public method added
- Files fixed: `runner.py`, `equivalence.py`, `optimize.py`, `search.py`, `mcp_server.py`

### Fixed — Code Quality

- **0 ruff lint errors** across quanta/ + tests/ (was 107+78)
- BitFlip/PhaseFlip distance corrected: `d=1` → `d=3` ([[3,1,3]])
- `sdg`/`tdg` gate mappings corrected to proper Hermitian conjugates
- `list.pop(0)` → `deque.popleft()` in DAG topological sort + surface code BFS
- `CircuitSpec` removed from `__all__` (dead code)
- `CY` gate added to public exports
- Turkish comments → English across 12+ files
- Thread-safe circuit builder: `_active_builders` → `threading.local()`

### Tests
- **457 tests** (445 passed, 12 MCP skipped when fastmcp not installed)
- MCP tests gracefully skip with `pytest.importorskip`

---

## [v0.6.0] - 2026-03-07

### Added — QEC Enhancements (Google quantumlib parity)
- **Color Code**: (`qec/color_code.py`)
  - Triangular lattice, 3-colorable plaquettes (R, G, B)
  - Restriction decoder (chromobius-inspired)
  - Transversal Clifford gates (H, S, CX)
- **QEC Decoders**: (`qec/decoder.py`)
  - MWPMDecoder: greedy minimum weight perfect matching
  - UnionFindDecoder: near-linear O(n·α(n)) cluster-based
- **Pauli Frame Simulator**: (`simulator/pauli_frame.py`)
  - Aaronson-Gottesman stabilizer tableau
  - O(n) per gate, O(n²) memory, 50-qubit GHZ in <5s

### Tests
- 37 new tests → **216 total**

---

## [v0.5.0] - 2026-03-07

### Added — Hardware Backends
- **IBM Quantum Backend**: (`backends/ibm.py`)
  - QASM 2.0 bridge to Qiskit
  - AerSimulator local + IBM Quantum hardware
- **IonQ Backend**: (`backends/ionq.py`)
  - Pure REST API, zero Python dependencies
  - IonQ native JSON gate format
- **Google Backend refactor**: (`backends/google.py`)
  - DAGCircuit interface (consistent with IBM/IonQ)
- Runner `backend=` parameter for all backends

### Tests
- 21 new backend tests → **179 total**

---

## [v0.4.0] - 2026-03-06

### Added — Real-World Applications
- **VQE**: Variational Quantum Eigensolver (`layer3/vqe.py`)
  - Hardware-efficient ansatz, parameter-shift gradients
  - H2: 100.00% accuracy, HeH+: 99.99% accuracy
- **Shor's Algorithm**: Integer factoring (`layer3/shor.py`)
  - QFT-based period finding, continued fractions
  - 15 = 3 × 5 verified
- **QSVM**: Quantum Support Vector Machine (`layer3/qsvm.py`)
  - ZZFeatureMap kernel, quantum-enhanced classification
  - 100% accuracy on linearly separable data
- **Hamiltonian Simulation**: (`layer3/hamiltonian.py`)
  - Trotterized time evolution
  - Pre-defined molecules: H2, LiH, HeH+
- **Portfolio Optimization**: (`layer3/finance.py`)
  - Markowitz mean-variance → QUBO, QAOA-inspired selection
  - Conservative vs aggressive profiles, Sharpe ratio
- **Surface Code**: (`qec/surface_code.py`)
  - [[d²,1,d]] logical qubits, error correction simulation
  - Threshold ~1%, error suppression verified
- **BB84 QKD Protocol**: Quantum key distribution (`examples/08_qkd_bb84.py`)
  - Eavesdropper detection via ~25% error rate

### Added — Benchmark Infrastructure
- **QASM Import**: (`export/qasm_import.py`)
  - Full QASM 2.0/3.0 → DAG parser
  - Registers, parametric gates, pi expressions
- **QASMBench Suite**: (`benchmark/qasmbench.py`)
  - 10 standard circuits: bell, ghz, qft, teleportation, deutsch-jozsa, grover, adder, vqe_ansatz, swap_test, random_10
  - Full pipeline benchmark: import → compile → simulate → metrics
  - 10/10 circuits pass
- **Benchpress Adapter**: (`benchmark/benchpress_adapter.py`)
  - `QuantaBenchpressBackend` compatible with Nation et al. framework
  - new_circuit, apply_gate, optimize, export_qasm, simulate

### Added — Demo Cases
- `06_molecule_energy.py` — H2 + HeH+ ground state (VQE)
- `07_portfolio_optimization.py` — Tech stocks + crypto portfolio
- `08_qkd_bb84.py` — Quantum Key Distribution
- `09_full_demo.py` — All features in one script
- `10_quantum_benchmark.py` — 8-test quality litmus test

### Tests
- 52 new test cases → **150 total** (was 98)
- `test_layer3_new.py`: VQE, Shor, QSVM, finance, hamiltonian
- `test_qec_surface.py`: Surface code parameters, error correction
- `test_qasm_import.py`: QASM parsing, round-trip, QASMBench, Benchpress

---

## [v0.3.0] - 2026-03-05

### Added — Structural Improvements
- Custom gates: `custom_gate("name", matrix)`
- Density matrix simulator: mixed states + Kraus noise
- Qubit routing: linear/ring/grid topology, SWAP insertion
- Accelerated backend: JAX/CuPy GPU (auto-detect, NumPy fallback)
- Parameter sweep: `sweep(circuit, params={...})`
- Cirq-style display: `print(result)`, `dirac_notation()`, `histogram()`
- 27-qubit support (100.3s, 2GB)

---

## [v0.2.0] - 2026-03-04

### Added — Performance
- Tensor contraction simulator: O(2^n) replacing O(4^n)
- Google Quantum Engine backend (QASM bridge)
- Load test + installation documentation

### Performance
| Qubits | v0.1 | v0.2 | Speedup |
|--------|------|------|---------|
| 12 | 1.785s | 0.003s | 685x |
| 20 | impossible | 0.073s | ∞ |
| 27 | impossible | 100.3s | ∞ |

---

## [v0.1.0] - 2026-03-03

### Added — Foundation
- Core: 17 gates, circuit decorator, measurement
- DAG: Directed acyclic graph representation
- Compiler: 3-pass optimization pipeline
- Simulator: Statevector (Kronecker)
- Layer 3: Grover, QAOA, multi-agent
- QEC: Bit-flip, phase-flip, Steane codes
- Export: OpenQASM 3.0
- 98 unit tests
