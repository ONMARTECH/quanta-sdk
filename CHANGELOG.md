# Changelog

All notable changes to Quanta SDK.

Format: [Semantic Versioning](https://semver.org/)

## [v1.2.0] - 2026-09-25 — 2026 Dual-Track FTQC, Daleckii-Krein Autograd, & Academic Whitepaper

### Added — 2026 Dual-Track Fault-Tolerant Quantum Computing (FTQC)
- `quanta/qec/decoder.py`: Edmonds Blossom Minimum-Weight Perfect Matching (`MWPMDecoder`) replacing greedy heuristics, achieving theoretical threshold ($p_{\text{th}} \approx 1\%$).
- `quanta/qec/qldpc.py`: Canonical Gross $[[144, 12, 12]]$ Bivariate Bicycle qLDPC code with native Normalized Min-Sum BP-OSD-0 decoder (1.54 ms syndrome decode, $12\times$ qubit footprint reduction).
- `quanta/qec/distillation.py`: 15-to-1 Bravyi-Kitaev magic state distillation factory ($\epsilon_{\text{out}} \le 35 p^3$) and planar lattice surgery.
- `quanta/qec/surface_code.py`: Google Willow-compliant 3D spacetime syndrome extraction tracking circuit and measurement errors across fault cycles.

### Added — Continuous Hilbert Gradients & Daleckii-Krein Autograd
- Closed-form Fréchet matrix exponential autograd eliminating Padé truncation errors ($>1.3\times 10^{-6}$) with exact machine precision ($9.99\times 10^{-16}$).
- `quanta/torch/lie_algebra.py`: Dynamic Lie algebra ($\dim(\mathfrak{g})$) dimension and analytical barren plateau pre-flight diagnosis.

### Added — MCP Server Tools Expansion & Academic Whitepaper
- 23 Model Context Protocol tools (`estimate_fault_tolerant_cost`, `quanta_reasoning_eval`, `transpile_for_target`).
- Academic Software Architecture Paper (`docs/papers/quanta_framework_paper.md` and `docs/arxiv/quanta_framework/main.tex`).
- Permanent CERN/Zenodo Release DOI: [10.5281/zenodo.22952779](https://doi.org/10.5281/zenodo.22952779).
- Total automated tests reached **2,076 tests** with 100% pass rate.

---

## [v1.1.0] - 2026-09-16 — Pillar 2: PyTorch Quantum Layer & Continuous Resonance

### Added — PyTorch Native Quantum Engine (`quanta.torch`)
- `quanta/torch/layer.py`: Added `QuantumLayer(nn.Module)` with custom `torch.autograd.Function` bridging Quanta quantum circuits to PyTorch deep learning models.
  - Analytical Parameter-Shift Rule for exact VJPs without numerical finite differences.
  - 4 built-in parameterized ansatz presets (`hardware_efficient`, `strong_entangling`, `real_amplitudes`, `reuploading`) and custom circuit builder support.
  - Full compatibility with `nn.Sequential`, batch inputs, Adam/SGD optimizers, and Apple Silicon MPS/Metal acceleration.
- `quanta/torch/continuous.py`: Added `ContinuousResonantLayer(nn.Module)` modeling brain-inspired continuous-time quantum network resonance ($U(t) = e^{-i H(x, \theta) t}$).
  - Non-sequential, all-at-once holistic quantum dynamics ("her yerden aynı anda ışıldayan kuantum dinamikleri") without discrete gate synchronization bottlenecks.
  - Learnable network coupling topology ($J$), local bias fields ($h$), input projections ($W$), and interaction duration ($t$).
  - Simultaneous multi-observable readout ($\langle Z_j \rangle, \langle X_j \rangle$) across all network nodes.
  - Exact analytical autograd engine powered by **Daleckii-Krein Fréchet matrix exponential derivatives** and **Ehrenfest theorem time derivatives**.
- `quanta/torch/ops.py`: Native statevector operations, Pauli Kronecker product caching, and normalized sinc kernel for degenerate spectra.
- `docs/theory/continuous_quantum_neural_dynamics.md`: 920-line authoritative academic whitepaper (51 citations) synthesizing EPR non-locality, continuous-time quantum walks, Orch-OR, Posner molecule nuclear spin coherence, biophotonics, and 5 analytical gradient theorems.
- 111 dedicated unit and E2E tests (`test_torch_layer.py`, `test_torch_continuous.py`, `test_torch_e2e.py`, `test_torch_ops.py`, `test_m1_mathematical_theorems.py`) with 96.37% test coverage on `quanta/torch`.

---

## [v1.0.0] - 2026-09-16 — Milestone Major Release

### Added — World's First Native Apple Silicon Metal/MLX Quantum Simulator
- `quanta/simulator/mlx.py`: Added `MLXSimulator(SimulatorBackend)` native Metal GPU accelerator.
  - Multi-dimensional $N$-axis tensor representation `[2]*N` for scalable 30+ qubit simulation on Apple Silicon Unified Memory.
  - Native gate tensor contractions on Apple M5 Pro GPU via Metal Performance Shaders.
  - **404x speedup** on 26 qubits (0.076s vs 30.7s CPU) and **24.5x speedup** on 30 qubits (1.88s vs 46.2s CPU).
- `quanta/simulator/accelerated.py`: Integrated Apple MLX as top-priority GPU acceleration on Darwin ARM64 (`mlx-metal` > `jax-gpu` > `cupy` > `numpy`).
- `quanta/simulator/router.py` & `factory.py`: Automatic simulator routing to `MLXSimulator` on Apple Silicon.
- `pyproject.toml`: Added `metal` optional dependency (`mlx>=0.20`, `mlx-metal>=0.20`) and upgraded status to `5 - Production/Stable`.

### Added — Multi-Cloud Hardware Production Support
- **IonQ REST API v0.3**: Live verified execution on 29-qubit cloud simulator with 100% fidelity.
- **Google Cirq & Quantum Engine**: Local Sycamore QASM simulation and Google Colab integration for GPU runtimes.
- **IBM Quantum IAM**: OpenQASM 3.0 compilation and IAM token integration.

### Added — 5 Strategic Pillars Roadmap
- Documented 5 strategic pillars in `ROADMAP.md`:
  1. Pillar 1: Native Apple Silicon Metal/MLX Quantum Engine (v1.0.0 — COMPLETED)
  2. Pillar 2: PyTorch Native Quantum Layer (`quanta.torch.QuantumLayer`) (v1.1.0)
  3. Pillar 3: Real-Time QEC & High-Speed Syndrome Decoder (MWPM/Union-Find) (v1.2.0)
  4. Pillar 4: Industrial QUBO & Large-Scale Graph Partitioning Decomposer (v1.3.0)
  5. Pillar 5: Agentic Quantum FinOps & Cost Arbiter for MCP (v1.4.0)

---

## [v0.9.3] - 2026-09-15

### Added — Apple Silicon M5 Pro (48 GB RAM) Hardware Optimization
- `quanta/config.py`: Added dynamic RAM detection (`get_system_memory_gb()`, `get_max_dense_qubits()`)
- `StateVectorSimulator`: Max qubits elevated from 27 to **30 qubits** on 48 GB local hardware ($2^{30} \times 16\text{ B} \approx 16\text{ GB}$)
- Dynamic hardware memory test in `tests/test_tier1_coverage.py`

### Added — 2026 Agentic MCP Tools (Gemini 3.8 / Claude 3.7 / GPT-5)
- Tool count increased from 20 to **23 tools**:
  - `estimate_fault_tolerant_cost`: Surface code distance $d$, physical qubit footprint, and T-factory budget calculation (Google Willow & IBM Starling compatible)
  - `quanta_reasoning_eval`: Deep reasoning evaluation for LLM-generated circuits (entangling gate density, redundant gate detection, architectural feedback)
  - `transpile_for_target`: Native target compilation for IBM Heron, Google Willow, and IonQ Aria

### Added — Google Willow Dynamic Surface Code (FTQC)
- `SurfaceCode.simulate_dynamic()`: Multi-cycle dynamic surface code simulation
- `DynamicSurfaceCodeResult`: 3D spacetime defect tracking ($\Delta s_t = s_t \oplus s_{t-1}$), measurement flip noise, and Willow exponential suppression factor ($\Lambda$)

### Changed & Hardened — Type Safety & Compilation
- Fixed 32+ Mypy type violations across core, simulator, and QEC modules
- `quanta/export/qasm.py`: Extended `to_qasm()` to accept both `CircuitDefinition` and compiled `DAGCircuit`
- `CustomGate`: Full conformance with `Gate` base class architecture, with isolated registry cleanup fixtures (%100 coverage)
- Converted all 14 tutorial and migration guides to latest SDK APIs

### Quality & Benchmark
- Tests: 820 → **889 passed, 1 skipped (0 failed)**
- Test Coverage: 80.37% → **90.11%** (exceeding strict >= 80% threshold)
- Mypy: 0 errors across 25 source files (`quanta/core`, `quanta/simulator`, `quanta/qec`)
- Ruff: 0 errors across entire repository
- MkDocs: builds in 1.47s with 0 errors

---

## [v0.9.2] - 2026-03-31

### Added — Multi-Backend Simulator Architecture
- `SimulatorBackend` ABC — abstract base class for all simulators
- `create_simulator(n, method="auto")` — factory with auto-selection
- `select_simulator(n, gate_names)` — circuit-aware routing
- New `quanta/simulator/__init__.py` public API exports

### Added — Sparse StateVector Simulator
- `SparseSimulator` — dict-based amplitude storage, O(k) memory
- Supports up to 50 qubits for sparse circuits (GHZ, oracle, product states)
- 35-qubit GHZ: 120 bytes vs 256 GB dense!
- Optimized 1q, 2q, and Nq gate paths with MSB qubit convention
- `apply_phase()` for Grover oracle support
- Diagnostics: `num_nonzero`, `sparsity`, `memory_bytes`

### Added — MPS Tensor Network Simulator
- `MPSSimulator` — SVD-based Matrix Product State decomposition
- O(n·χ²) memory: 100-qubit GHZ in ~32 KB, 200-qubit QAOA ✅
- Bond dimension control via `chi_max` parameter
- Adjacent gates: einsum + SVD split + truncation
- Non-adjacent gates: automatic SWAP chain insertion
- Sequential qubit-by-qubit sampling (O(n·χ²) per sample)
- `truncation_error`, `bond_dimensions`, `max_bond_dim` diagnostics

### Added — Circuit-Aware Router
- Clifford detection → PauliFrameSimulator (1000+ qubits)
- Qubit count routing: dense ≤27 → sparse ≤50 → MPS 50+
- `analyze_circuit()` — returns circuit analysis with recommendation

### Changed — L3 Architecture Refactor
- All 10 Layer 3 modules decoupled from `StateVectorSimulator`
- Now use `create_simulator()` factory: agent, clustering, entity_resolution,
  finance, optimize, qml, qsvm, search, shor, vqe
- Type hints use `SimulatorBackend` ABC instead of concrete class

### Security — v0.9.1 Hardening (included)
- **RCE Prevention**: `_validate_code()` + `_SAFE_BUILTINS` for exec()
- **eval() Elimination**: Pure AST recursive walker in qasm_import.py
- **Traceback Leak Fix**: All 14 MCP error responses sanitized with `_safe_error()`

### Quality
- Tests: 774 → 820 (+46 tests: 22 sparse + 24 MPS)
- Coverage: 91%
- Files: 84 Python modules
- Simulators: 4 → 6 (SparseSimulator + MPSSimulator)
- Max Qubits: 27 (dense) → 200+ (MPS)
- Ruff: 0 errors

---

## [v0.9.1] - 2026-03-31

### Added — Option Greeks (Finance)
- `compute_greeks()` — finite-difference Monte Carlo for Δ,Γ,ν,Θ,ρ
- `GreeksResult` dataclass with `summary()` method
- Supports european_call and european_put payoffs

### Added — QEC Decode/Correct API
- `QECCode.decode()` — inverse of encode() for all codes
- `QECCode.lookup_table()` — syndrome → correction mapping
- `correct_error(code, syndrome)` — correction action helper
- **ShorCode** [[9,1,3]] — Shor's 9-qubit code (encode + decode)
- BitFlip/PhaseFlip codes now have full decode + lookup

### Added — SDK Configuration
- `quanta/config.py` — `QuantaConfig` class for credential management
- TOML round-trip: load/save `~/.quanta/config.toml`
- Per-backend credential management (IBM, IonQ, Google)
- `describe()` with masked credential output
- `QUANTA_CONFIG_DIR` env override

### Added — QASM 3.0 Round-Trip Tests
- 5 verified round-trip tests: Bell, parametric, GHZ, QASM 2.0, from_qasm_gates
- Export → import preserves gates, parameters, and measurements

### Added — MCP Tools (18 → 20)
- `option_greeks` — compute option sensitivities via MCP
- `qec_diagnose` — syndrome → correction lookup via MCP

### Quality
- Tests: 748 → 774 (+26 tests)
- Coverage: 91%
- Files: 84 → 86
- QEC codes: 6 → 7 (ShorCode added)
- Ruff: 0 errors

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
