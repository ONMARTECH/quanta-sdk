# Changelog

All notable changes to Quanta SDK.

Format: [Semantic Versioning](https://semver.org/)

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
