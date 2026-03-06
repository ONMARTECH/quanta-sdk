# Quanta SDK

Clean, concise, multi-paradigm quantum computing SDK.

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
║  |00>  ████████████████████  50.5%               ║
║  |11>  ███████████████████  49.5%                ║
╠──────────────────────────────────────────────────╣
║  0.71|00> + 0.71|11>                             ║
╚══════════════════════════════════════════════════╝
```

## Architecture

3-layer abstraction — use whichever level fits:

```
Layer 3 (Declarative)  →  search(), optimize(), vqe(), factor()
Layer 2 (Algorithmic)  →  @circuit, H, CX, measure, run
Layer 1 (Physical)     →  DAG, compiler, routing, QASM
```

## Features

### Core
- **17 built-in gates** + `custom_gate()` for user-defined unitaries
- **`@circuit` decorator** — define circuits as Python functions
- **DAG IR** — directed acyclic graph, 3-pass compiler (cancel/merge/translate)
- **Qubit routing** — SWAP insertion for linear/ring/grid topologies
- **QASM 2.0/3.0** — import and export

### Simulators
- **Statevector** — tensor contraction, up to 27 qubits (100s, 2GB)
- **Density matrix** — mixed states + Kraus noise channels, up to 13 qubits
- **Accelerated backend** — JAX-GPU/CuPy auto-detect, NumPy fallback

### Algorithms
| Algorithm | Module | Use Case |
|-----------|--------|----------|
| Grover | `layer3.search` | Unstructured search (quadratic speedup) |
| QAOA | `layer3.optimize` | Combinatorial optimization |
| VQE | `layer3.vqe` | Molecular ground-state energy |
| Shor | `layer3.shor` | Integer factoring (RSA breaking) |
| QSVM | `layer3.qsvm` | Quantum machine learning |
| Multi-Agent | `layer3.agent` | Quantum agent-based modeling |
| Portfolio | `layer3.finance` | Financial portfolio optimization |
| Hamiltonian | `layer3.hamiltonian` | Molecular simulation (H2, LiH, HeH+) |

### Error Correction
- Bit-flip, Phase-flip, Steane [[7,1,3]]
- **Surface code** [[d²,1,d]] — logical qubits with threshold ~1%

### Security
- **BB84 QKD** — quantum key distribution with eavesdropper detection

## Quality Benchmark

### Turnusol Test (8/8)
```
1. ✅ Bell Fidelity        F=1.0000
2. ✅ CHSH Violation        S=2.8284 (Tsirelson bound)
3. ✅ Quantum Teleportation Unitarity preserved
4. ✅ Grover Amplification  99.9% target probability
5. ✅ VQE Convergence       0.000054 Ha error (< chemical accuracy)
6. ✅ Shor Factoring        15 = 3 × 5
7. ✅ QSVM Classification   100% accuracy
8. ✅ Surface Code QEC      0% logical error rate
```

### QASMBench (10/10)
All standard circuits import, compile, and simulate correctly:
bell, ghz, qft, teleportation, deutsch-jozsa, grover, adder, vqe_ansatz, swap_test, random

### Benchpress Compatible
Includes `QuantaBenchpressBackend` for cross-SDK benchmarking with Qiskit, Cirq, Braket.

## Install

```bash
pip install -e ".[dev]"
```

Optional GPU acceleration:
```bash
pip install jax jaxlib   # For JAX GPU
pip install cupy          # For NVIDIA GPU
```

## Test

```bash
pytest                                          # 150 unit tests
python -m quanta.examples.10_quantum_benchmark  # 8-test litmus test
python -m quanta.examples.09_full_demo          # Full feature showcase
```

## Examples

| # | Example | Description |
|---|---------|-------------|
| 01 | Bell State | EPR pair entanglement |
| 02 | GHZ State | Multi-qubit entanglement |
| 03 | Teleportation | Quantum state transfer |
| 04 | Deutsch-Jozsa | Oracle algorithm |
| 05 | Grover Search | Quadratic speedup demo |
| 06 | Molecule Energy | H2, HeH+ ground state via VQE |
| 07 | Portfolio | Tech stocks + crypto optimization |
| 08 | QKD BB84 | Quantum key distribution |
| 09 | Full Demo | All features in one script |
| 10 | Benchmark | SDK quality litmus test |

## Stats

- **50+ files** | **8,500+ lines** | **150+ tests**
- **v0.4.0** | Python 3.10+ | NumPy (+ optional JAX/CuPy)

## Author

**Abdullah Enes SARI** — [ONMARTECH](https://onmartech.com)

📧 info@onmartech.com

## Contributing

Contributions are welcome! Please open an issue or pull request.

## License

Apache License 2.0 — see [LICENSE](LICENSE)

Same license used by Google Cirq and IBM Qiskit.

