# Quanta SDK — Features

## Gate Set (17 Gates)

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
| U3(θ,φ,λ) | 1 | General single-qubit unitary |
| ISWAP | 2 | Imaginary SWAP |
| SX | 1 | Square root of X |

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
| Density Matrix | 13 | Mixed states, Kraus channels |
| Accelerated | 27 | Auto-detects JAX-GPU / CuPy |

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

## Error Correction Codes

| Code | Notation | Correctable Errors |
|------|----------|-------------------|
| BitFlip | [[3,1,1]] | 1 bit-flip |
| PhaseFlip | [[3,1,1]] | 1 phase-flip |
| Steane | [[7,1,3]] | 1 arbitrary single-qubit error |
| Surface Code | [[d²,1,d]] | ⌊(d-1)/2⌋ errors, threshold ~1% |

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

## MCP Server (AI Integration)

Quanta SDK can be used as an MCP (Model Context Protocol) server,
allowing AI assistants like Claude to perform quantum simulations.

| Tool | Description |
|------|-------------|
| `run_circuit` | Execute arbitrary quantum circuits |
| `create_bell_state` | Quick entanglement demonstration |
| `grover_search` | Grover's search algorithm |
| `shor_factor` | Shor's integer factoring |
| `simulate_noise` | Noisy circuit simulation (7 channels) |
| `list_gates` | Available gate reference |
| `explain_result` | Interpret measurement results |

```bash
# Local (Claude Desktop)
fastmcp install quanta/mcp_server.py --name "Quanta Quantum SDK"

# Remote (Cloud Run)
python -m quanta.mcp_server --transport sse --port 8080
```

## Deployment

| Target | Method | Use Case |
|--------|--------|----------|
| Local | `pip install quanta-sdk` | Development, research |
| Claude Desktop | `fastmcp install` | AI-assisted simulation |
| Cloud Run | Dockerfile.mcp + CI/CD | Always-on MCP server |
| Lambda/Functions | Lightweight package | Serverless computation |
| CI/CD Pipeline | `pip install quanta-sdk` | Automated QC testing |

**Lightweight advantage**: Pure Python + NumPy only. No heavy
framework dependencies. Ideal for serverless, edge computing,
and embedding in CI/CD pipelines.
