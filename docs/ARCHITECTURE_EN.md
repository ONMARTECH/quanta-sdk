# Quanta SDK — Architecture

## Overview

Quanta uses a **3-layer independent architecture**. Each layer can be used independently.

## Layer Diagram

```
+---------------------------------------------------------+
|              LAYER 3: DECLARATIVE API                   |
|  search() | optimize() | vqe() | factor() | qsvm()     |
|  portfolio_optimize() | resolve() | MultiAgentSystem    |
|  "What do you want?" -- no gate knowledge needed        |
+---------------------------------------------------------+
|              LAYER 2: ALGORITHMIC DSL                   |
|  @circuit | H/CX/RZ | measure() | run() | sweep()      |
|  custom_gate() | 25 built-in gates (IBM parity)          |
|  "How to build the circuit?"                            |
+---------------------------------------------------------+
|              LAYER 1: PHYSICAL ENGINE                   |
|  DAG | Compiler | Routing | Simulator | QEC | QASM I/O  |
|  "How will it run on hardware?"                         |
+---------------------------------------------------------+
```

## Dependency Graph

```
layer3/ -------> simulator/ -------> core/
                      |
runner.py -------> dag/ -------> core/
                      |
compiler/ -------> dag/ -------> core/
                      |
backends/ -------> simulator/ -------> core/
                      |
export/ -------> dag/ -------> core/
                      |
benchmark/ -------> export/ + simulator/ + compiler/
                      |
qec/ -------> core/
```

**Rule**: Dependencies always flow downward. No lower layer depends on an upper layer.

## Module Details

### core/ -- Building Blocks

| File | Responsibility |
|------|----------------|
| `types.py` | QubitRef, Instruction, QubitRegister |
| `gates.py` | 25 gates + broadcast (IBM Heron parity) |
| `circuit.py` | @circuit decorator, CircuitBuilder |
| `measure.py` | Flexible measurement (full, partial) |
| `equivalence.py` | Unitary comparison, fidelity |
| `custom_gate.py` | User-defined unitary gates |

### dag/ -- Directed Acyclic Graph

| File | Responsibility |
|------|----------------|
| `node.py` | InputNode, OpNode, OutputNode (immutable) |
| `dag_circuit.py` | Topological sort (Kahn's), depth, parallel layers |

### compiler/ -- Optimization Pipeline

| File | Responsibility |
|------|----------------|
| `pipeline.py` | CompilerPass Protocol, chaining, statistics |
| `passes/optimize.py` | CancelInverses (H.H=I), MergeRotations |
| `passes/translate.py` | IBM/Google/Quantinuum gate set transpilation |
| `passes/routing.py` | Topology-aware SWAP insertion (linear/ring/grid) |

### simulator/ -- Simulation Engines

| File | Responsibility |
|------|----------------|
| `statevector.py` | Tensor contraction, up to 27 qubits, `apply_phase()` + `apply_noise()` public API |
| `density_matrix.py` | Mixed states + Kraus noise channels, up to 13 qubits |
| `pauli_frame.py` | Aaronson-Gottesman stabilizer tableau, 50-qubit GHZ in <5s |
| `noise.py` | 7 noise channels: Depolarizing, BitFlip, PhaseFlip, AmplitudeDamping, T2Relaxation, Crosstalk, ReadoutError |
| `accelerated.py` | JAX-GPU / CuPy auto-detection, NumPy fallback |

### layer3/ -- Declarative API

| File | Responsibility |
|------|----------------|
| `search.py` | Auto Grover search |
| `optimize.py` | QAOA optimization |
| `agent.py` | Multi-agent decision modeling |
| `vqe.py` | Variational Quantum Eigensolver |
| `shor.py` | Integer factoring (period finding + QFT) |
| `qsvm.py` | Quantum kernel SVM classification |
| `finance.py` | Portfolio optimization (Markowitz + QAOA) |
| `hamiltonian.py` | Trotterized time evolution, molecular Hamiltonians |
| `entity_resolution.py` | QAOA-based customer deduplication |
| `monte_carlo.py` | Quantum Monte Carlo, amplitude estimation, option pricing |
| `clustering.py` | Quantum swap-test distances + k-means clustering |

### export/ -- QASM I/O

| File | Responsibility |
|------|----------------|
| `qasm.py` | OpenQASM 3.0 export |
| `qasm_import.py` | QASM 2.0/3.0 import to DAG |

### qec/ -- Error Correction

| File | Responsibility |
|------|----------------|
| `codes.py` | BitFlip [[3,1,3]], PhaseFlip [[3,1,3]], Steane [[7,1,3]] |
| `surface_code.py` | Surface code [[d^2,1,d]], stabilizer-based syndrome extraction |
| `color_code.py` | Color code, triangular lattice, restriction decoder |
| `decoder.py` | MWPM + Union-Find decoders |

### benchmark/ -- Quality Benchmarking

| File | Responsibility |
|------|----------------|
| `qasmbench.py` | 10 standard + 3 large QASMBench circuits |
| `benchpress_adapter.py` | Cross-SDK benchmarking API (Nation et al.) |

### Support Modules

| File | Responsibility |
|------|----------------|
| `runner.py` | 6-stage orchestrator: build > DAG > compile > sim > noise > sample > result |
| `result.py` | Measurement results, probabilities, Dirac notation |
| `visualize.py` | ASCII circuit diagram |
| `visualize_state.py` | Probability histogram, phase diagram |
| `mcp_server.py` | MCP server — 14 tools for AI-assisted quantum computation (SSE + stdio) |

## Data Flow

```
User Code               SDK Internals
    |                       |
@circuit(qubits=N) ---> CircuitDefinition
    |                       |
H(q[0]), CX(...)    ---> CircuitBuilder (lazy Instruction list)
    |                       |
measure(q)          ---> MeasureSpec
    |                       |
run(circuit)        ---> +- DAGCircuit.from_builder()
                         +- CompilerPipeline.run(dag)
                         +- StateVectorSimulator.apply(ops)
                         +- simulator.sample(shots)
                         +- Result(counts, probs, statevector)
```

## Design Decisions

1. **Lazy Evaluation**: Gates are recorded as Instructions, not applied immediately
2. **DAG Representation**: Enables parallelism detection and optimization
3. **Protocol-based**: CompilerPass is a Protocol -- duck typing is sufficient
4. **Immutable**: QubitRef, Instruction, nodes are frozen dataclasses
5. **Thread-local Builder**: Multiple circuits can be built concurrently
6. **Hybrid approach**: Classical blocking + quantum optimization for real-world problems
7. **Lightweight**: Pure Python + NumPy only — ideal for serverless (Lambda, Cloud Functions), edge computing, and CI/CD integration
8. **AI-native**: MCP server enables AI assistants to perform quantum computations directly
9. **Encapsulation**: All simulator state access through public API (`state`, `apply_phase`, `apply_noise`) — no `_state` external access
10. **Noise-first**: Noise channels integrated into `run()` pipeline, not bolted on
