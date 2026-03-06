# Quanta SDK — Architecture Documentation

## Overview

Quanta is designed with a **3-layer independent architecture**. Each layer can be used independently from the one above it.

## Layer Diagram

```
┌─────────────────────────────────────────────────────┐
│              LAYER 3: DECLARATIVE API               │
│  search()  │  optimize()  │  MultiAgentSystem       │
│  "What do you want?" — NO gate knowledge needed      │
├─────────────────────────────────────────────────────┤
│              LAYER 2: ALGORITHMIC DSL               │
│  @circuit  │  H/CX/RZ  │  measure()  │  run()      │
│  "How to build the circuit?"                         │
├─────────────────────────────────────────────────────┤
│              LAYER 1: PHYSICAL ENGINE               │
│  DAG  │  Compiler  │  Simulator  │  QEC  │  Export  │
│  "How will it run on hardware?"                      │
└─────────────────────────────────────────────────────┘
```

## Dependency Graph

```
layer3/ ──→ simulator/ ──→ core/
                │
runner.py ──→ dag/ ──→ core/
                │
compiler/ ──→ dag/ ──→ core/
                │
backends/ ──→ simulator/ ──→ core/
                │
export/ ──→ dag/ ──→ core/
                │
qec/ ──→ core/
```

**Rule**: Dependencies always flow downward. No lower layer depends on an upper layer.

## Module Details

### core/ — Fundamental Building Blocks

| File | Lines | Responsibility |
|------|-------|----------------|
| `types.py` | 164 | QubitRef, Instruction, QubitRegister, error classes |
| `gates.py` | 321 | 14 standard gates + 3 parametric (RX/RY/RZ) + broadcast |
| `circuit.py` | 174 | @circuit decorator, CircuitBuilder, CircuitDefinition |
| `measure.py` | 66 | Flexible measurement (full, partial, single) |
| `equivalence.py` | 153 | Unitary comparison, circuit fidelity |

### dag/ — Directed Acyclic Graph Engine

| File | Lines | Responsibility |
|------|-------|----------------|
| `node.py` | 78 | InputNode, OpNode, OutputNode (immutable) |
| `dag_circuit.py` | 227 | Topological sort (Kahn's), depth, parallel layers |

### compiler/ — Optimization Pipeline

| File | Lines | Responsibility |
|------|-------|----------------|
| `pipeline.py` | 137 | CompilerPass Protocol, chaining pipeline, statistics |
| `passes/optimize.py` | 232 | CancelInverses (H·H=I), MergeRotations (RZ(a)+RZ(b)) |
| `passes/translate.py` | 180 | IBM/Google/Quantinuum gate set transpilation |

### simulator/ — Simulation Engine

| File | Lines | Responsibility |
|------|-------|----------------|
| `statevector.py` | 233 | NumPy full statevector, Kronecker expansion |
| `noise.py` | 236 | 4 noise channels: Depolarizing, BitFlip, PhaseFlip, AmplDamp |

### layer3/ — Declarative API

| File | Lines | Responsibility |
|------|-------|----------------|
| `search.py` | 152 | Auto Grover — give target, it finds |
| `optimize.py` | 194 | QAOA-based — give cost function, it optimizes |
| `agent.py` | 265 | Quantum decision modeling — agents, interaction, correlation |

### Support Modules

| File | Lines | Responsibility |
|------|-------|----------------|
| `runner.py` | 129 | 6-stage orchestrator: build→DAG→compile→sim→sample→result |
| `result.py` | 89 | Measurement results, probabilities, summary |
| `visualize.py` | 114 | ASCII circuit diagram |
| `visualize_state.py` | 166 | Probability histogram, phase diagram |
| `export/qasm.py` | 162 | OpenQASM 3.0 export + parser |
| `qec/codes.py` | 226 | BitFlip [[3,1,1]], Steane [[7,1,3]] |
| `backends/base.py` | 68 | Backend abstract class |
| `backends/local.py` | 94 | Local NumPy simulator backend |

## Data Flow

```
User Code               SDK Internals
    │                       │
@circuit(qubits=N) ──→ CircuitDefinition
    │                       │
H(q[0]), CX(...)    ──→ CircuitBuilder (lazy Instruction list)
    │                       │
measure(q)          ──→ MeasureSpec
    │                       │
run(circuit)        ──→ ┌─ DAGCircuit.from_builder()
                        ├─ CompilerPipeline.run(dag)
                        ├─ StateVectorSimulator.apply(ops)
                        ├─ simulator.sample(shots)
                        └─ Result(counts, probs, statevector)
```

## Design Decisions

1. **Lazy Evaluation**: Gates are not applied immediately; recorded as Instructions
2. **DAG Representation**: Essential for parallelism detection and optimization
3. **Protocol-based**: CompilerPass is a Protocol — duck typing is sufficient
4. **Immutable**: QubitRef, Instruction, nodes are frozen dataclasses
5. **Thread-local Builder**: Multiple circuits can be built concurrently
