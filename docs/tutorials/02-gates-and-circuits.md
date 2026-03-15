# Gates and Circuits

> Tested with: Quanta SDK v0.8.1

## What You'll Learn

Master all 25 quantum gates, build multi-qubit circuits, use parametric and custom gates.

## Prerequisites

- [01 — Getting Started](01-getting-started.md)

## The 25 Gates

### Single-Qubit Gates (18)

```python
from quanta import (
    circuit, measure, run,
    H, X, Y, Z,        # Pauli + Hadamard
    S, T, SDG, TDG,     # Phase gates
    I, SX, SXdg,        # Identity, sqrt-X
    RX, RY, RZ, P, U,   # Parametric
)
import math

@circuit(qubits=1)
def demo_single(q):
    # Pauli gates — bit/phase flips
    X(q[0])              # |0⟩ → |1⟩ (bit flip)
    Y(q[0])              # Bit + phase flip
    Z(q[0])              # Phase flip only

    # Hadamard — superposition
    H(q[0])              # |0⟩ → (|0⟩+|1⟩)/√2

    # Phase gates
    S(q[0])              # π/2 phase
    T(q[0])              # π/4 phase
    SDG(q[0])            # −π/2 phase (S†)
    TDG(q[0])            # −π/4 phase (T†)

    # Parametric gates — angle as argument
    RX(math.pi/4)(q[0])  # X-axis rotation
    RY(math.pi/3)(q[0])  # Y-axis rotation
    RZ(math.pi/6)(q[0])  # Z-axis rotation
    P(math.pi/2)(q[0])   # Phase gate
    U(math.pi/4, 0, math.pi)(q[0])  # Universal gate (θ, φ, λ)

    return measure(q)
```

### Multi-Qubit Gates (7)

```python
from quanta import circuit, measure, run, H, CX, CZ, CY, SWAP, CCX, RXX, RZZ
import math

@circuit(qubits=4)
def demo_multi(q):
    H(q[0])

    # 2-qubit gates
    CX(q[0], q[1])       # CNOT — controlled NOT
    CZ(q[0], q[1])       # Controlled-Z
    CY(q[0], q[1])       # Controlled-Y
    SWAP(q[0], q[1])     # Swap two qubits

    # Parametric 2-qubit gates
    RXX(math.pi/4)(q[0], q[1])  # XX rotation
    RZZ(math.pi/4)(q[0], q[1])  # ZZ rotation

    # 3-qubit gate
    CCX(q[0], q[1], q[2])  # Toffoli (AND gate)

    return measure(q)
```

## Broadcast — Apply to All Qubits

```python
from quanta import circuit, H, measure, run

@circuit(qubits=4)
def broadcast_demo(q):
    H(q)                  # H on ALL 4 qubits at once
    return measure(q)

result = run(broadcast_demo, shots=1024)
print(f"States: {len(result.counts)}")
# All 16 basis states roughly equal probability
```

## Custom Gates

Define your own unitary:

```python
from quanta import custom_gate, circuit, measure, run, X
import numpy as np

# Define a √X gate
custom_gate("SqrtX", np.array([
    [0.5+0.5j, 0.5-0.5j],
    [0.5-0.5j, 0.5+0.5j]
]))

# Use built-in X gate to verify (SqrtX^2 = X)
@circuit(qubits=1)
def custom_demo(q):
    X(q[0])  # Flip to |1⟩
    return measure(q)

result = run(custom_demo, shots=100)
print(result.most_frequent)  # '1'
```

## Circuit Inspection

```python
from quanta import circuit, H, CX, measure
from quanta.export.qasm import to_qasm
from quanta.dag.dag_circuit import DAGCircuit

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

# Export to OpenQASM 3.0
print(to_qasm(bell))
# OPENQASM 3.0;
# include "stdgates.inc";
# qubit[2] q;
# ...

# Inspect as DAG
dag = DAGCircuit.from_builder(bell.build())
print(f"Gates: {dag.gate_count()}, Depth: {dag.depth()}")
```

## Try It Yourself

1. Build a circuit that puts qubit 0 in state |1⟩ using only `RY`
2. Create a SWAP gate using only 3 CX gates
3. Export your circuit to QASM and inspect the output

## What's Next

→ [03 — Simulation](03-simulation.md): Noise models, density matrix, fidelity
