# Gates and Circuits

> Tested with: Quanta SDK v1.3.0

## What You'll Learn

Master all 34 quantum gates (16 single-qubit + 18 multi-qubit), build multi-qubit circuits, and use parametric and custom gates.

## Prerequisites

- [01 — Getting Started](01-getting-started.md)

## The 34 Gates

### Single-Qubit Gates (16)

```python
from quanta import (
    circuit, measure, run,
    H, X, Y, Z,        # Pauli + Hadamard (4)
    S, T, SDG, TDG,    # Phase gates (4)
    I, SX, SXdg,       # Identity, sqrt-X, sqrt-X† (3)
    RX, RY, RZ, P, U,  # Parametric single-qubit gates (5)
)
import math

@circuit(qubits=1)
def demo_single(q):
    # Pauli gates — bit/phase flips
    X(q[0])              # |0⟩ → |1⟩ (bit flip)
    Y(q[0])              # Bit + phase flip
    Z(q[0])              # Phase flip only
    I(q[0])              # Identity (idle cycle)

    # Hadamard & sqrt-X — superposition
    H(q[0])              # |0⟩ → (|0⟩+|1⟩)/√2
    SX(q[0])             # √X (IBM Heron native)
    SXdg(q[0])           # √X† inverse

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

result = run(demo_single, shots=256, seed=42)
print(f"Single-qubit demo executed {result.gate_count} gates")
```

### Multi-Qubit Gates (18)

```python
from quanta import (
    circuit, measure, run, H,
    CX, CZ, CY, SWAP,        # Standard 2-qubit gates (4)
    ECR, CH,                 # Hardware-native & controlled-H (2)
    CP, RXX, RZZ, MS,        # Parametric 2-qubit gates (4)
    CCX, CSWAP, RCCX, RC3X,  # 3-qubit & 4-qubit multi-controlled gates (4)
)
from quanta.core.gates import CRX, CRY, CRZ, iSWAP  # Controlled rotations & iSWAP (4)
import math

@circuit(qubits=4)
def demo_multi(q):
    H(q[0])

    # Fixed 2-qubit gates
    CX(q[0], q[1])              # CNOT — controlled NOT
    CZ(q[0], q[1])              # Controlled-Z (IBM Heron native)
    CY(q[0], q[1])              # Controlled-Y
    SWAP(q[0], q[1])            # Swap two qubits
    ECR(q[0], q[1])             # Echoed cross-resonance (IBM Eagle native)
    iSWAP(q[0], q[1])           # Imaginary SWAP (Google superconducting native)
    CH(q[0], q[1])              # Controlled-Hadamard

    # Parametric 2-qubit gates
    CP(math.pi/4)(q[0], q[1])   # Controlled-Phase
    RXX(math.pi/4)(q[0], q[1])  # Ising XX coupling
    RZZ(math.pi/4)(q[0], q[1])  # Ising ZZ coupling
    MS(math.pi/4)(q[0], q[1])   # Mølmer-Sørensen (IonQ trapped-ion native)
    CRX(math.pi/4)(q[0], q[1])  # Controlled-RX
    CRY(math.pi/4)(q[0], q[1])  # Controlled-RY
    CRZ(math.pi/4)(q[0], q[1])  # Controlled-RZ

    # 3-qubit and 4-qubit multi-controlled gates
    CCX(q[0], q[1], q[2])       # Toffoli (CCNOT)
    CSWAP(q[0], q[1], q[2])     # Fredkin (controlled-SWAP)
    RCCX(q[0], q[1], q[2])      # Relative-phase Toffoli
    RC3X(q[0], q[1], q[2], q[3])  # 3-controlled X (4 qubits)

    return measure(q)

result = run(demo_multi, shots=256, seed=42)
print(f"Multi-qubit demo executed {result.gate_count} gates")
```

## Broadcast — Apply to All Qubits

```python
from quanta import circuit, H, measure, run

@circuit(qubits=4)
def broadcast_demo(q):
    H(q)                  # H on ALL 4 qubits at once
    return measure(q)

result = run(broadcast_demo, shots=1024, seed=42)
print(f"States: {len(result.counts)}")
# All 16 basis states roughly equal probability
```

## Custom Gates

Define your own unitary and apply it directly inside `@circuit`:

```python
from quanta import custom_gate, circuit, measure, run
import numpy as np

# Define a √X gate
SqrtX = custom_gate("SqrtX", np.array([
    [0.5+0.5j, 0.5-0.5j],
    [0.5-0.5j, 0.5+0.5j],
]))

# Apply SqrtX twice to verify SqrtX @ SqrtX = X
@circuit(qubits=1)
def custom_demo(q):
    SqrtX(q[0])
    SqrtX(q[0])  # Two √X applications flip |0⟩ → |1⟩
    return measure(q)

result = run(custom_demo, shots=100, seed=42)
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

1. Build a circuit that puts qubit 0 in state `|1⟩` using only `RY`
2. Create a `SWAP` gate using only 3 `CX` gates
3. Export your circuit to QASM and inspect the output

## What's Next

→ [03 — Simulation](03-simulation.md): Simulators, 7 noise channels, density matrix, fidelity
