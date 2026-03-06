# Quanta SDK — Features

## Gate Set (14 + 3 Parametric)

| Gate | Qubits | Description | Matrix Size |
|------|--------|-------------|-------------|
| H | 1 | Hadamard — creates superposition | 2×2 |
| X | 1 | Pauli-X — bit flip (NOT) | 2×2 |
| Y | 1 | Pauli-Y — bit + phase flip | 2×2 |
| Z | 1 | Pauli-Z — phase flip | 2×2 |
| S | 1 | S gate — π/2 phase | 2×2 |
| T | 1 | T gate — π/4 phase | 2×2 |
| CX | 2 | CNOT — controlled NOT | 4×4 |
| CZ | 2 | Controlled-Z — controlled phase | 4×4 |
| CY | 2 | Controlled-Y | 4×4 |
| SWAP | 2 | Qubit exchange | 4×4 |
| CCX | 3 | Toffoli — double controlled NOT | 8×8 |
| RX(θ) | 1 | X-axis rotation | 2×2 |
| RY(θ) | 1 | Y-axis rotation | 2×2 |
| RZ(θ) | 1 | Z-axis rotation | 2×2 |

## Broadcast Support

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

## Supported Hardware Gate Sets

| Hardware | Gate Set |
|----------|----------|
| IBM Heron | {CX, RZ, SX, X} |
| Google Sycamore | {CZ, RZ, RX, RY} |
| Quantinuum H-Series | {CX, RZ, RY, RX} |

## Noise Models

| Channel | Description | Parameter |
|---------|-------------|-----------|
| Depolarizing | Random Pauli error | p ∈ [0,1] |
| BitFlip | |0⟩↔|1⟩ flip | p ∈ [0,1] |
| PhaseFlip | Phase error (Z) | p ∈ [0,1] |
| AmplitudeDamping | Energy loss (T1) | γ ∈ [0,1] |

## Error Correction Codes

| Code | Notation | Correctable Errors |
|------|----------|-------------------|
| BitFlip | [[3,1,1]] | 1 bit-flip |
| PhaseFlip | [[3,1,1]] | 1 phase-flip |
| Steane | [[7,1,3]] | 1 arbitrary single-qubit error |

## Layer 3 — Declarative API

### search(num_bits, target, shots)
- Automatically applies Grover's algorithm
- Computes optimal iteration count: π/4·√(N/M)
- Target: int or lambda function
- Success rate: 96%+ (at 4 qubits)

### optimize(num_bits, cost, minimize, layers)
- Automatically applies QAOA algorithm
- Grid search parameter optimization
- Minimize or maximize
- 50 random trials to find best parameters

### MultiAgentSystem
- Agent = qubit (in superposition)
- Interaction = entanglement (strength: 0-1)
- Decision = measurement (collapse)
- Output: marginal probabilities, correlations

## Visualization

- **ASCII circuit diagram**: `draw(circuit)`
- **Probability histogram**: `show_probabilities(result)`
- **Statevector table**: `show_statevector(sv, n)`
- **Phase diagram**: `show_phases(sv, n)`

## OpenQASM 3.0 Support

```python
from quanta.export.qasm import to_qasm
print(to_qasm(bell_state))
# OPENQASM 3.0;
# include "stdgates.inc";
# qubit[2] q;
# h q[0];
# cx q[0], q[1];
```
