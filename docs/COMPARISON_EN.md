# Quanta SDK -- Comparison

## Quanta vs Existing SDKs

### Feature Comparison

| Feature | Quanta | Qiskit | Cirq | PennyLane |
|---------|--------|--------|------|-----------|
| **Language** | Python | Python | Python | Python |
| **Learning Curve** | Easy | Hard | Medium | Medium |
| **Declarative API** | Yes (Layer 3) | No | No | No |
| **No-Gate Usage** | Yes | No | No | No |
| **Broadcast** | `H(q)` | Manual | Manual | Partial |
| **@circuit Decorator** | Yes | No | No | `@qml.qnode` |
| **DAG Representation** | Built-in | Built-in | Moments | No |
| **Compiler Pipeline** | 3-pass + routing | PassManager | Optimizer | Limited |
| **Noise Model** | 4 channels | Extensive | Extensive | Plugin |
| **QEC Codes** | 4 codes (incl. surface) | External | External | No |
| **QASM Import/Export** | 2.0 + 3.0 | 2.0/3.0 | 2.0 | No |
| **Multi-Agent** | Yes | No | No | No |
| **VQE** | Built-in | qiskit-nature | Via cirq-core | Built-in |
| **Shor** | Built-in | External | No | No |
| **Entity Resolution** | Built-in (QAOA) | No | No | No |
| **Dependencies** | 1 (numpy) | 20+ | 10+ | 10+ |

### Code Comparison: Bell State

**Quanta (5 lines)**
```python
@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)
```

**Qiskit (10 lines)**
```python
from qiskit import QuantumCircuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
from qiskit_aer import AerSimulator
simulator = AerSimulator()
result = simulator.run(qc, shots=1024).result()
counts = result.get_counts()
```

**Cirq (12 lines)**
```python
import cirq
q = cirq.LineQubit.range(2)
circuit = cirq.Circuit([
    cirq.H(q[0]),
    cirq.CNOT(q[0], q[1]),
    cirq.measure(*q, key='result')
])
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1024)
counts = result.histogram(key='result')
```

### Search Comparison

**Quanta Layer 3 (1 line)**
```python
result = search(num_bits=4, target=13, shots=1024)
```

**Qiskit (30+ lines)**
```python
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit.algorithms import AmplificationProblem, Grover
# Define oracle, define problem, set up Grover, run...
```

### VQE Comparison

**Quanta (3 lines)**
```python
from quanta.layer3.vqe import vqe
result = vqe(2, hamiltonian=[("ZZ", 1.0), ("XI", 0.5)], layers=3)
print(result.energy)
```

**Qiskit (20+ lines)**
```python
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.circuit.library import EfficientSU2
# Set up mapper, ansatz, optimizer, VQE, run...
```

## Quanta's Differentiators

### 1. 3-Layer Abstraction
- **Layer 3**: Use quantum without knowing gates
- **Layer 2**: Standard circuit programming
- **Layer 1**: Hardware optimization

### 2. Real-World Use Cases
- Entity resolution (customer deduplication)
- Portfolio optimization (financial)
- Molecular simulation (H2, LiH, HeH+)

### 3. Multi-Agent Decision Modeling
Quantum mechanics applied to decision theory.
Superposition = choices, Entanglement = interaction, Measurement = decision.

### 4. Minimal Dependencies
NumPy only. No 200MB install, no Java, no Rust toolchain.

## Numerical Comparison

| Metric | Quanta | Qiskit |
|--------|--------|--------|
| Bell State code | 5 lines | 10 lines |
| Grover search | 1 line (L3) | 30+ lines |
| `pip install` size | ~1 MB | ~200 MB |
| Dependencies | 1 (numpy) | 20+ |
| Tests | 150+ | 5000+ |
| Max qubits (sim) | 27 | 32 |
