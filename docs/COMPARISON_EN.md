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
| **Noise Model** | 7 channels | Extensive | Extensive | Plugin |
| **QEC Codes** | 6 codes (surface + color) | External | External | No |
| **QASM Import/Export** | 2.0 + 3.0 | 2.0/3.0 | 2.0 | No |
| **Multi-Agent** | Yes | No | No | No |
| **VQE** | Built-in | qiskit-nature | Via cirq-core | Built-in |
| **Shor** | Built-in | External | No | No |
| **Entity Resolution** | Built-in (QAOA) | No | No | No |
| **Dependencies** | 1 (numpy) | 20+ | 10+ | 10+ |
| **MCP Server** | Built-in (16 tools) | No | No | No |
| **Gradients** | Parameter-shift + Natural | Manual | Manual | **Built-in (autograd)** |

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
| Tests | 457 | 5000+ |
| Max qubits (sim) | 27 | 32 |

## Differentiable Quantum Computing

PennyLane's key advantage is differentiable programming with autograd.
Quanta now provides comparable gradient support:

| Feature | Quanta | PennyLane |
|---------|--------|-----------|
| **Parameter-shift rule** | `parameter_shift()` | `qml.gradients.param_shift` |
| **Finite differences** | `finite_diff()` | `qml.gradients.finite_diff` |
| **Natural gradient** | `natural_gradient()` (QFIM) | `qml.QNGOptimizer` |
| **Expectation values** | `expectation()` | `qml.expval()` |
| **Backprop (autograd)** | Not yet | **Yes (JAX/Torch/TF)** |
| **Framework integration** | NumPy-native | JAX, PyTorch, TensorFlow |

### Gradient Example Comparison

**Quanta (4 lines)**
```python
from quanta.gradients import parameter_shift, expectation
from quanta.simulator.statevector import StateVectorSimulator

def cost(params):
    sim = StateVectorSimulator(1)
    sim.apply("RY", (0,), (params[0],))
    return expectation(sim.state, "Z", 1)

result = parameter_shift(cost, [0.5])
print(result.gradients)  # exact: [-sin(0.5)]
```

**PennyLane (6 lines)**
```python
import pennylane as qml

dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev, diff_method="parameter-shift")
def cost(theta):
    qml.RY(theta, wires=0)
    return qml.expval(qml.PauliZ(0))

grad_fn = qml.grad(cost)
print(grad_fn(0.5))  # exact: [-sin(0.5)]
```

### Quanta's Advantage
- **Zero dependencies**: Gradients work with just NumPy
- **Explicit control**: Choose method per-call, not per-device
- **QFIM built-in**: Natural gradient with Fubini-Study metric
- **MCP integration**: AI assistants can compute gradients remotely

### PennyLane's Advantage
- **Autograd backprop**: True reverse-mode AD through circuits
- **Framework bridges**: Native JAX/PyTorch/TensorFlow support
- **Larger ecosystem**: More optimizers, more devices

