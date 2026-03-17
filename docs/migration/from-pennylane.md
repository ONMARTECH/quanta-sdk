# Coming from PennyLane

> Tested with: Quanta SDK v0.8.1

## Why Switch?

| Feature | PennyLane | Quanta |
|---------|-----------|--------|
| Install size | ~200 MB | ~5 MB |
| Dependencies | TensorFlow/JAX/Torch optional | NumPy only |
| IBM Hardware | Via plugin | Built-in REST API |
| QEC | ❌ | ✅ (5 codes, 2 decoders) |
| MCP (AI tools) | ❌ | ✅ (16 tools) |
| QASM 3.0 | ❌ | ✅ (export + import) |

## Side-by-Side: 10 Common Patterns

### 1. Define a Circuit

```python
# ── PennyLane ──
# import pennylane as qml
# dev = qml.device("default.qubit", wires=2)
# @qml.qnode(dev)
# def bell(params):
#     qml.Hadamard(wires=0)
#     qml.CNOT(wires=[0, 1])
#     return qml.probs(wires=[0, 1])

# ── Quanta ──
from quanta import circuit, H, CX, measure, run

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell, shots=1024)
print(result.counts)
```

### 2. Parametric Gates

```python
# ── PennyLane ──
# qml.RY(theta, wires=0)
# qml.RZ(phi, wires=1)

# ── Quanta ──
from quanta import circuit, RY, RZ, measure, run
import math

@circuit(qubits=2)
def parametric(q):
    RY(math.pi / 4)(q[0])
    RZ(math.pi / 3)(q[1])
    return measure(q)

result = run(parametric, shots=1024)
print(result.most_frequent)
```

### 3. Gradients

```python
# ── PennyLane ──
# grad_fn = qml.grad(cost)
# gradients = grad_fn(params)

# ── Quanta ──
from quanta.gradients import parameter_shift
import numpy as np

def cost_fn(params):
    """Simple cost function for gradient demo."""
    return float(np.sin(params[0]) * np.cos(params[1]))

result = parameter_shift(cost_fn, np.array([0.5, 0.3]))
print(f"Gradient: {result.gradients}")
```

### 4. QML Classification

```python
# ── PennyLane ──
# @qml.qnode(dev, diff_method="parameter-shift")
# def classifier(x, params):
#     qml.AngleEmbedding(x, wires=range(2))
#     qml.StronglyEntanglingLayers(params, wires=range(2))
#     return qml.expval(qml.PauliZ(0))

# ── Quanta ──
from quanta.layer3.qml import QuantumClassifier
import numpy as np

X = np.array([[0.1, 0.2], [0.8, 0.9], [0.2, 0.1], [0.9, 0.8]])
y = np.array([0, 1, 0, 1])

clf = QuantumClassifier(n_qubits=2, n_layers=2, seed=42)
result = clf.fit(X, y, epochs=20)
print(f"Accuracy: {result.accuracy:.0%}")
```

### 5. Quantum Kernel

```python
# ── PennyLane ──
# kernel = qml.kernels.EmbeddingKernel(ansatz, dev)
# K = qml.kernels.kernel_matrix(X, X, kernel)

# ── Quanta ──
from quanta.layer3.qml import QuantumKernel
import numpy as np

X = np.array([[0.2, 0.3], [0.7, 0.8], [0.5, 0.5]])
kernel = QuantumKernel(n_qubits=2)
K = kernel.matrix(X)
print(f"Kernel matrix:\n{K.round(3)}")
```

### 6. Feature Maps

```python
# ── PennyLane ──
# qml.AngleEmbedding(x, wires=range(n))
# qml.IQPEmbedding(x, wires=range(n))

# ── Quanta ──
from quanta.layer3.qml import angle_encoding, zz_feature_map
from quanta.simulator.statevector import StateVectorSimulator
import numpy as np

x = np.array([0.4, 0.6])

sim = StateVectorSimulator(2)
angle_encoding(sim, x)
print(f"Angle-encoded state: {len(sim.state)} amplitudes")

sim2 = StateVectorSimulator(2)
zz_feature_map(sim2, x)
print(f"ZZ-encoded state: {len(sim2.state)} amplitudes")
```

### 7. Noise Simulation

```python
# ── PennyLane ──
# dev = qml.device("default.mixed", wires=2)
# qml.DepolarizingChannel(0.01, wires=0)

# ── Quanta ──
from quanta import circuit, H, CX, measure, run
from quanta.simulator.noise import NoiseModel, Depolarizing

noise = NoiseModel().add(Depolarizing(probability=0.01))

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell, shots=1024, noise=noise)
print(f"States: {result.counts}")
```

## Key Differences

| PennyLane | Quanta |
|-----------|--------|
| `qml.device(...)` | Built-in (no device setup) |
| `@qml.qnode(dev)` | `@circuit(qubits=N)` |
| `qml.RY(angle, wires=0)` | `RY(angle)(q[0])` |
| `qml.expval(qml.PauliZ(0))` | `result.counts` |
| `qml.grad(cost)` | `parameter_shift(fn, params)` |
| `qml.StronglyEntanglingLayers` | `QuantumClassifier(n_layers=N)` |

## Getting Help

- [Tutorials](../tutorials/01-getting-started.md) — 8-part learning path
- [Features](../FEATURES_EN.md) — Complete feature reference
- [GitHub](https://github.com/ONMARTECH/quanta-sdk) — Issues & discussions
