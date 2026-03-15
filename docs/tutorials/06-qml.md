# Quantum Machine Learning — Deep Dive

> Tested with: Quanta SDK v0.8.1

## What You'll Learn

Build quantum classifiers, use quantum kernels, and understand feature maps.

## Prerequisites

- [04 — Algorithms](04-algorithms.md)

## QuantumClassifier — Variational Quantum Circuit

The QuantumClassifier uses a parameterized quantum circuit (PQC) trained with parameter-shift gradients:

```python
from quanta.layer3.qml import QuantumClassifier
import numpy as np

# Simple binary classification
X = np.array([[0.1, 0.2], [0.8, 0.9], [0.2, 0.1], [0.9, 0.8]])
y = np.array([0, 1, 0, 1])

clf = QuantumClassifier(n_qubits=2, n_layers=2, seed=42)
result = clf.fit(X, y, epochs=20)
print(f"Accuracy: {result.accuracy:.0%}")
print(f"Parameters: {result.n_params}")
```

## Feature Maps

Feature maps encode classical data into quantum states:

```python
from quanta.layer3.qml import angle_encoding, zz_feature_map
from quanta.simulator.statevector import StateVectorSimulator
import numpy as np

x = np.array([0.5, 0.3])

# Angle encoding: x_i → RY(x_i)
sim_angle = StateVectorSimulator(2)
angle_encoding(sim_angle, x)
print(f"Angle encoding dim: {len(sim_angle.state)}")

# ZZ encoding: entangling feature map
sim_zz = StateVectorSimulator(2)
zz_feature_map(sim_zz, x)
print(f"ZZ encoding dim: {len(sim_zz.state)}")
```

## Quantum Kernel

Compute quantum kernel matrices for kernel-based ML:

```python
from quanta.layer3.qml import QuantumKernel
import numpy as np

X = np.array([[0.1, 0.2], [0.8, 0.9], [0.5, 0.5]])

kernel = QuantumKernel(n_qubits=2)
K = kernel.matrix(X)

print(f"Kernel matrix shape: {K.shape}")  # (3, 3)
print(f"K[0,0] (self-similarity): {K[0,0]:.4f}")  # ~1.0
print(f"K[0,1] (different points): {K[0,1]:.4f}")  # < 1.0
```

## Architecture

```
Classical Data → Feature Map → PQC (trainable) → Measurement → Prediction
    [x₁, x₂]     RY(x_i)     RY(θ_i) + CX       |ψ⟩→P(0)     0 or 1
```

- **Feature Map**: Encodes data into quantum state
- **Ansatz (PQC)**: Trainable rotation + entangling layers
- **Gradient**: Parameter-shift rule (exact, quantum-native)
- **Optimizer**: Gradient descent with configurable learning rate

## QML vs Classical ML

| Feature | Classical (sklearn) | Quanta QML |
|---------|-------------------|------------|
| Feature space | d dimensions | 2^n dimensions |
| Kernel trick | Polynomial, RBF | Quantum kernel |
| Training | SGD, Adam | Parameter-shift |
| Expressibility | Fixed | Exponential |
| Current advantage | Large datasets | Small, structured data |

## Try It Yourself

1. Try `n_layers=4` — does accuracy improve?
2. Compare `AngleFeatureMap` vs `ZZFeatureMap` for the same data
3. Use `QuantumKernel` with 3 features and 3 qubits

## What's Next

→ [07 — Error Correction](07-noise-and-qec.md): Protect your quantum computation
