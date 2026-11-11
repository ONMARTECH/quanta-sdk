# Quantum Classification — Quick Recipe

> Train a variational quantum classifier on a non-linearly separable XOR dataset.

## What It Does

A **Quantum Classifier** uses parameterized quantum circuits to classify data.
By configuring `feature_map="zz"`, it encodes classical features into entangled
quantum states via pairwise ZZ interactions, then optimizes rotation angles to
separate classes — learning non-linear decision boundaries via quantum interference.

## Code

```python
from quanta.layer3.qml import QuantumClassifier
import numpy as np

# XOR-like dataset (not linearly separable!)
X_train = np.array([
    [0.1, 0.9],  # Class 1
    [0.9, 0.1],  # Class 1
    [0.1, 0.1],  # Class 0
    [0.9, 0.9],  # Class 0
])
y_train = np.array([1, 1, 0, 0])

# Train with entangling ZZ feature map and Adam optimizer
clf = QuantumClassifier(
    n_qubits=2,
    n_layers=2,
    feature_map="zz",
    optimizer="adam",
    learning_rate=0.2,
    seed=42,
)
result = clf.fit(X_train, y_train, epochs=40)
print(f"Training accuracy: {result.accuracy:.0%}")

# Predict on new Class-1 points in the XOR quadrants
X_test = np.array([[0.2, 0.8], [0.8, 0.2]])
preds = clf.predict(X_test)
print(f"Predictions: {preds}")
```

## Expected Output

```
Training accuracy: 100%
Predictions: [1 1]
```

> **Key insight:** This XOR pattern is not linearly separable — a single
> classical perceptron cannot solve it. The `"zz"` quantum feature map projects
> pairwise interactions into Hilbert space where the classes become separable.

## Parameters Explained

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `n_qubits` | Number of qubits (= feature dimensions) | 2–8 |
| `n_layers` | Circuit depth (more layers = more expressive) | 1–5 |
| `feature_map` | Data encoding strategy | `"angle"` (default), `"zz"`, `"amplitude"` |
| `optimizer` | Parameter update rule | `"sgd"` (default), `"adam"` |
| `epochs` | Training iterations | 10–50 |
| `seed` | Random seed for reproducibility | Any integer |

## Try Next

- **Larger datasets**: Use sklearn's `make_moons` or `make_circles`
- **More qubits**: Encode 4+ features with `n_qubits=4`
- **Deeper circuits**: Try `n_layers=4` for complex boundaries
- **Real-world data**: See [Tutorial 09c — QML Fraud Detection](../tutorials/09c-qml-fraud.md)

## See Also

- [Domain Packs — Marketing/CRM](../domain-packs.md)
- [Tutorial 06 — Quantum ML](../tutorials/06-qml.md)
