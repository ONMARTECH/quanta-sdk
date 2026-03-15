# Quantum Classification — Quick Recipe

> Train a quantum classifier on a small dataset.

```python
from quanta.layer3.qml import QuantumClassifier
import numpy as np

# Training data
X_train = np.array([
    [0.1, 0.9],  # Class 1
    [0.9, 0.1],  # Class 1
    [0.1, 0.1],  # Class 0
    [0.9, 0.9],  # Class 0
])
y_train = np.array([1, 1, 0, 0])

# Train
clf = QuantumClassifier(n_qubits=2, n_layers=2, seed=42)
result = clf.fit(X_train, y_train, epochs=20)
print(f"Accuracy: {result.accuracy:.0%}")

# Predict
preds = clf.predict(np.array([[0.5, 0.8], [0.8, 0.5]]))
print(f"Predictions: {preds}")
```
