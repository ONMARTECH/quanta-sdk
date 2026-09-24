# PyTorch QuantumLayer & Continuous Resonance — Quick Recipe

> Build end-to-end differentiable hybrid quantum neural networks with PyTorch and Quanta's exact autograd.

---

## 1. Differentiable PyTorch QuantumLayer

`quanta.torch.QuantumLayer` integrates parameterized quantum circuits directly into PyTorch's `nn.Module` computational graph.

### Python Code

```python
import torch
import torch.nn as nn
from quanta.torch import QuantumLayer

# 1. Define a hybrid neural network
class HybridQuantumClassifier(nn.Module):
    def __init__(self, num_qubits: int = 4):
        super().__init__()
        self.pre_net = nn.Linear(8, num_qubits)
        self.quantum_layer = QuantumLayer(
            num_qubits=num_qubits,
            depth=2,
            ansatz="real_amplitudes",
        )
        self.post_net = nn.Linear(num_qubits, 2)

    def forward(self, x):
        features = torch.tanh(self.pre_net(x))
        quantum_out = self.quantum_layer(features)
        logits = self.post_net(quantum_out)
        return logits

# 2. Forward pass with autograd
model = HybridQuantumClassifier(num_qubits=4)
x = torch.randn(16, 8)  # Batch of 16 samples
logits = model(x)
print(f"Logits Shape: {logits.shape}")  # [16, 2]

# 3. Backward pass (computes quantum gradients automatically)
loss = logits.sum()
loss.backward()
print("Quantum Layer Gradient Norm:", model.quantum_layer.weights.grad.norm().item())
```

---

## 2. Continuous-Time Resonant Layer (`ContinuousResonantLayer`)

For continuous-variable quantum dynamics governed by Hamiltonian ODEs:
$$\frac{d|\psi(t)\rangle}{dt} = -i \hat{H}(\theta, t)|\psi(t)\rangle$$

```python
from quanta.torch import ContinuousResonantLayer

# Initialize continuous resonant layer with Daleckii-Krein matrix exponential autograd
resonant_layer = ContinuousResonantLayer(
    num_qubits=3,
    evolution_time=1.0,
    num_steps=50,
)

inputs = torch.randn(8, 3)
output = resonant_layer(inputs)
print(f"Continuous Resonance Output: {output.shape}")
```

---

## See Also
- [2026 Scientific Audit Report](../scientific_audit_september_2026.md)
- [API Reference — PyTorch](../api/torch.md)
