# Quantum Algorithms

> Tested with: Quanta SDK v0.8.1

## What You'll Learn

Use Quanta's one-function-call API for Grover search, QAOA optimization, VQE molecular energy, and Shor factoring.

## Prerequisites

- [03 — Simulation](03-simulation.md)

## Grover Search — Find a Needle in a Haystack

Grover's algorithm finds a target in an unsorted database with √N speedup:

```python
from quanta.layer3.search import search

# Find value 5 in a 4-qubit search space (16 items)
result = search(num_bits=4, target=5)
print(f"Found: {result.most_frequent}")
print(f"Shots: {result.shots}")
```

## QAOA — Combinatorial Optimization

Solve Max-Cut and other graph problems:

```python
from quanta.layer3.optimize import optimize

# Max-Cut on a 3-node graph
edges = [(0,1), (1,2), (2,0)]
result = optimize(
    num_bits=3,
    cost=lambda x: sum(1 for i,j in edges if ((x >> i) & 1) != ((x >> j) & 1)),
    layers=2,
)
print(f"Best solution: {result.best_bitstring}")
print(f"Cost: {result.best_cost}")
```

## VQE — Molecular Ground State Energy

Find the ground state energy of a molecule:

```python
from quanta.layer3.vqe import vqe

# H2 molecule Hamiltonian (simplified)
hamiltonian = [
    ("ZZ", -1.05),
    ("XX",  0.39),
    ("YY", -0.39),
    ("ZI", -0.47),
    ("IZ", -0.47),
]

result = vqe(num_qubits=2, hamiltonian=hamiltonian, layers=3)
print(f"Ground state energy: {result.energy:.4f}")
print(f"Iterations: {result.num_iterations}")
```

## Shor's Algorithm — Integer Factoring

Factor integers using quantum period finding:

```python
from quanta.layer3.shor import factor

result = factor(15)
print(f"15 = {result.factors[0]} x {result.factors[1]}")  # 3 x 5
```

## Quantum Machine Learning

Classify data with quantum circuits:

```python
from quanta.layer3.qml import QuantumClassifier
import numpy as np

# Training data (XOR-like pattern)
X_train = np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.1], [0.9, 0.9]])
y_train = np.array([1, 1, 0, 0])

# Train quantum classifier
clf = QuantumClassifier(n_qubits=2, n_layers=2, seed=42)
result = clf.fit(X_train, y_train, epochs=10)

print(f"Training accuracy: {result.accuracy:.0%}")
print(f"Parameters: {result.n_params}")
```

## Monte Carlo — Option Pricing

Price financial derivatives with quantum amplitude estimation:

```python
from quanta.layer3.monte_carlo import quantum_monte_carlo

result = quantum_monte_carlo(
    distribution="lognormal",
    payoff="european_call",
    params={"spot": 100, "strike": 105, "rate": 0.05, "vol": 0.2, "T": 1.0},
    n_qubits=5,
)

print(f"Quantum estimate: {result.estimated_value:.4f}")
print(f"Classical estimate: {result.classical_value:.4f}")
print(f"Grover iterations: {result.grover_iterations}")
```

## Algorithm Comparison

| Algorithm | Classical | Quanta | Speedup |
|-----------|----------|--------|---------|
| Grover | O(N) | `search(num_bits, target)` | √N |
| QAOA | NP-hard | `optimize(num_bits, cost)` | Heuristic |
| VQE | O(4^n) | `vqe(num_qubits, hamiltonian)` | Polynomial |
| Shor | O(e^(n^⅓)) | `factor(N)` | Exponential |
| QML | O(n·d) | `QuantumClassifier.fit()` | Feature space |

## Try It Yourself

1. Use Grover to find `target=7` with `num_bits=4` — verify √N iterations
2. Try VQE with `layers=5` vs `layers=1` — does energy improve?
3. Factor 21 using Shor — what are the factors?

## What's Next

→ [05 — IBM Hardware](05-ibm-hardware.md): Run circuits on real quantum hardware
