# Quantum Algorithms

> Tested with: Quanta SDK v1.3.0

## What You'll Learn

Use Quanta's one-function-call API for Grover search, QAOA optimization, VQE molecular energy, Shor factoring, QML, and Quantum Monte Carlo.

## Prerequisites

- [03 — Simulation](03-simulation.md)

## Grover Search — Find a Needle in a Haystack

Grover's algorithm finds a target in an unsorted database with $\sqrt{N}$ speedup:

```python
from quanta.layer3.search import search

# Find value 5 ('0101') in a 4-qubit search space (16 items)
result = search(num_bits=4, target=5, seed=42)
print(f"Found: {result.most_frequent}")
print(f"Oracle queries: {result.oracle_queries}, Shots: {result.shots}")
```

## QAOA — Combinatorial Optimization

Solve Max-Cut and other graph problems (setting `minimize=False` to maximize cut edges):

```python
from quanta.layer3.optimize import optimize

# Max-Cut on a 3-node triangle graph (max cut = 2 edges)
edges = [(0, 1), (1, 2), (2, 0)]
result = optimize(
    num_bits=3,
    cost=lambda x: sum(1 for i, j in edges if ((x >> i) & 1) != ((x >> j) & 1)),
    minimize=False,
    layers=2,
    seed=42,
)
print(f"Best solution: {result.best_bitstring}")
print(f"Cut edges: {result.best_cost}")
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

result = vqe(num_qubits=2, hamiltonian=hamiltonian, layers=3, seed=42)
print(f"Ground state energy: {result.energy:.4f}")
print(f"Iterations: {result.num_iterations}")
```

## Shor's Algorithm — Integer Factoring

Factor integers using genuine semiclassical quantum period finding:

```python
from quanta.layer3.shor import factor

result = factor(15, seed=42)
print(f"15 = {result.factors[0]} x {result.factors[1]}")  # 3 x 5
```

## Quantum Machine Learning

Classify non-linearly separable XOR data with an entangling ZZ quantum circuit:

```python
from quanta.layer3.qml import QuantumClassifier
import numpy as np

# Training data (XOR-like pattern)
X_train = np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.1], [0.9, 0.9]])
y_train = np.array([1, 1, 0, 0])

# Train quantum classifier with ZZ feature map and Adam
clf = QuantumClassifier(
    n_qubits=2,
    n_layers=2,
    feature_map="zz",
    optimizer="adam",
    learning_rate=0.2,
    seed=42,
)
result = clf.fit(X_train, y_train, epochs=30)

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
    params={"S0": 100, "K": 105, "r": 0.05, "sigma": 0.2, "T": 1.0},
    n_qubits=5,
    seed=42,
)

print(f"Quantum estimate: {result.estimated_value:.4f}")
print(f"Classical estimate: {result.classical_value:.4f}")
print(f"Grover iterations: {result.grover_iterations}")
```

## Algorithm Comparison

| Algorithm | Classical | Quanta | Speedup |
|-----------|----------|--------|---------|
| Grover | $O(N)$ | `search(num_bits, target)` | $\sqrt{N}$ |
| QAOA | NP-hard | `optimize(num_bits, cost, minimize=...)` | Variational heuristic |
| VQE | $O(4^n)$ | `vqe(num_qubits, hamiltonian)` | Polynomial state prep |
| Shor | Sub-exponential | `factor(N)` | Super-polynomial |
| QML | $O(n \cdot d)$ | `QuantumClassifier.fit()` | Entangled feature space |

## Try It Yourself

1. Use Grover to find `target=7` with `num_bits=4` — verify $\sqrt{N}$ iterations
2. Try VQE with `layers=5` vs `layers=1` — does energy improve?
3. Factor 21 using Shor — what are the factors?

## What's Next

→ [05 — IBM Hardware](05-ibm-hardware.md): Run circuits on real quantum hardware
