# Domain Use-Case Packs

Quanta SDK includes production-ready quantum algorithms for specific business domains.
Each pack combines quantum modules, MCP tools, and documentation.

---

## 🏦 Finance Pack

Quantum algorithms for quantitative finance and risk analysis.

### Modules

| Module | Description | MCP Tool |
|--------|-------------|----------|
| `layer3/monte_carlo.py` | Iterative Quantum Amplitude Estimation (IQAE) option pricing & CRN Greeks | `monte_carlo_price`, `option_greeks` |
| `layer3/finance.py` | Portfolio optimization via Markowitz + QAOA | `portfolio_optimize` |

### Quick Start

```python
# Option Pricing & Greeks — Quantum Monte Carlo (IQAE + CRN)
from quanta.layer3.monte_carlo import quantum_monte_carlo, compute_greeks

result = quantum_monte_carlo(
    distribution="lognormal",
    payoff="european_call",
    params={
        "spot": 100.0,
        "strike": 105.0,
        "volatility": 0.2,
        "rate": 0.05,
        "time_to_expiry": 1.0,
    },
    n_qubits=6,
    n_estimation=4,
    seed=42,
)
print(f"Quantum price: ${result.estimated_value:.2f}")
print(f"Classical price: ${result.classical_value:.2f}")

greeks = compute_greeks(
    spot=100.0, strike=105.0, volatility=0.2, rate=0.05, time_to_expiry=1.0, seed=42
)
print(f"Delta: {greeks.delta:.4f}, Vega: {greeks.vega:.4f}")
```

```python
# Portfolio Optimization — QAOA
from quanta.layer3.finance import portfolio_optimize

result = portfolio_optimize(
    assets=[
        {"name": "AAPL", "return": 0.12, "risk": 0.20},
        {"name": "MSFT", "return": 0.10, "risk": 0.15},
        {"name": "GOOG", "return": 0.11, "risk": 0.18},
        {"name": "BND",  "return": 0.05, "risk": 0.06},
    ],
    budget=2,  # Select 2 assets
    seed=42,
)
print(result.selected)
print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
```

### MCP Workflow

Use the `option-pricing` prompt for a guided quantum finance session:
1. `monte_carlo_price` — Price European calls and puts via IQAE
2. `option_greeks` — Compute Black-Scholes sensitivity Greeks ($\Delta, \Gamma, \mathcal{V}, \Theta, \rho$)
3. `portfolio_optimize` — Allocate discrete asset budgets under risk aversion

---

## 📊 Analytics / Marketing & CRM Pack

Quantum algorithms for customer analytics and marketing optimization.

### Modules

| Module | Description | MCP Tool |
|--------|-------------|----------|
| `layer3/entity_resolution.py` | Quantum-classical entity matching | — (Python API) |
| `layer3/clustering.py` | Buhrman Swap-Test quantum clustering | `cluster_data` |
| `layer3/qsvm.py` | Cortes-Vapnik Dual QP quantum kernel classification | — |
| `layer3/qml.py` | Variational quantum classifier | `qml_classify` |

### Quick Start

```python
# Entity Resolution — Customer Deduplication
from quanta.layer3.entity_resolution import resolve

records = [
    {"name": "John Smith", "email": "john@example.com", "phone": "555-0101"},
    {"name": "J. Smith", "email": "jsmith@example.com", "phone": "555-0101"},
    {"name": "Jane Doe", "email": "jane@example.com", "phone": "555-0202"},
]

result = resolve(records, threshold=0.65, seed=42)
print(result.clusters)  # Groups matching records
```

```python
# Quantum Clustering — Customer Segmentation
from quanta.layer3.clustering import quantum_cluster

data = [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [5.5, 7.5]]
result = quantum_cluster(data, k=2, seed=42)
print(result.labels)     # [0, 0, 1, 1]
print(result.centroids)
```

```python
# Quantum Classification — Churn Prediction
import numpy as np
from quanta.layer3.qml import QuantumClassifier

X_train = np.array([[0.1, 0.2], [0.2, 0.1], [2.8, 2.9], [2.9, 2.8]])
y_train = np.array([0, 0, 1, 1])
X_test, y_test = X_train, y_train

clf = QuantumClassifier(n_qubits=2, n_layers=2, feature_map="zz", seed=42)
clf.fit(X_train, y_train, epochs=15)
predictions = clf.predict(X_test)
print(f"Accuracy: {sum(predictions == y_test) / len(y_test):.1%}")
```

### MCP Workflow

1. Use `resolve()` Python API — Deduplicate customer records
2. `cluster_data` — Segment customers by behavior via Swap-Test distances
3. `qml_classify` — Train variational quantum classifiers on tabular customer features

---

## Summary

| Pack | Modules | MCP Tools | Use Cases |
|------|---------|-----------|-----------|
| **Finance** | 2 | **3** (`monte_carlo_price`, `option_greeks`, `portfolio_optimize`) | Option pricing, Black-Scholes Greeks, portfolio optimization |
| **Analytics (Marketing/CRM)** | 4 | **2** (`cluster_data`, `qml_classify`) | Entity resolution, customer segmentation, churn classification |
