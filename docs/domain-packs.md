# Domain Use-Case Packs

Quanta SDK includes production-ready quantum algorithms for specific business domains.
Each pack combines quantum modules, MCP tools, and documentation.

---

## 🏦 Finance Pack

Quantum algorithms for quantitative finance and risk analysis.

### Modules

| Module | Description | MCP Tool |
|--------|-------------|----------|
| `layer3/monte_carlo.py` | Quantum amplitude estimation for option pricing | `monte_carlo_price` |
| `layer3/finance.py` | Portfolio optimization via QAOA | — |

### Quick Start

```python
# Option Pricing — Quantum Monte Carlo
from quanta.layer3.monte_carlo import price_option

result = price_option(
    S0=100,       # Spot price
    K=105,        # Strike price
    sigma=0.2,    # Volatility
    T=1.0,        # Time to expiry (years)
    r=0.05,       # Risk-free rate
    option_type="call",
    n_qubits=6,
)
print(f"Quantum price: ${result.quantum_price:.2f}")
print(f"Classical price: ${result.classical_price:.2f}")
```

```python
# Portfolio Optimization — QAOA
from quanta.layer3.finance import optimize_portfolio

result = optimize_portfolio(
    returns=[0.08, 0.12, 0.06, 0.10],
    covariance=[[0.04, 0.01, 0.02, 0.01],
                [0.01, 0.09, 0.01, 0.03],
                [0.02, 0.01, 0.03, 0.01],
                [0.01, 0.03, 0.01, 0.05]],
    budget=2,  # Select 2 assets
)
print(result.selected_assets)
```

### MCP Workflow

Use the `option-pricing` prompt for a guided quantum finance session:
1. `monte_carlo_price` — Price calls and puts
2. Compare quantum vs classical pricing
3. Vary volatility and strike to explore the option surface

---

## 📊 Marketing / CRM Pack

Quantum algorithms for customer analytics and marketing optimization.

### Modules

| Module | Description | MCP Tool |
|--------|-------------|----------|
| `layer3/entity_resolution.py` | Quantum-classical entity matching | — (Python API) |
| `layer3/clustering.py` | Swap-test quantum clustering | `cluster_data` |
| `layer3/qsvm.py` | Quantum kernel classification | — |
| `layer3/qml.py` | Variational quantum classifier | — |

### Quick Start

```python
# Entity Resolution — Customer Deduplication
from quanta.layer3.entity_resolution import resolve_entities

records = [
    {"name": "John Smith", "email": "john@example.com", "phone": "555-0101"},
    {"name": "J. Smith", "email": "jsmith@example.com", "phone": "555-0101"},
    {"name": "Jane Doe", "email": "jane@example.com", "phone": "555-0202"},
]

result = resolve_entities(records, threshold=0.7)
print(result.clusters)  # Groups matching records
```

```python
# Quantum Clustering — Customer Segmentation
from quanta.layer3.clustering import quantum_cluster

data = [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [5.5, 7.5]]
result = quantum_cluster(data, n_clusters=2)
print(result.labels)     # [0, 0, 1, 1]
print(result.centroids)
```

```python
# Quantum Classification — Churn Prediction
from quanta.layer3.qml import QuantumClassifier

clf = QuantumClassifier(n_qubits=4, n_layers=3, feature_map="ZZFeatureMap")
clf.fit(X_train, y_train, epochs=30)
predictions = clf.predict(X_test)
print(f"Accuracy: {sum(predictions == y_test) / len(y_test):.1%}")
```

### MCP Workflow

1. Use `resolve_entities()` Python API — Deduplicate customer records
2. `cluster_data` — Segment customers by behavior
3. Use results for targeted marketing campaigns

---

## Summary

| Pack | Modules | MCP Tools | Use Cases |
|------|---------|-----------|-----------|
| **Finance** | 2 | 1 | Option pricing, portfolio optimization, risk analysis |
| **Marketing/CRM** | 4 | 2 | Entity resolution, customer segmentation, churn prediction |
