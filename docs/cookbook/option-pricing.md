# Option Pricing — Quick Recipe

> Quantum Monte Carlo for European call option pricing.

```python
from quanta.layer3.monte_carlo import quantum_monte_carlo

result = quantum_monte_carlo(
    distribution="lognormal",
    payoff="european_call",
    params={
        "spot": 100,      # Current stock price
        "strike": 105,    # Strike price
        "rate": 0.05,     # Risk-free rate (5%)
        "vol": 0.2,       # Volatility (20%)
        "T": 1.0,         # Time to expiry (1 year)
    },
    n_qubits=5,
)

print(f"Quantum estimate:   {result.estimated_value:.4f}")
print(f"Classical estimate: {result.classical_value:.4f}")
print(f"Grover iterations:  {result.grover_iterations}")
print(f"Confidence:         [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
```
