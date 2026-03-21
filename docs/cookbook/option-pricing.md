# Option Pricing — Quick Recipe

> Price a European call option using Quantum Monte Carlo amplitude estimation.

## What It Does

Classical Monte Carlo pricing generates random price paths and averages the
payoffs. **Quantum Monte Carlo** achieves the same result with a quadratic
speedup using **amplitude estimation** — `O(1/ε)` vs `O(1/ε²)`.

This recipe prices a European call option: the right to buy a stock at price
**K** at expiry **T**, profiting when the stock price exceeds the strike.

## Code

```python
from quanta.layer3.monte_carlo import quantum_monte_carlo

result = quantum_monte_carlo(
    distribution="lognormal",
    payoff="european_call",
    params={
        "spot": 100,      # Current stock price ($100)
        "strike": 105,    # Strike price ($105)
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

## Expected Output

```
Quantum estimate:   8.0214
Classical estimate: 8.0213
Grover iterations:  4
Confidence:         [7.8500, 8.1928]
```

> **Key insight:** The quantum and classical estimates converge, but quantum
> Monte Carlo reaches this accuracy with **quadratically fewer samples** on
> real quantum hardware.

## Parameters Explained

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `spot` | Current stock price | $50–$500 |
| `strike` | Exercise price | Near spot price |
| `rate` | Risk-free interest rate | 0.01–0.10 |
| `vol` | Annual volatility (σ) | 0.10–0.50 |
| `T` | Time to expiry (years) | 0.25–2.0 |
| `n_qubits` | Precision qubits | 3–8 (more = more accurate) |

## Try Next

- **Put options**: Change `payoff="european_put"` to price puts
- **Higher precision**: Increase `n_qubits=8` for tighter confidence
- **MCP integration**: Use `monte_carlo_price` MCP tool for AI-assisted pricing
- **Volatility surface**: Loop over `vol=[0.1, 0.2, 0.3, 0.4]` to map the surface

## See Also

- [Domain Packs — Finance](../domain-packs.md)
- [Tutorial 04 — Algorithms](../tutorials/04-algorithms.md)
