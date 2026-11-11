# Option Pricing — Quick Recipe

> Price a European call option and compute Greeks using Quantum Monte Carlo amplitude estimation.

## What It Does

Classical Monte Carlo pricing generates random price paths and averages the
payoffs. **Quantum Monte Carlo** achieves the same result with a quadratic
speedup using **amplitude estimation** — `O(1/ε)` vs `O(1/ε²)`.

This recipe prices a European call option: the right to buy a stock at strike
**K** at expiry **T**, profiting when the terminal stock price exceeds **K**.

## Code

```python
from quanta.layer3.monte_carlo import quantum_monte_carlo, compute_greeks

result = quantum_monte_carlo(
    distribution="lognormal",
    payoff="european_call",
    params={
        "S0": 100,        # Current spot price ($100)
        "K": 105,         # Strike price ($105)
        "r": 0.05,        # Risk-free rate (5%)
        "sigma": 0.2,     # Annual volatility (20%)
        "T": 1.0,         # Time to expiry (1 year)
    },
    n_qubits=5,
    seed=42,
)

print(f"Quantum estimate:   {result.estimated_value:.4f}")
print(f"Classical estimate: {result.classical_value:.4f}")
print(f"Grover iterations:  {result.grover_iterations}")
print(f"Confidence:         [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")

greeks = compute_greeks(
    spot=100, strike=105, volatility=0.2, rate=0.05, time_to_expiry=1.0, seed=42
)
print(f"Delta={greeks.delta:.4f}, Gamma={greeks.gamma:.4f}, Vega={greeks.vega:.4f}")
```

## Expected Output

```
Quantum estimate:   8.0608
Classical estimate: 8.0062
Grover iterations:  33
Confidence:         [7.4272, 8.6943]
Delta=0.5422, Gamma=0.0198, Vega=39.6714
```

> **Key insight:** The quantum and classical estimates converge around the analytical
> Black-Scholes call value (~8.0214), while iterative quantum amplitude estimation
> reaches target precision $\varepsilon$ with **quadratically fewer oracle queries**.

## Parameters Explained

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `S0` | Current spot stock price | \$50–\$500 |
| `K` | Strike (exercise) price | Near `S0` |
| `r` | Risk-free interest rate | 0.01–0.10 |
| `sigma` | Annual volatility ($\sigma$) | 0.10–0.50 |
| `T` | Time to expiry (years) | 0.25–2.0 |
| `n_qubits` | Discretization qubits | 3–8 (more = finer grid) |

## Try Next

- **Put options**: Change `payoff="european_put"` to price puts
- **Higher precision**: Increase `n_qubits=6` or `n_estimation=8` for tighter confidence intervals
- **MCP integration**: Use `monte_carlo_price` and `option_greeks` MCP tools for AI-assisted pricing
- **Volatility surface**: Loop over `sigma` in `[0.1, 0.2, 0.3, 0.4]` to map the implied volatility surface

## See Also

- [Domain Packs — Finance](../domain-packs.md)
- [Tutorial 04 — Algorithms](../tutorials/04-algorithms.md)
