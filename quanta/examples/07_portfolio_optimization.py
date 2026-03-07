"""
Example 07: Quantum Portfolio Optimization

Uses quantum-inspired optimization to select the best asset
combination from a portfolio. Demonstrates QAOA approach to
the Markowitz mean-variance optimization problem.

Real-world use case: hedge funds and investment banks use
quantum algorithms for portfolio selection.

Running:
    python -m quanta.examples.07_portfolio_optimization
"""

from quanta.layer3.finance import portfolio_optimize


def demo_tech_portfolio():
    """Optimize a tech stock portfolio."""
    print("=" * 55)
    print("  Quantum Portfolio Optimization — Tech Stocks")
    print("=" * 55)

    assets = [
        {"name": "AAPL",  "return": 0.12, "risk": 0.15},
        {"name": "GOOGL", "return": 0.10, "risk": 0.12},
        {"name": "TSLA",  "return": 0.28, "risk": 0.40},
        {"name": "MSFT",  "return": 0.11, "risk": 0.13},
        {"name": "NVDA",  "return": 0.35, "risk": 0.45},
        {"name": "META",  "return": 0.15, "risk": 0.20},
    ]

    print(f"\n  Assets: {len(assets)}")
    print(f"  Budget: select 3 from {len(assets)}")
    print()

    for a in assets:
        print(f"    {a['name']:>5}  return={a['return']:+.0%}  risk={a['risk']:.0%}")

    result = portfolio_optimize(
        assets, budget=3, risk_aversion=0.5, seed=42
    )

    print(f"\n{result.summary()}")


def demo_crypto_portfolio():
    """Optimize a crypto portfolio."""
    print("\n" + "=" * 55)
    print("  Quantum Portfolio Optimization — Crypto")
    print("=" * 55)

    assets = [
        {"name": "BTC",   "return": 0.40, "risk": 0.60},
        {"name": "ETH",   "return": 0.30, "risk": 0.50},
        {"name": "SOL",   "return": 0.50, "risk": 0.70},
        {"name": "BOND",  "return": 0.04, "risk": 0.02},
        {"name": "GOLD",  "return": 0.08, "risk": 0.10},
    ]

    # Conservative investor (high risk aversion)
    print("\n  Conservative investor (risk_aversion=1.5):")
    result_conservative = portfolio_optimize(
        assets, budget=2, risk_aversion=1.5, seed=42
    )
    print(f"  -> Selected: {result_conservative.selected}")
    print(f"     Return: {result_conservative.expected_return:.1%}  "
          f"Risk: {result_conservative.portfolio_risk:.1%}")

    # Aggressive investor (low risk aversion)
    print("\n  Aggressive investor (risk_aversion=0.1):")
    result_aggressive = portfolio_optimize(
        assets, budget=2, risk_aversion=0.1, seed=42
    )
    print(f"  -> Selected: {result_aggressive.selected}")
    print(f"     Return: {result_aggressive.expected_return:.1%}  "
          f"Risk: {result_aggressive.portfolio_risk:.1%}")


if __name__ == "__main__":
    demo_tech_portfolio()
    demo_crypto_portfolio()
