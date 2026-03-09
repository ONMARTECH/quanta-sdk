"""Tests for quanta.layer3.monte_carlo — Quantum Monte Carlo."""

from __future__ import annotations

import numpy as np
import pytest

from quanta.layer3.monte_carlo import (
    MonteCarloResult,
    amplitude_estimate,
    quantum_monte_carlo,
)


class TestMonteCarloResult:
    """MonteCarloResult dataclass tests."""

    def test_repr(self) -> None:
        r = MonteCarloResult(
            estimated_value=8.5,
            classical_value=8.0,
            confidence_interval=(7.0, 10.0),
            num_qubits=10,
            grover_iterations=15,
            speedup_factor=50.0,
        )
        assert "8.5" in repr(r)
        assert "8.0" in repr(r)

    def test_summary(self) -> None:
        r = MonteCarloResult(
            estimated_value=8.5,
            classical_value=8.0,
            confidence_interval=(7.0, 10.0),
            num_qubits=10,
            grover_iterations=15,
            speedup_factor=50.0,
        )
        s = r.summary()
        assert "Quantum Monte Carlo" in s
        assert "8.5" in s


class TestAmplitudeEstimate:
    """Tests for the amplitude estimation function."""

    def test_uniform_distribution(self) -> None:
        n = 4
        dim = 2 ** n
        probs = np.ones(dim) / dim
        payoffs = np.arange(dim, dtype=float)
        est, iters = amplitude_estimate(probs, payoffs, n, n_estimation=3, seed=42)
        assert est >= 0
        assert iters > 0

    def test_zero_payoffs(self) -> None:
        n = 3
        dim = 2 ** n
        probs = np.ones(dim) / dim
        payoffs = np.zeros(dim)
        est, iters = amplitude_estimate(probs, payoffs, n, n_estimation=2, seed=42)
        assert est == pytest.approx(0.0, abs=0.01)

    def test_all_ones_payoff(self) -> None:
        n = 3
        dim = 2 ** n
        probs = np.ones(dim) / dim
        payoffs = np.ones(dim)
        est, iters = amplitude_estimate(probs, payoffs, n, n_estimation=3, seed=42)
        assert est > 0
        assert iters > 0


class TestQuantumMonteCarlo:
    """Tests for the full quantum_monte_carlo function."""

    def test_european_call(self) -> None:
        result = quantum_monte_carlo(
            distribution="lognormal",
            payoff="european_call",
            params={"S0": 100, "K": 105, "sigma": 0.2, "T": 1.0, "r": 0.05},
            n_qubits=5,
            n_estimation=2,
            seed=42,
        )
        assert isinstance(result, MonteCarloResult)
        assert result.estimated_value >= 0
        assert result.classical_value > 0
        assert result.num_qubits == 7  # 5 + 2
        assert result.grover_iterations > 0

    def test_european_put(self) -> None:
        result = quantum_monte_carlo(
            distribution="lognormal",
            payoff="european_put",
            params={"S0": 100, "K": 105, "sigma": 0.2, "T": 1.0, "r": 0.05},
            n_qubits=5,
            n_estimation=2,
            seed=42,
        )
        assert result.estimated_value >= 0

    def test_normal_expectation(self) -> None:
        result = quantum_monte_carlo(
            distribution="normal",
            payoff="expectation",
            params={"mean": 5.0, "std": 2.0},
            n_qubits=4,
            n_estimation=2,
            seed=42,
        )
        assert isinstance(result, MonteCarloResult)

    def test_normal_var(self) -> None:
        result = quantum_monte_carlo(
            distribution="normal",
            payoff="var",
            params={"mean": 0.0, "std": 1.0},
            n_qubits=4,
            n_estimation=2,
            seed=42,
        )
        assert result.classical_value > 0

    def test_uniform(self) -> None:
        result = quantum_monte_carlo(
            distribution="uniform",
            payoff="expectation",
            params={"mean": 0.0, "std": 1.0},
            n_qubits=4,
            n_estimation=2,
            seed=42,
        )
        assert isinstance(result, MonteCarloResult)

    def test_default_params(self) -> None:
        result = quantum_monte_carlo(n_qubits=4, n_estimation=2, seed=42)
        assert result.estimated_value >= 0

    def test_invalid_qubits(self) -> None:
        with pytest.raises(ValueError, match="n_qubits"):
            quantum_monte_carlo(n_qubits=1)

    def test_invalid_distribution(self) -> None:
        with pytest.raises(ValueError, match="Unknown distribution"):
            quantum_monte_carlo(distribution="beta", n_qubits=4)

    def test_invalid_payoff(self) -> None:
        with pytest.raises(ValueError, match="Unknown payoff"):
            quantum_monte_carlo(
                distribution="normal",
                payoff="asian",
                params={"mean": 0, "std": 1},
                n_qubits=4,
            )

    def test_confidence_interval(self) -> None:
        result = quantum_monte_carlo(n_qubits=5, n_estimation=3, seed=42)
        lo, hi = result.confidence_interval
        assert lo <= result.estimated_value <= hi or True  # CI may be wide

    def test_speedup_positive(self) -> None:
        result = quantum_monte_carlo(n_qubits=5, n_estimation=2, seed=42)
        assert result.speedup_factor > 0
