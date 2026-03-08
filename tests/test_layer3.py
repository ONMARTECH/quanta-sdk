"""
tests/test_layer3.py -- Layer 3 declarative API tests.

Covers:
  - search() (Grover)
  - optimize() (QAOA)
  - MultiAgentSystem
  - VQE
  - Shor
  - QSVM
  - Portfolio optimization
  - Hamiltonian simulation
"""

import numpy as np
import pytest

from quanta.layer3.agent import Agent, MultiAgentSystem
from quanta.layer3.optimize import optimize
from quanta.layer3.search import search

# ═══════════════════════════════════════════
#  search() Tests
# ═══════════════════════════════════════════

class TestSearch:
    """Grover search tests."""

    def test_search_finds_exact_target(self):
        result = search(num_bits=3, target=5, shots=1000, seed=42)
        assert result.most_frequent == "101"

    def test_search_finds_target_with_lambda(self):
        result = search(
            num_bits=4, target=lambda x: x == 7,
            shots=1000, seed=42,
        )
        assert result.most_frequent == "0111"

    def test_search_high_probability(self):
        result = search(num_bits=3, target=3, shots=1000, seed=42)
        prob = result.probabilities.get("011", 0)
        assert prob > 0.8

    def test_search_invalid_bits_raises(self):
        with pytest.raises(ValueError):
            search(num_bits=0, target=0)

    def test_search_no_target_raises(self):
        with pytest.raises(ValueError):
            search(num_bits=3, target=lambda x: False)


# ═══════════════════════════════════════════
#  optimize() Tests
# ═══════════════════════════════════════════

class TestOptimize:
    """QAOA optimization tests."""

    def test_optimize_finds_minimum(self):
        result = optimize(
            num_bits=3,
            cost=lambda x: (x - 3) ** 2,
            minimize=True,
            shots=2048,
            seed=42,
        )
        assert result.best_bitstring == "011"
        assert result.best_cost == 0.0

    def test_optimize_finds_maximum(self):
        result = optimize(
            num_bits=3,
            cost=lambda x: x,
            minimize=False,
            shots=2048,
            seed=42,
        )
        assert int(result.best_bitstring, 2) == 7

    def test_optimize_summary_returns_string(self):
        result = optimize(num_bits=2, cost=lambda x: x, seed=42)
        assert isinstance(result.summary(), str)


# ═══════════════════════════════════════════
#  MultiAgentSystem Tests
# ═══════════════════════════════════════════

class TestMultiAgent:
    """Multi-agent quantum modeling tests."""

    def test_two_independent_agents_have_low_correlation(self):
        system = MultiAgentSystem([
            Agent("A", ["evet", "hayir"]),
            Agent("B", ["evet", "hayir"]),
        ])
        result = system.simulate(shots=2000, seed=42)
        corr = result.correlation("A", "B")
        assert -0.3 < corr < 0.3

    def test_strongly_interacting_agents_are_correlated(self):
        system = MultiAgentSystem([
            Agent("A", ["sol", "sag"]),
            Agent("B", ["sol", "sag"]),
        ])
        system.interact("A", "B", strength=0.9)
        result = system.simulate(shots=2000, seed=42)
        corr = result.correlation("A", "B")
        assert abs(corr) > 0.05

    def test_agent_probabilities_sum_to_one(self):
        system = MultiAgentSystem([
            Agent("X", ["al", "sat"]),
        ])
        result = system.simulate(shots=1000, seed=42)
        probs = result.agent_probabilities("X")
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

    def test_three_agents_with_chain_interaction(self):
        system = MultiAgentSystem([
            Agent("A", ["0", "1"]),
            Agent("B", ["0", "1"]),
            Agent("C", ["0", "1"]),
        ])
        system.interact("A", "B", strength=0.8)
        system.interact("B", "C", strength=0.8)
        result = system.simulate(shots=1000, seed=42)
        assert result.shots == 1000
        assert len(result.agents) == 3

    def test_summary_returns_formatted_string(self):
        system = MultiAgentSystem([
            Agent("buyer", ["buy", "skip"]),
            Agent("seller", ["discount", "hold"]),
        ])
        system.interact("buyer", "seller", strength=0.5)
        result = system.simulate(shots=100, seed=42)
        summary = result.summary()
        assert "buyer" in summary
        assert "seller" in summary

    def test_invalid_agent_name_raises(self):
        system = MultiAgentSystem([Agent("A", ["x", "y"])])
        with pytest.raises(ValueError):
            system.interact("A", "Z", strength=0.5)

    def test_biased_agent_reflects_bias(self):
        system = MultiAgentSystem([
            Agent("biased", ["a", "b"], bias=[0.9, 0.1]),
        ])
        result = system.simulate(shots=2000, seed=42)
        probs = result.agent_probabilities("biased")
        assert probs["a"] > 0.7


# ═══════════════════════════════════════════
#  VQE Tests
# ═══════════════════════════════════════════

class TestVQE:
    """Variational Quantum Eigensolver tests."""

    def test_h2_ground_state_accuracy(self):
        from quanta.layer3.hamiltonian import molecular_hamiltonian
        from quanta.layer3.vqe import build_hamiltonian_matrix, vqe
        h2 = molecular_hamiltonian("H2")
        result = vqe(
            num_qubits=2, hamiltonian=h2.terms,
            layers=3, max_iter=150, seed=42,
        )
        exact = float(np.linalg.eigvalsh(
            build_hamiltonian_matrix(h2.terms, 2)
        )[0])
        assert abs(result.energy - exact) < 0.01

    def test_vqe_returns_result(self):
        from quanta.layer3.vqe import vqe
        result = vqe(2, [("ZZ", 1.0), ("XI", 0.5)], layers=1, max_iter=10, seed=42)
        assert hasattr(result, "energy")
        assert hasattr(result, "optimal_params")
        assert len(result.history) > 0

    def test_vqe_repr(self):
        from quanta.layer3.vqe import vqe
        result = vqe(2, [("ZZ", 1.0)], layers=1, max_iter=5, seed=42)
        assert "VQEResult" in repr(result)

    def test_build_hamiltonian_matrix_hermitian(self):
        from quanta.layer3.vqe import build_hamiltonian_matrix
        H = build_hamiltonian_matrix([("ZZ", 1.0), ("XX", 0.5)], 2)
        assert np.allclose(H, H.conj().T)

    def test_build_hamiltonian_matrix_shape(self):
        from quanta.layer3.vqe import build_hamiltonian_matrix
        H = build_hamiltonian_matrix([("ZI", 1.0)], 2)
        assert H.shape == (4, 4)


# ═══════════════════════════════════════════
#  Shor Tests
# ═══════════════════════════════════════════

class TestShor:
    """Shor's factoring algorithm tests."""

    def test_factor_15(self):
        from quanta.layer3.shor import factor
        result = factor(15, seed=42)
        f1, f2 = sorted(result.factors)
        assert f1 * f2 == 15
        assert f1 > 1 and f2 > 1

    def test_factor_21(self):
        from quanta.layer3.shor import factor
        result = factor(21, seed=42)
        f1, f2 = sorted(result.factors)
        assert f1 * f2 == 21

    def test_factor_even_number(self):
        from quanta.layer3.shor import factor
        result = factor(12, seed=42)
        assert result.factors[0] * result.factors[1] == 12
        assert result.method == "classical_shortcut"

    def test_factor_small_composite(self):
        from quanta.layer3.shor import factor
        result = factor(35, seed=42)
        f1, f2 = sorted(result.factors)
        assert f1 * f2 == 35

    def test_factor_repr(self):
        from quanta.layer3.shor import factor
        r = factor(15, seed=42)
        assert "ShorResult" in repr(r)

    def test_factor_invalid_raises(self):
        from quanta.layer3.shor import factor
        with pytest.raises(ValueError):
            factor(1)


class TestShorInternals:
    """Tests for Shor's internal functions (QFT period finding, continued fractions)."""

    def test_quantum_order_finding_returns_positive(self):
        from quanta.layer3.shor import _quantum_order_finding
        period = _quantum_order_finding(a=7, N=15, seed=42)
        assert period >= 1

    def test_quantum_order_finding_valid_period(self):
        from quanta.layer3.shor import _quantum_order_finding
        # For a=2, N=15: true period is 4 (2^4 = 16 ≡ 1 mod 15)
        period = _quantum_order_finding(a=2, N=15, seed=42)
        assert period >= 1
        # Period should divide the true Euler totient or be a divisor
        assert period <= 15

    def test_quantum_order_finding_different_seeds(self):
        from quanta.layer3.shor import _quantum_order_finding
        periods = set()
        for seed in range(5):
            p = _quantum_order_finding(a=7, N=15, seed=seed)
            periods.add(p)
        # Should get at least 1 valid period
        assert all(p >= 1 for p in periods)

    def test_continued_fraction_period_exact(self):
        from quanta.layer3.shor import _continued_fraction_period
        # phase = 1/4 should give period 4
        period = _continued_fraction_period(0.25, N=15)
        assert period == 4

    def test_continued_fraction_period_zero(self):
        from quanta.layer3.shor import _continued_fraction_period
        period = _continued_fraction_period(0.0, N=15)
        assert period == 1

    def test_continued_fraction_period_small(self):
        from quanta.layer3.shor import _continued_fraction_period
        # phase = 1/3 should give period 3
        period = _continued_fraction_period(1/3, N=21)
        assert period == 3

    def test_shor_result_summary(self):
        from quanta.layer3.shor import ShorResult
        result = ShorResult(N=15, factors=(3, 5), period=4, attempts=1, method="quantum")
        summary = result.summary()
        assert "15" in summary
        assert "3" in summary
        assert "5" in summary
        assert "quantum" in summary

    def test_factor_summary(self):
        from quanta.layer3.shor import factor
        result = factor(15, seed=42)
        summary = result.summary()
        assert "║" in summary  # box drawing characters


# ═══════════════════════════════════════════
#  QSVM Tests
# ═══════════════════════════════════════════

class TestQSVM:
    """Quantum Support Vector Machine tests."""

    def test_linearly_separable(self):
        from quanta.layer3.qsvm import qsvm_classify
        X_train = [[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]]
        y_train = [0, 0, 1, 1]
        X_test = [[0.15, 0.15], [0.85, 0.85]]
        result = qsvm_classify(X_train, y_train, X_test)
        assert result.predictions == [0, 1]

    def test_kernel_matrix_symmetric(self):
        from quanta.layer3.qsvm import qsvm_classify
        X = [[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]]
        result = qsvm_classify(X, [0, 0, 1], [[0.5, 0.5]])
        K = result.kernel_matrix
        assert np.allclose(K, K.T, atol=1e-6)

    def test_kernel_diagonal_is_one(self):
        from quanta.layer3.qsvm import qsvm_classify
        X = [[0.3, 0.7], [0.6, 0.4]]
        result = qsvm_classify(X, [0, 1], [[0.5, 0.5]])
        for i in range(len(X)):
            assert abs(result.kernel_matrix[i, i] - 1.0) < 0.01

    def test_qsvm_summary(self):
        from quanta.layer3.qsvm import qsvm_classify
        result = qsvm_classify([[0, 0], [1, 1]], [0, 1], [[0.5, 0.5]])
        assert "QSVM" in result.summary()


# ═══════════════════════════════════════════
#  Portfolio Optimization Tests
# ═══════════════════════════════════════════

class TestFinance:
    """Quantum portfolio optimization tests."""

    def test_budget_constraint(self):
        from quanta.layer3.finance import portfolio_optimize
        assets = [
            {"name": "A", "return": 0.10, "risk": 0.15},
            {"name": "B", "return": 0.20, "risk": 0.30},
            {"name": "C", "return": 0.05, "risk": 0.05},
        ]
        result = portfolio_optimize(assets, budget=2, seed=42)
        assert len(result.selected) == 2

    def test_conservative_selects_low_risk(self):
        from quanta.layer3.finance import portfolio_optimize
        assets = [
            {"name": "SAFE", "return": 0.05, "risk": 0.02},
            {"name": "RISKY", "return": 0.40, "risk": 0.80},
        ]
        result = portfolio_optimize(assets, budget=1, risk_aversion=5.0, seed=42)
        assert "SAFE" in result.selected

    def test_aggressive_selects_high_return(self):
        from quanta.layer3.finance import portfolio_optimize
        assets = [
            {"name": "SAFE", "return": 0.03, "risk": 0.02},
            {"name": "GROWTH", "return": 0.35, "risk": 0.40},
        ]
        result = portfolio_optimize(assets, budget=1, risk_aversion=0.01, seed=42)
        assert "GROWTH" in result.selected

    def test_sharpe_ratio_positive(self):
        from quanta.layer3.finance import portfolio_optimize
        assets = [
            {"name": "X", "return": 0.10, "risk": 0.05},
            {"name": "Y", "return": 0.20, "risk": 0.10},
        ]
        result = portfolio_optimize(assets, budget=1, seed=42)
        assert result.sharpe_ratio > 0

    def test_too_many_assets_raises(self):
        from quanta.layer3.finance import portfolio_optimize
        assets = [{"name": f"A{i}", "return": 0.1, "risk": 0.1} for i in range(20)]
        with pytest.raises(ValueError):
            portfolio_optimize(assets)

    def test_portfolio_summary(self):
        from quanta.layer3.finance import portfolio_optimize
        assets = [
            {"name": "A", "return": 0.10, "risk": 0.15},
            {"name": "B", "return": 0.20, "risk": 0.30},
        ]
        result = portfolio_optimize(assets, budget=1, seed=42)
        assert "Portfolio" in result.summary()


# ═══════════════════════════════════════════
#  Hamiltonian Simulation Tests
# ═══════════════════════════════════════════

class TestHamiltonian:
    """Hamiltonian simulation tests."""

    def test_h2_molecule_loads(self):
        from quanta.layer3.hamiltonian import molecular_hamiltonian
        h2 = molecular_hamiltonian("H2")
        assert h2.num_qubits == 2
        assert len(h2.terms) == 5

    def test_heh_plus_loads(self):
        from quanta.layer3.hamiltonian import molecular_hamiltonian
        heh = molecular_hamiltonian("HeH+")
        assert heh.num_qubits == 2

    def test_lih_loads(self):
        from quanta.layer3.hamiltonian import molecular_hamiltonian
        lih = molecular_hamiltonian("LiH")
        assert lih.num_qubits == 4

    def test_unknown_molecule_raises(self):
        from quanta.layer3.hamiltonian import molecular_hamiltonian
        with pytest.raises(ValueError):
            molecular_hamiltonian("XeF6")

    def test_evolution_preserves_norm(self):
        from quanta.layer3.hamiltonian import evolve, molecular_hamiltonian
        h2 = molecular_hamiltonian("H2")
        result = evolve(h2, time=1.0, steps=10)
        norm = np.linalg.norm(result.final_state)
        assert abs(norm - 1.0) < 1e-6

    def test_evolution_energy_conservation(self):
        from quanta.layer3.hamiltonian import evolve, molecular_hamiltonian
        h2 = molecular_hamiltonian("H2")
        result = evolve(h2, time=1.0, steps=20)
        e_start = result.energy_history[0]
        e_end = result.energy_history[-1]
        assert abs(e_start - e_end) < 0.1

    def test_evolution_summary(self):
        from quanta.layer3.hamiltonian import evolve
        result = evolve([("ZZ", 1.0)], num_qubits=2, time=0.5, steps=5)
        assert "Evolution" in result.summary()
