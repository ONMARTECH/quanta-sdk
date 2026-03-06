"""
tests/test_layer3_new.py — Tests for new Layer 3 modules.

Covers: VQE, Shor, QSVM, Portfolio Optimization, Hamiltonian simulation.
"""

import pytest
import numpy as np


# ═══════════════════════════════════════════
#  VQE Tests
# ═══════════════════════════════════════════

class TestVQE:
    """Variational Quantum Eigensolver tests."""

    def test_h2_ground_state_accuracy(self):
        """VQE finds H2 ground state within chemical accuracy."""
        from quanta.layer3.vqe import vqe, build_hamiltonian_matrix
        from quanta.layer3.hamiltonian import molecular_hamiltonian

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
        """VQE returns VQEResult with expected attributes."""
        from quanta.layer3.vqe import vqe
        result = vqe(2, [("ZZ", 1.0), ("XI", 0.5)], layers=1, max_iter=10, seed=42)
        assert hasattr(result, "energy")
        assert hasattr(result, "optimal_params")
        assert hasattr(result, "history")
        assert len(result.history) > 0

    def test_vqe_repr(self):
        from quanta.layer3.vqe import vqe
        result = vqe(2, [("ZZ", 1.0)], layers=1, max_iter=5, seed=42)
        assert "VQEResult" in repr(result)

    def test_build_hamiltonian_matrix_hermitian(self):
        """Hamiltonian matrix must be Hermitian."""
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
        """15 = 3 × 5."""
        from quanta.layer3.shor import factor
        result = factor(15, seed=42)
        f1, f2 = sorted(result.factors)
        assert f1 * f2 == 15
        assert f1 > 1 and f2 > 1

    def test_factor_21(self):
        """21 = 3 × 7."""
        from quanta.layer3.shor import factor
        result = factor(21, seed=42)
        f1, f2 = sorted(result.factors)
        assert f1 * f2 == 21

    def test_factor_even_number(self):
        """Even number -> classical shortcut."""
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
        assert "15" in repr(r)

    def test_factor_invalid_raises(self):
        from quanta.layer3.shor import factor
        with pytest.raises(ValueError):
            factor(1)


# ═══════════════════════════════════════════
#  QSVM Tests
# ═══════════════════════════════════════════

class TestQSVM:
    """Quantum Support Vector Machine tests."""

    def test_linearly_separable(self):
        """QSVM correctly classifies linearly separable data."""
        from quanta.layer3.qsvm import qsvm_classify
        X_train = [[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]]
        y_train = [0, 0, 1, 1]
        X_test = [[0.15, 0.15], [0.85, 0.85]]

        result = qsvm_classify(X_train, y_train, X_test)
        assert result.predictions == [0, 1]

    def test_kernel_matrix_symmetric(self):
        """Quantum kernel matrix must be symmetric."""
        from quanta.layer3.qsvm import qsvm_classify
        X = [[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]]
        result = qsvm_classify(X, [0, 0, 1], [[0.5, 0.5]])
        K = result.kernel_matrix
        assert np.allclose(K, K.T, atol=1e-6)

    def test_kernel_diagonal_is_one(self):
        """K(x, x) should be 1.0 (self-fidelity)."""
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
        """Selected assets must match budget."""
        from quanta.layer3.finance import portfolio_optimize
        assets = [
            {"name": "A", "return": 0.10, "risk": 0.15},
            {"name": "B", "return": 0.20, "risk": 0.30},
            {"name": "C", "return": 0.05, "risk": 0.05},
        ]
        result = portfolio_optimize(assets, budget=2, seed=42)
        assert len(result.selected) == 2

    def test_conservative_selects_low_risk(self):
        """High risk aversion → low risk assets."""
        from quanta.layer3.finance import portfolio_optimize
        assets = [
            {"name": "SAFE", "return": 0.05, "risk": 0.02},
            {"name": "RISKY", "return": 0.40, "risk": 0.80},
        ]
        result = portfolio_optimize(assets, budget=1, risk_aversion=5.0, seed=42)
        assert "SAFE" in result.selected

    def test_aggressive_selects_high_return(self):
        """Low risk aversion → high return assets."""
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
        """Time evolution must preserve statevector norm."""
        from quanta.layer3.hamiltonian import molecular_hamiltonian, evolve
        h2 = molecular_hamiltonian("H2")
        result = evolve(h2, time=1.0, steps=10)
        norm = np.linalg.norm(result.final_state)
        assert abs(norm - 1.0) < 1e-6

    def test_evolution_energy_conservation(self):
        """Energy should be approximately conserved during evolution."""
        from quanta.layer3.hamiltonian import molecular_hamiltonian, evolve
        h2 = molecular_hamiltonian("H2")
        result = evolve(h2, time=1.0, steps=20)
        e_start = result.energy_history[0]
        e_end = result.energy_history[-1]
        assert abs(e_start - e_end) < 0.1

    def test_evolution_summary(self):
        from quanta.layer3.hamiltonian import evolve
        result = evolve([("ZZ", 1.0)], num_qubits=2, time=0.5, steps=5)
        assert "Evolution" in result.summary()
