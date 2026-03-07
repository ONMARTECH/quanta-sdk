"""
tests/test_gradients.py — Gradient computation tests.

Verifies parameter-shift, finite-diff, and natural gradient methods
against known analytical results.
"""

import numpy as np
import pytest

from quanta.gradients import (
    parameter_shift,
    finite_diff,
    natural_gradient,
    expectation,
    multi_expectation,
    GradientResult,
)
from quanta.simulator.statevector import StateVectorSimulator


# ═══════════════════════════════════════════
#  Expectation Value Tests
# ═══════════════════════════════════════════

class TestExpectation:
    """Expectation value computation tests."""

    def test_z_on_zero_state(self):
        """<0|Z|0> = 1."""
        state = np.array([1, 0], dtype=complex)
        assert expectation(state, "Z", 1) == pytest.approx(1.0)

    def test_z_on_one_state(self):
        """<1|Z|1> = -1."""
        state = np.array([0, 1], dtype=complex)
        assert expectation(state, "Z", 1) == pytest.approx(-1.0)

    def test_x_on_plus_state(self):
        """<+|X|+> = 1."""
        state = np.array([1, 1], dtype=complex) / np.sqrt(2)
        assert expectation(state, "X", 1) == pytest.approx(1.0)

    def test_zz_on_bell_state(self):
        """<Φ+|ZZ|Φ+> = 1 (both qubits always agree)."""
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        assert expectation(state, "ZZ", 2) == pytest.approx(1.0)

    def test_multi_expectation(self):
        """Sum of weighted Pauli terms."""
        state = np.array([1, 0], dtype=complex)
        result = multi_expectation(state, [("Z", 2.0), ("X", 1.0)], 1)
        # <0|Z|0> = 1, <0|X|0> = 0 → 2*1 + 1*0 = 2
        assert result == pytest.approx(2.0)


# ═══════════════════════════════════════════
#  Parameter-Shift Tests
# ═══════════════════════════════════════════

class TestParameterShift:
    """Parameter-shift rule gradient tests."""

    def _ry_cost(self, params: np.ndarray) -> float:
        """RY(θ)|0⟩ → <Z> = cos(θ)."""
        sim = StateVectorSimulator(1)
        sim.apply("RY", (0,), (params[0],))
        return expectation(sim.state, "Z", 1)

    def test_gradient_of_cosine(self):
        """d/dθ cos(θ) = -sin(θ)."""
        theta = 0.7
        result = parameter_shift(self._ry_cost, [theta])
        expected_grad = -np.sin(theta)
        assert result.gradients[0] == pytest.approx(expected_grad, abs=1e-10)

    def test_gradient_at_zero(self):
        """d/dθ cos(θ)|_{θ=0} = 0."""
        result = parameter_shift(self._ry_cost, [0.0])
        assert result.gradients[0] == pytest.approx(0.0, abs=1e-10)

    def test_gradient_at_pi(self):
        """d/dθ cos(θ)|_{θ=π} = 0."""
        result = parameter_shift(self._ry_cost, [np.pi])
        assert result.gradients[0] == pytest.approx(0.0, abs=1e-10)

    def test_gradient_at_pi_half(self):
        """d/dθ cos(θ)|_{θ=π/2} = -1."""
        result = parameter_shift(self._ry_cost, [np.pi / 2])
        assert result.gradients[0] == pytest.approx(-1.0, abs=1e-10)

    def test_function_value_stored(self):
        """GradientResult stores the function value."""
        result = parameter_shift(self._ry_cost, [0.0])
        assert result.function_value == pytest.approx(1.0, abs=1e-10)

    def test_num_evaluations(self):
        """2n+1 evaluations for n parameters."""
        result = parameter_shift(self._ry_cost, [0.5])
        assert result.num_evaluations == 3  # 2*1 + 1

    def test_method_name(self):
        result = parameter_shift(self._ry_cost, [0.5])
        assert result.method == "parameter-shift"

    def test_multi_param_gradient(self):
        """Gradient of RY(θ₁)·RZ(θ₂) circuit."""
        def cost(params):
            sim = StateVectorSimulator(1)
            sim.apply("RY", (0,), (params[0],))
            sim.apply("RZ", (0,), (params[1],))
            return expectation(sim.state, "Z", 1)

        result = parameter_shift(cost, [0.5, 1.0])
        assert len(result.gradients) == 2
        # RZ doesn't affect Z expectation after RY
        # d<Z>/dθ₁ = -sin(θ₁), d<Z>/dθ₂ = 0
        assert result.gradients[0] == pytest.approx(-np.sin(0.5), abs=1e-10)
        assert result.gradients[1] == pytest.approx(0.0, abs=1e-10)

    def test_repr(self):
        result = parameter_shift(self._ry_cost, [0.5])
        repr_str = repr(result)
        assert "parameter-shift" in repr_str


# ═══════════════════════════════════════════
#  Finite Difference Tests
# ═══════════════════════════════════════════

class TestFiniteDiff:
    """Finite difference gradient tests."""

    def _ry_cost(self, params: np.ndarray) -> float:
        sim = StateVectorSimulator(1)
        sim.apply("RY", (0,), (params[0],))
        return expectation(sim.state, "Z", 1)

    def test_central_diff_matches_analytical(self):
        """Central difference ≈ -sin(θ)."""
        theta = 0.7
        result = finite_diff(self._ry_cost, [theta], method="central")
        assert result.gradients[0] == pytest.approx(-np.sin(theta), abs=1e-6)

    def test_forward_diff_matches_analytical(self):
        """Forward difference ≈ -sin(θ) (less precise)."""
        theta = 0.7
        result = finite_diff(self._ry_cost, [theta], method="forward")
        assert result.gradients[0] == pytest.approx(-np.sin(theta), abs=1e-4)

    def test_central_more_accurate_than_forward(self):
        """Central diff should be closer to exact than forward."""
        theta = 0.7
        exact = -np.sin(theta)
        central = finite_diff(self._ry_cost, [theta], method="central")
        forward = finite_diff(self._ry_cost, [theta], method="forward")
        err_central = abs(central.gradients[0] - exact)
        err_forward = abs(forward.gradients[0] - exact)
        assert err_central < err_forward

    def test_method_name_central(self):
        result = finite_diff(self._ry_cost, [0.5], method="central")
        assert result.method == "finite-diff-central"

    def test_method_name_forward(self):
        result = finite_diff(self._ry_cost, [0.5], method="forward")
        assert result.method == "finite-diff-forward"


# ═══════════════════════════════════════════
#  Natural Gradient Tests
# ═══════════════════════════════════════════

class TestNaturalGradient:
    """Natural gradient (QFIM-based) tests."""

    def _ry_cost(self, params: np.ndarray) -> float:
        sim = StateVectorSimulator(1)
        sim.apply("RY", (0,), (params[0],))
        return expectation(sim.state, "Z", 1)

    def _ry_state(self, params: np.ndarray) -> np.ndarray:
        sim = StateVectorSimulator(1)
        sim.apply("RY", (0,), (params[0],))
        return sim.state

    def test_natural_gradient_direction(self):
        """Natural gradient should point in same direction as vanilla."""
        theta = 0.7
        ps = parameter_shift(self._ry_cost, [theta])
        ng = natural_gradient(
            self._ry_cost, self._ry_state, [theta],
        )
        # Same sign (same direction)
        assert np.sign(ps.gradients[0]) == np.sign(ng.gradients[0])

    def test_natural_gradient_method_name(self):
        result = natural_gradient(
            self._ry_cost, self._ry_state, [0.5],
        )
        assert result.method == "natural-gradient"

    def test_natural_gradient_returns_result(self):
        result = natural_gradient(
            self._ry_cost, self._ry_state, [0.5],
        )
        assert isinstance(result, GradientResult)
        assert len(result.gradients) == 1


# ═══════════════════════════════════════════
#  Parameter-Shift vs Finite Diff Agreement
# ═══════════════════════════════════════════

class TestGradientAgreement:
    """Cross-validate different gradient methods."""

    def test_all_methods_agree_on_simple_circuit(self):
        """All 3 methods should agree for a simple 1-qubit circuit."""
        def cost(params):
            sim = StateVectorSimulator(1)
            sim.apply("RY", (0,), (params[0],))
            return expectation(sim.state, "Z", 1)

        def state(params):
            sim = StateVectorSimulator(1)
            sim.apply("RY", (0,), (params[0],))
            return sim.state

        theta = 1.2
        ps = parameter_shift(cost, [theta])
        fd = finite_diff(cost, [theta])
        ng = natural_gradient(cost, state, [theta])

        # All should approximate -sin(1.2)
        assert ps.gradients[0] == pytest.approx(-np.sin(theta), abs=1e-8)
        assert fd.gradients[0] == pytest.approx(-np.sin(theta), abs=1e-5)
        # Natural gradient scales differently but same sign
        assert np.sign(ng.gradients[0]) == np.sign(-np.sin(theta))
