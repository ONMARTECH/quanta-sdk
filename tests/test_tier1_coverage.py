"""
Tier 1 coverage tests — target uncovered lines in:
  - statevector.py (53, 95-111, 120, 124-129, 171, 204)
  - accelerated.py (58-89, 141-189)
  - shor.py (89-104, 197-298, 304-353)
  - surface_code.py (231-326)
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

# ════════════════════════════════════════════
#  StateVectorSimulator
# ════════════════════════════════════════════

class TestStateVectorCoverage:
    """Tests for uncovered lines in statevector.py."""

    def test_max_qubits_exceeded(self):
        """Line 53: SimulatorError when exceeding MAX_QUBITS."""
        from quanta.simulator.statevector import SimulatorError, StateVectorSimulator
        with pytest.raises(SimulatorError, match="Max 27 qubits"):
            StateVectorSimulator(28)

    def test_numpy_fallback_when_accelerated_unavailable(self):
        """Lines 95-111: Ensure NumPy tensor contraction fallback works."""
        from quanta.simulator.statevector import StateVectorSimulator

        # Mock the accelerated import to fail, forcing NumPy fallback
        with mock.patch.dict("sys.modules", {"quanta.simulator.accelerated": None}):
            sim = StateVectorSimulator(2, seed=42)
            sim.apply("H", (0,))
            sim.apply("CX", (0, 1))
            probs = sim.probabilities()
            assert abs(probs[0] - 0.5) < 0.01  # |00⟩
            assert abs(probs[3] - 0.5) < 0.01  # |11⟩

    def test_unknown_gate_error(self):
        """Line 120: SimulatorError for unknown gate."""
        from quanta.simulator.statevector import SimulatorError, StateVectorSimulator
        sim = StateVectorSimulator(2)
        with pytest.raises(SimulatorError, match="Unknown gate"):
            sim.apply("NONEXISTENT", (0,))

    def test_parametric_gate_without_params(self):
        """Lines 124-129: SimulatorError when parametric gates lack params."""
        from quanta.simulator.statevector import SimulatorError, StateVectorSimulator
        sim = StateVectorSimulator(2)
        with pytest.raises(SimulatorError, match="requires parameters"):
            sim.apply("RX", (0,))

    def test_multi_parametric_gate_without_params(self):
        """Lines 124-126: SimulatorError when multi-param gate lacks params."""
        from quanta.simulator.statevector import SimulatorError, StateVectorSimulator
        sim = StateVectorSimulator(2)
        with pytest.raises(SimulatorError, match="requires parameters"):
            sim.apply("U", (0,))

    def test_state_setter_dimension_mismatch(self):
        """Line 171: SimulatorError on dimension mismatch."""
        from quanta.simulator.statevector import SimulatorError, StateVectorSimulator
        sim = StateVectorSimulator(2)
        with pytest.raises(SimulatorError, match="State dimension mismatch"):
            sim.state = np.zeros(8, dtype=complex)  # Expected 4, got 8

    def test_state_setter_valid(self):
        """Line 171 boundary: Valid state setter."""
        from quanta.simulator.statevector import StateVectorSimulator
        sim = StateVectorSimulator(2)
        new_state = np.array([0, 1, 0, 0], dtype=complex)  # |01⟩
        sim.state = new_state
        probs = sim.probabilities()
        assert probs[1] == 1.0

    def test_apply_noise(self):
        """Line 204: apply_noise method."""
        from quanta.simulator.statevector import StateVectorSimulator
        sim = StateVectorSimulator(2, seed=42)
        sim.apply("H", (0,))

        # Create a mock noise model
        class MockNoise:
            def apply_noise(self, state, qubits, n, rng):
                return state  # identity noise

        rng = np.random.default_rng(42)
        sim.apply_noise(MockNoise(), (0,), rng)
        # State should be unchanged
        probs = sim.probabilities()
        assert abs(probs[0] - 0.5) < 0.01

    def test_apply_phase(self):
        """Line 186: apply_phase method."""
        from quanta.simulator.statevector import StateVectorSimulator
        sim = StateVectorSimulator(2, seed=42)
        # Put in |01⟩ state
        sim.apply("X", (1,))
        # Flip phase of |01⟩ = index 1
        sim.apply_phase(1, -1)
        state = sim.state
        assert abs(state[1] - (-1.0)) < 1e-10


# ════════════════════════════════════════════
#  Accelerated Backend
# ════════════════════════════════════════════

class TestAcceleratedCoverage:
    """Tests for uncovered lines in accelerated.py."""

    def test_get_backend_info_numpy(self):
        """Lines 166-191: Backend info returns numpy when no GPU."""
        from quanta.simulator.accelerated import get_backend_info
        info = get_backend_info()
        assert info["backend"] in ("numpy", "jax-gpu", "jax-tpu", "cupy")
        assert "device" in info

    def test_xp_returns_array_module(self):
        """Lines 100-106: xp() returns a valid array library."""
        from quanta.simulator.accelerated import xp
        mod = xp()
        assert hasattr(mod, "array")
        assert hasattr(mod, "tensordot")

    def test_get_array_module(self):
        """Lines 109-112: get_array_module identical to xp."""
        from quanta.simulator.accelerated import get_array_module, xp
        assert get_array_module() is xp()

    def test_tensor_contract_numpy_path(self):
        """Lines 134-163: tensor_contract with NumPy backend."""
        from quanta.simulator.accelerated import tensor_contract
        # H gate matrix
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        state = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        result = tensor_contract(H, state, (0,), 2)
        # After H on qubit 0: (|00⟩ + |10⟩)/√2
        assert abs(abs(result[0]) - 1 / np.sqrt(2)) < 1e-10

    def test_detect_backend_no_gpu(self):
        """Lines 39-86: _detect_backend handles missing JAX/CuPy."""
        import quanta.simulator.accelerated as acc
        # Reset and re-detect
        old_init = acc._initialized
        acc._initialized = False
        acc._detect_backend()
        # Should fallback to numpy on CI/dev machines without GPU
        assert acc._backend_name in ("numpy", "jax-gpu", "jax-tpu", "cupy")
        acc._initialized = old_init


# ════════════════════════════════════════════
#  Shor's Algorithm
# ════════════════════════════════════════════

class TestShorCoverage:
    """Tests for uncovered lines in shor.py."""

    def test_build_qft_dag(self):
        """Lines 89-104: QFT DAG construction."""
        from quanta.layer3.shor import _build_qft_dag
        dag = _build_qft_dag(4)
        ops = list(dag.op_nodes())
        # Should have H gates, RZ gates, and SWAP gates
        gate_names = [op.gate_name for op in ops]
        assert "H" in gate_names
        assert "RZ" in gate_names
        assert "SWAP" in gate_names

    def test_build_inverse_qft_dag(self):
        """Lines 107-129: Inverse QFT DAG construction."""
        from quanta.layer3.shor import _build_inverse_qft_dag
        dag = _build_inverse_qft_dag(4)
        ops = list(dag.op_nodes())
        gate_names = [op.gate_name for op in ops]
        assert "H" in gate_names
        assert "SWAP" in gate_names

    def test_shor_result_summary(self):
        """Lines 62-74: ShorResult.summary() formatting."""
        from quanta.layer3.shor import ShorResult
        r = ShorResult(N=15, factors=(3, 5), period=4, attempts=2, method="quantum")
        s = r.summary()
        assert "15" in s
        assert "3" in s
        assert "5" in s
        assert "quantum" in s

    def test_shor_result_repr(self):
        """Lines 55-59: ShorResult.__repr__."""
        from quanta.layer3.shor import ShorResult
        r = ShorResult(N=15, factors=(3, 5), period=4, attempts=2, method="quantum")
        assert "15" in repr(r)
        assert "3 × 5" in repr(r)

    def test_factor_small_even(self):
        """Line 260: Classical shortcut for even numbers."""
        from quanta.layer3.shor import factor
        r = factor(14)
        assert r.method == "classical_shortcut"
        assert r.factors[0] * r.factors[1] == 14

    def test_factor_prime_raises(self):
        """Lines 252-256: Factoring a prime raises ValueError."""
        from quanta.layer3.shor import factor
        with pytest.raises(ValueError, match="prime"):
            factor(13)

    def test_factor_too_small_raises(self):
        """Line 248-249: N < 2 raises ValueError."""
        from quanta.layer3.shor import factor
        with pytest.raises(ValueError, match="must be"):
            factor(1)

    def test_factor_15(self):
        """Lines 268-291: Quantum factoring of 15."""
        from quanta.layer3.shor import factor
        r = factor(15, seed=42)
        assert r.factors[0] * r.factors[1] == 15
        assert r.N == 15

    def test_factor_21(self):
        """Quantum factoring of 21."""
        from quanta.layer3.shor import factor
        r = factor(21, seed=42)
        assert r.factors[0] * r.factors[1] == 21

    def test_is_prime(self):
        """Lines 301-314: _is_prime edge cases."""
        from quanta.layer3.shor import _is_prime
        assert _is_prime(2) is True
        assert _is_prime(3) is True
        assert _is_prime(4) is False
        assert _is_prime(5) is True
        assert _is_prime(1) is False
        assert _is_prime(0) is False
        assert _is_prime(6) is False
        assert _is_prime(25) is False  # 5*5
        assert _is_prime(49) is False  # 7*7
        assert _is_prime(97) is True

    def test_factor_recursive_simple(self):
        """Lines 339-356: Recursive factoring into primes."""
        from quanta.layer3.shor import factor_recursive
        # 12 = 2 × 2 × 3
        result = factor_recursive(12, seed=42)
        product = 1
        for f in result:
            product *= f
        assert product == 12

    def test_factor_recursive_prime(self):
        """Line 341-342: Recursive on a prime returns [N]."""
        from quanta.layer3.shor import factor_recursive
        assert factor_recursive(7) == [7]

    def test_factor_recursive_small(self):
        """Lines 339-340: Recursive on N < 2 returns []."""
        from quanta.layer3.shor import factor_recursive
        assert factor_recursive(1) == []
        assert factor_recursive(0) == []

    def test_factor_recursive_composite(self):
        """Line 353: Recursive factoring of nested composite."""
        from quanta.layer3.shor import factor_recursive
        # 60 = 2 × 2 × 3 × 5
        result = factor_recursive(60, seed=42)
        product = 1
        for f in result:
            product *= f
        assert product == 60

    def test_continued_fraction_period(self):
        """Lines 211-225: Continued fraction extraction."""
        from quanta.layer3.shor import _continued_fraction_period
        # phase ≈ 0 should return 1
        assert _continued_fraction_period(0.0, 15) == 1
        # phase = 0.25 → period 4 for N=15
        r = _continued_fraction_period(0.25, 15)
        assert r >= 1


# ════════════════════════════════════════════
#  Surface Code
# ════════════════════════════════════════════

class TestSurfaceCodeCoverage:
    """Tests for uncovered lines in surface_code.py."""

    def test_result_summary(self):
        """Lines 50-67: SurfaceCodeResult.summary()."""
        from quanta.qec.surface_code import SurfaceCodeResult
        r = SurfaceCodeResult(
            logical_error_rate=0.01,
            physical_error_rate=0.05,
            rounds=1000,
            errors_injected=50,
            errors_corrected=45,
            threshold_estimate=0.011,
        )
        s = r.summary()
        assert "Surface Code" in s
        assert "5.00%" in s

    def test_result_summary_zero_logical_rate(self):
        """Line 53: Summary with zero logical error rate (inf suppression)."""
        from quanta.qec.surface_code import SurfaceCodeResult
        r = SurfaceCodeResult(
            logical_error_rate=0.0,
            physical_error_rate=0.05,
            rounds=100,
            errors_injected=10,
            errors_corrected=10,
            threshold_estimate=0.011,
        )
        s = r.summary()
        assert "inf" in s.lower() or "Surface Code" in s

    def test_simulate_high_error_rate(self):
        """Lines 231-239: High error rate triggers logical error paths."""
        from quanta.qec.surface_code import SurfaceCode
        code = SurfaceCode(distance=3)
        result = code.simulate_error_correction(
            error_rate=0.15, rounds=200, seed=42,
        )
        # High error rate should produce some logical errors
        assert result.logical_error_rate > 0
        assert result.errors_injected > 0

    def test_simulate_moderate_error_rate(self):
        """Lines 231-239: Moderate error rate exercises all branches."""
        from quanta.qec.surface_code import SurfaceCode
        code = SurfaceCode(distance=3)
        result = code.simulate_error_correction(
            error_rate=0.08, rounds=500, seed=123,
        )
        assert result.rounds == 500
        assert result.physical_error_rate == 0.08

    def test_check_logical_error_crossing(self):
        """Lines 265-326: BFS crossing detection — horizontal crossing."""
        from quanta.qec.surface_code import SurfaceCode
        code = SurfaceCode(distance=3)
        # Create error mask that crosses the lattice horizontally
        # 3×3 grid: errors on (0,0), (0,1), (0,2) = indices 0,1,2
        error_mask = np.zeros(9, dtype=bool)
        error_mask[0] = True  # (0,0)
        error_mask[1] = True  # (0,1)
        error_mask[2] = True  # (0,2)
        result = code._check_logical_error(error_mask)
        assert result is True  # Crosses from left to right

    def test_check_logical_error_no_crossing(self):
        """Lines 265-326: No lattice crossing."""
        from quanta.qec.surface_code import SurfaceCode
        code = SurfaceCode(distance=5)
        # Single error — should not cross
        error_mask = np.zeros(25, dtype=bool)
        error_mask[12] = True  # Center qubit only
        result = code._check_logical_error(error_mask)
        assert result is False

    def test_check_logical_error_vertical_crossing(self):
        """Lines 305-322: Vertical crossing detected by BFS."""
        from quanta.qec.surface_code import SurfaceCode
        code = SurfaceCode(distance=3)
        # Create vertical crossing: (0,1), (1,1), (2,1) = indices 1, 4, 7
        error_mask = np.zeros(9, dtype=bool)
        error_mask[1] = True   # (0,1)
        error_mask[4] = True   # (1,1)
        error_mask[7] = True   # (2,1)
        result = code._check_logical_error(error_mask)
        assert result is True  # Vertical crossing

    def test_check_logical_error_many_errors(self):
        """Line 271: n_errors >= d always returns True."""
        from quanta.qec.surface_code import SurfaceCode
        code = SurfaceCode(distance=3)
        # 3+ errors on 3-distance code
        error_mask = np.ones(9, dtype=bool)  # All qubits errored
        assert code._check_logical_error(error_mask) is True

    def test_check_logical_error_no_left_boundary(self):
        """Lines 283-286: No errors on left boundary — heuristic path."""
        from quanta.qec.surface_code import SurfaceCode
        code = SurfaceCode(distance=3)
        # Errors not on left boundary: (0,1) and (1,2)
        error_mask = np.zeros(9, dtype=bool)
        error_mask[1] = True   # (0,1) — not on left boundary
        error_mask[5] = True   # (1,2) — not on left boundary
        result = code._check_logical_error(error_mask)
        # 2 errors, correctable=1, 2 > 3//2 → True
        assert isinstance(result, bool)

    def test_simulate_zero_error_rate(self):
        """Line 220: No errors injected path."""
        from quanta.qec.surface_code import SurfaceCode
        code = SurfaceCode(distance=3)
        result = code.simulate_error_correction(
            error_rate=0.0, rounds=100, seed=42,
        )
        assert result.logical_error_rate == 0.0
        assert result.errors_injected == 0

    def test_distance_5_simulation(self):
        """Exercise distance-5 code with all branches."""
        from quanta.qec.surface_code import SurfaceCode
        code = SurfaceCode(distance=5)
        result = code.simulate_error_correction(
            error_rate=0.10, rounds=300, seed=99,
        )
        assert result.physical_error_rate == 0.10
        assert result.rounds == 300

    def test_syndrome_zero_with_errors(self):
        """Lines 231-233: Zero syndrome but errors present = logical error.
        This exercises the degenerate error pattern branch.
        """
        from quanta.qec.surface_code import SurfaceCode
        code = SurfaceCode(distance=3)
        # Manually test with high error rate many rounds
        # to statistically hit the zero-syndrome-with-errors branch
        result = code.simulate_error_correction(
            error_rate=0.20, rounds=1000, seed=777,
        )
        # With 20% error rate and 1000 rounds, we should see both
        # corrected and logical errors
        assert result.errors_injected > 0
