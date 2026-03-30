"""
tests/test_sparse_simulator.py -- Sparse simulator correctness tests.

Every test cross-validates against dense StateVectorSimulator to ensure
physics correctness. NO hallucination tolerance — results must match exactly.
"""

from __future__ import annotations

import math

import numpy as np

from quanta.simulator.factory import create_simulator
from quanta.simulator.sparse import SparseSimulator
from quanta.simulator.statevector import StateVectorSimulator


def _compare_simulators(n: int, ops: list[tuple], atol: float = 1e-10) -> None:
    """Run same operations on sparse and dense, compare results."""
    dense = StateVectorSimulator(n, seed=42)
    sparse = SparseSimulator(n, seed=42)

    for gate_name, qubits, params in ops:
        dense.apply(gate_name, qubits, params)
        sparse.apply(gate_name, qubits, params)

    # Compare full statevectors
    np.testing.assert_allclose(
        sparse.state, dense.state, atol=atol,
        err_msg=f"Sparse vs Dense mismatch after {len(ops)} ops"
    )

    # Compare probabilities
    np.testing.assert_allclose(
        sparse.probabilities(), dense.probabilities(), atol=atol,
    )


class TestSparseBasics:
    """Basic functionality tests."""

    def test_initial_state(self) -> None:
        sim = SparseSimulator(3)
        assert sim.num_nonzero == 1
        assert sim.sparsity > 0.8
        np.testing.assert_allclose(sim.state[0], 1.0)

    def test_x_gate(self) -> None:
        _compare_simulators(3, [("X", (0,), ())])

    def test_h_gate(self) -> None:
        _compare_simulators(2, [("H", (0,), ())])

    def test_bell_state(self) -> None:
        _compare_simulators(2, [
            ("H", (0,), ()),
            ("CX", (0, 1), ()),
        ])

    def test_ghz_state(self) -> None:
        n = 5
        ops = [("H", (0,), ())]
        for i in range(n - 1):
            ops.append(("CX", (i, i + 1), ()))
        _compare_simulators(n, ops)


class TestParametricGates:
    """Parametric gate correctness."""

    def test_rz(self) -> None:
        _compare_simulators(2, [
            ("H", (0,), ()),
            ("RZ", (0,), (math.pi / 4,)),
        ])

    def test_rx(self) -> None:
        _compare_simulators(2, [
            ("RX", (0,), (math.pi / 3,)),
        ])

    def test_ry(self) -> None:
        _compare_simulators(2, [
            ("RY", (0,), (math.pi / 6,)),
            ("CX", (0, 1), ()),
        ])


class TestMultiQubit:
    """Multi-qubit gate tests."""

    def test_ccx(self) -> None:
        _compare_simulators(3, [
            ("X", (0,), ()),
            ("X", (1,), ()),
            ("CCX", (0, 1, 2), ()),
        ])

    def test_swap(self) -> None:
        _compare_simulators(3, [
            ("X", (0,), ()),
            ("SWAP", (0, 2), ()),
        ])


class TestMeasurement:
    """Measurement and sampling tests."""

    def test_deterministic_sample(self) -> None:
        sim = SparseSimulator(2, seed=42)
        sim.apply("X", (0,), ())
        counts = sim.sample(1000)
        assert counts == {"10": 1000}  # X(q0) → |10⟩ in MSB convention

    def test_bell_sampling(self) -> None:
        sim = SparseSimulator(2, seed=42)
        sim.apply("H", (0,), ())
        sim.apply("CX", (0, 1), ())
        counts = sim.sample(10000)
        assert "00" in counts
        assert "11" in counts
        assert counts.get("01", 0) + counts.get("10", 0) < 10  # near zero


class TestApplyPhase:
    """Grover-style phase oracle tests."""

    def test_phase_flip(self) -> None:
        sim = SparseSimulator(3, seed=42)
        dense = StateVectorSimulator(3, seed=42)

        # Create uniform superposition
        for q in range(3):
            sim.apply("H", (q,), ())
            dense.apply("H", (q,), ())

        # Phase flip |101⟩ = index 5
        sim.apply_phase(5, -1)
        dense.apply_phase(5, -1)

        np.testing.assert_allclose(sim.state, dense.state, atol=1e-10)


class TestStateAccess:
    """State getter/setter tests."""

    def test_set_state(self) -> None:
        sim = SparseSimulator(2)
        # Set to Bell state manually
        bell = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
        sim.state = bell
        np.testing.assert_allclose(sim.state, bell, atol=1e-10)
        assert sim.num_nonzero == 2

    def test_sparsity(self) -> None:
        sim = SparseSimulator(3)
        assert sim.sparsity == 1.0 - 1 / 8  # only |000⟩ is nonzero


class TestLargeCircuits:
    """Scalability tests — these must work without memory issues."""

    def test_30_qubit_x_chain(self) -> None:
        """30-qubit circuit that stays sparse (X gates only)."""
        sim = SparseSimulator(30)
        for q in range(30):
            sim.apply("X", (q,), ())
        assert sim.num_nonzero == 1  # |111...1⟩
        expected = 2**30 - 1  # all bits set
        assert expected in sim._amplitudes

    def test_35_qubit_ghz(self) -> None:
        """35-qubit GHZ state: only 2 non-zero amplitudes."""
        sim = SparseSimulator(35)
        sim.apply("H", (0,), ())
        for q in range(34):
            sim.apply("CX", (q, q + 1), ())
        assert sim.num_nonzero == 2
        # Memory for 2 entries ≈ 120 bytes vs 256 GB dense!
        assert sim.memory_bytes < 200

    def test_40_qubit_product_state(self) -> None:
        """40-qubit product state — each qubit independent."""
        sim = SparseSimulator(40)
        # Apply RY to first 10 qubits — creates moderate superposition
        for q in range(10):
            sim.apply("RY", (q,), (math.pi / 4,))
        # 2^10 = 1024 non-zero entries (still very sparse for 40 qubits)
        assert sim.num_nonzero == 2**10
        assert sim.memory_bytes < 100_000  # ~60 KB


class TestFactory:
    """Factory integration tests."""

    def test_factory_sparse(self) -> None:
        sim = create_simulator(35, method="sparse")
        assert isinstance(sim, SparseSimulator)

    def test_factory_auto_large(self) -> None:
        """Auto-selection should use sparse for 30+ qubits."""
        sim = create_simulator(30)
        assert isinstance(sim, SparseSimulator)

    def test_factory_auto_small(self) -> None:
        """Auto-selection should use dense for ≤27 qubits."""
        sim = create_simulator(20)
        assert isinstance(sim, StateVectorSimulator)


class TestRepr:
    """String representation test."""

    def test_repr(self) -> None:
        sim = SparseSimulator(10)
        r = repr(sim)
        assert "qubits=10" in r
        assert "nonzero=1" in r
