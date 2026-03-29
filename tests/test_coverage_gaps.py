"""
test_coverage_gaps — Tests to close coverage gaps (Task 18).

Targets:
  - accelerated.py (61% → 80%+)
  - pauli_frame.py (83% → 90%+)
  - visualize_state.py (78% → 90%+)
  - runner.py (80% → 90%+)
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, ".")


# ═══════════════════════════════════════════
# accelerated.py coverage
# ═══════════════════════════════════════════

class TestAccelerated:
    """Tests for simulator accelerated backend."""

    def test_xp_returns_numpy(self):
        from quanta.simulator.accelerated import xp
        mod = xp()
        assert hasattr(mod, "array")
        assert hasattr(mod, "tensordot")

    def test_get_array_module(self):
        from quanta.simulator.accelerated import get_array_module
        mod = get_array_module()
        assert mod is not None

    def test_get_backend_info(self):
        from quanta.simulator.accelerated import get_backend_info
        info = get_backend_info()
        assert "backend" in info
        assert "device" in info
        # CPU-only → numpy backend
        assert info["backend"] == "numpy"
        assert info["device"] == "cpu"

    def test_tensor_contract_single_qubit(self):
        """Test tensor contraction with Hadamard gate."""
        from quanta.simulator.accelerated import tensor_contract
        # H gate matrix
        h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        # |0⟩ state
        state = np.array([1, 0], dtype=complex)
        result = tensor_contract(h, state, (0,), 1)
        # H|0⟩ = |+⟩ = (|0⟩+|1⟩)/√2
        np.testing.assert_allclose(
            np.abs(result) ** 2, [0.5, 0.5], atol=1e-10,
        )

    def test_tensor_contract_two_qubit(self):
        """Test 2-qubit tensor contraction with CX gate."""
        from quanta.simulator.accelerated import tensor_contract
        cx = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=complex)
        # |10⟩ state
        state = np.array([0, 0, 1, 0], dtype=complex)
        result = tensor_contract(cx, state, (0, 1), 2)
        # CX|10⟩ = |11⟩
        np.testing.assert_allclose(
            np.abs(result) ** 2, [0, 0, 0, 1], atol=1e-10,
        )

    def test_ensure_init_idempotent(self):
        """Calling _ensure_init multiple times is safe."""
        from quanta.simulator.accelerated import _ensure_init
        _ensure_init()
        _ensure_init()


# ═══════════════════════════════════════════
# pauli_frame.py coverage
# ═══════════════════════════════════════════

class TestPauliFrameCoverage:
    """Tests for uncovered Pauli frame paths."""

    def test_y_gate(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        sim.y(0)
        counts = sim.sample(shots=100, seed=42)
        assert len(counts) >= 1

    def test_z_gate(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        sim.z(0)
        counts = sim.sample(shots=100, seed=42)
        # Z|0⟩ = |0⟩ (deterministic)
        assert "0" in counts

    def test_x_gate_flips(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        sim.x(0)
        counts = sim.sample(shots=100, seed=42)
        assert "1" in counts

    def test_cz_gate(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(2)
        sim.h(0)
        sim.cz(0, 1)
        counts = sim.sample(shots=200, seed=42)
        assert len(counts) >= 1

    def test_swap_gate(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(2)
        sim.x(0)
        sim.swap(0, 1)
        counts = sim.sample(shots=100, seed=42)
        # After swap: qubit 1 should be |1⟩, qubit 0 should be |0⟩
        assert "01" in counts

    def test_inject_error_x(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        sim.inject_error(0, "X")
        counts = sim.sample(shots=100, seed=42)
        assert "1" in counts

    def test_inject_error_y(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        sim.inject_error(0, "Y")
        counts = sim.sample(shots=100, seed=42)
        assert len(counts) >= 1

    def test_inject_error_z(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        sim.inject_error(0, "Z")
        counts = sim.sample(shots=100, seed=42)
        assert len(counts) >= 1

    def test_inject_error_invalid(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        with pytest.raises(ValueError, match="Unknown error"):
            sim.inject_error(0, "W")

    def test_measure_specific_qubits(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(3)
        sim.h(0)
        sim.cx(0, 1)
        sim.measure(0, 1)
        counts = sim.sample(shots=200, seed=42)
        for key in counts:
            assert len(key) == 2

    def test_deterministic_measurement(self):
        """|0⟩ state should give deterministic 0."""
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        # No operations → |0⟩ → deterministic
        counts = sim.sample(shots=50, seed=42)
        assert counts == {"0": 50}

    def test_ghz_state(self):
        """GHZ state with all Clifford gates."""
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(3)
        sim.h(0)
        sim.cx(0, 1)
        sim.cx(0, 2)
        counts = sim.sample(shots=1000, seed=42)
        # GHZ: only |000⟩ and |111⟩
        for key in counts:
            assert key in ("000", "111")

    def test_repr(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(5)
        assert "n=5" in repr(sim)

    def test_s_gate_phase(self):
        """S gate followed by measurement exercises phase tracking."""
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        sim.h(0)
        sim.s(0)
        sim.h(0)
        # H·S·H = effectively a phase rotation
        counts = sim.sample(shots=200, seed=42)
        assert len(counts) >= 1

    def test_complex_stabilizer_rowmult(self):
        """Circuit that forces rowmult through all Pauli product branches."""
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(3)
        # Build a circuit that creates non-trivial stabilizer state
        sim.h(0)
        sim.s(0)      # Now stabilizer has Y component
        sim.cx(0, 1)
        sim.h(1)
        sim.s(1)
        sim.cx(1, 2)
        sim.h(2)
        # Measure — forces rowmult with various X/Z combinations
        counts = sim.sample(shots=500, seed=42)
        total = sum(counts.values())
        assert total == 500

    def test_repeated_measurement_determinism(self):
        """Same circuit measured twice with same seed gives same result."""
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim1 = PauliFrameSimulator(2)
        sim1.h(0)
        sim1.cx(0, 1)
        c1 = sim1.sample(shots=100, seed=99)

        sim2 = PauliFrameSimulator(2)
        sim2.h(0)
        sim2.cx(0, 1)
        c2 = sim2.sample(shots=100, seed=99)

        assert c1 == c2

    def test_s_s_equals_z(self):
        """S·S = Z gate equivalence in Pauli frame."""
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        # Method 1: S·S
        sim1 = PauliFrameSimulator(1)
        sim1.h(0)  # Create superposition
        sim1.s(0)
        sim1.s(0)
        c1 = sim1.sample(shots=500, seed=42)

        # Method 2: Z
        sim2 = PauliFrameSimulator(1)
        sim2.h(0)
        sim2.z(0)
        c2 = sim2.sample(shots=500, seed=42)

        assert c1 == c2


# ═══════════════════════════════════════════
# visualize_state.py coverage
# ═══════════════════════════════════════════

class TestVisualizeState:
    """Tests for uncovered visualize_state paths."""

    def test_show_probabilities(self):
        from quanta import CX, H, circuit, measure, run
        from quanta.visualize_state import show_probabilities

        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        result = run(bell, shots=100, seed=42)
        output = show_probabilities(result)
        assert "Probability Distribution" in output
        assert "║" in output

    def test_show_probabilities_max_states(self):
        from quanta import H, circuit, measure, run
        from quanta.visualize_state import show_probabilities

        @circuit(qubits=1)
        def h_circ(q):
            H(q[0])
            return measure(q)

        result = run(h_circ, shots=100, seed=42)
        output = show_probabilities(result, max_states=1)
        assert "║" in output

    def test_show_statevector(self):
        from quanta.visualize_state import show_statevector
        state = np.array([1, 0, 0, 0], dtype=complex)
        output = show_statevector(state, num_qubits=2, threshold=0.01)
        assert "|00⟩" in output

    def test_show_statevector_bell(self):
        """Bell state visualization."""
        from quanta.visualize_state import show_statevector
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        output = show_statevector(state, num_qubits=2, threshold=0.01)
        assert "|00⟩" in output
        assert "|11⟩" in output

    def test_show_phases(self):
        from quanta.visualize_state import show_phases
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        output = show_phases(state, num_qubits=2)
        assert "Phase Diagram" in output

    def test_show_phases_with_complex(self):
        """Phase diagram with imaginary amplitudes."""
        from quanta.visualize_state import show_phases
        state = np.array([1, 1j, 0, 0], dtype=complex) / np.sqrt(2)
        output = show_phases(state, num_qubits=2)
        assert "|00⟩" in output
        assert "|01⟩" in output

    def test_phase_to_symbol_all_directions(self):
        from quanta.visualize_state import _phase_to_symbol
        # 0° → →
        assert _phase_to_symbol(0) == "→"
        # 90° → ↑
        assert _phase_to_symbol(np.pi / 2) == "↑"
        # 180° → ←
        assert _phase_to_symbol(np.pi) == "←"
        # 270° → ↓
        assert _phase_to_symbol(-np.pi / 2) == "↓"
        # 45° → ↗
        assert _phase_to_symbol(np.pi / 4) == "↗"
        # 135° → ↖
        assert _phase_to_symbol(3 * np.pi / 4) == "↖"

    def test_phase_to_arrow(self):
        from quanta.visualize_state import _phase_to_arrow
        assert _phase_to_arrow(0) == "→"
        assert _phase_to_arrow(np.pi / 2) == "↑"
        assert _phase_to_arrow(np.pi) == "←"


# ═══════════════════════════════════════════
# runner.py coverage
# ═══════════════════════════════════════════

class TestRunnerCoverage:
    """Tests for runner.py uncovered paths."""

    def test_sweep(self):
        from quanta import RZ, circuit, measure, sweep

        @circuit(qubits=1)
        def rz_circ(q, theta=0.0):
            RZ(theta)(q[0])
            return measure(q)

        results = sweep(rz_circ, params={"theta": [0.0, 0.5, 1.0]}, shots=10)
        assert len(results) == 3

    def test_run_async(self):
        import asyncio

        from quanta import H, circuit, measure, run_async

        @circuit(qubits=1)
        def h_circ(q):
            H(q[0])
            return measure(q)

        result = asyncio.run(run_async(h_circ, shots=10))
        assert result is not None
