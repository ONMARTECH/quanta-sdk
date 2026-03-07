"""
tests/test_coverage_gaps.py -- Tests for previously untested modules.

Covers: DensityMatrixSimulator, visualize.draw, Result display, CustomGate.
"""

import numpy as np
import pytest


# ═══════════════════════════════════════════
#  DensityMatrixSimulator
# ═══════════════════════════════════════════

class TestDensityMatrixSimulator:
    """Tests for density matrix simulator (82 lines, was 0%)."""

    def test_init_ground_state(self):
        from quanta.simulator.density_matrix import DensityMatrixSimulator
        sim = DensityMatrixSimulator(1)
        rho = sim.state
        assert rho[0, 0] == 1.0
        assert rho.shape == (2, 2)

    def test_init_max_qubits(self):
        from quanta.simulator.density_matrix import DensityMatrixSimulator, DensityMatrixError
        with pytest.raises(DensityMatrixError):
            DensityMatrixSimulator(14)

    def test_apply_h_gate(self):
        from quanta.simulator.density_matrix import DensityMatrixSimulator
        sim = DensityMatrixSimulator(1)
        sim.apply("H", (0,))
        probs = sim.probabilities()
        assert abs(probs[0] - 0.5) < 0.01
        assert abs(probs[1] - 0.5) < 0.01

    def test_apply_x_gate(self):
        from quanta.simulator.density_matrix import DensityMatrixSimulator
        sim = DensityMatrixSimulator(1)
        sim.apply("X", (0,))
        probs = sim.probabilities()
        assert abs(probs[0]) < 0.01
        assert abs(probs[1] - 1.0) < 0.01

    def test_apply_cx(self):
        from quanta.simulator.density_matrix import DensityMatrixSimulator
        sim = DensityMatrixSimulator(2)
        sim.apply("H", (0,))
        sim.apply("CX", (0, 1))
        probs = sim.probabilities()
        # Bell state: |00> + |11>
        assert abs(probs[0] - 0.5) < 0.01  # |00>
        assert abs(probs[3] - 0.5) < 0.01  # |11>

    def test_apply_parametric_gate(self):
        from quanta.simulator.density_matrix import DensityMatrixSimulator
        sim = DensityMatrixSimulator(1)
        sim.apply("RY", (0,), (np.pi,))
        probs = sim.probabilities()
        assert abs(probs[1] - 1.0) < 0.01

    def test_unknown_gate_raises(self):
        from quanta.simulator.density_matrix import DensityMatrixSimulator, DensityMatrixError
        sim = DensityMatrixSimulator(1)
        with pytest.raises(DensityMatrixError, match="Unknown"):
            sim.apply("FOOBAR", (0,))

    def test_apply_kraus(self):
        from quanta.simulator.density_matrix import DensityMatrixSimulator
        sim = DensityMatrixSimulator(1)
        # Bit flip channel: 50% chance of X
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        kraus = [np.sqrt(0.5) * I, np.sqrt(0.5) * X]
        sim.apply_kraus(kraus, (0,))
        probs = sim.probabilities()
        assert abs(probs[0] - 0.5) < 0.1
        assert abs(probs[1] - 0.5) < 0.1

    def test_depolarizing_noise(self):
        from quanta.simulator.density_matrix import DensityMatrixSimulator
        sim = DensityMatrixSimulator(1)
        sim.apply_depolarizing(0, p=0.1)
        # State should still be mostly |0> but slightly mixed
        probs = sim.probabilities()
        assert probs[0] > 0.9

    def test_depolarizing_zero_noise(self):
        from quanta.simulator.density_matrix import DensityMatrixSimulator
        sim = DensityMatrixSimulator(1)
        sim.apply_depolarizing(0, p=0.0)
        probs = sim.probabilities()
        assert abs(probs[0] - 1.0) < 0.01

    def test_purity_pure_state(self):
        from quanta.simulator.density_matrix import DensityMatrixSimulator
        sim = DensityMatrixSimulator(1)
        assert abs(sim.purity - 1.0) < 0.01

    def test_purity_mixed_state(self):
        from quanta.simulator.density_matrix import DensityMatrixSimulator
        sim = DensityMatrixSimulator(1)
        sim.apply_depolarizing(0, p=0.5)
        assert sim.purity < 1.0

    def test_sample(self):
        from quanta.simulator.density_matrix import DensityMatrixSimulator
        sim = DensityMatrixSimulator(1)
        counts = sim.sample(100)
        assert counts.get("0", 0) == 100

    def test_sample_superposition(self):
        from quanta.simulator.density_matrix import DensityMatrixSimulator
        sim = DensityMatrixSimulator(1, seed=42)
        sim.apply("H", (0,))
        counts = sim.sample(1000)
        assert "0" in counts and "1" in counts


# ═══════════════════════════════════════════
#  Visualize (draw)
# ═══════════════════════════════════════════

class TestVisualize:
    """Tests for ASCII circuit drawing (81 lines, was 0%)."""

    def test_draw_bell(self):
        from quanta.visualize import draw
        from quanta.core.circuit import circuit
        from quanta.core.gates import H, CX
        from quanta.core.measure import measure

        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        output = draw(bell)
        assert "q[0]" in output
        assert "q[1]" in output
        assert "H" in output
        assert "M" in output

    def test_draw_single_qubit(self):
        from quanta.visualize import draw
        from quanta.core.circuit import circuit
        from quanta.core.gates import X
        from quanta.core.measure import measure

        @circuit(qubits=1)
        def c(q):
            X(q[0])
            return measure(q)

        output = draw(c)
        assert "X" in output

    def test_draw_three_qubit(self):
        from quanta.visualize import draw
        from quanta.core.circuit import circuit
        from quanta.core.gates import H, CX, X
        from quanta.core.measure import measure

        @circuit(qubits=3)
        def c(q):
            H(q[0])
            CX(q[0], q[2])
            X(q[1])
            return measure(q)

        output = draw(c)
        assert "q[0]" in output
        assert "q[2]" in output

    def test_draw_swap(self):
        from quanta.visualize import draw
        from quanta.core.circuit import circuit
        from quanta.core.gates import SWAP
        from quanta.core.measure import measure

        @circuit(qubits=2)
        def c(q):
            SWAP(q[0], q[1])
            return measure(q)

        output = draw(c)
        assert "x" in output

    def test_draw_parametric(self):
        from quanta.visualize import draw
        from quanta.core.circuit import circuit
        from quanta.core.gates import RX
        from quanta.core.measure import measure

        @circuit(qubits=1)
        def c(q):
            RX(0.5)(q[0])
            return measure(q)

        output = draw(c)
        assert "Rx" in output

    def test_draw_vertical_connections(self):
        from quanta.visualize import draw
        from quanta.core.circuit import circuit
        from quanta.core.gates import CX
        from quanta.core.measure import measure

        @circuit(qubits=3)
        def c(q):
            CX(q[0], q[2])  # skip qubit 1 → vertical line
            return measure(q)

        output = draw(c)
        lines = output.split("\n")
        assert len(lines) == 3
        # Middle line (q[1]) should have vertical connector
        assert "┼" in lines[1]


# ═══════════════════════════════════════════
#  Result display methods
# ═══════════════════════════════════════════

class TestResultDisplay:
    """Tests for Result display methods (80 lines, was 49%)."""

    def _make_result(self, with_sv=True):
        from quanta.result import Result
        sv = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]) if with_sv else None
        return Result(
            counts={"00": 500, "11": 500},
            shots=1000,
            num_qubits=2,
            circuit_name="test_circuit",
            gate_count=2,
            depth=2,
            statevector=sv,
        )

    def test_dirac_with_statevector(self):
        r = self._make_result(with_sv=True)
        dirac = r.dirac_notation()
        assert "|00>" in dirac
        assert "|11>" in dirac
        assert "0.707" in dirac

    def test_dirac_without_statevector(self):
        r = self._make_result(with_sv=False)
        dirac = r.dirac_notation()
        assert "|00>" in dirac
        assert "|11>" in dirac

    def test_dirac_imaginary_amplitude(self):
        from quanta.result import Result
        sv = np.array([0, 1j], dtype=complex)
        r = Result(counts={"1": 100}, shots=100, num_qubits=1, statevector=sv)
        dirac = r.dirac_notation()
        assert "j" in dirac

    def test_dirac_complex_amplitude(self):
        from quanta.result import Result
        sv = np.array([0.5+0.5j, 0.5-0.5j], dtype=complex)
        r = Result(counts={"0": 50, "1": 50}, shots=100, num_qubits=1, statevector=sv)
        dirac = r.dirac_notation()
        assert "|0>" in dirac

    def test_dirac_empty_statevector(self):
        from quanta.result import Result
        sv = np.array([0, 0], dtype=complex)
        r = Result(counts={"0": 100}, shots=100, num_qubits=1, statevector=sv)
        dirac = r.dirac_notation()
        assert "|0>" in dirac

    def test_histogram(self):
        r = self._make_result()
        hist = r.histogram()
        assert "█" in hist
        assert "|00>" in hist
        assert "50.0%" in hist

    def test_histogram_empty(self):
        from quanta.result import Result
        r = Result(counts={}, shots=0, num_qubits=1)
        assert "No measurement" in r.histogram()

    def test_summary(self):
        r = self._make_result()
        s = r.summary()
        assert "test_circuit" in s
        assert "Qubits" in s
        assert "║" in s

    def test_summary_many_states(self):
        from quanta.result import Result
        counts = {format(i, "04b"): 1 for i in range(20)}
        r = Result(counts=counts, shots=20, num_qubits=4, circuit_name="big")
        s = r.summary()
        assert "+4 more" in s or "more states" in s

    def test_str(self):
        r = self._make_result()
        s = str(r)
        assert "Quanta Result" in s

    def test_repr(self):
        r = self._make_result()
        rp = repr(r)
        assert "Result(" in rp
        assert "test_circuit" in rp

    def test_most_frequent(self):
        r = self._make_result()
        mf = r.most_frequent
        assert mf in ("00", "11")

    def test_probabilities(self):
        r = self._make_result()
        probs = r.probabilities
        assert abs(probs["00"] - 0.5) < 0.01
        assert abs(probs["11"] - 0.5) < 0.01


# ═══════════════════════════════════════════
#  CustomGate
# ═══════════════════════════════════════════

class TestCustomGate:
    """Tests for custom gate creation (37 lines, was 43%)."""

    def test_create_valid_gate(self):
        from quanta.core.custom_gate import custom_gate
        from quanta.core.gates import GATE_REGISTRY
        # sqrt(X) gate
        mat = [[0.5+0.5j, 0.5-0.5j], [0.5-0.5j, 0.5+0.5j]]
        gate = custom_gate("SqrtX_test", mat)
        assert gate.name == "SqrtX_test"
        assert gate.num_qubits == 1
        assert "SqrtX_test" in GATE_REGISTRY
        # Cleanup
        del GATE_REGISTRY["SqrtX_test"]

    def test_not_square_raises(self):
        from quanta.core.custom_gate import custom_gate, CustomGateError
        with pytest.raises(CustomGateError, match="square"):
            custom_gate("bad", [[1, 0, 0], [0, 1, 0]])

    def test_not_power_of_2_raises(self):
        from quanta.core.custom_gate import custom_gate, CustomGateError
        with pytest.raises(CustomGateError, match="power"):
            custom_gate("bad3", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_not_unitary_raises(self):
        from quanta.core.custom_gate import custom_gate, CustomGateError
        with pytest.raises(CustomGateError, match="unitary"):
            custom_gate("bad_u", [[2, 0], [0, 1]])

    def test_duplicate_name_raises(self):
        from quanta.core.custom_gate import custom_gate, CustomGateError
        with pytest.raises(CustomGateError, match="already registered"):
            custom_gate("H", [[1, 0], [0, 1]])  # H already exists

    def test_gate_repr(self):
        from quanta.core.custom_gate import CustomGate
        mat = np.array([[1, 0], [0, 1j]], dtype=complex)
        g = CustomGate("TestRepr", mat)
        assert "TestRepr" in repr(g)
        assert "1q" in repr(g)

    def test_two_qubit_custom_gate(self):
        from quanta.core.custom_gate import custom_gate
        from quanta.core.gates import GATE_REGISTRY
        # SWAP equivalent
        mat = [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]
        gate = custom_gate("MySwap_test", mat)
        assert gate.num_qubits == 2
        del GATE_REGISTRY["MySwap_test"]

    def test_gate_in_circuit(self):
        from quanta.core.custom_gate import custom_gate
        from quanta.core.gates import GATE_REGISTRY
        from quanta.core.circuit import circuit
        from quanta.core.measure import measure
        from quanta.runner import run

        mat = [[1, 0], [0, 1j]]
        SqrtZ = custom_gate("SqrtZ_c", mat)

        @circuit(qubits=1)
        def c(q):
            SqrtZ(q[0])
            return measure(q)

        result = run(c, shots=100)
        assert result.counts.get("0", 0) > 0
        del GATE_REGISTRY["SqrtZ_c"]
