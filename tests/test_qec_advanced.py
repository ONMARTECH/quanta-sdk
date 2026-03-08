"""
tests/test_qec_advanced.py -- Tests for color code, decoders, Pauli frame sim.
"""

import numpy as np
import pytest

# ═══════════════════════════════════════════
#  Color Code
# ═══════════════════════════════════════════

class TestColorCode:
    """Tests for color code implementation."""

    def test_code_params_d3(self):
        from quanta.qec.color_code import ColorCode
        code = ColorCode(distance=3)
        assert code.n_data == 7
        assert code.code_params == "[[7, 1, 3]]"
        assert code.correctable_errors == 1

    def test_code_params_d5(self):
        from quanta.qec.color_code import ColorCode
        code = ColorCode(distance=5)
        assert code.n_data == 19
        assert code.correctable_errors == 2

    def test_code_params_d7(self):
        from quanta.qec.color_code import ColorCode
        code = ColorCode(distance=7)
        assert code.n_data == 37
        assert code.correctable_errors == 3

    def test_invalid_distance(self):
        from quanta.qec.color_code import ColorCode
        with pytest.raises(ValueError):
            ColorCode(distance=2)
        with pytest.raises(ValueError):
            ColorCode(distance=4)

    def test_plaquettes_d3(self):
        from quanta.qec.color_code import Color, ColorCode
        code = ColorCode(distance=3)
        assert len(code.plaquettes) == 3
        colors = {p.color for p in code.plaquettes}
        assert colors == {Color.RED, Color.GREEN, Color.BLUE}

    def test_syndrome_no_error(self):
        from quanta.qec.color_code import ColorCode
        code = ColorCode(distance=3)
        error_mask = np.zeros(7, dtype=bool)
        syndrome = code.get_syndrome(error_mask)
        assert not syndrome.any()

    def test_syndrome_single_error(self):
        from quanta.qec.color_code import ColorCode
        code = ColorCode(distance=3)
        error_mask = np.zeros(7, dtype=bool)
        error_mask[0] = True
        syndrome = code.get_syndrome(error_mask)
        # Qubit 0 is in all 3 plaquettes -> all 3 should fire
        assert syndrome.sum() == 3

    def test_restriction_decode_no_error(self):
        from quanta.qec.color_code import ColorCode
        code = ColorCode(distance=3)
        error_mask = np.zeros(7, dtype=bool)
        syndrome = code.get_syndrome(error_mask)
        assert code.restriction_decode(syndrome, error_mask) is True

    def test_restriction_decode_single_error(self):
        from quanta.qec.color_code import ColorCode
        code = ColorCode(distance=3)
        error_mask = np.zeros(7, dtype=bool)
        error_mask[1] = True  # single error, correctable
        syndrome = code.get_syndrome(error_mask)
        assert code.restriction_decode(syndrome, error_mask) is True

    def test_simulate_low_error_rate(self):
        from quanta.qec.color_code import ColorCode
        code = ColorCode(distance=3)
        result = code.simulate_error_correction(
            error_rate=0.001, rounds=500, seed=42
        )
        assert result.logical_error_rate < 0.1
        assert result.physical_error_rate == 0.001
        assert result.rounds == 500

    def test_simulate_result_summary(self):
        from quanta.qec.color_code import ColorCode
        code = ColorCode(distance=3)
        result = code.simulate_error_correction(
            error_rate=0.01, rounds=100, seed=42
        )
        summary = result.summary()
        assert "Color Code" in summary
        assert "Logical" in summary

    def test_summary(self):
        from quanta.qec.color_code import ColorCode
        code = ColorCode(distance=3)
        s = code.summary()
        assert "Color Code" in s
        assert "Transversal" in s
        assert "7" in s

    def test_n_stabilizers(self):
        from quanta.qec.color_code import ColorCode
        code = ColorCode(distance=3)
        # 3 plaquettes -> 6 stabilizers (3 X + 3 Z)
        assert code.n_stabilizers == 6

    def test_repr(self):
        from quanta.qec.color_code import ColorCode
        code = ColorCode(distance=5)
        assert "d=5" in repr(code)
        assert "19" in repr(code)


# ═══════════════════════════════════════════
#  Decoders
# ═══════════════════════════════════════════

class TestMWPMDecoder:
    """Tests for Minimum Weight Perfect Matching decoder."""

    def test_no_defects(self):
        from quanta.qec.decoder import MWPMDecoder
        decoder = MWPMDecoder()
        syndrome = np.zeros(4, dtype=bool)
        result = decoder.decode(syndrome, code_distance=3)
        assert result.success is True
        assert result.correction == ()
        assert result.weight == 0

    def test_two_defects(self):
        from quanta.qec.decoder import MWPMDecoder
        decoder = MWPMDecoder()
        syndrome = np.zeros(9, dtype=bool)
        syndrome[0] = True
        syndrome[1] = True
        result = decoder.decode(syndrome, code_distance=3)
        assert result.success is True
        assert len(result.correction) >= 1

    def test_single_defect_boundary(self):
        from quanta.qec.decoder import MWPMDecoder
        decoder = MWPMDecoder()
        syndrome = np.zeros(9, dtype=bool)
        syndrome[4] = True  # center defect
        result = decoder.decode(syndrome, code_distance=3)
        assert len(result.correction) >= 1

    def test_four_defects(self):
        from quanta.qec.decoder import MWPMDecoder
        decoder = MWPMDecoder()
        syndrome = np.zeros(25, dtype=bool)
        syndrome[0] = True
        syndrome[4] = True
        syndrome[20] = True
        syndrome[24] = True
        result = decoder.decode(syndrome, code_distance=5)
        assert len(result.correction) >= 2

    def test_greedy_matching(self):
        from quanta.qec.decoder import MWPMDecoder
        dist = np.array([[0, 1, 3, 5],
                         [1, 0, 2, 4],
                         [3, 2, 0, 1],
                         [5, 4, 1, 0]])
        pairs = MWPMDecoder._greedy_matching(dist, 4)
        assert len(pairs) == 2
        # Should match (0,1) and (2,3) for minimum weight
        total = sum(dist[i, j] for i, j in pairs)
        assert total <= 3  # optimal is 1+1=2


class TestUnionFindDecoder:
    """Tests for Union-Find decoder."""

    def test_no_defects(self):
        from quanta.qec.decoder import UnionFindDecoder
        decoder = UnionFindDecoder()
        syndrome = np.zeros(9, dtype=bool)
        result = decoder.decode(syndrome, code_distance=3)
        assert result.success is True
        assert result.correction == ()

    def test_two_adjacent_defects(self):
        from quanta.qec.decoder import UnionFindDecoder
        decoder = UnionFindDecoder()
        syndrome = np.zeros(9, dtype=bool)
        syndrome[0] = True
        syndrome[1] = True
        result = decoder.decode(syndrome, code_distance=3)
        assert result.success is True
        assert 0 in result.correction
        assert 1 in result.correction

    def test_single_defect(self):
        from quanta.qec.decoder import UnionFindDecoder
        decoder = UnionFindDecoder()
        syndrome = np.zeros(9, dtype=bool)
        syndrome[4] = True
        result = decoder.decode(syndrome, code_distance=3)
        assert 4 in result.correction

    def test_many_defects_d5(self):
        from quanta.qec.decoder import UnionFindDecoder
        decoder = UnionFindDecoder()
        syndrome = np.zeros(25, dtype=bool)
        syndrome[0] = True
        syndrome[1] = True
        syndrome[5] = True
        syndrome[6] = True
        result = decoder.decode(syndrome, code_distance=5)
        assert result.success is True
        assert len(result.correction) >= 2


# ═══════════════════════════════════════════
#  Pauli Frame Simulator
# ═══════════════════════════════════════════

class TestPauliFrameSimulator:
    """Tests for Pauli frame simulator."""

    def test_identity(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        counts = sim.sample(shots=100, seed=42)
        # |0> state -> always "0"
        assert "0" in counts
        assert counts.get("0", 0) == 100

    def test_x_gate(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        sim.x(0)
        counts = sim.sample(shots=100, seed=42)
        # X|0> = |1>
        assert counts.get("1", 0) == 100

    def test_hadamard_superposition(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        sim.h(0)
        counts = sim.sample(shots=1000, seed=42)
        # H|0> = |+> -> ~50% each
        assert "0" in counts
        assert "1" in counts
        assert abs(counts["0"] - 500) < 150

    def test_bell_state(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(2)
        sim.h(0)
        sim.cx(0, 1)
        counts = sim.sample(shots=1000, seed=42)
        # Bell state: ~50% |00>, ~50% |11>
        assert "00" in counts or "11" in counts
        n_correlated = counts.get("00", 0) + counts.get("11", 0)
        assert n_correlated == 1000  # perfect correlation

    def test_ghz_state(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(3)
        sim.h(0)
        sim.cx(0, 1)
        sim.cx(0, 2)
        counts = sim.sample(shots=1000, seed=42)
        # GHZ: |000> + |111>
        total = counts.get("000", 0) + counts.get("111", 0)
        assert total == 1000

    def test_z_gate(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        sim.z(0)
        counts = sim.sample(shots=100, seed=42)
        # Z|0> = |0>
        assert counts.get("0", 0) == 100

    def test_s_gate(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        sim.s(0)
        counts = sim.sample(shots=100, seed=42)
        # S|0> = |0>
        assert counts.get("0", 0) == 100

    def test_cz_gate(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(2)
        sim.h(0)
        sim.h(1)
        sim.cz(0, 1)
        counts = sim.sample(shots=1000, seed=42)
        # Should produce all 4 outcomes
        assert len(counts) >= 2

    def test_swap_gate(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(2)
        sim.x(0)  # |10>
        sim.swap(0, 1)  # -> |01>
        counts = sim.sample(shots=100, seed=42)
        assert counts.get("01", 0) == 100

    def test_error_injection(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        sim.inject_error(0, "X")
        counts = sim.sample(shots=100, seed=42)
        assert counts.get("1", 0) == 100

    def test_error_injection_invalid(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(1)
        with pytest.raises(ValueError, match="Unknown"):
            sim.inject_error(0, "W")

    def test_measure_subset(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(3)
        sim.x(2)
        sim.measure(0, 2)
        counts = sim.sample(shots=100, seed=42)
        # Measuring qubits 0 and 2: should get "01"
        assert counts.get("01", 0) == 100

    def test_large_circuit(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        # 50 qubits -- would be impossible with statevector
        sim = PauliFrameSimulator(50)
        sim.h(0)
        for i in range(49):
            sim.cx(i, i + 1)
        counts = sim.sample(shots=100, seed=42)
        # GHZ-50: only |0...0> and |1...1>
        total = counts.get("0" * 50, 0) + counts.get("1" * 50, 0)
        assert total == 100

    def test_repr(self):
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        sim = PauliFrameSimulator(5)
        assert "n=5" in repr(sim)
