"""
tests/test_qec_surface.py — Surface code tests.

Validates:
  - Surface code parameters ([[n,k,d]])
  - Error correction simulation
  - Threshold behavior
"""

import pytest
import numpy as np

from quanta.qec.surface_code import SurfaceCode


class TestSurfaceCode:
    """Surface code error correction tests."""

    def test_distance_3_params(self):
        """[[9,1,3]]: 9 physical qubits, 1 logical, distance 3."""
        code = SurfaceCode(distance=3)
        assert code.n_physical == 9
        assert code.n_logical == 1
        assert code.correctable_errors == 1
        assert "[[9, 1, 3]]" in code.code_params

    def test_distance_5_params(self):
        """[[25,1,5]]: 25 physical qubits, corrects 2 errors."""
        code = SurfaceCode(distance=5)
        assert code.n_physical == 25
        assert code.correctable_errors == 2

    def test_distance_7_params(self):
        code = SurfaceCode(distance=7)
        assert code.n_physical == 49
        assert code.correctable_errors == 3

    def test_invalid_distance_raises(self):
        """Even or < 3 distance should raise error."""
        with pytest.raises(ValueError):
            SurfaceCode(distance=2)
        with pytest.raises(ValueError):
            SurfaceCode(distance=4)

    def test_error_correction_below_threshold(self):
        """Below threshold: logical error rate < physical error rate."""
        code = SurfaceCode(distance=5)
        result = code.simulate_error_correction(
            error_rate=0.001, rounds=2000, seed=42
        )
        assert result.logical_error_rate <= result.physical_error_rate

    def test_error_correction_rounds(self):
        code = SurfaceCode(distance=3)
        result = code.simulate_error_correction(rounds=500, seed=42)
        assert result.rounds == 500

    def test_summary_string(self):
        code = SurfaceCode(distance=3)
        s = code.summary()
        assert "Surface Code" in s
        assert "Physical qubits" in s

    def test_repr(self):
        code = SurfaceCode(distance=3)
        assert "SurfaceCode" in repr(code)

    def test_correction_result_summary(self):
        code = SurfaceCode(distance=3)
        result = code.simulate_error_correction(rounds=100, seed=42)
        assert "Surface Code" in result.summary()

    def test_zero_error_rate(self):
        """Zero errors → zero logical errors."""
        code = SurfaceCode(distance=3)
        result = code.simulate_error_correction(error_rate=0.0, rounds=100, seed=42)
        assert result.logical_error_rate == 0
        assert result.errors_injected == 0


# ═══════════════════════════════════════════
#  QEC Codes -- Syndrome & Correction Tests
# ═══════════════════════════════════════════

class TestBitFlipCodeSyndrome:
    """Tests BitFlipCode syndrome measurement and encoding."""

    def test_encode_circuit_structure(self):
        from quanta.qec.codes import BitFlipCode
        code = BitFlipCode()
        enc = code.encode()
        dag = enc.build()
        ops = list(dag.instructions)
        # Should have 2 CX gates
        cx_ops = [op for op in ops if op.gate_name == "CX"]
        assert len(cx_ops) == 2

    def test_syndrome_circuit_structure(self):
        from quanta.qec.codes import BitFlipCode
        code = BitFlipCode()
        syn = code.syndrome_measure()
        dag = syn.build()
        ops = list(dag.instructions)
        # Should have 4 CX gates (2 per syndrome qubit)
        cx_ops = [op for op in ops if op.gate_name == "CX"]
        assert len(cx_ops) == 4

    def test_syndrome_returns_circuit(self):
        from quanta.qec.codes import BitFlipCode, QECCode
        from quanta.core.circuit import CircuitDefinition
        code = BitFlipCode()
        syn = code.syndrome_measure()
        assert isinstance(syn, CircuitDefinition)

    def test_syndrome_qubit_count(self):
        from quanta.qec.codes import BitFlipCode
        code = BitFlipCode()
        syn = code.syndrome_measure()
        dag = syn.build()
        assert dag.num_qubits == 5  # 3 code + 2 ancilla

    def test_info(self):
        from quanta.qec.codes import BitFlipCode
        code = BitFlipCode()
        info = code.info
        assert info.n == 3
        assert info.k == 1
        assert info.d == 3
        assert "BitFlip" in info.name
        assert info.correctable_errors == 1  # d=3 -> (3-1)/2 = 1

    def test_info_repr(self):
        from quanta.qec.codes import BitFlipCode
        info = BitFlipCode().info
        r = repr(info)
        assert "[[3,1,3]]" in r
        assert "BitFlip" in r


class TestPhaseFlipCodeCoverage:
    """Tests PhaseFlipCode encoding and info."""

    def test_encode_has_hadamards(self):
        from quanta.qec.codes import PhaseFlipCode
        code = PhaseFlipCode()
        enc = code.encode()
        dag = enc.build()
        ops = list(dag.instructions)
        h_ops = [op for op in ops if op.gate_name == "H"]
        assert len(h_ops) == 3  # H on all 3 qubits

    def test_encode_has_cx(self):
        from quanta.qec.codes import PhaseFlipCode
        code = PhaseFlipCode()
        enc = code.encode()
        dag = enc.build()
        ops = list(dag.instructions)
        cx_ops = [op for op in ops if op.gate_name == "CX"]
        assert len(cx_ops) == 2

    def test_info(self):
        from quanta.qec.codes import PhaseFlipCode
        info = PhaseFlipCode().info
        assert info.n == 3 and info.k == 1 and info.d == 3
        assert "PhaseFlip" in info.name


class TestSteaneCodeCoverage:
    """Tests Steane code syndrome measurement and structure."""

    def test_encode_qubit_count(self):
        from quanta.qec.codes import SteaneCode
        code = SteaneCode()
        enc = code.encode()
        dag = enc.build()
        assert dag.num_qubits == 7

    def test_syndrome_qubit_count(self):
        from quanta.qec.codes import SteaneCode
        code = SteaneCode()
        syn = code.syndrome_measure()
        dag = syn.build()
        assert dag.num_qubits == 13  # 7 code + 6 syndrome

    def test_syndrome_has_cx_gates(self):
        from quanta.qec.codes import SteaneCode
        code = SteaneCode()
        syn = code.syndrome_measure()
        dag = syn.build()
        ops = list(dag.instructions)
        cx_ops = [op for op in ops if op.gate_name == "CX"]
        # X stabilizers: 3 * 4 = 12 CX, Z stabilizers: 3 * 4 = 12 CX
        assert len(cx_ops) == 24

    def test_syndrome_has_hadamards(self):
        from quanta.qec.codes import SteaneCode
        code = SteaneCode()
        syn = code.syndrome_measure()
        dag = syn.build()
        ops = list(dag.instructions)
        h_ops = [op for op in ops if op.gate_name == "H"]
        # Z stabilizers need H before and after: 3 * 2 = 6
        assert len(h_ops) == 6

    def test_info(self):
        from quanta.qec.codes import SteaneCode
        info = SteaneCode().info
        assert info.n == 7, info.k == 1
        assert info.d == 3
        assert info.correctable_errors == 1
        assert "Steane" in repr(info)

