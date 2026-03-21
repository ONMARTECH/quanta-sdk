"""
QASM round-trip tests — export → import → compare.

Verifies that Quanta circuits survive the export/import cycle:
  Quanta circuit → QASM 3.0 string → DAGCircuit → compare gates
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quanta import CX, CZ, RX, RY, RZ, H, S, T, X, Y, Z, circuit, measure
from quanta.export.qasm import from_qasm_gates, to_qasm
from quanta.export.qasm_import import from_qasm


class TestQASMRoundTrip:
    """Round-trip: Quanta → QASM → parse back → verify gates match."""

    def test_bell_state(self):
        """Bell state: H, CX, measure."""
        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        qasm = to_qasm(bell)
        assert "OPENQASM 3.0" in qasm
        assert 'include "stdgates.inc"' in qasm
        assert "h q[0]" in qasm
        assert "cx q[0], q[1]" in qasm
        assert "measure" in qasm

        # Parse back
        dag = from_qasm(qasm)
        ops = list(dag.op_nodes())
        assert len(ops) == 2
        assert ops[0].gate_name == "H"
        assert ops[1].gate_name == "CX"

    def test_ghz_3qubit(self):
        """GHZ state: H + 2 CX."""
        @circuit(qubits=3)
        def ghz(q):
            H(q[0])
            CX(q[0], q[1])
            CX(q[1], q[2])
            return measure(q)

        qasm = to_qasm(ghz)
        dag = from_qasm(qasm)
        ops = list(dag.op_nodes())
        assert len(ops) == 3
        gate_names = [op.gate_name for op in ops]
        assert gate_names == ["H", "CX", "CX"]

    def test_rotation_gates(self):
        """Parametric gates: RX, RY, RZ with angle preservation."""
        @circuit(qubits=2)
        def rotations(q):
            RX(math.pi / 4)(q[0])
            RY(math.pi / 3)(q[1])
            RZ(math.pi / 6)(q[0])
            return measure(q)

        qasm = to_qasm(rotations)
        dag = from_qasm(qasm)
        ops = list(dag.op_nodes())
        assert len(ops) == 3

        # Check gate preservation
        assert ops[0].gate_name == "RX"
        assert ops[1].gate_name == "RY"
        assert ops[2].gate_name == "RZ"

        # Verify angles are preserved (within float precision)
        assert abs(ops[0].params[0] - math.pi / 4) < 1e-5
        assert abs(ops[1].params[0] - math.pi / 3) < 1e-5
        assert abs(ops[2].params[0] - math.pi / 6) < 1e-5

    def test_pauli_gates(self):
        """Pauli gates: X, Y, Z."""
        @circuit(qubits=3)
        def paulis(q):
            X(q[0])
            Y(q[1])
            Z(q[2])
            return measure(q)

        qasm = to_qasm(paulis)
        dag = from_qasm(qasm)
        ops = list(dag.op_nodes())
        gate_names = [op.gate_name for op in ops]
        assert gate_names == ["X", "Y", "Z"]

    def test_phase_gates(self):
        """Phase gates: S, T."""
        @circuit(qubits=2)
        def phases(q):
            S(q[0])
            T(q[1])
            return measure(q)

        qasm = to_qasm(phases)
        dag = from_qasm(qasm)
        ops = list(dag.op_nodes())
        assert len(ops) == 2
        assert ops[0].gate_name == "S"
        assert ops[1].gate_name == "T"

    def test_cz_gate(self):
        """CZ gate round-trip."""
        @circuit(qubits=2)
        def cz(q):
            H(q[0])
            CZ(q[0], q[1])
            return measure(q)

        qasm = to_qasm(cz)
        assert "cz" in qasm
        dag = from_qasm(qasm)
        ops = list(dag.op_nodes())
        assert any(op.gate_name == "CZ" for op in ops)

    def test_from_qasm_gates_helper(self):
        """Test the simple from_qasm_gates parser."""
        qasm = (
            'OPENQASM 3.0;\n'
            'include "stdgates.inc";\n'
            'qubit[2] q;\n'
            'bit[2] c;\n'
            'h q[0];\n'
            'cx q[0], q[1];\n'
            'c[0] = measure q[0];\n'
        )
        gates = from_qasm_gates(qasm)
        assert len(gates) == 2
        assert gates[0] == ("H", (0,))
        assert gates[1] == ("CX", (0, 1))

    def test_qasm2_import(self):
        """QASM 2.0 format import."""
        qasm2 = '''
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0],q[1];
            measure q -> c;
        '''
        dag = from_qasm(qasm2)
        ops = list(dag.op_nodes())
        assert len(ops) == 2
        assert ops[0].gate_name == "H"
        assert ops[1].gate_name == "CX"

    def test_empty_circuit_roundtrip(self):
        """Empty circuit (no gates) should not crash."""
        @circuit(qubits=2)
        def empty(q):
            return measure(q)

        qasm = to_qasm(empty)
        assert "OPENQASM 3.0" in qasm
        dag = from_qasm(qasm)
        assert dag.gate_count() == 0


class TestDecoderBase:
    """Tests for the DecoderBase ABC."""

    def test_decoder_base_is_abstract(self):
        """Cannot instantiate DecoderBase directly."""
        from quanta.qec.decoder import DecoderBase
        with pytest.raises(TypeError):
            DecoderBase()

    def test_mwpm_inherits_base(self):
        """MWPMDecoder is a DecoderBase."""
        from quanta.qec.decoder import DecoderBase, MWPMDecoder
        d = MWPMDecoder()
        assert isinstance(d, DecoderBase)
        assert d.name == "MWPM"

    def test_union_find_inherits_base(self):
        """UnionFindDecoder is a DecoderBase."""
        from quanta.qec.decoder import DecoderBase, UnionFindDecoder
        d = UnionFindDecoder()
        assert isinstance(d, DecoderBase)
        assert d.name == "Union-Find"

    def test_custom_decoder_plugin(self):
        """Custom decoder can be created by subclassing DecoderBase."""
        from quanta.qec.decoder import DecoderBase, DecoderResult

        class TrivialDecoder(DecoderBase):
            @property
            def name(self):
                return "trivial"

            def decode(self, syndrome, code_distance, lattice_size=None):
                return DecoderResult(correction=(), success=True, weight=0)

        d = TrivialDecoder()
        assert d.name == "trivial"
        result = d.decode(np.zeros(4, dtype=int), code_distance=3)
        assert result.success is True


class TestCircuitReprHTML:
    """Tests for CircuitDefinition._repr_html_."""

    def test_circuit_repr_html_returns_svg(self):
        """_repr_html_ returns HTML with embedded SVG."""
        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        html = bell._repr_html_()
        assert "<svg" in html
        assert "bell" in html
        assert "2q" in html

    def test_result_repr_html_returns_html(self):
        """Result._repr_html_ returns styled HTML."""
        from quanta.result import Result
        r = Result(
            counts={"00": 512, "11": 512},
            shots=1024,
            num_qubits=2,
            circuit_name="bell",
        )
        html = r._repr_html_()
        assert "bell" in html
        assert "50.0%" in html
        assert "Quanta" in html
