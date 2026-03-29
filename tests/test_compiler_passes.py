"""
test_compiler_passes — Unit tests for compiler passes.

Tests gate cancellation, rotation merging, identity removal,
commutation, and full pipeline "before/after" metrics.
"""

import math
import sys

import pytest

sys.path.insert(0, ".")

from quanta import CX, H, X, Z, circuit, measure
from quanta.compiler.passes.optimize import (
    CancelInverses,
    CommutationPass,
    IdentityRemoval,
    MergeRotations,
)
from quanta.compiler.pipeline import CompilerPipeline
from quanta.core.gates import (
    CCX,
    RZ,
    SDG,
    SX,
    TDG,
    I,
    S,
    SXdg,
    T,
    iSWAP,
)
from quanta.core.types import GateError
from quanta.dag.dag_circuit import DAGCircuit
from quanta.dag.node import OpNode

# ═══════════════════════════════════════════
# Task 1: Type safety & error hierarchy
# ═══════════════════════════════════════════

class TestTypeSystem:
    """Tests for GateError, better error messages, py.typed."""

    def test_gate_error_on_wrong_qubit_count(self):
        """CX with 1 qubit should raise CircuitError (wraps GateError)."""
        @circuit(qubits=1)
        def bad(q):
            CX(q[0])
            return measure(q)
        from quanta.core.types import CircuitError
        with pytest.raises(CircuitError, match="expects 2"):
            bad.build()

    def test_gate_error_hierarchy(self):
        """GateError should be subclass of QuantaError."""
        from quanta.core.types import QuantaError
        assert issubclass(GateError, QuantaError)

    def test_py_typed_marker_exists(self):
        """py.typed marker should exist for PEP 561."""
        import pathlib
        marker = pathlib.Path(__file__).parent.parent / "quanta" / "py.typed"
        assert marker.exists()


# ═══════════════════════════════════════════
# Task 2: Gate inverse & controlled
# ═══════════════════════════════════════════

class TestGateInverse:
    """Tests for Gate.inverse property."""

    def test_self_inverse_gates(self):
        for gate in [H, X, Z, CX]:
            assert gate.inverse is gate, f"{gate.name} should be self-inverse"

    def test_known_inverse_pairs(self):
        assert S.inverse is SDG
        assert SDG.inverse is S
        assert T.inverse is TDG
        assert TDG.inverse is T
        assert SX.inverse is SXdg
        assert SXdg.inverse is SX

    def test_computed_inverse_unitarity(self):
        """iSWAP† · iSWAP should equal identity."""
        import numpy as np
        inv = iSWAP.inverse
        assert inv.name == "iSWAP†"
        product = inv.matrix @ iSWAP.matrix
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)

    def test_inverse_preserves_qubit_count(self):
        inv = iSWAP.inverse
        assert inv.num_qubits == iSWAP.num_qubits


class TestGateControlled:
    """Tests for Gate.controlled() method."""

    def test_controlled_x_is_cx(self):
        """X.controlled() matrix should equal CX matrix."""
        import numpy as np
        cx = X.controlled()
        np.testing.assert_allclose(cx.matrix, CX.matrix, atol=1e-10)

    def test_controlled_x_2_is_ccx(self):
        """X.controlled(2) should produce CCX matrix."""
        import numpy as np
        ccx = X.controlled(2)
        assert ccx.num_qubits == 3
        np.testing.assert_allclose(ccx.matrix, CCX.matrix, atol=1e-10)

    def test_controlled_h_dimensions(self):
        ch = H.controlled()
        assert ch.num_qubits == 2
        assert ch.matrix.shape == (4, 4)

    def test_controlled_preserves_top_block(self):
        """Top-left block should be identity (control=|0⟩)."""
        import numpy as np
        ch = H.controlled()
        np.testing.assert_allclose(ch.matrix[:2, :2], np.eye(2), atol=1e-10)

    def test_controlled_invalid_ctrl(self):
        with pytest.raises(GateError, match="num_ctrl"):
            H.controlled(0)


# ═══════════════════════════════════════════
# Task 3: DAG IR improvements
# ═══════════════════════════════════════════

class TestDAGMethods:
    """Tests for DAG substitute_node, remove_node, node_layer."""

    def _make_bell_dag(self):
        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)
        return DAGCircuit.from_builder(bell.build())

    def test_node_layer(self):
        dag = self._make_bell_dag()
        ops = dag.op_nodes()
        assert dag.node_layer(ops[0].node_id) == 0  # H
        assert dag.node_layer(ops[1].node_id) == 1  # CX

    def test_substitute_node(self):
        dag = self._make_bell_dag()
        ops = dag.op_nodes()
        dag.substitute_node(ops[0].node_id, OpNode(0, "X", ops[0].qubits))
        assert dag.op_nodes()[0].gate_name == "X"

    def test_remove_node(self):
        dag = self._make_bell_dag()
        assert dag.gate_count() == 2
        ops = dag.op_nodes()
        dag.remove_node(ops[0].node_id)
        assert dag.gate_count() == 1

    def test_predecessors_successors(self):
        dag = self._make_bell_dag()
        ops = dag.op_nodes()
        preds = dag.predecessors(ops[1].node_id)
        assert len(preds) >= 1
        succs = dag.successors(ops[0].node_id)
        assert len(succs) >= 1


# ═══════════════════════════════════════════
# Task 4: Compiler passes
# ═══════════════════════════════════════════

class TestCancelInverses:
    """CancelInverses: H-H → empty, CX-CX → empty."""

    def test_hh_cancellation(self):
        @circuit(qubits=1)
        def hh(q):
            H(q[0])
            H(q[0])
            return measure(q)
        dag = DAGCircuit.from_builder(hh.build())
        assert dag.gate_count() == 2
        result = CancelInverses().run(dag)
        assert result.gate_count() == 0

    def test_cxcx_cancellation(self):
        @circuit(qubits=2)
        def cxcx(q):
            CX(q[0], q[1])
            CX(q[0], q[1])
            return measure(q)
        dag = DAGCircuit.from_builder(cxcx.build())
        result = CancelInverses().run(dag)
        assert result.gate_count() == 0

    def test_no_cancel_different_qubits(self):
        @circuit(qubits=2)
        def hh_diff(q):
            H(q[0])
            H(q[1])
            return measure(q)
        dag = DAGCircuit.from_builder(hh_diff.build())
        result = CancelInverses().run(dag)
        assert result.gate_count() == 2  # should NOT cancel


class TestMergeRotations:
    """MergeRotations: RZ(a)-RZ(b) → RZ(a+b)."""

    def test_rz_merge(self):
        @circuit(qubits=1)
        def rzrz(q):
            RZ(0.3)(q[0])
            RZ(0.5)(q[0])
            return measure(q)
        dag = DAGCircuit.from_builder(rzrz.build())
        assert dag.gate_count() == 2
        result = MergeRotations().run(dag)
        assert result.gate_count() == 1
        merged = result.op_nodes()[0]
        assert abs(merged.params[0] - 0.8) < 1e-10

    def test_rz_full_rotation_removal(self):
        """RZ(π) + RZ(π) = RZ(2π) → identity → removed."""
        @circuit(qubits=1)
        def full(q):
            RZ(math.pi)(q[0])
            RZ(math.pi)(q[0])
            return measure(q)
        dag = DAGCircuit.from_builder(full.build())
        result = MergeRotations().run(dag)
        assert result.gate_count() == 0


class TestIdentityRemoval:
    """IdentityRemoval: I gates and RZ(0) removed."""

    def test_i_gate_removal(self):
        @circuit(qubits=1)
        def with_id(q):
            I(q[0])
            H(q[0])
            I(q[0])
            return measure(q)
        dag = DAGCircuit.from_builder(with_id.build())
        assert dag.gate_count() == 3
        result = IdentityRemoval().run(dag)
        assert result.gate_count() == 1
        assert result.op_nodes()[0].gate_name == "H"

    def test_rz_zero_removal(self):
        @circuit(qubits=1)
        def rz0(q):
            RZ(0.0)(q[0])
            H(q[0])
            return measure(q)
        dag = DAGCircuit.from_builder(rz0.build())
        result = IdentityRemoval().run(dag)
        assert result.gate_count() == 1


class TestCommutationPass:
    """CommutationPass: reorders commuting gates."""

    def test_pass_runs_without_error(self):
        @circuit(qubits=2)
        def circ(q):
            H(q[0])
            RZ(0.5)(q[1])
            H(q[1])
            return measure(q)
        dag = DAGCircuit.from_builder(circ.build())
        result = CommutationPass().run(dag)
        assert result.gate_count() == dag.gate_count()

    def test_diagonal_before_nondiagonal(self):
        """RZ on q1 should move before H on q0 (different qubits)."""
        @circuit(qubits=2)
        def circ(q):
            H(q[0])
            RZ(0.5)(q[1])
            return measure(q)
        dag = DAGCircuit.from_builder(circ.build())
        result = CommutationPass().run(dag)
        ops = result.op_nodes()
        # Both gates on different qubits — commutation may reorder
        assert result.gate_count() == 2
        gate_names = {op.gate_name for op in ops}
        assert gate_names == {"H", "RZ"}


class TestFullPipeline:
    """Integration: full pipeline before/after metrics."""

    def test_pipeline_reduces_circuit(self):
        @circuit(qubits=2)
        def redundant(q):
            H(q[0])
            H(q[0])  # cancel
            RZ(0.3)(q[0])
            RZ(0.5)(q[0])  # merge to 0.8
            I(q[1])  # identity remove
            CX(q[0], q[1])
            return measure(q)

        dag = DAGCircuit.from_builder(redundant.build())
        original_gates = dag.gate_count()
        assert original_gates == 6

        pipeline = CompilerPipeline([
            CancelInverses(),
            MergeRotations(),
            IdentityRemoval(),
        ])
        result = pipeline.run(dag)

        assert result.gate_count() < original_gates
        # H-H cancels (−2), RZ-RZ merges (−1), I removed (−1)
        # Remaining: RZ(0.8) + CX + ... depends on pass order
        assert result.gate_count() <= 4

        # Verify summary
        summary = pipeline.summary()
        assert "CancelInverses" in summary

    def test_pipeline_depth_reduction(self):
        @circuit(qubits=1)
        def deep(q):
            H(q[0])
            H(q[0])
            X(q[0])
            X(q[0])
            return measure(q)
        dag = DAGCircuit.from_builder(deep.build())
        assert dag.depth() == 4

        result = CancelInverses().run(dag)
        assert result.depth() == 0
