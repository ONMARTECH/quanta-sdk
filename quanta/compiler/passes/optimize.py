"""
quanta.compiler.passes.optimize — Gate optimization passes.


Example:
    >>> from quanta.compiler.passes.optimize import CancelInverses
    >>> dag = CancelInverses().run(dag)
"""

from __future__ import annotations

import math

from quanta.core.types import Instruction
from quanta.dag.dag_circuit import DAGCircuit
from quanta.dag.node import OpNode

# ── Public API ──
__all__ = ["CancelInverses", "MergeRotations"]

_SELF_INVERSE_GATES = frozenset({"H", "X", "Y", "Z", "CX", "CZ", "SWAP"})

_ROTATION_GATES = frozenset({"RX", "RY", "RZ"})

class CancelInverses:
    """Cancels sequential inverse gates.


    Example:
    """

    name = "CancelInverses"

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Cancels inverse gates and returns a new DAG.

        Args:

        Returns:
        """
        ops = dag.op_nodes()

        if len(ops) < 2:
            return dag

        to_remove: set[int] = set()

        for i in range(len(ops) - 1):
            if i in to_remove:
                continue

            current = ops[i]
            next_op = ops[i + 1]

            if self._can_cancel(current, next_op, ops, i):
                to_remove.add(i)
                to_remove.add(i + 1)

        # Build new DAG from remaining instructions
        if not to_remove:
            return dag

        return self._rebuild_dag(dag, ops, to_remove)

    def _can_cancel(
        self, a: OpNode, b: OpNode, ops: list[OpNode], idx: int
    ) -> bool:
        """Checks if two sequential gates can be cancelled."""
        if a.gate_name != b.gate_name:
            return False
        if a.qubits != b.qubits:
            return False
        if a.gate_name not in _SELF_INVERSE_GATES:
            return False
        return self._are_adjacent_on_qubits(a, b, ops, idx)

    def _are_adjacent_on_qubits(
        self, a: OpNode, b: OpNode, ops: list[OpNode], idx: int
    ) -> bool:
        """Checks if gates a and b are adjacent on their shared qubits.

        Returns True if no other gate touches shared qubits between a and b.
        """
        shared_qubits = set(a.qubits)
        for k in range(idx + 1, len(ops)):
            if ops[k].node_id == b.node_id:
                return True
            if set(ops[k].qubits) & shared_qubits:
                return False
        return False  # b not found after a

    def _rebuild_dag(
        self,
        original: DAGCircuit,
        ops: list[OpNode],
        to_remove: set[int],
    ) -> DAGCircuit:
        """Builds a new DAG without removed nodes."""
        from quanta.core.circuit import CircuitBuilder

        builder = CircuitBuilder(original.num_qubits)
        builder.measurement = original.measurement

        for i, op in enumerate(ops):
            if i not in to_remove:
                builder.record(Instruction(
                    gate_name=op.gate_name,
                    qubits=op.qubits,
                    params=op.params,
                ))

        return DAGCircuit.from_builder(builder)

class MergeRotations:
    """Merges sequential same-axis rotations.


    Example:
    """

    name = "MergeRotations"

    EPSILON = 1e-10

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Merges rotations.

        Args:

        Returns:
        """
        ops = dag.op_nodes()
        if len(ops) < 2:
            return dag

        merged_ops: list[OpNode | None] = list(ops)
        changed = False

        i = 0
        while i < len(merged_ops) - 1:
            if merged_ops[i] is None:
                i += 1
                continue

            current = merged_ops[i]
            j = i + 1
            while j < len(merged_ops) and merged_ops[j] is None:
                j += 1

            if j >= len(merged_ops):
                break

            next_op = merged_ops[j]

            if (
                current.gate_name == next_op.gate_name
                and current.gate_name in _ROTATION_GATES
                and current.qubits == next_op.qubits
                and current.params
                and next_op.params
            ):
                new_angle = current.params[0] + next_op.params[0]
                changed = True

                if abs(new_angle % (2 * math.pi)) < self.EPSILON:
                    merged_ops[i] = None
                    merged_ops[j] = None
                else:
                    merged_ops[i] = OpNode(
                        node_id=current.node_id,
                        gate_name=current.gate_name,
                        qubits=current.qubits,
                        params=(new_angle,),
                    )
                    merged_ops[j] = None
            else:
                i = j
                continue
            i = j

        if not changed:
            return dag

        # Build new DAG
        return self._rebuild_from_ops(dag, merged_ops)

    def _rebuild_from_ops(
        self, original: DAGCircuit, ops: list[OpNode | None]
    ) -> DAGCircuit:
        """Rebuilds a new DAG from the merged op list."""
        from quanta.core.circuit import CircuitBuilder

        builder = CircuitBuilder(original.num_qubits)
        builder.measurement = original.measurement

        for op in ops:
            if op is not None:
                builder.record(Instruction(
                    gate_name=op.gate_name,
                    qubits=op.qubits,
                    params=op.params,
                ))

        return DAGCircuit.from_builder(builder)
