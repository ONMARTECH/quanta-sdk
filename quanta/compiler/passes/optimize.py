"""quanta.compiler.passes.optimize — Topological per-wire gate optimization passes.

Provides DAG-aware optimization passes operating along per-qubit dependency wires:
  - ``CancelInverses``: Cancels consecutive self-inverse gates and adjoint pairs (S/SDG,
    T/TDG, SX/SXdg) on their shared qubit wires regardless of interleaved independent gates.
  - ``MergeRotations``: Merges arbitrary-length chains of consecutive same-axis rotations
    along each wire, wrapping angles into (-pi, pi] and removing near-zero rotations.
  - ``IdentityRemoval``: Removes explicit I gates and 2*pi-periodic zero-angle rotations.
  - ``CommutationPass``: Reorders commuting operations to expose further cancellations.
"""

from __future__ import annotations

import math

from quanta.core.types import Instruction
from quanta.dag.dag_circuit import DAGCircuit
from quanta.dag.node import OpNode

__all__ = [
    "CancelInverses",
    "MergeRotations",
    "IdentityRemoval",
    "CommutationPass",
]

_SELF_INVERSE_GATES = frozenset(
    {"H", "X", "Y", "Z", "I", "CX", "CZ", "CY", "SWAP", "CCX", "CSWAP", "ECR"}
)

_INVERSE_PAIRS: dict[str, str] = {
    "S": "SDG",
    "SDG": "S",
    "T": "TDG",
    "TDG": "T",
    "SX": "SXdg",
    "SXdg": "SX",
}

_ROTATION_GATES = frozenset(
    {"RX", "RY", "RZ", "P", "RXX", "RZZ", "CRX", "CRY", "CRZ", "CP", "MS"}
)

_DIAGONAL_1Q = frozenset({"RZ", "P", "Z", "S", "T", "SDG", "TDG", "I"})


def _wrap_angle_pi(angle: float) -> float:
    """Wraps an angle into (-pi, pi], mapping -pi cleanly to +pi."""
    wrapped = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped <= -math.pi + 1e-15:
        return math.pi
    return wrapped


class CancelInverses:
    """Cancels inverse gate pairs along per-qubit topological wires.

    Cancels both self-inverse gates (H·H, X·X, CX·CX, SWAP·SWAP, ECR·ECR, ...) and
    mutually inverse gate pairs (S·SDG, T·TDG, SX·SXdg) whenever no intervening gate
    acts on their shared qubit wire(s), including cascading cancellations.
    """

    name = "CancelInverses"

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Cancels inverse gates per wire and returns an optimized DAGCircuit."""
        ops = dag.op_nodes()
        if len(ops) < 2:
            return dag

        active_ops: list[OpNode | None] = list(ops)
        wire_stacks: dict[int, list[int]] = {q: [] for q in range(dag.num_qubits)}
        removed_any = False

        for idx, op in enumerate(ops):
            if op.gate_name == "I":
                active_ops[idx] = None
                removed_any = True
                continue

            qubits = op.qubits
            if qubits and all(bool(wire_stacks[q]) for q in qubits):
                candidate_idx = wire_stacks[qubits[0]][-1]
                if all(wire_stacks[q][-1] == candidate_idx for q in qubits):
                    prev_op = active_ops[candidate_idx]
                    if prev_op is not None and self._is_inverse_pair(prev_op, op):
                        for q in qubits:
                            wire_stacks[q].pop()
                        active_ops[candidate_idx] = None
                        active_ops[idx] = None
                        removed_any = True
                        continue

            for q in qubits:
                wire_stacks[q].append(idx)

        if not removed_any:
            return dag

        from quanta.core.circuit import CircuitBuilder

        builder = CircuitBuilder(dag.num_qubits)
        builder.measurement = dag.measurement
        for op_item in active_ops:
            if op_item is not None:
                builder.record(
                    Instruction(
                        gate_name=op_item.gate_name,
                        qubits=op_item.qubits,
                        params=op_item.params,
                    )
                )
        return DAGCircuit.from_builder(builder)

    @staticmethod
    def _is_inverse_pair(a: OpNode, b: OpNode) -> bool:
        if a.qubits != b.qubits:
            return False
        if a.gate_name == b.gate_name and a.gate_name in _SELF_INVERSE_GATES:
            return True
        return _INVERSE_PAIRS.get(a.gate_name) == b.gate_name


class MergeRotations:
    """Merges arbitrary-length chains of consecutive same-axis rotations per qubit wire.

    Wraps merged angles into (-pi, pi] via ``(merged + pi) % (2 * pi) - pi`` and
    eliminates rotations whose wrapped magnitude is smaller than ``EPSILON``.
    """

    name = "MergeRotations"

    EPSILON = 1e-10

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Merges same-axis rotations along each wire and returns an optimized DAGCircuit."""
        ops = dag.op_nodes()
        if not ops:
            return dag

        active_ops: list[OpNode | None] = list(ops)
        raw_angles: dict[int, float] = {}
        wire_stacks: dict[int, list[int]] = {q: [] for q in range(dag.num_qubits)}
        changed = False

        for idx, op in enumerate(ops):
            qubits = op.qubits
            is_rot = op.gate_name in _ROTATION_GATES and len(op.params) == 1

            if is_rot and qubits and all(bool(wire_stacks[q]) for q in qubits):
                candidate_idx = wire_stacks[qubits[0]][-1]
                if all(wire_stacks[q][-1] == candidate_idx for q in qubits):
                    prev_op = active_ops[candidate_idx]
                    if (
                        prev_op is not None
                        and prev_op.gate_name == op.gate_name
                        and prev_op.qubits == qubits
                        and len(prev_op.params) == 1
                    ):
                        total_raw = raw_angles.get(candidate_idx, prev_op.params[0]) + op.params[0]
                        wrapped = _wrap_angle_pi(total_raw)
                        changed = True
                        active_ops[idx] = None

                        if abs(wrapped) < self.EPSILON:
                            for q in qubits:
                                wire_stacks[q].pop()
                            active_ops[candidate_idx] = None
                            raw_angles.pop(candidate_idx, None)
                        else:
                            raw_angles[candidate_idx] = total_raw
                            active_ops[candidate_idx] = OpNode(
                                node_id=prev_op.node_id,
                                gate_name=prev_op.gate_name,
                                qubits=qubits,
                                params=(wrapped,),
                            )
                        continue

            if is_rot:
                wrapped_single = _wrap_angle_pi(op.params[0])
                if abs(wrapped_single) < self.EPSILON:
                    active_ops[idx] = None
                    changed = True
                    continue
                raw_angles[idx] = op.params[0]
                if abs(wrapped_single - op.params[0]) > 1e-15:
                    active_ops[idx] = OpNode(
                        node_id=op.node_id,
                        gate_name=op.gate_name,
                        qubits=qubits,
                        params=(wrapped_single,),
                    )
                    changed = True

            for q in qubits:
                wire_stacks[q].append(idx)

        if not changed:
            return dag

        from quanta.core.circuit import CircuitBuilder

        builder = CircuitBuilder(dag.num_qubits)
        builder.measurement = dag.measurement
        for op_item in active_ops:
            if op_item is not None:
                builder.record(
                    Instruction(
                        gate_name=op_item.gate_name,
                        qubits=op_item.qubits,
                        params=op_item.params,
                    )
                )
        return DAGCircuit.from_builder(builder)


class IdentityRemoval:
    """Removes identity gates (I, RZ(0), RX(0), RY(0), ...) from the circuit."""

    name = "IdentityRemoval"

    EPSILON = 1e-10

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Removes identity-equivalent gates."""
        ops = dag.op_nodes()
        to_remove: set[int] = set()

        for i, op in enumerate(ops):
            if op.gate_name == "I" or (
                op.gate_name in _ROTATION_GATES
                and op.params
                and abs(_wrap_angle_pi(op.params[0])) < self.EPSILON
            ):
                to_remove.add(i)

        if not to_remove:
            return dag

        from quanta.core.circuit import CircuitBuilder

        builder = CircuitBuilder(dag.num_qubits)
        builder.measurement = dag.measurement
        for i, op in enumerate(ops):
            if i not in to_remove:
                builder.record(
                    Instruction(
                        gate_name=op.gate_name,
                        qubits=op.qubits,
                        params=op.params,
                    )
                )
        return DAGCircuit.from_builder(builder)


class CommutationPass:
    """Reorders commuting gates to expose more cancellation opportunities."""

    name = "CommutationPass"

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Reorders commuting op sequences."""
        ops = list(dag.op_nodes())
        if len(ops) < 2:
            return dag

        changed = True
        max_iters = len(ops)
        iteration = 0

        while changed and iteration < max_iters:
            changed = False
            iteration += 1
            for i in range(len(ops) - 1):
                a, b = ops[i], ops[i + 1]
                if self._should_swap(a, b):
                    ops[i], ops[i + 1] = ops[i + 1], ops[i]
                    changed = True

        from quanta.core.circuit import CircuitBuilder

        builder = CircuitBuilder(dag.num_qubits)
        builder.measurement = dag.measurement
        for op in ops:
            builder.record(
                Instruction(
                    gate_name=op.gate_name,
                    qubits=op.qubits,
                    params=op.params,
                )
            )
        return DAGCircuit.from_builder(builder)

    @staticmethod
    def _should_swap(a: OpNode, b: OpNode) -> bool:
        """Returns True if swapping a and b enables optimization."""
        if set(a.qubits) & set(b.qubits):
            return False
        if a.gate_name == b.gate_name:
            return False
        return b.gate_name in _DIAGONAL_1Q and a.gate_name not in _DIAGONAL_1Q
