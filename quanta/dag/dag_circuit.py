"""
quanta.dag.dag_circuit — DAG-based circuit representation.


DAG sayesinde:

Example:
    >>> builder = bell_state.build()
    >>> dag = DAGCircuit.from_builder(builder)
    >>> dag.depth()
    2
    >>> dag.gate_count()
    2
"""

from __future__ import annotations

from collections import defaultdict

from quanta.core.circuit import CircuitBuilder
from quanta.core.types import MeasureSpec
from quanta.dag.node import DAGNode, InputNode, OpNode, OutputNode

# ── Public API ──
__all__ = ["DAGCircuit"]

class DAGCircuit:
    """DAG representation of a quantum circuit.


    Attributes:
        measurement: Measurement specification.
    """

    __slots__ = (
        "num_qubits",
        "measurement",
        "_nodes",
        "_edges",
        "_reverse_edges",
        "_node_counter",
        "_input_nodes",
        "_output_nodes",
    )

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits
        self.measurement: MeasureSpec | None = None

        self._nodes: dict[int, DAGNode] = {}
        self._edges: dict[int, list[int]] = defaultdict(list)  # node_id → successors
        self._reverse_edges: dict[int, list[int]] = defaultdict(list)  # node_id → predecessors
        self._node_counter = 0

        self._input_nodes: dict[int, InputNode] = {}
        self._output_nodes: dict[int, OutputNode] = {}

        for q in range(num_qubits):
            in_node = self._add_node(InputNode(self._node_counter, qubit=q))
            self._input_nodes[q] = in_node

    def _add_node(self, node: DAGNode) -> DAGNode:
        node_id = self._node_counter
        if isinstance(node, InputNode):
            node = InputNode(node_id=node_id, qubit=node.qubit)
        elif isinstance(node, OpNode):
            node = OpNode(node_id=node_id, gate_name=node.gate_name,
                         qubits=node.qubits, params=node.params)
        elif isinstance(node, OutputNode):
            node = OutputNode(node_id=node_id, qubit=node.qubit)

        self._nodes[node_id] = node
        self._node_counter += 1
        return node

    def _add_edge(self, from_id: int, to_id: int) -> None:
        """Adds a directed edge between two nodes."""
        self._edges[from_id].append(to_id)
        self._reverse_edges[to_id].append(from_id)

    @classmethod
    def from_builder(cls, builder: CircuitBuilder) -> DAGCircuit:
        """Builds DAGCircuit from a CircuitBuilder.

        Args:

        Returns:
        """
        dag = cls(builder.num_qubits)
        dag.measurement = builder.measurement

        last_on_qubit: dict[int, int] = {
            q: dag._input_nodes[q].node_id
            for q in range(builder.num_qubits)
        }

        for instr in builder.instructions:
            op = dag._add_node(OpNode(
                node_id=0, gate_name=instr.gate_name,
                qubits=instr.qubits, params=instr.params,
            ))

            for q in instr.qubits:
                dag._add_edge(last_on_qubit[q], op.node_id)
                last_on_qubit[q] = op.node_id

        for q in range(builder.num_qubits):
            out = dag._add_node(OutputNode(node_id=0, qubit=q))
            dag._output_nodes[q] = out
            dag._add_edge(last_on_qubit[q], out.node_id)

        return dag

    def op_nodes(self) -> list[OpNode]:
        ordered = self.topological_sort()
        return [n for n in ordered if isinstance(n, OpNode)]

    def gate_count(self) -> int:
        """Total gate count in the circuit."""
        return sum(1 for n in self._nodes.values() if isinstance(n, OpNode))

    def depth(self) -> int:
        """Circuit depth (critical path length).

        """
        if not self._nodes:
            return 0

        longest: dict[int, int] = {}

        for node in self.topological_sort():
            if isinstance(node, InputNode):
                longest[node.node_id] = 0
            else:
                preds = self._reverse_edges.get(node.node_id, [])
                max_pred = max((longest.get(p, 0) for p in preds), default=0)
                longest[node.node_id] = max_pred + (1 if isinstance(node, OpNode) else 0)

        return max(longest.values(), default=0)

    def topological_sort(self) -> list[DAGNode]:
        """Topological sort of the DAG using Kahn's algorithm.

        Uses deque for O(1) popleft instead of list.pop(0) O(n).
        """
        from collections import deque as _deque

        in_degree: dict[int, int] = {nid: 0 for nid in self._nodes}

        for nid in self._nodes:
            for succ in self._edges.get(nid, []):
                in_degree[succ] += 1

        queue = _deque(nid for nid, deg in in_degree.items() if deg == 0)
        result: list[DAGNode] = []

        while queue:
            nid = queue.popleft()
            result.append(self._nodes[nid])

            for succ in self._edges.get(nid, []):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        return result

    def layers(self) -> list[list[OpNode]]:
        """Splits circuit into parallel layers (moments).


        Returns:
            List of layers. Each layer is a list of OpNode.
        """
        if not self._nodes:
            return []

        depth_map: dict[int, int] = {}
        for node in self.topological_sort():
            if isinstance(node, InputNode):
                depth_map[node.node_id] = -1
            elif isinstance(node, OpNode):
                preds = self._reverse_edges.get(node.node_id, [])
                depth_map[node.node_id] = max(
                    (depth_map.get(p, -1) for p in preds), default=-1
                ) + 1
            else:
                preds = self._reverse_edges.get(node.node_id, [])
                depth_map[node.node_id] = max(
                    (depth_map.get(p, -1) for p in preds), default=-1
                )

        # Group into layers
        layer_map: dict[int, list[OpNode]] = defaultdict(list)
        for node in self._nodes.values():
            if isinstance(node, OpNode):
                layer_map[depth_map[node.node_id]].append(node)

        return [layer_map[d] for d in sorted(layer_map)]

    def node_layer(self, node_id: int) -> int:
        """Returns the layer index (time step) for a given node.

        Args:
            node_id: The DAG node identifier.

        Returns:
            Zero-based layer index. Returns -1 for input nodes.
        """
        depth_map: dict[int, int] = {}
        for node in self.topological_sort():
            if isinstance(node, InputNode):
                depth_map[node.node_id] = -1
            elif isinstance(node, OpNode):
                preds = self._reverse_edges.get(node.node_id, [])
                depth_map[node.node_id] = max(
                    (depth_map.get(p, -1) for p in preds), default=-1
                ) + 1
            else:
                preds = self._reverse_edges.get(node.node_id, [])
                depth_map[node.node_id] = max(
                    (depth_map.get(p, -1) for p in preds), default=-1
                )
        return depth_map.get(node_id, -1)

    def substitute_node(self, node_id: int, new_op: OpNode) -> None:
        """Replaces an OpNode in-place, preserving edges.

        Args:
            node_id: ID of the node to replace.
            new_op: Replacement OpNode (node_id is updated automatically).

        Raises:
            KeyError: If node_id does not exist.
            TypeError: If the target node is not an OpNode.
        """
        old = self._nodes[node_id]
        if not isinstance(old, OpNode):
            raise TypeError(f"Cannot substitute non-OpNode: {old!r}")
        self._nodes[node_id] = OpNode(
            node_id=node_id,
            gate_name=new_op.gate_name,
            qubits=new_op.qubits,
            params=new_op.params,
        )

    def remove_node(self, node_id: int) -> None:
        """Removes an OpNode and reconnects its predecessors to successors.

        Args:
            node_id: ID of the OpNode to remove.

        Raises:
            KeyError: If node_id does not exist.
            TypeError: If the target node is not an OpNode.
        """
        node = self._nodes[node_id]
        if not isinstance(node, OpNode):
            raise TypeError(f"Cannot remove non-OpNode: {node!r}")

        preds = self._reverse_edges.get(node_id, [])
        succs = self._edges.get(node_id, [])

        # Reconnect: each predecessor → each successor
        for p in preds:
            self._edges[p] = [s for s in self._edges[p] if s != node_id]
            self._edges[p].extend(succs)
        for s in succs:
            self._reverse_edges[s] = [
                p for p in self._reverse_edges[s] if p != node_id
            ]
            self._reverse_edges[s].extend(preds)

        # Remove the node itself
        del self._nodes[node_id]
        self._edges.pop(node_id, None)
        self._reverse_edges.pop(node_id, None)

    def predecessors(self, node_id: int) -> list[DAGNode]:
        """Returns predecessor nodes of the given node."""
        return [self._nodes[p] for p in self._reverse_edges.get(node_id, [])]

    def successors(self, node_id: int) -> list[DAGNode]:
        """Returns successor nodes of the given node."""
        return [self._nodes[s] for s in self._edges.get(node_id, [])]

    def __repr__(self) -> str:
        return (
            f"DAGCircuit(qubits={self.num_qubits}, "
            f"gates={self.gate_count()}, depth={self.depth()})"
        )
