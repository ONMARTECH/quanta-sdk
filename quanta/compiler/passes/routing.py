"""
quanta.compiler.passes.routing -- Qubit routing for hardware connectivity.

Maps logical qubits to physical qubits by inserting SWAP gates
where direct connectivity doesn't exist.

Supports topologies:
  - linear: 0-1-2-3-...-n
  - ring: 0-1-2-...-n-0
  - grid: r x c rectangular grid
  - custom: user-defined adjacency set

Example:
    >>> from quanta.compiler.passes.routing import RouteToTopology
    >>> routing = RouteToTopology(topology="linear", num_qubits=5)
    >>> routed_dag = routing.run(dag)
"""

from __future__ import annotations

from quanta.compiler.pipeline import CompilerPass
from quanta.dag.dag_circuit import DAGCircuit

__all__ = ["RouteToTopology"]


def linear_topology(n: int) -> set[tuple[int, int]]:
    """Linear chain: 0-1-2-...-n."""
    return {(i, i + 1) for i in range(n - 1)}


def ring_topology(n: int) -> set[tuple[int, int]]:
    """Ring: 0-1-2-...-n-0."""
    edges = linear_topology(n)
    edges.add((0, n - 1))
    return edges


def grid_topology(rows: int, cols: int) -> set[tuple[int, int]]:
    """Rectangular grid with nearest-neighbor connectivity."""
    edges: set[tuple[int, int]] = set()
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.add((q, q + 1))
            if r + 1 < rows:
                edges.add((q, q + cols))
    return edges


class RouteToTopology(CompilerPass):
    """Routes a circuit to match hardware connectivity constraints.

    Inserts SWAP gates where two-qubit gates operate on non-adjacent qubits.
    Uses greedy nearest-neighbor routing strategy.

    Args:
        topology: "linear", "ring", or "grid".
        num_qubits: Number of physical qubits.
        grid_rows: Rows for grid topology.
        grid_cols: Columns for grid topology.
        custom_edges: Custom adjacency set {(i, j), ...}.
    """

    def __init__(
        self,
        topology: str = "linear",
        num_qubits: int = 0,
        grid_rows: int = 0,
        grid_cols: int = 0,
        custom_edges: set[tuple[int, int]] | None = None,
    ) -> None:
        self._topology = topology
        self._num_qubits = num_qubits

        if custom_edges is not None:
            self._edges = custom_edges
        elif topology == "linear":
            self._edges = linear_topology(num_qubits)
        elif topology == "ring":
            self._edges = ring_topology(num_qubits)
        elif topology == "grid":
            self._edges = grid_topology(grid_rows, grid_cols)
        else:
            raise ValueError(f"Unknown topology: {topology}")

        # Build adjacency set (bidirectional)
        self._adjacent: set[tuple[int, int]] = set()
        for a, b in self._edges:
            self._adjacent.add((min(a, b), max(a, b)))

    def _are_adjacent(self, q0: int, q1: int) -> bool:
        """Checks if two physical qubits are directly connected."""
        return (min(q0, q1), max(q0, q1)) in self._adjacent

    def _shortest_path(self, src: int, dst: int, n: int) -> list[int]:
        """BFS shortest path between two qubits on the topology."""
        if src == dst:
            return [src]

        # Build adjacency list
        adj: dict[int, list[int]] = {i: [] for i in range(n)}
        for a, b in self._adjacent:
            adj[a].append(b)
            adj[b].append(a)

        # BFS
        visited = {src}
        queue = [(src, [src])]
        while queue:
            node, path = queue.pop(0)
            for neighbor in adj[node]:
                if neighbor == dst:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []  # No path (disconnected graph)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Routes the circuit by inserting SWAPs for non-adjacent gates.

        Uses greedy approach: for each non-adjacent two-qubit gate,
        insert SWAPs along the shortest path to bring qubits together.
        """
        n = dag.num_qubits
        if n < 2:
            return dag

        # Logical -> physical mapping (initially identity)
        l2p = list(range(n))  # logical qubit i -> physical qubit l2p[i]
        p2l = list(range(n))  # physical qubit i -> logical qubit p2l[i]

        from quanta.core.types import Instruction, MeasureSpec
        from quanta.core.circuit import CircuitBuilder

        builder = CircuitBuilder(n)

        for op in dag.op_nodes():
            if len(op.qubits) < 2:
                # Single-qubit gate: just map qubit
                builder.record(Instruction(
                    gate_name=op.gate_name,
                    qubits=(l2p[op.qubits[0]],),
                    params=op.params,
                ))
                continue

            # Two-qubit gate: check adjacency
            pq0 = l2p[op.qubits[0]]
            pq1 = l2p[op.qubits[1]]

            if not self._are_adjacent(pq0, pq1):
                # Insert SWAPs along shortest path
                path = self._shortest_path(pq0, pq1, n)
                for i in range(len(path) - 2):
                    a, b = path[i], path[i + 1]
                    builder.record(Instruction(
                        gate_name="SWAP",
                        qubits=(a, b),
                        params=(),
                    ))
                    # Update mappings
                    la, lb = p2l[a], p2l[b]
                    l2p[la], l2p[lb] = b, a
                    p2l[a], p2l[b] = lb, la

                pq0 = l2p[op.qubits[0]]
                pq1 = l2p[op.qubits[1]]

            builder.record(Instruction(
                gate_name=op.gate_name,
                qubits=(pq0, pq1),
                params=op.params,
            ))

        builder.measurement = dag.measurement
        return DAGCircuit.from_builder(builder)
