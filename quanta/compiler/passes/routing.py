"""quanta.compiler.passes.routing -- Topology-aware qubit routing & IBM Heavy-Hex graphs.

Maps logical qubits to physical qubits by inserting SWAP gates where direct hardware
connectivity does not exist, decomposing 3Q+ gates cleanly into 1Q+2Q operations
prior to routing, and preserving exact logical qubit measurement order.

Supports topologies:
  - linear: 0-1-2-3-...-n
  - ring: 0-1-2-...-n-0
  - grid: r x c rectangular grid
  - heavy_hex: genuine IBM Heavy-Hex degree-2/3 coupling lattice
  - custom: user-defined adjacency set
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from quanta.compiler.pipeline import CompilerPass
from quanta.core.types import Instruction, MeasureSpec
from quanta.dag.dag_circuit import DAGCircuit

__all__ = ["RouteToTopology", "Topology"]


# ── Topology Factories ──


def _linear_edges(n: int) -> set[tuple[int, int]]:
    """Linear chain: 0-1-2-...-n."""
    return {(i, i + 1) for i in range(max(0, n - 1))}


def _ring_edges(n: int) -> set[tuple[int, int]]:
    """Ring: 0-1-2-...-n-0."""
    edges = _linear_edges(n)
    if n > 2:
        edges.add((0, n - 1))
    return edges


def _grid_edges(rows: int, cols: int) -> set[tuple[int, int]]:
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


def _bfs_renumber(
    adj: dict[object, list[object]],
    start: object,
) -> tuple[int, set[tuple[int, int]]]:
    """Renumbers an arbitrary connected graph in BFS order starting from ``start`` -> 0.

    Guarantees that every prefix subgraph {0, 1, ..., k - 1} is connected!
    """
    order: list[object] = []
    visited: set[object] = {start}
    queue: deque[object] = deque([start])
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                queue.append(v)

    # Append any remaining components deterministically if present
    for node in adj:
        if node not in visited:
            visited.add(node)
            queue.append(node)
            while queue:
                u = queue.popleft()
                order.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)

    idx_map = {node: idx for idx, node in enumerate(order)}
    edges: set[tuple[int, int]] = set()
    for u, nbrs in adj.items():
        iu = idx_map[u]
        for v in nbrs:
            iv = idx_map[v]
            if iu < iv:
                edges.add((iu, iv))
    return len(order), edges


def _build_heavy_hex_chains(
    chain_lengths: list[int],
    bridge_columns_per_gap: list[list[int]],
    chain_offsets: list[int] | None = None,
) -> tuple[int, set[tuple[int, int]]]:
    """Constructs a Heavy-Hex graph from horizontal chain lengths and staggered vertical bridges.

    Every bridge vertex connects strictly 1 upper chain vertex and 1 lower chain vertex (degree 2),
    and staggered bridge columns ensure no chain vertex ever receives both an upward and downward
    bridge, bounding maximum vertex degree strictly at <= 3.
    """
    num_rows = len(chain_lengths)
    offsets = chain_offsets if chain_offsets is not None else [0] * num_rows
    adj: dict[object, list[object]] = {}

    def _add_node(node: object) -> None:
        if node not in adj:
            adj[node] = []

    def _add_edge(u: object, v: object) -> None:
        _add_node(u)
        _add_node(v)
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)

    for r, length in enumerate(chain_lengths):
        x0 = offsets[r]
        for k in range(length):
            _add_node(("c", r, x0 + k))
            if k > 0:
                _add_edge(("c", r, x0 + k - 1), ("c", r, x0 + k))

    for r, cols in enumerate(bridge_columns_per_gap):
        for x in cols:
            top = ("c", r, x)
            bot = ("c", r + 1, x)
            if top in adj and bot in adj:
                bridge = ("b", r, x)
                _add_edge(top, bridge)
                _add_edge(bridge, bot)

    start_node = ("c", 0, offsets[0])
    return _bfs_renumber(adj, start_node)


def _heavy_hex_edges(rows: int, cols: int) -> tuple[int, set[tuple[int, int]]]:
    """Builds a parametric IBM Heavy-Hex lattice for ``rows`` chains and ``cols`` unit cells."""
    if rows < 2 or cols < 1:
        raise ValueError(
            f"Heavy-hex topology requires rows >= 2 and cols >= 1, got rows={rows}, cols={cols}."
        )
    chain_len = 4 * cols + 1
    chain_lengths = [chain_len] * rows
    bridges: list[list[int]] = []
    for r in range(rows - 1):
        if r % 2 == 0:
            bridges.append([4 * c for c in range(cols + 1)])
        else:
            bridges.append([4 * c + 2 for c in range(cols)])
    return _build_heavy_hex_chains(chain_lengths, bridges)


# Exact 3Q/4Q decomposition rules into 1Q+2Q gates prior to topology routing
_MULTI_QUBIT_LOWERING: dict[str, list[tuple[str, tuple[int, ...], tuple[float, ...]]]] = {
    "CCX": [
        ("H", (2,), ()),
        ("CX", (1, 2), ()),
        ("RZ", (2,), (-math.pi / 4,)),
        ("CX", (0, 2), ()),
        ("RZ", (2,), (math.pi / 4,)),
        ("CX", (1, 2), ()),
        ("RZ", (2,), (-math.pi / 4,)),
        ("CX", (0, 2), ()),
        ("RZ", (1,), (math.pi / 4,)),
        ("RZ", (2,), (math.pi / 4,)),
        ("H", (2,), ()),
        ("CX", (0, 1), ()),
        ("RZ", (0,), (math.pi / 4,)),
        ("RZ", (1,), (-math.pi / 4,)),
        ("CX", (0, 1), ()),
    ],
    "CSWAP": [
        ("CX", (2, 1), ()),
        ("CCX", (0, 1, 2), ()),
        ("CX", (2, 1), ()),
    ],
    "RCCX": [
        ("CRX", (1, 2), (math.pi / 2,)),
        ("CX", (0, 1), ()),
        ("CRX", (1, 2), (-math.pi / 2,)),
        ("CX", (0, 1), ()),
        ("CRX", (0, 2), (math.pi / 2,)),
    ],
    "RC3X": [
        ("CRX", (2, 3), (math.pi / 2,)),
        ("CCX", (0, 1, 2), ()),
        ("CRX", (2, 3), (-math.pi / 2,)),
        ("CCX", (0, 1, 2), ()),
        ("CRX", (1, 3), (math.pi / 4,)),
        ("CX", (0, 1), ()),
        ("CRX", (1, 3), (-math.pi / 4,)),
        ("CX", (0, 1), ()),
        ("CRX", (0, 3), (math.pi / 4,)),
    ],
}


def _expand_to_2q_ops(
    gate_name: str,
    qubits: tuple[int, ...],
    params: tuple[float, ...],
) -> list[Instruction]:
    """Recursively expands supported 3Q+ gates into 1Q+2Q Instructions."""
    if len(qubits) <= 2:
        return [Instruction(gate_name=gate_name, qubits=qubits, params=params)]
    if gate_name not in _MULTI_QUBIT_LOWERING:
        raise ValueError(
            f"RouteToTopology cannot route {len(qubits)}-qubit gate '{gate_name}'; "
            f"decompose >3Q gates into 1Q/2Q operations before routing."
        )
    expanded: list[Instruction] = []
    for sub_name, offsets, sub_params in _MULTI_QUBIT_LOWERING[gate_name]:
        sub_qubits = tuple(qubits[i] for i in offsets)
        expanded.extend(_expand_to_2q_ops(sub_name, sub_qubits, sub_params))
    return expanded


@dataclass
class Topology:
    """Describes hardware qubit connectivity.

    Example:
        >>> t = Topology.line(5)
        >>> t = Topology.grid(4, 4)
        >>> t = Topology.heavy_hex(3, 2)
        >>> t = Topology.from_backend("ibm_fez")
    """

    name: str
    num_qubits: int
    edges: set[tuple[int, int]]

    @classmethod
    def line(cls, n: int) -> Topology:
        """Linear chain topology."""
        return cls(name=f"line({n})", num_qubits=n, edges=_linear_edges(n))

    @classmethod
    def ring(cls, n: int) -> Topology:
        """Ring topology."""
        return cls(name=f"ring({n})", num_qubits=n, edges=_ring_edges(n))

    @classmethod
    def grid(cls, rows: int, cols: int) -> Topology:
        """Rectangular grid topology."""
        return cls(
            name=f"grid({rows}x{cols})",
            num_qubits=rows * cols,
            edges=_grid_edges(rows, cols),
        )

    @classmethod
    def heavy_hex(cls, rows: int, cols: int) -> Topology:
        """Genuine IBM Heavy-Hex coupling topology with degree-2 bridges and degree-3 junctions."""
        n, edges = _heavy_hex_edges(rows, cols)
        return cls(
            name=f"heavy_hex({rows}x{cols})",
            num_qubits=n,
            edges=edges,
        )

    @classmethod
    def custom(cls, edges: list[tuple[int, int]]) -> Topology:
        """User-defined topology from edge list."""
        edge_set = {(min(a, b), max(a, b)) for a, b in edges if a != b}
        n = max(max(a, b) for a, b in edge_set) + 1 if edge_set else 0
        return cls(name="custom", num_qubits=n, edges=edge_set)

    @classmethod
    def from_backend(cls, backend_name: str) -> Topology:
        """Creates hardware topology matching real quantum processor coupling maps.

        Args:
            backend_name: "ibm_fez", "ibm_heron", "ibm_brisbane", "ibm_eagle",
                "ionq_aria", "ionq", "google_willow".
        """
        name_lower = backend_name.lower()
        if name_lower in ("ibm_fez", "ibm_heron"):
            # Authentic IBM Heron 156-qubit heavy-hex lattice:
            # 8 horizontal chains of 16 qubits (128) + 7 staggered layers of 4 degree-2 bridges (28) = 156q
            chain_lengths = [16] * 8
            bridges = [
                [3, 7, 11, 15] if r % 2 == 0 else [1, 5, 9, 13]
                for r in range(7)
            ]
            n, edges = _build_heavy_hex_chains(chain_lengths, bridges)
            return cls(name=name_lower, num_qubits=n, edges=edges)

        if name_lower in ("ibm_brisbane", "ibm_eagle"):
            # Authentic IBM Eagle r3 127-qubit heavy-hex lattice:
            # 7 horizontal chains [14, 15, 15, 15, 15, 15, 14] (103) + 6 layers of 4 bridges (24) = 127q
            chain_lengths = [14, 15, 15, 15, 15, 15, 14]
            offsets = [0, 0, 0, 0, 0, 0, 1]
            bridges = [
                [0, 4, 8, 12] if r % 2 == 0 else [2, 6, 10, 14]
                for r in range(6)
            ]
            n, edges = _build_heavy_hex_chains(chain_lengths, bridges, chain_offsets=offsets)
            return cls(name=name_lower, num_qubits=n, edges=edges)

        if name_lower in ("ionq_aria", "ionq"):
            n = 25
            edges = {(i, j) for i in range(n) for j in range(i + 1, n)}
            return cls(name="ionq_aria", num_qubits=n, edges=edges)

        if name_lower in ("google_willow", "google"):
            return cls.grid(6, 12)

        raise ValueError(
            f"Unknown backend '{backend_name}'. "
            f"Try: ibm_fez, ibm_heron, ibm_brisbane, ionq_aria, google_willow"
        )

    def is_connected(self, u: int, v: int) -> bool:
        """Returns True if physical qubits ``u`` and ``v`` share a direct hardware edge."""
        return (min(u, v), max(u, v)) in self.edges

    def neighbors(self, qubit: int) -> tuple[int, ...]:
        """Returns sorted adjacent physical qubits of ``qubit``."""
        nbrs = [b if a == qubit else a for a, b in self.edges if a == qubit or b == qubit]
        return tuple(sorted(nbrs))

    def shortest_path(self, src: int, dst: int) -> list[int]:
        """Returns a shortest path ``[src, ..., dst]`` along topology edges (empty if unreachable)."""
        if src == dst:
            return [src]
        adj: dict[int, list[int]] = {i: [] for i in range(self.num_qubits)}
        for a, b in self.edges:
            if a in adj and b in adj:
                adj[a].append(b)
                adj[b].append(a)
        visited = {src}
        queue: deque[list[int]] = deque([[src]])
        while queue:
            path = queue.popleft()
            for nb in sorted(adj.get(path[-1], ())):
                if nb == dst:
                    return [*path, dst]
                if nb not in visited:
                    visited.add(nb)
                    queue.append([*path, nb])
        return []

    def max_degree(self) -> int:
        """Returns the maximum vertex degree in the coupling graph."""
        if not self.edges:
            return 0
        deg: dict[int, int] = {}
        for a, b in self.edges:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        return max(deg.values())

    def __repr__(self) -> str:
        return f"Topology('{self.name}', qubits={self.num_qubits}, edges={len(self.edges)})"


class RouteToTopology(CompilerPass):
    """Routes a circuit to match hardware connectivity constraints.

    Inserts nearest-neighbor SWAP gates where two-qubit gates act on non-adjacent
    physical qubits, decomposes 3Q+ gates (CCX, CSWAP, RCCX, RC3X) prior to routing,
    and ensures final measurement bitstrings and statevectors match the exact original
    logical qubit order.

    Args:
        topology: "linear", "ring", "grid", "heavy_hex", or a Topology instance.
        num_qubits: Number of physical qubits.
        grid_rows: Rows for grid / heavy_hex topology.
        grid_cols: Columns for grid / heavy_hex topology.
        custom_edges: Custom adjacency set {(i, j), ...}.
        restore_layout: If True (default), restores the final logical-to-physical
            permutation via spanning-tree leaf-peeling SWAPs so both direct unitary/statevector
            simulation and measurements match logical order identically. If False, omits
            trailing SWAPs and remaps ``new_dag.measurement.qubits`` to the final physical
            locations ``tuple(l2p[q] for q in measured)``.
    """

    def __init__(
        self,
        topology: str | Topology = "linear",
        num_qubits: int = 0,
        grid_rows: int = 0,
        grid_cols: int = 0,
        custom_edges: set[tuple[int, int]] | None = None,
        restore_layout: bool = True,
    ) -> None:
        if isinstance(topology, Topology):
            self._topology = topology.name
            self._num_qubits = topology.num_qubits
            self._edges = topology.edges
        elif custom_edges is not None:
            self._topology = "custom"
            self._num_qubits = num_qubits
            self._edges = custom_edges
        elif topology == "linear":
            self._topology = topology
            self._num_qubits = num_qubits
            self._edges = _linear_edges(num_qubits)
        elif topology == "ring":
            self._topology = topology
            self._num_qubits = num_qubits
            self._edges = _ring_edges(num_qubits)
        elif topology == "grid":
            self._topology = topology
            self._num_qubits = grid_rows * grid_cols
            self._edges = _grid_edges(grid_rows, grid_cols)
        elif topology == "heavy_hex":
            hh_n, hh_edges = _heavy_hex_edges(grid_rows, grid_cols)
            self._topology = topology
            self._num_qubits = hh_n
            self._edges = hh_edges
        else:
            raise ValueError(f"Unknown topology: {topology}")

        self._restore_layout = restore_layout
        self.final_layout: tuple[int, ...] = ()

        # Build bidirectional canonical adjacency set
        self._adjacent: set[tuple[int, int]] = set()
        for a, b in self._edges:
            if a != b:
                self._adjacent.add((min(a, b), max(a, b)))

    def _are_adjacent(self, q0: int, q1: int) -> bool:
        """Checks if two physical qubits are directly connected."""
        return (min(q0, q1), max(q0, q1)) in self._adjacent

    def _shortest_path(
        self,
        src: int,
        dst: int,
        n: int,
        allowed_nodes: set[int] | None = None,
    ) -> list[int]:
        """BFS shortest path between two physical qubits on the coupling graph."""
        if src == dst:
            return [src]

        nodes = allowed_nodes if allowed_nodes is not None else set(range(n))
        if src not in nodes or dst not in nodes:
            return []

        adj: dict[int, list[int]] = {i: [] for i in nodes}
        for a, b in self._adjacent:
            if a in adj and b in adj:
                adj[a].append(b)
                adj[b].append(a)

        visited = {src}
        queue: deque[tuple[int, list[int]]] = deque([(src, [src])])
        while queue:
            node, path = queue.popleft()
            for neighbor in sorted(adj[node]):
                if neighbor == dst:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []

    def _active_physical_size(self, logical_n: int) -> int:
        """Determines minimal connected physical qubit count >= logical_n."""
        if self._num_qubits > 0 and logical_n > self._num_qubits:
            raise ValueError(
                f"Circuit requires {logical_n} qubits, exceeding topology capacity {self._num_qubits}."
            )
        # Check if induced subgraph on {0 .. logical_n - 1} is already connected
        prefix = set(range(logical_n))
        if logical_n <= 1 or all(
            bool(self._shortest_path(0, q, logical_n, allowed_nodes=prefix))
            for q in range(1, logical_n)
        ):
            return logical_n
        # Otherwise use full topology size so bridge qubits connect {0 .. logical_n - 1}
        full_n = max(logical_n, self._num_qubits)
        full_set = set(range(full_n))
        if all(
            bool(self._shortest_path(0, q, full_n, allowed_nodes=full_set))
            for q in range(1, logical_n)
        ):
            return full_n
        raise ValueError(
            f"Topology '{self._topology}' is disconnected on logical qubits 0..{logical_n - 1}."
        )

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Routes the circuit onto hardware edges, preserving logical qubit order."""
        n = dag.num_qubits
        if n < 2:
            self.final_layout = tuple(range(n))
            return dag

        # 1. Expand any 3Q+ operations (CCX, CSWAP, RCCX, RC3X) into 1Q+2Q ops
        flat_ops: list[Instruction] = []
        for op in dag.op_nodes():
            flat_ops.extend(_expand_to_2q_ops(op.gate_name, op.qubits, op.params))

        phys_n = self._active_physical_size(n)
        l2p = list(range(phys_n))  # logical i -> physical l2p[i]
        p2l = list(range(phys_n))  # physical i -> logical p2l[i]
        active_set = set(range(phys_n))

        from quanta.core.circuit import CircuitBuilder

        builder = CircuitBuilder(phys_n)

        def _apply_swap(pa: int, pb: int) -> None:
            builder.record(Instruction(gate_name="SWAP", qubits=(pa, pb), params=()))
            la, lb = p2l[pa], p2l[pb]
            l2p[la], l2p[lb] = pb, pa
            p2l[pa], p2l[pb] = lb, la

        for inst in flat_ops:
            if len(inst.qubits) == 1:
                builder.record(
                    Instruction(
                        gate_name=inst.gate_name,
                        qubits=(l2p[inst.qubits[0]],),
                        params=inst.params,
                    )
                )
                continue

            pq0 = l2p[inst.qubits[0]]
            pq1 = l2p[inst.qubits[1]]

            if not self._are_adjacent(pq0, pq1):
                path = self._shortest_path(pq0, pq1, phys_n, allowed_nodes=active_set)
                if not path:
                    raise ValueError(
                        f"No coupling path between physical qubits {pq0} and {pq1} "
                        f"on topology '{self._topology}'."
                    )
                for i in range(len(path) - 2):
                    _apply_swap(path[i], path[i + 1])

                pq0 = l2p[inst.qubits[0]]
                pq1 = l2p[inst.qubits[1]]

            builder.record(
                Instruction(
                    gate_name=inst.gate_name,
                    qubits=(pq0, pq1),
                    params=inst.params,
                )
            )

        # 2. Restore identity logical-to-physical layout via spanning-tree leaf peeling
        #    when restore_layout=True and any logical qubit moved.
        if self._restore_layout and any(l2p[q] != q for q in range(phys_n)):
            # Build BFS order of active_set rooted at 0 so peeling from end keeps tree connected
            bfs_order: list[int] = []
            seen = {0}
            q_bfs: deque[int] = deque([0])
            adj_map: dict[int, list[int]] = {i: [] for i in active_set}
            for a, b in self._adjacent:
                if a in adj_map and b in adj_map:
                    adj_map[a].append(b)
                    adj_map[b].append(a)
            while q_bfs:
                u = q_bfs.popleft()
                bfs_order.append(u)
                for v in sorted(adj_map[u]):
                    if v not in seen:
                        seen.add(v)
                        q_bfs.append(v)

            remaining = set(bfs_order)
            for target_phys in reversed(bfs_order):
                cur_phys = l2p[target_phys]
                if cur_phys != target_phys:
                    path = self._shortest_path(
                        cur_phys, target_phys, phys_n, allowed_nodes=remaining
                    )
                    for i in range(len(path) - 1):
                        _apply_swap(path[i], path[i + 1])
                remaining.remove(target_phys)

        self.final_layout = tuple(l2p[:n])

        # 3. Remap measurement qubits through final l2p mapping so bitstrings are ALWAYS
        #    returned in the exact original logical qubit order!
        if dag.measurement is not None:
            orig_measured = (
                dag.measurement.qubits
                if dag.measurement.qubits
                else tuple(range(n))
            )
            remapped = tuple(l2p[lq] for lq in orig_measured)
            if remapped == tuple(range(phys_n)) and not dag.measurement.qubits:
                builder.measurement = dag.measurement
            else:
                builder.measurement = MeasureSpec(qubits=remapped)

        return DAGCircuit.from_builder(builder)
