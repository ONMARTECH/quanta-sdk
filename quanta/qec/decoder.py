"""
quanta.qec.decoder -- Error correction decoders (MWPM and Union-Find).

Decoders take a syndrome measurement and determine which physical data qubits
experienced errors, returning physical Pauli corrections over GF(2).

Implemented decoders:
  - MWPMDecoder: Edmonds' Blossom Minimum Weight Perfect Matching with GF(2)
    symmetric-difference path accumulation.
  - UnionFindDecoder: Delfosse-Nickerson (Quantum 5, 595, 2021) weighted
    half-edge cluster growth + spanning-forest peeling decoder.

Example:
    >>> from quanta.qec.decoder import MWPMDecoder, UnionFindDecoder
    >>> decoder = UnionFindDecoder()
    >>> correction = decoder.decode(syndrome, code_distance=3)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

__all__ = ["DecoderResult", "MWPMDecoder", "UnionFindDecoder"]


@dataclass(frozen=True)
class DecoderResult:
    """Output of a syndrome decoder.

    Attributes:
        correction: Physical data qubit indices to apply Pauli correction.
        weight: Total correction weight (number of corrected physical qubits).
        algorithm: Decoder name used.
        success: Whether the correction clears the input syndrome over GF(2).
        pauli: Mapping from physical qubit index to Pauli operator ('X', 'Z', 'Y').
    """

    correction: tuple[int, ...] = ()
    weight: int = 0
    algorithm: str = "custom"
    success: bool = True
    pauli: dict[int, str] = field(default_factory=dict)


def _build_rotated_surface_stabilizers(
    distance: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Builds rotated d×d surface code (x_stabilizers, z_stabilizers)."""
    d = distance
    x_stabs: list[list[int]] = []
    z_stabs: list[list[int]] = []

    def idx(r: int, c: int) -> int:
        return r * d + c

    for r in range(d - 1):
        for c in range(d - 1):
            qubits = [idx(r, c), idx(r, c + 1), idx(r + 1, c), idx(r + 1, c + 1)]
            if (r + c) % 2 == 0:
                x_stabs.append(qubits)
            else:
                z_stabs.append(qubits)

    for c in range(d - 1):
        if c % 2 == 0:
            z_stabs.append([idx(0, c), idx(0, c + 1)])
        else:
            z_stabs.append([idx(d - 1, c), idx(d - 1, c + 1)])

    for r in range(d - 1):
        if r % 2 == 1:
            x_stabs.append([idx(r, 0), idx(r + 1, 0)])
        else:
            x_stabs.append([idx(r, d - 1), idx(r + 1, d - 1)])

    return x_stabs, z_stabs


def _build_planar_grid_stabilizers(distance: int) -> list[list[int]]:
    """Builds a d×d vertex-check planar Tanner lattice with physical edge qubits.

    Used when callers pass an unrotated d^2-length syndrome array without
    explicit stabilizers. Boundary data qubits are indexed first (so corner
    check 0 has boundary qubit 0 of weight 1), followed by internal horizontal
    and vertical data-qubit edges.
    """
    d = distance
    stabs: list[list[int]] = [[] for _ in range(d * d)]
    next_q = 0

    # Boundary data-qubit edges (1 check incident -> connects check to boundary B)
    for r in range(d):
        for c in range(d):
            if r in (0, d - 1) or c in (0, d - 1):
                stabs[r * d + c].append(next_q)
                next_q += 1

    # Horizontal internal data-qubit edges
    for r in range(d):
        for c in range(d - 1):
            u = r * d + c
            v = r * d + (c + 1)
            stabs[u].append(next_q)
            stabs[v].append(next_q)
            next_q += 1

    # Vertical internal data-qubit edges
    for r in range(d - 1):
        for c in range(d):
            u = r * d + c
            v = (r + 1) * d + c
            stabs[u].append(next_q)
            stabs[v].append(next_q)
            next_q += 1

    return stabs


def _resolve_stabilizers(
    syndrome: np.ndarray,
    code_distance: int,
    stabilizers: list[list[int]] | None,
    error_type: str,
) -> list[list[int]]:
    """Resolves stabilizer check supports from explicit argument or code geometry."""
    if stabilizers is not None:
        return stabilizers

    m = len(syndrome)
    if code_distance >= 3 and code_distance % 2 == 1:
        half = (code_distance * code_distance - 1) // 2
        if m == half:
            x_stabs, z_stabs = _build_rotated_surface_stabilizers(code_distance)
            return z_stabs if error_type.upper() == "X" else x_stabs
        if m == code_distance * code_distance:
            return _build_planar_grid_stabilizers(code_distance)

    # Generic fallback: 1D chain with boundary endpoints
    stabs = [[] for _ in range(m)]
    if m > 0:
        stabs[0].append(0)
        for i in range(m - 1):
            stabs[i].append(i + 1)
            stabs[i + 1].append(i + 1)
        stabs[m - 1].append(m)
    return stabs


def _verify_syndrome_clearance(
    syndrome: np.ndarray,
    stabilizers: list[list[int]],
    correction: set[int] | tuple[int, ...],
) -> bool:
    """Checks (H @ corr_vec) % 2 == syndrome over GF(2)."""
    corr_set = set(correction)
    for s_idx, stab in enumerate(stabilizers):
        parity = sum(1 for q in stab if q in corr_set) % 2
        if parity != int(bool(syndrome[s_idx])):
            return False
    return True


def _merge_css_results(
    res_z_err: DecoderResult,
    res_x_err: DecoderResult,
    algorithm: str,
) -> DecoderResult:
    """Combines Z-error (from X-syndrome) and X-error (from Z-syndrome) results."""
    pauli: dict[int, str] = {}
    for q in res_x_err.correction:
        pauli[q] = "X"
    for q in res_z_err.correction:
        pauli[q] = "Y" if q in pauli else "Z"
    combined = tuple(sorted(pauli.keys()))
    return DecoderResult(
        correction=combined,
        weight=len(combined),
        algorithm=algorithm,
        success=res_z_err.success and res_x_err.success,
        pauli=pauli,
    )


class DecoderBase(ABC):
    """Abstract base for syndrome decoders."""

    name: str = "base"

    @abstractmethod
    def decode(
        self,
        syndrome: np.ndarray,
        code_distance: int,
        stabilizers: list[list[int]] | None = None,
        error_type: str = "X",
    ) -> DecoderResult:
        """Decodes a syndrome into physical data-qubit corrections."""


class MWPMDecoder(DecoderBase):
    """Minimum Weight Perfect Matching decoder using Edmonds' Blossom algorithm.

    Pairs syndrome defects via minimum-weight paths on the stabilizer graph
    (with virtual boundary replication) and accumulates physical data-qubit
    corrections over GF(2) (XOR / symmetric difference).
    """

    name = "MWPM"

    def decode(
        self,
        syndrome: np.ndarray,
        code_distance: int,
        stabilizers: list[list[int]] | None = None,
        error_type: str = "X",
    ) -> DecoderResult:
        syn = np.asarray(syndrome, dtype=bool).ravel()

        # Combined [syn_x, syn_z] CSS syndrome on rotated surface code
        if (
            stabilizers is None
            and code_distance >= 3
            and code_distance % 2 == 1
            and len(syn) == code_distance * code_distance - 1
        ):
            half = len(syn) // 2
            x_stabs, z_stabs = _build_rotated_surface_stabilizers(code_distance)
            res_z = self.decode(syn[:half], code_distance, stabilizers=x_stabs, error_type="Z")
            res_x = self.decode(syn[half:], code_distance, stabilizers=z_stabs, error_type="X")
            return _merge_css_results(res_z, res_x, self.name)

        stabs = _resolve_stabilizers(syn, code_distance, stabilizers, error_type)
        defect_indices = [int(i) for i in np.where(syn)[0]]
        n = len(defect_indices)

        if n == 0:
            return DecoderResult(
                correction=(),
                weight=0,
                algorithm=self.name,
                success=True,
                pauli={},
            )

        n_qubits = max((max(s) for s in stabs if s), default=-1) + 1
        G_stab = nx.Graph()
        for s_idx in range(len(stabs)):
            G_stab.add_node(s_idx)
        G_stab.add_node("B")

        for q in range(n_qubits):
            inc = [idx for idx, s in enumerate(stabs) if q in s]
            if len(inc) == 2 and not G_stab.has_edge(inc[0], inc[1]):
                G_stab.add_edge(inc[0], inc[1], qubit=q, weight=1)
            elif len(inc) == 1 and not G_stab.has_edge(inc[0], "B"):
                G_stab.add_edge(inc[0], "B", qubit=q, weight=1)

        # Precompute shortest paths from each defect to other defects and boundary "B"
        lengths_from: dict[int, dict[int | str, int]] = {}
        paths_from: dict[int, dict[int | str, list[int | str]]] = {}
        for u in defect_indices:
            lengths_from[u] = dict(nx.single_source_shortest_path_length(G_stab, u))
            paths_from[u] = dict(nx.single_source_shortest_path(G_stab, u))

        G_match = nx.Graph()
        big_m = 10 * max(code_distance, len(stabs), 1)

        for i in range(n):
            u = defect_indices[i]
            for j in range(i + 1, n):
                v = defect_indices[j]
                dist_uv = lengths_from[u].get(v, big_m)
                G_match.add_edge(i, j, weight=float(dist_uv))

            dist_ub = lengths_from[u].get("B", big_m)
            for b in range(n, 2 * n):
                G_match.add_edge(i, b, weight=float(dist_ub))

        for b1 in range(n, 2 * n):
            for b2 in range(b1 + 1, 2 * n):
                G_match.add_edge(b1, b2, weight=0.0)

        matching = nx.min_weight_matching(G_match)

        # Accumulate data qubits along matched paths over GF(2) (XOR)
        correction_set: set[int] = set()
        for u_node, v_node in matching:
            if u_node >= n and v_node >= n:
                continue
            if u_node < n and v_node < n:
                s1 = defect_indices[u_node]
                s2 = defect_indices[v_node]
                if s2 in paths_from[s1]:
                    node_path = paths_from[s1][s2]
                    edge_qubits = {
                        int(G_stab[node_path[k]][node_path[k + 1]]["qubit"])
                        for k in range(len(node_path) - 1)
                    }
                    correction_set.symmetric_difference_update(edge_qubits)
            else:
                real_idx = u_node if u_node < n else v_node
                s1 = defect_indices[real_idx]
                if "B" in paths_from[s1]:
                    node_path = paths_from[s1]["B"]
                    edge_qubits = {
                        int(G_stab[node_path[k]][node_path[k + 1]]["qubit"])
                        for k in range(len(node_path) - 1)
                    }
                    correction_set.symmetric_difference_update(edge_qubits)

        correction_tuple = tuple(sorted(correction_set))
        ok = _verify_syndrome_clearance(syn, stabs, correction_set)
        p_char = error_type.upper()
        pauli_map = {q: p_char for q in correction_tuple}

        return DecoderResult(
            correction=correction_tuple,
            weight=len(correction_tuple),
            algorithm=self.name,
            success=ok,
            pauli=pauli_map,
        )

    @staticmethod
    def _greedy_matching(
        dist: np.ndarray, n: int
    ) -> list[tuple[int, int]]:
        """Greedy minimum weight matching (exposed for benchmark comparison)."""
        used: set[int] = set()
        pairs: list[tuple[int, int]] = []

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((dist[i, j], i, j))
        edges.sort()

        for _, i, j in edges:
            if i not in used and j not in used:
                pairs.append((i, j))
                used.add(i)
                used.add(j)
                if len(used) >= n - 1:
                    break

        return pairs


class UnionFindDecoder(DecoderBase):
    """Delfosse-Nickerson Union-Find + Spanning-Forest Peeling decoder.

    Reference:
        N. Delfosse and N. H. Nickerson, "Almost-linear time decoding algorithm
        for topological codes", Quantum 5, 595 (2021).

    Algorithm:
      1. Build Tanner decoding graph: check vertices 0..m-1 plus virtual
         boundary vertex B = m; physical data qubits form edges (capacity 2
         half-edges each).
      2. Weighted cluster growth: grow odd-parity clusters not yet touching B
         along incident half-edges (smallest cluster first) and merge components
         via disjoint-set union whenever an edge reaches 2 half-edges.
      3. Spanning-forest peeling: build a spanning tree per grown component
         (rooted at B if boundary-connected) and peel leaves in reverse BFS
         post-order over GF(2) to emit physical data-qubit corrections.
    """

    name = "Union-Find"

    def decode(
        self,
        syndrome: np.ndarray,
        code_distance: int,
        stabilizers: list[list[int]] | None = None,
        error_type: str = "X",
    ) -> DecoderResult:
        syn = np.asarray(syndrome, dtype=bool).ravel()

        # Combined [syn_x, syn_z] CSS syndrome on rotated surface code
        if (
            stabilizers is None
            and code_distance >= 3
            and code_distance % 2 == 1
            and len(syn) == code_distance * code_distance - 1
        ):
            half = len(syn) // 2
            x_stabs, z_stabs = _build_rotated_surface_stabilizers(code_distance)
            res_z = self.decode(syn[:half], code_distance, stabilizers=x_stabs, error_type="Z")
            res_x = self.decode(syn[half:], code_distance, stabilizers=z_stabs, error_type="X")
            return _merge_css_results(res_z, res_x, self.name)

        stabs = _resolve_stabilizers(syn, code_distance, stabilizers, error_type)
        if not np.any(syn):
            return DecoderResult(
                correction=(),
                weight=0,
                algorithm=self.name,
                success=True,
                pauli={},
            )

        m = len(stabs)
        boundary_node = m
        n_verts = m + 1
        n_qubits = max((max(s) for s in stabs if s), default=-1) + 1

        # Build edges: each physical data qubit q gives an edge (u, v, q)
        edges: list[tuple[int, int, int]] = []
        incident_edges: list[list[int]] = [[] for _ in range(n_verts)]
        for q in range(n_qubits):
            inc = [s_idx for s_idx, s in enumerate(stabs) if q in s]
            if len(inc) == 2:
                u, v = inc[0], inc[1]
            elif len(inc) == 1:
                u, v = inc[0], boundary_node
            else:
                continue
            e_id = len(edges)
            edges.append((u, v, q))
            incident_edges[u].append(e_id)
            incident_edges[v].append(e_id)

        n_edges = len(edges)
        edge_growth = [0] * n_edges  # 0, 1, or 2 half-edges

        # Disjoint-set union state
        parent = list(range(n_verts))
        comp_size = [1] * n_verts
        parity = [int(bool(syn[u])) for u in range(m)] + [0]
        at_boundary = [False] * m + [True]
        boundary_lists: list[list[int]] = [[v] for v in range(n_verts)]

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> int:
            rx, ry = find(x), find(y)
            if rx == ry:
                return rx
            if comp_size[rx] < comp_size[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            comp_size[rx] += comp_size[ry]
            parity[rx] ^= parity[ry]
            at_boundary[rx] = at_boundary[rx] or at_boundary[ry]
            boundary_lists[rx].extend(boundary_lists[ry])
            boundary_lists[ry].clear()
            return rx

        # Phase 1: Weighted half-edge cluster growth (Delfosse-Nickerson Sec. 3.3)
        while True:
            odd_roots = [
                r
                for r in range(m)
                if parent[r] == r and parity[r] == 1 and not at_boundary[r]
            ]
            if not odd_roots:
                break

            # Grow smallest active odd cluster(s) by 1 half-edge
            min_sz = min(comp_size[r] for r in odd_roots)
            chosen_roots = [r for r in odd_roots if comp_size[r] == min_sz]

            newly_grown: list[int] = []
            progressed = False

            for r in chosen_roots:
                if parent[r] != r or parity[r] == 0 or at_boundary[r]:
                    continue
                new_b_verts: list[int] = []
                candidate_edges: list[int] = []
                for v in boundary_lists[r]:
                    has_external = False
                    for e_id in incident_edges[v]:
                        if edge_growth[e_id] < 2:
                            u_e, v_e, _ = edges[e_id]
                            if find(u_e) != find(v_e):
                                has_external = True
                                candidate_edges.append(e_id)
                    if has_external:
                        new_b_verts.append(v)
                boundary_lists[r] = new_b_verts

                for e_id in candidate_edges:
                    if edge_growth[e_id] < 2:
                        edge_growth[e_id] += 1
                        progressed = True
                        if edge_growth[e_id] == 2:
                            newly_grown.append(e_id)

            for e_id in newly_grown:
                u_e, v_e, _ = edges[e_id]
                union(u_e, v_e)

            if not progressed:
                break

        # Phase 2: Build spanning forest over fully grown edges (edge_growth == 2)
        grown_adj: list[list[tuple[int, int]]] = [[] for _ in range(n_verts)]
        touched_vertices: set[int] = set()
        for e_id, (u_e, v_e, q_e) in enumerate(edges):
            if edge_growth[e_id] == 2:
                grown_adj[u_e].append((v_e, q_e))
                grown_adj[v_e].append((u_e, q_e))
                touched_vertices.add(u_e)
                touched_vertices.add(v_e)

        visited = [False] * n_verts
        tree_edges_bfs: list[tuple[int, int, int]] = []  # (child, parent_v, qubit)

        # Visit boundary-connected component FIRST, rooted at boundary_node!
        start_order: list[int] = []
        if boundary_node in touched_vertices:
            start_order.append(boundary_node)
        start_order.extend(sorted(v for v in touched_vertices if v != boundary_node))

        for root_v in start_order:
            if visited[root_v]:
                continue
            visited[root_v] = True
            queue: deque[int] = deque([root_v])
            while queue:
                curr = queue.popleft()
                for nxt, q_e in grown_adj[curr]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        tree_edges_bfs.append((nxt, curr, q_e))
                        queue.append(nxt)

        # Phase 3: Peel spanning forest leaves-to-root in reverse BFS post-order
        defects = [int(bool(syn[u])) for u in range(m)] + [0]
        correction_set: set[int] = set()

        for child, par, q_e in reversed(tree_edges_bfs):
            if defects[child] == 1:
                if q_e in correction_set:
                    correction_set.remove(q_e)
                else:
                    correction_set.add(q_e)
                defects[child] = 0
                if par != boundary_node:
                    defects[par] ^= 1

        correction_tuple = tuple(sorted(correction_set))
        ok = _verify_syndrome_clearance(syn, stabs, correction_set)
        p_char = error_type.upper()
        pauli_map = {q: p_char for q in correction_tuple}

        return DecoderResult(
            correction=correction_tuple,
            weight=len(correction_tuple),
            algorithm=self.name,
            success=ok,
            pauli=pauli_map,
        )
