"""
quanta.qec.decoder -- Quantum error correction decoders.

Decoders take a syndrome (stabilizer measurement results) and determine
which correction to apply. All decoders implement the ``DecoderBase``
abstract interface.

Provided decoders:

  - MWPMDecoder: Minimum Weight Perfect Matching
    Optimal but O(n^3). Pairs syndrome defects with minimum total weight.

  - UnionFindDecoder: Union-Find based decoder
    Near-linear O(n·α(n)). Clusters defects using union-find, then
    corrects each cluster independently.

To create a custom decoder (e.g., ML-based), subclass ``DecoderBase``
and implement the ``decode()`` method.

Example:
    >>> from quanta.qec.decoder import MWPMDecoder, UnionFindDecoder
    >>> from quanta.qec.surface_code import SurfaceCode
    >>> decoder = MWPMDecoder()
    >>> correction = decoder.decode(syndrome, code_distance=3)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

__all__ = [
    "DecoderBase",
    "MWPMDecoder",
    "UnionFindDecoder",
    "DecoderResult",
    "CorrectionResult",
]


@dataclass
class DecoderResult:
    """Result of decoding a syndrome.

    Attributes:
        correction: Indices of qubits to correct.
        success: Whether the decoder believes correction will succeed.
        weight: Total weight (distance) of the correction.
        pauli: Mapping of physical qubit index to Pauli correction operator ('X', 'Y', 'Z').
        pauli_string: String representation of Pauli correction on physical qubits.
    """
    correction: tuple[int, ...]
    success: bool
    weight: int
    pauli: dict[int, str] = field(default_factory=dict)
    pauli_string: str = ""


CorrectionResult = DecoderResult


class DecoderBase(abc.ABC):
    """Abstract base class for QEC decoders.

    All decoders must implement the ``decode()`` method. This enables
    plugin-based decoder architectures — subclass ``DecoderBase`` to
    create custom decoders (e.g., ML-based, lookup-table, etc.).

    Example:
        >>> class MyDecoder(DecoderBase):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my-decoder"
        ...     def decode(self, syndrome, code_distance, lattice_size=None):
        ...         # custom decoding logic
        ...         return DecoderResult(correction=(), success=True, weight=0)
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable decoder name."""

    @abc.abstractmethod
    def decode(
        self,
        syndrome: np.ndarray,
        code_distance: int,
        lattice_size: int | None = None,
        stabilizers: list[list[int]] | None = None,
        error_type: str = "X",
    ) -> DecoderResult:
        """Decodes a syndrome into a correction.

        Args:
            syndrome: Boolean array of excited stabilizers.
            code_distance: Code distance d.
            lattice_size: Lattice dimension (default: d).
            stabilizers: Optional list of stabilizer support data qubit lists.
            error_type: Type of Pauli correction ('X' or 'Z').

        Returns:
            DecoderResult with correction qubits and success flag.
        """


class MWPMDecoder(DecoderBase):
    """Minimum Weight Perfect Matching decoder using Edmonds' Blossom algorithm.

    Pairs syndrome defects (excited stabilizers) such that the total
    graph distance is strictly minimized. Uses ``networkx.min_weight_matching``
    over a complete bipartite/replicated boundary defect graph to allow
    independent boundary matching regardless of defect parity (even or odd).

    Algorithm:
        1. Construct defect graph with k defects and k virtual boundary nodes.
        2. Pairwise distances: shortest path on stabilizer graph (if stabilizers
           provided) or Manhattan distance on the lattice.
        3. Virtual boundary-to-boundary edges with weight 0 ensure exact perfect
           matching for any defect subset.
        4. Solve minimum weight perfect matching via Edmonds' Blossom algorithm.
        5. Reconstruct shortest-path Pauli correction chains on primal/dual lattice.

    Complexity: O(V^3) where V is the defect count.
    """

    @property
    def name(self) -> str:
        """Decoder name."""
        return "MWPM"

    def decode(
        self,
        syndrome: np.ndarray,
        code_distance: int,
        lattice_size: int | None = None,
        stabilizers: list[list[int]] | None = None,
        error_type: str = "X",
    ) -> DecoderResult:
        """Decodes a syndrome into a correction.

        Args:
            syndrome: Boolean array of excited stabilizers.
            code_distance: Code distance d.
            lattice_size: Lattice dimension (default: d).
            stabilizers: Optional stabilizer qubit support for exact physical mapping.
            error_type: Pauli operator type ('X' or 'Z').

        Returns:
            DecoderResult with correction qubits and success flag.
        """
        d = lattice_size or code_distance
        defects = np.where(syndrome)[0]
        k = len(defects)

        if k == 0:
            n_q = d * d if stabilizers is None else max(max(s, default=0) for s in stabilizers) + 1
            return DecoderResult(
                correction=(),
                success=True,
                weight=0,
                pauli={},
                pauli_string="I" * n_q,
            )

        G = nx.Graph()

        if stabilizers is not None:
            # Build stabilizer graph where edges are physical data qubits
            G_stab = nx.Graph()
            all_qubits: set[int] = set()
            for s in stabilizers:
                all_qubits.update(s)
            for q in all_qubits:
                inc = [idx for idx, s in enumerate(stabilizers) if q in s]
                if len(inc) == 2:
                    G_stab.add_edge(inc[0], inc[1], qubit=q, weight=1)
                elif len(inc) == 1:
                    G_stab.add_edge(inc[0], "B", qubit=q, weight=1)

            def dist_stabs(s1: int, s2: int) -> int:
                if nx.has_path(G_stab, s1, s2):
                    return int(nx.shortest_path_length(G_stab, s1, s2, weight="weight"))
                return 2 * d

            def dist_bound(s: int) -> int:
                if nx.has_path(G_stab, s, "B"):
                    return int(nx.shortest_path_length(G_stab, s, "B", weight="weight"))
                return d

            for i in range(k):
                s_i = int(defects[i])
                for j in range(i + 1, k):
                    s_j = int(defects[j])
                    G.add_edge(i, j, weight=dist_stabs(s_i, s_j))
                w_b = dist_bound(s_i)
                for b in range(k, 2 * k):
                    G.add_edge(i, b, weight=w_b)
        else:
            coords = [divmod(int(x), d) for x in defects]
            b_dists = [min(r, c, d - 1 - r, d - 1 - c) + 1 for r, c in coords]
            for i in range(k):
                r1, c1 = coords[i]
                for j in range(i + 1, k):
                    r2, c2 = coords[j]
                    G.add_edge(i, j, weight=abs(r1 - r2) + abs(c1 - c2))
                for b in range(k, 2 * k):
                    G.add_edge(i, b, weight=b_dists[i])

        # Zero-weight edges between all virtual boundary nodes
        for b1 in range(k, 2 * k):
            for b2 in range(b1 + 1, 2 * k):
                G.add_edge(b1, b2, weight=0)

        # Edmonds' Blossom Minimum Weight Perfect Matching
        matching = nx.min_weight_matching(G)

        correction: set[int] = set()
        total_weight = 0
        num_matches = 0

        if stabilizers is not None:
            for u, v in matching:
                if u >= k and v >= k:
                    continue
                num_matches += 1
                if u < k and v < k:
                    s1 = int(defects[u])
                    s2 = int(defects[v])
                    if nx.has_path(G_stab, s1, s2):
                        path = nx.shortest_path(G_stab, s1, s2, weight="weight")
                        for p_idx in range(len(path) - 1):
                            correction.add(G_stab[path[p_idx]][path[p_idx + 1]]["qubit"])
                        total_weight += nx.shortest_path_length(G_stab, s1, s2, weight="weight")
                else:
                    def_idx = u if u < k else v
                    s = int(defects[def_idx])
                    if nx.has_path(G_stab, s, "B"):
                        path = nx.shortest_path(G_stab, s, "B", weight="weight")
                        for p_idx in range(len(path) - 1):
                            correction.add(G_stab[path[p_idx]][path[p_idx + 1]]["qubit"])
                        total_weight += nx.shortest_path_length(G_stab, s, "B", weight="weight")
        else:
            for u, v in matching:
                if u >= k and v >= k:
                    continue
                num_matches += 1
                if u < k and v < k:
                    r1, c1 = coords[u]
                    r2, c2 = coords[v]
                    w = abs(r1 - r2) + abs(c1 - c2)
                    total_weight += w
                    curr_r, curr_c = r1, c1
                    correction.add(curr_r * d + curr_c)
                    while curr_r != r2:
                        curr_r += 1 if r2 > curr_r else -1
                        correction.add(curr_r * d + curr_c)
                    while curr_c != c2:
                        curr_c += 1 if c2 > curr_c else -1
                        correction.add(curr_r * d + curr_c)
                else:
                    def_idx = u if u < k else v
                    r, c = coords[def_idx]
                    w = b_dists[def_idx]
                    total_weight += w
                    correction.add(r * d + c)
                    min_edge = min(r, c, d - 1 - r, d - 1 - c)
                    curr_r, curr_c = r, c
                    if min_edge == r:
                        while curr_r > 0:
                            curr_r -= 1
                            correction.add(curr_r * d + curr_c)
                    elif min_edge == d - 1 - r:
                        while curr_r < d - 1:
                            curr_r += 1
                            correction.add(curr_r * d + curr_c)
                    elif min_edge == c:
                        while curr_c > 0:
                            curr_c -= 1
                            correction.add(curr_r * d + curr_c)
                    else:
                        while curr_c < d - 1:
                            curr_c += 1
                            correction.add(curr_r * d + curr_c)

        t = (code_distance - 1) // 2
        success = bool(total_weight <= max(1, t * max(1, num_matches)))
        corr_tuple = tuple(sorted(correction))
        pauli_map = {q: error_type for q in corr_tuple}

        num_qubits = d * d
        if stabilizers is not None:
            max_q = max((max(s, default=0) for s in stabilizers), default=d * d - 1)
            num_qubits = max(num_qubits, max_q + 1)
        elif corr_tuple:
            num_qubits = max(num_qubits, max(corr_tuple) + 1)

        pauli_chars = ["I"] * num_qubits
        for q in corr_tuple:
            if q < num_qubits:
                pauli_chars[q] = error_type
        pauli_str = "".join(pauli_chars)

        return DecoderResult(
            correction=corr_tuple,
            success=success,
            weight=int(total_weight),
            pauli=pauli_map,
            pauli_string=pauli_str,
        )

    @staticmethod
    def _greedy_matching(dist: np.ndarray, n: int) -> list[tuple[int, int]]:
        """Greedy approximation to minimum weight perfect matching.

        Retained for baseline benchmarking and backward compatibility.
        """
        matched = set()
        pairs: list[tuple[int, int]] = []

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((dist[i, j], i, j))
        edges.sort()

        for _w, i, j in edges:
            if i not in matched and j not in matched:
                pairs.append((i, j))
                matched.add(i)
                matched.add(j)
            if len(matched) == n:
                break

        return pairs


class UnionFindDecoder(DecoderBase):
    """Union-Find based decoder.

    Clusters syndrome defects into groups using the union-find data
    structure, then corrects each cluster. Near-linear time complexity
    makes it practical for large codes.

    Algorithm:
        1. Initialize each defect as its own cluster
        2. Grow clusters by increasing radius
        3. Merge overlapping clusters via union-find
        4. For each fully-grown cluster, apply minimum correction

    Complexity: O(n·α(n)) amortized, where α is inverse Ackermann.
    """

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._rank: dict[int, int] = {}

    @property
    def name(self) -> str:
        """Decoder name."""
        return "Union-Find"

    def _find(self, x: int) -> int:
        """Find with path compression."""
        if self._parent[x] != x:
            self._parent[x] = self._find(self._parent[x])
        return self._parent[x]

    def _union(self, x: int, y: int) -> None:
        """Union by rank."""
        rx, ry = self._find(x), self._find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def decode(
        self,
        syndrome: np.ndarray,
        code_distance: int,
        lattice_size: int | None = None,
        stabilizers: list[list[int]] | None = None,
        error_type: str = "X",
    ) -> DecoderResult:
        """Decodes a syndrome using union-find clustering.

        Args:
            syndrome: Boolean array of excited stabilizers.
            code_distance: Code distance d.
            lattice_size: Lattice dimension (default: d).
            stabilizers: Optional stabilizer qubit support.
            error_type: Pauli operator type ('X' or 'Z').

        Returns:
            DecoderResult with correction qubits and success flag.
        """
        d = lattice_size or code_distance
        defects = list(np.where(syndrome)[0])

        if not defects:
            n_q = d * d if stabilizers is None else max(max(s, default=0) for s in stabilizers) + 1
            return DecoderResult(
                correction=(),
                success=True,
                weight=0,
                pauli={},
                pauli_string="I" * n_q,
            )

        # Initialize union-find
        self._parent = {i: i for i in defects}
        self._rank = {i: 0 for i in defects}

        # Grow clusters: merge defects within radius r
        for radius in range(1, d + 1):
            for i, d1 in enumerate(defects):
                for d2 in defects[i + 1:]:
                    r1, c1 = divmod(int(d1), d)
                    r2, c2 = divmod(int(d2), d)
                    dist = abs(r1 - r2) + abs(c1 - c2)
                    if dist <= radius:
                        self._union(int(d1), int(d2))

            # Check: all clusters have even parity?
            clusters: dict[int, list[int]] = {}
            for defect in defects:
                root = self._find(int(defect))
                clusters.setdefault(root, []).append(int(defect))

            all_even = all(len(v) % 2 == 0 for v in clusters.values())
            if all_even:
                break

        # Build correction from clusters
        correction = set()
        total_weight = 0
        for members in clusters.values():
            if len(members) >= 2:
                # Connect consecutive defects within cluster
                members_sorted = sorted(members)
                for k in range(0, len(members_sorted) - 1, 2):
                    correction.add(members_sorted[k])
                    correction.add(members_sorted[k + 1])
                    r1, c1 = divmod(members_sorted[k], d)
                    r2, c2 = divmod(members_sorted[k + 1], d)
                    total_weight += abs(r1 - r2) + abs(c1 - c2)
            elif len(members) == 1:
                # Boundary correction
                correction.add(members[0])
                row, col = divmod(members[0], d)
                total_weight += min(row, col, d - 1 - row, d - 1 - col) + 1

        t = (code_distance - 1) // 2
        success = len(defects) <= 2 * t
        corr_tuple = tuple(sorted(correction))
        pauli_map = {q: error_type for q in corr_tuple}

        num_qubits = d * d
        if stabilizers is not None:
            max_q = max((max(s, default=0) for s in stabilizers), default=d * d - 1)
            num_qubits = max(num_qubits, max_q + 1)
        elif corr_tuple:
            num_qubits = max(num_qubits, max(corr_tuple) + 1)

        pauli_chars = ["I"] * num_qubits
        for q in corr_tuple:
            if q < num_qubits:
                pauli_chars[q] = error_type
        pauli_str = "".join(pauli_chars)

        return DecoderResult(
            correction=corr_tuple,
            success=success,
            weight=total_weight,
            pauli=pauli_map,
            pauli_string=pauli_str,
        )
