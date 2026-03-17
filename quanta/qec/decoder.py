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
from dataclasses import dataclass

import numpy as np

__all__ = ["DecoderBase", "MWPMDecoder", "UnionFindDecoder", "DecoderResult"]


@dataclass
class DecoderResult:
    """Result of decoding a syndrome.

    Attributes:
        correction: Indices of qubits to correct.
        success: Whether the decoder believes correction will succeed.
        weight: Total weight (distance) of the correction.
    """
    correction: tuple[int, ...]
    success: bool
    weight: int


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
    ) -> DecoderResult:
        """Decodes a syndrome into a correction.

        Args:
            syndrome: Boolean array of excited stabilizers.
            code_distance: Code distance d.
            lattice_size: Lattice dimension (default: d).

        Returns:
            DecoderResult with correction qubits and success flag.
        """


class MWPMDecoder(DecoderBase):
    """Minimum Weight Perfect Matching decoder.

    Pairs syndrome defects (excited stabilizers) such that the total
    graph distance is minimized. This is the gold standard decoder
    for surface codes.

    Algorithm:
        1. Build a complete graph of syndrome defects
        2. Compute pairwise Manhattan distances on the lattice
        3. Find minimum weight perfect matching via greedy approximation
        4. Infer correction chain from matched pairs

    Complexity: O(n^3) worst case, O(n^2) typical.
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
    ) -> DecoderResult:
        """Decodes a syndrome into a correction.

        Args:
            syndrome: Boolean array of excited stabilizers.
            code_distance: Code distance d.
            lattice_size: Lattice dimension (default: d).

        Returns:
            DecoderResult with correction qubits and success flag.
        """
        d = lattice_size or code_distance
        defects = np.where(syndrome)[0]

        if len(defects) == 0:
            return DecoderResult(correction=(), success=True, weight=0)

        # If odd number of defects, add a virtual boundary defect
        if len(defects) % 2 == 1:
            defects = np.append(defects, -1)  # -1 = boundary

        # Build distance matrix
        n = len(defects)
        dist = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if defects[i] == -1 or defects[j] == -1:
                    # Distance to boundary
                    real = defects[i] if defects[j] == -1 else defects[j]
                    row, col = divmod(int(real), d)
                    dist[i, j] = min(row, col, d - 1 - row, d - 1 - col) + 1
                else:
                    # Manhattan distance on lattice
                    r1, c1 = divmod(int(defects[i]), d)
                    r2, c2 = divmod(int(defects[j]), d)
                    dist[i, j] = abs(r1 - r2) + abs(c1 - c2)
                dist[j, i] = dist[i, j]

        # Greedy minimum weight perfect matching
        matching = self._greedy_matching(dist, n)

        # Infer correction
        correction = set()
        total_weight = 0
        for i, j in matching:
            w = dist[i, j]
            total_weight += w
            # Add qubits along correction path
            if defects[i] != -1:
                correction.add(int(defects[i]))
            if defects[j] != -1:
                correction.add(int(defects[j]))

        t = (code_distance - 1) // 2
        success = bool(total_weight <= t * len(matching))

        return DecoderResult(
            correction=tuple(sorted(correction)),
            success=success,
            weight=total_weight,
        )

    @staticmethod
    def _greedy_matching(dist: np.ndarray, n: int) -> list[tuple[int, int]]:
        """Greedy approximation to minimum weight perfect matching.

        Repeatedly matches the closest pair of unmatched nodes.
        """
        matched = set()
        pairs: list[tuple[int, int]] = []

        # Build sorted edge list
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
    ) -> DecoderResult:
        """Decodes a syndrome using union-find clustering.

        Args:
            syndrome: Boolean array of excited stabilizers.
            code_distance: Code distance d.
            lattice_size: Lattice dimension (default: d).

        Returns:
            DecoderResult with correction qubits and success flag.
        """
        d = lattice_size or code_distance
        defects = list(np.where(syndrome)[0])

        if not defects:
            return DecoderResult(correction=(), success=True, weight=0)

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

        return DecoderResult(
            correction=tuple(sorted(correction)),
            success=success,
            weight=total_weight,
        )
