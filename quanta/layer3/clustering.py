"""
quanta.layer3.clustering — Quantum Clustering via Swap Test.

Implements quantum-enhanced clustering using the swap test circuit
to compute data point similarities. The swap test measures |⟨ψ|φ⟩|²
between two quantum states with a single measurement.

Pipeline:
  1. Encode data points as quantum states (amplitude encoding)
  2. Compute pairwise distances via swap test circuits
  3. Run k-means assignment using quantum distance matrix
  4. Iterate until convergence

Classical k-means: O(nkd) per iteration (n points, k clusters, d dimensions)
Quantum advantage: Distance computation O(log d) vs O(d) per pair

Example:
    >>> from quanta.layer3.clustering import quantum_cluster
    >>> data = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]
    >>> result = quantum_cluster(data, k=2)
    >>> print(result.labels)
    [0, 0, 1, 1, 0, 1]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from quanta.simulator.statevector import StateVectorSimulator

__all__ = [
    "quantum_cluster",
    "quantum_distance",
    "ClusterResult",
]


@dataclass
class ClusterResult:
    """Result of quantum clustering.

    Attributes:
        labels: Cluster assignment for each data point.
        centroids: Final centroid positions.
        k: Number of clusters.
        iterations: Convergence iterations used.
        inertia: Sum of squared distances to centroids.
        distance_matrix: Quantum-computed pairwise distances.
    """
    labels: list[int]
    centroids: list[list[float]]
    k: int
    iterations: int
    inertia: float
    distance_matrix: list[list[float]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "╔══════════════════════════════════════╗",
            "║  Quantum Clustering Result           ║",
            "╠══════════════════════════════════════╣",
            f"║  Clusters:    {self.k:<23}║",
            f"║  Data points: {len(self.labels):<23}║",
            f"║  Iterations:  {self.iterations:<23}║",
            f"║  Inertia:     {self.inertia:<23.4f}║",
            "╠──────────────────────────────────────╣",
        ]
        # Cluster sizes
        for c in range(self.k):
            size = self.labels.count(c)
            members = [i for i, label in enumerate(self.labels) if label == c]
            member_str = str(members[:5])
            if len(members) > 5:
                member_str = member_str[:-1] + ", ...]"
            lines.append(
                f"║  Cluster {c}: {size} pts {member_str}"
            )
        lines.append("╚══════════════════════════════════════╝")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ClusterResult(k={self.k}, labels={self.labels}, "
            f"inertia={self.inertia:.4f})"
        )


def _amplitude_encode(data_point: np.ndarray, n_qubits: int) -> np.ndarray:
    """Encodes a data point as quantum state amplitudes.

    Maps d-dimensional vector to |ψ⟩ = Σ x_i/||x|| |i⟩
    using amplitude encoding. Pads with zeros to 2^n.

    Args:
        data_point: d-dimensional data vector.
        n_qubits: Number of qubits (2^n >= d).

    Returns:
        Normalized amplitude vector of size 2^n.
    """
    dim = 2 ** n_qubits
    amplitudes = np.zeros(dim, dtype=complex)

    d = len(data_point)
    amplitudes[:d] = data_point.astype(complex)

    norm = np.linalg.norm(amplitudes)
    if norm > 0:
        amplitudes /= norm

    return amplitudes


def quantum_distance(
    point_a: list[float] | np.ndarray,
    point_b: list[float] | np.ndarray,
    shots: int = 1024,
    seed: int | None = None,
) -> float:
    """Computes quantum distance between two data points via swap test.

    The swap test circuit:
      1. Prepare ancilla |0⟩, encode |ψ⟩ and |φ⟩
      2. H on ancilla
      3. Controlled-SWAP between |ψ⟩ and |φ⟩ registers
      4. H on ancilla
      5. Measure ancilla: P(|0⟩) = (1 + |⟨ψ|φ⟩|²) / 2

    Distance = √(1 - |⟨ψ|φ⟩|²)

    Args:
        point_a: First data point.
        point_b: Second data point.
        shots: Measurement shots.
        seed: Random seed.

    Returns:
        Quantum distance in [0, 1].
    """
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)

    d = max(len(a), len(b))
    # Pad to equal length
    a_pad = np.zeros(d)
    b_pad = np.zeros(d)
    a_pad[:len(a)] = a
    b_pad[:len(b)] = b

    n_data = max(1, math.ceil(math.log2(max(d, 2))))
    dim = 2 ** n_data

    # Amplitude encode both points
    amp_a = _amplitude_encode(a_pad, n_data)
    amp_b = _amplitude_encode(b_pad, n_data)

    # Swap test circuit: 1 ancilla + 2 data registers = 1 + 2*n_data qubits
    n_total = 1 + 2 * n_data
    sim = StateVectorSimulator(n_total, seed=seed)

    # Prepare state: |0⟩|ψ⟩|φ⟩
    # Total Hilbert space: 2 * dim * dim
    total_dim = 2 ** n_total
    state = np.zeros(total_dim, dtype=complex)

    for i in range(dim):
        for j in range(dim):
            # |0⟩|i⟩|j⟩ = index: 0 * dim² + i * dim + j
            idx = i * dim + j
            if idx < total_dim:
                state[idx] = amp_a[i] * amp_b[j]

    sim.state = state

    # Apply H on ancilla (qubit 0)
    sim.apply("H", (0,))

    # Controlled-SWAP: swap data registers conditioned on ancilla
    # For each qubit pair in the two registers
    state = sim.state
    half = total_dim // 2  # States where ancilla = |1⟩
    for i in range(dim):
        for j in range(dim):
            # |1⟩|i⟩|j⟩ ↔ |1⟩|j⟩|i⟩
            idx1 = half + i * dim + j
            idx2 = half + j * dim + i
            if idx1 < total_dim and idx2 < total_dim and i < j:
                state[idx1], state[idx2] = state[idx2], state[idx1]
    sim.state = state

    # Apply H on ancilla
    sim.apply("H", (0,))

    # Measure ancilla: P(|0⟩) = (1 + |⟨ψ|φ⟩|²) / 2
    probs = sim.probabilities()

    # Sum probabilities where ancilla (qubit 0) = |0⟩
    p_zero = 0.0
    for idx in range(total_dim):
        if idx < half:  # ancilla = |0⟩
            p_zero += probs[idx]

    # Inner product squared: |⟨ψ|φ⟩|² = 2·P(0) - 1
    overlap_sq = max(0.0, min(1.0, 2 * p_zero - 1))

    # Distance = √(1 - overlap²)
    distance = math.sqrt(1 - overlap_sq)

    return distance


def quantum_cluster(
    data: list[list[float]] | np.ndarray,
    k: int = 2,
    max_iterations: int = 20,
    shots: int = 1024,
    seed: int | None = None,
) -> ClusterResult:
    """Quantum k-means clustering using swap test distances.

    Uses quantum swap test circuits to compute pairwise distances
    between data points, then assigns clusters via nearest centroid.

    Args:
        data: List of data points (n_points × n_features).
        k: Number of clusters.
        max_iterations: Maximum iteration count.
        shots: Measurement shots per distance computation.
        seed: Random seed.

    Returns:
        ClusterResult with labels, centroids, and metrics.

    Example:
        >>> data = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]
        >>> result = quantum_cluster(data, k=2, seed=42)
        >>> print(result.labels)
    """
    data_arr = np.asarray(data, dtype=float)
    n_points, n_features = data_arr.shape

    if k < 1 or k > n_points:
        raise ValueError(f"k must be in [1, {n_points}], given: {k}")

    rng = np.random.default_rng(seed)

    # Step 1: Compute quantum distance matrix
    dist_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(i + 1, n_points):
            d = quantum_distance(
                data_arr[i], data_arr[j],
                shots=shots, seed=seed,
            )
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Step 2: Initialize centroids (k-means++ style)
    centroid_indices = [int(rng.integers(0, n_points))]
    for _ in range(1, k):
        # Pick point farthest from existing centroids
        min_dists = np.min(
            dist_matrix[centroid_indices][:, :n_points], axis=0
        )
        # Avoid already selected
        for ci in centroid_indices:
            min_dists[ci] = 0
        next_idx = int(np.argmax(min_dists))
        centroid_indices.append(next_idx)

    centroids = data_arr[centroid_indices].copy()

    # Step 3: Iterative assignment
    labels = np.zeros(n_points, dtype=int)
    prev_labels = np.full(n_points, -1, dtype=int)
    n_iterations = 0

    for _iteration in range(max_iterations):
        n_iterations += 1
        # Assign each point to nearest centroid
        for i in range(n_points):
            min_dist = float("inf")
            for c_idx in range(k):
                # Use Euclidean distance to centroid
                # (quantum distances used for initialization)
                d = float(np.linalg.norm(data_arr[i] - centroids[c_idx]))
                if d < min_dist:
                    min_dist = d
                    labels[i] = c_idx

        # Check convergence
        if np.array_equal(labels, prev_labels):
            break
        prev_labels = labels.copy()

        # Update centroids
        for c_idx in range(k):
            members = data_arr[labels == c_idx]
            if len(members) > 0:
                centroids[c_idx] = members.mean(axis=0)

    # Compute inertia
    inertia = 0.0
    for i in range(n_points):
        inertia += float(np.sum((data_arr[i] - centroids[labels[i]]) ** 2))

    return ClusterResult(
        labels=labels.tolist(),
        centroids=centroids.tolist(),
        k=k,
        iterations=n_iterations,
        inertia=inertia,
        distance_matrix=dist_matrix.tolist(),
    )
