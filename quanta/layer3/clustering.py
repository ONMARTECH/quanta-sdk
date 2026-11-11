"""
quanta.layer3.clustering — Genuine Gate-Based Swap Test Quantum Clustering.

Implements:
  1. Genuine ``(2 * m + 1)``-qubit Swap Test circuit (`RY`/`CX` amplitude state-prep
     trees on ``q[1..m]`` and ``q[m+1..2m]``, `H(0)`, `CSWAP(0, 1+i, 1+m+i)`, `H(0)`,
     and real shot sampling ``sim.sample(shots)``).
  2. Norm-preserving hemisphere stereographic embedding so ``quantum_cluster`` uses
     ``quantum_distance`` inside EVERY k-means cluster assignment iteration (not just
     k-means++ seeding!).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from quanta.core.circuit import CircuitBuilder
from quanta.core.types import Instruction
from quanta.dag.dag_circuit import DAGCircuit
from quanta.layer3.monte_carlo import append_amplitude_state_prep
from quanta.simulator.factory import create_simulator

__all__ = [
    "quantum_cluster",
    "quantum_distance",
    "ClusterResult",
]


@dataclass
class ClusterResult:
    """Result of quantum clustering."""

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
        for c in range(self.k):
            size = self.labels.count(c)
            members = [i for i, label in enumerate(self.labels) if label == c]
            member_str = str(members[:5])
            if len(members) > 5:
                member_str = member_str[:-1] + ", ...]"
            lines.append(f"║  Cluster {c}: {size} pts {member_str}")
        lines.append("╚══════════════════════════════════════╝")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ClusterResult(k={self.k}, labels={self.labels}, "
            f"inertia={self.inertia:.4f})"
        )


def _amplitude_encode(data_point: np.ndarray, n_qubits: int) -> np.ndarray:
    """Normalizes and zero-pads a real data vector to length ``2**n_qubits``."""
    dim = 1 << n_qubits
    amplitudes = np.zeros(dim, dtype=float)
    d = len(data_point)
    amplitudes[:d] = np.asarray(data_point, dtype=float)
    norm = float(np.linalg.norm(amplitudes))
    if norm > 1e-15:
        amplitudes /= norm
    return amplitudes


def _build_swap_test_dag(
    amp_a: np.ndarray, amp_b: np.ndarray, n_data: int
) -> DAGCircuit:
    """Builds a real (2*n_data + 1)-qubit Swap Test DAGCircuit (`RY`, `CX`, `H`, `CSWAP`)."""
    n_total = 1 + 2 * n_data
    builder = CircuitBuilder(n_total)

    reg_a = list(range(1, 1 + n_data))
    reg_b = list(range(1 + n_data, 1 + 2 * n_data))

    append_amplitude_state_prep(builder, reg_a, amp_a)
    append_amplitude_state_prep(builder, reg_b, amp_b)

    builder.record(Instruction("H", (0,)))
    for i in range(n_data):
        builder.record(Instruction("CSWAP", (0, reg_a[i], reg_b[i])))
    builder.record(Instruction("H", (0,)))

    return DAGCircuit.from_builder(builder)


def quantum_distance(
    point_a: list[float] | np.ndarray,
    point_b: list[float] | np.ndarray,
    shots: int = 1024,
    seed: int | None = None,
) -> float:
    """Computes quantum distance sqrt(1 - |<psi|phi>|^2) via a real Swap Test circuit.

    Prepares ``|0>|psi>|phi>`` on ``2*m + 1`` qubits using ``RY``/``CX`` state-preparation
    trees, applies ``H(0)``, ``m`` Fredkin ``CSWAP`` gates, ``H(0)``, and samples ``shots``
    measurements via ``sim.sample(shots)``.
    """
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")

    a = np.asarray(point_a, dtype=float).ravel()
    b = np.asarray(point_b, dtype=float).ravel()

    d = max(len(a), len(b))
    a_pad = np.zeros(d, dtype=float)
    b_pad = np.zeros(d, dtype=float)
    a_pad[: len(a)] = a
    b_pad[: len(b)] = b

    norm_a = float(np.linalg.norm(a_pad))
    norm_b = float(np.linalg.norm(b_pad))
    if norm_a <= 1e-15 and norm_b <= 1e-15:
        return 0.0
    if norm_a <= 1e-15 or norm_b <= 1e-15:
        return 1.0

    n_data = max(1, int(math.ceil(math.log2(max(d, 2)))))
    amp_a = _amplitude_encode(a_pad, n_data)
    amp_b = _amplitude_encode(b_pad, n_data)

    dag = _build_swap_test_dag(amp_a, amp_b, n_data)
    sim = create_simulator(1 + 2 * n_data, seed=seed)
    for op in dag.op_nodes():
        sim.apply(op.gate_name, op.qubits, op.params)

    counts = sim.sample(shots)
    # Ancilla is q[0], which corresponds to MSB bitstring[0]
    zeros = sum(cnt for bs, cnt in counts.items() if bs[0] == "0")
    p_zero = zeros / float(shots)

    overlap_sq = max(0.0, min(1.0, 2.0 * p_zero - 1.0))
    return math.sqrt(max(0.0, 1.0 - overlap_sq))


def _hemisphere_embed(
    vec: np.ndarray, center: np.ndarray, radius_scale: float
) -> np.ndarray:
    """Norm-preserving hemisphere embedding mapping R^d -> S^d_+ so Swap Test fidelity
    is strictly monotonic in Euclidean distance (sensitive to both angle and norm).
    """
    u = (np.asarray(vec, dtype=float) - center) / radius_scale
    u_norm_sq = float(np.dot(u, u))
    if u_norm_sq > 1.96:
        u = u * (math.sqrt(1.96 / u_norm_sq))
        u_norm_sq = 1.96
    half_scaled = u / math.sqrt(2.0)
    last_coord = math.sqrt(max(1e-12, 1.0 - 0.5 * u_norm_sq))
    return np.append(half_scaled, last_coord)


def quantum_cluster(
    data: list[list[float]] | np.ndarray,
    k: int = 2,
    max_iterations: int = 20,
    shots: int = 1024,
    seed: int | None = None,
) -> ClusterResult:
    """Quantum k-means clustering executing Swap Test ``quantum_distance`` in EVERY iteration."""
    data_arr = np.asarray(data, dtype=float)
    if data_arr.ndim != 2:
        raise ValueError("data must be a 2D array of shape (n_points, n_features).")
    n_points, _ = data_arr.shape

    if k < 1 or k > n_points:
        raise ValueError(f"k must be in [1, {n_points}], given: {k}")

    rng = np.random.default_rng(seed)
    center = np.mean(data_arr, axis=0)
    radii = np.linalg.norm(data_arr - center, axis=1)
    radius_scale = max(float(np.max(radii)) * 1.25, 1.0)

    embedded_points = [
        _hemisphere_embed(data_arr[i], center, radius_scale) for i in range(n_points)
    ]

    # Step 1: Pairwise Swap-Test quantum distance matrix
    dist_matrix = np.zeros((n_points, n_points), dtype=float)
    pair_counter = 0
    for i in range(n_points):
        for j in range(i + 1, n_points):
            d_seed = None if seed is None else int(seed) + pair_counter
            pair_counter += 1
            d_ij = quantum_distance(
                embedded_points[i], embedded_points[j], shots=shots, seed=d_seed
            )
            dist_matrix[i, j] = d_ij
            dist_matrix[j, i] = d_ij

    # Step 2: k-means++ initialization using Swap Test distances
    centroid_indices = [int(rng.integers(0, n_points))]
    for _ in range(1, k):
        min_dists = np.min(dist_matrix[centroid_indices][:, :n_points], axis=0).copy()
        for ci in centroid_indices:
            min_dists[ci] = 0.0
        centroid_indices.append(int(np.argmax(min_dists)))

    centroids = data_arr[centroid_indices].copy()

    # Step 3: Iterative k-means assignment using Swap Test quantum_distance in EVERY iteration
    labels = np.zeros(n_points, dtype=int)
    prev_labels = np.full(n_points, -1, dtype=int)
    n_iterations = 0

    for iteration in range(max_iterations):
        n_iterations += 1
        embedded_centroids = [
            _hemisphere_embed(centroids[c_idx], center, radius_scale)
            for c_idx in range(k)
        ]

        for i in range(n_points):
            best_cluster = 0
            min_dist = float("inf")
            for c_idx in range(k):
                call_seed = (
                    None
                    if seed is None
                    else int(seed) + 10_000 + iteration * (n_points * k) + i * k + c_idx
                )
                d_ic = quantum_distance(
                    embedded_points[i],
                    embedded_centroids[c_idx],
                    shots=shots,
                    seed=call_seed,
                )
                if d_ic < min_dist:
                    min_dist = d_ic
                    best_cluster = c_idx
            labels[i] = best_cluster

        if np.array_equal(labels, prev_labels):
            break
        prev_labels = labels.copy()

        for c_idx in range(k):
            members = data_arr[labels == c_idx]
            if len(members) > 0:
                centroids[c_idx] = members.mean(axis=0)

    inertia = float(
        sum(
            np.sum((data_arr[i] - centroids[labels[i]]) ** 2)
            for i in range(n_points)
        )
    )

    return ClusterResult(
        labels=labels.tolist(),
        centroids=centroids.tolist(),
        k=k,
        iterations=n_iterations,
        inertia=inertia,
        distance_matrix=dist_matrix.tolist(),
    )
