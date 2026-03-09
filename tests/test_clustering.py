"""Tests for quanta.layer3.clustering — Quantum Clustering."""

from __future__ import annotations

import pytest

from quanta.layer3.clustering import (
    ClusterResult,
    quantum_cluster,
    quantum_distance,
)


class TestQuantumDistance:
    """Tests for the quantum_distance function."""

    def test_identical_points(self) -> None:
        d = quantum_distance([1, 0], [1, 0], seed=42)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_orthogonal_points(self) -> None:
        d = quantum_distance([1, 0], [0, 1], seed=42)
        assert d > 0

    def test_symmetry(self) -> None:
        a, b = [1, 2], [3, 4]
        d1 = quantum_distance(a, b, seed=42)
        d2 = quantum_distance(b, a, seed=42)
        assert d1 == pytest.approx(d2, abs=0.01)

    def test_non_negative(self) -> None:
        d = quantum_distance([1, 2, 3], [4, 5, 6], seed=42)
        assert d >= 0

    def test_range_zero_one(self) -> None:
        d = quantum_distance([1, 0], [0, 1], seed=42)
        assert 0 <= d <= 1.01  # Small tolerance


class TestClusterResult:
    """Tests for the ClusterResult dataclass."""

    def test_repr(self) -> None:
        r = ClusterResult(
            labels=[0, 0, 1, 1],
            centroids=[[1, 2], [5, 6]],
            k=2,
            iterations=3,
            inertia=5.0,
        )
        assert "k=2" in repr(r)
        assert "inertia=5.0" in repr(r)

    def test_summary(self) -> None:
        r = ClusterResult(
            labels=[0, 0, 1, 1],
            centroids=[[1, 2], [5, 6]],
            k=2,
            iterations=3,
            inertia=5.0,
        )
        s = r.summary()
        assert "Quantum Clustering" in s
        assert "Cluster 0" in s
        assert "Cluster 1" in s


class TestQuantumCluster:
    """Tests for the quantum_cluster function."""

    def test_two_clusters(self) -> None:
        data = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]
        result = quantum_cluster(data, k=2, seed=42)
        assert isinstance(result, ClusterResult)
        assert len(result.labels) == 6
        assert result.k == 2
        assert all(0 <= l < 2 for l in result.labels)
        assert result.iterations >= 1
        assert result.inertia >= 0

    def test_single_cluster(self) -> None:
        data = [[1, 1], [1.1, 0.9], [0.9, 1.1]]
        result = quantum_cluster(data, k=1, seed=42)
        assert all(l == 0 for l in result.labels)

    def test_three_clusters(self) -> None:
        data = [
            [0, 0], [0.1, 0.1],
            [10, 10], [10.1, 9.9],
            [20, 0], [19.9, 0.1],
        ]
        result = quantum_cluster(data, k=3, seed=42)
        assert result.k == 3
        assert len(set(result.labels)) <= 3

    def test_centroids_shape(self) -> None:
        data = [[1, 2], [3, 4], [5, 6], [7, 8]]
        result = quantum_cluster(data, k=2, seed=42)
        assert len(result.centroids) == 2
        assert len(result.centroids[0]) == 2

    def test_distance_matrix(self) -> None:
        data = [[1, 2], [3, 4], [5, 6]]
        result = quantum_cluster(data, k=2, seed=42)
        assert len(result.distance_matrix) == 3
        assert len(result.distance_matrix[0]) == 3

    def test_invalid_k(self) -> None:
        data = [[1, 2], [3, 4]]
        with pytest.raises(ValueError, match="k must be"):
            quantum_cluster(data, k=0, seed=42)

    def test_k_too_large(self) -> None:
        data = [[1, 2], [3, 4]]
        with pytest.raises(ValueError, match="k must be"):
            quantum_cluster(data, k=5, seed=42)

    def test_convergence(self) -> None:
        data = [[0, 0], [100, 100]]
        result = quantum_cluster(data, k=2, seed=42)
        # Should converge quickly with well-separated data
        assert result.iterations <= 5
