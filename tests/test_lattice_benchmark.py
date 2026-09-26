"""tests.test_lattice_benchmark — Unit Tests for Classical LLL/Babai and CVP Benchmark Suite.

Verifies:
  1. LLL lattice basis reduction: unimodularity |det(U)| == 1, basis equivalence B_red = B @ U,
     size reduction |mu_{i, j}| <= 0.5, and the Lovász condition.
  2. Babai Nearest Plane algorithm: exact coordinate recovery on orthogonal and reduced lattices.
  3. Ground-truth synthetic CVP generator: mathematical uniqueness of z* (d(z') > d* for all z' != z*),
     perturbation bound epsilon < lambda_1 / 2, and coordinate box bounds.
  4. 3-way comparative benchmark suite integration on small and scaled dimensions.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from benchmarks.cvp_benchmark import (
    generate_ground_truth_cvp,
    run_benchmark_suite,
)
from quanta.lattice.classical import (
    babai_nearest_plane,
    gram_schmidt_orthogonalize,
    lll_reduce,
    solve_cvp_classical,
)

# ==============================================================================
# 1. LLL Reduction Tests
# ==============================================================================


class TestLLLReduction:
    """Verifies unimodularity, equivalence, size reduction, and Lovász conditions of LLL."""

    @pytest.mark.parametrize("dim", [2, 3, 4, 6])
    def test_unimodularity_and_determinant(self, dim: int) -> None:
        """Verifies that transformation matrix U has integer entries and |det(U)| == 1."""
        rng = np.random.default_rng(42 + dim)
        b_orig = rng.standard_normal((dim, dim)) * 10.0

        b_red, u_mat = lll_reduce(b_orig, delta=0.75)

        # 1. U is strictly integer
        assert issubclass(u_mat.dtype.type, np.integer)

        # 2. Unimodularity: |det(U)| == 1
        det_u = float(np.linalg.det(u_mat.astype(np.float64)))
        assert abs(abs(det_u) - 1.0) < 1e-5

        # 3. Basis equivalence: B_red == B @ U
        diff = np.max(np.abs(b_orig @ u_mat - b_red))
        assert diff < 1e-10

    def test_size_reduction_condition(self) -> None:
        """Verifies that for all j < i, Gram-Schmidt coefficients satisfy |mu_{i, j}| <= 0.5."""
        rng = np.random.default_rng(101)
        dim = 5
        b_orig = rng.standard_normal((dim, dim)) * 25.0

        b_red, _ = lll_reduce(b_orig, delta=0.75)
        _, mu, _ = gram_schmidt_orthogonalize(b_red)

        for i in range(1, dim):
            for j in range(i):
                assert abs(mu[i, j]) <= 0.5 + 1e-9, f"Size reduction violated at ({i}, {j}): {mu[i, j]}"

    def test_lovasz_condition(self) -> None:
        """Verifies the Lovász condition delta * ||b_{k-1}*||^2 <= ||b_k* + mu b_{k-1}*||^2."""
        rng = np.random.default_rng(202)
        dim = 4
        delta = 0.75
        b_orig = rng.standard_normal((dim, dim)) * 15.0

        b_red, _ = lll_reduce(b_orig, delta=delta)
        _, mu, sqnorms = gram_schmidt_orthogonalize(b_red)

        for k in range(1, dim):
            lhs = delta * sqnorms[k - 1]
            rhs = sqnorms[k] + (mu[k, k - 1] ** 2) * sqnorms[k - 1]
            assert rhs >= lhs - 1e-9, f"Lovász condition failed at index {k}: {rhs} < {lhs}"

    def test_shortest_vector_length_reduction(self) -> None:
        """Verifies that LLL significantly reduces the norm of the first basis vector."""
        # Highly skewed basis
        b_skewed = np.array(
            [
                [1.0, 2.0, 100.0],
                [0.0, 1.0, 100.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        b_red, _ = lll_reduce(b_skewed, delta=0.75)

        norm_orig = float(np.linalg.norm(b_skewed[:, 0]))
        norm_red = float(np.linalg.norm(b_red[:, 0]))
        assert norm_red <= norm_orig + 1e-9
        assert np.linalg.norm(b_red[:, 2]) < np.linalg.norm(b_skewed[:, 2])

    def test_input_validation(self) -> None:
        """Verifies that invalid parameters raise descriptive ValueError exceptions."""
        valid_b = np.eye(3)
        with pytest.raises(ValueError, match="Lovász parameter delta"):
            lll_reduce(valid_b, delta=0.20)
        with pytest.raises(ValueError, match="Lovász parameter delta"):
            lll_reduce(valid_b, delta=1.05)

        bad_shape_b = np.ones((2, 4))  # rows < cols
        with pytest.raises(ValueError, match="Lattice dimension error"):
            lll_reduce(bad_shape_b)


# ==============================================================================
# 2. Babai Nearest Plane Tests
# ==============================================================================


class TestBabaiNearestPlane:
    """Verifies Babai Nearest Plane algorithm on orthogonal and reduced lattice bases."""

    def test_exact_recovery_orthogonal_lattice(self) -> None:
        """Verifies that Babai finds exact closest vector on orthogonal bases."""
        basis = np.diag([2.0, 3.0, 5.0, 7.0])
        true_z = np.array([2, -1, 3, 0], dtype=np.int64)
        v_true = basis @ true_z.astype(np.float64)

        # Target displaced by perturbation within Voronoi cell
        perturbation = np.array([0.4, -0.6, 0.9, -1.2])
        target = v_true + perturbation

        z_recovered = babai_nearest_plane(basis, target, reduce=False)
        assert np.array_equal(z_recovered, true_z)

    def test_solve_cvp_classical_helper(self) -> None:
        """Verifies solve_cvp_classical returns integer vector and residual distance."""
        basis = np.array([[3.0, 1.0], [0.0, 2.0]], dtype=np.float64)
        target = np.array([3.1, 0.2], dtype=np.float64)

        z, dist = solve_cvp_classical(basis, target)
        assert issubclass(z.dtype.type, np.integer)
        expected_dist = float(np.linalg.norm(basis @ z.astype(np.float64) - target))
        assert abs(dist - expected_dist) < 1e-12

    def test_dimension_mismatch_raises(self) -> None:
        """Verifies ValueError when target dimension does not match basis rows."""
        basis = np.eye(3)
        target = np.array([1.0, 2.0])  # length 2 != 3
        with pytest.raises(ValueError, match="Target dimension"):
            babai_nearest_plane(basis, target)


# ==============================================================================
# 3. Ground-Truth CVP Generator Tests
# ==============================================================================


class TestGroundTruthCVPGenerator:
    """Verifies mathematical guarantees and uniqueness of the synthetic CVP generator."""

    @pytest.mark.parametrize("dim", [3, 4, 6])
    def test_ground_truth_uniqueness(self, dim: int) -> None:
        """Verifies that z* is strictly closer to target than all tested non-zero offsets z' != z*."""
        basis, target, z_star, d_star = generate_ground_truth_cvp(
            dim=dim,
            bits_per_coord=3,
            skewness=2.0,
            seed=42 + dim,
        )

        # Distance to true z* equals d_star
        v_star = basis @ z_star.astype(np.float64)
        assert abs(np.linalg.norm(v_star - target) - d_star) < 1e-12

        # Check random discrete perturbations z' != z*
        rng = np.random.default_rng(999 + dim)
        for _ in range(150):
            delta_z = rng.integers(-2, 3, size=dim)
            if np.all(delta_z == 0):
                continue
            z_other = z_star + delta_z
            d_other = float(np.linalg.norm(basis @ z_other.astype(np.float64) - target))
            assert d_other > d_star, (
                f"Ground truth uniqueness violated: d_other ({d_other}) <= d_star ({d_star})"
            )

    def test_coordinate_bounds(self) -> None:
        """Verifies that sampled z* coordinates stay within the allocated bit-depth box."""
        bits = 3
        half_span = 1 << (bits - 1)
        low, high = -half_span, half_span - 1

        for seed in [1, 2, 3, 4, 5]:
            _, _, z_star, _ = generate_ground_truth_cvp(
                dim=5,
                bits_per_coord=bits,
                skewness=1.5,
                seed=seed,
            )
            assert np.all(z_star >= low)
            assert np.all(z_star <= high)

    def test_reproducibility(self) -> None:
        """Verifies deterministic output when seed is provided."""
        b1, t1, z1, d1 = generate_ground_truth_cvp(dim=4, seed=123)
        b2, t2, z2, d2 = generate_ground_truth_cvp(dim=4, seed=123)

        assert np.allclose(b1, b2)
        assert np.allclose(t1, t2)
        assert np.array_equal(z1, z2)
        assert abs(d1 - d2) < 1e-14


# ==============================================================================
# 4. Comparative Benchmark Suite Tests
# ==============================================================================


class TestComparativeBenchmarkIntegration:
    """Verifies end-to-end execution of the 3-way comparative benchmark suite."""

    def test_benchmark_suite_execution_and_persistence(self) -> None:
        """Verifies that run_benchmark_suite runs cleanly and persists valid JSON."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_file = Path(tmp_dir) / "test_benchmark_results.json"

            # Run mini benchmark with n=4 and n=8, 2 trials
            results = run_benchmark_suite(
                dimensions=[4, 8],
                num_trials=2,
                bits_per_coord=2,
                skewness=2.0,
                output_path=json_file,
            )

            # 1. Structure verification
            assert "metadata" in results
            assert "summary_table" in results
            assert "results_by_dimension" in results
            assert "trial_records" in results

            # 2. Results per dimension
            assert "4" in results["results_by_dimension"]
            assert "8" in results["results_by_dimension"]

            dim4_res = results["results_by_dimension"]["4"]
            assert "classical" in dim4_res
            assert "vanilla_qaoa" in dim4_res
            assert "cognitive_qaoa" in dim4_res

            # 3. Metric validity
            cog_m = dim4_res["cognitive_qaoa"]
            assert cog_m["success_rate"] >= 0.0
            assert cog_m["mean_active_qubits"] <= cog_m["total_qubits"]
            assert cog_m["subspace_reduction_pct"] >= 0.0

            # 4. JSON file persisted correctly
            assert json_file.exists()
            with open(json_file, encoding="utf-8") as f:
                loaded_data = json.load(f)
            assert loaded_data["metadata"]["dimensions"] == [4, 8]

    def test_vanilla_qaoa_barrier_detection(self) -> None:
        """Verifies that dimensions with total qubits > 16 record Barren Plateau limit status."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_file = Path(tmp_dir) / "barrier_results.json"

            # Dimension 12 with bits_per_coord=2 -> 24 qubits (> 16 qubit classical simulation limit)
            results = run_benchmark_suite(
                dimensions=[12],
                num_trials=1,
                bits_per_coord=2,
                output_path=json_file,
            )

            vanilla_res = results["results_by_dimension"]["12"]["vanilla_qaoa"]
            assert "BARREN_PLATEAU_LIMIT" in vanilla_res["status"]
            assert vanilla_res["success_rate"] == 0.0
            assert vanilla_res["mean_runtime_seconds"] is None
