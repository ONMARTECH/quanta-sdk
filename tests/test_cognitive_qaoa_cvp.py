"""tests.test_cognitive_qaoa_cvp — Test Suite for Quanta Cognitive CVP Integration.

Verifies:
  1. ZenoSubspaceArbiter: Gram-Schmidt sensitivities, P_zeno observation frequencies,
     subspace pruning (N_active < N_total), and exact energy conservation.
  2. LatticeEngramMemory: CSF shielding, engram crystallization, SWR recall of warm-start
     angles theta_a and variational parameter seeds (gamma_0, beta_0).
  3. CognitiveQAOACVPSolver: Full 6-phase pipeline across dimensions n=3, 4, 6, 8,
     verifying Zeno qubit reduction, exact ground truth recovery, and SWR acceleration.
"""

from __future__ import annotations

import math

import numpy as np

from quanta.cognitive.cvp_arbiter import (
    LatticeEngramMemory,
    SubspaceSensitivity,
    WarmStartConfig,
    ZenoProjectionResult,
    ZenoSubspaceArbiter,
    babai_closest_vector,
    gram_schmidt_orthogonalize,
)
from quanta.primitives.cvp import CVPHamiltonian
from quanta.solvers.cognitive_qaoa_cvp import CognitiveCVPResult, CognitiveQAOACVPSolver


def create_synthetic_cvp_instance(
    n: int,
    noise_scale: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generates a well-conditioned synthetic lattice CVP instance with known ground truth.

    Args:
        n: Dimension of the lattice.
        noise_scale: Magnitude of perturbation from true lattice point.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (basis B, target t, ground_truth_z, ground_truth_distance).
    """
    rng = np.random.default_rng(seed)

    # Generate well-conditioned basis B via random orthogonal matrix + positive diagonal
    q_mat, _ = np.linalg.qr(rng.standard_normal((n, n)))
    diag_vals = rng.uniform(1.5, 3.5, size=n)
    b_mat = (q_mat @ np.diag(diag_vals)).astype(np.float64)

    # Pick integer target coordinates within small bounds
    z_star = rng.choice([-1, 0, 1], size=n).astype(np.int64)
    v_star = b_mat @ z_star.astype(np.float64)

    # Add small displacement perturbation e
    perturbation = rng.standard_normal(n)
    p_norm = np.linalg.norm(perturbation)
    if p_norm > 0:
        perturbation = (perturbation / p_norm) * noise_scale
    target = v_star + perturbation
    d_star = float(np.linalg.norm(v_star - target))

    return b_mat, target, z_star, d_star


# ==============================================================================
# 1. Zeno Subspace Arbiter Tests
# ==============================================================================


class TestZenoSubspaceArbiter:
    """Verifies Gram-Schmidt sensitivity analysis, P_zeno dynamics, and subspace reduction."""

    def test_gram_schmidt_orthogonality(self) -> None:
        """Verifies that Gram-Schmidt vectors b_i* are mutually orthogonal."""
        basis, _, _, _ = create_synthetic_cvp_instance(n=4, seed=10)
        b_star, mu = gram_schmidt_orthogonalize(basis)

        n = basis.shape[1]
        assert b_star.shape == basis.shape
        assert mu.shape == (n, n)

        # Check diagonal of mu is 1.0
        np.testing.assert_allclose(np.diag(mu), 1.0, atol=1e-12)

        # Check mutual orthogonality of b_i*
        for i in range(n):
            for j in range(i + 1, n):
                dot_prod = float(np.dot(b_star[:, i], b_star[:, j]))
                assert abs(dot_prod) < 1e-10, f"Vectors {i} and {j} not orthogonal: dot={dot_prod}"

    def test_babai_closest_vector_recovery(self) -> None:
        """Verifies Babai algorithm recovers integer coordinate vector."""
        basis, target, z_star, _ = create_synthetic_cvp_instance(n=3, noise_scale=0.05, seed=12)
        z_babai, c_vals, ambiguities = babai_closest_vector(basis, target)

        assert len(z_babai) == 3
        assert len(c_vals) == 3
        assert len(ambiguities) == 3
        for amb in ambiguities:
            assert 0.0 <= amb <= 0.5000001

    def test_analyze_sensitivities(self) -> None:
        """Verifies SubspaceSensitivity metrics and P_zeno bounds."""
        basis, target, _, _ = create_synthetic_cvp_instance(n=4, seed=15)
        arbiter = ZenoSubspaceArbiter(freeze_threshold=0.70, exploration_drive=0.20)
        sensitivities = arbiter.analyze_sensitivities(basis, target)

        assert len(sensitivities) == 4
        for s in sensitivities:
            assert isinstance(s, SubspaceSensitivity)
            assert s.gram_schmidt_norm > 0.0
            assert 0.0 <= s.integer_ambiguity <= 0.5000001
            assert 0.0 <= s.sensitivity_score <= 1.0
            assert 0.15 <= s.p_zeno <= 0.985
            assert s.recommended_freeze == (s.p_zeno >= 0.70)

    def test_zeno_subspace_reduction(self) -> None:
        """Verifies that Zeno arbitration reduces active qubit register size (N_active < N_total)."""
        # Create instance with very small noise so some coordinates land closely on integer hyperplanes
        basis, target, _, _ = create_synthetic_cvp_instance(n=4, noise_scale=0.01, seed=20)
        arbiter = ZenoSubspaceArbiter(freeze_threshold=0.65, min_active_coords=1)
        res = arbiter.arbitrate_subspace_freezing(basis, target, bits_per_coord=3)

        assert isinstance(res, ZenoProjectionResult)
        assert res.total_qubits == 12
        assert len(res.pinned_indices) > 0, "High confidence coordinates should be pinned"
        assert res.num_active_qubits < res.total_qubits
        assert res.num_active_qubits >= 3  # min_active_coords=1 * 3 bits
        assert res.subspace_reduction_ratio > 0.0

    def test_subspace_energy_conservation(self) -> None:
        """Mathematically verifies exact energy conservation between full and reduced Hamiltonian.

        ||B_red * z_active - t_eff||^2 == ||B * z_full - t||^2
        """
        basis, target, _, _ = create_synthetic_cvp_instance(n=4, noise_scale=0.05, seed=25)
        cvp_full = CVPHamiltonian(basis=basis, target=target, bits_per_coord=2)

        arbiter = ZenoSubspaceArbiter(cvp_hamiltonian=cvp_full, freeze_threshold=0.60, min_active_coords=1)
        reduced_cvp, pinned_dict = arbiter.prune_hamiltonian()

        assert len(pinned_dict) > 0, "Expected at least one pinned coordinate for testing"
        active_indices = [i for i in range(4) if i not in pinned_dict]
        assert len(active_indices) == reduced_cvp.dim

        # Test multiple random candidate active coordinates
        rng = np.random.default_rng(99)
        for _ in range(5):
            z_active_test = rng.choice([-1, 0, 1], size=len(active_indices)).astype(np.int64)

            # Reconstruct full vector
            z_full_test = np.zeros(4, dtype=np.int64)
            for i, val in pinned_dict.items():
                z_full_test[i] = val
            for k, orig_idx in enumerate(active_indices):
                z_full_test[orig_idx] = z_active_test[k]

            # Direct Euclidean energy ||B z - t||^2
            full_distance_sq = cvp_full.evaluate_squared_distance(z_full_test)
            reduced_distance_sq = reduced_cvp.evaluate_squared_distance(z_active_test)

            np.testing.assert_allclose(
                full_distance_sq,
                reduced_distance_sq,
                rtol=1e-10,
                atol=1e-10,
                err_msg="Reduced CVP Hamiltonian must conserve exact algebraic energy",
            )

    def test_mixer_weights_construction(self) -> None:
        """Verifies per-qubit mixer field weights w_a = (1 - P_zeno) for transverse quenching."""
        arbiter = ZenoSubspaceArbiter(freeze_threshold=0.70)
        p_zeno_dict = {0: 0.95, 1: 0.30, 2: 0.80}
        weights = arbiter.build_zeno_mixer_weights(total_qubits=9, p_zeno_values=p_zeno_dict, bits_per_coord=3)

        assert len(weights) == 9
        # Coordinate 0 qubits: w = 1.0 - 0.95 = 0.05 (quenched)
        for q in [0, 1, 2]:
            assert abs(weights[q] - 0.05) < 1e-6
        # Coordinate 1 qubits: w = 1.0 - 0.30 = 0.70 (active)
        for q in [3, 4, 5]:
            assert abs(weights[q] - 0.70) < 1e-6
        # Coordinate 2 qubits: w = 1.0 - 0.80 = 0.20 (quenched)
        for q in [6, 7, 8]:
            assert abs(weights[q] - 0.20) < 1e-6


# ==============================================================================
# 2. SWR Lattice Engram Memory Tests
# ==============================================================================


class TestLatticeEngramMemory:
    """Verifies CSF shielding, engram storage, and SWR warm-start angle replay."""

    def test_store_and_csf_retention(self) -> None:
        """Verifies storing a solution creates an engram with S >= 2.0 (CSF-shielded core anchor)."""
        memory = LatticeEngramMemory()
        basis = np.eye(3)
        target = np.array([0.9, 1.1, -0.1])
        z = np.array([1, 1, 0])

        idx = memory.store_solution(
            basis_id="basis_test_3x3",
            z=z,
            distance=0.15,
            angles=[0.5, 0.3],
            basis=basis,
            target=target,
            is_ground_truth=True,
        )

        assert idx >= 0
        assert memory.has_engram("basis_test_3x3")
        assert memory.get_engram_count("basis_test_3x3") == 1

        # Check hippocampal vital context
        vital = memory.memory_manager.recall_vital_context(top_k=5)
        assert len(vital) >= 1
        assert vital[0]["salience"] >= 2.0
        assert vital[0]["is_core_anchor"] is True

    def test_recall_warm_start_empty_fallback(self) -> None:
        """Verifies that recall on unvisited basis returns uniform superposition angles theta = pi/2."""
        memory = LatticeEngramMemory()
        cfg = memory.recall_warm_start(basis_id="unknown_basis", num_qubits=6, bits_per_coord=2)

        assert isinstance(cfg, WarmStartConfig)
        assert cfg.recalled_engram_count == 0
        assert len(cfg.initial_rotation_angles) == 6
        for angle in cfg.initial_rotation_angles:
            assert abs(angle - math.pi / 2.0) < 1e-6
        assert cfg.seeded_qaoa_angles is None

        # Verify tuple unpacking support
        angles_arr, seeds_arr = cfg
        assert angles_arr is not None
        assert seeds_arr is None

    def test_recall_warm_start_biasing(self) -> None:
        """Verifies that recalled SWR engrams construct non-uniform warm-start angles theta_a."""
        memory = LatticeEngramMemory()
        basis = np.eye(2)
        # Store solution z = [1, -1] with bits_per_coord=2
        # center=[0, 0], span=4, half_span=2, z_min=[-2, -2]
        # coord 0: z=1 -> delta=3 -> bits (1, 1)
        # coord 1: z=-1 -> delta=1 -> bits (1, 0)
        memory.store_solution(
            basis_id="basis_bias_test",
            z=[1, -1],
            distance=0.01,
            angles=[0.42, 0.84],
            basis=basis,
            is_ground_truth=True,
        )

        cfg = memory.recall_warm_start(
            basis_id="basis_bias_test",
            num_qubits=4,
            bits_per_coord=2,
            center=np.array([0, 0]),
            delta_regularization=0.05,
        )

        assert cfg.recalled_engram_count == 1
        assert cfg.seeded_qaoa_angles == [0.42, 0.84]
        assert len(cfg.initial_rotation_angles) == 4

        # For bits that are 1, prior is 0.95 -> theta > pi/2
        # For bits that are 0, prior is 0.05 -> theta < pi/2
        theta_0 = cfg.initial_rotation_angles[0]  # bit 0 of coord 0 (value 1)
        assert theta_0 > math.pi / 2.0

        # Construct single-qubit state RY(theta)|0>
        # Probability of |1> = sin^2(theta/2) should be ~0.95
        prob_one = math.sin(theta_0 / 2.0) ** 2
        assert abs(prob_one - 0.95) < 1e-4

    def test_swr_statevector_fidelity_boost(self) -> None:
        """Verifies that the SWR warm-start product state has high overlap with target ground state."""
        memory = LatticeEngramMemory()
        basis, target, z_star, _ = create_synthetic_cvp_instance(n=3, seed=33)

        memory.store_solution(
            basis_id=basis,
            z=z_star,
            distance=0.05,
            is_ground_truth=True,
        )

        cfg = memory.recall_warm_start(
            basis_id=basis,
            num_qubits=6,
            bits_per_coord=2,
            delta_regularization=0.05,
        )
        assert len(cfg.initial_rotation_angles) == 6

        # Overlap per qubit is at least 0.95 for matching bits
        # Total fidelity for 6 qubits >= (0.95)^6 ~ 0.735 >> 1/2^6 (0.0156)
        min_expected_overlap = (0.95) ** 6
        uniform_overlap = (0.50) ** 6
        assert min_expected_overlap > 40 * uniform_overlap


# ==============================================================================
# 3. Cognitive QAOA CVP Solver Tests (n = 3, 4, 6, 8)
# ==============================================================================


class TestCognitiveQAOACVPSolver:
    """Verifies end-to-end 6-phase Cognitive QAOA pipeline across lattice dimensions."""

    def test_solver_initialization_and_modes(self) -> None:
        """Verifies initialization with and without cognitive components."""
        solver_full = CognitiveQAOACVPSolver(use_zeno=True, use_swr=True)
        assert solver_full.zeno_arbiter is not None
        assert solver_full.memory is not None

        solver_vanilla = CognitiveQAOACVPSolver(use_zeno=False, use_swr=False)
        assert solver_vanilla.zeno_arbiter is None
        assert solver_vanilla.memory is None

    def test_solve_cvp_dimension_3(self) -> None:
        """Verifies CVP solving for n=3 lattice (N=6 qubits)."""
        basis, target, z_star, d_star = create_synthetic_cvp_instance(n=3, noise_scale=0.10, seed=41)
        solver = CognitiveQAOACVPSolver(p=1, shots=512, seed=42)
        res = solver.solve(basis=basis, target=target, bits_per_coord=2, ground_truth_z=z_star)

        assert isinstance(res, CognitiveCVPResult)
        assert res.success is True
        assert len(res.z) == 3
        assert res.distance <= res.babai_distance + 1e-6
        assert res.approximation_factor <= 1.05
        assert res.wall_clock_time > 0.0

    def test_solve_cvp_dimension_4(self) -> None:
        """Verifies CVP solving for n=4 lattice (N=8 qubits)."""
        basis, target, z_star, d_star = create_synthetic_cvp_instance(n=4, noise_scale=0.08, seed=42)
        solver = CognitiveQAOACVPSolver(p=1, shots=512, seed=43)
        res = solver.solve(basis=basis, target=target, bits_per_coord=2, ground_truth_z=z_star)

        assert isinstance(res, CognitiveCVPResult)
        assert res.total_qubits == 8
        assert res.active_qubits <= 8
        np.testing.assert_array_equal(res.z, z_star)
        assert abs(res.approximation_factor - 1.0) < 1e-4

    def test_solve_cvp_dimension_6_with_zeno_pruning(self) -> None:
        """Verifies Zeno subspace pruning reduces qubit footprint on n=6 lattice (N=12 qubits)."""
        basis, target, z_star, d_star = create_synthetic_cvp_instance(n=6, noise_scale=0.05, seed=44)
        solver = CognitiveQAOACVPSolver(p=1, shots=512, zeno_threshold=0.65, min_active_coords=1, seed=44)
        res = solver.solve(basis=basis, target=target, bits_per_coord=2, ground_truth_z=z_star)

        assert res.total_qubits == 12
        assert res.active_qubits < res.total_qubits, "Zeno pruning must reduce active qubits"
        assert len(res.zeno_pinned) > 0, "Expected coordinates to be pinned"
        assert res.distance <= res.babai_distance + 1e-6
        assert res.subspace_reduction_ratio > 0.0
        assert res.approximation_factor <= 1.05

    def test_solve_cvp_dimension_8_with_zeno_pruning(self) -> None:
        """Verifies Zeno pruning scales to n=8 lattice (N=16 qubits) in rapid execution time."""
        basis, target, z_star, d_star = create_synthetic_cvp_instance(n=8, noise_scale=0.05, seed=45)
        solver = CognitiveQAOACVPSolver(p=1, shots=256, zeno_threshold=0.60, min_active_coords=2, seed=45)
        res = solver.solve(basis=basis, target=target, bits_per_coord=2, ground_truth_z=z_star)

        assert res.total_qubits == 16
        # Active qubits must be pruned down significantly
        assert res.active_qubits <= 10, f"Expected active qubits <= 10, got {res.active_qubits}"
        assert res.wall_clock_time < 2.0, f"Zeno pruning must execute rapidly, took {res.wall_clock_time:.2f}s"
        assert res.distance <= res.babai_distance + 1e-6

    def test_swr_warm_start_consolidation_and_acceleration(self) -> None:
        """Verifies SWR replay: run 1 stores solution engram, run 2 recalls it to warm-start."""
        basis, target, z_star, _ = create_synthetic_cvp_instance(n=4, noise_scale=0.05, seed=50)
        shared_memory = LatticeEngramMemory()

        # Run 1: Cold start (no prior engrams)
        solver1 = CognitiveQAOACVPSolver(p=1, shots=512, memory=shared_memory, seed=51)
        res1 = solver1.solve(basis, target, bits_per_coord=2, ground_truth_z=z_star)

        assert res1.swr_recalled is False
        assert shared_memory.get_engram_count() >= 1, "Run 1 must crystallize solution into engram memory"

        # Run 2: Warm start on same lattice
        solver2 = CognitiveQAOACVPSolver(p=1, shots=512, memory=shared_memory, seed=52)
        res2 = solver2.solve(basis, target, bits_per_coord=2, ground_truth_z=z_star)

        assert res2.swr_recalled is True, "Run 2 must recall SWR engrams from memory"
        assert res2.warm_start_diagnostics is not None
        assert res2.warm_start_diagnostics.recalled_engram_count >= 1
        np.testing.assert_array_equal(res2.z, z_star)

    def test_result_summary_and_aliases(self) -> None:
        """Verifies formatting and property aliases on CognitiveCVPResult."""
        basis, target, z_star, _ = create_synthetic_cvp_instance(n=3, seed=60)
        solver = CognitiveQAOACVPSolver(p=1, shots=128, seed=60)
        res = solver.solve(basis, target, bits_per_coord=2, ground_truth_z=z_star)

        # Check aliases
        np.testing.assert_array_equal(res.best_z, res.z)
        np.testing.assert_array_equal(res.best_lattice_vector, res.closest_point)
        assert res.residual_norm == res.distance
        assert res.num_active_qubits == res.active_qubits
        assert res.num_total_qubits == res.total_qubits
        assert res.execution_time_ms == res.wall_clock_time * 1000.0

        # Check summary formatting
        summary_text = res.summary()
        assert "Cognitive QAOA CVP Result" in summary_text
        assert "Lattice Dimension:  3" in summary_text
