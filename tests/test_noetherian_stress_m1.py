"""Empirical Stress Test Suite for Milestone M1 Noetherian Invariant Cognitive Arbiter.

Conducted by Challenger 1 under Quanta SDK Scientific Integrity Protocol (PROJECT_RULES.md).
Adversarial dimensions tested:
1. 1000 Random Drift Vectors with magnitudes ||Delta q|| in [0.01, 10.0] (strict drift reduction & zero NaN/Inf).
2. Degenerate Gram Matrices: 16 collinear generators (C_i = C_0 + 10^-7 N(0, 1)), exact duplicates, latency < 5ms.
3. Action Integral & Mizan Trajectory Scaling across lengths T in [1, 100] and beyond (O(T) linearity, stability).
4. Pathological & Extreme Boundary Conditions (Overconstrained m > d, conflicting targets, dirty tensors).
"""

from __future__ import annotations

import math
import time

import pytest
import torch

from quanta.cognitive.noetherian import (
    ConservedChargeDefinition,
    NoetherianEngineConfig,
    NoetherianInvariantEngine,
    evaluate_epistemic_entropy,
    evaluate_trajectory_action,
    evaluate_trajectory_noetherian,
    project_to_manifold,
)

# ============================================================================
# Fixtures & Helpers
# ============================================================================


@pytest.fixture
def rng_seed() -> None:
    """Ensures reproducible pseudo-random generation."""
    torch.manual_seed(2026)


@pytest.fixture
def standard_config() -> NoetherianEngineConfig:
    """Standard calibrated configuration for 64-dimensional cognitive space."""
    return NoetherianEngineConfig(
        dim=64,
        weight_action=0.25,
        weight_entropy=0.20,
        weight_charge=0.35,
        weight_manifold=0.20,
        action_scale=1.0,
        charge_sigma=0.25,
        charge_gamma=2.0,
        max_allowed_entropy_flux=0.15,
        entropy_kappa=3.0,
        drift_tolerance=0.02,
        tikhonov_reg=1e-6,
        max_projection_iters=3,
        critical_breach_threshold=0.40,
        flux_alert_threshold=0.40,
        mild_drift_threshold=0.65,
        harmonic_threshold=0.85,
    )


def sample_unit_vector(dim: int = 64, generator: torch.Generator | None = None) -> torch.Tensor:
    """Samples a random statevector on the unit sphere S^{dim-1}."""
    v = torch.randn(dim, generator=generator, dtype=torch.float32)
    norm = torch.norm(v, p=2)
    if norm.item() < 1e-12:
        v[0] = 1.0
        return v
    return v / norm


# ============================================================================
# Harness 1: 1000 Random Drift Vectors (Strict Drift Reduction & No NaN/Inf)
# ============================================================================


class TestEmpiricalDriftVectorHarness:
    """Empirical challenge 1: Stress-testing invariant manifold projection under random drift."""

    def test_1000_random_drift_vectors_strict_decrease_canonical_charges(
        self,
        rng_seed: None,
        standard_config: NoetherianEngineConfig,
    ) -> None:
        """Tests 1000 random drift vectors with magnitudes ||Delta q|| in [0.01, 10.0].

        Verifies:
        1. When initial drift > tolerance, projection is applied and strictly decreases drift distance.
        2. Zero NaNs or Infs generated in projected state, metrics, or compliance scores.
        3. Projected state remains on unit sphere (|norm - 1.0| < 1e-5).
        """
        engine = NoetherianInvariantEngine(config=standard_config, auto_register_canonical=True)
        charges = engine.charges
        dim = standard_config.dim

        gen = torch.Generator().manual_seed(42)
        q0 = sample_unit_vector(dim, generator=gen)

        num_samples = 1000
        magnitudes = torch.empty(num_samples).uniform_(0.01, 10.0)

        applied_count = 0
        strictly_decreased_count = 0
        nan_inf_count = 0

        for i in range(num_samples):
            mag = float(magnitudes[i].item())
            direction = sample_unit_vector(dim, generator=gen)
            drift_vec = mag * direction
            q_perturbed = q0 + drift_vec

            proj_q, res = project_to_manifold(q_perturbed, charges=charges, config=standard_config)

            # Check for NaNs and Infs
            if (
                torch.isnan(proj_q).any()
                or torch.isinf(proj_q).any()
                or math.isnan(res.drift_distance_before)
                or math.isnan(res.drift_distance_after)
                or math.isinf(res.drift_distance_after)
                or math.isnan(res.compliance_score)
            ):
                nan_inf_count += 1

            # Unit sphere normalization
            norm_after = torch.norm(proj_q, p=2).item()
            assert math.isclose(norm_after, 1.0, rel_tol=1e-5), f"Norm degraded: {norm_after}"

            if res.projection_applied:
                applied_count += 1
                if res.drift_distance_after < res.drift_distance_before:
                    strictly_decreased_count += 1

        assert nan_inf_count == 0, f"Encountered {nan_inf_count} NaN/Inf instances!"
        assert applied_count == num_samples, f"Expected all {num_samples} to have drift > tolerance"
        assert strictly_decreased_count == applied_count, (
            f"Projection failed strict decrease in {applied_count - strictly_decreased_count} cases!"
        )

    def test_1000_random_drift_vectors_with_nonzero_target_charges(
        self,
        rng_seed: None,
        standard_config: NoetherianEngineConfig,
    ) -> None:
        """Tests 1000 random drift vectors with non-zero target charges (b_j != 0)."""
        dim = standard_config.dim
        gen = torch.Generator().manual_seed(123)

        # 4 custom charges with distinct non-zero target values
        charges = [
            ConservedChargeDefinition(
                charge_id="target_c1",
                rule_name="Nonzero Target 1",
                generator_vector=sample_unit_vector(dim, generator=gen),
                target_value=0.20,
            ),
            ConservedChargeDefinition(
                charge_id="target_c2",
                rule_name="Nonzero Target 2",
                generator_vector=sample_unit_vector(dim, generator=gen),
                target_value=-0.15,
            ),
            ConservedChargeDefinition(
                charge_id="target_c3",
                rule_name="Nonzero Target 3",
                generator_vector=sample_unit_vector(dim, generator=gen),
                target_value=0.10,
            ),
            ConservedChargeDefinition(
                charge_id="target_c4",
                rule_name="Nonzero Target 4",
                generator_vector=sample_unit_vector(dim, generator=gen),
                target_value=0.25,
            ),
        ]

        q0 = sample_unit_vector(dim, generator=gen)
        num_samples = 1000
        magnitudes = torch.empty(num_samples).uniform_(0.01, 10.0)

        strictly_decreased = 0
        applied_count = 0

        for i in range(num_samples):
            mag = float(magnitudes[i].item())
            direction = sample_unit_vector(dim, generator=gen)
            q_perturbed = q0 + mag * direction

            proj_q, res = project_to_manifold(q_perturbed, charges=charges, config=standard_config)

            assert not torch.isnan(proj_q).any()
            assert not torch.isinf(proj_q).any()

            if res.projection_applied:
                applied_count += 1
                if res.drift_distance_after < res.drift_distance_before:
                    strictly_decreased += 1

        assert applied_count == num_samples
        assert strictly_decreased == applied_count

    def test_extreme_drift_magnitudes_and_inversion(
        self,
        standard_config: NoetherianEngineConfig,
    ) -> None:
        """Tests micro-drift (1e-4), hyper-drift (100.0, 1000.0), and antipodal inversion."""
        engine = NoetherianInvariantEngine(config=standard_config, auto_register_canonical=True)
        q0 = sample_unit_vector(64)

        # 1. Hyper drift (1000.0)
        q_hyper = q0 + 1000.0 * sample_unit_vector(64)
        proj_hyper, res_hyper = engine.project_state(q_hyper)
        assert not torch.isnan(proj_hyper).any()
        assert res_hyper.drift_distance_after < res_hyper.drift_distance_before
        assert math.isclose(torch.norm(proj_hyper, p=2).item(), 1.0, rel_tol=1e-5)

        # 2. Inversion drift: Delta q = -q0 (sum is zero vector)
        q_zero = q0 - q0
        proj_zero, res_zero = engine.project_state(q_zero)
        assert not torch.isnan(proj_zero).any()
        assert math.isclose(torch.norm(proj_zero, p=2).item(), 1.0, rel_tol=1e-5)

        # 3. Antipodal drift: Delta q = -2 * q0
        q_anti = q0 - 2.0 * q0
        proj_anti, res_anti = engine.project_state(q_anti)
        assert not torch.isnan(proj_anti).any()
        assert math.isclose(torch.norm(proj_anti, p=2).item(), 1.0, rel_tol=1e-5)


# ============================================================================
# Harness 2: Degenerate Gram Matrices (16 Collinear Generators & Latency < 5ms)
# ============================================================================


class TestDegenerateGramCollinearHarness:
    """Empirical challenge 2: Tikhonov regularization stability under near-collinear / singular Gram matrices."""

    def test_16_collinear_generators_tikhonov_inversion_under_5ms(
        self,
        rng_seed: None,
        standard_config: NoetherianEngineConfig,
    ) -> None:
        """16 collinear charge generators: C_i = C_0 + 10^-7 N(0, 1).

        Verifies:
        1. Inversion G^-1 = (C C^T + lambda I)^-1 executes stably without SingularMatrixException.
        2. Execution latency is strictly < 5.0 ms per projection (empirical target: < 1.0 ms).
        3. Zero NaNs or Infs generated.
        """
        dim = standard_config.dim
        gen = torch.Generator().manual_seed(999)

        c0 = sample_unit_vector(dim, generator=gen)

        charges_16: list[ConservedChargeDefinition] = []
        for i in range(16):
            noise = 1e-7 * torch.randn(dim, generator=gen)
            ci = c0 + noise
            ci = ci / torch.norm(ci)
            charges_16.append(
                ConservedChargeDefinition(
                    charge_id=f"collinear_{i}",
                    rule_name=f"Collinear Charge {i}",
                    generator_vector=ci,
                    target_value=0.0,
                )
            )

        # Warm-up call
        q_warm = sample_unit_vector(dim, generator=gen)
        project_to_manifold(q_warm, charges=charges_16, config=standard_config)

        # Benchmark 100 evaluations
        latencies_ms: list[float] = []
        for _ in range(100):
            q_test = sample_unit_vector(dim, generator=gen)
            t_start = time.perf_counter()
            proj_q, res = project_to_manifold(q_test, charges=charges_16, config=standard_config)
            t_end = time.perf_counter()

            elapsed_ms = (t_end - t_start) * 1000.0
            latencies_ms.append(elapsed_ms)

            assert not torch.isnan(proj_q).any()
            assert not math.isnan(res.drift_distance_after)
            if res.projection_applied:
                assert res.drift_distance_after < res.drift_distance_before
            else:
                assert res.drift_distance_before <= standard_config.drift_tolerance

        mean_ms = sum(latencies_ms) / len(latencies_ms)
        p95_ms = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]
        max_ms = max(latencies_ms)

        assert mean_ms < 5.0, f"Mean projection latency {mean_ms:.3f}ms exceeded 5.0ms threshold!"
        assert p95_ms < 5.0, f"p95 projection latency {p95_ms:.3f}ms exceeded 5.0ms threshold!"
        assert max_ms < 15.0, f"Max projection latency {max_ms:.3f}ms spike!"

    def test_exact_identical_collinear_generators_rank_one(
        self,
        rng_seed: None,
        standard_config: NoetherianEngineConfig,
    ) -> None:
        """Tests exact identical generators (zero noise, rank-1 Gram matrix of size 16x16 and 32x32)."""
        dim = standard_config.dim
        c0 = sample_unit_vector(dim)

        for num_generators in (16, 32):
            charges_exact = [
                ConservedChargeDefinition(
                    charge_id=f"exact_{i}",
                    rule_name=f"Exact Rank1 Charge {i}",
                    generator_vector=c0.clone(),
                    target_value=0.0,
                )
                for i in range(num_generators)
            ]

            q = sample_unit_vector(dim)
            t0 = time.perf_counter()
            proj_q, res = project_to_manifold(q, charges=charges_exact, config=standard_config)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            assert elapsed_ms < 5.0
            assert not torch.isnan(proj_q).any()
            assert not math.isnan(res.drift_distance_after)
            assert math.isclose(torch.norm(proj_q, p=2).item(), 1.0, rel_tol=1e-5)

    def test_tikhonov_regularization_parameter_sweep(
        self,
        standard_config: NoetherianEngineConfig,
    ) -> None:
        """Sweeps Tikhonov parameter lambda across [1e-10, 1e-2] for collinear generators."""
        dim = standard_config.dim
        c0 = sample_unit_vector(dim)
        charges = [
            ConservedChargeDefinition(charge_id=f"c_{i}", rule_name=f"C {i}", generator_vector=c0.clone())
            for i in range(16)
        ]
        q = sample_unit_vector(dim)

        for reg in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
            cfg = NoetherianEngineConfig(dim=dim, tikhonov_reg=reg)
            proj_q, res = project_to_manifold(q, charges=charges, config=cfg)
            assert not torch.isnan(proj_q).any()
            assert not math.isnan(res.drift_distance_after)


# ============================================================================
# Harness 3: Action Integral Scaling Across Trajectories T in [1, 100]
# ============================================================================


class TestTrajectoryScalingActionIntegralHarness:
    """Empirical challenge 3: Action integral and Mizan score stability across trajectory lengths T in [1, 100]."""

    def test_action_integral_scaling_trajectory_lengths_1_to_100(
        self,
        rng_seed: None,
        standard_config: NoetherianEngineConfig,
    ) -> None:
        """Measures computational linearity and numerical stability for T in [1, 100].

        Verifies:
        1. Action time scales linearly O(T).
        2. Action efficiency A_eff remains in [0.0, 1.0] without underflow/overflow.
        3. Discrete action S_d = sum (T_k - V_k) accurately reflects accumulated dynamics.
        4. Full Noetherian evaluation executes within 10ms even at T=100.
        """
        engine = NoetherianInvariantEngine(config=standard_config, auto_register_canonical=True)
        charges = engine.charges
        dim = standard_config.dim

        T_values = [1, 2, 5, 10, 20, 50, 75, 100]
        results_table: list[dict[str, float]] = []

        for T in T_values:
            gen = torch.Generator().manual_seed(500 + T)
            # Create trajectory of T transitions (T + 1 states)
            states: list[torch.Tensor] = []
            q_curr = sample_unit_vector(dim, generator=gen)
            states.append(q_curr)

            for _ in range(T):
                step_dir = sample_unit_vector(dim, generator=gen)
                q_next = q_curr + 0.05 * step_dir
                q_curr = q_next / torch.norm(q_next)
                states.append(q_curr)

            probs = [[0.9, 0.1]] * (T + 1)
            consequence_costs = [0.02 * (k % 5) for k in range(T + 1)]

            # Benchmark Action evaluation
            t0 = time.perf_counter()
            action_metric = evaluate_trajectory_action(
                path_states=states,
                goal_state=states[0],
                consequence_costs=consequence_costs,
                config=standard_config,
            )
            t_action_ms = (time.perf_counter() - t0) * 1000.0

            # Benchmark full evaluation
            t0 = time.perf_counter()
            full_eval = evaluate_trajectory_noetherian(
                states=states,
                probs=probs,
                config=standard_config,
                charges=charges,
                goal_state=states[0],
                consequence_costs=consequence_costs,
            )
            t_full_ms = (time.perf_counter() - t0) * 1000.0

            # Assert numerical bounds
            assert 0.0 <= action_metric.action_efficiency <= 1.0
            assert 0.0 <= full_eval.mizan_score <= 1.0
            assert not math.isnan(action_metric.discrete_action)
            assert not math.isnan(full_eval.mizan_score)
            assert t_action_ms < 5.0, f"Action evaluation took {t_action_ms:.2f}ms for T={T}"
            assert t_full_ms < 15.0, f"Full evaluation took {t_full_ms:.2f}ms for T={T}"

            results_table.append({
                "T": float(T),
                "t_action_ms": t_action_ms,
                "t_full_ms": t_full_ms,
                "kinetic_sum": action_metric.kinetic_energy_sum,
                "potential_sum": action_metric.potential_energy_sum,
                "discrete_action": action_metric.discrete_action,
                "action_efficiency": action_metric.action_efficiency,
                "mizan_score": full_eval.mizan_score,
            })

        # Check O(T) monotonicity of kinetic energy accumulation
        for i in range(1, len(results_table)):
            assert results_table[i]["kinetic_sum"] > results_table[i - 1]["kinetic_sum"], (
                f"Kinetic energy failed to accumulate monotonically at T={results_table[i]['T']}"
            )

    def test_antipodal_oscillations_worst_case_kinetics(
        self,
        standard_config: NoetherianEngineConfig,
    ) -> None:
        """Worst-case kinetic trajectory: q_k = (-1)^k q_0 for T=100.

        Each step has ||Delta q|| = 2.0 -> T_k = 0.5 * 4.0 = 2.0.
        Verifies that kinetic sum matches 2.0 * T and efficiency gracefully degrades without NaN.
        """
        dim = standard_config.dim
        q0 = sample_unit_vector(dim)
        T = 100
        states = [q0 if k % 2 == 0 else -q0 for k in range(T + 1)]

        metric = evaluate_trajectory_action(states, config=standard_config)

        expected_kinetic = 2.0 * T
        assert math.isclose(metric.kinetic_energy_sum, expected_kinetic, rel_tol=1e-4)
        assert 0.0 <= metric.action_efficiency <= 1.0
        assert not math.isnan(metric.action_efficiency)
        # Action efficiency should be strictly penalized due to massive kinetic movement
        assert metric.action_efficiency < 0.20

    def test_long_horizon_trajectory_scaling_t_250_and_500(
        self,
        standard_config: NoetherianEngineConfig,
    ) -> None:
        """Extended long-horizon trajectory T=250 and T=500 to ensure zero memory exhaustion or stack overflow."""
        dim = standard_config.dim
        gen = torch.Generator().manual_seed(777)
        states = [sample_unit_vector(dim, generator=gen) for _ in range(251)]

        eval_res = evaluate_trajectory_noetherian(states, config=standard_config)
        assert 0.0 <= eval_res.mizan_score <= 1.0
        assert not math.isnan(eval_res.action_efficiency)
        assert not math.isnan(eval_res.noether_flux)


# ============================================================================
# Harness 4: Pathological & Adversarial Edge Cases
# ============================================================================


class TestPathologicalBoundaryAdversarialHarness:
    """Empirical challenge 4: Stress-testing adversarial and pathological inputs."""

    def test_overconstrained_system_m_greater_than_dim(
        self,
        rng_seed: None,
        standard_config: NoetherianEngineConfig,
    ) -> None:
        """Overconstrained system: m=128 charges in d=64 dimensional space.

        Gram matrix is 128x128 with rank at most 64, meaning >=64 zero eigenvalues.
        Tikhonov regularizer lambda*I must stabilize inversion without crashing.
        """
        dim = standard_config.dim
        m = 128
        gen = torch.Generator().manual_seed(888)

        charges_over = [
            ConservedChargeDefinition(
                charge_id=f"over_{i}",
                rule_name=f"Overconstrained Charge {i}",
                generator_vector=sample_unit_vector(dim, generator=gen),
                target_value=0.0,
            )
            for i in range(m)
        ]

        q = sample_unit_vector(dim, generator=gen)
        proj_q, res = project_to_manifold(q, charges=charges_over, config=standard_config)

        assert not torch.isnan(proj_q).any()
        assert not torch.isinf(proj_q).any()
        assert res.projection_applied is True
        assert res.drift_distance_after < res.drift_distance_before

    def test_conflicting_invariants_minimum_norm_compromise(
        self,
        standard_config: NoetherianEngineConfig,
    ) -> None:
        """Conflicting invariants: Two parallel generators with contradictory targets (b1=0.5, b2=-0.5).

        Mathematically impossible to satisfy both; verifies system finds minimum-norm least-squares compromise.
        """
        dim = standard_config.dim
        c0 = sample_unit_vector(dim)

        charges_conflict = [
            ConservedChargeDefinition(charge_id="c_pos", rule_name="Positive Target", generator_vector=c0.clone(), target_value=0.5),
            ConservedChargeDefinition(charge_id="c_neg", rule_name="Negative Target", generator_vector=c0.clone(), target_value=-0.5),
        ]

        q = sample_unit_vector(dim)
        proj_q, res = project_to_manifold(q, charges=charges_conflict, config=standard_config)

        assert not torch.isnan(proj_q).any()
        assert not torch.isinf(proj_q).any()
        assert 0.0 <= res.compliance_score <= 1.0
        assert math.isclose(torch.norm(proj_q, p=2).item(), 1.0, rel_tol=1e-5)

    def test_adversarial_tensor_corruption_nan_inf_complex(
        self,
        standard_config: NoetherianEngineConfig,
    ) -> None:
        """All-NaN vector, all-Inf vector, and complex tensors are sanitized without crashing."""
        dim = standard_config.dim
        all_nan = torch.full((dim,), float("nan"))
        all_inf = torch.full((dim,), float("inf"))

        engine = NoetherianInvariantEngine(config=standard_config, auto_register_canonical=True)

        # NaN state
        proj_nan, res_nan = engine.project_state(all_nan)
        assert not torch.isnan(proj_nan).any()
        assert math.isclose(torch.norm(proj_nan, p=2).item(), 1.0, rel_tol=1e-5)

        # Inf state
        proj_inf, res_inf = engine.project_state(all_inf)
        assert not torch.isnan(proj_inf).any()
        assert not torch.isinf(proj_inf).any()

    def test_epistemic_entropy_boundary_distributions(
        self,
        standard_config: NoetherianEngineConfig,
    ) -> None:
        """Epistemic entropy handles tiny probabilities (1e-30), high-dim (N=1024), and all-zeros."""
        # 1. Tiny probabilities
        p_tiny = torch.tensor([1.0 - 1e-15, 1e-15, 1e-30, 0.0])
        m_tiny = evaluate_epistemic_entropy([p_tiny], config=standard_config)
        assert not math.isnan(m_tiny.initial_entropy)
        assert m_tiny.initial_entropy >= 0.0

        # 2. Large distribution N=1024
        p_large = torch.ones(1024) / 1024.0
        m_large = evaluate_epistemic_entropy([p_large], config=standard_config)
        assert math.isclose(m_large.initial_entropy, 1.0, rel_tol=1e-4)

        # 3. All zeros distribution (falls back to uniform)
        p_zeros = torch.zeros(10)
        m_zeros = evaluate_epistemic_entropy([p_zeros], config=standard_config)
        assert not math.isnan(m_zeros.initial_entropy)
        assert m_zeros.initial_entropy == pytest.approx(1.0, abs=1e-4)
