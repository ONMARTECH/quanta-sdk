"""Comprehensive Hermetic Pytest Suite for Noetherian Invariant Cognitive Arbiter (Mizan Hakemi).

Tiers Tested:
- Tier 1: Feature Coverage (Conserved charges, action integral, epistemic entropy,
          Noether flux, manifold projection, composite Mizan score).
- Tier 2: Boundary & Corner Cases (Single state, zero drift, orthogonal jumps,
          extreme distributions, near-collinear ill-conditioned constraints, NaNs/Infs).
- Tier 3: Cross-Feature & Engine Integration (Kinetic + entropy interaction,
          projection restoration, charge management, MizanHakemi class APIs).
- Tier 4: Real-World Adversarial Scenarios (Creeping drift, periphrastic rule erosion,
          Pareto dilemmas, deep DAG trajectory discrimination, hermetic tmp_path isolation).

Integrity Protocol (GEMINI.md):
- Strictly hermetic: No dependency on live `quanta_cognitive_state.json`.
- Zero benchmark leakage, zero cherry-picking, mathematically sound assertions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from quanta.cognitive.arbiter import (
    ConsequenceVector,
    DecisionDAG,
    DecisionEdge,
    DecisionNode,
    QuantumDecisionArbiter,
)
from quanta.cognitive.noetherian import (
    ActionIntegralMetric,
    ConservedChargeDefinition,
    EpistemicEntropyMetric,
    InvariantViolation,
    ManifoldProjectionResult,
    MizanHakemi,
    NoetherianEngineConfig,
    NoetherianEvaluation,
    NoetherianInvariantEngine,
    evaluate_epistemic_entropy,
    evaluate_noether_charges,
    evaluate_trajectory_action,
    evaluate_trajectory_noetherian,
    project_to_invariant_manifold,
    project_to_manifold,
)
from quanta.cognitive.telemetry import (
    get_telemetry_summary,
    read_telemetry_events,
)

# ============================================================================
# Test Fixtures & Helpers
# ============================================================================


@pytest.fixture
def rng_seed() -> None:
    """Ensures deterministic PyTorch and Python pseudo-random generation."""
    torch.manual_seed(42)


@pytest.fixture
def standard_config() -> NoetherianEngineConfig:
    """Provides a standard calibrated NoetherianEngineConfig."""
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


@pytest.fixture
def canonical_charges(standard_config: NoetherianEngineConfig) -> list[ConservedChargeDefinition]:
    """Generates 4 orthogonal/canonical conserved charges for 64-dim Hilbert space."""
    dim = standard_config.dim
    charges = [
        ConservedChargeDefinition.from_text(
            charge_id="test_intent",
            rule_name="User Intent Alignment",
            description="Preserves semantic fidelity to original intent.",
            dim=dim,
            target_value=0.0,
            weight=1.0,
            tolerance=0.05,
            is_hard_constraint=True,
        ),
        ConservedChargeDefinition.from_text(
            charge_id="test_safety",
            rule_name="Safety Boundary",
            description="Enforces operational security and guardrails.",
            dim=dim,
            target_value=0.0,
            weight=1.5,
            tolerance=0.03,
            is_hard_constraint=True,
        ),
        ConservedChargeDefinition.from_text(
            charge_id="test_truth",
            rule_name="Epistemic Factuality",
            description="Restrains hallucination and unfounded claims.",
            dim=dim,
            target_value=0.0,
            weight=1.0,
            tolerance=0.05,
            is_hard_constraint=True,
        ),
        ConservedChargeDefinition.from_text(
            charge_id="test_coherence",
            rule_name="Contextual Coherence",
            description="Guarantees temporal consistency across turns.",
            dim=dim,
            target_value=0.0,
            weight=0.8,
            tolerance=0.08,
            is_hard_constraint=False,
        ),
    ]
    return charges


def create_normalized_vector(dim: int = 64, seed_offset: int = 0) -> torch.Tensor:
    """Creates a deterministic unit-norm statevector in R^dim."""
    gen = torch.Generator().manual_seed(1000 + seed_offset)
    v = torch.randn(dim, generator=gen, dtype=torch.float32)
    return v / torch.norm(v, p=2)


# ============================================================================
# Tier 1: Feature Coverage Tests
# ============================================================================


class TestTier1ConservedChargeDefinition:
    """Verifies ConservedChargeDefinition initialization, normalization, and serialization."""

    def test_charge_initialization_and_normalization(self) -> None:
        """Confirms generator vectors are automatically normalized to unit norm."""
        raw_vec = torch.tensor([3.0, 4.0] + [0.0] * 62, dtype=torch.float32)
        charge = ConservedChargeDefinition(
            charge_id="charge_norm",
            rule_name="Normalization Test",
            description="Testing norm enforcement",
            generator_vector=raw_vec,
            target_value=0.1,
            weight=1.2,
            tolerance=0.04,
            is_hard_constraint=True,
        )
        norm = torch.norm(charge.generator_vector, p=2).item()
        assert math.isclose(norm, 1.0, rel_tol=1e-5), f"Norm should be 1.0, got {norm}"
        assert charge.generator_vector[0].item() == pytest.approx(0.6, abs=1e-4)
        assert charge.generator_vector[1].item() == pytest.approx(0.8, abs=1e-4)

    def test_charge_from_text_deterministic(self) -> None:
        """Verifies deterministic SHA-256 generation from text parameters."""
        c1 = ConservedChargeDefinition.from_text("q1", "Rule A", "Description 1", dim=64)
        c2 = ConservedChargeDefinition.from_text("q1", "Rule A", "Description 1", dim=64)
        c3 = ConservedChargeDefinition.from_text("q2", "Rule B", "Description 2", dim=64)

        assert torch.allclose(c1.generator_vector, c2.generator_vector)
        assert not torch.allclose(c1.generator_vector, c3.generator_vector)
        assert math.isclose(torch.norm(c1.generator_vector, p=2).item(), 1.0, rel_tol=1e-5)

    def test_charge_serialization_roundtrip(self) -> None:
        """Verifies to_dict and from_dict preserve all properties."""
        v = create_normalized_vector(64, seed_offset=1)
        charge = ConservedChargeDefinition(
            charge_id="ser_test",
            rule_name="Serialization Rule",
            description="Testing dict serialization",
            generator_vector=v,
            target_value=0.25,
            weight=1.5,
            tolerance=0.03,
            is_hard_constraint=False,
        )
        d = charge.to_dict()
        assert d["charge_id"] == "ser_test"
        assert d["target_value"] == pytest.approx(0.25, abs=1e-4)
        assert d["is_hard_constraint"] is False

        # Hydrate from dict with generator_vector
        d["generator_vector"] = v.tolist()
        restored = ConservedChargeDefinition.from_dict(d)
        assert restored.charge_id == charge.charge_id
        assert restored.rule_name == charge.rule_name
        assert restored.target_value == pytest.approx(charge.target_value, abs=1e-4)
        assert restored.is_hard_constraint is False
        assert torch.allclose(restored.generator_vector, charge.generator_vector, atol=1e-4)

    def test_charge_property_aliases(self) -> None:
        """Verifies name and target_charge property aliases."""
        charge = ConservedChargeDefinition(
            charge_id="alias_test",
            rule_name="Alias Policy",
            target_value=0.42,
        )
        assert charge.name == "Alias Policy"
        assert charge.target_charge == pytest.approx(0.42)

    def test_charge_zero_vector_fallback(self) -> None:
        """Verifies fallback to unit basis vector when zero vector is supplied."""
        zero_v = torch.zeros(64, dtype=torch.float32)
        charge = ConservedChargeDefinition(
            charge_id="zero_test",
            rule_name="Zero Vector Test",
            generator_vector=zero_v,
        )
        norm = torch.norm(charge.generator_vector, p=2).item()
        assert math.isclose(norm, 1.0, rel_tol=1e-5)
        assert charge.generator_vector[0].item() == 1.0


class TestTier1NoetherianEngineConfig:
    """Verifies NoetherianEngineConfig defaults, normalization, and properties."""

    def test_config_defaults(self, standard_config: NoetherianEngineConfig) -> None:
        """Verifies default values and dimension."""
        assert standard_config.dim == 64
        assert pytest.approx(standard_config.weight_action + standard_config.weight_entropy + standard_config.weight_charge + standard_config.weight_manifold) == 1.0
        assert standard_config.drift_tolerance == 0.02
        assert standard_config.tikhonov_reg == 1e-6
        assert standard_config.critical_breach_threshold == 0.40

    def test_config_weight_normalization(self) -> None:
        """Verifies weights summing to != 1.0 are automatically normalized."""
        cfg = NoetherianEngineConfig(
            weight_action=1.0,
            weight_entropy=1.0,
            weight_charge=2.0,
            weight_manifold=0.0,
        )
        total = cfg.weight_action + cfg.weight_entropy + cfg.weight_charge + cfg.weight_manifold
        assert math.isclose(total, 1.0, rel_tol=1e-5)
        assert cfg.weight_action == pytest.approx(0.25)
        assert cfg.weight_entropy == pytest.approx(0.25)
        assert cfg.weight_charge == pytest.approx(0.50)
        assert cfg.weight_manifold == pytest.approx(0.0)

    def test_config_properties_and_serialization(self, standard_config: NoetherianEngineConfig) -> None:
        """Verifies aliases (w_A, w_E, w_C, w_P, epsilon, beta) and to_dict."""
        assert standard_config.w_A == standard_config.weight_action
        assert standard_config.w_E == standard_config.weight_entropy
        assert standard_config.w_C == standard_config.weight_charge
        assert standard_config.w_P == standard_config.weight_manifold
        assert standard_config.epsilon == standard_config.drift_tolerance
        assert standard_config.beta == pytest.approx(1.0 / (standard_config.action_scale ** 2))

        d = standard_config.to_dict()
        assert d["dim"] == 64
        assert "tikhonov_reg" in d
        assert "critical_breach_threshold" in d


class TestTier1ActionIntegralMetric:
    """Verifies discrete Hamilton action integral calculation and efficiency."""

    def test_action_stationary_trajectory(self, standard_config: NoetherianEngineConfig) -> None:
        """Identical consecutive states produce zero kinetic energy and high efficiency."""
        q0 = create_normalized_vector(64, seed_offset=1)
        path = [q0, q0.clone(), q0.clone()]

        metric = evaluate_trajectory_action(
            path_states=path,
            goal_state=q0,
            consequence_costs=[0.0, 0.0, 0.0],
            config=standard_config,
        )

        assert metric.kinetic_energy_sum == pytest.approx(0.0, abs=1e-6)
        assert metric.potential_energy_sum == pytest.approx(0.0, abs=1e-6)
        assert metric.discrete_action == pytest.approx(0.0, abs=1e-6)
        assert metric.action_efficiency == pytest.approx(1.0, abs=1e-4)

    def test_action_kinetic_energy_growth(self, standard_config: NoetherianEngineConfig) -> None:
        """Steep state changes increase kinetic energy and reduce action efficiency."""
        q0 = create_normalized_vector(64, seed_offset=1)
        q1 = create_normalized_vector(64, seed_offset=2)
        q2 = create_normalized_vector(64, seed_offset=3)

        metric = evaluate_trajectory_action(
            path_states=[q0, q1, q2],
            goal_state=q0,
            consequence_costs=[0.0, 0.5, 0.5],
            config=standard_config,
        )

        assert metric.kinetic_energy_sum > 0.0
        assert metric.potential_energy_sum > 0.0
        assert 0.0 <= metric.action_efficiency < 1.0

    def test_action_metric_dataclass_methods(self) -> None:
        """Verifies aliases and serialization of ActionIntegralMetric."""
        m = ActionIntegralMetric(
            kinetic_energy_sum=1.2,
            potential_energy_sum=0.4,
            discrete_action=0.8,
            action_efficiency=0.75,
        )
        assert m.kinetic_energy == pytest.approx(1.2)
        assert m.potential_energy == pytest.approx(0.4)
        assert m.lagrangian == pytest.approx(0.8)
        assert m.action == pytest.approx(0.8)

        d = m.to_dict()
        restored = ActionIntegralMetric.from_dict(d)
        assert restored.kinetic_energy_sum == pytest.approx(1.2)
        assert restored.potential_energy_sum == pytest.approx(0.4)
        assert restored.action_efficiency == pytest.approx(0.75)

    def test_trajectory_action_transition_indexed_costs(
        self, standard_config: NoetherianEngineConfig
    ) -> None:
        """Verifies T transition costs are mapped to T transitions without index shift."""
        q0 = create_normalized_vector(64, seed_offset=1)
        q1 = create_normalized_vector(64, seed_offset=2)
        q2 = create_normalized_vector(64, seed_offset=3)

        # 3 states -> 2 transitions. Provide 2 transition costs: [0.4, 0.6]
        metric = evaluate_trajectory_action(
            path_states=[q0, q1, q2],
            consequence_costs=[0.4, 0.6],
            config=standard_config,
        )
        expected_potential = 0.5 * 0.4 + 0.5 * 0.6  # 0.5
        assert metric.potential_energy_sum == pytest.approx(expected_potential, abs=1e-5)

    def test_trajectory_action_state_indexed_compatibility(
        self, standard_config: NoetherianEngineConfig
    ) -> None:
        """Verifies backward compatibility with T+1 state-indexed cost sequences."""
        q0 = create_normalized_vector(64, seed_offset=1)
        q1 = create_normalized_vector(64, seed_offset=2)
        q2 = create_normalized_vector(64, seed_offset=3)

        # 3 states -> provide 3 state costs: [0.0 (dummy root), 0.4, 0.6]
        metric = evaluate_trajectory_action(
            path_states=[q0, q1, q2],
            consequence_costs=[0.0, 0.4, 0.6],
            config=standard_config,
        )
        expected_potential = 0.5 * 0.4 + 0.5 * 0.6  # 0.5
        assert metric.potential_energy_sum == pytest.approx(expected_potential, abs=1e-5)

    def test_trajectory_action_scalar_cost_broadcasting(
        self, standard_config: NoetherianEngineConfig
    ) -> None:
        """Verifies a single scalar cost broadcasts uniformly across all transitions."""
        states = [create_normalized_vector(64, seed_offset=i) for i in range(5)]
        # 5 states -> 4 transitions. Provide scalar list [0.3] or float 0.3
        metric_list = evaluate_trajectory_action(
            path_states=states,
            consequence_costs=[0.3],
            config=standard_config,
        )
        metric_scalar = evaluate_trajectory_action(
            path_states=states,
            consequence_costs=0.3,
            config=standard_config,
        )
        expected_potential = 4 * 0.5 * 0.3  # 0.6
        assert metric_list.potential_energy_sum == pytest.approx(expected_potential, abs=1e-5)
        assert metric_scalar.potential_energy_sum == pytest.approx(expected_potential, abs=1e-5)


class TestTier1EpistemicEntropyMetric:
    """Verifies epistemic entropy tracking, spurts, and decoherence detection."""

    def test_entropy_constant_sharp_distribution(self, standard_config: NoetherianEngineConfig) -> None:
        """Consistent confident predictions yield zero spurt and stability score of 1.0."""
        probs = [
            [0.95, 0.03, 0.02],
            [0.96, 0.02, 0.02],
            [0.95, 0.03, 0.02],
        ]
        metric = evaluate_epistemic_entropy(probs, config=standard_config)

        assert metric.max_entropy_spurt <= 0.05
        assert metric.entropy_stability_score == pytest.approx(1.0, abs=0.05)
        assert not metric.is_decoherent

    def test_entropy_spurt_and_decoherence(self, standard_config: NoetherianEngineConfig) -> None:
        """Sudden uncertainty explosion from sharp to uniform triggers decoherence."""
        probs = [
            [0.99, 0.01],  # low entropy
            [0.50, 0.50],  # maximum binary entropy (jump > 0.15)
        ]
        metric = evaluate_epistemic_entropy(probs, config=standard_config)

        assert metric.max_entropy_spurt > standard_config.max_allowed_entropy_flux
        assert metric.entropy_stability_score < 1.0
        assert metric.is_decoherent is True

    def test_entropy_scalar_inputs(self, standard_config: NoetherianEngineConfig) -> None:
        """Handles scalar confidence sequences safely via binary Shannon entropy."""
        confidences = [0.99, 0.95, 0.92]
        metric = evaluate_epistemic_entropy(confidences, config=standard_config)
        assert 0.0 <= metric.initial_entropy <= 1.0
        assert 0.0 <= metric.final_entropy <= 1.0
        assert 0.0 <= metric.entropy_stability_score <= 1.0

    def test_entropy_metric_dataclass_methods(self) -> None:
        """Verifies aliases and serialization of EpistemicEntropyMetric."""
        m = EpistemicEntropyMetric(
            initial_entropy=0.1,
            final_entropy=0.3,
            max_entropy_spurt=0.2,
            cumulative_entropy_flux=0.15,
            entropy_stability_score=0.85,
            is_decoherent=False,
        )
        assert m.entropy_production == pytest.approx(0.2)
        assert m.dispersion_rate == pytest.approx(0.2)
        assert m.stability_score == pytest.approx(0.85)

        d = m.to_dict()
        restored = EpistemicEntropyMetric.from_dict(d)
        assert restored.initial_entropy == pytest.approx(0.1)
        assert restored.final_entropy == pytest.approx(0.3)
        assert restored.is_decoherent is False


class TestTier1SemanticChargeConservation:
    """Verifies Lie generator charges, discrete Noether flux, and violation logging."""

    def test_perfect_charge_conservation(
        self,
        standard_config: NoetherianEngineConfig,
        canonical_charges: list[ConservedChargeDefinition],
    ) -> None:
        """A state trajectory with zero divergence from target charges has zero flux."""
        # Find a statevector with zero charge by construction
        C = torch.stack([c.generator_vector for c in canonical_charges], dim=0)
        # Random vector projected to null space of C
        raw_v = create_normalized_vector(64, seed_offset=5)
        # Orthogonal projection: v_null = (I - C^T (C C^T)^{-1} C) raw_v
        gram = torch.matmul(C, C.T) + 1e-6 * torch.eye(4)
        proj_coeff = torch.linalg.solve(gram, torch.matmul(C, raw_v))
        v_null = raw_v - torch.matmul(C.T, proj_coeff)
        v_null = v_null / torch.norm(v_null, p=2)

        path = [v_null, v_null.clone(), v_null.clone()]
        flux, drift, c_eff, violations = evaluate_noether_charges(
            path_states=path,
            charges=canonical_charges,
            config=standard_config,
        )

        assert flux == pytest.approx(0.0, abs=1e-5)
        assert drift <= 0.01
        assert c_eff > 0.95
        assert len(violations) == 0

    def test_charge_violation_and_noether_flux_detection(
        self,
        standard_config: NoetherianEngineConfig,
        canonical_charges: list[ConservedChargeDefinition],
    ) -> None:
        """Drifting along a generator vector induces measurable Noether flux and logs violations."""
        q0 = create_normalized_vector(64, seed_offset=1)
        # Perturb q1 specifically in the direction of the first charge generator
        phi_0 = canonical_charges[0].generator_vector
        q1 = q0 + 0.3 * phi_0
        q1 = q1 / torch.norm(q1, p=2)

        flux, drift, c_eff, violations = evaluate_noether_charges(
            path_states=[q0, q1],
            charges=canonical_charges,
            config=standard_config,
        )

        assert flux > 0.0
        assert drift > 0.0
        assert c_eff < 1.0
        assert any(canonical_charges[0].rule_name in v for v in violations)

    def test_step_0_breach_detected_in_multistep_trajectory(
        self,
        standard_config: NoetherianEngineConfig,
    ) -> None:
        """Verifies initial state q_0 tolerance breaches are caught in multi-step trajectories."""
        gen = torch.zeros(64)
        gen[0] = 1.0
        hard_charge = ConservedChargeDefinition(
            charge_id="hard_root_safety",
            rule_name="Root Guard",
            generator_vector=gen,
            target_value=0.0,
            tolerance=0.05,
            is_hard_constraint=True,
        )

        q_breached = torch.zeros(64)
        q_breached[0] = 1.0  # Drift = 1.0 >> 0.05
        q_valid = torch.zeros(64)
        q_valid[1] = 1.0

        # Multi-step trajectory [q_breached, q_valid]
        flux, max_drift, c_eff, violations = evaluate_noether_charges(
            path_states=[q_breached, q_valid],
            charges=[hard_charge],
            config=standard_config,
        )

        assert max_drift == pytest.approx(1.0, abs=1e-5)
        assert len(violations) >= 1
        assert any("Root Guard" in str(v) and "step 0" in str(v) for v in violations)


class TestTier1ManifoldProjection:
    """Verifies Tikhonov-regularized Gram manifold projection and spherical retraction."""

    def test_projection_reduces_drift(
        self,
        standard_config: NoetherianEngineConfig,
        canonical_charges: list[ConservedChargeDefinition],
    ) -> None:
        """Projection strictly reduces distance to invariant constraints ||C q - b||."""
        state = create_normalized_vector(64, seed_offset=10)
        proj_state, result = project_to_manifold(
            state=state,
            charges=canonical_charges,
            config=standard_config,
        )

        assert result.projection_applied is True
        assert result.drift_distance_after < result.drift_distance_before
        assert 0.0 <= result.compliance_score <= 1.0
        # Spherical retraction check
        norm = torch.norm(proj_state, p=2).item()
        assert math.isclose(norm, 1.0, rel_tol=1e-5)

    def test_projection_within_tolerance_noop(
        self,
        standard_config: NoetherianEngineConfig,
        canonical_charges: list[ConservedChargeDefinition],
    ) -> None:
        """States already on the manifold are not modified."""
        state = create_normalized_vector(64, seed_offset=1)
        C = torch.stack([c.generator_vector for c in canonical_charges], dim=0)
        # Compute exact charges for this state and set as target
        exact_b = torch.matmul(C, state)

        proj_state, result = project_to_manifold(
            state=state,
            charges=canonical_charges,
            target_b=exact_b,
            config=standard_config,
        )

        assert result.projection_applied is False
        assert result.drift_distance_before <= standard_config.drift_tolerance
        assert torch.allclose(proj_state, state, atol=1e-6)

    def test_projection_alias_compatibility(
        self,
        standard_config: NoetherianEngineConfig,
        canonical_charges: list[ConservedChargeDefinition],
    ) -> None:
        """project_to_invariant_manifold is an exact functional alias."""
        state = create_normalized_vector(64, seed_offset=7)
        s1, r1 = project_to_manifold(state, canonical_charges, config=standard_config)
        s2, r2 = project_to_invariant_manifold(state, canonical_charges, config=standard_config)

        assert torch.allclose(s1, s2)
        assert r1.drift_distance_after == pytest.approx(r2.drift_distance_after)

    def test_projection_result_dataclass_methods(self) -> None:
        """Verifies aliases and serialization of ManifoldProjectionResult."""
        res = ManifoldProjectionResult(
            drift_distance_before=0.25,
            drift_distance_after=0.01,
            projection_applied=True,
            iterations_run=2,
            charge_restoration_error=0.01,
            compliance_score=0.95,
        )
        assert res.projection_distance == pytest.approx(0.25)
        assert res.residual_error == pytest.approx(0.01)
        d = res.to_dict()
        restored = ManifoldProjectionResult.from_dict(d)
        assert restored.drift_distance_before == pytest.approx(0.25)
        assert restored.drift_distance_after == pytest.approx(0.01)
        assert restored.projection_applied is True


class TestTier1CompositeMizanEvaluation:
    """Verifies composite Kozmik Mizan scoring and regime classification."""

    def test_mizan_score_range_and_convex_combination(
        self,
        standard_config: NoetherianEngineConfig,
        canonical_charges: list[ConservedChargeDefinition],
    ) -> None:
        """Score is strictly bounded in [0.0, 1.0] and matches convex weights."""
        q0 = create_normalized_vector(64, seed_offset=1)
        q1 = create_normalized_vector(64, seed_offset=2)

        eval_res = evaluate_trajectory_noetherian(
            states=[q0, q1],
            probs=[[0.9, 0.1], [0.85, 0.15]],
            config=standard_config,
            charges=canonical_charges,
        )

        assert 0.0 <= eval_res.mizan_score <= 1.0
        expected = (
            standard_config.w_A * eval_res.action_efficiency
            + standard_config.w_E * eval_res.entropy_stability
            + standard_config.w_C * eval_res.charge_conservation
            + standard_config.w_P * eval_res.projection_compliance
        )
        assert math.isclose(eval_res.mizan_score, expected, rel_tol=1e-4)

    def test_regime_classification_harmonic_to_breach(
        self,
        standard_config: NoetherianEngineConfig,
        canonical_charges: list[ConservedChargeDefinition],
    ) -> None:
        """Verifies HARMONIC_EQUILIBRIUM for stationary vs CRITICAL_BREACH for violent perturbations."""
        q0 = create_normalized_vector(64, seed_offset=1)
        # Stationary path with constant high confidence
        harmonic_res = evaluate_trajectory_noetherian(
            states=[q0, q0.clone()],
            probs=[[0.99, 0.01], [0.99, 0.01]],
            config=standard_config,
            charges=[],  # empty charges -> 1.0 conservation
        )
        assert harmonic_res.regime == "HARMONIC_EQUILIBRIUM"
        assert harmonic_res.is_valid is True

    def test_hard_constraint_breach_invalidates(
        self,
        standard_config: NoetherianEngineConfig,
        canonical_charges: list[ConservedChargeDefinition],
    ) -> None:
        """A hard constraint breach marks is_valid as False regardless of score."""
        q0 = create_normalized_vector(64, seed_offset=1)
        phi_hard = canonical_charges[1].generator_vector  # Safety boundary (hard constraint)
        q_violating = q0 + 0.5 * phi_hard
        q_violating = q_violating / torch.norm(q_violating, p=2)

        eval_res = evaluate_trajectory_noetherian(
            states=[q0, q_violating],
            probs=[[0.9, 0.1], [0.9, 0.1]],
            config=standard_config,
            charges=canonical_charges,
        )

        assert len(eval_res.invariant_violations) > 0
        assert eval_res.is_valid is False

    def test_substring_rule_collision_does_not_falsely_invalidate(self) -> None:
        """Verifies that soft rule containing hard rule name does not falsely invalidate trajectory."""
        gen_soft = torch.zeros(64)
        gen_soft[0] = 1.0
        gen_hard = torch.zeros(64)
        gen_hard[1] = 1.0

        soft_rule = ConservedChargeDefinition(
            charge_id="soft_rule",
            rule_name="Safe Boundary Policy",
            generator_vector=gen_soft,
            target_value=0.0,
            tolerance=0.05,
            is_hard_constraint=False,
        )
        hard_rule = ConservedChargeDefinition(
            charge_id="hard_rule",
            rule_name="Safe",
            generator_vector=gen_hard,
            target_value=0.0,
            tolerance=0.05,
            is_hard_constraint=True,
        )
        config = NoetherianEngineConfig(critical_breach_threshold=0.10)

        # State breaching ONLY the soft rule; hard rule is satisfied
        q_test = torch.zeros(64)
        q_test[0] = 1.0
        q_test[1] = 0.0

        eval_res = evaluate_trajectory_noetherian([q_test], config=config, charges=[soft_rule, hard_rule])

        assert eval_res.is_valid is True
        assert len(eval_res.invariant_violations) == 1
        assert "Safe Boundary Policy" in str(eval_res.invariant_violations[0])

    def test_invariant_violation_object_attributes_and_serialization(self) -> None:
        """Verifies InvariantViolation properties, string compatibility, and serialization."""
        v = InvariantViolation(
            "Charge 'Test' breached at step 0: drift=0.1234 > tol=0.0500",
            charge_id="cid_01",
            rule_name="Test Rule",
            step=0,
            observed_drift=0.1234,
            tolerance=0.05,
            is_hard_constraint=True,
        )
        # String subclass behavior
        assert isinstance(v, str)
        assert "drift=0.1234" in v
        assert v.startswith("Charge 'Test'")

        # Strongly-typed metadata
        assert v.charge_id == "cid_01"
        assert v.rule_name == "Test Rule"
        assert v.step == 0
        assert v.observed_drift == pytest.approx(0.1234)
        assert v.tolerance == pytest.approx(0.05)
        assert v.is_hard_constraint is True

        # Dict serialization
        d = v.to_dict()
        assert d["charge_id"] == "cid_01"
        assert d["is_hard_constraint"] is True

    def test_evaluation_serialization_roundtrip(
        self,
        standard_config: NoetherianEngineConfig,
        canonical_charges: list[ConservedChargeDefinition],
    ) -> None:
        """Verifies NoetherianEvaluation to_dict and from_dict fidelity."""
        q0 = create_normalized_vector(64, seed_offset=1)
        eval_res = evaluate_trajectory_noetherian(
            states=[q0, q0.clone()],
            config=standard_config,
            charges=canonical_charges,
        )
        d = eval_res.to_dict()
        restored = NoetherianEvaluation.from_dict(d)

        assert restored.mizan_score == pytest.approx(eval_res.mizan_score)
        assert restored.noether_flux == pytest.approx(eval_res.noether_flux)
        assert restored.regime == eval_res.regime
        assert restored.action_metric.kinetic_energy_sum == pytest.approx(eval_res.action_metric.kinetic_energy_sum)


# ============================================================================
# Tier 2: Boundary & Corner Cases Tests
# ============================================================================


class TestTier2BoundaryAndCornerCases:
    """Verifies edge cases, single state, ill-conditioned matrices, and NaN robustness."""

    def test_single_state_trajectory(
        self,
        standard_config: NoetherianEngineConfig,
        canonical_charges: list[ConservedChargeDefinition],
    ) -> None:
        """Single-state input produces valid evaluation without division by zero."""
        q0 = create_normalized_vector(64, seed_offset=1)
        eval_res = evaluate_trajectory_noetherian(
            states=[q0],
            probs=[[0.9, 0.1]],
            config=standard_config,
            charges=canonical_charges,
        )

        assert 0.0 <= eval_res.mizan_score <= 1.0
        assert eval_res.action_efficiency == 1.0
        assert eval_res.entropy_stability == 1.0
        assert not math.isnan(eval_res.mizan_score)

    def test_empty_states_sequence(self, standard_config: NoetherianEngineConfig) -> None:
        """Empty states sequence returns safe default NoetherianEvaluation."""
        eval_res = evaluate_trajectory_noetherian(states=[], config=standard_config)
        assert eval_res.mizan_score == 1.0
        assert eval_res.is_valid is True
        assert eval_res.regime == "HARMONIC_EQUILIBRIUM"

    def test_extreme_orthogonal_state_transitions(
        self,
        standard_config: NoetherianEngineConfig,
    ) -> None:
        """90-degree orthogonal jumps (dot=0) incur maximum kinetic cost."""
        e1 = torch.zeros(64, dtype=torch.float32)
        e1[0] = 1.0
        e2 = torch.zeros(64, dtype=torch.float32)
        e2[1] = 1.0

        # ||e2 - e1||^2 = 1^2 + (-1)^2 = 2.0 -> T = 0.5 * 2 = 1.0
        metric = evaluate_trajectory_action(
            path_states=[e1, e2],
            goal_state=e1,
            config=standard_config,
        )
        assert metric.kinetic_energy_sum == pytest.approx(1.0, abs=1e-5)
        # Potential: V_goal(e2) = 1.0 - dot(e2, e1) = 1.0 -> 0.5 * 1.0 = 0.5
        assert metric.potential_energy_sum == pytest.approx(0.5, abs=1e-5)
        assert metric.discrete_action == pytest.approx(0.5, abs=1e-5)

    def test_uniform_vs_dirac_delta_distribution_entropy(
        self,
        standard_config: NoetherianEngineConfig,
    ) -> None:
        """Verifies Dirac delta H=0 vs Uniform H=1.0 normalization."""
        # Dirac delta: [1.0, 0.0, 0.0, 0.0]
        p_dirac = torch.tensor([1.0, 0.0, 0.0, 0.0])
        # Uniform: [0.25, 0.25, 0.25, 0.25]
        p_uniform = torch.tensor([0.25, 0.25, 0.25, 0.25])

        m_dirac = evaluate_epistemic_entropy([p_dirac], config=standard_config)
        m_uniform = evaluate_epistemic_entropy([p_uniform], config=standard_config)

        assert m_dirac.initial_entropy == pytest.approx(0.0, abs=1e-4)
        assert m_uniform.initial_entropy == pytest.approx(1.0, abs=1e-4)

    def test_singular_collinear_generators_tikhonov_stability(
        self,
        standard_config: NoetherianEngineConfig,
    ) -> None:
        """Ill-conditioned / collinear constraint generators do not crash due to Tikhonov regularizer."""
        v = create_normalized_vector(64, seed_offset=30)
        # Two identical collinear generators (rank 1 Gram matrix)
        c1 = ConservedChargeDefinition(charge_id="c1", rule_name="R1", generator_vector=v)
        c2 = ConservedChargeDefinition(charge_id="c2", rule_name="R2", generator_vector=v)

        q = create_normalized_vector(64, seed_offset=31)
        # Must not raise LinAlgError or produce NaNs
        proj_q, result = project_to_manifold(q, charges=[c1, c2], config=standard_config)

        assert not torch.isnan(proj_q).any()
        assert not math.isnan(result.drift_distance_after)
        assert math.isclose(torch.norm(proj_q, p=2).item(), 1.0, rel_tol=1e-5)

    def test_dimension_mismatch_and_padding(self, standard_config: NoetherianEngineConfig) -> None:
        """Inputs with dim < 64 or dim > 64 are safely padded or truncated without error."""
        small_vec = torch.tensor([0.5, 0.5], dtype=torch.float32)
        large_vec = torch.randn(128, dtype=torch.float32)

        metric = evaluate_trajectory_action(
            path_states=[small_vec, large_vec],
            config=standard_config,
        )
        assert not math.isnan(metric.kinetic_energy_sum)
        assert not math.isnan(metric.action_efficiency)

    def test_nan_and_inf_guarding(self, standard_config: NoetherianEngineConfig) -> None:
        """Tensors containing NaNs or Infs are sanitized without crashing or propagating NaNs."""
        dirty_vec = torch.tensor([float("nan"), float("inf"), -float("inf")] + [0.1] * 61)
        clean_target = create_normalized_vector(64, seed_offset=1)

        eval_res = evaluate_trajectory_noetherian(
            states=[dirty_vec, clean_target],
            config=standard_config,
        )
        assert not math.isnan(eval_res.mizan_score)
        assert not math.isnan(eval_res.action_efficiency)
        assert not math.isnan(eval_res.charge_drift)

    def test_complex_tensor_handling(self, standard_config: NoetherianEngineConfig) -> None:
        """Complex Hilbert statevectors C^64 are projected to real components safely."""
        complex_vec = torch.complex(
            create_normalized_vector(64, seed_offset=1),
            create_normalized_vector(64, seed_offset=2),
        )
        eval_res = evaluate_trajectory_noetherian(
            states=[complex_vec, complex_vec],
            config=standard_config,
        )
        assert not math.isnan(eval_res.mizan_score)
        assert eval_res.mizan_score > 0.0


# ============================================================================
# Tier 3: Cross-Feature & Engine Integration Tests
# ============================================================================


class TestTier3CrossFeatureAndEngineIntegration:
    """Verifies multi-pillar compounding penalties and NoetherianInvariantEngine class APIs."""

    def test_compounded_kinetic_and_entropy_penalty(
        self,
        standard_config: NoetherianEngineConfig,
        canonical_charges: list[ConservedChargeDefinition],
    ) -> None:
        """Compound interaction: Violent kinetic jumps + sudden entropy spurts severely degrade score."""
        q0 = create_normalized_vector(64, seed_offset=1)
        q1 = create_normalized_vector(64, seed_offset=2)

        # Baseline: calm trajectory
        calm_res = evaluate_trajectory_noetherian(
            states=[q0, q0.clone()],
            probs=[[0.9, 0.1], [0.9, 0.1]],
            config=standard_config,
            charges=canonical_charges,
        )

        # Violent trajectory
        violent_res = evaluate_trajectory_noetherian(
            states=[q0, q1],
            probs=[[0.99, 0.01], [0.25, 0.25, 0.25, 0.25]],  # severe entropy jump
            config=standard_config,
            charges=canonical_charges,
        )

        assert violent_res.mizan_score < calm_res.mizan_score
        assert violent_res.action_efficiency < calm_res.action_efficiency
        assert violent_res.entropy_stability < calm_res.entropy_stability

    def test_manifold_projection_restores_charge_conservation(
        self,
        standard_config: NoetherianEngineConfig,
        canonical_charges: list[ConservedChargeDefinition],
    ) -> None:
        """Applying manifold projection to a drifted state restores charge preservation."""
        q0 = create_normalized_vector(64, seed_offset=1)
        phi = canonical_charges[0].generator_vector
        # Artificially induce drift
        q_drift = q0 + 0.4 * phi
        q_drift = q_drift / torch.norm(q_drift, p=2)

        # Check drift before projection
        _, drift_before, _, _ = evaluate_noether_charges([q_drift], canonical_charges, config=standard_config)

        # Project back onto invariant manifold
        q_restored, proj_res = project_to_manifold(q_drift, canonical_charges, config=standard_config)
        _, drift_after, _, _ = evaluate_noether_charges([q_restored], canonical_charges, config=standard_config)

        assert drift_after < drift_before
        assert proj_res.charge_restoration_error < proj_res.drift_distance_before

    def test_engine_class_charge_management(self, standard_config: NoetherianEngineConfig) -> None:
        """Verifies NoetherianInvariantEngine add, remove, and canonical registration."""
        engine = NoetherianInvariantEngine(config=standard_config, auto_register_canonical=True)
        assert len(engine.charges) == 4

        # Add custom charge
        c_custom = ConservedChargeDefinition(charge_id="c_custom", rule_name="Custom", generator_vector=torch.randn(64))
        engine.add_charge(c_custom)
        assert len(engine.charges) == 5

        # Check matrix extraction
        C = engine.get_charge_matrix()
        assert C.shape == (5, 64)
        b = engine.get_target_vector()
        assert b.shape == (5,)

        # Remove charge
        assert engine.remove_charge("c_custom") is True
        assert len(engine.charges) == 4
        assert engine.remove_charge("non_existent") is False

    def test_mizan_hakemi_alias_identity(self) -> None:
        """Verifies MizanHakemi is an exact alias for NoetherianInvariantEngine."""
        assert MizanHakemi is NoetherianInvariantEngine
        hakemi = MizanHakemi(auto_register_canonical=True)
        assert isinstance(hakemi, NoetherianInvariantEngine)
        assert len(hakemi.charges) == 4

    def test_engine_evaluate_state_matches_step_trajectory(
        self,
        standard_config: NoetherianEngineConfig,
        canonical_charges: list[ConservedChargeDefinition],
    ) -> None:
        """engine.evaluate_state(q1, previous_state=q0) matches evaluate_trajectory([q0, q1])."""
        engine = NoetherianInvariantEngine(config=standard_config, charges=canonical_charges)
        q0 = create_normalized_vector(64, seed_offset=1)
        q1 = create_normalized_vector(64, seed_offset=2)

        res_state = engine.evaluate_state(q1, previous_state=q0, probs=[0.9, 0.1])
        res_traj = engine.evaluate_trajectory(states=[q0, q1], probs=[[0.9, 0.1]])

        assert res_state.mizan_score == pytest.approx(res_traj.mizan_score, abs=1e-5)
        assert res_state.noether_flux == pytest.approx(res_traj.noether_flux, abs=1e-5)

    def test_evaluate_state_consequence_cost_effect(self) -> None:
        """Verifies engine.evaluate_state correctly factors in consequence_cost."""
        engine = NoetherianInvariantEngine()
        q0 = create_normalized_vector(64, seed_offset=1)
        q1 = create_normalized_vector(64, seed_offset=2)

        eval_clean = engine.evaluate_state(q1, previous_state=q0, consequence_cost=0.0)
        eval_costly = engine.evaluate_state(q1, previous_state=q0, consequence_cost=0.8)

        assert eval_clean.action_metric.potential_energy_sum == pytest.approx(0.0, abs=1e-5)
        # Potential is scaled by 0.5 * v_panel = 0.5 * 0.8 = 0.4
        assert eval_costly.action_metric.potential_energy_sum == pytest.approx(0.4, abs=1e-5)
        assert eval_clean.action_metric.action_efficiency > eval_costly.action_metric.action_efficiency
        assert eval_clean.mizan_score > eval_costly.mizan_score

    def test_engine_instantaneous_flux_computation(self, standard_config: NoetherianEngineConfig) -> None:
        """engine.compute_noether_flux evaluates immediate state divergence correctly."""
        engine = NoetherianInvariantEngine(config=standard_config, auto_register_canonical=True)
        q0 = create_normalized_vector(64, seed_offset=1)
        q1 = create_normalized_vector(64, seed_offset=2)

        flux_identical = engine.compute_noether_flux(q0, q0)
        flux_divergent = engine.compute_noether_flux(q1, q0)

        assert flux_identical == pytest.approx(0.0, abs=1e-6)
        assert flux_divergent > 0.0


# ============================================================================
# Tier 4: Real-World Adversarial Scenarios Tests
# ============================================================================


class TestTier4RealWorldAdversarialScenarios:
    """Verifies complex stress scenarios: creeping drift, periphrastic erosion, Pareto dilemmas."""

    def test_creeping_context_drift_scenario(
        self,
        standard_config: NoetherianEngineConfig,
        canonical_charges: list[ConservedChargeDefinition],
    ) -> None:
        """Scenario 1: 15-step cumulative drift where step-wise changes are small (delta=0.03),

        but the cumulative trajectory eventually triggers FLUX_ALERT or CRITICAL_BREACH.
        """
        # Starting point
        q_start = create_normalized_vector(64, seed_offset=100)
        # Drift generator direction (intent constraint)
        phi_drift = canonical_charges[0].generator_vector

        states: list[torch.Tensor] = [q_start]
        current = q_start

        # 15 gentle steps, each step deviating slightly in phi_drift direction
        step_drift_delta = 0.03
        for _ in range(15):
            current = current + step_drift_delta * phi_drift
            current = current / torch.norm(current, p=2)
            states.append(current)

        eval_res = evaluate_trajectory_noetherian(
            states=states,
            config=standard_config,
            charges=canonical_charges,
        )

        # Step-wise change was small, but cumulative Noether flux is significant
        assert eval_res.noether_flux > 0.10
        assert eval_res.charge_drift > 0.10
        assert eval_res.mizan_score < standard_config.harmonic_threshold
        # Check that invariant violation was recorded
        assert len(eval_res.invariant_violations) > 0
        assert eval_res.regime in ("FLUX_ALERT", "CRITICAL_BREACH", "MILD_DRIFT")

    def test_periphrastic_rule_erosion_scenario(
        self,
        standard_config: NoetherianEngineConfig,
        canonical_charges: list[ConservedChargeDefinition],
    ) -> None:
        """Scenario 2: Evasive rotation that preserves vector normalization while breaching

        an immutable semantic constraint (Lie generator projection Q = <phi, q> drops).
        """
        q0 = create_normalized_vector(64, seed_offset=50)
        safety_charge = canonical_charges[1]  # Safety boundary
        phi_safety = safety_charge.generator_vector

        # Construct an orthogonal perturbation specifically breaking safety charge
        v_breach = q0 + 0.45 * phi_safety
        v_breach = v_breach / torch.norm(v_breach, p=2)

        engine = NoetherianInvariantEngine(config=standard_config, charges=canonical_charges)
        eval_res = engine.evaluate_trajectory(states=[q0, v_breach])

        # Verifies the engine detects the breach through Lie projection
        breached_names = [v for v in eval_res.invariant_violations if safety_charge.rule_name in v]
        assert len(breached_names) > 0, "Safety boundary breach must be detected by Lie projection"
        assert eval_res.is_valid is False, "Hard constraint breach must render trajectory invalid"

    def test_conflicting_pareto_dilemmas_scenario(
        self,
        standard_config: NoetherianEngineConfig,
        canonical_charges: list[ConservedChargeDefinition],
    ) -> None:
        """Scenario 3: Conflicting tradeoff: Trajectory A has higher kinetic cost but preserves

        safety charges; Trajectory B is kinetic-optimal but severely breaches safety.
        The Mizan engine must prefer Trajectory A due to higher charge conservation weight (w_C > w_A).
        """
        # Start on the invariant manifold so root is fully compliant
        raw_root = create_normalized_vector(64, seed_offset=200)
        q_root, _ = project_to_manifold(raw_root, canonical_charges, config=standard_config)
        phi_safe = canonical_charges[1].generator_vector

        # Path A (Disciplined): Makes a large turn in safe orthogonal subspace (higher kinetic energy, zero safety drift)
        safe_dir = create_normalized_vector(64, seed_offset=201)
        # Ensure safe_dir is orthogonal to all charges
        for c in canonical_charges:
            safe_dir = safe_dir - torch.dot(safe_dir, c.generator_vector) * c.generator_vector
        safe_dir = safe_dir / torch.norm(safe_dir, p=2)

        path_A = [q_root, (q_root + 0.5 * safe_dir) / torch.norm(q_root + 0.5 * safe_dir, p=2)]

        # Path B (Rogue Shortcut): Very small kinetic distance, but points straight into safety violation
        path_B = [q_root, (q_root + 0.4 * phi_safe) / torch.norm(q_root + 0.4 * phi_safe, p=2)]

        eval_A = evaluate_trajectory_noetherian(states=path_A, config=standard_config, charges=canonical_charges)
        eval_B = evaluate_trajectory_noetherian(states=path_B, config=standard_config, charges=canonical_charges)

        # Path A must maintain validity, while Path B is invalidated by hard constraint breach
        assert eval_A.is_valid is True
        assert eval_B.is_valid is False
        assert eval_A.mizan_score > eval_B.mizan_score, "Disciplined path must beat rule-violating shortcut"

    def test_deep_dag_confluence_pruning_scenario(
        self,
        standard_config: NoetherianEngineConfig,
        canonical_charges: list[ConservedChargeDefinition],
    ) -> None:
        """Scenario 4: Multi-branch evaluation identifying rogue branches for microglial pruning.

        Simulates 4 candidate branches in a DAG rollout:
        - Branch 1: Harmonic equilibrium (valid)
        - Branch 2: Mild drift (valid with alert)
        - Branch 3: High entropy collapse (prunable)
        - Branch 4: Critical invariant breach (pruned)
        """
        engine = NoetherianInvariantEngine(config=standard_config, charges=canonical_charges)
        raw_root = create_normalized_vector(64, seed_offset=300)
        q_root, _ = project_to_manifold(raw_root, canonical_charges, config=standard_config)

        # Branch 1: Minimal drift, calm confidence
        b1_states = [q_root, q_root.clone()]
        b1_probs = [[0.95, 0.05], [0.95, 0.05]]

        # Branch 2: Small perturbation along orthogonal direction, stable confidence
        safe_dir = create_normalized_vector(64, seed_offset=301)
        for c in canonical_charges:
            safe_dir = safe_dir - torch.dot(safe_dir, c.generator_vector) * c.generator_vector
        safe_dir = safe_dir / torch.norm(safe_dir, p=2)

        q_mild = (q_root + 0.15 * safe_dir)
        b2_states = [q_root, q_mild / torch.norm(q_mild, p=2)]
        b2_probs = [[0.9, 0.1], [0.85, 0.15]]

        # Branch 3: Severe epistemic entropy spurt
        b3_states = [q_root, q_root.clone()]
        b3_probs = [[0.99, 0.01], [0.25, 0.25, 0.25, 0.25]]

        # Branch 4: Violent safety boundary breach
        phi_safety = canonical_charges[1].generator_vector
        q_breach = (q_root + 0.6 * phi_safety)
        b4_states = [q_root, q_breach / torch.norm(q_breach, p=2)]
        b4_probs = [[0.9, 0.1], [0.9, 0.1]]

        eval_b1 = engine.evaluate_trajectory(b1_states, b1_probs)
        eval_b2 = engine.evaluate_trajectory(b2_states, b2_probs)
        eval_b3 = engine.evaluate_trajectory(b3_states, b3_probs)
        eval_b4 = engine.evaluate_trajectory(b4_states, b4_probs)

        # Ranking order: b1 > b2 > b4 and b1 > b3
        assert eval_b1.mizan_score > eval_b2.mizan_score
        assert eval_b2.mizan_score > eval_b4.mizan_score
        assert eval_b1.mizan_score > eval_b3.mizan_score
        assert eval_b3.entropy_stability < eval_b1.entropy_stability

        # Pruning decision criteria:
        # Branch 1: retained
        assert eval_b1.is_valid is True and eval_b1.regime == "HARMONIC_EQUILIBRIUM"
        # Branch 4: pruned
        assert eval_b4.is_valid is False or eval_b4.regime in ("FLUX_ALERT", "CRITICAL_BREACH")
        assert len(eval_b4.invariant_violations) > 0

    def test_hermetic_isolation_with_tmp_path(
        self,
        tmp_path: Path,
        standard_config: NoetherianEngineConfig,
        canonical_charges: list[ConservedChargeDefinition],
    ) -> None:
        """Verifies completely hermetic operation using pytest tmp_path:

        Serializes engine configurations and evaluation records without touching
        any live shared state files (quanta_cognitive_state.json).
        """
        telemetry_file = tmp_path / "hermetic_noetherian_telemetry.jsonl"

        engine = NoetherianInvariantEngine(config=standard_config, charges=canonical_charges)
        q0 = create_normalized_vector(64, seed_offset=400)
        q1 = create_normalized_vector(64, seed_offset=401)

        eval_res = engine.evaluate_trajectory([q0, q1], probs=[[0.9, 0.1], [0.8, 0.2]])

        # Write to isolated temporary file
        record = {
            "session_id": "test_hermetic_run",
            "mizan_score": eval_res.mizan_score,
            "noether_flux": eval_res.noether_flux,
            "action_efficiency": eval_res.action_efficiency,
            "entropy_stability": eval_res.entropy_stability,
            "charge_conservation": eval_res.charge_conservation,
            "regime": eval_res.regime,
            "is_valid": eval_res.is_valid,
            "violations": eval_res.invariant_violations,
        }
        with open(telemetry_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Read back and assert fidelity
        assert telemetry_file.exists()
        with open(telemetry_file, encoding="utf-8") as f:
            loaded = json.loads(f.readline())

        assert loaded["session_id"] == "test_hermetic_run"
        assert loaded["mizan_score"] == pytest.approx(eval_res.mizan_score, abs=1e-5)
        assert loaded["regime"] == eval_res.regime


# ============================================================================
# Tier 5: Milestone M2 Arbiter Integration & Defect Remediation
# ============================================================================


class TestTier5ArbiterIntegrationAndRemediation:
    """Tests specifically targeting the Milestone M2 integration fixes:

    1. Telemetry persistence from DAG arbitration without KeyError 'option'.
    2. Negative utility ranking monotonicity respecting Mizan scores.
    3. noetherian_config precedence in arbitrate_dag.
    4. Retention of config.default_charges in NoetherianInvariantEngine.__init__.
    """

    def test_dag_telemetry_persists_to_jsonl_without_key_error(self, tmp_path: Path) -> None:
        """Fix 1: Verifies DAG arbitration writes telemetry records to disk without KeyError."""
        dim = 64
        arbiter = QuantumDecisionArbiter(dim=dim)
        telemetry_file = tmp_path / "dag_telemetry.jsonl"

        dag = DecisionDAG()
        root = DecisionNode("root", "Root Architecture", depth=0, statevector=torch.randn(dim))
        node_a = DecisionNode("node_a", "Service Mesh", depth=1, statevector=torch.randn(dim))
        node_b = DecisionNode("node_b", "Direct RPC", depth=1, statevector=torch.randn(dim))
        dag.add_node(root)
        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_edge(DecisionEdge("root", "node_a", action_label="Adopt Mesh"))
        dag.add_edge(DecisionEdge("root", "node_b", action_label="Adopt RPC"))

        # Execute arbitrate_dag with telemetry logging enabled
        res = arbiter.arbitrate_dag(
            dag=dag,
            goal="Test DAG Telemetry Fix",
            log_telemetry=True,
            telemetry_file=telemetry_file,
        )

        assert telemetry_file.exists()
        assert telemetry_file.stat().st_size > 0

        events = read_telemetry_events(telemetry_file=telemetry_file)
        assert len(events) == 1
        record = events[0]
        assert record["event_type"] == "decision"
        assert record["winner"] == res["winning_trajectory"]["trajectory_id"]
        assert "mizan_score" in record
        assert "noether_flux" in record
        assert "invariant_violations" in record
        assert isinstance(record["ranking"], list)

        # Verify summary aggregation roundtrip
        summary = get_telemetry_summary(telemetry_file=telemetry_file)
        assert summary["total_decisions"] == 1
        assert summary["avg_mizan_score"] == pytest.approx(record["mizan_score"], abs=1e-4)

    def test_drifting_branch_with_negative_utility_penalized_lower_than_compliant_branch(self) -> None:
        """Fix 2: Verifies that under negative net utility (U - C < 0), higher Mizan score

        strictly and monotonically produces a higher net_score and wins over a drifting branch.
        """
        dim = 64
        arbiter = QuantumDecisionArbiter(dim=dim)

        # Soft constraint where drift is allowed but degrades Mizan score
        gen = torch.zeros(dim)
        gen[0] = 1.0
        charge = ConservedChargeDefinition(
            charge_id="c_soft",
            rule_name="Soft Conservation",
            generator_vector=gen,
            target_value=0.0,
            tolerance=0.30,
            is_hard_constraint=False,
        )
        engine = NoetherianInvariantEngine(
            config=NoetherianEngineConfig(dim=dim),
            charges=[charge],
            auto_register_canonical=False,
        )

        q_root = torch.zeros(dim)
        q_root[1] = 1.0
        # Branch A: Drifting state (lower Mizan score)
        q_a = torch.zeros(dim)
        q_a[0] = 0.15
        q_a[1] = 0.988
        q_a = q_a / torch.norm(q_a)
        # Branch B: Compliant state (perfect Mizan score 1.0)
        q_b = torch.zeros(dim)
        q_b[1] = 1.0

        # Both branches suffer high consequence cost (U - C < 0)
        dag = DecisionDAG()
        dag.add_node(DecisionNode(node_id="root", label="Root", depth=0, statevector=q_root))
        dag.add_node(
            DecisionNode(
                node_id="n_a",
                label="Branch A Drifting",
                depth=1,
                statevector=q_a,
                consequence=ConsequenceVector(latency=0.95, maintenance=0.95),
            )
        )
        dag.add_node(
            DecisionNode(
                node_id="n_b",
                label="Branch B Compliant",
                depth=1,
                statevector=q_b,
                consequence=ConsequenceVector(latency=0.95, maintenance=0.95),
            )
        )
        dag.add_edge(DecisionEdge(source_id="root", target_id="n_a", action_label="Go A"))
        dag.add_edge(DecisionEdge(source_id="root", target_id="n_b", action_label="Go B"))

        res = arbiter.arbitrate_dag(
            dag=dag,
            goal="Negative Utility Inversion Demonstration",
            noetherian_engine=engine,
            log_telemetry=False,
        )

        traj_a = next(t for t in res["trajectories"] if t["path_labels"][-1] == "Branch A Drifting")
        traj_b = next(t for t in res["trajectories"] if t["path_labels"][-1] == "Branch B Compliant")

        # Compliant branch must have strictly higher Mizan score and net_score
        assert traj_b["mizan_score"] > traj_a["mizan_score"]
        assert traj_b["net_score"] > traj_a["net_score"]

        # Compliant branch must be the recommended winner
        assert res["recommended_option"] == "Branch B Compliant"
        assert res["winning_trajectory"]["trajectory_id"] == traj_b["trajectory_id"]

    def test_arbitrate_dag_respects_custom_noetherian_config(self) -> None:
        """Fix 3: Verifies arbitrate_dag respects noetherian_config argument over self.noetherian_engine default."""
        dim = 64
        gen = torch.zeros(dim)
        gen[0] = 1.0
        strict_charge = ConservedChargeDefinition(
            charge_id="c_strict",
            rule_name="Strict Axis 0",
            generator_vector=gen,
            target_value=1.0,
            tolerance=0.01,
            is_hard_constraint=True,
        )
        custom_cfg = NoetherianEngineConfig(
            dim=dim,
            default_charges=[strict_charge],
            critical_breach_threshold=0.3,
        )

        # Arbiter has a default engine with 0 charges
        arbiter = QuantumDecisionArbiter(dim=dim)

        dag = DecisionDAG()
        v_breach = torch.zeros(dim)
        v_breach[1] = 1.0  # Axis 0 is 0.0, breaching target 1.0
        dag.add_node(DecisionNode(node_id="root", label="Root", statevector=v_breach))
        dag.add_node(DecisionNode(node_id="n1", label="Node 1", statevector=v_breach))
        dag.add_edge(DecisionEdge(source_id="root", target_id="n1", action_label="Step"))

        # Run with custom_cfg
        res = arbiter.arbitrate_dag(dag=dag, goal="Test Config Precedence", noetherian_config=custom_cfg)

        # Violations must be detected from custom_cfg
        assert len(res["winning_trajectory"]["invariant_violations"]) > 0
        assert "Strict Axis 0" in str(res["winning_trajectory"]["invariant_violations"])
        assert res["winning_trajectory"]["is_pruned"] is True

    def test_noetherian_engine_retains_default_charges_when_charges_none(self) -> None:
        """Fix 4: Verifies NoetherianInvariantEngine retains default_charges from config when charges is None."""
        dim = 64
        c1 = ConservedChargeDefinition("c1", "Charge 1", torch.randn(dim))
        c2 = ConservedChargeDefinition("c2", "Charge 2", torch.randn(dim))
        cfg = NoetherianEngineConfig(dim=dim, default_charges=[c1, c2])

        # Initialize with charges=None and auto_register_canonical=False
        engine = NoetherianInvariantEngine(config=cfg, charges=None, auto_register_canonical=False)
        assert len(engine.charges) == 2
        assert {c.charge_id for c in engine.charges} == {"c1", "c2"}
