"""Empirical Challenger 2 Test Suite for Milestone M1 (Noetherian Invariant Engine).

This test suite empirically tests boundary cases, extreme probability distributions,
trajectory length scalings ($N=1, 2, 50$), and hard constraint breach detection
against `quanta/cognitive/noetherian.py`.

Findings Demonstrated:
1. Extreme probability distributions (Dirac delta, uniform, unnormalized, near-zero 1e-30)
   behave stably with zero NaNs and zero infinities.
2. Trajectory evaluation across N in {1, 2, 50} produces Mizan scores strictly bounded in [0.0, 1.0].
3. BUG CONFIRMED: Initial state (step 0) invariant breaches in trajectories of length N >= 2
   are completely missed by `evaluate_noether_charges`, resulting in `is_valid=True`,
   empty violations, and zero charge drift.
4. BUG CONFIRMED: Substring collisions in rule name matching in `evaluate_trajectory_noetherian`
   falsely invalidate trajectories when a soft constraint rule name contains a hard constraint
   rule name as a substring.
"""

from __future__ import annotations

import math

import pytest
import torch

from quanta.cognitive.noetherian import (
    ConservedChargeDefinition,
    NoetherianEngineConfig,
    evaluate_epistemic_entropy,
    evaluate_trajectory_noetherian,
)


class TestExtremeProbabilityDistributions:
    """Requirement 1: Empirical stress-testing of epistemic entropy under extreme probability inputs."""

    @pytest.mark.parametrize(
        "probs,expected_initial_range",
        [
            # Dirac delta distributions: should yield ~0 normalized entropy
            ([[1.0, 0.0]], (0.0, 1e-5)),
            ([[1.0] + [0.0] * 99], (0.0, 1e-5)),
            ([[0.0] * 50 + [1.0] + [0.0] * 49], (0.0, 1e-5)),
            # Pure uniform distributions: should yield 1.0 normalized entropy
            ([[0.5, 0.5]], (0.9999, 1.0001)),
            ([[0.25] * 4], (0.9999, 1.0001)),
            ([[1.0 / 64.0] * 64], (0.9999, 1.0001)),
            # Unnormalized probabilities: should auto-normalize
            ([[10.0, 50.0]], (0.64, 0.66)),
            ([[1e10, 5e10]], (0.64, 0.66)),
            ([[1e-10, 5e-10]], (0.64, 0.66)),
            # Near-zero values (1e-30)
            ([[1e-30, 1e-30]], (0.9999, 1.0001)),  # fallback to uniform when sum <= 1e-12
            ([[1.0, 1e-30]], (0.0, 1e-5)),         # dominated by 1.0
            ([[1e-30] * 10], (0.9999, 1.0001)),    # uniform fallback
            ([[0.0, 0.0, 0.0]], (0.9999, 1.0001)), # all zero fallback
            # NaNs and Infs in distribution
            ([[float("nan"), 1.0]], (0.0, 1e-5)),
            ([[float("inf"), 1.0]], (0.0, 1e-5)),
            # Scalar probabilities
            ([0.7], (0.87, 0.89)),
            ([0.0], (0.0, 1e-4)),
            ([1.0], (0.0, 1e-4)),
        ],
    )
    def test_probability_distribution_boundary_cases(
        self, probs: list[any], expected_initial_range: tuple[float, float]
    ) -> None:
        """Confirms entropy calculation produces finite values within expected range without NaNs or Infs."""
        config = NoetherianEngineConfig()
        result = evaluate_epistemic_entropy(probs, config)

        assert not math.isnan(result.initial_entropy), "initial_entropy must not be NaN"
        assert not math.isinf(result.initial_entropy), "initial_entropy must not be Inf"
        assert not math.isnan(result.final_entropy), "final_entropy must not be NaN"
        assert not math.isinf(result.final_entropy), "final_entropy must not be Inf"
        assert not math.isnan(result.entropy_stability_score), "entropy_stability_score must not be NaN"
        assert 0.0 <= result.entropy_stability_score <= 1.0, "stability score must be in [0, 1]"

        low, high = expected_initial_range
        assert low <= result.initial_entropy <= high, (
            f"Expected initial entropy in [{low}, {high}], got {result.initial_entropy}"
        )


class TestTrajectoryMizanScoreBounds:
    """Requirement 2: Empirical testing of trajectory evaluation with N=1, 2, 50 steps."""

    @pytest.mark.parametrize("n_steps", [1, 2, 50])
    @pytest.mark.parametrize(
        "state_mode", ["random", "stationary", "antiparallel", "zeros", "nan_inf"]
    )
    def test_mizan_score_strictly_bounded_across_lengths(
        self, n_steps: int, state_mode: str
    ) -> None:
        """Confirms Mizan score is strictly bounded in [0.0, 1.0] across trajectory lengths and conditions."""
        config = NoetherianEngineConfig()

        # Generate state sequence according to mode
        if state_mode == "random":
            torch.manual_seed(42 + n_steps)
            states = [torch.randn(64) for _ in range(n_steps)]
            states = [s / torch.norm(s) for s in states]
        elif state_mode == "stationary":
            s = torch.randn(64)
            s = s / torch.norm(s)
            states = [s.clone() for _ in range(n_steps)]
        elif state_mode == "antiparallel":
            s = torch.randn(64)
            s = s / torch.norm(s)
            states = [s * ((-1) ** i) for i in range(n_steps)]
        elif state_mode == "zeros":
            states = [torch.zeros(64) for _ in range(n_steps)]
        elif state_mode == "nan_inf":
            states = []
            for i in range(n_steps):
                s = torch.randn(64)
                if i % 2 == 0:
                    s[0] = float("nan")
                else:
                    s[1] = float("inf")
                states.append(s)

        # Test with diverse probability patterns
        prob_patterns = [
            None,
            [[1.0, 0.0] for _ in range(n_steps)],
            [[0.25] * 4 for _ in range(n_steps)],
            [[10.0, 50.0] for _ in range(n_steps)],
            [[1e-30, 1e-30] for _ in range(n_steps)],
        ]

        cost_patterns = [
            None,
            [0.0] * n_steps,
            [1.0] * n_steps,
            [1e6] * n_steps,
            [-50.0] * n_steps,
        ]

        for probs in prob_patterns:
            for costs in cost_patterns:
                eval_res = evaluate_trajectory_noetherian(
                    states=states,
                    probs=probs,
                    config=config,
                    consequence_costs=costs,
                )

                score = eval_res.mizan_score
                assert not math.isnan(score), f"mizan_score is NaN for N={n_steps}, mode={state_mode}"
                assert not math.isinf(score), f"mizan_score is Inf for N={n_steps}, mode={state_mode}"
                assert 0.0 <= score <= 1.0, (
                    f"mizan_score {score} outside [0.0, 1.0] for N={n_steps}, mode={state_mode}"
                )


class TestHardConstraintBreachDetection:
    """Requirement 3: Empirical verification of hard constraint breach gating and failure mode exposure."""

    def test_terminal_hard_constraint_breach_invalidates(self) -> None:
        """Verifies that a hard constraint breach at trajectory terminus correctly sets is_valid=False."""
        gen = torch.zeros(64)
        gen[0] = 1.0
        hard_charge = ConservedChargeDefinition(
            charge_id="hard_safety",
            rule_name="Safety Rule",
            generator_vector=gen,
            target_value=0.0,
            tolerance=0.05,
            is_hard_constraint=True,
        )

        config = NoetherianEngineConfig(
            default_charges=[hard_charge],
            critical_breach_threshold=0.10,
        )

        # Terminal state breaches hard constraint
        q_valid = torch.zeros(64)
        q_valid[1] = 1.0

        q_breached = torch.zeros(64)
        q_breached[0] = 1.0  # Q = 1.0 >> 0.05

        eval_res = evaluate_trajectory_noetherian(
            states=[q_valid, q_breached],
            config=config,
            charges=[hard_charge],
        )

        assert eval_res.is_valid is False
        assert len(eval_res.invariant_violations) > 0
        assert "Safety Rule" in eval_res.invariant_violations[0]

    def test_bug_step_0_breach_blindspot_in_multi_step_trajectory(self) -> None:
        """DEMONSTRATES BUG 1: Step 0 invariant breach in multi-step trajectories (N >= 2) is ignored.

        In `evaluate_noether_charges`, when len(path_states) >= 2, the loop runs:
            `for k in range(1, len(sanitized_states)):`
        Step 0 (q_0) is NEVER checked for tolerance breaches, nor is its drift recorded in
        `max_charge_drift`. If q_0 breaches a hard invariant and q_1 is valid, the engine
        incorrectly yields `is_valid=True` and `invariant_violations=[]`.
        """
        gen = torch.zeros(64)
        gen[0] = 1.0
        hard_charge = ConservedChargeDefinition(
            charge_id="hard_safety",
            rule_name="Hard Invariant",
            generator_vector=gen,
            target_value=0.0,
            tolerance=0.05,
            is_hard_constraint=True,
        )

        config = NoetherianEngineConfig(
            default_charges=[hard_charge],
            critical_breach_threshold=0.10,
        )

        q_breached = torch.zeros(64)
        q_breached[0] = 1.0  # Drift = 1.0 >> 0.05

        q_valid = torch.zeros(64)
        q_valid[1] = 1.0    # Drift = 0.0 <= 0.05

        # Single state: correctly detected as breached
        eval_single = evaluate_trajectory_noetherian([q_breached], config=config, charges=[hard_charge])
        assert eval_single.is_valid is False
        assert len(eval_single.invariant_violations) == 1

        # Multi-step N=2: q_0 breached, q_1 valid.
        # The engine must reject this (is_valid=False) because q_0 breached a hard invariant.
        eval_two = evaluate_trajectory_noetherian([q_breached, q_valid], config=config, charges=[hard_charge])

        # VERIFIED RESOLUTION:
        # Step 0 breach is now properly detected in multi-step trajectories (N >= 2)
        assert eval_two.is_valid is False
        assert len(eval_two.invariant_violations) == 1
        assert "Hard Invariant" in eval_two.invariant_violations[0]
        assert "step 0" in eval_two.invariant_violations[0]
        assert eval_two.charge_drift == pytest.approx(1.0)

    def test_bug_substring_matching_false_invalidation(self) -> None:
        """VERIFIES BUG 2 RESOLUTION: Substring collision in rule name matching does not falsely invalidate.

        If a soft constraint is named "Safe Boundary Policy" (breached) and a hard constraint
        is named "Safe" (unbreached), zero-collision matching guarantees the soft rule breach
        does not falsely invalidate the trajectory.
        """
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
            is_hard_constraint=False,  # SOFT CONSTRAINT
        )

        hard_rule = ConservedChargeDefinition(
            charge_id="hard_rule",
            rule_name="Safe",
            generator_vector=gen_hard,
            target_value=0.0,
            tolerance=0.05,
            is_hard_constraint=True,   # HARD CONSTRAINT
        )

        config = NoetherianEngineConfig(critical_breach_threshold=0.10)

        # State that breaches ONLY the soft rule, hard rule is completely satisfied
        q_test = torch.zeros(64)
        q_test[0] = 1.0
        q_test[1] = 0.0

        eval_res = evaluate_trajectory_noetherian([q_test], config=config, charges=[soft_rule, hard_rule])

        # The hard rule was NOT breached (q_test[1] == 0.0). Only the soft rule breached.
        # Mizan score is well above critical_breach_threshold (0.10).
        # Multi-tiered zero-collision rule matching guarantees is_valid is True.
        assert eval_res.is_valid is True
        assert len(eval_res.invariant_violations) == 1
        assert "Safe Boundary Policy" in eval_res.invariant_violations[0]
