"""Empirical Challenger 1 Test Suite for Milestone M1 Iteration 2 (Noetherian Invariant Engine).

This test suite rigorously evaluates:
1. Consequence cost dual-mode indexing across trajectory lengths T in {1, 2, 5, 20}
   under transition-indexed, state-indexed, and scalar broadcast configurations against analytical oracles.
2. Step 0 (initial state q_0) drift stress harness across 100 randomized trajectories
   with invariant breaches followed by compliant transitions, verifying 100% detection rate and invalidation.
3. Edge cases and adversarial inputs: extreme costs, NaNs, Infs, partial/oversized sequences,
   soft vs hard rule isolation at step 0, and single-state static rollouts.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from quanta.cognitive.noetherian import (
    ConservedChargeDefinition,
    NoetherianEngineConfig,
    NoetherianInvariantEngine,
    evaluate_trajectory_action,
    evaluate_trajectory_noetherian,
)


class TestConsequenceCostDualModeIndexing:
    """Requirement 1: Empirical stress-testing of consequence cost dual-mode indexing across T in {1, 2, 5, 20}."""

    @pytest.mark.parametrize("T", [1, 2, 5, 20])
    @pytest.mark.parametrize(
        "mode",
        ["transition_indexed", "state_indexed", "scalar_float", "scalar_list"],
    )
    def test_consequence_cost_dual_mode_across_t(self, T: int, mode: str) -> None:
        """Verifies exact potential energy and action efficiency for T transitions under all 3 indexing modes."""
        dim = 64
        config = NoetherianEngineConfig(dim=dim, action_scale=1.0)

        # Generate orthogonal state sequence of length T + 1: [q_0, ..., q_T]
        states: list[torch.Tensor] = []
        for k in range(T + 1):
            q = torch.zeros(dim, dtype=torch.float32)
            q[k % dim] = 1.0
            states.append(q)

        # Calculate baseline kinetic energy for validation:
        # For orthogonal unit vectors, ||q_k - q_{k-1}||^2 = 2.0, so T_k = 0.5 * 2.0 = 1.0
        expected_kinetic_sum = 1.0 * T

        if mode == "transition_indexed":
            # Length T sequence: index k - 1 corresponds to transition k
            costs = [min(1.0, 0.04 * (i + 1)) for i in range(T)]
            expected_potential_sum = sum(0.5 * min(1.0, max(0.0, c)) for c in costs)
        elif mode == "state_indexed":
            # Length T + 1 sequence: index k corresponds to arrival state q_k; index 0 is q_0
            c_arrivals = [min(1.0, 0.04 * (i + 1)) for i in range(T)]
            costs = [0.99] + c_arrivals  # index 0 is arbitrary and must not contribute to potential
            expected_potential_sum = sum(0.5 * min(1.0, max(0.0, c)) for c in c_arrivals)
        elif mode == "scalar_float":
            # Float scalar broadcast uniformly across all T transitions
            costs = 0.65
            expected_potential_sum = T * 0.5 * 0.65
        elif mode == "scalar_list":
            # Length-1 list broadcast uniformly across all T transitions
            costs = [0.65]
            expected_potential_sum = T * 0.5 * 0.65
        else:
            pytest.fail(f"Unknown mode {mode}")

        res = evaluate_trajectory_action(path_states=states, consequence_costs=costs, config=config)

        # 1. Kinetic energy matches analytical oracle
        assert res.kinetic_energy_sum == pytest.approx(expected_kinetic_sum, abs=1e-5), (
            f"T={T} mode={mode}: kinetic energy {res.kinetic_energy_sum} != {expected_kinetic_sum}"
        )

        # 2. Potential energy from consequence cost matches exact analytical oracle
        assert res.potential_energy_sum == pytest.approx(expected_potential_sum, abs=1e-5), (
            f"T={T} mode={mode}: potential energy {res.potential_energy_sum} != {expected_potential_sum}"
        )

        # 3. Discrete action S_d = kinetic - potential
        assert res.discrete_action == pytest.approx(
            expected_kinetic_sum - expected_potential_sum, abs=1e-5
        )

        # 4. Action efficiency = exp(-mean_cost / sigma_A^2)
        mean_cost = (expected_kinetic_sum + expected_potential_sum) / T
        expected_efficiency = math.exp(-mean_cost / (config.action_scale**2))
        assert res.action_efficiency == pytest.approx(expected_efficiency, abs=1e-5)

    @pytest.mark.parametrize("T", [1, 2, 5, 20])
    @pytest.mark.parametrize("dummy_q0_cost", [0.0, 0.5, 1.0, 100.0, -50.0])
    def test_state_indexed_ignores_initial_state_cost(self, T: int, dummy_q0_cost: float) -> None:
        """Confirms that index 0 in state-indexed consequence_costs has strictly zero contribution to potential."""
        dim = 64
        states = [torch.zeros(dim) for _ in range(T + 1)]
        for k in range(T + 1):
            states[k][k % dim] = 1.0

        # Uniform arrival costs of 0.4
        c_arrivals = [0.4] * T
        costs = [dummy_q0_cost] + c_arrivals

        res = evaluate_trajectory_action(path_states=states, consequence_costs=costs)
        expected_v = T * 0.5 * 0.4
        assert res.potential_energy_sum == pytest.approx(expected_v, abs=1e-5)

    @pytest.mark.parametrize("cost", [0.0, 0.1, 0.25, 0.5, 0.8, 1.0])
    def test_evaluate_state_consequence_cost_effect_and_monotonicity(self, cost: float) -> None:
        """Verifies evaluate_state correctly applies consequence cost in single-transition mode."""
        engine = NoetherianInvariantEngine()
        q0 = torch.zeros(64)
        q0[0] = 1.0
        q1 = torch.zeros(64)
        q1[1] = 1.0

        eval_res = engine.evaluate_state(state=q1, previous_state=q0, consequence_cost=cost)

        # In evaluate_state, previous_state to state is T=1 transition
        expected_potential = 0.5 * cost
        assert eval_res.action_metric.potential_energy_sum == pytest.approx(expected_potential, abs=1e-5)

        # As consequence cost increases, action efficiency must monotonically decrease
        expected_eff = math.exp(-(1.0 + expected_potential) / 1.0)
        assert eval_res.action_efficiency == pytest.approx(expected_eff, abs=1e-5)

    def test_consequence_cost_boundary_clamping(self) -> None:
        """Verifies boundary clamping of negative, over-range, NaN, and Inf consequence costs."""
        dim = 64
        states = [torch.zeros(dim) for _ in range(5)]  # T = 4 transitions
        for k in range(5):
            states[k][k] = 1.0

        adversarial_costs = [-1.5, 2.5, float("nan"), float("inf")]
        # Clamped:
        # k=1: -1.5 -> 0.0 => V = 0.0
        # k=2: 2.5 -> 1.0 => V = 0.5
        # k=3: nan -> 0.0 => V = 0.0
        # k=4: inf -> 0.0 => V = 0.0
        # Total potential = 0.5
        res = evaluate_trajectory_action(path_states=states, consequence_costs=adversarial_costs)
        assert res.potential_energy_sum == pytest.approx(0.5, abs=1e-5)

    def test_consequence_cost_empty_and_none(self) -> None:
        """Verifies that None, empty list, and 0.0 costs produce zero potential energy without error."""
        states = [torch.zeros(64) for _ in range(3)]
        for k in range(3):
            states[k][k] = 1.0

        for empty_val in [None, [], 0.0, [0.0]]:
            res = evaluate_trajectory_action(path_states=states, consequence_costs=empty_val)
            assert res.potential_energy_sum == pytest.approx(0.0, abs=1e-5)

    def test_consequence_cost_undersized_and_oversized_sequences(self) -> None:
        """Verifies safe handling when cost sequence length does not match T or T+1."""
        states = [torch.zeros(64) for _ in range(6)]  # T = 5 transitions
        for k in range(6):
            states[k][k] = 1.0

        # Undersized: len = 2 < 5
        # Steps 1 and 2 receive costs, steps 3..5 are out of range and safely ignored
        undersized_costs = [0.3, 0.7]
        res_under = evaluate_trajectory_action(path_states=states, consequence_costs=undersized_costs)
        assert res_under.potential_energy_sum == pytest.approx(0.5 * (0.3 + 0.7), abs=1e-5)

        # Oversized: len = 10 > 6
        # Treated as state-indexed (len >= 6), uses indices k=1..5
        oversized_costs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        res_over = evaluate_trajectory_action(path_states=states, consequence_costs=oversized_costs)
        expected_v = 0.5 * (0.1 + 0.2 + 0.3 + 0.4 + 0.5)
        assert res_over.potential_energy_sum == pytest.approx(expected_v, abs=1e-5)

    def test_tensor_input_consequence_costs(self) -> None:
        """Verifies evaluate_trajectory_action accepts PyTorch 1D tensors as consequence costs."""
        states = [torch.zeros(64) for _ in range(4)]
        for k in range(4):
            states[k][k] = 1.0

        costs_tensor = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float32)
        res = evaluate_trajectory_action(path_states=states, consequence_costs=costs_tensor)
        assert res.potential_energy_sum == pytest.approx(0.5 * (0.2 + 0.4 + 0.6), abs=1e-5)


class TestStep0InitialStateDriftStressHarness:
    """Requirement 2: Empirical stress-testing of initial state (q_0) drift across 100 randomized trajectories."""

    @pytest.fixture
    def canonical_test_setup(self) -> tuple[NoetherianEngineConfig, list[ConservedChargeDefinition]]:
        """Sets up a standardized suite of hard and soft invariant charges."""
        dim = 64
        gen1 = torch.zeros(dim)
        gen1[0] = 1.0
        hard1 = ConservedChargeDefinition(
            charge_id="hard_safety_1",
            rule_name="Operational Safety Boundary",
            generator_vector=gen1,
            target_value=0.0,
            tolerance=0.04,
            is_hard_constraint=True,
        )

        gen2 = torch.zeros(dim)
        gen2[1] = 1.0
        hard2 = ConservedChargeDefinition(
            charge_id="hard_intent_2",
            rule_name="Core Intent Invariant",
            generator_vector=gen2,
            target_value=0.0,
            tolerance=0.05,
            is_hard_constraint=True,
        )

        gen_soft = torch.zeros(dim)
        gen_soft[2] = 1.0
        soft = ConservedChargeDefinition(
            charge_id="soft_coherence_3",
            rule_name="Contextual Coherence Buffer",
            generator_vector=gen_soft,
            target_value=0.0,
            tolerance=0.08,
            is_hard_constraint=False,
        )

        charges = [hard1, hard2, soft]
        config = NoetherianEngineConfig(
            dim=dim,
            default_charges=charges,
            critical_breach_threshold=0.30,
        )
        return config, charges

    def test_100_trajectories_random_q0_breaches(
        self, canonical_test_setup: tuple[NoetherianEngineConfig, list[ConservedChargeDefinition]]
    ) -> None:
        """Generates 100 randomized trajectories with q_0 breaches and valid steps; asserts 100% detection rate."""
        config, charges = canonical_test_setup
        dim = config.dim

        torch.manual_seed(42)
        random.seed(42)

        hard_charges = [c for c in charges if c.is_hard_constraint]
        trajectory_lengths = [1, 2, 3, 5, 8, 12, 20, 50]

        detected_step_0_count = 0
        is_valid_false_count = 0
        total_trajectories = 100

        for traj_idx in range(total_trajectories):
            T = trajectory_lengths[traj_idx % len(trajectory_lengths)]
            target_charge = hard_charges[traj_idx % len(hard_charges)]
            breach_coord = 0 if target_charge.charge_id == "hard_safety_1" else 1

            # 1. Synthesize breached initial state q_0
            q0 = torch.zeros(dim, dtype=torch.float32)
            # Drift magnitude between 2.0x and 6.0x tolerance
            drift_mag = random.uniform(target_charge.tolerance * 2.0, target_charge.tolerance * 6.0)
            sign = 1.0 if (traj_idx % 2 == 0) else -1.0
            q0[breach_coord] = sign * drift_mag
            # Add orthogonal random noise
            q0[10:] = torch.randn(dim - 10) * 0.05
            # Retract to unit sphere
            q0 = q0 / torch.norm(q0, p=2)

            # Ensure drift persists after normalization
            while abs(q0[breach_coord].item() - target_charge.target_value) <= target_charge.tolerance:
                q0[breach_coord] *= 1.5
                q0 = q0 / torch.norm(q0, p=2)

            actual_drift = abs(q0[breach_coord].item() - target_charge.target_value)
            assert actual_drift > target_charge.tolerance, "Synthetic q_0 must breach tolerance"

            # 2. Synthesize subsequent compliant states q_1 ... q_T
            states = [q0]
            for _k in range(1, T + 1):
                qk = torch.zeros(dim, dtype=torch.float32)
                # Keep charge projection coordinates strictly at zero
                qk[0] = 0.0
                qk[1] = 0.0
                qk[2] = 0.0
                # Active subspace orthogonal to generators
                qk[10:] = torch.randn(dim - 10)
                qk = qk / torch.norm(qk, p=2)
                states.append(qk)

            # 3. Evaluate multi-step trajectory
            eval_res = evaluate_trajectory_noetherian(states, config=config, charges=charges)

            # 4. Check step 0 breach detection
            has_step_0_violation = any(
                getattr(v, "step", None) == 0 or "step 0:" in str(v)
                for v in eval_res.invariant_violations
            )

            if has_step_0_violation:
                detected_step_0_count += 1

            if not eval_res.is_valid:
                is_valid_false_count += 1

            # 5. Invariant assertions per trajectory
            assert has_step_0_violation, (
                f"Trajectory #{traj_idx} (T={T}): Failed to record step 0 breach in violations! "
                f"Violations: {eval_res.invariant_violations}"
            )
            assert not eval_res.is_valid, (
                f"Trajectory #{traj_idx} (T={T}): is_valid must be False due to step 0 hard breach!"
            )
            assert eval_res.charge_drift >= (actual_drift - 1e-4), (
                f"Trajectory #{traj_idx}: charge_drift ({eval_res.charge_drift}) "
                f"did not record step 0 drift ({actual_drift})!"
            )

        # 6. Overall 100% detection rate verification
        assert detected_step_0_count == total_trajectories, (
            f"Expected {total_trajectories}/100 step 0 detections, got {detected_step_0_count}"
        )
        assert is_valid_false_count == total_trajectories, (
            f"Expected {total_trajectories}/100 invalidations, got {is_valid_false_count}"
        )

    def test_100_trajectories_valid_baseline(
        self, canonical_test_setup: tuple[NoetherianEngineConfig, list[ConservedChargeDefinition]]
    ) -> None:
        """Verifies 100% pass rate (is_valid=True, zero violations) for trajectories with compliant q_0."""
        config, charges = canonical_test_setup
        dim = config.dim

        torch.manual_seed(123)
        random.seed(123)

        total_trajectories = 100
        passed_count = 0

        for _traj_idx in range(total_trajectories):
            T = random.randint(1, 15)
            states = []
            for _ in range(T + 1):
                q = torch.zeros(dim, dtype=torch.float32)
                # Keep charge projection coordinates strictly at zero
                q[0] = 0.0
                q[1] = 0.0
                q[2] = 0.0
                q[10:] = torch.randn(dim - 10)
                q = q / torch.norm(q, p=2)
                states.append(q)

            eval_res = evaluate_trajectory_noetherian(states, config=config, charges=charges)

            if eval_res.is_valid and len(eval_res.invariant_violations) == 0:
                passed_count += 1

        assert passed_count == total_trajectories, (
            f"Expected {total_trajectories}/100 valid trajectories to pass, got {passed_count}"
        )

    def test_step_0_soft_rule_breach_non_invalidation(
        self, canonical_test_setup: tuple[NoetherianEngineConfig, list[ConservedChargeDefinition]]
    ) -> None:
        """Verifies that step 0 breaches on soft constraints record a violation but do not set is_valid=False."""
        config, charges = canonical_test_setup
        dim = config.dim

        # Breach coordinate 2 (soft_coherence_3)
        soft_charge = next(c for c in charges if c.charge_id == "soft_coherence_3")
        assert not soft_charge.is_hard_constraint

        for T in [1, 2, 5, 10]:
            q0 = torch.zeros(dim)
            q0[2] = 0.5  # Drift 0.5 > tolerance 0.08
            q0[10] = 0.5
            q0 = q0 / torch.norm(q0, p=2)

            states = [q0]
            for _ in range(T):
                qk = torch.zeros(dim)
                qk[10] = 1.0
                states.append(qk)

            eval_res = evaluate_trajectory_noetherian(states, config=config, charges=charges)

            # Step 0 violation must be recorded
            assert len(eval_res.invariant_violations) >= 1
            assert any(getattr(v, "step", None) == 0 for v in eval_res.invariant_violations)

            # Because it is a soft constraint and Mizan score is healthy, is_valid remains True
            assert eval_res.is_valid is True, f"T={T}: Soft breach falsely set is_valid to False"

    def test_static_single_state_step_0_breach_and_valid(
        self, canonical_test_setup: tuple[NoetherianEngineConfig, list[ConservedChargeDefinition]]
    ) -> None:
        """Verifies step 0 breach detection in static single-state trajectories (T=0, N=1)."""
        config, charges = canonical_test_setup
        dim = config.dim

        # Breached static state
        q_breached = torch.zeros(dim)
        q_breached[0] = 1.0
        eval_breached = evaluate_trajectory_noetherian([q_breached], config=config, charges=charges)
        assert eval_breached.is_valid is False
        assert len(eval_breached.invariant_violations) == 1
        assert eval_breached.invariant_violations[0].step == 0
        assert eval_breached.charge_drift == pytest.approx(1.0, abs=1e-5)

        # Valid static state
        q_valid = torch.zeros(dim)
        q_valid[10] = 1.0
        eval_valid = evaluate_trajectory_noetherian([q_valid], config=config, charges=charges)
        assert eval_valid.is_valid is True
        assert len(eval_valid.invariant_violations) == 0
        assert eval_valid.charge_drift == pytest.approx(0.0, abs=1e-5)

    def test_multi_charge_simultaneous_breach_at_step_0(
        self, canonical_test_setup: tuple[NoetherianEngineConfig, list[ConservedChargeDefinition]]
    ) -> None:
        """Verifies that simultaneous breaches across multiple hard constraints at step 0 are all recorded."""
        config, charges = canonical_test_setup
        dim = config.dim

        # Breaches both hard1 (coord 0) and hard2 (coord 1)
        q0 = torch.zeros(dim)
        q0[0] = 0.5
        q0[1] = 0.5
        q0 = q0 / torch.norm(q0, p=2)

        q1 = torch.zeros(dim)
        q1[10] = 1.0

        eval_res = evaluate_trajectory_noetherian([q0, q1], config=config, charges=charges)
        assert eval_res.is_valid is False

        step_0_violations = [v for v in eval_res.invariant_violations if getattr(v, "step", None) == 0]
        assert len(step_0_violations) == 2
        rule_names = {v.rule_name for v in step_0_violations}
        assert "Operational Safety Boundary" in rule_names
        assert "Core Intent Invariant" in rule_names
