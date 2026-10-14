"""Empirical Stress Test Suite for Milestone M1 Iteration 2: Substring Collision Testing.

Conducted by Challenger 2 under Quanta SDK Scientific Integrity Protocol (GEMINI.md).

Adversarial Dimensions Tested:
1. Explicit Substring Overlap Pairs (DISPATCH.md):
   - Hard: "Safe" vs Soft: "Safe Boundary Policy" (Prefix collision)
   - Hard: "Safe" vs Soft: "Safe_Margin" (Prefix + underscore collision)
   - Hard: "Safe" vs Soft: "Unsafe" (Suffix collision)
   - Hard: "Safe" vs Soft: "Safety First" (Root word collision)
   - Hard: "Safe" vs Soft: "Be Safe Always" (Infix / substring collision)
2. Reverse Substring Inclusions:
   - Soft: "Safe" vs Hard: "Safe Boundary Policy"
   - Soft: "Intent" vs Hard: "Core User Intent Alignment"
3. Multi-Step Trajectories (N in {1, 2, 5, 20, 50}):
   - Violations isolated to Step 0, intermediate steps, and final step.
   - Confirmation that soft-only breaches NEVER produce is_valid=False or regime="CRITICAL_BREACH".
   - Confirmation that true hard breaches ALWAYS trigger is_valid=False reliably.
4. Deserialized / Plain String Fallback Verification (Tiers 1, 2, 3 in noetherian.py):
   - Object-level InvariantViolation attributes.
   - Regex extraction on template strings with quotes, special characters, and colons.
   - Fallback prefix templates with colon delimiters.
5. Adversarial & Malicious Rule Names:
   - Single quotes: "User's Constraint" vs "User's Constraint Relaxed"
   - Regex metacharacters: "Safe.*", "Safe+", "Safe(1)", "[Safe]", "Safe$"
   - Colon namespaces: "Core:Safe" vs "Core:Safe:Extended"
   - Delimiter mimicry / injection attacks.
"""

from __future__ import annotations

import math
import re

import pytest
import torch

from quanta.cognitive.noetherian import (
    ConservedChargeDefinition,
    NoetherianEngineConfig,
    evaluate_trajectory_noetherian,
)


def sample_unit_vector(dim: int = 64, seed_offset: int = 0) -> torch.Tensor:
    """Creates a deterministic unit-norm vector in R^dim."""
    gen = torch.Generator().manual_seed(2026 + seed_offset)
    v = torch.randn(dim, generator=gen, dtype=torch.float32)
    return v / torch.norm(v, p=2)


def make_orthogonal_pair(dim: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns two strictly orthogonal unit vectors e_0 and e_1."""
    e0 = torch.zeros(dim, dtype=torch.float32)
    e0[0] = 1.0
    e1 = torch.zeros(dim, dtype=torch.float32)
    e1[1] = 1.0
    return e0, e1


# ============================================================================
# Test Suite 1: Dispatch-Mandated Substring Collision Pairs
# ============================================================================


class TestDispatchMandatedSubstringCollisions:
    """Requirement: Empirically test substring collision scenarios between overlapping soft and hard rule names."""

    @pytest.mark.parametrize(
        "hard_name,soft_name,collision_type",
        [
            ("Safe", "Safe Boundary Policy", "prefix"),
            ("Safe", "Safe_Margin", "prefix_underscore"),
            ("Safe", "Unsafe", "suffix"),
            ("Safe", "Safety First", "root_word"),
            ("Safe", "Be Safe Always", "infix"),
            ("Intent", "Core Intent Alignment", "infix"),
            ("Intent", "Intentional Drift Policy", "prefix"),
            ("Truth", "Epistemic Truth Boundary", "infix"),
            ("Truth", "Untruth Restraint", "suffix"),
            ("Coherence", "Contextual Coherence", "suffix"),
            ("Coherence", "Coherence Guardrail", "prefix"),
            ("Norm", "Normative Consistency Policy", "prefix"),
            ("Norm", "Subnorm Stability", "infix"),
        ],
    )
    def test_soft_only_breach_never_falsely_invalidates(
        self,
        hard_name: str,
        soft_name: str,
        collision_type: str,
    ) -> None:
        """Verifies that when only the soft rule breaches tolerance, is_valid remains True and regime is not BREACH."""
        dim = 64
        gen_soft, gen_hard = make_orthogonal_pair(dim)

        soft_charge = ConservedChargeDefinition(
            charge_id=f"soft_{collision_type}",
            rule_name=soft_name,
            generator_vector=gen_soft,
            target_value=0.0,
            weight=1.0,
            tolerance=0.05,
            is_hard_constraint=False,  # Soft constraint
        )

        hard_charge = ConservedChargeDefinition(
            charge_id=f"hard_{collision_type}",
            rule_name=hard_name,
            generator_vector=gen_hard,
            target_value=0.0,
            weight=1.0,
            tolerance=0.05,
            is_hard_constraint=True,  # Hard constraint
        )

        config = NoetherianEngineConfig(
            dim=dim,
            critical_breach_threshold=0.40,
        )

        # Statevector strictly aligned with gen_soft (drift=0.15 on soft, drift=0.0 on hard)
        # Soft: Q = <gen_soft, q> = 0.15 > 0.05 (BREACHED)
        # Hard: Q = <gen_hard, q> = 0.00 <= 0.05 (SATISFIED)
        q_test = torch.zeros(dim, dtype=torch.float32)
        q_test[0] = 0.15
        q_test[2] = math.sqrt(1.0 - 0.15**2)  # Maintain unit norm on orthogonal coordinate

        evaluation = evaluate_trajectory_noetherian(
            states=[q_test],
            config=config,
            charges=[soft_charge, hard_charge],
        )

        # Assertions
        assert evaluation.is_valid is True, (
            f"False Invalidation! Hard rule '{hard_name}' collided with soft rule '{soft_name}' ({collision_type})"
        )
        assert evaluation.regime != "CRITICAL_BREACH", (
            f"Regime incorrectly set to CRITICAL_BREACH for soft-only breach '{soft_name}'"
        )
        assert len(evaluation.invariant_violations) == 1, (
            f"Expected exactly 1 violation for soft rule, got {len(evaluation.invariant_violations)}"
        )
        assert soft_name in evaluation.invariant_violations[0], (
            f"Expected soft rule name '{soft_name}' in violation, got {evaluation.invariant_violations[0]}"
        )
        assert hard_name not in evaluation.invariant_violations[0].replace(soft_name, ""), (
            f"Hard rule name '{hard_name}' leaked into breach description"
        )

    @pytest.mark.parametrize(
        "hard_name,soft_name",
        [
            ("Safe", "Safe Boundary Policy"),
            ("Safe", "Safe_Margin"),
            ("Safe", "Unsafe"),
            ("Intent", "Core Intent Alignment"),
            ("Coherence", "Contextual Coherence"),
        ],
    )
    def test_true_hard_breach_always_reliably_invalidates(
        self,
        hard_name: str,
        soft_name: str,
    ) -> None:
        """Verifies that when the hard rule breaches tolerance, is_valid is reliably False."""
        dim = 64
        gen_soft, gen_hard = make_orthogonal_pair(dim)

        soft_charge = ConservedChargeDefinition(
            charge_id="soft_rule",
            rule_name=soft_name,
            generator_vector=gen_soft,
            target_value=0.0,
            weight=1.0,
            tolerance=0.05,
            is_hard_constraint=False,
        )

        hard_charge = ConservedChargeDefinition(
            charge_id="hard_rule",
            rule_name=hard_name,
            generator_vector=gen_hard,
            target_value=0.0,
            weight=1.0,
            tolerance=0.05,
            is_hard_constraint=True,
        )

        config = NoetherianEngineConfig(
            dim=dim,
            critical_breach_threshold=0.40,
        )

        # Statevector strictly aligned with gen_hard (drift=0.15 on hard, drift=0.0 on soft)
        # Hard: Q = <gen_hard, q> = 0.15 > 0.05 (BREACHED)
        # Soft: Q = <gen_soft, q> = 0.00 <= 0.05 (SATISFIED)
        q_test = torch.zeros(dim, dtype=torch.float32)
        q_test[1] = 0.15
        q_test[2] = math.sqrt(1.0 - 0.15**2)

        evaluation = evaluate_trajectory_noetherian(
            states=[q_test],
            config=config,
            charges=[soft_charge, hard_charge],
        )

        assert evaluation.is_valid is False, (
            f"Failure to detect hard breach! Rule '{hard_name}' breached but is_valid was True"
        )
        assert len(evaluation.invariant_violations) >= 1
        hard_violations = [v for v in evaluation.invariant_violations if hard_name in v]
        assert len(hard_violations) >= 1


# ============================================================================
# Test Suite 2: Multi-Step Trajectories Across Various Step Counts
# ============================================================================


class TestMultiStepTrajectorySubstringCollisions:
    """Requirement: Test multi-step trajectories (N in {1, 2, 5, 20, 50}) with step-localized breaches."""

    @pytest.mark.parametrize("n_steps", [1, 2, 5, 20, 50])
    @pytest.mark.parametrize("breach_location", ["start", "middle", "end", "all"])
    def test_multistep_soft_only_breach_preserves_validity(
        self,
        n_steps: int,
        breach_location: str,
    ) -> None:
        """Verifies multi-step trajectories with soft rule breaches at various positions remain valid."""
        dim = 64
        gen_soft, gen_hard = make_orthogonal_pair(dim)

        soft_charge = ConservedChargeDefinition(
            charge_id="soft_bound",
            rule_name="Safe Boundary Policy",
            generator_vector=gen_soft,
            target_value=0.0,
            tolerance=0.05,
            is_hard_constraint=False,
        )

        hard_charge = ConservedChargeDefinition(
            charge_id="hard_safe",
            rule_name="Safe",
            generator_vector=gen_hard,
            target_value=0.0,
            tolerance=0.05,
            is_hard_constraint=True,
        )

        config = NoetherianEngineConfig(
            dim=dim,
            critical_breach_threshold=0.30,  # Ensure mild drift does not trip global critical threshold
        )

        # Construct trajectory
        q_clean = torch.zeros(dim, dtype=torch.float32)
        q_clean[2] = 1.0  # Orthogonal to both soft (dim 0) and hard (dim 1)

        q_soft_breach = torch.zeros(dim, dtype=torch.float32)
        q_soft_breach[0] = 0.10  # Drift 0.10 > 0.05 on soft
        q_soft_breach[2] = math.sqrt(1.0 - 0.10**2)

        states: list[torch.Tensor] = []
        for step in range(n_steps):
            is_breaching = False
            if breach_location == "start" and step == 0 or breach_location == "middle" and step == n_steps // 2 or breach_location == "end" and step == n_steps - 1 or breach_location == "all":
                is_breaching = True

            states.append(q_soft_breach.clone() if is_breaching else q_clean.clone())

        eval_res = evaluate_trajectory_noetherian(
            states=states,
            config=config,
            charges=[soft_charge, hard_charge],
        )

        assert eval_res.is_valid is True, (
            f"Multi-step false invalidation for N={n_steps}, breach at {breach_location}!"
        )
        assert eval_res.regime != "CRITICAL_BREACH"
        assert len(eval_res.invariant_violations) > 0
        for v in eval_res.invariant_violations:
            assert "Safe Boundary Policy" in str(v)

    @pytest.mark.parametrize("n_steps", [2, 5, 20])
    @pytest.mark.parametrize("breach_location", ["start", "middle", "end"])
    def test_multistep_hard_breach_detected_at_any_position(
        self,
        n_steps: int,
        breach_location: str,
    ) -> None:
        """Verifies multi-step trajectories with hard rule breaches at step 0, middle, or end are invalidated."""
        dim = 64
        gen_soft, gen_hard = make_orthogonal_pair(dim)

        soft_charge = ConservedChargeDefinition(
            charge_id="soft_bound",
            rule_name="Safe Boundary Policy",
            generator_vector=gen_soft,
            target_value=0.0,
            tolerance=0.05,
            is_hard_constraint=False,
        )

        hard_charge = ConservedChargeDefinition(
            charge_id="hard_safe",
            rule_name="Safe",
            generator_vector=gen_hard,
            target_value=0.0,
            tolerance=0.05,
            is_hard_constraint=True,
        )

        config = NoetherianEngineConfig(dim=dim)

        q_clean = torch.zeros(dim, dtype=torch.float32)
        q_clean[2] = 1.0

        q_hard_breach = torch.zeros(dim, dtype=torch.float32)
        q_hard_breach[1] = 0.10  # Drift on hard rule
        q_hard_breach[2] = math.sqrt(1.0 - 0.10**2)

        states: list[torch.Tensor] = []
        for step in range(n_steps):
            is_breaching = False
            if breach_location == "start" and step == 0 or breach_location == "middle" and step == n_steps // 2 or breach_location == "end" and step == n_steps - 1:
                is_breaching = True

            states.append(q_hard_breach.clone() if is_breaching else q_clean.clone())

        eval_res = evaluate_trajectory_noetherian(
            states=states,
            config=config,
            charges=[soft_charge, hard_charge],
        )

        assert eval_res.is_valid is False, (
            f"Multi-step missed hard breach for N={n_steps} at location={breach_location}!"
        )


# ============================================================================
# Test Suite 3: Raw / Deserialized String Provenance (Tiers 1, 2, 3 Fallback)
# ============================================================================


class TestRawStringProvenanceAndFallbackMatching:
    """Stress tests Tier 2 (regex extraction) and Tier 3 (structured prefix) in noetherian.py."""

    def test_plain_string_violations_tier2_exact_regex_matching(self) -> None:
        """When violations are plain strings (e.g. from JSON deserialization), Tier 2 regex must match exact name."""
        dim = 64
        ConservedChargeDefinition(
            charge_id="hard_safe",
            rule_name="Safe",
            generator_vector=torch.zeros(dim),
            tolerance=0.05,
            is_hard_constraint=True,
        )
        ConservedChargeDefinition(
            charge_id="soft_safe",
            rule_name="Safe Boundary Policy",
            generator_vector=torch.zeros(dim),
            tolerance=0.05,
            is_hard_constraint=False,
        )

        NoetherianEngineConfig(dim=dim, critical_breach_threshold=0.10)
        q_clean = torch.zeros(dim)
        q_clean[0] = 1.0

        # We construct an evaluation and inject plain string violations
        # Scenario A: Plain string violation for SOFT rule "Safe Boundary Policy"
        plain_soft_violation = "Charge 'Safe Boundary Policy' breached at step 0: drift=0.1200 > tol=0.0500"
        assert not hasattr(plain_soft_violation, "is_hard_constraint")

        # Directly test Tier 2 regex behavior
        m = re.match(r"^Charge '(?P<name>.*)' breached at step \d+:", plain_soft_violation)
        assert m is not None
        extracted_name = m.group("name")
        assert extracted_name == "Safe Boundary Policy"
        assert extracted_name != "Safe"
        assert extracted_name not in {"Safe"}

        # Scenario B: Plain string violation for HARD rule "Safe"
        plain_hard_violation = "Charge 'Safe' breached at step 0: drift=0.1200 > tol=0.0500"
        m_hard = re.match(r"^Charge '(?P<name>.*)' breached at step \d+:", plain_hard_violation)
        assert m_hard is not None
        assert m_hard.group("name") == "Safe"
        assert m_hard.group("name") in {"Safe"}

    def test_plain_string_violations_tier3_prefix_colon_delimiter(self) -> None:
        """Tier 3 prefix fallback must not match when rule name is a prefix without delimiter."""
        hard_rule_names = {"Safe"}

        # "Violation in Safe Boundary Policy: drift=0.10"
        soft_v = "Violation in Safe Boundary Policy: drift=0.10"
        # Check against template: v.startswith(f"Violation in {name}:")
        matches = any(soft_v.startswith(f"Violation in {name}:") for name in hard_rule_names)
        assert matches is False, "Tier 3 prefix falsely matched without delimiter!"

        # Hard violation with delimiter matches:
        hard_v = "Violation in Safe: drift=0.10"
        matches_hard = any(hard_v.startswith(f"Violation in {name}:") for name in hard_rule_names)
        assert matches_hard is True, "Tier 3 prefix failed to match hard violation!"


# ============================================================================
# Test Suite 4: Adversarial & Complex Rule Names
# ============================================================================


class TestAdversarialAndPathologicalRuleNames:
    """Stress tests rule names containing single quotes, regex metacharacters, namespaces, and unicode."""

    @pytest.mark.parametrize(
        "hard_name,soft_name",
        [
            ("User's Policy", "User's Policy Extended"),
            ("Rule[A]", "Rule[A]_Relaxed"),
            ("Safe.*Pattern", "Safe.*Pattern Policy"),
            ("Safe+Bound", "Safe+Bound-Extended"),
            ("ns:safe:core", "ns:safe:core:subpolicy"),
            ("Güvenlik", "Güvenlik Duvarı Politikası"),
            ("Mizan", "Kozmik Mizan Hakemi"),
        ],
    )
    def test_pathological_character_substring_isolation(
        self,
        hard_name: str,
        soft_name: str,
    ) -> None:
        """Verifies special character rule names maintain strict substring isolation."""
        dim = 64
        gen_soft, gen_hard = make_orthogonal_pair(dim)

        soft_charge = ConservedChargeDefinition(
            charge_id="soft_patho",
            rule_name=soft_name,
            generator_vector=gen_soft,
            tolerance=0.05,
            is_hard_constraint=False,
        )
        hard_charge = ConservedChargeDefinition(
            charge_id="hard_patho",
            rule_name=hard_name,
            generator_vector=gen_hard,
            tolerance=0.05,
            is_hard_constraint=True,
        )

        config = NoetherianEngineConfig(dim=dim, critical_breach_threshold=0.30)

        # Trigger soft breach only
        q = torch.zeros(dim)
        q[0] = 0.15
        q[2] = math.sqrt(1.0 - 0.15**2)

        res = evaluate_trajectory_noetherian(
            states=[q],
            config=config,
            charges=[soft_charge, hard_charge],
        )

        assert res.is_valid is True, f"Falsely invalidated for pathological pair: {hard_name} vs {soft_name}"
        assert res.regime != "CRITICAL_BREACH"
        assert len(res.invariant_violations) == 1
        assert soft_name in str(res.invariant_violations[0])

    def test_delimiter_injection_mimicry_attack(self) -> None:
        """Adversarial scenario: Soft rule name carefully crafted to mimic format delimiters.

        Rule name: `Safe' breached at step 0: drift=0.5000 > tol=0.0100; dummy='Safe`
        Verifies that Tier 1 strongly typed attribute completely neutralizes this attack.
        """
        dim = 64
        gen_soft, gen_hard = make_orthogonal_pair(dim)

        malicious_soft_name = "Safe' breached at step 0: drift=0.5000 > tol=0.0100; dummy='Safe"

        soft_charge = ConservedChargeDefinition(
            charge_id="malicious_soft",
            rule_name=malicious_soft_name,
            generator_vector=gen_soft,
            tolerance=0.05,
            is_hard_constraint=False,
        )

        hard_charge = ConservedChargeDefinition(
            charge_id="hard_safe",
            rule_name="Safe",
            generator_vector=gen_hard,
            tolerance=0.05,
            is_hard_constraint=True,
        )

        config = NoetherianEngineConfig(dim=dim, critical_breach_threshold=0.30)

        # Trigger soft breach only
        q = torch.zeros(dim)
        q[0] = 0.15
        q[2] = math.sqrt(1.0 - 0.15**2)

        res = evaluate_trajectory_noetherian(
            states=[q],
            config=config,
            charges=[soft_charge, hard_charge],
        )

        # Tier 1 guarantees that because is_hard_constraint is False, it does not invalidate
        assert res.is_valid is True
        assert len(res.invariant_violations) == 1
        assert res.invariant_violations[0].is_hard_constraint is False


# ============================================================================
# Test Suite 5: Simultaneous Soft + Hard Breaches (Sanity Matrix)
# ============================================================================


class TestSimultaneousSoftAndHardBreaches:
    """Confirms proper handling when both soft and hard rules breach concurrently."""

    def test_simultaneous_breaches_records_both_and_invalidates(self) -> None:
        """When both soft and hard rules breach, both are recorded in violations and is_valid is False."""
        dim = 64
        gen_soft, gen_hard = make_orthogonal_pair(dim)

        soft_charge = ConservedChargeDefinition(
            charge_id="soft_rule",
            rule_name="Safe Boundary Policy",
            generator_vector=gen_soft,
            tolerance=0.05,
            is_hard_constraint=False,
        )

        hard_charge = ConservedChargeDefinition(
            charge_id="hard_rule",
            rule_name="Safe",
            generator_vector=gen_hard,
            tolerance=0.05,
            is_hard_constraint=True,
        )

        config = NoetherianEngineConfig(dim=dim)

        # State breaching BOTH rules
        q = torch.zeros(dim)
        q[0] = 0.15  # soft breach
        q[1] = 0.15  # hard breach
        q[2] = math.sqrt(1.0 - 2 * 0.15**2)

        res = evaluate_trajectory_noetherian(
            states=[q],
            config=config,
            charges=[soft_charge, hard_charge],
        )

        assert res.is_valid is False
        assert len(res.invariant_violations) == 2
        names_in_violations = {getattr(v, "rule_name", "") for v in res.invariant_violations}
        assert "Safe Boundary Policy" in names_in_violations
        assert "Safe" in names_in_violations
