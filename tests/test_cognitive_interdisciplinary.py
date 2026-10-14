"""Comprehensive test suite for Interdisciplinary Biomorphic Cognitive Panel & Arbitration.

Validates multi-disciplinary evaluations spanning Neurobiology, Psychiatry, and Sociology,
alongside QuantumDecisionArbiter multi-criteria integration, DAG microglial pruning,
backwards-compatible ConsequenceVector interpolation, and architectural invariants.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from quanta.cognitive.arbiter import (
    CognitivePanelScore,
    ConsequenceVector,
    DecisionDAG,
    DecisionEdge,
    DecisionNode,
    NeurobiologicalEvaluation,
    PsychiatricEvaluation,
    QuantumDecisionArbiter,
    SociologicalEvaluation,
    _clamp01,
    _safe_float,
    _safe_int,
)
from quanta.cognitive.tom_analyzer import (
    DreamSeed,
    TheoryOfMindAnalyzer,
)
from quanta.cognitive.tom_analyzer import (
    _safe_float as tom_safe_float,
)
from quanta.cognitive.tom_analyzer import (
    _safe_int as tom_safe_int,
)
from scripts.hooks.cognitive_lifecycle_hook import (
    FastBiomorphicMemory,
    distill_semantic_gist,
    is_actionable_resolution,
)

# ============================================================================
# 1. Neurobiological Evaluation Tests
# ============================================================================


class TestNeurobiologicalEvaluation:
    """Verifies neurobiological metrics: synaptic saturation, energy budget, sleep affinity."""

    def test_synaptic_saturation_calculation(self) -> None:
        """Asserts turn count, message volume, and buffer density scale synaptic saturation."""
        tom = TheoryOfMindAnalyzer()

        # 1. Conversational volume baseline: 15 messages -> 15/30 = 0.50
        msgs_15 = [{"role": "user", "content": f"query step {i}"} for i in range(15)]
        panel_15 = tom.evaluate_cognitive_panel(msgs_15)
        assert 0.49 <= panel_15.neurobiology.synaptic_saturation <= 0.51

        # 2. Context turn_count and buffer_density override
        ctx = {"turn_count": 27, "buffer_density": 0.85}
        panel_ctx = tom.evaluate_cognitive_panel(msgs_15, context=ctx)
        # turn_count 27 / 30.0 = 0.90, which exceeds buffer_density 0.85 and base 0.50
        assert panel_ctx.neurobiology.synaptic_saturation >= 0.90

        # 3. Test failure stress scaling: test_failures=2 adds min(0.30, 2 * 0.10) = 0.20 stress
        panel_stress = tom.evaluate_cognitive_panel(
            msgs_15, project_state={"test_failures": 2}
        )
        assert panel_stress.neurobiology.synaptic_saturation >= 0.69

        # 4. Aggregate cost evaluation with default and custom weights
        neuro = NeurobiologicalEvaluation(
            synaptic_saturation=0.80,
            energy_expenditure=0.60,
            sleep_consolidation_affinity=0.40,
        )
        # Default weights: (0.40, 0.35, 0.25) -> 0.4*0.8 + 0.35*0.6 + 0.25*0.4 = 0.32+0.21+0.10 = 0.63
        assert pytest.approx(neuro.aggregate_cost(), abs=1e-6) == 0.63

        custom_weights = (0.50, 0.30, 0.20)
        # 0.5*0.8 + 0.3*0.6 + 0.2*0.4 = 0.40 + 0.18 + 0.08 = 0.66
        assert pytest.approx(neuro.aggregate_cost(weights=custom_weights), abs=1e-6) == 0.66
        assert 0.0 <= neuro.aggregate_cost() <= 1.0

    def test_energy_budgeting_penalty(self) -> None:
        """Asserts high metabolic cost, token density, and tool call bursts elevate energy cost."""
        tom = TheoryOfMindAnalyzer()

        # Low activity: short text, 0 tools
        low_msgs = [{"role": "user", "content": "quick test"}]
        panel_low = tom.evaluate_cognitive_panel(low_msgs)
        assert panel_low.neurobiology.energy_expenditure < 0.10

        # High activity: heavy character volume and multiple tool calls
        heavy_msgs: list[dict[str, Any]] = [
            {"role": "assistant", "content": "A" * 2000, "tool_call": "run_sim_1"},
            {"role": "assistant", "content": "B" * 2000, "tool_calls": ["t2", "t3"]},
            {"role": "assistant", "content": "C" * 2000, "tool": "exec_4"},
        ]
        panel_heavy = tom.evaluate_cognitive_panel(heavy_msgs)
        # Base chars: 6000 / 4000 = 1.5 -> clamped to 1.0; tools: 3 * 0.05 = 0.15 -> capped at 1.0
        assert panel_heavy.neurobiology.energy_expenditure >= 0.95

        # Direct context override via metabolic_cost
        panel_ctx_meta = tom.evaluate_cognitive_panel(
            low_msgs, context={"metabolic_cost": 0.75}
        )
        assert panel_ctx_meta.neurobiology.energy_expenditure >= 0.75

        # Aggregate penalty is higher for the heavy workload
        assert panel_heavy.neurobiology.aggregate_cost() > panel_low.neurobiology.aggregate_cost()

    def test_sleep_wake_consolidation_affinity(self) -> None:
        """Asserts quiet periods, resolved milestones, and turn exhaustion yield high sleep affinity."""
        tom = TheoryOfMindAnalyzer()

        # Quiescent project state signals high consolidation readiness
        panel_quiescent = tom.evaluate_cognitive_panel(
            [{"role": "user", "content": "status check"}],
            project_state={"quiescent": True},
        )
        assert panel_quiescent.neurobiology.sleep_consolidation_affinity >= 0.85

        # Resolved state also triggers high consolidation readiness
        panel_resolved = tom.evaluate_cognitive_panel(
            [{"role": "user", "content": "status check"}],
            project_state={"resolved": True},
        )
        assert panel_resolved.neurobiology.sleep_consolidation_affinity >= 0.85

        # 0 test failures with conversational history >= 5 messages triggers moderate sleep affinity
        msgs_5 = [{"role": "user", "content": f"turn {i}"} for i in range(5)]
        panel_stable = tom.evaluate_cognitive_panel(
            msgs_5, project_state={"test_failures": 0}
        )
        assert panel_stable.neurobiology.sleep_consolidation_affinity >= 0.60

        # Prolonged conversation (> 20 messages) elevates sleep affinity for SHY downscaling
        msgs_25 = [{"role": "user", "content": f"extended turn {i}"} for i in range(25)]
        panel_long = tom.evaluate_cognitive_panel(msgs_25)
        assert panel_long.neurobiology.sleep_consolidation_affinity >= 0.70

        # Explicit context override
        panel_override = tom.evaluate_cognitive_panel(
            msgs_5, context={"sleep_affinity": 0.92}
        )
        assert panel_override.neurobiology.sleep_consolidation_affinity >= 0.92


# ============================================================================
# 2. Psychiatric Evaluation Tests
# ============================================================================


class TestPsychiatricEvaluation:
    """Verifies psychiatric stability metrics: anti-rumination, perseveration, threat appraisal."""

    def test_anti_rumination_detection_and_bounding(self) -> None:
        """Asserts repeating circular domain topics triggers anti-rumination bounds."""
        tom = TheoryOfMindAnalyzer()

        # 1. Non-repetitive healthy conversation has 0 rumination risk
        healthy_msgs = [
            {"role": "user", "content": "Let us inspect the memory manager."},
            {"role": "assistant", "content": "Now tuning qos parameters."},
            {"role": "user", "content": "Checking darwin evolutionary selection."},
        ]
        panel_healthy = tom.evaluate_cognitive_panel(healthy_msgs)
        assert panel_healthy.psychiatry.rumination_risk == 0.0

        # 2. Circular conversation repeating 'memory' >= 3 times
        circular_msgs = [
            {"role": "user", "content": "Check memory leak in cache."},
            {"role": "assistant", "content": "Investigating memory buffer."},
            {"role": "user", "content": "Still failing in memory allocation."},
            {"role": "assistant", "content": "Memory consumption remains high."},
        ]
        panel_circular = tom.evaluate_cognitive_panel(circular_msgs)
        # 4 repetitions -> 0.30 + 0.20 * (4 - 2) = 0.70
        assert panel_circular.psychiatry.rumination_risk >= 0.65

        # 3. Context repetition_count or circular_depth triggers elevated rumination
        panel_ctx = tom.evaluate_cognitive_panel(
            healthy_msgs, context={"repetition_count": 5}
        )
        assert panel_ctx.psychiatry.rumination_risk >= 0.85

        # 4. Project state rumination flag triggers >= 0.75
        panel_state = tom.evaluate_cognitive_panel(
            healthy_msgs, project_state={"rumination": True}
        )
        assert panel_state.psychiatry.rumination_risk >= 0.75

        # 5. Strict bounding in [0.0, 1.0] even with extreme repetition count
        panel_extreme = tom.evaluate_cognitive_panel(
            healthy_msgs, context={"repetition_count": 1000}
        )
        assert panel_extreme.psychiatry.rumination_risk <= 1.0

    def test_perseveration_penalty_vs_flexibility(self) -> None:
        """Asserts repeating failed tool actions incurs perseveration penalty while flexibility is rewarded."""
        tom = TheoryOfMindAnalyzer()

        # 1. Flexible behavior: no tool failures, no repeated error terms
        flex_msgs = [
            {"role": "user", "content": "Running test suite."},
            {"role": "assistant", "content": "All green, proceeding with deployment."},
        ]
        panel_flex = tom.evaluate_cognitive_panel(flex_msgs)
        assert panel_flex.psychiatry.perseveration_penalty == 0.0

        # 2. Repeated tool failures in context triggers perseveration penalty >= 0.65
        panel_tool_fail = tom.evaluate_cognitive_panel(
            flex_msgs, context={"repeated_tool_failures": 3}
        )
        assert panel_tool_fail.psychiatry.perseveration_penalty >= 0.65

        # 3. Conversational perseveration: multiple failing turns with test_failures > 1
        fail_msgs = [
            {"role": "user", "content": "test fail on line 42"},
            {"role": "assistant", "content": "encountered exception and traceback"},
            {"role": "user", "content": "bug still broken"},
        ]
        panel_conv_fail = tom.evaluate_cognitive_panel(
            fail_msgs, project_state={"test_failures": 3}
        )
        assert panel_conv_fail.psychiatry.perseveration_penalty >= 0.60

        # 4. Aggregate cost reflects perseveration penalty
        psych = PsychiatricEvaluation(
            rumination_risk=0.10,
            perseveration_penalty=0.80,
            threat_distortion=0.10,
        )
        # Default weights: (0.40, 0.35, 0.25) -> 0.4*0.1 + 0.35*0.8 + 0.25*0.1 = 0.04 + 0.28 + 0.025 = 0.345
        assert pytest.approx(psych.aggregate_cost(), abs=1e-6) == 0.345

    def test_calibrated_threat_appraisal(self) -> None:
        """Asserts catastrophic distortion is distinguished from calibrated realistic threat."""
        tom = TheoryOfMindAnalyzer()

        # 1. Catastrophic alarmism with 0 actual test failures represents uncalibrated distortion
        alarmist_msgs = [
            {"role": "user", "content": "This is a catastrophic disaster and fatal failure!"}
        ]
        panel_distorted = tom.evaluate_cognitive_panel(
            alarmist_msgs, project_state={"test_failures": 0}
        )
        assert panel_distorted.psychiatry.threat_distortion >= 0.45

        # 2. Genuine blocker / real test failures calibrate the threat appraisal
        panel_calibrated = tom.evaluate_cognitive_panel(
            alarmist_msgs, project_state={"test_failures": 5, "blocker": True}
        )
        # Under genuine verified blocker/failures, threat appraisal is calibrated, not distorted
        assert panel_calibrated.psychiatry.threat_distortion <= 0.25

        # 3. Calm conversational message history has zero threat distortion
        calm_msgs = [{"role": "user", "content": "Status report: all systems nominal."}]
        panel_calm = tom.evaluate_cognitive_panel(calm_msgs)
        assert panel_calm.psychiatry.threat_distortion == 0.0


# ============================================================================
# 3. Sociological Evaluation Tests
# ============================================================================


class TestSociologicalEvaluation:
    """Verifies sociological alignment metrics: user fatigue, social framing, coordination friction."""

    def test_user_fatigue_detection_from_terse_cadence(self) -> None:
        """Asserts terse conversational cadence (<= 3 words) signals user cognitive fatigue."""
        tom = TheoryOfMindAnalyzer()

        # 1. Detailed user messages indicate engaged, non-fatigued collaborator
        engaged_msgs = [
            {"role": "user", "content": "Please implement the new quantum arbiter module with unit tests."},
            {"role": "assistant", "content": "Implementing module now."},
            {"role": "user", "content": "Make sure to verify backwards compatibility with 4D costs."},
        ]
        panel_engaged = tom.evaluate_cognitive_panel(engaged_msgs)
        assert panel_engaged.sociology.user_fatigue_impact <= 0.15

        # 2. Terse user turns (>= 50% user turns <= 3 words) trigger fatigue detection
        terse_msgs = [
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "Detailed step analysis..."},
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "Another long explanation..."},
            {"role": "user", "content": "done?"},
        ]
        panel_terse = tom.evaluate_cognitive_panel(terse_msgs)
        # 3/3 terse turns = 100% -> base_fatigue = 0.55 + 0.40 * (1.0 - 0.5) = 0.75
        assert panel_terse.sociology.user_fatigue_impact >= 0.70

        # 3. Direct project_state or context signal
        panel_signaled = tom.evaluate_cognitive_panel(
            engaged_msgs, project_state={"user_fatigue": True}
        )
        assert panel_signaled.sociology.user_fatigue_impact >= 0.70

    def test_social_context_framing(self) -> None:
        """Asserts collaborative tone minimizes misalignment while hostile language is penalized."""
        tom = TheoryOfMindAnalyzer()

        # 1. Collaborative phrasing ("thanks", "team", "together", "please")
        collab_msgs = [
            {"role": "user", "content": "Thank you team, please let's work together."},
            {"role": "assistant", "content": "Great teamwork."},
        ]
        panel_collab = tom.evaluate_cognitive_panel(collab_msgs)
        assert panel_collab.sociology.social_misalignment == 0.0

        # 2. Hostile or misalignment phrasing ("wrong", "terrible", "useless")
        misalign_msgs = [
            {"role": "user", "content": "That is wrong and terrible."},
            {"role": "user", "content": "This approach is useless."},
        ]
        panel_misalign = tom.evaluate_cognitive_panel(misalign_msgs)
        # 2 messages * 0.40 = 0.80
        assert panel_misalign.sociology.social_misalignment >= 0.75

        # 3. Collaborative tokens mitigate misalignment tokens
        mixed_msgs = [
            {"role": "user", "content": "That is wrong."},
            {"role": "user", "content": "Thank you team, let's fix it together."},
        ]
        panel_mixed = tom.evaluate_cognitive_panel(mixed_msgs)
        # Misalign (1 msg * 0.40) mitigated by collab (1 msg * 0.15) -> 0.25
        assert panel_mixed.sociology.social_misalignment < panel_misalign.sociology.social_misalignment

    def test_collective_intelligence_friction(self) -> None:
        """Asserts multi-agent coordination friction words or state metrics trigger friction score."""
        tom = TheoryOfMindAnalyzer()

        # 1. Smooth interaction has 0 friction
        smooth_msgs = [{"role": "user", "content": "Dispatching worker m4 with clear specifications."}]
        panel_smooth = tom.evaluate_cognitive_panel(smooth_msgs)
        assert panel_smooth.sociology.coordination_friction == 0.0

        # 2. Friction words ("deadlock", "conflict", "desync")
        single_fric_msgs = [
            {"role": "user", "content": "Encountered deadlock between peer agents."},
        ]
        panel_single = tom.evaluate_cognitive_panel(single_fric_msgs)
        # 1 message * 0.35 = 0.35
        assert pytest.approx(panel_single.sociology.coordination_friction, abs=1e-6) == 0.35

        multi_fric_msgs = [
            {"role": "user", "content": "Encountered deadlock between peer agents."},
            {"role": "assistant", "content": "Desync and conflict detected during coordination."},
        ]
        panel_multi = tom.evaluate_cognitive_panel(multi_fric_msgs)
        # 2 messages * 0.35 = 0.70
        assert panel_multi.sociology.coordination_friction >= 0.65

        # 3. Aggregate sociological cost computation
        socio = SociologicalEvaluation(
            social_misalignment=0.20,
            user_fatigue_impact=0.80,
            coordination_friction=0.40,
        )
        # Default weights: (0.35, 0.40, 0.25) -> 0.35*0.20 + 0.40*0.80 + 0.25*0.40 = 0.07 + 0.32 + 0.10 = 0.49
        assert pytest.approx(socio.aggregate_cost(), abs=1e-6) == 0.49


# ============================================================================
# 4. Multi-Criteria Cognitive Panel Tests
# ============================================================================


class TestMultiCriteriaCognitivePanel:
    """Verifies composite cognitive panel aggregation, ConsequenceVector backwards compatibility & interpolation."""

    def test_cognitive_panel_aggregate_penalty(self) -> None:
        """Asserts composite discipline weighting computes exact aggregate penalty."""
        neuro = NeurobiologicalEvaluation(0.5, 0.4, 0.3)
        psych = PsychiatricEvaluation(0.6, 0.2, 0.1)
        socio = SociologicalEvaluation(0.1, 0.8, 0.2)
        panel = CognitivePanelScore(neurobiology=neuro, psychiatry=psych, sociology=socio)

        # Default weights: (0.30, 0.35, 0.35)
        # neuro cost = 0.4*0.5 + 0.35*0.4 + 0.25*0.3 = 0.415
        # psych cost = 0.4*0.6 + 0.35*0.2 + 0.25*0.1 = 0.335
        # socio cost = 0.35*0.1 + 0.4*0.8 + 0.25*0.2 = 0.405
        # 0.30 * 0.415 + 0.35 * 0.335 + 0.35 * 0.405 = 0.1245 + 0.11725 + 0.14175 = 0.3835
        assert pytest.approx(panel.aggregate_penalty(), abs=1e-6) == 0.3835

        # Custom discipline weights: (0.50, 0.30, 0.20)
        custom_weights = (0.50, 0.30, 0.20)
        expected_custom = 0.50 * 0.415 + 0.30 * 0.335 + 0.20 * 0.405
        assert pytest.approx(panel.aggregate_penalty(discipline_weights=custom_weights), abs=1e-6) == expected_custom

    def test_consequence_vector_backwards_compatibility(self) -> None:
        """Strictly asserts ConsequenceVector with panel_weight=0.0 equals exact 4D operational cost."""
        cv = ConsequenceVector(latency=0.20, maintenance=0.40, risk=0.10, metabolic=0.30)
        # Expected 4D cost: 0.25*0.20 + 0.35*0.40 + 0.25*0.10 + 0.15*0.30 = 0.05 + 0.14 + 0.025 + 0.045 = 0.26
        exact_4d = 0.26

        # 1. Default signature without arguments
        assert pytest.approx(cv.weighted_cost(), abs=1e-9) == exact_4d

        # 2. Explicit panel_weight = 0.0
        assert pytest.approx(cv.weighted_cost(panel_weight=0.0), abs=1e-9) == exact_4d

        # 3. Negative panel_weight falls back to 4D cost
        assert pytest.approx(cv.weighted_cost(panel_weight=-0.5), abs=1e-9) == exact_4d

        # 4. Custom 4D weights without panel weight
        custom_4d_weights = (0.10, 0.20, 0.30, 0.40)
        # 0.1*0.2 + 0.2*0.4 + 0.3*0.1 + 0.4*0.3 = 0.02 + 0.08 + 0.03 + 0.12 = 0.25
        assert pytest.approx(cv.weighted_cost(weights=custom_4d_weights), abs=1e-9) == 0.25

    def test_consequence_vector_with_panel_interpolation(self) -> None:
        """Asserts ConsequenceVector linear interpolation between 4D operational cost and cognitive panel."""
        panel = CognitivePanelScore(
            neurobiology=NeurobiologicalEvaluation(0.8, 0.8, 0.8),
            psychiatry=PsychiatricEvaluation(0.8, 0.8, 0.8),
            sociology=SociologicalEvaluation(0.8, 0.8, 0.8),
        )
        c_panel = panel.aggregate_penalty()
        assert pytest.approx(c_panel, abs=1e-6) == 0.80

        cv = ConsequenceVector(
            latency=0.10,
            maintenance=0.10,
            risk=0.10,
            metabolic=0.10,
            cognitive_panel=panel,
        )
        c_4d = cv.weighted_cost(panel_weight=0.0)
        assert pytest.approx(c_4d, abs=1e-6) == 0.10

        # Linear interpolation at panel_weight = 0.30: (1 - 0.3)*0.10 + 0.3*0.80 = 0.07 + 0.24 = 0.31
        blended = cv.weighted_cost(panel_weight=0.30)
        assert pytest.approx(blended, abs=1e-6) == 0.31

        # Boundary at panel_weight = 1.0
        assert pytest.approx(cv.weighted_cost(panel_weight=1.0), abs=1e-6) == c_panel

        # Boundary clamping above 1.0 (e.g. 1.5)
        assert pytest.approx(cv.weighted_cost(panel_weight=1.5), abs=1e-6) == c_panel

    def test_serialization_and_deserialization_roundtrip(self) -> None:
        """Asserts lossless dictionary and JSON serialization round-trip across all panel dataclasses."""
        neuro = NeurobiologicalEvaluation(0.25, 0.50, 0.75)
        psych = PsychiatricEvaluation(0.15, 0.45, 0.65)
        socio = SociologicalEvaluation(0.35, 0.55, 0.85)
        panel = CognitivePanelScore(neurobiology=neuro, psychiatry=psych, sociology=socio)
        cv = ConsequenceVector(
            latency=0.12,
            maintenance=0.34,
            risk=0.56,
            metabolic=0.78,
            cognitive_panel=panel,
        )

        # 1. Dataclass to_dict and from_dict
        neuro_rt = NeurobiologicalEvaluation.from_dict(neuro.to_dict())
        assert neuro_rt == neuro

        psych_rt = PsychiatricEvaluation.from_dict(psych.to_dict())
        assert psych_rt == psych

        socio_rt = SociologicalEvaluation.from_dict(socio.to_dict())
        assert socio_rt == socio

        panel_rt = CognitivePanelScore.from_dict(panel.to_dict())
        assert pytest.approx(panel_rt.aggregate_penalty(), abs=1e-6) == panel.aggregate_penalty()

        cv_rt = ConsequenceVector.from_dict(cv.to_dict())
        assert cv_rt.latency == cv.latency
        assert cv_rt.maintenance == cv.maintenance
        assert cv_rt.risk == cv.risk
        assert cv_rt.metabolic == cv.metabolic
        assert pytest.approx(cv_rt.cognitive_panel.aggregate_penalty(), abs=1e-6) == panel.aggregate_penalty()

        # 2. JSON dumps and loads round-trip
        json_str = json.dumps(cv.to_dict())
        cv_from_json = ConsequenceVector.from_dict(json.loads(json_str))
        assert pytest.approx(cv_from_json.weighted_cost(panel_weight=0.4), abs=1e-6) == cv.weighted_cost(panel_weight=0.4)

        # 3. Robust fallback from empty or None dict
        assert NeurobiologicalEvaluation.from_dict(None) == NeurobiologicalEvaluation()
        assert PsychiatricEvaluation.from_dict({}) == PsychiatricEvaluation()
        assert SociologicalEvaluation.from_dict({}) == SociologicalEvaluation()
        assert CognitivePanelScore.from_dict(None).aggregate_penalty() == 0.0


# ============================================================================
# 5. Interdisciplinary Arbitration Tests
# ============================================================================


class TestInterdisciplinaryArbitration:
    """Verifies quantum decision arbitration, microglial DAG pruning, and ToM synthesis."""

    def test_quantum_arbiter_penalizes_ruminative_option(self) -> None:
        """Asserts QuantumDecisionArbiter applies panel penalties to demote circular ruminative options."""
        arbiter = QuantumDecisionArbiter(seed=42)

        # Candidate A: Healthy, flexible, low-stress architecture
        healthy_option = {
            "name": "Adaptive Modular Pipeline",
            "description": "Decomposes monolithic state into independent micro-stages with bounded memory.",
            "cognitive_panel": CognitivePanelScore(
                neurobiology=NeurobiologicalEvaluation(0.1, 0.1, 0.1),
                psychiatry=PsychiatricEvaluation(0.05, 0.05, 0.05),
                sociology=SociologicalEvaluation(0.1, 0.1, 0.1),
            ),
        }

        # Candidate B: Ruminative, circular, perseverative architecture
        ruminative_option = {
            "name": "Recursive Monolithic Loop",
            "description": "Continuously retries identical monolithic state traversal in a circular loop.",
            "cognitive_panel": CognitivePanelScore(
                neurobiology=NeurobiologicalEvaluation(0.8, 0.8, 0.8),
                psychiatry=PsychiatricEvaluation(0.95, 0.90, 0.85),
                sociology=SociologicalEvaluation(0.7, 0.7, 0.7),
            ),
        }

        result = arbiter.arbitrate(
            goal="Design high-stability autonomous decision execution pipeline",
            options=[healthy_option, ruminative_option],
            panel_weight=0.60,
        )

        assert "recommended_option" in result
        # The adaptive option must be chosen over the severely ruminative one
        assert result["recommended_option"] == "Adaptive Modular Pipeline"
        ranked = result["ranked_options"]
        assert len(ranked) == 2
        assert ranked[0]["option"] == "Adaptive Modular Pipeline"
        assert ranked[0]["score"] > ranked[1]["score"]
        assert ranked[0]["probability"] > ranked[1]["probability"]

    def test_dag_microglial_pruning_under_human_fatigue(self) -> None:
        """Asserts DAG rollout microglial phagocytic pruning eliminates trajectories with severe user fatigue."""
        arbiter = QuantumDecisionArbiter(seed=42)

        dag = DecisionDAG()
        dag.add_node(DecisionNode("root", "Initial State"))

        # Path 1: Concise, ergonomic trajectory (minimal human fatigue)
        dag.add_node(
            DecisionNode(
                "concise_action",
                "Concise Autonomous Action",
                consequence=ConsequenceVector(
                    latency=0.10,
                    maintenance=0.10,
                    risk=0.10,
                    metabolic=0.10,
                    cognitive_panel=CognitivePanelScore(
                        neurobiology=NeurobiologicalEvaluation(0.1, 0.1, 0.1),
                        psychiatry=PsychiatricEvaluation(0.1, 0.1, 0.1),
                        sociology=SociologicalEvaluation(0.05, 0.05, 0.05),
                    ),
                ),
            )
        )

        # Path 2: Exhausting, high-friction trajectory with identical 4D cost but extreme user fatigue
        dag.add_node(
            DecisionNode(
                "exhausting_action",
                "High Fatigue Verbose Inquiry",
                consequence=ConsequenceVector(
                    latency=0.12,
                    maintenance=0.12,
                    risk=0.10,
                    metabolic=0.10,
                    cognitive_panel=CognitivePanelScore(
                        neurobiology=NeurobiologicalEvaluation(0.8, 0.8, 0.8),
                        psychiatry=PsychiatricEvaluation(0.8, 0.8, 0.8),
                        sociology=SociologicalEvaluation(0.95, 0.95, 0.95),
                    ),
                ),
            )
        )

        dag.add_edge(DecisionEdge("root", "concise_action", "Execute Concise"))
        dag.add_edge(DecisionEdge("root", "exhausting_action", "Prompt User Verbose"))

        goal = "Autonomous Pipeline Execution with Minimal Cognitive Friction"

        # Under panel_weight=0.0: Both trajectories survive (4D operational costs are close)
        res_unweighted = arbiter.arbitrate_dag(
            dag,
            goal=goal,
            prune_threshold=0.25,
            zeno_lock_threshold=0.50,
            panel_weight=0.0,
        )
        exhausting_unweighted = next(
            t for t in res_unweighted["trajectories"] if t["path_nodes"][-1] == "exhausting_action"
        )
        assert exhausting_unweighted["is_pruned"] is False

        # Under panel_weight=0.85: Microglial phagocytic pruning eliminates the high-fatigue branch
        res_weighted = arbiter.arbitrate_dag(
            dag,
            goal=goal,
            prune_threshold=0.25,
            zeno_lock_threshold=0.50,
            panel_weight=0.85,
        )
        concise_weighted = next(
            t for t in res_weighted["trajectories"] if t["path_nodes"][-1] == "concise_action"
        )
        exhausting_weighted = next(
            t for t in res_weighted["trajectories"] if t["path_nodes"][-1] == "exhausting_action"
        )

        assert concise_weighted["is_pruned"] is False
        assert exhausting_weighted["is_pruned"] is True
        assert exhausting_weighted["prune_reason"] is not None
        assert "Dominated by winner" in exhausting_weighted["prune_reason"]

    def test_tom_interdisciplinary_analysis(self) -> None:
        """Asserts TheoryOfMindAnalyzer.analyze_interdisciplinary adheres strictly to 3-tuple return contract."""
        tom = TheoryOfMindAnalyzer()
        messages = [
            {"role": "user", "content": "How can we structure the biomorphic cache?"},
            {"role": "assistant", "content": "We should use calibrated Lindblad-Ebbinghaus dephasing."},
        ]
        project_state = {"test_failures": 0, "milestone_urgency": False}

        urgency, seeds, panel = tom.analyze_interdisciplinary(messages, project_state=project_state)

        assert isinstance(urgency, float)
        assert 0.2 <= urgency <= 5.0
        assert isinstance(seeds, list)
        for s in seeds:
            assert isinstance(s, DreamSeed)
        assert isinstance(panel, CognitivePanelScore)
        assert isinstance(panel.neurobiology, NeurobiologicalEvaluation)
        assert isinstance(panel.psychiatry, PsychiatricEvaluation)
        assert isinstance(panel.sociology, SociologicalEvaluation)

    def test_tom_synthesizes_interdisciplinary_dream_seeds(self) -> None:
        """Asserts TheoryOfMindAnalyzer synthesizes targeted dream seeds for prolonged turns, rumination, and fatigue."""
        tom = TheoryOfMindAnalyzer()

        # 1. Prolonged conversation (> 20 messages) produces synaptic_homeostasis seed
        long_messages = [{"role": "user", "content": f"turn number {i}"} for i in range(22)]
        _, seeds_long = tom.analyze_conversation(long_messages)
        homeostasis_seed = next((s for s in seeds_long if s.topic == "synaptic_homeostasis"), None)
        assert homeostasis_seed is not None
        assert isinstance(homeostasis_seed, DreamSeed)
        assert homeostasis_seed.urgency >= 2.5
        assert "synaptic_homeostasis" in homeostasis_seed.context_keys

        # 2. Circular topic repetition produces anti_rumination seed
        repetitive_messages = [
            {"role": "user", "content": "memory cache fault"},
            {"role": "assistant", "content": "memory allocator review"},
            {"role": "user", "content": "still memory issue"},
        ]
        _, seeds_rumination = tom.analyze_conversation(repetitive_messages)
        rumination_seed = next((s for s in seeds_rumination if s.topic == "anti_rumination"), None)
        assert rumination_seed is not None
        assert isinstance(rumination_seed, DreamSeed)
        assert rumination_seed.urgency >= 2.8
        assert "anti_rumination" in rumination_seed.context_keys

        # 3. Terse cadence (>= 50% short turns) produces user_fatigue_reduction seed
        terse_messages = [
            {"role": "user", "content": "ok"},
            {"role": "user", "content": "next"},
        ]
        _, seeds_fatigue = tom.analyze_conversation(terse_messages)
        fatigue_seed = next((s for s in seeds_fatigue if s.topic == "user_fatigue_reduction"), None)
        assert fatigue_seed is not None
        assert isinstance(fatigue_seed, DreamSeed)
        assert fatigue_seed.urgency >= 2.2
        assert "user_fatigue_reduction" in fatigue_seed.context_keys


# ============================================================================
# 6. Interdisciplinary Invariants & Regressions Tests
# ============================================================================


class TestInterdisciplinaryInvariantsAndRegressions:
    """Verifies bounds in [0.0, 1.0], zero/nan/inf safety, and subconscious hook/daemon integration."""

    def test_all_evaluation_values_bounded_zero_one(self) -> None:
        """Property test asserting all evaluation sub-metrics and aggregate costs remain strictly in [0.0, 1.0]."""
        # Testing extreme inputs
        test_inputs = [-100.0, -1.0, 0.0, 0.5, 1.0, 2.0, 1000.0]
        for val in test_inputs:
            neuro = NeurobiologicalEvaluation(val, val, val)
            psych = PsychiatricEvaluation(val, val, val)
            socio = SociologicalEvaluation(val, val, val)

            # Invariant: clamped bounded values
            assert 0.0 <= _clamp01(neuro.synaptic_saturation) <= 1.0
            assert 0.0 <= _clamp01(neuro.energy_expenditure) <= 1.0
            assert 0.0 <= _clamp01(neuro.sleep_consolidation_affinity) <= 1.0

            assert 0.0 <= _clamp01(psych.rumination_risk) <= 1.0
            assert 0.0 <= _clamp01(psych.perseveration_penalty) <= 1.0
            assert 0.0 <= _clamp01(psych.threat_distortion) <= 1.0

            assert 0.0 <= _clamp01(socio.social_misalignment) <= 1.0
            assert 0.0 <= _clamp01(socio.user_fatigue_impact) <= 1.0
            assert 0.0 <= _clamp01(socio.coordination_friction) <= 1.0

            # Deserialization clamp invariant
            neuro_clamped = NeurobiologicalEvaluation.from_dict({
                "synaptic_saturation": val,
                "energy_expenditure": val,
                "sleep_consolidation_affinity": val,
            })
            assert 0.0 <= neuro_clamped.synaptic_saturation <= 1.0
            assert 0.0 <= neuro_clamped.energy_expenditure <= 1.0
            assert 0.0 <= neuro_clamped.sleep_consolidation_affinity <= 1.0
            assert 0.0 <= neuro_clamped.aggregate_cost() <= 1.0

    def test_zero_nan_inf_safety(self) -> None:
        """Asserts TheoryOfMindAnalyzer and arbiter gracefully withstand NaN, Inf, None, and bad types without error."""
        tom = TheoryOfMindAnalyzer()

        # 1. Test defensive helper functions directly
        assert _safe_float(None, 4.2) == 4.2
        assert _safe_float("invalid", 4.2) == 4.2
        assert math.isnan(_safe_float(float("nan"), 4.2))
        assert _safe_float(3.14) == 3.14

        assert _safe_int(None, 7) == 7
        assert _safe_int("invalid", 7) == 7
        assert _safe_int("9") == 9

        # Test tom_analyzer module helpers
        assert tom_safe_float(None, 1.5) == 1.5
        assert tom_safe_float("bad", 1.5) == 1.5
        assert tom_safe_float(float("nan"), 1.5) == 1.5
        assert tom_safe_float(float("inf"), 1.5) == 1.5
        assert tom_safe_float("-inf", 1.5) == 1.5
        assert tom_safe_float(2.72) == 2.72

        assert tom_safe_int(None, 4) == 4
        assert tom_safe_int("bad", 4) == 4
        assert tom_safe_int(float("nan"), 4) == 4
        assert tom_safe_int(float("inf"), 4) == 4
        assert tom_safe_int("42") == 42
        assert tom_safe_int(42.9) == 42

        # Test clamp01 boundary and NaN/Inf handling
        assert _clamp01(float("nan"), 0.5) == 0.5
        assert _clamp01(float("inf"), 0.5) == 0.5
        assert _clamp01(-10.0) == 0.0
        assert _clamp01(10.0) == 1.0

        # 2. Adversarial inputs to evaluate_cognitive_panel
        # None in repetition_count, test_failures, turn_count, repeated_tool_failures
        adversarial_context: dict[str, Any] = {
            "turn_count": None,
            "buffer_density": float("nan"),
            "repetition_count": None,
            "repeated_tool_failures": None,
            "energy_expenditure": float("inf"),
        }
        adversarial_project_state: dict[str, Any] = {
            "test_failures": None,
            "repetition_count": None,
            "user_fatigue": "not_a_bool",
        }
        adversarial_messages: list[dict[str, Any]] = [
            {"role": None, "content": None},
            {"role": "user", "content": 42},
            {},
        ]

        # Must NOT raise TypeError or ValueError
        panel = tom.evaluate_cognitive_panel(
            adversarial_messages,
            project_state=adversarial_project_state,
            context=adversarial_context,
        )
        assert isinstance(panel, CognitivePanelScore)
        assert math.isfinite(panel.aggregate_penalty())
        assert 0.0 <= panel.aggregate_penalty() <= 1.0

        # Adversarial inputs to analyze_conversation
        urgency, seeds = tom.analyze_conversation(
            adversarial_messages, project_state=adversarial_project_state
        )
        assert math.isfinite(urgency)
        assert 0.2 <= urgency <= 5.0
        assert isinstance(seeds, list)

    def test_subconscious_daemon_and_hook_integration(self) -> None:
        """Asserts integration with FastBiomorphicMemory: calibrated micro-step decay and gist crystallization."""
        mem = FastBiomorphicMemory(capacity=64)

        # 1. Micro-step scaling (R1): Transient decision remains retained across >= 20 tool steps
        mem.record_transient(
            key="transient_arch_decision",
            content="Use event-driven pub-sub for cross-module telemetry dispatch.",
            salience=0.50,
        )
        # Advance 20 tool steps at dt=0.2 (total dt = 4.0)
        for _ in range(20):
            mem.step(dt=0.2)

        engram = next((e for e in mem.engrams if e["key"] == "transient_arch_decision"), None)
        assert engram is not None
        # Invariant: S=0.50 engram maintains fidelity >= 0.885 after 20 tool steps
        assert engram["fidelity"] >= 0.885
        assert engram["fidelity"] > 0.70  # Has NOT reached pruning threshold

        # 2. Fuzzy-Trace Semantic Gist Crystallization (R2)
        assert is_actionable_resolution("Architectural Decision: We must adopt event-driven pub-sub.")
        gist_key, distilled = distill_semantic_gist(
            "Architectural Decision: We must adopt event-driven pub-sub."
        )
        assert gist_key.startswith("gist_")
        assert len(distilled) > 0

        # Decay engram below 0.70 threshold to trigger crystallization
        for _ in range(35):
            mem.step(dt=1.0)
        assert engram["fidelity"] < 0.70

        # Run pruning: actionable engram crystallizes semantic gist before detail eviction
        pruned = mem.prune_obsolete(fidelity_threshold=0.70)
        assert len(pruned) >= 1

        # Confirm semantic gist is recorded as a core anchor (S >= 1.80, is_core_anchor=True)
        gists = [e for e in mem.engrams if e.get("category") == "semantic_gist"]
        assert len(gists) >= 1
        gist_engram = gists[0]
        assert gist_engram["is_core_anchor"] is True
        assert gist_engram["salience"] >= 1.80
        assert gist_engram["fidelity"] >= 0.999
