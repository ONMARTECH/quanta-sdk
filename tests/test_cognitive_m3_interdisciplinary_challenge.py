"""tests/test_cognitive_m3_interdisciplinary_challenge.py

Empirical stress testing and verification harness for Milestone 3 (R3):
Multi-Disciplinary Cognitive Evaluation Panel.

Authored by teamwork_preview_challenger_m3_1 (EMPIRICAL CHALLENGER):
1. Multi-criteria decision arbitration: candidate options with varying psychiatric
   rumination and human fatigue penalties. Verify options with high rumination are
   systematically penalized and ranked below cognitively healthy alternatives.
2. ConsequenceVector: verify cv.weighted_cost() returns identical values to baseline
   4D cost when panel_weight=0.0. Verify linear interpolation when panel_weight=0.3.
3. Boundary values: extreme values (1.0, 0.0), None, non-finite (NaN, Inf, -Inf) inputs.
   Verify _clamp01 protects against crashes and guarantees [0.0, 1.0] bounds.
4. Round-trip serialization: to_dict and from_dict across all 4 evaluation dataclasses
   and ConsequenceVector (including JSON round-trip and flat backward compatibility).
5. TheoryOfMindAnalyzer interdisciplinary conversational signals and DreamSeed synthesis.
6. DAG rollout microglial pruning under cognitive fatigue and psychiatric perseveration.
"""

from __future__ import annotations

import json
import math
import random
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
)
from quanta.cognitive.tom_analyzer import TheoryOfMindAnalyzer

# ============================================================================
# 1. Multi-Criteria Decision Arbitration & Penalty Stress Tests
# ============================================================================


class TestMultiCriteriaDecisionArbitrationStress:
    """Empirical challenge 1: Multi-criteria decision arbitration must penalize

    psychiatric rumination and user fatigue, systematically ranking unhealthy
    options below cognitively sound alternatives.
    """

    @pytest.fixture
    def arbiter(self) -> QuantumDecisionArbiter:
        return QuantumDecisionArbiter(dim=64, num_heads=4, seed=42)

    def test_rumination_penalizes_and_demotes_candidate(
        self,
        arbiter: QuantumDecisionArbiter,
    ) -> None:
        """Verify candidate with severe psychiatric rumination is ranked below healthy choice."""
        goal = "Select optimal database indexing strategy for high-throughput queries"

        healthy_panel = CognitivePanelScore(
            neurobiology=NeurobiologicalEvaluation(
                synaptic_saturation=0.10,
                energy_expenditure=0.15,
                sleep_consolidation_affinity=0.10,
            ),
            psychiatry=PsychiatricEvaluation(
                rumination_risk=0.05,
                perseveration_penalty=0.05,
                threat_distortion=0.05,
            ),
            sociology=SociologicalEvaluation(
                social_misalignment=0.05,
                user_fatigue_impact=0.10,
                coordination_friction=0.05,
            ),
        )

        ruminative_panel = CognitivePanelScore(
            neurobiology=NeurobiologicalEvaluation(
                synaptic_saturation=0.80,
                energy_expenditure=0.75,
                sleep_consolidation_affinity=0.85,
            ),
            psychiatry=PsychiatricEvaluation(
                rumination_risk=0.95,
                perseveration_penalty=0.90,
                threat_distortion=0.85,
            ),
            sociology=SociologicalEvaluation(
                social_misalignment=0.70,
                user_fatigue_impact=0.80,
                coordination_friction=0.65,
            ),
        )

        options = [
            {
                "name": "B-Tree Indexing on Primary Key",
                "option": "B-Tree Indexing on Primary Key",
                "cognitive_panel": healthy_panel,
            },
            {
                "name": "Circular Deliberation Over 15 Index Algorithms",
                "option": "Circular Deliberation Over 15 Index Algorithms",
                "cognitive_panel": ruminative_panel,
            },
        ]

        res = arbiter.arbitrate(
            goal=goal,
            options=options,
            panel_weight=0.5,
        )

        ranking = res["ranked_options"]
        assert len(ranking) == 2
        winner = ranking[0]
        loser = ranking[1]

        assert winner["option"] == "B-Tree Indexing on Primary Key"
        assert loser["option"] == "Circular Deliberation Over 15 Index Algorithms"
        assert winner["score"] > loser["score"]
        assert winner["panel_penalty"] < loser["panel_penalty"]
        assert loser["panel_penalty"] >= 0.70

    def test_rumination_overcomes_raw_alignment_advantage(
        self,
        arbiter: QuantumDecisionArbiter,
    ) -> None:
        """When an option has slight lexical/raw advantage, severe rumination

        must depress its alignment and demote it when panel_weight is active.
        """
        goal = "Resolve cache contention and thread lock contention in memory subsystem"

        # Option A: Clean, healthy approach
        panel_healthy = CognitivePanelScore(
            psychiatry=PsychiatricEvaluation(
                rumination_risk=0.02,
                perseveration_penalty=0.02,
                threat_distortion=0.02,
            ),
            sociology=SociologicalEvaluation(user_fatigue_impact=0.05),
        )

        # Option B: High rumination (looping endlessly on locks)
        panel_ruminative = CognitivePanelScore(
            psychiatry=PsychiatricEvaluation(
                rumination_risk=0.98,
                perseveration_penalty=0.95,
                threat_distortion=0.90,
            ),
            sociology=SociologicalEvaluation(user_fatigue_impact=0.90),
        )

        options = [
            {
                "name": "Lock-Free Ring Buffer Queue",
                "option": "Lock-Free Ring Buffer Queue",
                "cognitive_panel": panel_healthy,
            },
            {
                "name": "Exhaustive Lock Contention Retries in Loop",
                "option": "Exhaustive Lock Contention Retries in Loop",
                "cognitive_panel": panel_ruminative,
            },
        ]

        res_with_panel = arbiter.arbitrate(
            goal=goal,
            options=options,
            panel_weight=0.60,
        )

        assert res_with_panel["recommended_option"] == "Lock-Free Ring Buffer Queue"
        assert (
            res_with_panel["ranked_options"][0]["score"]
            > res_with_panel["ranked_options"][1]["score"]
        )

    def test_human_fatigue_penalizes_and_ranks_below_healthy(
        self,
        arbiter: QuantumDecisionArbiter,
    ) -> None:
        """Options imposing high cognitive fatigue on user are demoted."""
        goal = "Communicate architectural status report to engineering lead"

        panel_concise = CognitivePanelScore(
            sociology=SociologicalEvaluation(
                user_fatigue_impact=0.10,
                social_misalignment=0.05,
                coordination_friction=0.05,
            ),
        )

        panel_fatiguing = CognitivePanelScore(
            sociology=SociologicalEvaluation(
                user_fatigue_impact=0.95,
                social_misalignment=0.85,
                coordination_friction=0.80,
            ),
        )

        options = [
            {
                "name": "Concise Executive Summary with Actionable Insights",
                "option": "Concise Executive Summary with Actionable Insights",
                "cognitive_panel": panel_concise,
            },
            {
                "name": "Unstructured 50-Page Trace Dump Requiring Manual Sifting",
                "option": "Unstructured 50-Page Trace Dump Requiring Manual Sifting",
                "cognitive_panel": panel_fatiguing,
            },
        ]

        res = arbiter.arbitrate(
            goal=goal,
            options=options,
            panel_weight=0.50,
        )

        assert (
            res["recommended_option"]
            == "Concise Executive Summary with Actionable Insights"
        )
        assert res["ranked_options"][0]["score"] > res["ranked_options"][1]["score"]
        assert (
            res["ranked_options"][1]["cognitive_panel"]["sociology"][
                "user_fatigue_impact"
            ]
            == 0.95
        )

    def test_consequence_vector_in_arbitrate_candidates(
        self,
        arbiter: QuantumDecisionArbiter,
    ) -> None:
        """Arbitrate seamlessly extracts cognitive panel from consequence dict/vector."""
        cv_clean = ConsequenceVector(
            latency=0.1,
            maintenance=0.1,
            cognitive_panel=CognitivePanelScore(
                psychiatry=PsychiatricEvaluation(rumination_risk=0.05)
            ),
        )
        cv_ruminative = ConsequenceVector(
            latency=0.1,
            maintenance=0.1,
            cognitive_panel=CognitivePanelScore(
                psychiatry=PsychiatricEvaluation(
                    rumination_risk=0.95,
                    perseveration_penalty=0.90,
                )
            ),
        )

        options = [
            {"name": "Clean Path", "consequence": cv_clean},
            {"name": "Ruminative Path", "consequence": cv_ruminative},
        ]

        res = arbiter.arbitrate(
            goal="Execute data pipeline batch sync",
            options=options,
            panel_weight=0.5,
        )
        assert res["recommended_option"] == "Clean Path"
        assert res["ranked_options"][0]["score"] > res["ranked_options"][1]["score"]

    def test_dag_rollout_microglial_pruning_with_panel_weight(
        self,
        arbiter: QuantumDecisionArbiter,
    ) -> None:
        """DAG rollout with panel_weight > 0 microglially prunes branches imposing

        extreme user fatigue and rumination.
        """
        dag = DecisionDAG()

        root = DecisionNode(
            node_id="root",
            label="Root Inception",
            depth=0,
            consequence=ConsequenceVector(),
        )
        dag.add_node(root)

        # Branch 1: Streamlined canary with low fatigue
        b1_cp = CognitivePanelScore(
            psychiatry=PsychiatricEvaluation(rumination_risk=0.05),
            sociology=SociologicalEvaluation(user_fatigue_impact=0.05),
        )
        b1 = DecisionNode(
            node_id="b1",
            label="Automated Progressive Canary Rollout",
            depth=1,
            consequence=ConsequenceVector(
                latency=0.1,
                maintenance=0.1,
                risk=0.1,
                metabolic=0.1,
                cognitive_panel=b1_cp,
            ),
        )
        dag.add_node(b1)
        dag.add_edge(
            DecisionEdge(
                source_id="root",
                target_id="b1",
                action_label="Automated Progressive Canary Rollout",
            )
        )

        # Branch 2: High cognitive fatigue & perseveration
        b2_cp = CognitivePanelScore(
            neurobiology=NeurobiologicalEvaluation(synaptic_saturation=0.9),
            psychiatry=PsychiatricEvaluation(
                rumination_risk=0.95,
                perseveration_penalty=0.95,
            ),
            sociology=SociologicalEvaluation(
                user_fatigue_impact=0.98,
                social_misalignment=0.90,
            ),
        )
        b2 = DecisionNode(
            node_id="b2",
            label="Manual 100-Stage Interactive Confirmation Gating",
            depth=1,
            consequence=ConsequenceVector(
                latency=0.1,
                maintenance=0.1,
                risk=0.1,
                metabolic=0.1,
                cognitive_panel=b2_cp,
            ),
        )
        dag.add_node(b2)
        dag.add_edge(
            DecisionEdge(
                source_id="root",
                target_id="b2",
                action_label="Manual 100-Stage Interactive Confirmation Gating",
            )
        )

        goal = "Deploy service mesh with canary deployment"
        # When panel_weight=0.5, b2 incurs heavy penalty and is pruned
        res_weighted = arbiter.arbitrate_dag(
            dag=dag,
            goal=goal,
            panel_weight=0.5,
            prune_threshold=0.15,
        )
        trajs_weighted = {
            t["path_nodes"][-1]: t for t in res_weighted["trajectories"]
        }
        traj_b1 = trajs_weighted["b1"]
        traj_b2 = trajs_weighted["b2"]

        assert traj_b1["cumulative_cost"] < traj_b2["cumulative_cost"]
        assert res_weighted["winning_trajectory"]["trajectory_id"] == traj_b1["trajectory_id"]
        assert traj_b2["is_pruned"] is True
        assert "Dominated by winner" in str(traj_b2["prune_reason"])

        # When panel_weight=0.0, consequence is strictly 4D cost (both are 0.1)
        res_zero = arbiter.arbitrate_dag(
            dag=dag,
            goal=goal,
            panel_weight=0.0,
            prune_threshold=0.15,
        )
        trajs_zero = {t["path_nodes"][-1]: t for t in res_zero["trajectories"]}
        assert trajs_zero["b1"]["cumulative_cost"] == pytest.approx(
            trajs_zero["b2"]["cumulative_cost"], abs=1e-4
        )


# ============================================================================
# 2. ConsequenceVector Baseline & Interpolation Tests
# ============================================================================


class TestConsequenceVectorBaselineAndInterpolation:
    """Empirical challenge 2: ConsequenceVector weighted cost baseline identity

    and linear interpolation invariants.
    """

    def test_baseline_4d_cost_identity_at_panel_weight_zero(self) -> None:
        """When panel_weight == 0.0, returns exact 4D cost."""
        for _ in range(25):
            lat = random.uniform(0.0, 1.0)
            mnt = random.uniform(0.0, 1.0)
            rsk = random.uniform(0.0, 1.0)
            met = random.uniform(0.0, 1.0)

            # Assign arbitrarily high panel score
            cp = CognitivePanelScore(
                neurobiology=NeurobiologicalEvaluation(
                    synaptic_saturation=random.uniform(0.7, 1.0),
                    energy_expenditure=random.uniform(0.7, 1.0),
                ),
                psychiatry=PsychiatricEvaluation(
                    rumination_risk=random.uniform(0.7, 1.0),
                    perseveration_penalty=random.uniform(0.7, 1.0),
                ),
                sociology=SociologicalEvaluation(
                    user_fatigue_impact=random.uniform(0.7, 1.0),
                ),
            )

            cv = ConsequenceVector(
                latency=lat,
                maintenance=mnt,
                risk=rsk,
                metabolic=met,
                cognitive_panel=cp,
            )

            expected_4d = 0.25 * lat + 0.35 * mnt + 0.25 * rsk + 0.15 * met
            cost_default = cv.weighted_cost()
            cost_zero = cv.weighted_cost(panel_weight=0.0)

            assert cost_default == pytest.approx(expected_4d, abs=1e-9)
            assert cost_zero == pytest.approx(expected_4d, abs=1e-9)

    def test_baseline_at_negative_panel_weight(self) -> None:
        """Negative panel_weight is clamped/treated as <= 0.0, returning exact 4D cost."""
        cv = ConsequenceVector(
            latency=0.3,
            maintenance=0.4,
            risk=0.5,
            metabolic=0.6,
            cognitive_panel=CognitivePanelScore(
                psychiatry=PsychiatricEvaluation(rumination_risk=0.99)
            ),
        )
        expected_4d = 0.25 * 0.3 + 0.35 * 0.4 + 0.25 * 0.5 + 0.15 * 0.6
        assert cv.weighted_cost(panel_weight=-0.5) == pytest.approx(expected_4d, abs=1e-9)
        assert cv.weighted_cost(panel_weight=-10.0) == pytest.approx(expected_4d, abs=1e-9)

    def test_linear_interpolation_at_panel_weight_0_3(self) -> None:
        """When panel_weight == 0.3, cost is strictly (1 - 0.3)*C_4D + 0.3*C_panel."""
        for _ in range(50):
            lat = random.uniform(0.0, 1.0)
            mnt = random.uniform(0.0, 1.0)
            rsk = random.uniform(0.0, 1.0)
            met = random.uniform(0.0, 1.0)

            sat = random.uniform(0.0, 1.0)
            rum = random.uniform(0.0, 1.0)
            fat = random.uniform(0.0, 1.0)

            cp = CognitivePanelScore(
                neurobiology=NeurobiologicalEvaluation(synaptic_saturation=sat),
                psychiatry=PsychiatricEvaluation(rumination_risk=rum),
                sociology=SociologicalEvaluation(user_fatigue_impact=fat),
            )

            cv = ConsequenceVector(
                latency=lat,
                maintenance=mnt,
                risk=rsk,
                metabolic=met,
                cognitive_panel=cp,
            )

            c_4d = 0.25 * lat + 0.35 * mnt + 0.25 * rsk + 0.15 * met
            c_panel = cp.aggregate_penalty()
            expected = 0.7 * c_4d + 0.3 * c_panel

            actual = cv.weighted_cost(panel_weight=0.3)
            assert actual == pytest.approx(expected, abs=1e-9)

    def test_panel_weight_boundary_one_and_above(self) -> None:
        """When panel_weight == 1.0 or > 1.0, cost strictly equals c_panel."""
        cp = CognitivePanelScore(
            neurobiology=NeurobiologicalEvaluation(
                synaptic_saturation=0.5,
                energy_expenditure=0.6,
                sleep_consolidation_affinity=0.7,
            ),
            psychiatry=PsychiatricEvaluation(
                rumination_risk=0.8,
                perseveration_penalty=0.9,
                threat_distortion=0.4,
            ),
            sociology=SociologicalEvaluation(
                social_misalignment=0.3,
                user_fatigue_impact=0.7,
                coordination_friction=0.2,
            ),
        )

        cv = ConsequenceVector(
            latency=0.1,
            maintenance=0.2,
            risk=0.1,
            metabolic=0.1,
            cognitive_panel=cp,
        )

        expected_panel = cp.aggregate_penalty()
        assert cv.weighted_cost(panel_weight=1.0) == pytest.approx(expected_panel, abs=1e-9)
        # Clamped above 1.0
        assert cv.weighted_cost(panel_weight=1.5) == pytest.approx(expected_panel, abs=1e-9)

    def test_custom_consequence_weights_with_panel_weight(self) -> None:
        """Custom consequence weights tuple correctly scales 4D portion."""
        cv = ConsequenceVector(
            latency=0.2,
            maintenance=0.3,
            risk=0.4,
            metabolic=0.5,
            cognitive_panel=CognitivePanelScore(
                psychiatry=PsychiatricEvaluation(rumination_risk=0.8)
            ),
        )
        custom_weights = (0.1, 0.2, 0.3, 0.4)
        c_4d = 0.1 * 0.2 + 0.2 * 0.3 + 0.3 * 0.4 + 0.4 * 0.5
        c_panel = cv.cognitive_panel.aggregate_penalty()

        w_p = 0.4
        expected = (1.0 - w_p) * c_4d + w_p * c_panel
        actual = cv.weighted_cost(weights=custom_weights, panel_weight=w_p)
        assert actual == pytest.approx(expected, abs=1e-9)


# ============================================================================
# 3. Boundary Values & Non-Finite Inputs Stress Tests
# ============================================================================


class TestBoundaryValuesAndRobustness:
    """Empirical challenge 3: Boundary values (1.0, 0.0), None, non-finite

    (NaN, Inf, -Inf) inputs and _clamp01 robustness.
    """

    def test_clamp01_all_cases(self) -> None:
        """Exhaustively verify _clamp01 on nominal, out-of-bound, and invalid inputs."""
        # Exact boundaries
        assert _clamp01(0.0) == 0.0
        assert _clamp01(1.0) == 1.0
        assert _clamp01(0.5) == 0.5

        # Out of bounds
        assert _clamp01(-0.001) == 0.0
        assert _clamp01(-100.0) == 0.0
        assert _clamp01(1.001) == 1.0
        assert _clamp01(9999.0) == 1.0

        # Non-finite floats
        assert _clamp01(float("nan")) == 0.0
        assert _clamp01(float("inf")) == 0.0
        assert _clamp01(-float("inf")) == 0.0

        # Custom default for non-finite / None
        assert _clamp01(float("nan"), default=0.42) == 0.42
        assert _clamp01(float("inf"), default=0.75) == 0.75
        assert _clamp01(None, default=0.55) == 0.55

        # None and types
        assert _clamp01(None) == 0.0
        assert _clamp01("invalid_string") == 0.0
        assert _clamp01("0.85") == pytest.approx(0.85, abs=1e-6)
        assert _clamp01([]) == 0.0
        assert _clamp01({}) == 0.0
        assert _clamp01(object()) == 0.0

    def test_dataclasses_from_dict_non_finite_and_none(self) -> None:
        """Verify from_dict protects against NaN, Inf, None across all dataclasses."""
        adversarial_dict = {
            "synaptic_saturation": float("nan"),
            "energy_expenditure": float("inf"),
            "sleep_consolidation_affinity": -float("inf"),
            "rumination_risk": None,
            "perseveration_penalty": "not_a_number",
            "threat_distortion": 25.0,  # Clamped to 1.0
            "social_misalignment": -5.0,  # Clamped to 0.0
            "user_fatigue_impact": float("nan"),
            "coordination_friction": float("inf"),
        }

        # Neurobiology
        neuro = NeurobiologicalEvaluation.from_dict(adversarial_dict)
        assert neuro.synaptic_saturation == 0.0
        assert neuro.energy_expenditure == 0.0
        assert neuro.sleep_consolidation_affinity == 0.0
        assert math.isfinite(neuro.aggregate_cost())

        # Psychiatry
        psych = PsychiatricEvaluation.from_dict(adversarial_dict)
        assert psych.rumination_risk == 0.0
        assert psych.perseveration_penalty == 0.0
        assert psych.threat_distortion == 1.0
        assert math.isfinite(psych.aggregate_cost())

        # Sociology
        soc = SociologicalEvaluation.from_dict(adversarial_dict)
        assert soc.social_misalignment == 0.0
        assert soc.user_fatigue_impact == 0.0
        assert soc.coordination_friction == 0.0
        assert math.isfinite(soc.aggregate_cost())

        # CognitivePanelScore
        panel = CognitivePanelScore.from_dict({
            "neurobiology": adversarial_dict,
            "psychiatry": adversarial_dict,
            "sociology": adversarial_dict,
        })
        assert math.isfinite(panel.aggregate_penalty())
        assert 0.0 <= panel.aggregate_penalty() <= 1.0

        # ConsequenceVector
        cv = ConsequenceVector.from_dict({
            "latency": float("nan"),
            "maintenance": float("inf"),
            "risk": None,
            "metabolic": -10.0,
            "cognitive_panel": panel.to_dict(),
        })
        assert cv.latency == 0.0
        assert cv.maintenance == 0.0
        assert cv.risk == 0.0
        assert cv.metabolic == 0.0
        assert math.isfinite(cv.weighted_cost(panel_weight=0.5))

    def test_from_dict_null_and_malformed_fallbacks(self) -> None:
        """from_dict handles None, empty dict, or non-dict inputs gracefully."""
        for cls in (
            NeurobiologicalEvaluation,
            PsychiatricEvaluation,
            SociologicalEvaluation,
            CognitivePanelScore,
            ConsequenceVector,
        ):
            assert isinstance(cls.from_dict(None), cls)
            assert isinstance(cls.from_dict({}), cls)
            assert isinstance(cls.from_dict("malformed"), cls)  # type: ignore[arg-type]
            assert isinstance(cls.from_dict([1, 2, 3]), cls)  # type: ignore[arg-type]

    def test_tom_analyzer_adversarial_inputs_survive(self) -> None:
        """TheoryOfMindAnalyzer handles empty, None, and adversarial signals without crashing."""
        tom = TheoryOfMindAnalyzer()

        # Empty messages
        res_empty = tom.evaluate_cognitive_panel([])
        assert isinstance(res_empty, CognitivePanelScore)
        assert res_empty.aggregate_penalty() == 0.0

        # Malformed messages with None content, non-string types, and float NaN stress
        malformed_msgs: list[dict[str, Any]] = [
            {"role": None, "content": None},
            {"role": "user", "content": 12345},
            {"content": ""},
            {"invalid_key": "some_value"},
        ]
        res_malformed = tom.evaluate_cognitive_panel(
            messages=malformed_msgs,
            project_state={"test_failures": float("nan"), "repetition_count": 0},
            context={"turn_count": 100, "buffer_density": float("nan")},
        )
        assert isinstance(res_malformed, CognitivePanelScore)
        assert math.isfinite(res_malformed.aggregate_penalty())
        assert 0.0 <= res_malformed.aggregate_penalty() <= 1.0

    def test_tom_analyzer_vulnerability_none_repetition_count(self) -> None:
        """Verifies that project_state['repetition_count'] = None is safely handled

        via _safe_int fallback without raising an unhandled TypeError.
        """
        tom = TheoryOfMindAnalyzer()
        res = tom.evaluate_cognitive_panel(
            messages=[{"role": "user", "content": "check status"}],
            project_state={"repetition_count": None},
        )
        assert isinstance(res, CognitivePanelScore)
        assert math.isfinite(res.aggregate_penalty())

    def test_tom_analyzer_vulnerability_none_turn_count(self) -> None:
        """Verifies that context['turn_count'] = None is safely handled

        via _safe_float fallback without raising an unhandled TypeError.
        """
        tom = TheoryOfMindAnalyzer()
        res = tom.evaluate_cognitive_panel(
            messages=[{"role": "user", "content": "check status"}],
            context={"turn_count": None},
        )
        assert isinstance(res, CognitivePanelScore)
        assert math.isfinite(res.aggregate_penalty())

    def test_tom_analyzer_vulnerability_none_tool_failures(self) -> None:
        """Verifies that context['repeated_tool_failures'] = None is safely handled

        via _safe_int fallback without raising an unhandled TypeError.
        """
        tom = TheoryOfMindAnalyzer()
        res = tom.evaluate_cognitive_panel(
            messages=[{"role": "user", "content": "check status"}],
            context={"repeated_tool_failures": None},
        )
        assert isinstance(res, CognitivePanelScore)
        assert math.isfinite(res.aggregate_penalty())


# ============================================================================
# 4. Round-Trip Serialization Across Dataclasses
# ============================================================================


class TestRoundTripSerializationAllDataclasses:
    """Empirical challenge 4: Round-trip serialization (to_dict / from_dict)

    across all 4 evaluation dataclasses and ConsequenceVector.
    """

    def test_neurobiological_serialization_roundtrip(self) -> None:
        """NeurobiologicalEvaluation round-trip serialization and JSON compatibility."""
        test_cases = [
            NeurobiologicalEvaluation(),
            NeurobiologicalEvaluation(
                synaptic_saturation=0.0,
                energy_expenditure=0.0,
                sleep_consolidation_affinity=0.0,
            ),
            NeurobiologicalEvaluation(
                synaptic_saturation=1.0,
                energy_expenditure=1.0,
                sleep_consolidation_affinity=1.0,
            ),
            NeurobiologicalEvaluation(
                synaptic_saturation=0.25,
                energy_expenditure=0.50,
                sleep_consolidation_affinity=0.75,
            ),
        ]

        for orig in test_cases:
            d = orig.to_dict()
            assert isinstance(d, dict)
            # JSON round-trip
            json_str = json.dumps(d)
            restored_dict = json.loads(json_str)
            restored = NeurobiologicalEvaluation.from_dict(restored_dict)

            assert orig.synaptic_saturation == pytest.approx(
                restored.synaptic_saturation, abs=1e-6
            )
            assert orig.energy_expenditure == pytest.approx(
                restored.energy_expenditure, abs=1e-6
            )
            assert orig.sleep_consolidation_affinity == pytest.approx(
                restored.sleep_consolidation_affinity, abs=1e-6
            )
            assert orig.aggregate_cost() == pytest.approx(
                restored.aggregate_cost(), abs=1e-6
            )

    def test_psychiatric_serialization_roundtrip(self) -> None:
        """PsychiatricEvaluation round-trip serialization and JSON compatibility."""
        test_cases = [
            PsychiatricEvaluation(),
            PsychiatricEvaluation(
                rumination_risk=0.88,
                perseveration_penalty=0.77,
                threat_distortion=0.66,
            ),
        ]

        for orig in test_cases:
            d = orig.to_dict()
            json_str = json.dumps(d)
            restored = PsychiatricEvaluation.from_dict(json.loads(json_str))

            assert orig.rumination_risk == pytest.approx(restored.rumination_risk, abs=1e-6)
            assert orig.perseveration_penalty == pytest.approx(
                restored.perseveration_penalty, abs=1e-6
            )
            assert orig.threat_distortion == pytest.approx(
                restored.threat_distortion, abs=1e-6
            )
            assert orig.aggregate_cost() == pytest.approx(
                restored.aggregate_cost(), abs=1e-6
            )

    def test_sociological_serialization_roundtrip(self) -> None:
        """SociologicalEvaluation round-trip serialization and JSON compatibility."""
        test_cases = [
            SociologicalEvaluation(),
            SociologicalEvaluation(
                social_misalignment=0.33,
                user_fatigue_impact=0.66,
                coordination_friction=0.99,
            ),
        ]

        for orig in test_cases:
            d = orig.to_dict()
            json_str = json.dumps(d)
            restored = SociologicalEvaluation.from_dict(json.loads(json_str))

            assert orig.social_misalignment == pytest.approx(
                restored.social_misalignment, abs=1e-6
            )
            assert orig.user_fatigue_impact == pytest.approx(
                restored.user_fatigue_impact, abs=1e-6
            )
            assert orig.coordination_friction == pytest.approx(
                restored.coordination_friction, abs=1e-6
            )
            assert orig.aggregate_cost() == pytest.approx(
                restored.aggregate_cost(), abs=1e-6
            )

    def test_cognitive_panel_score_serialization_roundtrip(self) -> None:
        """CognitivePanelScore composite nested round-trip serialization."""
        orig = CognitivePanelScore(
            neurobiology=NeurobiologicalEvaluation(
                synaptic_saturation=0.45,
                energy_expenditure=0.55,
                sleep_consolidation_affinity=0.65,
            ),
            psychiatry=PsychiatricEvaluation(
                rumination_risk=0.20,
                perseveration_penalty=0.30,
                threat_distortion=0.40,
            ),
            sociology=SociologicalEvaluation(
                social_misalignment=0.15,
                user_fatigue_impact=0.85,
                coordination_friction=0.35,
            ),
        )

        d = orig.to_dict()
        assert "aggregate_penalty" in d
        assert "neurobiology" in d
        assert "psychiatry" in d
        assert "sociology" in d

        json_str = json.dumps(d)
        restored = CognitivePanelScore.from_dict(json.loads(json_str))

        assert orig.aggregate_penalty() == pytest.approx(
            restored.aggregate_penalty(), abs=1e-4
        )
        assert orig.neurobiology.synaptic_saturation == pytest.approx(
            restored.neurobiology.synaptic_saturation, abs=1e-6
        )
        assert orig.psychiatry.rumination_risk == pytest.approx(
            restored.psychiatry.rumination_risk, abs=1e-6
        )
        assert orig.sociology.user_fatigue_impact == pytest.approx(
            restored.sociology.user_fatigue_impact, abs=1e-6
        )

    def test_consequence_vector_nested_and_flat_serialization(self) -> None:
        """ConsequenceVector supports both nested and flattened backward compatibility."""
        cp = CognitivePanelScore(
            neurobiology=NeurobiologicalEvaluation(synaptic_saturation=0.5),
            psychiatry=PsychiatricEvaluation(rumination_risk=0.8),
            sociology=SociologicalEvaluation(user_fatigue_impact=0.7),
        )
        orig_cv = ConsequenceVector(
            latency=0.12,
            maintenance=0.34,
            risk=0.56,
            metabolic=0.78,
            cognitive_panel=cp,
        )

        # Standard nested dict
        d = orig_cv.to_dict()
        restored_nested = ConsequenceVector.from_dict(d)
        assert restored_nested.latency == pytest.approx(0.12, abs=1e-6)
        assert restored_nested.maintenance == pytest.approx(0.34, abs=1e-6)
        assert restored_nested.risk == pytest.approx(0.56, abs=1e-6)
        assert restored_nested.metabolic == pytest.approx(0.78, abs=1e-6)
        assert (
            restored_nested.cognitive_panel.psychiatry.rumination_risk
            == pytest.approx(0.8, abs=1e-6)
        )

        # Flat dictionary hydration (backward compatibility)
        flat_dict = {
            "latency": 0.20,
            "maintenance": 0.30,
            "risk": 0.40,
            "metabolic": 0.50,
            "psychiatry": {"rumination_risk": 0.75, "perseveration_penalty": 0.60},
            "sociology": {"user_fatigue_impact": 0.85},
        }
        restored_flat = ConsequenceVector.from_dict(flat_dict)
        assert restored_flat.latency == pytest.approx(0.20, abs=1e-6)
        assert (
            restored_flat.cognitive_panel.psychiatry.rumination_risk
            == pytest.approx(0.75, abs=1e-6)
        )
        assert (
            restored_flat.cognitive_panel.sociology.user_fatigue_impact
            == pytest.approx(0.85, abs=1e-6)
        )


# ============================================================================
# 5. TheoryOfMind Conversational Interdisciplinary Lens Tests
# ============================================================================


class TestTheoryOfMindInterdisciplinaryEvaluation:
    """Empirical challenge 5: TheoryOfMindAnalyzer conversational panel evaluation

    and interdisciplinary DreamSeed synthesis.
    """

    @pytest.fixture
    def tom(self) -> TheoryOfMindAnalyzer:
        return TheoryOfMindAnalyzer()

    def test_repetitive_topics_generate_anti_rumination_dream_seed_and_penalty(
        self,
        tom: TheoryOfMindAnalyzer,
    ) -> None:
        """Circular repetition of domain topics triggers rumination risk and anti_rumination seed."""
        circular_msgs = [
            {"role": "user", "content": "How do we handle zeno simulation?"},
            {"role": "assistant", "content": "We can use Quanta zeno attention dynamics."},
            {"role": "user", "content": "Let us check zeno memory again."},
            {"role": "assistant", "content": "Zeno memory uses quantum pinning."},
            {"role": "user", "content": "What about zeno fidelity once more?"},
        ]

        score = tom.evaluate_cognitive_panel(circular_msgs)
        assert score.psychiatry.rumination_risk >= 0.30

        urgency, seeds = tom.analyze_conversation(circular_msgs)
        topics = [s.topic for s in seeds]
        assert "anti_rumination" in topics

    def test_excessive_turns_generate_synaptic_homeostasis_seed(
        self,
        tom: TheoryOfMindAnalyzer,
    ) -> None:
        """Conversations exceeding 20 turns trigger synaptic homeostasis seed and sleep affinity."""
        long_msgs = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Turn {i} content"}
            for i in range(25)
        ]

        score = tom.evaluate_cognitive_panel(long_msgs)
        assert score.neurobiology.synaptic_saturation >= 0.60
        assert score.neurobiology.sleep_consolidation_affinity >= 0.70

        urgency, seeds = tom.analyze_conversation(long_msgs)
        topics = [s.topic for s in seeds]
        assert "synaptic_homeostasis" in topics

    def test_terse_user_turns_generate_fatigue_seed_and_penalty(
        self,
        tom: TheoryOfMindAnalyzer,
    ) -> None:
        """High ratio of terse user messages triggers user fatigue impact and fatigue reduction seed."""
        fatigue_msgs = [
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "Detailed 20-line explanation..."},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "Another 30-line response..."},
            {"role": "user", "content": "done"},
        ]

        score = tom.evaluate_cognitive_panel(fatigue_msgs)
        assert score.sociology.user_fatigue_impact >= 0.50

        urgency, seeds = tom.analyze_conversation(fatigue_msgs)
        topics = [s.topic for s in seeds]
        assert "user_fatigue_reduction" in topics

    def test_analyze_interdisciplinary_contract(
        self,
        tom: TheoryOfMindAnalyzer,
    ) -> None:
        """analyze_interdisciplinary returns (urgency, seeds, panel_score) with expected types."""
        messages = [
            {"role": "user", "content": "Deploy the distributed pipeline."},
            {"role": "assistant", "content": "Service is deployed."},
        ]

        urgency, seeds, panel = tom.analyze_interdisciplinary(messages)
        assert isinstance(urgency, float)
        assert 1.0 <= urgency <= 5.0
        assert isinstance(seeds, list)
        assert isinstance(panel, CognitivePanelScore)
