"""tests/test_challenger_m3_2_empirical.py -- Empirical Challenger M3_2 Verification Suite.

Adversarially tests and independently verifies:
1. Theory of Mind Conversational Evaluation:
   - User fatigue simulation: terse prompts ("ok", "fix it", "done") vs normal detailed prompts.
     Verifies user_fatigue_impact >= 0.5 and DreamSeed "user_fatigue_reduction" synthesis.
   - Circular rumination: repetitive returns to failing topics.
     Verifies rumination_risk >= 0.6 and DreamSeed "anti_rumination" synthesis.
   - High conversational turn count (25 turns):
     Verifies synaptic_saturation >= 0.7 and DreamSeed "synaptic_homeostasis" synthesis.
   - Adversarial boundary testing: empty conversations, corrupted dictionaries, NaN/Inf contexts.
2. DAG Rollout Trajectory Microglial Pruning under Human Fatigue:
   - Multi-branch decision DAG comparing calm vs high-fatigue trajectories.
   - Confirms that with panel_weight == 0.0, high-fatigue branch is unpruned.
   - Confirms that with panel_weight > 0.0, high-fatigue branch is microglially pruned.
   - Verifies DMN speculative preservation and panel_weight sensitivity sweeps.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

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
)
from quanta.cognitive.tom_analyzer import TheoryOfMindAnalyzer

# ============================================================================
# 1. Theory of Mind Conversational Evaluation Harness
# ============================================================================


class TestEmpiricalConversationalTheoryOfMind:
    """Empirical verification harness for conversational cadence and fatigue analysis."""

    def test_user_fatigue_terse_prompts_vs_normal_prompts(self) -> None:
        """Simulates terse prompts vs normal prompts to verify fatigue impact and DreamSeed."""
        tom = TheoryOfMindAnalyzer()

        # Terse prompt sequence (< 4 words per user message)
        terse_conversation = [
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "Processing the quantum compilation pipeline..."},
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "Applying the phase cancellation pass..."},
            {"role": "user", "content": "done"},
        ]

        score_terse = tom.evaluate_cognitive_panel(terse_conversation)
        urg_terse, seeds_terse = tom.analyze_conversation(terse_conversation)
        seed_topics_terse = {s.topic for s in seeds_terse}

        # Assertions for terse prompts
        assert score_terse.sociology.user_fatigue_impact >= 0.5, (
            f"Expected user_fatigue_impact >= 0.5, got {score_terse.sociology.user_fatigue_impact}"
        )
        assert "user_fatigue_reduction" in seed_topics_terse, (
            f"Expected 'user_fatigue_reduction' DreamSeed, got {seed_topics_terse}"
        )
        assert urg_terse >= 1.0

        # Normal, detailed prompt sequence
        normal_conversation = [
            {
                "role": "user",
                "content": (
                    "Could you please synthesize a surface code stabilizer lattice with "
                    "distance 5 and report the physical qubit overhead?"
                ),
            },
            {
                "role": "assistant",
                "content": "Synthesizing distance-5 rotated surface code lattice...",
            },
            {
                "role": "user",
                "content": (
                    "Thank you. Now please evaluate whether MWPM decoding can correct "
                    "correlated measurement errors within 15 syndrome extraction cycles."
                ),
            },
        ]

        score_normal = tom.evaluate_cognitive_panel(normal_conversation)
        _, seeds_normal = tom.analyze_conversation(normal_conversation)
        seed_topics_normal = {s.topic for s in seeds_normal}

        # Assertions for normal prompts
        assert score_normal.sociology.user_fatigue_impact < 0.5, (
            f"Expected user_fatigue_impact < 0.5, got {score_normal.sociology.user_fatigue_impact}"
        )
        assert "user_fatigue_reduction" not in seed_topics_normal, (
            f"Unexpected 'user_fatigue_reduction' DreamSeed in normal conversation: "
            f"{seed_topics_normal}"
        )

    def test_rumination_repetitive_failing_topic(self) -> None:
        """Simulates user repeatedly returning to the same failing topic."""
        tom = TheoryOfMindAnalyzer()

        # Repetitive dialogue circling around memory and hook failures
        rumination_conversation = [
            {"role": "user", "content": "The memory hook is failing with exception"},
            {"role": "assistant", "content": "Investigating memory hook dephasing..."},
            {"role": "user", "content": "Still failing in memory hook step 12"},
            {"role": "assistant", "content": "Refactoring memory hook state persistence..."},
            {"role": "user", "content": "Memory hook broken: traceback in test suite"},
            {"role": "assistant", "content": "Checking CSF attenuation in memory hook..."},
            {"role": "user", "content": "Why is the memory hook failing continuously?"},
        ]

        score = tom.evaluate_cognitive_panel(rumination_conversation)
        urgency, seeds = tom.analyze_conversation(rumination_conversation)
        seed_topics = {s.topic for s in seeds}

        assert score.psychiatry.rumination_risk >= 0.6, (
            f"Expected rumination_risk >= 0.6, got {score.psychiatry.rumination_risk}"
        )
        assert "anti_rumination" in seed_topics, (
            f"Expected 'anti_rumination' DreamSeed, got {seed_topics}"
        )
        assert urgency >= 2.8, f"Expected elevated urgency >= 2.8, got {urgency}"

    def test_high_turn_count_synaptic_saturation(self) -> None:
        """Simulates 25 conversational turns to verify synaptic saturation and homeostasis."""
        tom = TheoryOfMindAnalyzer()

        turns_25 = [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Turn {i}: Analyzing FTQC resource budgeting and compilation passes.",
            }
            for i in range(25)
        ]

        score = tom.evaluate_cognitive_panel(turns_25)
        _, seeds = tom.analyze_conversation(turns_25)
        seed_topics = {s.topic for s in seeds}

        assert score.neurobiology.synaptic_saturation >= 0.7, (
            f"Expected synaptic_saturation >= 0.7, got {score.neurobiology.synaptic_saturation}"
        )
        assert "synaptic_homeostasis" in seed_topics, (
            f"Expected 'synaptic_homeostasis' DreamSeed, got {seed_topics}"
        )
        assert score.neurobiology.sleep_consolidation_affinity >= 0.7, (
            "Expected elevated sleep_consolidation_affinity after 25 turns, "
            f"got {score.neurobiology.sleep_consolidation_affinity}"
        )

    def test_interdisciplinary_composite_analysis_pipeline(self) -> None:
        """Verifies analyze_interdisciplinary() returns all three composite outputs."""
        tom = TheoryOfMindAnalyzer()
        messages = [
            {"role": "user", "content": "ok"},
            {"role": "user", "content": "fix memory hook"},
            {"role": "user", "content": "done"},
        ]

        urgency, seeds, panel = tom.analyze_interdisciplinary(messages)
        assert isinstance(urgency, float)
        assert isinstance(seeds, list)
        assert isinstance(panel, CognitivePanelScore)
        assert panel.sociology.user_fatigue_impact >= 0.5
        assert 0.0 <= panel.aggregate_penalty() <= 1.0

    def test_boundary_and_corrupted_inputs_resilience(self) -> None:
        """Stress-tests TheoryOfMindAnalyzer with empty, malformed, and non-finite inputs."""
        tom = TheoryOfMindAnalyzer()

        # 1. Empty conversation list
        empty_score = tom.evaluate_cognitive_panel([])
        assert empty_score.aggregate_penalty() == 0.0
        assert empty_score.neurobiology.synaptic_saturation == 0.0
        assert empty_score.sociology.user_fatigue_impact == 0.0

        empty_urgency, empty_seeds = tom.analyze_conversation([])
        assert empty_urgency == 1.0
        assert len(empty_seeds) == 0

        # 2. Corrupted message entries (missing keys, non-string types, None values)
        corrupted_messages: list[dict[str, Any]] = [
            {},
            {"role": "user"},
            {"content": None},
            {"role": "user", "content": 42},
            {"role": "system", "content": ["invalid", "list"]},
            {"role": None, "content": "Valid text string"},
        ]
        corrupted_score = tom.evaluate_cognitive_panel(corrupted_messages)
        assert 0.0 <= corrupted_score.aggregate_penalty() <= 1.0

        # 3. Context dictionary containing NaN, Inf, and out-of-range floats
        extreme_context = {
            "buffer_density": float("nan"),
            "energy_expenditure": float("inf"),
            "rumination_risk": -100.0,
            "user_fatigue_impact": 999.0,
            "sleep_affinity": float("-inf"),
        }
        extreme_score = tom.evaluate_cognitive_panel([], context=extreme_context)
        assert not math.isnan(extreme_score.aggregate_penalty())
        assert not math.isinf(extreme_score.aggregate_penalty())
        assert 0.0 <= extreme_score.aggregate_penalty() <= 1.0
        assert extreme_score.sociology.user_fatigue_impact == 1.0  # clamped from 999.0


# ============================================================================
# 2. DAG Rollout Microglial Pruning under Human Fatigue
# ============================================================================


class TestEmpiricalDAGRolloutPruningUnderFatigue:
    """Empirical verification harness for Decision DAG pruning under human cognitive fatigue."""

    def _build_calm_vs_fatigue_dag(self) -> DecisionDAG:
        """Helper building a 2-branch DAG comparing calm vs high-fatigue trajectories."""
        dag = DecisionDAG()
        root = DecisionNode(node_id="root", label="Workflow Orchestration")

        # Fixed unit basis statevector for identical semantic alignment
        vec = torch.zeros(64, dtype=torch.float32)
        vec[0] = 1.0

        # Branch 1: High user fatigue / cognitive load
        # Low 4D operational cost (fast script: 0.05), but severe human fatigue (0.95)
        panel_fatigue = CognitivePanelScore(
            neurobiology=NeurobiologicalEvaluation(
                synaptic_saturation=0.90, energy_expenditure=0.85
            ),
            psychiatry=PsychiatricEvaluation(
                rumination_risk=0.85, perseveration_penalty=0.80
            ),
            sociology=SociologicalEvaluation(
                user_fatigue_impact=0.95, coordination_friction=0.90
            ),
        )
        node_fatigue = DecisionNode(
            node_id="branch_fatigue",
            label="Excessive Manual Interactive Prompts and Micro-Confirmations",
            statevector=vec.clone(),
            consequence=ConsequenceVector(
                latency=0.05,
                maintenance=0.05,
                risk=0.05,
                metabolic=0.05,
                cognitive_panel=panel_fatigue,
            ),
        )

        # Branch 2: Calm, autonomous, streamlined trajectory
        # Slightly higher 4D operational cost (0.10), but negligible human fatigue (0.05)
        panel_calm = CognitivePanelScore(
            neurobiology=NeurobiologicalEvaluation(
                synaptic_saturation=0.05, energy_expenditure=0.05
            ),
            psychiatry=PsychiatricEvaluation(
                rumination_risk=0.05, perseveration_penalty=0.05
            ),
            sociology=SociologicalEvaluation(
                user_fatigue_impact=0.05, coordination_friction=0.05
            ),
        )
        node_calm = DecisionNode(
            node_id="branch_calm",
            label="Autonomous Cognitive Streamlining with Calm Defaults",
            statevector=vec.clone(),
            consequence=ConsequenceVector(
                latency=0.10,
                maintenance=0.10,
                risk=0.10,
                metabolic=0.10,
                cognitive_panel=panel_calm,
            ),
        )

        dag.add_node(root)
        dag.add_node(node_fatigue)
        dag.add_node(node_calm)

        dag.add_edge(
            DecisionEdge(
                source_id="root",
                target_id="branch_fatigue",
                action_label="Select Manual Micro-Prompts",
            )
        )
        dag.add_edge(
            DecisionEdge(
                source_id="root",
                target_id="branch_calm",
                action_label="Select Autonomous Calm",
            )
        )
        return dag

    def test_dag_microglial_pruning_under_human_fatigue(self) -> None:
        """Empirically confirms high-fatigue trajectory is pruned only when panel_weight > 0.0."""
        arbiter = QuantumDecisionArbiter(dim=64, num_heads=4, seed=42)

        # 1. Baseline: panel_weight = 0.0 (strictly 4D operational costs)
        dag_baseline = self._build_calm_vs_fatigue_dag()
        res_baseline = arbiter.arbitrate_dag(
            dag=dag_baseline,
            goal="Autonomous Workflow Orchestration",
            panel_weight=0.0,
            prune_threshold=0.20,
            zeno_lock_threshold=0.40,
        )

        traj_fatigue_base = next(
            t for t in res_baseline["trajectories"] if "branch_fatigue" in t["path_nodes"]
        )
        traj_calm_base = next(
            t for t in res_baseline["trajectories"] if "branch_calm" in t["path_nodes"]
        )

        # In baseline 4D, fatigue branch has lower operational cost (0.05 vs 0.10)
        assert res_baseline["winning_trajectory"]["path_nodes"] == ["root", "branch_fatigue"]
        assert traj_fatigue_base["is_pruned"] is False
        assert traj_calm_base["is_pruned"] is False
        assert res_baseline["pruned_count"] == 0

        # 2. Interdisciplinary evaluation: panel_weight = 0.70
        dag_panel = self._build_calm_vs_fatigue_dag()
        res_panel = arbiter.arbitrate_dag(
            dag=dag_panel,
            goal="Autonomous Workflow Orchestration",
            panel_weight=0.70,
            prune_threshold=0.20,
            zeno_lock_threshold=0.40,
        )

        traj_fatigue_panel = next(
            t for t in res_panel["trajectories"] if "branch_fatigue" in t["path_nodes"]
        )
        traj_calm_panel = next(
            t for t in res_panel["trajectories"] if "branch_calm" in t["path_nodes"]
        )

        # Under panel_weight = 0.70, human fatigue penalty dominates:
        # Calm branch becomes the winner; high-fatigue branch is microglially pruned
        assert res_panel["winning_trajectory"]["path_nodes"] == ["root", "branch_calm"]
        assert traj_calm_panel["is_pruned"] is False
        assert traj_fatigue_panel["is_pruned"] is True
        assert "Dominated by winner" in str(traj_fatigue_panel["prune_reason"])
        assert res_panel["pruned_count"] == 1
        assert res_panel["zeno_lock_engaged"] is True

    def test_dag_dmn_speculative_preservation_under_fatigue(self) -> None:
        """Verifies that DMN speculative branches survive moderate fatigue gaps."""
        arbiter = QuantumDecisionArbiter(dim=64, num_heads=4, seed=42)
        dag = DecisionDAG()
        root = DecisionNode(node_id="root", label="System Selection")

        vec = torch.zeros(64, dtype=torch.float32)
        vec[0] = 1.0

        # Winner: calm branch
        node_calm = DecisionNode(
            node_id="calm",
            label="Calm Streamlined",
            statevector=vec.clone(),
            consequence=ConsequenceVector(
                latency=0.05,
                maintenance=0.05,
                risk=0.05,
                metabolic=0.05,
                cognitive_panel=CognitivePanelScore(
                    sociology=SociologicalEvaluation(user_fatigue_impact=0.05)
                ),
            ),
        )

        # Speculative branch with elevated fatigue: marked with metadata is_speculative=True
        node_spec = DecisionNode(
            node_id="spec",
            label="Speculative Exploration with Moderate Fatigue",
            metadata={"is_speculative": True},
            statevector=vec.clone(),
            consequence=ConsequenceVector(
                latency=0.05,
                maintenance=0.05,
                risk=0.05,
                metabolic=0.05,
                cognitive_panel=CognitivePanelScore(
                    neurobiology=NeurobiologicalEvaluation(
                        synaptic_saturation=0.8, energy_expenditure=0.8
                    ),
                    psychiatry=PsychiatricEvaluation(
                        rumination_risk=0.8, perseveration_penalty=0.8
                    ),
                    sociology=SociologicalEvaluation(
                        user_fatigue_impact=0.9, coordination_friction=0.85
                    ),
                ),
            ),
        )

        # Standard non-speculative branch with identical fatigue penalty
        node_std = DecisionNode(
            node_id="std",
            label="Standard Trajectory with Identical Fatigue",
            metadata={"is_speculative": False},
            statevector=vec.clone(),
            consequence=ConsequenceVector(
                latency=0.05,
                maintenance=0.05,
                risk=0.05,
                metabolic=0.05,
                cognitive_panel=CognitivePanelScore(
                    neurobiology=NeurobiologicalEvaluation(
                        synaptic_saturation=0.8, energy_expenditure=0.8
                    ),
                    psychiatry=PsychiatricEvaluation(
                        rumination_risk=0.8, perseveration_penalty=0.8
                    ),
                    sociology=SociologicalEvaluation(
                        user_fatigue_impact=0.9, coordination_friction=0.85
                    ),
                ),
            ),
        )

        for n in [root, node_calm, node_spec, node_std]:
            dag.add_node(n)

        dag.add_edge(DecisionEdge("root", "calm", "Choose Calm"))
        dag.add_edge(DecisionEdge("root", "spec", "Choose Speculative"))
        dag.add_edge(DecisionEdge("root", "std", "Choose Standard"))

        # prune_threshold = 0.20 -> standard pruned at gap > 0.20; speculative pruned at gap > 0.40
        res = arbiter.arbitrate_dag(
            dag=dag,
            goal="System Selection",
            panel_weight=0.60,
            prune_threshold=0.20,
            zeno_lock_threshold=0.30,
        )

        traj_spec = next(t for t in res["trajectories"] if "spec" in t["path_nodes"])
        traj_std = next(t for t in res["trajectories"] if "std" in t["path_nodes"])

        # Speculative is protected by DMN lateral exploration (gap ~ 0.35 <= 0.40)
        assert traj_spec["is_speculative"] is True
        assert traj_spec["is_pruned"] is False
        assert traj_spec["prune_reason"] is None

        # Standard branch is microglially pruned (gap ~ 0.35 > 0.20)
        assert traj_std["is_speculative"] is False
        assert traj_std["is_pruned"] is True
        assert "Dominated by winner" in str(traj_std["prune_reason"])

    @pytest.mark.parametrize("panel_weight", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    def test_dag_panel_weight_sweep_monotonicity(self, panel_weight: float) -> None:
        """Verifies cumulative cost of high-fatigue trajectory increases with panel_weight."""
        arbiter = QuantumDecisionArbiter(dim=64, num_heads=4, seed=42)
        dag = self._build_calm_vs_fatigue_dag()

        res = arbiter.arbitrate_dag(
            dag=dag,
            goal="Autonomous Workflow Orchestration",
            panel_weight=panel_weight,
            prune_threshold=0.20,
            zeno_lock_threshold=0.40,
        )

        traj_fatigue = next(
            t for t in res["trajectories"] if "branch_fatigue" in t["path_nodes"]
        )

        expected_4d = 0.05
        node = dag.get_node("branch_fatigue")
        assert node is not None
        panel_penalty = node.consequence.cognitive_panel.aggregate_penalty()

        expected_cost = round((1.0 - panel_weight) * expected_4d + panel_weight * panel_penalty, 4)
        assert traj_fatigue["cumulative_cost"] == pytest.approx(expected_cost, abs=1e-3)

        if panel_weight >= 0.5:
            assert traj_fatigue["is_pruned"] is True
        elif panel_weight == 0.0:
            assert traj_fatigue["is_pruned"] is False
