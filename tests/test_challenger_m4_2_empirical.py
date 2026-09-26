"""tests/test_challenger_m4_2_empirical.py — Empirical Challenger M4_2 Verification Suite.

Adversarially tests and independently verifies Milestone 4:
1. Mathematical Bounds:
   - All costs, penalties, affinities, and aggregate scores strictly in [0.0, 1.0]
     across 10,000 randomized and adversarial inputs (including negative, huge, NaN,
     Inf, and None values).
2. Linear Interpolation Exactness:
   - ConsequenceVector.weighted_cost(panel_weight=w) matches (1 - w)*C_4D + w*C_panel
     within floating-point epsilon (< 1e-12) across 10,000 trials.
3. Microglial Pruning Under Human Fatigue & Topology Invariance:
   - High human fatigue (> 0.8) with panel_weight=0.85 triggers microglial pruning on
     unviable trajectories.
   - Verified across multiple DAG topologies (2-branch, 4-branch, 3-ply tree, diamond).
   - Confirms zero topological corruption: node sets, edge sets, and adjacency graphs
     are 100% preserved before and after arbitration.
4. Latency Benchmarks:
   - evaluate_cognitive_panel executes in < 5.0ms (P50, P95, and mean).
   - analyze_interdisciplinary executes in < 15.0ms (P50, P95, and mean).
"""

from __future__ import annotations

import random
import time
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
)
from quanta.cognitive.tom_analyzer import TheoryOfMindAnalyzer

# ============================================================================
# Section 1: Mathematical Bounds Invariance (10,000 Randomized Trials)
# ============================================================================


class TestMathematicalBoundsInvariance:
    """Verifies all evaluations, penalties, and aggregate scores are strictly in [0.0, 1.0]."""

    def test_randomized_dataclass_and_serialization_bounds_10000_runs(self) -> None:
        """Evaluates 10,000 randomized dictionary inputs to verify strict [0.0, 1.0] bounds."""
        rng = random.Random(1337)

        # Diverse adversarial value generators
        def random_adversarial_value() -> Any:
            choice = rng.randint(0, 7)
            if choice == 0:
                return rng.uniform(-1000.0, 1000.0)
            if choice == 1:
                return float("nan")
            if choice == 2:
                return float("inf") if rng.random() > 0.5 else float("-inf")
            if choice == 3:
                return None
            if choice == 4:
                return "corrupted_string"
            if choice == 5:
                return rng.choice([True, False])
            if choice == 6:
                return [1, 2, 3]
            return rng.uniform(0.0, 1.0)

        for trial_idx in range(10000):
            # 1. NeurobiologicalEvaluation
            n_data = {
                "synaptic_saturation": random_adversarial_value(),
                "energy_expenditure": random_adversarial_value(),
                "sleep_consolidation_affinity": random_adversarial_value(),
            }
            n_eval = NeurobiologicalEvaluation.from_dict(n_data)
            assert 0.0 <= n_eval.synaptic_saturation <= 1.0, f"Trial {trial_idx}: {n_eval}"
            assert 0.0 <= n_eval.energy_expenditure <= 1.0, f"Trial {trial_idx}: {n_eval}"
            assert 0.0 <= n_eval.sleep_consolidation_affinity <= 1.0, f"Trial {trial_idx}: {n_eval}"
            cost_n = n_eval.aggregate_cost()
            assert 0.0 <= cost_n <= 1.0, f"Trial {trial_idx} neuro cost: {cost_n}"

            # 2. PsychiatricEvaluation
            p_data = {
                "rumination_risk": random_adversarial_value(),
                "perseveration_penalty": random_adversarial_value(),
                "threat_distortion": random_adversarial_value(),
            }
            p_eval = PsychiatricEvaluation.from_dict(p_data)
            assert 0.0 <= p_eval.rumination_risk <= 1.0, f"Trial {trial_idx}: {p_eval}"
            assert 0.0 <= p_eval.perseveration_penalty <= 1.0, f"Trial {trial_idx}: {p_eval}"
            assert 0.0 <= p_eval.threat_distortion <= 1.0, f"Trial {trial_idx}: {p_eval}"
            cost_p = p_eval.aggregate_cost()
            assert 0.0 <= cost_p <= 1.0, f"Trial {trial_idx} psych cost: {cost_p}"

            # 3. SociologicalEvaluation
            s_data = {
                "social_misalignment": random_adversarial_value(),
                "user_fatigue_impact": random_adversarial_value(),
                "coordination_friction": random_adversarial_value(),
            }
            s_eval = SociologicalEvaluation.from_dict(s_data)
            assert 0.0 <= s_eval.social_misalignment <= 1.0, f"Trial {trial_idx}: {s_eval}"
            assert 0.0 <= s_eval.user_fatigue_impact <= 1.0, f"Trial {trial_idx}: {s_eval}"
            assert 0.0 <= s_eval.coordination_friction <= 1.0, f"Trial {trial_idx}: {s_eval}"
            cost_s = s_eval.aggregate_cost()
            assert 0.0 <= cost_s <= 1.0, f"Trial {trial_idx} soc cost: {cost_s}"

            # 4. CognitivePanelScore
            panel = CognitivePanelScore(neurobiology=n_eval, psychiatry=p_eval, sociology=s_eval)
            penalty = panel.aggregate_penalty()
            assert 0.0 <= penalty <= 1.0, f"Trial {trial_idx} panel penalty: {penalty}"

            # 5. ConsequenceVector
            c_data = {
                "latency": random_adversarial_value(),
                "maintenance": random_adversarial_value(),
                "risk": random_adversarial_value(),
                "metabolic": random_adversarial_value(),
                "cognitive_panel": panel.to_dict(),
            }
            cv = ConsequenceVector.from_dict(c_data)
            assert 0.0 <= cv.latency <= 1.0, f"Trial {trial_idx}: {cv}"
            assert 0.0 <= cv.maintenance <= 1.0, f"Trial {trial_idx}: {cv}"
            assert 0.0 <= cv.risk <= 1.0, f"Trial {trial_idx}: {cv}"
            assert 0.0 <= cv.metabolic <= 1.0, f"Trial {trial_idx}: {cv}"

            w_panel = rng.uniform(0.0, 1.0)
            cost_cv = cv.weighted_cost(panel_weight=w_panel)
            assert 0.0 <= cost_cv <= 1.0, f"Trial {trial_idx} weighted cost: {cost_cv}"

    def test_evaluate_cognitive_panel_bounds_2000_conversations(self) -> None:
        """Evaluates 2,000 synthesized conversations with random contexts to assert strict bounds."""
        rng = random.Random(42)
        tom = TheoryOfMindAnalyzer()

        vocab = list(TheoryOfMindAnalyzer.DOMAIN_TOPICS) + [
            "error",
            "exception",
            "fail",
            "catastrophic",
            "fatal",
            "thanks",
            "agree",
            "collaborate",
            "conflict",
            "deadlock",
            "ok",
            "quantum",
            "pipeline",
        ]

        for _ in range(2000):
            turn_count = rng.randint(0, 40)
            messages: list[dict[str, Any]] = []
            for _ in range(turn_count):
                words = rng.choices(vocab, k=rng.randint(1, 10))
                role = rng.choice(["user", "assistant"])
                msg: dict[str, Any] = {"role": role, "content": " ".join(words)}
                if rng.random() < 0.2:
                    msg["tool"] = "run_command"
                messages.append(msg)

            # Adversarial context and project state
            context: dict[str, Any] = {
                "turn_count": rng.choice([None, -5, 50, "invalid", 10.5]),
                "buffer_density": rng.choice([None, -1.0, 2.5, 0.45]),
                "repetition_count": rng.choice([None, -2, 10, "5"]),
                "repeated_tool_failures": rng.choice([None, -1, 4]),
                "sleep_affinity": rng.choice([None, 0.8, -0.5, 1.5]),
            }
            project_state: dict[str, Any] = {
                "test_failures": rng.choice([None, -3, 12, "0", float("nan")]),
                "repetition_count": rng.choice([None, 0, 7]),
                "quiescent": rng.choice([True, False, None]),
                "user_fatigue": rng.choice([True, False, None]),
                "blocker": rng.choice([True, False, None]),
            }

            panel = tom.evaluate_cognitive_panel(
                messages, project_state=project_state, context=context
            )

            # Assert all attributes strictly bounded in [0.0, 1.0]
            assert 0.0 <= panel.neurobiology.synaptic_saturation <= 1.0
            assert 0.0 <= panel.neurobiology.energy_expenditure <= 1.0
            assert 0.0 <= panel.neurobiology.sleep_consolidation_affinity <= 1.0
            assert 0.0 <= panel.neurobiology.aggregate_cost() <= 1.0

            assert 0.0 <= panel.psychiatry.rumination_risk <= 1.0
            assert 0.0 <= panel.psychiatry.perseveration_penalty <= 1.0
            assert 0.0 <= panel.psychiatry.threat_distortion <= 1.0
            assert 0.0 <= panel.psychiatry.aggregate_cost() <= 1.0

            assert 0.0 <= panel.sociology.social_misalignment <= 1.0
            assert 0.0 <= panel.sociology.user_fatigue_impact <= 1.0
            assert 0.0 <= panel.sociology.coordination_friction <= 1.0
            assert 0.0 <= panel.sociology.aggregate_cost() <= 1.0

            assert 0.0 <= panel.aggregate_penalty() <= 1.0


# ============================================================================
# Section 2: Exactness of Linear Interpolation (< 10^-12)
# ============================================================================


class TestLinearInterpolationExactness:
    """Verifies ConsequenceVector.weighted_cost matches mathematical definition within < 1e-12."""

    def test_linear_interpolation_10000_trials(self) -> None:
        """Evaluates 10,000 trials across random 4D vectors, panel penalties, and weights."""
        rng = random.Random(2026)
        max_discrepancy = 0.0

        for trial_idx in range(10000):
            # Random 4D costs in [0.0, 1.0]
            lat = rng.uniform(0.0, 1.0)
            maint = rng.uniform(0.0, 1.0)
            risk = rng.uniform(0.0, 1.0)
            meta = rng.uniform(0.0, 1.0)

            # Random 4D weights summing to 1.0
            raw_w = [rng.uniform(0.01, 1.0) for _ in range(4)]
            sum_w = sum(raw_w)
            w4d = (raw_w[0] / sum_w, raw_w[1] / sum_w, raw_w[2] / sum_w, raw_w[3] / sum_w)

            # Random panel components
            panel = CognitivePanelScore(
                neurobiology=NeurobiologicalEvaluation(
                    synaptic_saturation=rng.uniform(0.0, 1.0),
                    energy_expenditure=rng.uniform(0.0, 1.0),
                    sleep_consolidation_affinity=rng.uniform(0.0, 1.0),
                ),
                psychiatry=PsychiatricEvaluation(
                    rumination_risk=rng.uniform(0.0, 1.0),
                    perseveration_penalty=rng.uniform(0.0, 1.0),
                    threat_distortion=rng.uniform(0.0, 1.0),
                ),
                sociology=SociologicalEvaluation(
                    social_misalignment=rng.uniform(0.0, 1.0),
                    user_fatigue_impact=rng.uniform(0.0, 1.0),
                    coordination_friction=rng.uniform(0.0, 1.0),
                ),
            )

            cv = ConsequenceVector(
                latency=lat,
                maintenance=maint,
                risk=risk,
                metabolic=meta,
                cognitive_panel=panel,
            )

            # Pick panel_weight: test boundary values and continuous values
            if trial_idx == 0:
                w_p = 0.0
            elif trial_idx == 1:
                w_p = 1.0
            elif trial_idx == 2:
                w_p = 0.5
            elif trial_idx == 3:
                w_p = 0.85
            elif trial_idx == 4:
                w_p = 1e-15
            elif trial_idx == 5:
                w_p = 1.0 - 1e-15
            else:
                w_p = rng.uniform(0.0, 1.0)

            # Mathematical theoretical formula:
            # c_4d = sum(w_i * cost_i)
            # c_panel = panel.aggregate_penalty()
            # expected = (1.0 - w_p) * c_4d + w_p * c_panel
            c_4d_expected = (
                w4d[0] * lat + w4d[1] * maint + w4d[2] * risk + w4d[3] * meta
            )
            c_panel_expected = panel.aggregate_penalty()

            if w_p <= 0.0:
                expected_cost = float(c_4d_expected)
            else:
                clamped_wp = min(1.0, max(0.0, w_p))
                expected_cost = float((1.0 - clamped_wp) * c_4d_expected + clamped_wp * c_panel_expected)

            actual_cost = cv.weighted_cost(weights=w4d, panel_weight=w_p)

            discrepancy = abs(actual_cost - expected_cost)
            if discrepancy > max_discrepancy:
                max_discrepancy = discrepancy

            assert discrepancy < 1e-12, (
                f"Trial {trial_idx}: Discrepancy {discrepancy:.2e} >= 1e-12. "
                f"Actual={actual_cost}, Expected={expected_cost}, w_p={w_p}"
            )

        print(
            f"\n[Interpolation Exactness Benchmark - 10,000 Trials]\n"
            f"  Max Discrepancy: {max_discrepancy:.3e} (Strict Epsilon Target < 1.000e-12)\n"
            f"  Pass Rate: 100.0% (10,000/10,000)\n"
        )
        assert max_discrepancy < 1e-12


# ============================================================================
# Section 3: Microglial Pruning Under Human Fatigue & Topology Invariance
# ============================================================================


class TestMicroglialPruningUnderFatigueAndTopologyInvariance:
    """Verifies microglial pruning under human fatigue (> 0.8) without DAG topological corruption."""

    def test_pruning_and_topology_invariance_2_branch(self) -> None:
        """Tests 2-branch rollout comparing calm autonomous action vs exhausting verbose inquiry."""
        arbiter = QuantumDecisionArbiter(seed=42)

        dag = DecisionDAG()
        dag.add_node(DecisionNode("root", "Initial State", depth=0))

        # Path 1: Concise autonomous action (minimal fatigue: 0.05)
        dag.add_node(
            DecisionNode(
                "concise_action",
                "Concise Autonomous Action",
                depth=1,
                consequence=ConsequenceVector(
                    latency=0.10,
                    maintenance=0.10,
                    risk=0.10,
                    metabolic=0.10,
                    cognitive_panel=CognitivePanelScore(
                        neurobiology=NeurobiologicalEvaluation(0.1, 0.1, 0.1),
                        psychiatry=PsychiatricEvaluation(0.1, 0.1, 0.1),
                        sociology=SociologicalEvaluation(
                            social_misalignment=0.05,
                            user_fatigue_impact=0.05,
                            coordination_friction=0.05,
                        ),
                    ),
                ),
            )
        )

        # Path 2: Exhausting high-friction trajectory with extreme user fatigue (> 0.8)
        dag.add_node(
            DecisionNode(
                "exhausting_action",
                "High Fatigue Verbose Inquiry",
                depth=1,
                consequence=ConsequenceVector(
                    latency=0.12,
                    maintenance=0.12,
                    risk=0.10,
                    metabolic=0.10,
                    cognitive_panel=CognitivePanelScore(
                        neurobiology=NeurobiologicalEvaluation(0.85, 0.85, 0.85),
                        psychiatry=PsychiatricEvaluation(0.85, 0.85, 0.85),
                        sociology=SociologicalEvaluation(
                            social_misalignment=0.85,
                            user_fatigue_impact=0.95,
                            coordination_friction=0.85,
                        ),
                    ),
                ),
            )
        )

        dag.add_edge(DecisionEdge("root", "concise_action", "Execute Concise"))
        dag.add_edge(DecisionEdge("root", "exhausting_action", "Prompt User Verbose"))

        # Snapshot topology before arbitration
        before_nodes = set(dag.nodes.keys())
        before_edges = {
            (u, e.target_id, e.action_label)
            for u, edges in dag.adj.items()
            for e in edges
        }

        goal = "Autonomous Pipeline Execution with Minimal Cognitive Friction"

        # 1. Under panel_weight=0.0: Both trajectories survive (4D operational costs are close)
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

        # 2. Under panel_weight=0.85: Microglial pruning eliminates the high-fatigue branch
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

        # 3. VERIFY TOPOLOGY INVARIANCE: Graph topology must remain completely uncorrupted
        after_nodes = set(dag.nodes.keys())
        after_edges = {
            (u, e.target_id, e.action_label)
            for u, edges in dag.adj.items()
            for e in edges
        }
        assert before_nodes == after_nodes
        assert before_edges == after_edges

    def test_pruning_and_topology_invariance_multi_branch(self) -> None:
        """Tests 4-branch Decision DAG verifying selective microglial pruning under panel_weight=0.85."""
        arbiter = QuantumDecisionArbiter(seed=42)

        dag = DecisionDAG()
        dag.add_node(DecisionNode("root", "System Baseline Root", depth=0))

        branches = [
            ("branch_a", "Concise Autonomous Action", 0.05, 0.05),
            ("branch_b", "Autonomous Quick Confirmation", 0.25, 0.25),
            ("branch_c", "High Fatigue Repetitive Prompt Loop", 0.85, 0.85),
            ("branch_d", "Exhausting Multi-Turn Questionnaire", 0.95, 0.95),
        ]

        for b_id, b_label, fatigue, penalty in branches:
            dag.add_node(
                DecisionNode(
                    b_id,
                    b_label,
                    depth=1,
                    consequence=ConsequenceVector(
                        latency=0.10,
                        maintenance=0.10,
                        risk=0.10,
                        metabolic=0.10,
                        cognitive_panel=CognitivePanelScore(
                            neurobiology=NeurobiologicalEvaluation(penalty, penalty, penalty),
                            psychiatry=PsychiatricEvaluation(penalty, penalty, penalty),
                            sociology=SociologicalEvaluation(
                                social_misalignment=0.05,
                                user_fatigue_impact=fatigue,
                                coordination_friction=0.05,
                            ),
                        ),
                    ),
                )
            )
            dag.add_edge(DecisionEdge("root", b_id, f"Transition -> {b_label}"))

        # Snapshot topology before arbitration
        before_nodes = set(dag.nodes.keys())
        before_edges = {
            (u, e.target_id, e.action_label)
            for u, edges in dag.adj.items()
            for e in edges
        }
        before_adj_counts = {u: len(edges) for u, edges in dag.adj.items()}

        goal = "Concise Autonomous Action with Zero Cognitive Friction"

        # 1. Evaluate with panel_weight = 0.0 (baseline 4D cost only)
        res_unweighted = arbiter.arbitrate_dag(
            dag,
            goal=goal,
            prune_threshold=0.25,
            zeno_lock_threshold=0.45,
            panel_weight=0.0,
        )

        # All branches have identical 4D costs (0.10), so none should be pruned
        for traj in res_unweighted["trajectories"]:
            assert traj["is_pruned"] is False, (
                f"Trajectory {traj['trajectory_id']} unexpectedly pruned under panel_weight=0.0"
            )

        # 2. Evaluate with panel_weight = 0.85 (heavy cognitive panel weighting)
        res_weighted = arbiter.arbitrate_dag(
            dag,
            goal=goal,
            prune_threshold=0.25,
            zeno_lock_threshold=0.45,
            panel_weight=0.85,
        )

        trajs_by_leaf = {
            t["path_nodes"][-1]: t for t in res_weighted["trajectories"]
        }

        # Branch A (minimal fatigue) must survive and win
        assert trajs_by_leaf["branch_a"]["is_pruned"] is False
        assert res_weighted["recommended_option"] == "Concise Autonomous Action"

        # Branch B (moderate fatigue 0.25) also survives as a viable candidate
        assert trajs_by_leaf["branch_b"]["is_pruned"] is False

        # Branch C (0.85) and Branch D (0.95) must be microglially pruned
        assert trajs_by_leaf["branch_c"]["is_pruned"] is True
        assert trajs_by_leaf["branch_d"]["is_pruned"] is True
        assert "Dominated by winner" in trajs_by_leaf["branch_c"]["prune_reason"]
        assert "Dominated by winner" in trajs_by_leaf["branch_d"]["prune_reason"]

        # 3. VERIFY TOPOLOGY INVARIANCE: Graph topology must remain completely uncorrupted
        after_nodes = set(dag.nodes.keys())
        after_edges = {
            (u, e.target_id, e.action_label)
            for u, edges in dag.adj.items()
            for e in edges
        }
        after_adj_counts = {u: len(edges) for u, edges in dag.adj.items()}

        assert before_nodes == after_nodes, "DAG nodes set was corrupted during arbitration"
        assert before_edges == after_edges, "DAG edge transitions were corrupted during arbitration"
        assert before_adj_counts == after_adj_counts, "DAG adjacency list counts altered"

        # Check DAG reachability
        visited = set()
        stack = ["root"]
        while stack:
            curr = stack.pop()
            visited.add(curr)
            for e in dag.adj.get(curr, []):
                assert e.target_id in dag.nodes
                if e.target_id not in visited:
                    stack.append(e.target_id)

        assert visited == before_nodes, "Graph reachability corrupted"

    def test_pruning_and_topology_invariance_deep_tree(self) -> None:
        """Tests pruning in a multi-level 3-ply tree where deeper high-fatigue paths are pruned."""
        arbiter = QuantumDecisionArbiter(seed=42)

        dag = DecisionDAG()
        dag.add_node(DecisionNode("root", "System Root", depth=0))

        # c1: Concise Autonomous Path
        dag.add_node(
            DecisionNode(
                "c1",
                "Concise Autonomous Execution",
                depth=1,
                consequence=ConsequenceVector(
                    latency=0.1,
                    maintenance=0.1,
                    risk=0.1,
                    metabolic=0.1,
                    cognitive_panel=CognitivePanelScore(
                        neurobiology=NeurobiologicalEvaluation(0.1, 0.1, 0.1),
                        psychiatry=PsychiatricEvaluation(0.1, 0.1, 0.1),
                        sociology=SociologicalEvaluation(
                            social_misalignment=0.05,
                            user_fatigue_impact=0.1,
                            coordination_friction=0.05,
                        ),
                    ),
                ),
            )
        )
        # f1: High Fatigue Interactive Prompting
        dag.add_node(
            DecisionNode(
                "f1",
                "High Fatigue Repetitive Queries",
                depth=1,
                consequence=ConsequenceVector(
                    latency=0.1,
                    maintenance=0.1,
                    risk=0.1,
                    metabolic=0.1,
                    cognitive_panel=CognitivePanelScore(
                        neurobiology=NeurobiologicalEvaluation(0.85, 0.85, 0.85),
                        psychiatry=PsychiatricEvaluation(0.85, 0.85, 0.85),
                        sociology=SociologicalEvaluation(
                            social_misalignment=0.85,
                            user_fatigue_impact=0.9,
                            coordination_friction=0.85,
                        ),
                    ),
                ),
            )
        )

        # Leaves under c1
        dag.add_node(
            DecisionNode(
                "c1_leaf1",
                "Finalize Concise Autonomous Action",
                depth=2,
                consequence=ConsequenceVector(
                    latency=0.1,
                    maintenance=0.1,
                    risk=0.1,
                    metabolic=0.1,
                    cognitive_panel=CognitivePanelScore(
                        neurobiology=NeurobiologicalEvaluation(0.1, 0.1, 0.1),
                        psychiatry=PsychiatricEvaluation(0.1, 0.1, 0.1),
                        sociology=SociologicalEvaluation(
                            social_misalignment=0.05,
                            user_fatigue_impact=0.05,
                            coordination_friction=0.05,
                        ),
                    ),
                ),
            )
        )
        dag.add_node(
            DecisionNode(
                "c1_leaf2",
                "Verify Concise Autonomous Output",
                depth=2,
                consequence=ConsequenceVector(
                    latency=0.12,
                    maintenance=0.12,
                    risk=0.1,
                    metabolic=0.1,
                    cognitive_panel=CognitivePanelScore(
                        neurobiology=NeurobiologicalEvaluation(0.15, 0.15, 0.15),
                        psychiatry=PsychiatricEvaluation(0.15, 0.15, 0.15),
                        sociology=SociologicalEvaluation(
                            social_misalignment=0.05,
                            user_fatigue_impact=0.15,
                            coordination_friction=0.05,
                        ),
                    ),
                ),
            )
        )

        # Leaves under f1
        dag.add_node(
            DecisionNode(
                "f1_leaf1",
                "Exhausting Confirmation Prompt Loop 1",
                depth=2,
                consequence=ConsequenceVector(
                    latency=0.1,
                    maintenance=0.1,
                    risk=0.1,
                    metabolic=0.1,
                    cognitive_panel=CognitivePanelScore(
                        neurobiology=NeurobiologicalEvaluation(0.9, 0.9, 0.9),
                        psychiatry=PsychiatricEvaluation(0.9, 0.9, 0.9),
                        sociology=SociologicalEvaluation(
                            social_misalignment=0.9,
                            user_fatigue_impact=0.95,
                            coordination_friction=0.9,
                        ),
                    ),
                ),
            )
        )
        dag.add_node(
            DecisionNode(
                "f1_leaf2",
                "Exhausting Confirmation Prompt Loop 2",
                depth=2,
                consequence=ConsequenceVector(
                    latency=0.1,
                    maintenance=0.1,
                    risk=0.1,
                    metabolic=0.1,
                    cognitive_panel=CognitivePanelScore(
                        neurobiology=NeurobiologicalEvaluation(0.95, 0.95, 0.95),
                        psychiatry=PsychiatricEvaluation(0.95, 0.95, 0.95),
                        sociology=SociologicalEvaluation(
                            social_misalignment=0.95,
                            user_fatigue_impact=0.98,
                            coordination_friction=0.95,
                        ),
                    ),
                ),
            )
        )

        dag.add_edge(DecisionEdge("root", "c1", "Take Concise Autonomous Path"))
        dag.add_edge(DecisionEdge("root", "f1", "Take High Fatigue Path"))
        dag.add_edge(DecisionEdge("c1", "c1_leaf1", "Finish Action 1"))
        dag.add_edge(DecisionEdge("c1", "c1_leaf2", "Finish Action 2"))
        dag.add_edge(DecisionEdge("f1", "f1_leaf1", "Prompt Loop 1"))
        dag.add_edge(DecisionEdge("f1", "f1_leaf2", "Prompt Loop 2"))

        before_nodes = set(dag.nodes.keys())
        before_edges = {
            (u, e.target_id, e.action_label)
            for u, edges in dag.adj.items()
            for e in edges
        }

        goal = "Concise Autonomous Execution and Output"

        res = arbiter.arbitrate_dag(
            dag,
            goal=goal,
            prune_threshold=0.25,
            zeno_lock_threshold=0.45,
            panel_weight=0.85,
        )

        trajs_by_leaf = {t["path_nodes"][-1]: t for t in res["trajectories"]}

        # Clean branches survive
        assert trajs_by_leaf["c1_leaf1"]["is_pruned"] is False
        assert trajs_by_leaf["c1_leaf2"]["is_pruned"] is False

        # Fatigued branches are pruned
        assert trajs_by_leaf["f1_leaf1"]["is_pruned"] is True
        assert trajs_by_leaf["f1_leaf2"]["is_pruned"] is True
        assert "Dominated by winner" in trajs_by_leaf["f1_leaf1"]["prune_reason"]
        assert "Dominated by winner" in trajs_by_leaf["f1_leaf2"]["prune_reason"]

        # Check topology invariance
        after_nodes = set(dag.nodes.keys())
        after_edges = {
            (u, e.target_id, e.action_label)
            for u, edges in dag.adj.items()
            for e in edges
        }
        assert before_nodes == after_nodes
        assert before_edges == after_edges

    def test_pruning_and_topology_invariance_diamond_dag(self) -> None:
        """Tests diamond DAG where two paths diverge from root and reconverge on join node."""
        arbiter = QuantumDecisionArbiter(seed=42)

        dag = DecisionDAG()
        dag.add_node(DecisionNode("root", "Diamond Root", depth=0))

        # Path 1: Concise intermediate node
        dag.add_node(
            DecisionNode(
                "p_concise",
                "Concise Autonomous Step",
                depth=1,
                consequence=ConsequenceVector(
                    latency=0.1,
                    maintenance=0.1,
                    risk=0.1,
                    metabolic=0.1,
                    cognitive_panel=CognitivePanelScore(
                        sociology=SociologicalEvaluation(user_fatigue_impact=0.05)
                    ),
                ),
            )
        )
        # Path 2: Fatiguing intermediate node
        dag.add_node(
            DecisionNode(
                "p_fatigue",
                "Exhausting Interrogative Step",
                depth=1,
                consequence=ConsequenceVector(
                    latency=0.1,
                    maintenance=0.1,
                    risk=0.1,
                    metabolic=0.1,
                    cognitive_panel=CognitivePanelScore(
                        neurobiology=NeurobiologicalEvaluation(0.85, 0.85, 0.85),
                        psychiatry=PsychiatricEvaluation(0.85, 0.85, 0.85),
                        sociology=SociologicalEvaluation(
                            social_misalignment=0.85,
                            user_fatigue_impact=0.95,
                            coordination_friction=0.85,
                        ),
                    ),
                ),
            )
        )
        # Join node
        dag.add_node(
            DecisionNode(
                "join",
                "Reconverged Milestone Goal",
                depth=2,
                consequence=ConsequenceVector(
                    latency=0.05,
                    maintenance=0.05,
                    risk=0.05,
                    metabolic=0.05,
                ),
            )
        )

        dag.add_edge(DecisionEdge("root", "p_concise", "Go Concise"))
        dag.add_edge(DecisionEdge("root", "p_fatigue", "Go Fatiguing"))
        dag.add_edge(DecisionEdge("p_concise", "join", "Join From Concise"))
        dag.add_edge(DecisionEdge("p_fatigue", "join", "Join From Fatigue"))

        before_nodes = set(dag.nodes.keys())
        before_edges = {
            (u, e.target_id, e.action_label)
            for u, edges in dag.adj.items()
            for e in edges
        }

        res = arbiter.arbitrate_dag(
            dag,
            goal="Concise Autonomous Goal Completion",
            prune_threshold=0.25,
            zeno_lock_threshold=0.45,
            panel_weight=0.85,
        )

        trajs = res["trajectories"]
        assert len(trajs) == 2

        concise_traj = next(t for t in trajs if "p_concise" in t["path_nodes"])
        fatigue_traj = next(t for t in trajs if "p_fatigue" in t["path_nodes"])

        assert concise_traj["is_pruned"] is False
        assert fatigue_traj["is_pruned"] is True
        assert "Dominated by winner" in fatigue_traj["prune_reason"]

        # Topology invariance
        after_nodes = set(dag.nodes.keys())
        after_edges = {
            (u, e.target_id, e.action_label)
            for u, edges in dag.adj.items()
            for e in edges
        }
        assert before_nodes == after_nodes
        assert before_edges == after_edges


# ============================================================================
# Section 4: Latency Benchmark (< 5ms and < 15ms)
# ============================================================================


class TestCognitivePanelLatencyBenchmark:
    """Empirical latency benchmark for evaluate_cognitive_panel and analyze_interdisciplinary."""

    def test_evaluate_cognitive_panel_latency_benchmark(self) -> None:
        """Benchmarks evaluate_cognitive_panel across 1,000 iterations.

        Target: strictly < 5.0ms (mean and P95).
        """
        tom = TheoryOfMindAnalyzer()
        test_messages = [
            {"role": "user", "content": "Please implement the biomorphic memory kinetics"},
            {"role": "assistant", "content": "Implementing calibrated biological time scaling"},
            {"role": "user", "content": "The test failed with broken assertion on line 42"},
            {"role": "assistant", "content": "Analyzing exception and repairing synaptic weights", "tool": "run_command"},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "Applying set-shifting flexibility fix", "tool": "run_command"},
            {"role": "user", "content": "done"},
        ]
        test_context = {
            "turn_count": 7,
            "buffer_density": 0.45,
            "repeated_tool_failures": 1,
            "repetition_count": 0,
        }
        test_state = {
            "test_failures": 1,
            "repetition_count": 0,
            "quiescent": False,
        }

        # Warmup
        for _ in range(50):
            tom.evaluate_cognitive_panel(test_messages, test_state, test_context)

        latencies_ms: list[float] = []
        iterations = 1000

        for _ in range(iterations):
            t0 = time.perf_counter()
            panel = tom.evaluate_cognitive_panel(test_messages, test_state, test_context)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            assert panel is not None

        mean_lat = sum(latencies_ms) / len(latencies_ms)
        sorted_lat = sorted(latencies_ms)
        p50 = sorted_lat[int(0.50 * iterations)]
        p95 = sorted_lat[int(0.95 * iterations)]
        p99 = sorted_lat[int(0.99 * iterations)]
        max_lat = max(latencies_ms)

        print(
            f"\n[Latency Benchmark - evaluate_cognitive_panel ({iterations} runs)]\n"
            f"  Mean: {mean_lat:.4f} ms (Target < 5.0 ms)\n"
            f"  P50:  {p50:.4f} ms\n"
            f"  P95:  {p95:.4f} ms (Target < 5.0 ms)\n"
            f"  P99:  {p99:.4f} ms\n"
            f"  Max:  {max_lat:.4f} ms\n"
        )

        assert mean_lat < 5.0, f"Mean latency {mean_lat:.2f}ms exceeded 5.0ms target"
        assert p95 < 5.0, f"P95 latency {p95:.2f}ms exceeded 5.0ms target"

    def test_analyze_interdisciplinary_latency_benchmark(self) -> None:
        """Benchmarks analyze_interdisciplinary across 1,000 iterations.

        Target: strictly < 15.0ms (mean and P95).
        """
        tom = TheoryOfMindAnalyzer()
        test_messages = [
            {"role": "user", "content": "Please implement the biomorphic memory kinetics"},
            {"role": "assistant", "content": "Implementing calibrated biological time scaling"},
            {"role": "user", "content": "The test failed with broken assertion on line 42"},
            {"role": "assistant", "content": "Analyzing exception and repairing synaptic weights", "tool": "run_command"},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "Applying set-shifting flexibility fix", "tool": "run_command"},
            {"role": "user", "content": "done"},
        ]
        test_state = {
            "test_failures": 1,
            "repetition_count": 0,
            "quiescent": False,
        }

        # Warmup
        for _ in range(50):
            tom.analyze_interdisciplinary(test_messages, test_state)

        latencies_ms: list[float] = []
        iterations = 1000

        for _ in range(iterations):
            t0 = time.perf_counter()
            urgency, seeds, panel = tom.analyze_interdisciplinary(test_messages, test_state)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            assert urgency is not None
            assert seeds is not None
            assert panel is not None

        mean_lat = sum(latencies_ms) / len(latencies_ms)
        sorted_lat = sorted(latencies_ms)
        p50 = sorted_lat[int(0.50 * iterations)]
        p95 = sorted_lat[int(0.95 * iterations)]
        p99 = sorted_lat[int(0.99 * iterations)]
        max_lat = max(latencies_ms)

        print(
            f"\n[Latency Benchmark - analyze_interdisciplinary ({iterations} runs)]\n"
            f"  Mean: {mean_lat:.4f} ms (Target < 15.0 ms)\n"
            f"  P50:  {p50:.4f} ms\n"
            f"  P95:  {p95:.4f} ms (Target < 15.0 ms)\n"
            f"  P99:  {p99:.4f} ms\n"
            f"  Max:  {max_lat:.4f} ms\n"
        )

        assert mean_lat < 15.0, f"Mean latency {mean_lat:.2f}ms exceeded 15.0ms target"
        assert p95 < 15.0, f"P95 latency {p95:.2f}ms exceeded 15.0ms target"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--no-cov"])
