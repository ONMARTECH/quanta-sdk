"""Empirical Challenger 1 Test Suite for Milestone M2.

Noetherian Invariant Cognitive Arbiter (Mizan Hakemi) DAG Arbitration & Telemetry.

This suite provides rigorous empirical stress tests for:
1. Branching DAG arbitration where branch A has high utility but breaches semantic charges,
   and branch B with lower utility is compliant.
2. Microglial pruning verification on hard invariant breaches.
3. Telemetry persistence verification for mizan_score, noether_flux, and invariant_violations.
4. Identification and empirical reproduction of defects in arbitrate_dag and telemetry.py.
"""

from __future__ import annotations

import json
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
    ConservedChargeDefinition,
    NoetherianEngineConfig,
    NoetherianInvariantEngine,
)
from quanta.cognitive.telemetry import (
    get_telemetry_summary,
    record_decision_telemetry,
)


@pytest.fixture
def clean_engine() -> NoetherianInvariantEngine:
    """Provides a calibrated NoetherianInvariantEngine with a hard security charge."""
    dim = 64
    gen = torch.zeros(dim)
    gen[0] = 1.0  # Axis 0 generator
    charge = ConservedChargeDefinition(
        charge_id="charge_security",
        rule_name="Zero Drift Security",
        generator_vector=gen,
        target_value=0.0,
        tolerance=0.05,
        is_hard_constraint=True,
    )
    engine = NoetherianInvariantEngine(
        config=NoetherianEngineConfig(dim=dim),
        auto_register_canonical=False,
    )
    engine.add_charge(charge)
    return engine


class TestBranchingDAGArbitration:
    """Stress tests DAG arbitration with branching compliant and rogue paths."""

    def test_branching_dag_hard_breach_microglial_pruning(
        self, clean_engine: NoetherianInvariantEngine
    ) -> None:
        """Branch A has high utility (low cost) but breaches semantic charge.

        Branch B has lower utility (high cost) but zero charge breach.
        Verify that Branch A is microglially pruned and Branch B wins.
        """
        dim = 64
        arbiter = QuantumDecisionArbiter(dim=dim)

        q_root = torch.zeros(dim)
        q_root[1] = 1.0

        # Branch A: High utility, but breaches charge on axis 0 (0.8 > 0.05)
        q_a = torch.zeros(dim)
        q_a[0] = 0.8
        q_a[1] = 0.6
        q_a = q_a / torch.norm(q_a)

        # Branch B: Low utility, zero breach on axis 0
        q_b = torch.zeros(dim)
        q_b[1] = 0.7
        q_b[2] = 0.714
        q_b = q_b / torch.norm(q_b)

        dag = DecisionDAG()
        dag.add_node(DecisionNode(node_id="root", label="Root State", depth=0, statevector=q_root))
        dag.add_node(
            DecisionNode(
                node_id="node_a",
                label="Rogue Branch A",
                depth=1,
                statevector=q_a,
                consequence=ConsequenceVector(latency=0.05, maintenance=0.05, risk=0.05),
            )
        )
        dag.add_node(
            DecisionNode(
                node_id="node_b",
                label="Compliant Branch B",
                depth=1,
                statevector=q_b,
                consequence=ConsequenceVector(latency=0.60, maintenance=0.60, risk=0.60),
            )
        )

        dag.add_edge(DecisionEdge(source_id="root", target_id="node_a", action_label="Choose A"))
        dag.add_edge(DecisionEdge(source_id="root", target_id="node_b", action_label="Choose B"))

        res = arbiter.arbitrate_dag(
            dag=dag,
            goal="Execute secure transaction",
            noetherian_engine=clean_engine,
            log_telemetry=False,
        )

        # 1. Compliant Branch B must win
        assert res["recommended_option"] == "Compliant Branch B"
        assert res["winning_trajectory"]["trajectory_id"] == "tau_2"
        assert res["winning_trajectory"]["is_pruned"] is False
        assert res["winning_trajectory"]["mizan_score"] >= 0.85
        assert len(res["winning_trajectory"]["invariant_violations"]) == 0

        # 2. Rogue Branch A must be microglially pruned
        assert res["pruned_count"] == 1
        pruned_ids = [t["trajectory_id"] for t in res["pruned_trajectories"]]
        assert "tau_1" in pruned_ids

        traj_a = next(t for t in res["trajectories"] if t["trajectory_id"] == "tau_1")
        assert traj_a["is_pruned"] is True
        assert "Noetherian Invariant Breach" in traj_a["prune_reason"]
        assert "Zero Drift Security" in traj_a["prune_reason"]
        assert len(traj_a["invariant_violations"]) > 0

    def test_multi_branch_dag_pareto_breach(
        self, clean_engine: NoetherianInvariantEngine
    ) -> None:
        """3-Branch DAG: Rogue (highest utility, breaches), Compliant-Good (medium utility),

        Compliant-Weak (low utility). Rogue must be pruned; Compliant-Good must win.
        """
        dim = 64
        arbiter = QuantumDecisionArbiter(dim=dim)

        q_root = torch.zeros(dim)
        q_root[1] = 1.0

        # Rogue: breaches axis 0
        q_rogue = torch.zeros(dim)
        q_rogue[0] = 0.75
        q_rogue[1] = 0.66
        q_rogue = q_rogue / torch.norm(q_rogue)

        # Compliant Good: 0 on axis 0, medium cost
        q_good = torch.zeros(dim)
        q_good[1] = 0.8
        q_good[2] = 0.6
        q_good = q_good / torch.norm(q_good)

        # Compliant Weak: 0 on axis 0, high cost
        q_weak = torch.zeros(dim)
        q_weak[2] = 0.7
        q_weak[3] = 0.714
        q_weak = q_weak / torch.norm(q_weak)

        dag = DecisionDAG()
        dag.add_node(DecisionNode(node_id="root", label="Root", depth=0, statevector=q_root))
        dag.add_node(
            DecisionNode(
                node_id="n_rogue",
                label="Rogue",
                depth=1,
                statevector=q_rogue,
                consequence=ConsequenceVector(latency=0.01),
            )
        )
        dag.add_node(
            DecisionNode(
                node_id="n_good",
                label="Compliant Good",
                depth=1,
                statevector=q_good,
                consequence=ConsequenceVector(latency=0.30),
            )
        )
        dag.add_node(
            DecisionNode(
                node_id="n_weak",
                label="Compliant Weak",
                depth=1,
                statevector=q_weak,
                consequence=ConsequenceVector(latency=0.85),
            )
        )

        dag.add_edge(DecisionEdge(source_id="root", target_id="n_rogue", action_label="to Rogue"))
        dag.add_edge(DecisionEdge(source_id="root", target_id="n_good", action_label="to Good"))
        dag.add_edge(DecisionEdge(source_id="root", target_id="n_weak", action_label="to Weak"))

        res = arbiter.arbitrate_dag(
            dag=dag,
            goal="Balanced Performance",
            noetherian_engine=clean_engine,
            log_telemetry=False,
        )

        assert res["recommended_option"] == "Compliant Good"
        assert res["winning_trajectory"]["trajectory_id"] == "tau_2"
        # Rogue is pruned
        pruned_reasons = {t["trajectory_id"]: t["prune_reason"] for t in res["pruned_trajectories"]}
        assert "tau_1" in pruned_reasons
        assert "Noetherian Invariant Breach" in str(pruned_reasons["tau_1"])

    def test_deep_dag_trajectory_noetherian_accumulation(
        self, clean_engine: NoetherianInvariantEngine
    ) -> None:
        """Deep multi-step trajectory (depth=3) where drift accumulates at step 3.

        Step 1 & 2 compliant, Step 3 breaches tolerance -> pruned.
        """
        dim = 64
        arbiter = QuantumDecisionArbiter(dim=dim)

        q0 = torch.zeros(dim)
        q0[1] = 1.0
        q1 = torch.zeros(dim)
        q1[1] = 0.99
        q1[2] = 0.14
        q1 = q1 / torch.norm(q1)
        q2 = torch.zeros(dim)
        q2[1] = 0.98
        q2[2] = 0.20
        q2 = q2 / torch.norm(q2)
        # Deep step 3 injects breach on axis 0
        q3_bad = torch.zeros(dim)
        q3_bad[0] = 0.50
        q3_bad[1] = 0.86
        q3_bad = q3_bad / torch.norm(q3_bad)
        # Deep step 3 safe
        q3_safe = torch.zeros(dim)
        q3_safe[1] = 0.95
        q3_safe[2] = 0.31
        q3_safe = q3_safe / torch.norm(q3_safe)

        dag = DecisionDAG()
        dag.add_node(DecisionNode(node_id="n0", label="Step 0", depth=0, statevector=q0))
        dag.add_node(DecisionNode(node_id="n1", label="Step 1", depth=1, statevector=q1))
        dag.add_node(DecisionNode(node_id="n2", label="Step 2", depth=2, statevector=q2))
        dag.add_node(DecisionNode(node_id="n3_bad", label="Step 3 Rogue", depth=3, statevector=q3_bad))
        dag.add_node(DecisionNode(node_id="n3_safe", label="Step 3 Safe", depth=3, statevector=q3_safe))

        dag.add_edge(DecisionEdge(source_id="n0", target_id="n1", action_label="0->1"))
        dag.add_edge(DecisionEdge(source_id="n1", target_id="n2", action_label="1->2"))
        dag.add_edge(DecisionEdge(source_id="n2", target_id="n3_bad", action_label="2->bad"))
        dag.add_edge(DecisionEdge(source_id="n2", target_id="n3_safe", action_label="2->safe"))

        res = arbiter.arbitrate_dag(
            dag=dag,
            goal="Multi-step Alignment",
            noetherian_engine=clean_engine,
            log_telemetry=False,
        )

        assert res["recommended_option"] == "Step 1"
        assert res["winning_trajectory"]["path_labels"][-1] == "Step 3 Safe"
        bad_traj = next(t for t in res["trajectories"] if t["path_labels"][-1] == "Step 3 Rogue")
        assert bad_traj["is_pruned"] is True
        assert "Noetherian Invariant Breach" in bad_traj["prune_reason"]


class TestEmpiricalBugReproduction:
    """Empirical reproduction of identified defects."""

    def test_bug_1_arbitrate_dag_telemetry_key_error_swallowed(
        self, clean_engine: NoetherianInvariantEngine, tmp_path: Path
    ) -> None:
        """REMEDIATED: arbitrate_dag passes ranking=[t.to_dict() ...] which uses

        'trajectory_id' instead of 'option'. Inside record_decision_telemetry (line 121),
        identifier extraction safely falls back to 'trajectory_id' and persists telemetry.
        """
        dim = 64
        arbiter = QuantumDecisionArbiter(dim=dim)

        dag = DecisionDAG()
        dag.add_node(DecisionNode(node_id="root", label="Root", depth=0))
        dag.add_node(DecisionNode(node_id="a", label="A", depth=1))
        dag.add_edge(DecisionEdge(source_id="root", target_id="a", action_label="to A"))

        custom_log = tmp_path / "test_telemetry.jsonl"

        # Calling record_decision_telemetry with the exact payload from arbitrate_dag
        res = arbiter.arbitrate_dag(
            dag=dag,
            goal="Bug Verification Goal",
            noetherian_engine=clean_engine,
            log_telemetry=False,
        )
        winner = res["winning_trajectory"]
        trajectories = res["trajectories"]

        ranking = [t for t in trajectories]

        # Call record_decision_telemetry with arbitrate_dag payload
        record_decision_telemetry(
            goal="Bug Verification Goal",
            options=[t["trajectory_id"] for t in trajectories],
            winner=winner["trajectory_id"],
            confidence=winner["net_score"],
            zeno_pinning_factor=winner["mean_p_zeno"],
            anti_zeno_kickback=0.0,
            regime="Zeno Pinning",
            latency_ms=10.0,
            ranking=ranking,
            telemetry_file=custom_log,
        )

        # File exists and has positive bytes because KeyError was resolved
        assert custom_log.exists() and custom_log.stat().st_size > 0, (
            "Telemetry log should persist successfully without KeyError"
        )
        with open(custom_log, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["winner"] == winner["trajectory_id"]
        assert rec["confidence"] == winner["net_score"]

    def test_bug_2_negative_utility_mizan_score_inversion(self) -> None:
        """REMEDIATED: In arbitrate_dag line 1475:

        trajectories.sort respects item.net_score so higher Mizan score is monotonically
        rewarded even when (util - cost) is negative.
        """
        dim = 64
        arbiter = QuantumDecisionArbiter(dim=dim)

        # Soft constraint where drift is allowed but reduces Mizan score
        gen = torch.zeros(dim)
        gen[0] = 1.0
        charge = ConservedChargeDefinition(
            charge_id="charge_soft",
            rule_name="Soft Coherence",
            generator_vector=gen,
            target_value=0.0,
            tolerance=0.30,
            is_hard_constraint=False,
        )
        engine = NoetherianInvariantEngine(
            config=NoetherianEngineConfig(dim=dim),
            auto_register_canonical=False,
        )
        engine.add_charge(charge)

        q_root = torch.zeros(dim)
        q_root[1] = 1.0
        # Branch A: Drifting state (lower Mizan score ~0.75, valid)
        q_a = torch.zeros(dim)
        q_a[0] = 0.15
        q_a[1] = 0.988
        q_a = q_a / torch.norm(q_a)
        # Branch B: Zero drift state (perfect Mizan score 1.0, valid)
        q_b = torch.zeros(dim)
        q_b[1] = 1.0

        # Consequence cost is high (0.9), creating negative (utility - cost)
        dag = DecisionDAG()
        dag.add_node(DecisionNode(node_id="root", label="Root", depth=0, statevector=q_root))
        dag.add_node(
            DecisionNode(
                node_id="n_a",
                label="Branch A Drifting",
                depth=1,
                statevector=q_a,
                consequence=ConsequenceVector(latency=0.9, maintenance=0.9),
            )
        )
        dag.add_node(
            DecisionNode(
                node_id="n_b",
                label="Branch B Compliant",
                depth=1,
                statevector=q_b,
                consequence=ConsequenceVector(latency=0.9, maintenance=0.9),
            )
        )
        dag.add_edge(DecisionEdge(source_id="root", target_id="n_a", action_label="to A"))
        dag.add_edge(DecisionEdge(source_id="root", target_id="n_b", action_label="to B"))

        res = arbiter.arbitrate_dag(
            dag=dag,
            goal="Negative Utility Inversion Demonstration",
            noetherian_engine=engine,
            log_telemetry=False,
        )

        traj_a = next(t for t in res["trajectories"] if t["path_labels"][1] == "Branch A Drifting")
        traj_b = next(t for t in res["trajectories"] if t["path_labels"][1] == "Branch B Compliant")

        # In net_score, Branch B correctly beats Branch A (positive normalized scores)
        assert traj_b["mizan_score"] > traj_a["mizan_score"]
        assert traj_b["net_score"] > traj_a["net_score"]

        # With the fix, Branch B Compliant is correctly selected as the winner
        assert res["recommended_option"] == "Branch B Compliant"
        assert res["winning_trajectory"]["trajectory_id"] == traj_b["trajectory_id"]


class TestTelemetryFieldPersistenceWhenCorrected:
    """Verifies that telemetry records all required Noetherian fields when payload is well-formed."""

    def test_telemetry_records_mizan_flux_and_violations(self, tmp_path: Path) -> None:
        """When called with well-formed ranking/tr_rho_pi, record_decision_telemetry

        faithfully persists mizan_score, noether_flux, and invariant_violations.
        """
        test_file = tmp_path / "test_quanta_telemetry.jsonl"

        record_decision_telemetry(
            goal="Verification of Invariant Telemetry",
            options=["tau_compliant", "tau_rogue"],
            winner="tau_compliant",
            confidence=0.88,
            zeno_pinning_factor=0.82,
            anti_zeno_kickback=0.04,
            regime="HARMONIC_EQUILIBRIUM",
            latency_ms=8.5,
            ranking=[
                {"option": "tau_compliant", "score": 0.88},
                {"option": "tau_rogue", "score": 0.12},
            ],
            tr_rho_pi={"tau_compliant": 0.88, "tau_rogue": 0.12},
            telemetry_file=test_file,
            mizan_score=0.9421,
            noether_flux=0.0152,
            invariant_violations=["Charge 'Security' breached at step 1: drift=0.12 > tol=0.05"],
        )

        assert test_file.exists()
        with open(test_file, encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 1
        record = json.loads(lines[0])

        assert record["event_type"] == "decision"
        assert record["winner"] == "tau_compliant"
        assert record["mizan_score"] == 0.9421
        assert record["noether_flux"] == 0.0152
        assert record["invariant_violations"] == [
            "Charge 'Security' breached at step 1: drift=0.12 > tol=0.05"
        ]

        summary = get_telemetry_summary(telemetry_file=test_file)
        assert summary["avg_mizan_score"] == 0.9421
        assert summary["total_invariant_violations"] == 1
        assert summary["noether_flux_p95"] == 0.0152
