"""Empirical Invalidation and Boundary Test Suite for Milestone M2.

Challenger 2 Protocol:
1. Invalidation Case 1: arbitrate_dag when all trajectories violate hard constraints.
2. Invalidation Case 2: get_telemetry_summary with zero records, corrupt JSON lines, non-dict entries, and missing Mizan fields.
3. Invalidation Case 3: Metric serialization roundtrip (mizan_score, noether_flux, invariant_violations) across JSONL persistence and summary aggregation.
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
    InvariantViolation,
    NoetherianEngineConfig,
    NoetherianInvariantEngine,
)
from quanta.cognitive.telemetry import (
    get_telemetry_summary,
    read_telemetry_events,
    record_decision_telemetry,
)

# ============================================================================
# Invalidation Case 1: All Trajectories Breach Hard Constraints in arbitrate_dag
# ============================================================================


class TestArbitrateDagAllTrajectoriesBreaching:
    """Verifies arbitrate_dag behavior when all branches breach hard invariant constraints."""

    @pytest.fixture
    def arbiter_and_dag(self) -> tuple[QuantumDecisionArbiter, DecisionDAG, ConservedChargeDefinition]:
        """Creates an arbiter and a 3-branch DecisionDAG with an unavoidable hard constraint."""
        dim = 64
        # Create a hard charge with generator along dimension 0, target value 1.0, tolerance 0.01
        gen_vec = torch.zeros(dim, dtype=torch.float32)
        gen_vec[0] = 1.0

        hard_charge = ConservedChargeDefinition(
            charge_id="hard_must_be_one",
            rule_name="Strict Dimension 0 Charge",
            description="Requires state to have projection 1.0 on axis 0",
            generator_vector=gen_vec,
            target_value=1.0,
            tolerance=0.01,
            is_hard_constraint=True,
        )

        config = NoetherianEngineConfig(
            dim=dim,
            critical_breach_threshold=0.40,
        )
        # Register the hard charge explicitly in the engine
        engine = NoetherianInvariantEngine(config=config, charges=[hard_charge])
        arbiter = QuantumDecisionArbiter(dim=dim, num_heads=4, noetherian_engine=engine)

        # Build a DAG where ALL nodes have statevectors along dimension 1 (orthogonal to dim 0)
        # So charge is 0.0 everywhere, producing a massive drift |0.0 - 1.0| = 1.0 >> 0.01
        dag = DecisionDAG()
        v_root = torch.zeros(dim, dtype=torch.float32)
        v_root[1] = 1.0
        dag.add_node(DecisionNode(node_id="root", label="Root Node", statevector=v_root))

        for i in range(1, 4):
            v_branch = torch.zeros(dim, dtype=torch.float32)
            v_branch[1] = 0.8
            v_branch[i + 1] = 0.6
            v_branch = v_branch / torch.norm(v_branch, p=2)

            node_id = f"branch_{i}"
            dag.add_node(
                DecisionNode(
                    node_id=node_id,
                    label=f"Violating Option {i}",
                    statevector=v_branch,
                    consequence=ConsequenceVector(latency=0.1 * i, risk=0.2 * i),
                )
            )
            dag.add_edge(
                DecisionEdge(
                    source_id="root",
                    target_id=node_id,
                    action_label=f"Select Option {i}",
                )
            )

        return arbiter, dag, hard_charge

    def test_all_trajectories_breached_no_exception(
        self,
        arbiter_and_dag: tuple[QuantumDecisionArbiter, DecisionDAG, ConservedChargeDefinition],
    ) -> None:
        """Confirms that arbitrate_dag does not raise uncaught exceptions when all paths violate hard constraints."""
        arbiter, dag, _ = arbiter_and_dag
        res = arbiter.arbitrate_dag(
            dag=dag,
            goal="Test Goal",
            log_telemetry=False,
        )
        assert isinstance(res, dict)
        assert "winning_trajectory" in res
        assert "trajectories" in res
        assert "surviving_trajectories" in res
        assert "pruned_trajectories" in res

    def test_all_branches_pruned_and_flagged(
        self,
        arbiter_and_dag: tuple[QuantumDecisionArbiter, DecisionDAG, ConservedChargeDefinition],
    ) -> None:
        """Verifies that all trajectories are pruned and properly flagged with breach reasons."""
        arbiter, dag, _ = arbiter_and_dag
        res = arbiter.arbitrate_dag(
            dag=dag,
            goal="Test Goal",
            log_telemetry=False,
        )

        total = res["total_trajectories"]
        pruned_count = res["pruned_count"]
        assert total == 3
        assert pruned_count == 3, f"Expected all 3 trajectories to be pruned, got {pruned_count}"

        # Surviving trajectories should be empty
        assert len(res["surviving_trajectories"]) == 0

        # All pruned trajectories must have is_pruned == True and prune_reason mentioning invariant breach
        for t in res["pruned_trajectories"]:
            assert t["is_pruned"] is True
            assert "Noetherian Invariant Breach" in str(t["prune_reason"])
            assert len(t["invariant_violations"]) > 0

    def test_winner_fallback_structure_preserved(
        self,
        arbiter_and_dag: tuple[QuantumDecisionArbiter, DecisionDAG, ConservedChargeDefinition],
    ) -> None:
        """Verifies that winning_trajectory returns the best relative fallback despite being pruned."""
        arbiter, dag, _ = arbiter_and_dag
        res = arbiter.arbitrate_dag(
            dag=dag,
            goal="Test Goal",
            log_telemetry=False,
        )
        winner = res["winning_trajectory"]
        assert winner["trajectory_id"].startswith("tau_")
        assert len(winner["path_nodes"]) == 2
        assert winner["is_pruned"] is True  # Marked pruned because all are invalid
        assert "Noetherian Invariant Breach" in str(winner["prune_reason"])
        # Confidence penalized by factor of 0.1
        assert winner["net_score"] >= 0.0

    def test_multibranch_wrapper_with_all_breaching(
        self,
        arbiter_and_dag: tuple[QuantumDecisionArbiter, DecisionDAG, ConservedChargeDefinition],
    ) -> None:
        """Verifies arbitrate_multibranch does not crash when all branches breach."""
        arbiter, dag, _ = arbiter_and_dag
        # Run arbitrate_multibranch with equivalent branch dictionary
        branches = {
            "Option 1": {"consequences": {"latency": 0.1}},
            "Option 2": {"consequences": {"latency": 0.2}},
        }
        res = arbiter.arbitrate_multibranch(
            goal="Goal",
            branches=branches,
            log_telemetry=False,
        )
        assert "executive_summary" in res
        assert "winning_trajectory" in res
        assert res["total_trajectories"] == 2

    def test_single_node_single_trajectory_breach(self) -> None:
        """Boundary test: 1-node DAG where the single node breaches hard constraint."""
        dim = 64
        gen = torch.zeros(dim, dtype=torch.float32)
        gen[0] = 1.0
        hard_charge = ConservedChargeDefinition(
            charge_id="c1",
            rule_name="C1",
            generator_vector=gen,
            target_value=1.0,
            tolerance=0.01,
            is_hard_constraint=True,
        )
        engine = NoetherianInvariantEngine(
            config=NoetherianEngineConfig(dim=dim),
            charges=[hard_charge],
        )
        arbiter = QuantumDecisionArbiter(dim=dim, noetherian_engine=engine)

        dag = DecisionDAG()
        v_bad = torch.zeros(dim, dtype=torch.float32)
        v_bad[1] = 1.0  # violates C1
        dag.add_node(DecisionNode(node_id="single", label="Single Node", statevector=v_bad))

        res = arbiter.arbitrate_dag(dag=dag, goal="Goal", log_telemetry=False)
        assert res["total_trajectories"] == 1
        assert res["pruned_count"] == 1
        assert len(res["surviving_trajectories"]) == 0
        assert res["winning_trajectory"]["is_pruned"] is True

    def test_contrast_one_valid_two_invalid(self) -> None:
        """Verifies that when at least 1 trajectory is valid, it survives unpruned while breached are pruned."""
        dim = 64
        gen = torch.zeros(dim, dtype=torch.float32)
        gen[0] = 1.0
        hard_charge = ConservedChargeDefinition(
            charge_id="c_axis0",
            rule_name="Axis 0 Strict",
            generator_vector=gen,
            target_value=1.0,
            tolerance=0.05,
            is_hard_constraint=True,
        )
        engine = NoetherianInvariantEngine(
            config=NoetherianEngineConfig(dim=dim),
            charges=[hard_charge],
        )
        arbiter = QuantumDecisionArbiter(dim=dim, noetherian_engine=engine)

        dag = DecisionDAG()
        # Root satisfies constraint
        v_root = torch.zeros(dim, dtype=torch.float32)
        v_root[0] = 1.0
        dag.add_node(DecisionNode(node_id="root", label="Root", statevector=v_root))

        # Branch 1 satisfies constraint (q[0] = 1.0)
        v_valid = torch.zeros(dim, dtype=torch.float32)
        v_valid[0] = 1.0
        dag.add_node(DecisionNode(node_id="valid_b", label="Valid Branch", statevector=v_valid))
        dag.add_edge(DecisionEdge(source_id="root", target_id="valid_b", action_label="Go Valid"))

        # Branch 2 violates constraint (q[1] = 1.0, q[0] = 0.0)
        v_invalid = torch.zeros(dim, dtype=torch.float32)
        v_invalid[1] = 1.0
        dag.add_node(DecisionNode(node_id="invalid_b", label="Invalid Branch", statevector=v_invalid))
        dag.add_edge(DecisionEdge(source_id="root", target_id="invalid_b", action_label="Go Invalid"))

        res = arbiter.arbitrate_dag(dag=dag, goal="Goal", log_telemetry=False)
        assert res["total_trajectories"] == 2
        assert len(res["surviving_trajectories"]) == 1
        survivor_ids = [t["path_nodes"][-1] for t in res["surviving_trajectories"]]
        assert "valid_b" in survivor_ids
        assert "invalid_b" not in survivor_ids
        assert res["winning_trajectory"]["path_nodes"][-1] == "valid_b"
        assert res["winning_trajectory"]["is_pruned"] is False


# ============================================================================
# Invalidation Case 2: get_telemetry_summary Robustness with Corrupted Records
# ============================================================================


class TestTelemetrySummaryRobustness:
    """Tests get_telemetry_summary against empty, corrupted, and malformed telemetry files."""

    def test_nonexistent_and_empty_file(self, tmp_path: Path) -> None:
        """Verifies graceful defaults when telemetry file does not exist or is 0 bytes."""
        non_existent = tmp_path / "does_not_exist.jsonl"
        summary_empty = get_telemetry_summary(telemetry_file=non_existent)
        assert summary_empty["total_decisions"] == 0
        assert summary_empty["total_hook_steps"] == 0
        assert summary_empty["avg_mizan_score"] == 1.0
        assert summary_empty["noether_flux_p95"] == 0.0
        assert summary_empty["total_invariant_violations"] == 0

        # Now test 0-byte file
        zero_file = tmp_path / "zero_bytes.jsonl"
        zero_file.touch()
        summary_zero = get_telemetry_summary(telemetry_file=zero_file)
        assert summary_zero["total_decisions"] == 0
        assert summary_zero["total_hook_steps"] == 0
        assert summary_zero["avg_mizan_score"] == 1.0
        assert summary_zero["noether_flux_p95"] == 0.0
        assert summary_zero["total_invariant_violations"] == 0

    def test_corrupt_json_lines_and_non_dict_entries(self, tmp_path: Path) -> None:
        """Verifies parser handles completely corrupted JSON and primitive types gracefully."""
        log_file = tmp_path / "corrupt.jsonl"
        lines = [
            "",  # empty line
            "   ",  # whitespace
            "{broken json raw text :::",  # syntax error
            "12345",  # int
            '"just a string"',  # string
            "[1, 2, 3]",  # list
            "true",  # bool
            "null",  # null
            json.dumps({"event_type": "unknown_event", "foo": "bar"}),  # unrelated event
            json.dumps({  # valid decision record
                "event_type": "decision",
                "goal": "Test Goal",
                "winner": "opt_a",
                "confidence": 0.88,
                "zeno_pinning_factor": 0.85,
                "latency_ms": 12.5,
                "mizan_score": 0.92,
                "noether_flux": 0.04,
                "invariant_violations": ["breach_1"],
            }),
            "{trailing broken line",
        ]
        log_file.write_text("\n".join(lines), encoding="utf-8")

        summary = get_telemetry_summary(telemetry_file=log_file)
        assert summary["total_decisions"] == 1
        assert summary["avg_mizan_score"] == pytest.approx(0.92, abs=1e-4)
        assert summary["noether_flux_p95"] == pytest.approx(0.04, abs=1e-4)
        assert summary["total_invariant_violations"] == 1

    def test_missing_mizan_fields_legacy_records(self, tmp_path: Path) -> None:
        """Verifies backward compatibility with pre-M2 legacy records lacking Mizan fields."""
        log_file = tmp_path / "legacy.jsonl"
        records = [
            {
                "event_type": "decision",
                "goal": "Legacy Decision 1",
                "winner": "w1",
                "confidence": 0.75,
                "zeno_pinning_factor": 0.80,
                "latency_ms": 20.0,
                # Missing all mizan fields
            },
            {
                "event_type": "decision",
                "goal": "Legacy Decision 2",
                "winner": "w2",
                "confidence": 0.85,
                "zeno_pinning_factor": 0.82,
                "latency_ms": 15.0,
                # Missing all mizan fields
            },
        ]
        with open(log_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        summary = get_telemetry_summary(telemetry_file=log_file)
        assert summary["total_decisions"] == 2
        # Defaults when no valid mizan records exist
        assert summary["avg_mizan_score"] == 1.0
        assert summary["noether_flux_p95"] == 0.0
        assert summary["total_invariant_violations"] == 0

    def test_corrupted_field_types(self, tmp_path: Path) -> None:
        """Verifies resilience against extreme types (NaN, Inf, strings, nested objects) in metrics."""
        log_file = tmp_path / "bad_types.jsonl"
        records = [
            {
                "event_type": "decision",
                "goal": "Bad types record 1",
                "winner": "w1",
                "confidence": "high",  # non-float
                "zeno_pinning_factor": False,  # bool instead of float
                "latency_ms": None,
                "mizan_score": "invalid_string",
                "noether_flux": {"nested": "dict"},
                "invariant_violations": "single_string_instead_of_list",
            },
            {
                "event_type": "decision",
                "goal": "Bad types record 2",
                "winner": "w2",
                "confidence": 0.90,
                "zeno_pinning_factor": 0.70,
                "latency_ms": 10.0,
                "mizan_score": 0.85,
                "noether_flux": 0.15,
                "invariant_violations": ["v1", "v2"],
            },
        ]
        with open(log_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        summary = get_telemetry_summary(telemetry_file=log_file)
        assert summary["total_decisions"] == 2
        # Only record 2 has valid float mizan_score
        assert summary["avg_mizan_score"] == pytest.approx(0.85, abs=1e-4)
        assert summary["noether_flux_p95"] == pytest.approx(0.15, abs=1e-4)
        # record 2 has 2 violations; record 1 had a string which is safely ignored
        assert summary["total_invariant_violations"] == 2


# ============================================================================
# Invalidation Case 3: Metric Serialization Roundtrip & Aggregation
# ============================================================================


class TestMetricSerializationRoundtrip:
    """Verifies mizan_score, noether_flux, and invariant_violations roundtrip serialization."""

    def test_record_decision_telemetry_direct_args(self, tmp_path: Path) -> None:
        """Tests that record_decision_telemetry correctly writes explicit Mizan arguments."""
        log_file = tmp_path / "direct_args.jsonl"
        record_decision_telemetry(
            goal="Goal 1",
            options=["a", "b"],
            winner="a",
            confidence=0.9123,
            zeno_pinning_factor=0.8456,
            anti_zeno_kickback=0.1544,
            regime="Zeno Pinning",
            latency_ms=18.4,
            mizan_score=0.8765,
            noether_flux=0.0345,
            invariant_violations=["Violation A", "Violation B"],
            telemetry_file=log_file,
        )

        events = read_telemetry_events(telemetry_file=log_file)
        assert len(events) == 1
        rec = events[0]
        assert rec["mizan_score"] == pytest.approx(0.8765, abs=1e-4)
        assert rec["noether_flux"] == pytest.approx(0.0345, abs=1e-4)
        assert rec["invariant_violations"] == ["Violation A", "Violation B"]

    def test_record_decision_telemetry_via_kwargs(self, tmp_path: Path) -> None:
        """Tests that record_decision_telemetry correctly captures Mizan metrics when passed via kwargs."""
        log_file = tmp_path / "kwargs.jsonl"
        record_decision_telemetry(
            goal="Goal 2",
            options=["x", "y"],
            winner="x",
            confidence=0.88,
            zeno_pinning_factor=0.80,
            anti_zeno_kickback=0.20,
            regime="Zeno Pinning",
            latency_ms=15.0,
            telemetry_file=log_file,
            # Passed via kwargs
            mizan_score=0.7654,
            noether_flux=0.0891,
            invariant_violations=["Kwarg Breach 1"],
        )

        events = read_telemetry_events(telemetry_file=log_file)
        assert len(events) == 1
        rec = events[0]
        assert rec["mizan_score"] == pytest.approx(0.7654, abs=1e-4)
        assert rec["noether_flux"] == pytest.approx(0.0891, abs=1e-4)
        assert rec["invariant_violations"] == ["Kwarg Breach 1"]

    def test_strongly_typed_invariant_violation_objects_serialization(self, tmp_path: Path) -> None:
        """Verifies that InvariantViolation objects serialize without TypeError to pure strings."""
        log_file = tmp_path / "violation_types.jsonl"
        typed_violations = [
            InvariantViolation(
                "Breach of Safety Rule",
                charge_id="c_safety",
                rule_name="Safety Rule",
                step=1,
                observed_drift=0.09,
                tolerance=0.03,
                is_hard_constraint=True,
            ),
            InvariantViolation(
                "Breach of Intent Rule",
                charge_id="c_intent",
                rule_name="Intent Rule",
                step=2,
                observed_drift=0.06,
                tolerance=0.04,
                is_hard_constraint=False,
            ),
        ]

        record_decision_telemetry(
            goal="Goal with typed violations",
            options=["o1", "o2"],
            winner="o1",
            confidence=0.85,
            zeno_pinning_factor=0.82,
            anti_zeno_kickback=0.18,
            regime="Zeno Pinning",
            latency_ms=12.0,
            mizan_score=0.65,
            noether_flux=0.12,
            invariant_violations=typed_violations,
            telemetry_file=log_file,
        )

        events = read_telemetry_events(telemetry_file=log_file)
        assert len(events) == 1
        rec = events[0]
        assert len(rec["invariant_violations"]) == 2
        assert rec["invariant_violations"][0] == "Breach of Safety Rule"
        assert rec["invariant_violations"][1] == "Breach of Intent Rule"

    def test_summary_mathematical_aggregation_roundtrip(self, tmp_path: Path) -> None:
        """Verifies that avg_mizan_score, noether_flux_p95, and violations count compute accurately."""
        log_file = tmp_path / "aggregation.jsonl"
        # Record 10 decisions with deterministic values:
        # mizan: 0.50, 0.55, 0.60, ..., 0.95 (mean = 0.725)
        # flux: 0.01, 0.02, 0.03, ..., 0.10 (p95 index = 9 -> 0.10)
        # violations: 1 violation each = 10 total
        for i in range(10):
            m_score = 0.50 + 0.05 * i
            flux = 0.01 * (i + 1)
            record_decision_telemetry(
                goal=f"Decision {i}",
                options=["opt_a", "opt_b"],
                winner="opt_a",
                confidence=0.85,
                zeno_pinning_factor=0.82,
                anti_zeno_kickback=0.18,
                regime="Zeno Pinning",
                latency_ms=10.0,
                mizan_score=m_score,
                noether_flux=flux,
                invariant_violations=[f"Violation {i}"],
                telemetry_file=log_file,
            )

        summary = get_telemetry_summary(telemetry_file=log_file)
        assert summary["total_decisions"] == 10
        expected_avg_mizan = round(sum(0.50 + 0.05 * i for i in range(10)) / 10.0, 4)
        assert summary["avg_mizan_score"] == pytest.approx(expected_avg_mizan, abs=1e-4)
        assert summary["noether_flux_p95"] == pytest.approx(0.10, abs=1e-4)
        assert summary["total_invariant_violations"] == 10

    def test_boundary_values_nan_inf_none(self, tmp_path: Path) -> None:
        """Verifies that NaN, +/-Inf, and None in mizan_score and noether_flux fall back safely."""
        log_file = tmp_path / "extremes.jsonl"

        record_decision_telemetry(
            goal="Goal with NaNs",
            options=["o1", "o2"],
            winner="o1",
            confidence=0.85,
            zeno_pinning_factor=0.82,
            anti_zeno_kickback=0.18,
            regime="Zeno Pinning",
            latency_ms=10.0,
            mizan_score=float("nan"),
            noether_flux=float("inf"),
            invariant_violations=None,
            telemetry_file=log_file,
        )

        events = read_telemetry_events(telemetry_file=log_file)
        assert len(events) == 1
        rec = events[0]
        # Should be sanitized to 0.0 in record_decision_telemetry
        assert rec["mizan_score"] == 0.0
        assert rec["noether_flux"] == 0.0
        assert "invariant_violations" not in rec or rec["invariant_violations"] is None or rec["invariant_violations"] == []

        summary = get_telemetry_summary(telemetry_file=log_file)
        assert summary["total_decisions"] == 1
        assert summary["avg_mizan_score"] == 0.0
        assert summary["noether_flux_p95"] == 0.0
        assert summary["total_invariant_violations"] == 0
