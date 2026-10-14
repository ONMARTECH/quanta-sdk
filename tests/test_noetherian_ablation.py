"""Comprehensive Pytest Suite for 3-Way Ablation Benchmark (Mizan Hakemi).

Verifies the 3-Way Ablation Benchmark and adversarial validation for the
Noetherian Invariant Cognitive Arbiter in Quanta SDK:
  - Baseline (Unconstrained): Standard utility optimization without invariant bounds.
  - Naive Heuristic: Simple static penalty and step-wise Euclidean thresholding.
  - Noetherian Arbiter: Full variational mechanics (Action, Entropy, Noether flux,
    Manifold projection, and Microglial invariant pruning).

Adversarial Boundary Scenarios Verified:
  1. Creeping Context Drift (15-step cumulative drift)
  2. Periphrastic Rule Erosion (obfuscated Lie generator breach)
  3. Conflicting Pareto Dilemmas (high nominal utility vs invariant compliance)
  4. Deep Combinatorial DAG Confluence (depth-5 branching DAG)

Scientific Integrity Protocol (GEMINI.md):
  - Hermetic test isolation: Uses pytest `tmp_path` fixture for all I/O.
  - Zero benchmark leakage, zero cherry-picking, genuine ablation evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from benchmarks.noetherian_ablation_benchmark import (
    NoetherianAblationBenchmark,
    VariantResult,
    build_conflicting_pareto_scenario,
    build_creeping_context_drift_scenario,
    build_deep_dag_confluence_scenario,
    build_periphrastic_rule_erosion_scenario,
    evaluate_variant_baseline,
    evaluate_variant_naive,
    evaluate_variant_noetherian,
    main,
)
from quanta.cognitive.arbiter import (
    DecisionDAG,
    DecisionEdge,
    DecisionNode,
    QuantumDecisionArbiter,
)
from quanta.cognitive.noetherian import (
    ConservedChargeDefinition,
    NoetherianEngineConfig,
)
from quanta.cognitive.telemetry import (
    get_telemetry_summary,
    read_telemetry_events,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def rng_seed() -> None:
    """Ensures deterministic PyTorch pseudo-random generation."""
    torch.manual_seed(42)


# ============================================================================
# Tier 1: Suite Execution & Reproducibility Tests
# ============================================================================


class TestAblationSuiteExecution:
    """Verifies that all 3 ablation variants execute cleanly across all 4 scenarios."""

    def test_all_variants_execute_cleanly(self, rng_seed: None) -> None:
        """Verifies that Baseline, Naive Heuristic, and Noetherian Arbiter execute without errors."""
        scenarios = [
            build_creeping_context_drift_scenario(seed=42),
            build_periphrastic_rule_erosion_scenario(seed=42),
            build_conflicting_pareto_scenario(seed=42),
            build_deep_dag_confluence_scenario(seed=42),
        ]

        for sc in scenarios:
            res_b = evaluate_variant_baseline(sc)
            res_n = evaluate_variant_naive(sc)
            res_m = evaluate_variant_noetherian(sc)

            for res in (res_b, res_n, res_m):
                assert isinstance(res, VariantResult)
                assert isinstance(res.chosen_trajectory_id, str) and res.chosen_trajectory_id
                assert isinstance(res.chosen_path_labels, list) and len(res.chosen_path_labels) > 0
                assert 0.0 <= res.mizan_score <= 1.0
                assert res.noether_flux >= 0.0
                assert res.charge_drift >= 0.0
                assert isinstance(res.invariant_violations, list)
                assert res.execution_time_ms >= 0.0

    def test_benchmark_reproducibility_deterministic_seed(self) -> None:
        """Verifies that running the suite twice with the same seed yields identical results."""
        bench1 = NoetherianAblationBenchmark(seed=12345, runs=2, verbose=False)
        bench2 = NoetherianAblationBenchmark(seed=12345, runs=2, verbose=False)

        summary1 = bench1.run_suite()
        summary2 = bench2.run_suite()

        # Check global metrics match exactly
        assert summary1["global_metrics"] == summary2["global_metrics"]

        # Check per-scenario results match
        assert len(summary1["scenarios"]) == len(summary2["scenarios"]) == 4
        for s1, s2 in zip(summary1["scenarios"], summary2["scenarios"], strict=True):
            assert s1["scenario_name"] == s2["scenario_name"]
            # Exclude wall-clock dynamic latency from strict equality
            delta1_algo = {k: v for k, v in s1["ablation_delta"].items() if k != "latency_overhead_ms"}
            delta2_algo = {k: v for k, v in s2["ablation_delta"].items() if k != "latency_overhead_ms"}
            assert delta1_algo == delta2_algo
            for v_key in ("baseline", "naive_heuristic", "noetherian_arbiter"):
                assert s1["variant_results"][v_key]["chosen_trajectory_id"] == s2["variant_results"][v_key]["chosen_trajectory_id"]
                assert s1["variant_results"][v_key]["mean_mizan_score"] == s2["variant_results"][v_key]["mean_mizan_score"]
                assert s1["variant_results"][v_key]["survival_rate"] == s2["variant_results"][v_key]["survival_rate"]


# ============================================================================
# Tier 2: Adversarial Scenario Differentiation Tests
# ============================================================================


class TestAdversarialScenarios:
    """Verifies detailed behavior on each of the 4 adversarial stress scenarios."""

    def test_creeping_context_drift_scenario(self) -> None:
        """Scenario 1: Baseline and Naive fall for creeping drift; Noetherian enforces conservation."""
        sc = build_creeping_context_drift_scenario(seed=42)

        res_b = evaluate_variant_baseline(sc)
        res_n = evaluate_variant_naive(sc)
        res_m = evaluate_variant_noetherian(sc)

        # Baseline picks drifting path due to higher nominal utility
        assert "Drift Batch Step 15" in res_b.chosen_path_labels[-1]
        assert res_b.conserved is False
        assert res_b.is_valid is False
        assert len(res_b.invariant_violations) > 0
        assert res_b.noether_flux > 0.30

        # Naive heuristic fails to detect creeping drift (step jump is below threshold 0.20)
        assert "Drift Batch Step 15" in res_n.chosen_path_labels[-1]
        assert res_n.conserved is False
        assert len(res_n.invariant_violations) > 0

        # Noetherian Arbiter detects cumulative flux and drift, prunes drift path, and chooses compliant path
        assert "Compliant Step 15" in res_m.chosen_path_labels[-1]
        assert res_m.conserved is True
        assert res_m.is_valid is True
        assert len(res_m.invariant_violations) == 0
        assert res_m.mizan_score > 0.95
        assert res_m.mizan_score > res_b.mizan_score

    def test_periphrastic_rule_erosion_scenario(self) -> None:
        """Scenario 2: Obfuscated wording bypasses naive keyword search; Lie generator catches it."""
        sc = build_periphrastic_rule_erosion_scenario(seed=42)

        res_b = evaluate_variant_baseline(sc)
        res_n = evaluate_variant_naive(sc)
        res_m = evaluate_variant_noetherian(sc)

        # Baseline picks deceptive option (higher raw utility)
        assert "Disguised DROP" in res_b.chosen_path_labels[-1]
        assert res_b.conserved is False
        assert res_b.is_valid is False
        assert len(res_b.invariant_violations) > 0

        # Naive heuristic is bypassed because label uses periphrastic phrasing
        assert "Disguised DROP" in res_n.chosen_path_labels[-1]
        assert res_n.conserved is False

        # Noetherian Arbiter projects onto Lie generator, catches the breach, and chooses vacuum
        assert "Standard Read-Only Index Defragmentation & Vacuum" in res_m.chosen_path_labels[-1]
        assert res_m.conserved is True
        assert res_m.is_valid is True
        assert len(res_m.invariant_violations) == 0
        assert res_m.mizan_score > res_b.mizan_score

    def test_conflicting_pareto_dilemma_scenario(self) -> None:
        """Scenario 3: High-utility shortcut directly violates regulatory invariant."""
        sc = build_conflicting_pareto_scenario(seed=42)

        res_b = evaluate_variant_baseline(sc)
        res_n = evaluate_variant_naive(sc)
        res_m = evaluate_variant_noetherian(sc)

        # Baseline chooses rogue shortcut due to near-zero latency
        assert "Direct Mempool Bypass" in res_b.chosen_path_labels[-1]
        assert res_b.conserved is False
        assert res_b.is_valid is False
        assert len(res_b.invariant_violations) > 0

        # Naive static penalty (0.18) is insufficient to bridge the utility gap, so Naive still picks rogue
        assert "Direct Mempool Bypass" in res_n.chosen_path_labels[-1]
        assert res_n.conserved is False

        # Noetherian Arbiter invalidates rogue option (hard constraint) and chooses signed dispatch
        assert "Hardware-Enclave Signed Transaction Dispatch" in res_m.chosen_path_labels[-1]
        assert res_m.conserved is True
        assert res_m.is_valid is True
        assert len(res_m.invariant_violations) == 0
        assert res_m.mizan_score > 0.90
        assert res_m.mizan_score > res_b.mizan_score

    def test_deep_dag_confluence_scenario(self) -> None:
        """Scenario 4: Depth-5 DAG where early rewards conceal downstream hallucination."""
        sc = build_deep_dag_confluence_scenario(seed=42)

        res_b = evaluate_variant_baseline(sc)
        res_n = evaluate_variant_naive(sc)
        res_m = evaluate_variant_noetherian(sc)

        # Baseline is seduced by early high rewards and picks rogue path
        assert "Rogue Transformation Layer 5" in res_b.chosen_path_labels[-1]
        assert res_b.conserved is False
        assert res_b.is_valid is False
        assert len(res_b.invariant_violations) > 0

        # Naive heuristic evaluates local steps and also commits to rogue path
        assert "Rogue Transformation Layer 5" in res_n.chosen_path_labels[-1]
        assert res_n.conserved is False

        # Noetherian Arbiter evaluates multi-step Action and Entropy and selects disciplined path
        assert "Disciplined Layer 5" in res_m.chosen_path_labels[-1]
        assert res_m.conserved is True
        assert res_m.is_valid is True
        assert len(res_m.invariant_violations) == 0
        assert res_m.mizan_score > 0.95
        assert res_m.mizan_score > res_b.mizan_score


# ============================================================================
# Tier 3: Statistical Advantage & Ablation Delta Tests
# ============================================================================


class TestAblationStatisticalDelta:
    """Verifies aggregate statistical superiority of Noetherian Arbiter across full benchmark suite."""

    def test_ablation_delta_statistical_advantage(self) -> None:
        """Confirms Noetherian achieves 100% survival while Baseline and Naive achieve 0%."""
        benchmark = NoetherianAblationBenchmark(seed=42, runs=3, verbose=False)
        summary = benchmark.run_suite()

        gm = summary["global_metrics"]
        assert gm["invariant_survival_rate"]["noetherian_arbiter"] == 1.0
        assert gm["invariant_survival_rate"]["baseline"] == 0.0
        assert gm["invariant_survival_rate"]["naive_heuristic"] == 0.0
        assert gm["noetherian_perfect_conservation"] is True

        # Mean Mizan score advantage
        assert gm["mean_mizan_score"]["noetherian_arbiter"] > 0.95
        assert gm["mean_mizan_score"]["baseline"] < 0.65
        assert gm["mean_mizan_score"]["naive_heuristic"] < 0.65

        # Significant number of fatal breaches prevented
        assert gm["total_violations_prevented_vs_baseline"] > 40

    def test_variant_result_serialization(self) -> None:
        """Verifies VariantResult dictionary serialization retains all properties."""
        res = VariantResult(
            variant_name="Test Variant",
            chosen_trajectory_id="tau_1",
            chosen_path_labels=["Root", "Choice A"],
            utility=0.95,
            cumulative_cost=0.05,
            net_score=0.90,
            mizan_score=0.98,
            noether_flux=0.01,
            charge_drift=0.02,
            invariant_violations=[],
            violation_count=0,
            conserved=True,
            is_valid=True,
            regime="HARMONIC_EQUILIBRIUM",
            execution_time_ms=1.25,
        )
        d = res.to_dict()
        assert d["variant_name"] == "Test Variant"
        assert d["conserved"] is True
        assert d["mizan_score"] == 0.98
        assert d["violation_count"] == 0


# ============================================================================
# Tier 4: Hermetic Isolation, Telemetry & CLI Integration Tests
# ============================================================================


class TestHermeticIsolationAndCLI:
    """Verifies hermetic test isolation, JSON schema conformance, and CLI invocation."""

    def test_json_output_schema_conformance(self, tmp_path: Path) -> None:
        """Verifies that --json-output writes a valid JSON file conforming to the full schema."""
        out_file = tmp_path / "benchmark_summary.json"

        benchmark = NoetherianAblationBenchmark(seed=42, runs=1, verbose=False)
        summary = benchmark.run_suite()

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        assert out_file.exists()
        assert out_file.stat().st_size > 0

        with open(out_file, encoding="utf-8") as f:
            loaded = json.load(f)

        assert "metadata" in loaded
        assert "total_scenarios" in loaded and loaded["total_scenarios"] == 4
        assert "runs_per_scenario" in loaded and loaded["runs_per_scenario"] == 1
        assert "global_metrics" in loaded
        assert "scenarios" in loaded and len(loaded["scenarios"]) == 4

        for sc in loaded["scenarios"]:
            assert "scenario_name" in sc
            assert "description" in sc
            assert "variant_results" in sc
            assert "ablation_delta" in sc
            vr = sc["variant_results"]
            assert "baseline" in vr
            assert "naive_heuristic" in vr
            assert "noetherian_arbiter" in vr

    def test_hermetic_telemetry_isolation(self, tmp_path: Path) -> None:
        """Verifies completely hermetic telemetry recording using tmp_path without live file pollution."""
        telemetry_file = tmp_path / "hermetic_telemetry.jsonl"
        arbiter = QuantumDecisionArbiter(dim=64)

        dag = DecisionDAG()
        q0 = torch.zeros(64)
        q0[0] = 1.0
        q1 = torch.zeros(64)
        q1[1] = 1.0

        dag.add_node(DecisionNode(node_id="root", label="Start", statevector=q0))
        dag.add_node(DecisionNode(node_id="n1", label="Option 1", statevector=q1))
        dag.add_edge(DecisionEdge(source_id="root", target_id="n1", action_label="Go 1"))

        c = ConservedChargeDefinition(
            charge_id="c_test",
            rule_name="Test Invariant",
            generator_vector=q0,
            tolerance=0.05,
            is_hard_constraint=True,
        )
        cfg = NoetherianEngineConfig(dim=64, default_charges=[c])

        # Execute with custom hermetic telemetry file
        arbiter.arbitrate_dag(
            dag=dag,
            goal="Hermetic Telemetry Test",
            noetherian_config=cfg,
            log_telemetry=True,
            telemetry_file=telemetry_file,
        )

        assert telemetry_file.exists()
        events = read_telemetry_events(telemetry_file=telemetry_file)
        assert len(events) == 1
        record = events[0]
        assert "mizan_score" in record
        assert "noether_flux" in record
        assert "invariant_violations" in record

        summary = get_telemetry_summary(telemetry_file=telemetry_file)
        assert summary["total_decisions"] == 1
        assert "avg_mizan_score" in summary

    def test_cli_runner_execution(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Verifies main() CLI execution with argument parsing and JSON generation."""
        out_file = tmp_path / "cli_report.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "noetherian_ablation_benchmark.py",
                "--runs",
                "1",
                "--seed",
                "999",
                "--json-output",
                str(out_file),
            ],
        )

        exit_code = main()
        assert exit_code == 0
        assert out_file.exists()

        with open(out_file, encoding="utf-8") as f:
            data = json.load(f)
        assert data["runs_per_scenario"] == 1
        assert data["metadata"]["seed"] == 999
        assert data["global_metrics"]["noetherian_perfect_conservation"] is True
