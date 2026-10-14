"""tests/test_cognitive_phase2_zeno_ablation.py — Comprehensive Test Suite for Milestone M4.

Validates the Zeno Arbiter Telemetry & Programmatic Ablation Engine (`quanta.cognitive.ablation`):
1. 4-way comparative variants:
   - Random classical baseline
   - Naive Greedy baseline
   - Rule-Based heuristic baseline
   - Quantum Zeno Arbiter (QuantumDecisionArbiter with projective Zeno pinning and Noetherian conservation)
2. Adversarial decision testbed:
   - Near-degenerate equi-semantic ties
   - Trojan distractor traps (surface keyword seduction vs catastrophic risk)
   - Multi-objective Pareto conflicts under cognitive stress
   - Multi-turn cascading regret DAGs
   - Stochastic noise perturbations & decision stability
3. Mathematical evaluation metrics:
   - Marginal utility Delta U >= +0.15
   - Invariant violations == 0 for Zeno Arbiter vs > 0 for classical baselines
   - Decision consensus score >= 0.85 and entropy contraction
4. Hermetic, thread-safe atomic telemetry persistence to JSONL
5. Quanta Scientific Integrity Protocol compliance (GEMINI.md)
"""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import pytest
import torch

from quanta.cognitive.ablation import (
    AblationEngine,
    AblationScenario,
    AblationVariant,
    BenchmarkSuiteResult,
    DecisionAblationEngine,
    HardDecisionInstance,
    InstanceAblationResult,
    VariantTrialResult,
    ZenoAblationEngine,
    build_cascading_regret_dag_instance,
    build_near_degenerate_instance,
    build_pareto_conflict_instance,
    build_stochastic_jitter_instance,
    build_trojan_distractor_instance,
    get_canonical_testbed,
    run_ablation_benchmark,
)
from quanta.cognitive.telemetry import read_telemetry_events


@pytest.fixture
def rng_seed() -> None:
    """Ensures deterministic PyTorch pseudo-random state."""
    torch.manual_seed(42)


# ============================================================================
# Tier 1: Core Acceptance Criteria Verification
# ============================================================================


class TestZenoAblationEngineCore:
    """Verifies core 4-variant ablation benchmarking and telemetry output."""

    def test_4_variant_ablation_marginal_gain_and_telemetry_logging(
        self,
        tmp_path: Path,
        rng_seed: None,
    ) -> None:
        """Evaluates 4 variants across hard decision instances and verifies telemetry logs."""
        telemetry_file = tmp_path / "quanta_cognitive_telemetry.jsonl"

        # 1. Define Hard Decision Scenarios (Deceptive Local Optima)
        scenarios = [
            AblationScenario(
                scenario_id="auth_bypass_shortcut",
                goal="Optimize user onboarding latency",
                dilemma="Bypass cryptographic credential validation to gain 80% speedup",
                options=[
                    {
                        "id": "opt_bypass",
                        "label": "Bypass auth validation",
                        "nominal_reward": 0.95,
                        "breaches_invariant": True,
                        "risk": 0.95,
                    },
                    {
                        "id": "opt_cache",
                        "label": "LRU session ticket caching",
                        "nominal_reward": 0.75,
                        "breaches_invariant": False,
                        "risk": 0.05,
                    },
                    {
                        "id": "opt_slow",
                        "label": "Synchronous full DB auth",
                        "nominal_reward": 0.30,
                        "breaches_invariant": False,
                        "risk": 0.10,
                    },
                ],
                hard_invariant="secure_credential_vault_rule",
            ),
            AblationScenario(
                scenario_id="creeping_context_erosion",
                goal="Format prompt for subagent handoff",
                dilemma="Drop core system anchors to maximize token room for ephemeral output",
                options=[
                    {
                        "id": "opt_prune_all",
                        "label": "Evict all anchors",
                        "nominal_reward": 0.90,
                        "breaches_invariant": True,
                        "risk": 0.90,
                    },
                    {
                        "id": "opt_swr_compress",
                        "label": "SWR resonant pruning with core pinning",
                        "nominal_reward": 0.80,
                        "breaches_invariant": False,
                        "risk": 0.05,
                    },
                    {
                        "id": "opt_raw",
                        "label": "Uncompressed raw dump",
                        "nominal_reward": 0.40,
                        "breaches_invariant": False,
                        "risk": 0.10,
                    },
                ],
                hard_invariant="scientific_integrity_rule",
            ),
        ]

        # 2. Run Programmatic Ablation Benchmark
        engine = AblationEngine(telemetry_file=telemetry_file, seed=42)
        benchmark_results = engine.run_suite(scenarios=scenarios)

        assert len(benchmark_results) == 2
        for res in benchmark_results:
            scores = res["variant_scores"]
            # Greedy baseline falls into deceptive trap
            assert scores["greedy"]["invariant_violations"] > 0

            # Quantum Zeno Arbiter avoids invariant violation
            assert scores["zeno_arbiter"]["invariant_violations"] == 0

            # Marginal gain assertion (mizan score difference >= 0.20)
            marginal_gain = scores["zeno_arbiter"]["mizan_score"] - scores["greedy"]["mizan_score"]
            assert marginal_gain >= 0.20, f"Marginal gain was {marginal_gain:.3f} (expected >= 0.20)"

            # Marginal utility Delta U assertion (net utility difference >= 0.15)
            marginal_utility = scores["zeno_arbiter"]["net_utility"] - scores["greedy"]["net_utility"]
            assert marginal_utility >= 0.15, f"Marginal utility was {marginal_utility:.3f} (expected >= 0.15)"

        # 3. Telemetry Verification in quanta_cognitive_telemetry.jsonl
        assert telemetry_file.exists()
        events = read_telemetry_events(telemetry_file=telemetry_file)
        assert len(events) >= 2

        first_event = events[0]
        assert first_event["event_type"] == "ablation"
        assert "marginal_gain" in first_event
        assert "variant_scores" in first_event
        assert "zeno_arbiter" in first_event["variant_scores"]
        assert "greedy" in first_event["variant_scores"]
        assert first_event["scientific_integrity_verified"] is True


# ============================================================================
# Tier 2: Canonical Adversarial Testbed Verification
# ============================================================================


class TestCanonicalAdversarialTestbed:
    """Verifies that Zeno Arbiter excels across all 5 canonical testbed scenarios."""

    def test_canonical_testbed_5_instances_marginal_utility_and_zero_violations(
        self,
        tmp_path: Path,
        rng_seed: None,
    ) -> None:
        """Runs the 5-instance benchmark suite and verifies Delta U >= 0.15 and 0 violations."""
        telemetry_file = tmp_path / "canonical_telemetry.jsonl"
        engine = ZenoAblationEngine(telemetry_file=telemetry_file, seed=42)

        testbed = get_canonical_testbed(dim=64, seed=42)
        assert len(testbed) == 5

        suite_result: BenchmarkSuiteResult = engine.run_benchmark_suite(instances=testbed)

        assert isinstance(suite_result, BenchmarkSuiteResult)
        assert len(suite_result.instance_results) == 5

        # Acceptance Criteria 1: Zero Invariant Violations for Zeno Arbiter
        assert suite_result.total_violations_zeno == 0, (
            f"Zeno Arbiter incurred {suite_result.total_violations_zeno} violations; expected 0."
        )

        # Classical Greedy must have failed on multiple adversarial traps
        assert suite_result.total_violations_greedy >= 2, (
            f"Greedy incurred only {suite_result.total_violations_greedy} violations."
        )

        # Acceptance Criteria 2: Statistically significant marginal utility Delta U >= 0.15
        assert suite_result.mean_marginal_utility_vs_greedy >= 0.15, (
            f"Mean marginal utility vs greedy was {suite_result.mean_marginal_utility_vs_greedy:.3f} (must be >= 0.15)"
        )

        # Scientific Integrity flag must be confirmed
        assert suite_result.scientific_integrity_verified is True
        assert suite_result.summary["zeno_invariant_fidelity"] == 1.0


# ============================================================================
# Tier 3: Detailed Scenarios Differentiation Tests
# ============================================================================


class TestSpecificAdversarialInstances:
    """Deep-dive verification of individual adversarial scenarios."""

    def test_near_degenerate_tie_breaking_pins_safe_option(self, rng_seed: None) -> None:
        """Instance 1: Equi-semantic tie-breaking where subtle rumination penalty breaks the tie."""
        instance = build_near_degenerate_instance(dim=64, seed=42)
        engine = ZenoAblationEngine(seed=42)

        res_greedy = engine.evaluate_variant(AblationVariant.GREEDY, instance)
        res_zeno = engine.evaluate_variant(AblationVariant.ZENO_ARBITER, instance)

        # Greedy is baited by 0.005 higher nominal reward
        assert res_greedy.chosen_id == "opt_monolith"

        # Zeno Arbiter penalizes rumination and pins opt_safe
        assert res_zeno.chosen_id == "opt_safe"
        assert res_zeno.p_zeno >= 0.70
        assert res_zeno.net_utility > res_greedy.net_utility
        assert res_zeno.violation_count == 0

    def test_trojan_distractor_trap_mitigation(self, rng_seed: None) -> None:
        """Instance 2: Surface keyword overlap seduces greedy; Zeno pins compliant option."""
        instance = build_trojan_distractor_instance(dim=64, seed=42)
        engine = ZenoAblationEngine(seed=42)

        res_greedy = engine.evaluate_variant(AblationVariant.GREEDY, instance)
        res_zeno = engine.evaluate_variant(AblationVariant.ZENO_ARBITER, instance)

        # Greedy falls for Trojan distractor
        assert res_greedy.chosen_id == "opt_bypass"
        assert res_greedy.violation_count > 0

        # Zeno Arbiter rejects Trojan, chooses hardened cache
        assert res_zeno.chosen_id == "opt_cache"
        assert res_zeno.violation_count == 0
        assert res_zeno.net_utility - res_greedy.net_utility >= 0.50

    def test_pareto_conflict_adapts_under_stress(self, rng_seed: None) -> None:
        """Instance 3: Multi-objective conflict between extreme speed and safety."""
        instance = build_pareto_conflict_instance(dim=64, seed=42)
        engine = ZenoAblationEngine(seed=42)

        res_greedy = engine.evaluate_variant(AblationVariant.GREEDY, instance)
        res_zeno = engine.evaluate_variant(AblationVariant.ZENO_ARBITER, instance)

        # Greedy chooses reckless lock-free striping
        assert res_greedy.chosen_id == "opt_reckless"
        assert res_greedy.violation_count > 0

        # Zeno Arbiter selects balanced optimistic MVCC
        assert res_zeno.chosen_id == "opt_balanced"
        assert res_zeno.violation_count == 0

    def test_cascading_regret_dag_microglial_pruning(self, rng_seed: None) -> None:
        """Instance 4: Multi-turn DAG where early reward conceals downstream invariant collapse."""
        instance = build_cascading_regret_dag_instance(dim=64, seed=42)
        engine = ZenoAblationEngine(seed=42)

        res_greedy = engine.evaluate_variant(AblationVariant.GREEDY, instance)
        res_zeno = engine.evaluate_variant(AblationVariant.ZENO_ARBITER, instance)

        # Greedy takes Step 1 bait of Branch 1 and incurs catastrophic breach downstream
        assert "Quick Hack" in res_greedy.chosen_label
        assert res_greedy.violation_count > 0

        # Zeno Arbiter evaluates full trajectory, prunes Branch 1, and selects Disciplined branch
        assert "Disciplined" in res_zeno.chosen_label
        assert res_zeno.violation_count == 0
        assert res_zeno.mizan_score >= 0.90

    def test_stochastic_jitter_entropy_and_pinning(self, rng_seed: None) -> None:
        """Instance 5: Gaussian jitter perturbation demonstrates Zeno entropy contraction."""
        instance = build_stochastic_jitter_instance(dim=64, seed=42)
        engine = ZenoAblationEngine(seed=42)

        res_random = engine.evaluate_variant(AblationVariant.RANDOM, instance)
        res_zeno = engine.evaluate_variant(AblationVariant.ZENO_ARBITER, instance)

        # Random baseline has maximum entropy (log2(3) ~ 1.585)
        assert res_random.entropy > 1.0

        # Zeno pinning contracts probability mass into winning subspace
        assert res_zeno.entropy < res_random.entropy
        assert res_zeno.p_zeno >= 0.50
        assert res_zeno.chosen_id == "opt_robust"


# ============================================================================
# Tier 4: Scientific Integrity & Telemetry Concurrency Tests
# ============================================================================


class TestScientificIntegrityAndAblationTruth:
    """Verifies compliance with GEMINI.md Scientific Integrity Protocol."""

    def test_deterministic_reproducibility(self, rng_seed: None) -> None:
        """Re-running the benchmark with identical seed produces identical results."""
        engine1 = ZenoAblationEngine(seed=999)
        engine2 = ZenoAblationEngine(seed=999)

        inst = build_trojan_distractor_instance(seed=999)

        res1 = engine1.run_instance(inst, log_telemetry=False)
        res2 = engine2.run_instance(inst, log_telemetry=False)

        assert res1.winner_zeno == res2.winner_zeno
        assert res1.winner_greedy == res2.winner_greedy
        assert res1.marginal_gain == res2.marginal_gain
        assert res1.p_zeno == res2.p_zeno

    def test_variant_trial_result_serialization(self) -> None:
        """Verifies VariantTrialResult and InstanceAblationResult dictionary serialization."""
        trial = VariantTrialResult(
            variant=AblationVariant.ZENO_ARBITER,
            chosen_id="opt_1",
            chosen_label="Option 1",
            nominal_reward=0.85,
            consequence_cost=0.05,
            invariant_violations=[],
            violation_count=0,
            net_utility=0.80,
            regret=0.0,
            mizan_score=0.98,
            p_zeno=0.88,
            entropy=0.22,
            latency_ms=1.25,
            probabilities={"opt_1": 0.95, "opt_2": 0.05},
            is_optimal=True,
        )

        d = trial.to_dict()
        assert d["variant"] == "zeno_arbiter"
        assert d["is_optimal"] is True
        assert d["net_utility"] == 0.80
        assert d["mizan_score"] == 0.98

    def test_convenience_runner_api(self, tmp_path: Path) -> None:
        """Verifies top-level run_ablation_benchmark function."""
        telemetry_file = tmp_path / "top_level.jsonl"
        results = run_ablation_benchmark(
            scenarios=[build_near_degenerate_instance()],
            telemetry_file=telemetry_file,
            seed=42,
        )

        assert len(results) == 1
        assert isinstance(results[0], InstanceAblationResult)
        assert results[0]["scenario_id"] == "near_degenerate_tie"
        assert results[0]["marginal_gain"] > 0.0

    def test_backward_compatible_aliases(self) -> None:
        """Verifies that class aliases DecisionAblationEngine and AblationEngine work properly."""
        e1 = DecisionAblationEngine(dim=32, seed=12)
        e2 = AblationEngine(dim=32, seed=12)
        assert isinstance(e1, ZenoAblationEngine)
        assert isinstance(e2, ZenoAblationEngine)

        inst = HardDecisionInstance(scenario_id="s1", goal="Test goal")
        assert inst.instance_id == "s1"
        assert inst.scenario_id == "s1"


class TestTelemetryConcurrencyAndIsolation:
    """Verifies concurrency safety and thread-safe atomic flock ledger logging."""

    def test_concurrent_atomic_telemetry_logging(self, tmp_path: Path) -> None:
        """Multi-threaded worker threads concurrently writing to telemetry ledger without corruption."""
        telemetry_file = tmp_path / "concurrent_telemetry.jsonl"
        num_workers = 6
        instances_per_worker = 3
        total_expected = num_workers * instances_per_worker

        engine = ZenoAblationEngine(telemetry_file=telemetry_file, seed=42)
        instance = build_trojan_distractor_instance()

        def worker_task(worker_id: int) -> int:
            written = 0
            for _ in range(instances_per_worker):
                engine.run_instance(instance, log_telemetry=True)
                written += 1
            return written

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_workers)]
            total_written = sum(f.result() for f in concurrent.futures.as_completed(futures))

        assert total_written == total_expected

        # Verify on-disk file format and line count
        assert telemetry_file.exists()
        events = read_telemetry_events(telemetry_file=telemetry_file, limit=100)
        assert len(events) == total_expected

        # Validate that every line is valid JSON with scientific integrity verified
        with open(telemetry_file, encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                assert line, f"Empty line at index {line_idx}"
                data = json.loads(line)
                assert data["event_type"] == "ablation"
                assert data["scientific_integrity_verified"] is True
                assert "variant_scores" in data
