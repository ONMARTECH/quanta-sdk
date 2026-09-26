"""tests/test_challenger_m4_empirical.py — Challenger M4 Empirical Verification Suite.

Adversarially benchmarks and independently verifies Milestone M4 Preemption SLA:
1. Challenge 1: Preemption Latency Under Sleep Load:
   - Measure physical HID preemption interrupt latency during active REM thought vector perturbation.
   - Measure preemption latency during active SWS synaptic downscaling.
   - Verify latency is strictly < 20ms across 100 consecutive preemptions in each regime.
   - Verify zero state corruption, zero lock contention, and clean phase transition to PREEMPTED.
2. Challenge 2: Full Regression Invariants Verification.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

from quanta.cognitive.consolidation import SubconsciousConsolidator
from quanta.cognitive.daemon import (
    SleepPhase,
    SubconsciousDaemon,
)
from quanta.cognitive.memory import CognitiveMemoryManager
from quanta.cognitive.mind_wander import cosine_similarity, perturb_thought_vector
from quanta.cognitive.tom_analyzer import DreamSeed


class TestChallengerM4PreemptionLatency:
    """Empirical adversarial benchmark for preemption latency under active sleep load."""

    def test_preemption_latency_during_active_rem_thought_perturbation_100_runs(
        self, tmp_path: Path
    ) -> None:
        """Measures preemption latency while the daemon is actively perturbing thought vectors in REM.

        Verifies that across 100 consecutive preemptions, latency is strictly < 20.0ms.
        """
        state_file = tmp_path / "rem_preempt_state.json"
        pid_file = tmp_path / "rem_preempt.pid"
        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)

        seed = DreamSeed(
            topic="rem_perturbation_stress",
            speculative_question="Does preemption reliably halt active thought perturbation?",
            urgency=1.0,
            context_keys=["challenger_m4"],
        )

        # Realistic 200us cognitive micro-deliberation per turn
        orig_simulate_step = daemon.wander_engine.simulator.simulate_step

        def delayed_simulate_step(*args: Any, **kwargs: Any) -> Any:
            time.sleep(0.0002)
            return orig_simulate_step(*args, **kwargs)

        daemon.wander_engine.simulator.simulate_step = delayed_simulate_step  # type: ignore[method-assign]

        latencies_ms: list[float] = []
        run_count = 100

        for i in range(run_count):
            daemon._preemption_event.clear()
            daemon._stop_event.clear()
            daemon.current_sleep_phase = SleepPhase.WAKE

            result_holder: list[Any] = []

            def rem_worker(res_target: list[Any] = result_holder) -> None:
                res = daemon.run_rem_cycle(seed)
                res_target.append(res)

            worker_thread = threading.Thread(target=rem_worker, name=f"rem-worker-{i}")
            worker_thread.start()

            # Wait slightly to guarantee thread has entered REM turn loop
            time.sleep(0.0005)

            t0 = time.perf_counter()
            daemon.interrupt_immediate()
            worker_thread.join(timeout=2.0)
            t1 = time.perf_counter()

            latency = (t1 - t0) * 1000.0
            latencies_ms.append(latency)

            assert not worker_thread.is_alive(), f"Iteration {i}: worker thread did not terminate"
            assert len(result_holder) == 1, f"Iteration {i}: worker did not return result"
            assert result_holder[0] is None, (
                f"Iteration {i}: cycle was not preempted (returned result)"
            )
            assert daemon.current_sleep_phase == SleepPhase.PREEMPTED, (
                f"Iteration {i}: phase is {daemon.current_sleep_phase}, expected PREEMPTED"
            )
            assert latency < 20.0, (
                f"Iteration {i}: preemption latency {latency:.3f}ms exceeded strict 20ms SLA"
            )

        avg_lat = statistics.mean(latencies_ms)
        max_lat = max(latencies_ms)
        min_lat = min(latencies_ms)
        p50 = statistics.median(latencies_ms)
        sorted_lat = sorted(latencies_ms)
        p95 = sorted_lat[int(0.95 * run_count)]
        p99 = sorted_lat[int(0.99 * run_count)]

        print(
            f"\n[Challenge 1A: REM Thought Perturbation Preemption ({run_count} runs)]\n"
            f"  Avg: {avg_lat:.4f} ms (SLA < 20.0 ms)\n"
            f"  Max: {max_lat:.4f} ms (SLA < 20.0 ms)\n"
            f"  Min: {min_lat:.4f} ms\n"
            f"  P50: {p50:.4f} ms | P95: {p95:.4f} ms | P99: {p99:.4f} ms\n"
        )

        assert len(latencies_ms) == run_count
        assert max_lat < 20.0, f"Worst-case latency {max_lat:.2f}ms >= 20.0ms"
        assert avg_lat < 10.0, f"Average latency {avg_lat:.2f}ms unexpectedly high"

    def test_preemption_latency_during_active_sws_synaptic_downscaling_100_runs(
        self, tmp_path: Path
    ) -> None:
        """Measures preemption latency while the daemon is actively downscaling synapses in SWS.

        Verifies that across 100 consecutive preemptions, latency is strictly < 20.0ms.
        """
        state_file = tmp_path / "sws_preempt_state.json"
        pid_file = tmp_path / "sws_preempt.pid"
        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)

        # Pre-seed cognitive memory with 50 engrams of varying salience
        for j in range(50):
            daemon.consolidator.memory_manager.record_decision(
                key=f"sws_engram_{j}",
                content=f"Synthetic task memory decision {j} for testing downscaling.",
                salience=0.60 + (j % 10) * 0.15,
                category="transient_task",
                overwrite=True,
            )
        daemon.consolidator.save_state_atomic()

        latencies_ms: list[float] = []
        run_count = 100

        for i in range(run_count):
            for j in range(50):
                daemon.consolidator.memory_manager.record_decision(
                    key=f"sws_engram_{j}",
                    content=f"Synthetic task memory decision {j} for testing downscaling.",
                    salience=0.60 + (j % 10) * 0.15,
                    category="transient_task",
                    overwrite=True,
                )

            daemon._preemption_event.clear()
            daemon._stop_event.clear()
            daemon.current_sleep_phase = SleepPhase.WAKE

            result_holder: list[Any] = []

            def sws_worker(res_target: list[Any] = result_holder) -> None:
                res = daemon.run_nrem_cycle()
                res_target.append(res)

            worker_thread = threading.Thread(target=sws_worker, name=f"sws-worker-{i}")
            worker_thread.start()

            # Microsecond stagger calibrated to strike at different active stages of downscaling
            time.sleep(0.00008 + (i % 6) * 0.00010)

            t0 = time.perf_counter()
            daemon.interrupt_immediate()
            worker_thread.join(timeout=2.0)
            t1 = time.perf_counter()

            latency = (t1 - t0) * 1000.0
            latencies_ms.append(latency)

            assert not worker_thread.is_alive(), f"Iteration {i}: worker thread did not terminate"
            assert len(result_holder) == 1, f"Iteration {i}: worker did not return result"
            assert result_holder[0] is None, (
                f"Iteration {i}: cycle was not preempted (returned result)"
            )
            assert daemon.current_sleep_phase == SleepPhase.PREEMPTED, (
                f"Iteration {i}: phase is {daemon.current_sleep_phase}, expected PREEMPTED"
            )
            assert latency < 20.0, (
                f"Iteration {i}: preemption latency {latency:.3f}ms exceeded strict 20ms SLA"
            )

        avg_lat = statistics.mean(latencies_ms)
        max_lat = max(latencies_ms)
        min_lat = min(latencies_ms)
        p50 = statistics.median(latencies_ms)
        sorted_lat = sorted(latencies_ms)
        p95 = sorted_lat[int(0.95 * run_count)]
        p99 = sorted_lat[int(0.99 * run_count)]

        print(
            f"\n[Challenge 1B: SWS Synaptic Downscaling Preemption ({run_count} runs)]\n"
            f"  Avg: {avg_lat:.4f} ms (SLA < 20.0 ms)\n"
            f"  Max: {max_lat:.4f} ms (SLA < 20.0 ms)\n"
            f"  Min: {min_lat:.4f} ms\n"
            f"  P50: {p50:.4f} ms | P95: {p95:.4f} ms | P99: {p99:.4f} ms\n"
        )

        assert len(latencies_ms) == run_count
        assert max_lat < 20.0, f"Worst-case latency {max_lat:.2f}ms >= 20.0ms"
        assert avg_lat < 10.0, f"Average latency {avg_lat:.2f}ms unexpectedly high"

    def test_physical_darwin_hid_preemption_latency_100_runs(
        self, tmp_path: Path
    ) -> None:
        """Measures physical Darwin HID input preemption latency across 100 consecutive cycles.

        Simulates active HID interrupts where Quartz/CoreGraphics reports user input
        (idle drops from >= 1.0s quiescent down to 0.0s) during ongoing sleep cycles.
        Verifies that across 100 consecutive physical HID interruptions, latency is < 20.0ms.
        """
        state_file = tmp_path / "hid_preempt_state.json"
        pid_file = tmp_path / "hid_preempt.pid"
        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)
        daemon._is_running = True

        seed = DreamSeed(
            topic="darwin_hid_stress",
            speculative_question="Does Darwin Quartz HID event abort sleep in <20ms?",
            urgency=1.0,
            context_keys=["darwin_hid"],
        )

        latencies_ms: list[float] = []
        run_count = 100

        for i in range(run_count):
            daemon._preemption_event.clear()
            daemon._stop_event.clear()
            daemon.current_sleep_phase = SleepPhase.WAKE

            call_count = [0]
            t0_holder = [0.0]

            def dynamic_hid_idle(
                cnt: list[int] = call_count,
                t_holder: list[float] = t0_holder,
            ) -> float:
                cnt[0] += 1
                if cnt[0] == 1:
                    return 5.0  # Initial quiescent state (>= 1.0s)
                # Physical user HID event arrives during active cycle (mouse/key)
                if t_holder[0] == 0.0:
                    t_holder[0] = time.perf_counter()
                return 0.0

            with patch(
                "quanta.cognitive.daemon.get_user_idle_seconds", side_effect=dynamic_hid_idle
            ):
                res = (
                    daemon.run_rem_cycle(seed)
                    if i % 2 == 0
                    else daemon.run_nrem_cycle()
                )
                t1 = time.perf_counter()

                latency = (
                    (t1 - t0_holder[0]) * 1000.0 if t0_holder[0] > 0.0 else 0.0
                )
                latencies_ms.append(latency)

                assert res is None, f"Iteration {i}: cycle was not preempted by HID"
                assert daemon.current_sleep_phase == SleepPhase.PREEMPTED, (
                    f"Iteration {i}: phase is {daemon.current_sleep_phase}, expected PREEMPTED"
                )
                assert latency < 20.0, (
                    f"Iteration {i}: HID preemption latency {latency:.3f}ms exceeded 20ms SLA"
                )

        avg_lat = statistics.mean(latencies_ms)
        max_lat = max(latencies_ms)
        min_lat = min(latencies_ms)
        p50 = statistics.median(latencies_ms)
        sorted_lat = sorted(latencies_ms)
        p95 = sorted_lat[int(0.95 * run_count)]
        p99 = sorted_lat[int(0.99 * run_count)]

        print(
            f"\n[Challenge 1C: Physical Darwin HID Preemption ({run_count} runs)]\n"
            f"  Avg: {avg_lat:.4f} ms (SLA < 20.0 ms)\n"
            f"  Max: {max_lat:.4f} ms (SLA < 20.0 ms)\n"
            f"  Min: {min_lat:.4f} ms\n"
            f"  P50: {p50:.4f} ms | P95: {p95:.4f} ms | P99: {p99:.4f} ms\n"
        )

        assert len(latencies_ms) == run_count
        assert max_lat < 20.0, f"Worst-case HID latency {max_lat:.2f}ms >= 20.0ms"
        assert avg_lat < 10.0, f"Average HID latency {avg_lat:.2f}ms unexpectedly high"

    def test_state_integrity_and_recovery_after_preemption_burst(self, tmp_path: Path) -> None:
        """Verifies that 50 rapid preemptions leave zero corrupt state and daemon recovers."""
        state_file = tmp_path / "burst_state.json"
        pid_file = tmp_path / "burst.pid"
        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)

        seed = DreamSeed(
            topic="burst_recovery",
            speculative_question="Does state survive rapid preemption interrupts?",
            urgency=1.0,
            context_keys=["quanta"],
        )

        # Prepopulate state
        daemon.consolidator.memory_manager.record_decision(
            key="core_invariant_1",
            content="Core architectural decision that must survive burst preemption.",
            salience=3.0,
            category="core_decision",
            overwrite=True,
        )
        daemon.consolidator.save_state_atomic()

        # Fire 50 preemptions
        for _ in range(50):
            daemon.interrupt_immediate()
            res = daemon.run_single_cycle(seed)
            assert res is None
            assert daemon.current_sleep_phase == SleepPhase.PREEMPTED
            daemon._preemption_event.clear()

        # Check file validity
        with open(state_file, encoding="utf-8") as f:
            data = json.load(f)
            assert "engrams" in data
            assert len(data["engrams"]) >= 1

        # Verify that subsequent sleep cycle completes normally
        daemon._preemption_event.clear()
        normal_res = daemon.run_nrem_cycle()
        assert normal_res is not None
        assert normal_res.success is True
        assert daemon.current_sleep_phase == SleepPhase.WAKE


class TestChallengerM4EmpiricalRemediation:
    """Empirical verification for Challenger 1 & 2 remediation requirements."""

    def test_perturb_thought_vector_dimension_scaled_noise_20000_runs(self) -> None:
        """Verifies dimension-scaled orthogonal noise guarantees cosine similarity in [0.80, 0.99].

        Tests 1,000 trials across all d in {8, 16, 32, 64, 128} and sigma in {0.05, 0.10, 0.15, 0.20}
        (20,000 total vectors). Demonstrates:
        - 100% L2 unit norm preservation (1.000000 +/- 1e-6).
        - 100% cosine similarity strictly within [0.80, 0.99] across all dimensions.
        - Zero NaN, Inf, or ZeroDivisionError.
        """
        rng = random.Random(1337)
        total_vectors = 0
        norm_passes = 0
        cosine_passes = 0

        for d in (8, 16, 32, 64, 128):
            base_v = [1.0 / math.sqrt(d)] * d
            for sig in (0.05, 0.10, 0.15, 0.20):
                for _ in range(1000):
                    total_vectors += 1
                    v_prime, p_norm = perturb_thought_vector(base_v, sigma=sig, rng=rng)

                    # Norm check
                    norm = math.sqrt(sum(x * x for x in v_prime))
                    if math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
                        norm_passes += 1

                    # Numerical sanity
                    assert all(not math.isnan(x) and not math.isinf(x) for x in v_prime)
                    assert not math.isnan(p_norm) and not math.isinf(p_norm)
                    assert p_norm > 0.0

                    # Angular cosine similarity check
                    sim = cosine_similarity(base_v, v_prime)
                    if 0.80 <= sim <= 0.99:
                        cosine_passes += 1

        assert total_vectors == 20000
        assert norm_passes == 20000, f"Norm pass rate: {norm_passes}/20000"
        assert cosine_passes == 20000, f"Cosine similarity pass rate: {cosine_passes}/20000"

    def test_50_alternating_sleep_cycles_memory_buffer_capacity_bounded(
        self, tmp_path: Path
    ) -> None:
        """Verifies buffer capacity remains strictly bounded over 50 SWS-REM cycles."""
        state_file = tmp_path / "capacity_bounded_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file, capacity=8)

        # Pre-seed two permanent system core anchors
        consolidator.memory_manager.record_decision(
            key="sys_core_invariant_1",
            content="Permanent architectural invariant 1",
            salience=3.0,
            category="system_core",
            is_system_core=True,
        )
        consolidator.memory_manager.record_decision(
            key="sys_core_invariant_2",
            content="Permanent architectural invariant 2",
            salience=3.0,
            category="system_core",
            is_system_core=True,
        )

        initial_cap = consolidator.memory_manager.buffer.capacity
        assert initial_cap == 8

        # Run 50 alternating SWS-REM cycles
        for cycle in range(50):
            # SWS: Synaptic downscaling and pruning
            consolidator.apply_wake_sleep_plasticity(SleepPhase.NREM_SWS)

            # REM: Dialectic dreaming generates new dream insight
            consolidator.memory_manager.record_decision(
                key=f"hybrid_dream_insight_{cycle}",
                content=f"Emergent bisociative insight generated in REM cycle {cycle}.",
                salience=2.5,
                category="hybrid_dream",
            )
            consolidator.apply_wake_sleep_plasticity(SleepPhase.REM_DREAM)

        final_cap = consolidator.memory_manager.buffer.capacity
        final_size = len(consolidator.memory_manager.buffer.buffer)

        # Capacity must remain strictly bounded (never inflated to 50+)
        assert final_cap <= 8, f"Buffer capacity inflated to {final_cap} (expected <= 8)"
        assert final_size <= 8, f"Buffer size grew to {final_size} (expected <= 8)"

        # Both permanent system core invariants must survive all 50 cycles
        keys = [e.get("metadata", {}).get("key") for e in consolidator.memory_manager.buffer.buffer]
        assert "sys_core_invariant_1" in keys, "Permanent core invariant 1 was evicted!"
        assert "sys_core_invariant_2" in keys, "Permanent core invariant 2 was evicted!"

    def test_untagged_noise_decays_and_pruned_in_sws_without_stalling(self) -> None:
        """Verifies untagged non-core chatter decays past 0.20 and prunes without stalling."""
        mgr = CognitiveMemoryManager(capacity=16, auto_prune=False)
        mgr.record_decision(
            key="untagged_task_chatter",
            content="General ephemeral reasoning step without explicit noise category.",
            salience=0.80,
            category="general",
        )

        pruned_cycle = None
        for cycle in range(15):
            mgr.apply_wake_sleep_plasticity(SleepPhase.NREM_SWS, shy_factor=0.90)
            engram = next(
                (e for e in mgr.buffer.buffer if e.get("metadata", {}).get("key") == "untagged_task_chatter"),
                None,
            )
            if engram is None:
                pruned_cycle = cycle
                break

        assert pruned_cycle is not None, "Untagged noise was never pruned (stalled in memory)!"
        assert pruned_cycle <= 10, f"Pruning took {pruned_cycle} cycles, expected <= 10"
