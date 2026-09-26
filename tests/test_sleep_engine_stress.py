"""tests/test_sleep_engine_stress.py — Adversarial Stress Test Suite.

Adversarially challenges and benchmarks:
1. High-frequency preemption triggering (<20ms latency guarantee under stress).
2. Alternating pacing transitions across 50+ consecutive cycles (ensuring no phase drift or deadlocks).
3. Thread safety of sleep state mutations and telemetry logging under concurrency.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from quanta.cognitive.consolidation import SubconsciousConsolidator
from quanta.cognitive.daemon import (
    NREMExecutionResult,
    REMExecutionResult,
    SleepPhase,
    SubconsciousDaemon,
)
from quanta.cognitive.telemetry import read_telemetry_events, record_sleep_cycle_telemetry
from quanta.cognitive.tom_analyzer import DreamSeed


class TestHighFrequencyPreemptionStress:
    """Stress-tests preemption latency and reliability under adversarial load."""

    def test_preemption_latency_distribution_under_heavy_buffer(self, tmp_path: Path) -> None:
        """Benchmarks preemption latency across 100 trials with a populated cognitive state."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        # Populate buffer with 50 diverse engrams
        for i in range(50):
            consolidator.memory_manager.record_decision(
                key=f"stress_engram_{i}",
                content=f"Subconscious architectural decision {i} with detailed context and parameters.",
                salience=0.5 + (i % 20) * 0.1,
                category="decision" if i % 2 == 0 else "transient",
            )
        consolidator.save_state_atomic()

        daemon = SubconsciousDaemon(
            state_file=state_file,
            pid_file=pid_file,
            use_agy_cli=False,
        )

        latencies_ms: list[float] = []

        # Run 100 high-frequency preemption checks
        for _ in range(100):
            daemon._preemption_event.set()
            t0 = time.perf_counter()
            res = daemon.run_nrem_cycle()
            dur_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dur_ms)

            assert res is None, "Cycle must be aborted when preemption event is set"
            assert daemon.current_sleep_phase == SleepPhase.PREEMPTED

            daemon.current_sleep_phase = SleepPhase.WAKE
            daemon._preemption_event.clear()

        max_latency = max(latencies_ms)
        avg_latency = sum(latencies_ms) / len(latencies_ms)
        p99_latency = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]

        print("\n[Preemption Latency Benchmark - 100 Trials]")
        print(f"  Average: {avg_latency:.4f} ms")
        print(f"  P99:     {p99_latency:.4f} ms")
        print(f"  Max:     {max_latency:.4f} ms")

        # Formal check: preemption must complete strictly within 20ms
        assert max_latency < 20.0, f"Max preemption latency {max_latency:.2f}ms exceeded 20ms limit"
        assert avg_latency < 1.0, f"Average preemption latency {avg_latency:.2f}ms unexpectedly high"

    def test_nrem_individual_phase_execution_ceilings(self, tmp_path: Path) -> None:
        """Verifies individual NREM phases execute within bounded latency budgets (<20ms)."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        # Fill to capacity (64 engrams)
        for i in range(64):
            consolidator.memory_manager.record_decision(
                key=f"full_cap_{i}",
                content=f"Engram capacity filling item {i}",
                salience=1.0 + (i % 10) * 0.1,
                category="transient" if i % 2 == 0 else "decision",
            )
        consolidator.save_state_atomic()

        # Measure Step 1: Glymphatic clearance
        t0 = time.perf_counter()
        clearance_res = consolidator.execute_glymphatic_clearance()
        t_clearance_ms = (time.perf_counter() - t0) * 1000.0
        assert "files_cleared" in clearance_res

        # Measure Step 2: Synaptic homeostasis
        t1 = time.perf_counter()
        shy_res = consolidator.execute_synaptic_homeostasis(shy_factor=0.90)
        t_shy_ms = (time.perf_counter() - t1) * 1000.0
        assert "downscaled_count" in shy_res

        # Measure Step 3: Hippocampal consolidation
        t2 = time.perf_counter()
        swr_count = consolidator.execute_hippocampal_neocortical_consolidation()
        t_swr_ms = (time.perf_counter() - t2) * 1000.0
        assert swr_count >= 0

        print("\n[NREM Component Latencies at Max Capacity (64 engrams)]")
        print(f"  1. Glymphatic clearance:   {t_clearance_ms:.2f} ms")
        print(f"  2. Synaptic Homeostasis:   {t_shy_ms:.2f} ms")
        print(f"  3. Hippocampal Replay:     {t_swr_ms:.2f} ms")

        # Each sub-phase must remain well-behaved to keep preemption checks frequent
        assert t_clearance_ms < 20.0, f"Clearance {t_clearance_ms:.2f}ms exceeded 20ms"
        assert t_shy_ms < 20.0, f"SHY downscaling {t_shy_ms:.2f}ms exceeded 20ms"
        assert t_swr_ms < 20.0, f"Hippocampal replay {t_swr_ms:.2f}ms exceeded 20ms"

    def test_asynchronous_preemption_interrupt_in_flight_nrem(self, tmp_path: Path) -> None:
        """Challenges preemption reflex while NREM cycle is actively running in a background thread."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(
            state_file=state_file,
            pid_file=pid_file,
            use_agy_cli=False,
        )

        for i in range(20):
            daemon.consolidator.memory_manager.record_decision(
                key=f"item_{i}", content=f"Decision {i}", salience=1.0, category="decision"
            )
        daemon.consolidator.save_state_atomic()

        interrupted_count = 0
        total_trials = 25

        for _ in range(total_trials):
            daemon._preemption_event.clear()
            daemon.current_sleep_phase = SleepPhase.WAKE

            result_holder: list[NREMExecutionResult | None] = []
            start_barrier = threading.Barrier(2)

            def make_worker(
                barrier: threading.Barrier, holder: list[NREMExecutionResult | None]
            ) -> Callable[[], None]:
                def worker() -> None:
                    barrier.wait()
                    res = daemon.run_nrem_cycle()
                    holder.append(res)
                return worker

            t = threading.Thread(target=make_worker(start_barrier, result_holder))
            t.start()

            start_barrier.wait()
            daemon.interrupt_immediate()
            t.join(timeout=2.0)

            assert not t.is_alive(), "Worker thread must not hang or deadlock"
            assert len(result_holder) == 1
            if result_holder[0] is None:
                interrupted_count += 1
                assert daemon.current_sleep_phase == SleepPhase.PREEMPTED

        print(f"\n[Asynchronous NREM Interruption] {interrupted_count}/{total_trials} cycles caught in-flight")

    def test_asynchronous_preemption_interrupt_in_flight_rem(self, tmp_path: Path) -> None:
        """Challenges preemption reflex while REM dreaming cycle is in-flight across dialectic turns."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(
            state_file=state_file,
            pid_file=pid_file,
            use_agy_cli=False,
        )

        seed = DreamSeed(
            topic="stress_preemption_topic",
            speculative_question="Can REM dreaming be interrupted sub-20ms in flight?",
            urgency=1.0,
            context_keys=["stress_test"],
        )

        preemption_latencies: list[float] = []

        for _ in range(20):
            daemon._preemption_event.clear()
            daemon.current_sleep_phase = SleepPhase.WAKE
            result_holder: list[REMExecutionResult | None] = []

            cycle_started = threading.Event()
            orig_simulate = daemon.wander_engine.simulator.simulate_step

            def hook_simulate(
                *args: Any,
                _started: threading.Event = cycle_started,
                _orig: Any = orig_simulate,
                **kwargs: Any,
            ) -> Any:
                _started.set()
                time.sleep(0.002)
                return _orig(*args, **kwargs)

            def make_rem_worker(
                holder: list[REMExecutionResult | None]
            ) -> Callable[[], None]:
                def worker() -> None:
                    res = daemon.run_rem_cycle(seed=seed)
                    holder.append(res)
                return worker

            with patch.object(daemon.wander_engine.simulator, "simulate_step", side_effect=hook_simulate):
                t = threading.Thread(target=make_rem_worker(result_holder))
                t.start()

                assert cycle_started.wait(timeout=2.0), "Dream cycle did not start in time"

                t_signal = time.perf_counter()
                daemon.interrupt_immediate()
                t.join(timeout=2.0)
                dur_interrupt_ms = (time.perf_counter() - t_signal) * 1000.0
                preemption_latencies.append(dur_interrupt_ms)

            assert not t.is_alive(), "REM worker thread must terminate"
            assert len(result_holder) == 1
            assert result_holder[0] is None, "In-flight REM cycle must return None when preempted"
            assert daemon.current_sleep_phase == SleepPhase.PREEMPTED

        max_rem_lat = max(preemption_latencies)
        avg_rem_lat = sum(preemption_latencies) / len(preemption_latencies)
        print("\n[In-Flight REM Preemption Latency - 20 Trials]")
        print(f"  Average: {avg_rem_lat:.4f} ms")
        print(f"  Max:     {max_rem_lat:.4f} ms")
        assert max_rem_lat < 20.0, f"In-flight REM preemption {max_rem_lat:.2f}ms exceeded 20ms limit"

    def test_rapid_burst_preemption_signals_no_state_leak(self, tmp_path: Path) -> None:
        """Floods the daemon with 500 rapid preemption calls while checking status."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(
            state_file=state_file,
            pid_file=pid_file,
            use_agy_cli=False,
        )

        def bombard() -> None:
            for _ in range(50):
                daemon.interrupt_immediate()
                time.sleep(0.0001)

        threads = [threading.Thread(target=bombard) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert daemon._preemption_event.is_set()
        stat = daemon.status()
        assert stat["running"] is False
        assert stat["current_sleep_phase"] in ("WAKE", "PREEMPTED")


class TestAlternatingPacingTransitionsStress:
    """Stress-tests alternating pacing transitions across 50 consecutive cycles."""

    def test_50_consecutive_cycles_no_phase_drift(self, tmp_path: Path) -> None:
        """Executes 50 consecutive sleep cycles verifying deterministic 2:1 biomorphic alternation."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(
            state_file=state_file,
            pid_file=pid_file,
            nrem_to_rem_ratio=2,
            use_agy_cli=False,
        )

        executed_phases: list[str] = []
        nrem_expected = 0
        rem_expected = 0

        t_start = time.perf_counter()

        for cycle_idx in range(50):
            if nrem_expected < (rem_expected + 1) * 2:
                expected_phase = "NREM_SWS"
                nrem_expected += 1
            else:
                expected_phase = "REM_DREAM"
                rem_expected += 1

            result = daemon.run_single_cycle()
            assert result is not None, f"Cycle {cycle_idx} failed unexpectedly"

            if isinstance(result, NREMExecutionResult):
                actual_phase = "NREM_SWS"
            elif isinstance(result, REMExecutionResult):
                actual_phase = "REM_DREAM"
            else:
                pytest.fail(f"Unknown result type: {type(result)}")

            executed_phases.append(actual_phase)
            assert actual_phase == expected_phase, (
                f"Phase drift at cycle {cycle_idx}: expected {expected_phase}, got {actual_phase}. "
                f"(NREM completed={daemon.nrem_cycles_completed}, REM completed={daemon.rem_cycles_completed})"
            )

        elapsed_total = time.perf_counter() - t_start

        # Validate final counts
        assert daemon.nrem_cycles_completed == 34
        assert daemon.rem_cycles_completed == 16
        assert daemon._total_cycles == 50
        assert daemon.status()["nrem_cycles_completed"] == 34
        assert daemon.status()["rem_cycles_completed"] == 16

        print("\n[50 Consecutive Cycles Benchmark]")
        print(f"  Total time: {elapsed_total:.2f}s ({elapsed_total/50*1000:.2f} ms/cycle)")
        print(f"  NREM completed: {daemon.nrem_cycles_completed}")
        print(f"  REM completed:  {daemon.rem_cycles_completed}")

    def test_pacing_stability_under_intermittent_preemption(self, tmp_path: Path) -> None:
        """Verifies that preempted cycles do NOT advance counters and cause phase drift."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(
            state_file=state_file,
            pid_file=pid_file,
            nrem_to_rem_ratio=2,
            use_agy_cli=False,
        )

        completed_phases: list[str] = []
        preemptions_induced = 0

        for step in range(60):
            # Intermittently induce preemption every 7th step
            if step % 7 == 0:
                daemon._preemption_event.set()
                res = daemon.run_single_cycle()
                assert res is None
                assert daemon.current_sleep_phase == SleepPhase.PREEMPTED
                preemptions_induced += 1
                daemon._preemption_event.clear()
                daemon.current_sleep_phase = SleepPhase.WAKE
            else:
                res = daemon.run_single_cycle()
                assert res is not None
                if isinstance(res, NREMExecutionResult):
                    completed_phases.append("NREM_SWS")
                else:
                    completed_phases.append("REM_DREAM")

        n_completed = len(completed_phases)
        for i, phase in enumerate(completed_phases):
            expected = "REM_DREAM" if (i % 3 == 2) else "NREM_SWS"
            assert phase == expected, f"Drift detected at completed index {i}: expected {expected}, got {phase}"

        print("\n[Intermittent Preemption Resilience]")
        print(f"  Preemptions induced: {preemptions_induced}")
        print(f"  Valid completed cycles: {n_completed}")

    def test_alternative_pacing_ratios_50_cycles(self, tmp_path: Path) -> None:
        """Verifies pacing across different ratios (1:1 and 3:1) for 50 cycles."""
        for ratio in (1, 3):
            state_file = tmp_path / f"state_{ratio}.json"
            pid_file = tmp_path / f"quanta_dream_{ratio}.pid"
            daemon = SubconsciousDaemon(
                state_file=state_file,
                pid_file=pid_file,
                nrem_to_rem_ratio=ratio,
                use_agy_cli=False,
            )

            nrem_exp = 0
            rem_exp = 0
            for i in range(50):
                if nrem_exp < (rem_exp + 1) * ratio:
                    exp = "NREM"
                    nrem_exp += 1
                else:
                    exp = "REM"
                    rem_exp += 1

                res = daemon.run_single_cycle()
                assert res is not None
                act = "NREM" if isinstance(res, NREMExecutionResult) else "REM"
                assert act == exp, f"Ratio {ratio} mismatch at cycle {i}: expected {exp}, got {act}"

            assert daemon.nrem_cycles_completed == nrem_exp
            assert daemon.rem_cycles_completed == rem_exp


class TestThreadSafetyAndConcurrencyStress:
    """Stress-tests concurrency, thread safety of sleep mutations, and telemetry logging."""

    def test_concurrent_telemetry_logging_integrity(self, tmp_path: Path) -> None:
        """Fires 500 concurrent sleep telemetry records across 10 threads to verify JSONL integrity."""
        telem_file = tmp_path / "concurrent_sleep_telemetry.jsonl"
        num_threads = 10
        records_per_thread = 50
        total_expected = num_threads * records_per_thread

        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                for j in range(records_per_thread):
                    phase = "NREM_SWS" if j % 2 == 0 else "REM_DREAM"
                    record_sleep_cycle_telemetry(
                        phase=phase,
                        duration_ms=10.0 + j,
                        metrics={"thread_id": thread_id, "iter": j, "token_count": 100 + j},
                        telemetry_file=telem_file,
                        workspace="stress_concurrency",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - t0

        assert not errors, f"Telemetry logging encountered thread errors: {errors}"
        assert telem_file.exists()

        events = read_telemetry_events(
            event_type="sleep_cycle",
            limit=total_expected + 50,
            telemetry_file=telem_file,
        )
        assert len(events) >= total_expected

        with open(telem_file, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == total_expected, f"Expected {total_expected} JSONL lines, got {len(lines)}"

        for line_no, line in enumerate(lines, 1):
            try:
                parsed = json.loads(line)
                assert parsed["event_type"] == "sleep_cycle"
                assert parsed["sleep_phase"] in ("NREM_SWS", "REM_DREAM")
            except json.JSONDecodeError as err:
                pytest.fail(f"Corrupted JSON at line {line_no}: {line} -> {err}")

        print(f"\n[Concurrent Telemetry Benchmark - {total_expected} Events]")
        print(f"  Total time: {elapsed:.2f}s ({total_expected/elapsed:.1f} events/sec)")
        print(f"  All {total_expected} records parsed with 0 JSON errors")

    def test_concurrent_sleep_state_mutations_and_inspections(self, tmp_path: Path) -> None:
        """Executes concurrent state modifications and inspections across 8 parallel threads."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(
            state_file=state_file,
            pid_file=pid_file,
            use_agy_cli=False,
        )

        for i in range(10):
            daemon.consolidator.memory_manager.record_decision(
                key=f"init_key_{i}",
                content=f"Initial state decision {i}",
                salience=1.5,
                category="decision",
            )
        daemon.consolidator.save_state_atomic()

        stop_workers = threading.Event()
        mutation_errors: list[Exception] = []
        cycles_run = 0
        cycles_lock = threading.Lock()

        def cycle_runner() -> None:
            nonlocal cycles_run
            while not stop_workers.is_set():
                try:
                    res = daemon.run_single_cycle()
                    if res is not None:
                        with cycles_lock:
                            cycles_run += 1
                    time.sleep(0.005)
                except Exception as e:
                    mutation_errors.append(e)

        def status_inspector() -> None:
            while not stop_workers.is_set():
                try:
                    st = daemon.status()
                    assert "current_sleep_phase" in st
                    assert "nrem_cycles_completed" in st
                    assert "rem_cycles_completed" in st
                    time.sleep(0.002)
                except Exception as e:
                    mutation_errors.append(e)

        def disk_loader() -> None:
            while not stop_workers.is_set():
                try:
                    data = daemon.consolidator.load_state()
                    assert isinstance(data, dict)
                    assert "engrams" in data
                    time.sleep(0.003)
                except Exception as e:
                    mutation_errors.append(e)

        threads: list[threading.Thread] = []
        for _ in range(3):
            threads.append(threading.Thread(target=cycle_runner))
        for _ in range(3):
            threads.append(threading.Thread(target=status_inspector))
        for _ in range(2):
            threads.append(threading.Thread(target=disk_loader))

        for t in threads:
            t.start()

        time.sleep(2.5)
        stop_workers.set()

        for t in threads:
            t.join(timeout=3.0)
            assert not t.is_alive(), "Worker thread hung during concurrent stress"

        assert not mutation_errors, f"Concurrent execution produced errors: {mutation_errors}"
        assert cycles_run > 0, "No cycles were completed during stress run"

        final_state = daemon.consolidator.load_state()
        assert isinstance(final_state, dict)
        assert len(final_state.get("engrams", [])) > 0

        print("\n[Concurrent Sleep State Mutations Benchmark]")
        print(f"  Cycles successfully completed under concurrency: {cycles_run}")
        print(f"  Total errors: {len(mutation_errors)}")
        print(f"  Final disk state valid: {len(final_state['engrams'])} engrams")
