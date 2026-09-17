"""tests/test_m4_empirical_challenge.py — Empirical challenge suite for Milestone M4.

Validates:
1. Preemption Latency Budget:
   - 1,200 empirical preemption checks via interrupt_immediate() and SIGUSR1.
   - Verifies average latency strictly < 20ms (target) and worst-case latency < 50ms (hard limit).
2. Stress Test SWR Consolidation:
   - 50 consecutive calls to SubconsciousConsolidator.consolidate_insight() with pruning.
   - Verifies valid JSON structure, CSF biophysical quantum shielding intact, and SHY
     microglial downscaling removes decayed items without dropping protected dream insights.
3. Subconscious Daemon PID Locking:
   - Verifies running two daemons simultaneously is prevented (in-process and cross-process).
   - Verifies stale PID files referencing non-existent PIDs are detected and
     recovered automatically.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch

from quanta.cognitive.consolidation import SubconsciousConsolidator
from quanta.cognitive.daemon import SubconsciousDaemon, _is_pid_alive
from quanta.cognitive.mind_wander import DreamInsight
from quanta.cognitive.tom_analyzer import DreamSeed
from quanta.torch.brain import CSFShieldedEnvironment

# ============================================================================
# Section 1: Preemption Latency Empirical Benchmark (1,000+ Runs)
# ============================================================================


class TestPreemptionLatencyEmpirical:
    """Empirical timing and latency distribution benchmark for preemption reflex."""

    def test_preemption_interrupt_immediate_1000_runs(self, tmp_path: Path) -> None:
        """Runs 1,000 preemption checks via interrupt_immediate().

        Asserts average latency < 20ms and worst-case latency < 50ms across all 1,000 runs.
        """
        state_file = tmp_path / "bench_state.json"
        pid_file = tmp_path / "bench_daemon.pid"
        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)

        seed = DreamSeed(
            topic="preemption_stress",
            speculative_question="Can preemption maintain sub-20ms latency under 1,000 iterations?",
            urgency=1.0,
            context_keys=["darwin_idle"],
        )

        latencies_ms: list[float] = []
        run_count = 1000

        for i in range(run_count):
            daemon._preemption_event.clear()
            daemon._stop_event.clear()

            # Record exact trigger timestamp
            t0 = time.perf_counter()
            daemon.interrupt_immediate()
            res = daemon.run_single_cycle(seed)
            t1 = time.perf_counter()

            latency = (t1 - t0) * 1000.0
            latencies_ms.append(latency)

            assert res is None, f"Iteration {i}: cycle was not preempted"

        total = len(latencies_ms)
        avg_latency = sum(latencies_ms) / total
        max_latency = max(latencies_ms)
        min_latency = min(latencies_ms)
        sorted_lat = sorted(latencies_ms)
        p50 = sorted_lat[int(0.50 * total)]
        p95 = sorted_lat[int(0.95 * total)]
        p99 = sorted_lat[int(0.99 * total)]

        print(
            f"\n[Preemption Benchmark - interrupt_immediate ({total} runs)]\n"
            f"  Avg: {avg_latency:.4f} ms (target < 20.0 ms)\n"
            f"  Max (worst-case): {max_latency:.4f} ms (hard limit < 50.0 ms)\n"
            f"  Min: {min_latency:.4f} ms\n"
            f"  P50: {p50:.4f} ms | P95: {p95:.4f} ms | P99: {p99:.4f} ms\n"
        )

        assert total >= 1000
        assert avg_latency < 20.0, f"Average latency {avg_latency:.2f}ms exceeded 20ms target"
        assert max_latency < 50.0, f"Worst-case latency {max_latency:.2f}ms exceeded 50ms limit"

    def test_preemption_mid_cycle_concurrent_100_runs(self, tmp_path: Path) -> None:
        """Runs 100 preemption checks where interrupt_immediate() strikes an
        active worker thread."""
        state_file = tmp_path / "mid_state.json"
        pid_file = tmp_path / "mid_daemon.pid"
        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)

        seed = DreamSeed(
            topic="mid_cycle_stress",
            speculative_question="Can active thread preemption halt in sub-20ms?",
            urgency=1.0,
            context_keys=["darwin_idle"],
        )

        # Introduce realistic 500us deliberation per turn
        orig_simulate_step = daemon.wander_engine.simulator.simulate_step

        def delayed_simulate_step(*args: Any, **kwargs: Any) -> Any:
            time.sleep(0.0005)
            return orig_simulate_step(*args, **kwargs)

        daemon.wander_engine.simulator.simulate_step = delayed_simulate_step  # type: ignore[method-assign]

        latencies_ms: list[float] = []
        run_count = 100

        for i in range(run_count):
            daemon._preemption_event.clear()
            daemon._stop_event.clear()

            cycle_res: list[Any] = []

            def worker(target_res: list[Any] = cycle_res) -> None:
                target_res.append(daemon.run_single_cycle(seed))

            t = threading.Thread(target=worker, name=f"MidWorker-{i}")
            t.start()
            # Allow worker thread to enter turn 0
            time.sleep(0.0008)

            t0 = time.perf_counter()
            daemon.interrupt_immediate()
            t.join(timeout=1.0)
            t1 = time.perf_counter()

            latency = (t1 - t0) * 1000.0
            latencies_ms.append(latency)

            assert len(cycle_res) == 1, f"Iteration {i}: worker did not append result"
            assert cycle_res[0] is None, f"Iteration {i}: worker was not preempted"

        total = len(latencies_ms)
        avg_latency = sum(latencies_ms) / total
        max_latency = max(latencies_ms)

        print(
            f"\n[Preemption Benchmark - Mid-Cycle Concurrent ({total} runs)]\n"
            f"  Avg: {avg_latency:.4f} ms (target < 20.0 ms)\n"
            f"  Max (worst-case): {max_latency:.4f} ms (hard limit < 50.0 ms)\n"
        )

        assert avg_latency < 20.0
        assert max_latency < 50.0

    def test_preemption_sigusr1_signal_latency_100_runs(self, tmp_path: Path) -> None:
        """Runs 100 preemption checks via OS SIGUSR1 signal delivery to the main thread."""
        state_file = tmp_path / "sig_state.json"
        pid_file = tmp_path / "sig_daemon.pid"
        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)

        seed = DreamSeed(
            topic="sigusr1_stress",
            speculative_question="Does SIGUSR1 signal delivery meet preemption latency budget?",
            urgency=1.0,
            context_keys=["darwin_idle"],
        )

        latencies_ms: list[float] = []
        run_count = 100
        daemon._setup_signals()

        for i in range(run_count):
            daemon._preemption_event.clear()
            daemon._stop_event.clear()

            t0 = time.perf_counter()
            os.kill(os.getpid(), signal.SIGUSR1)
            res = daemon.run_single_cycle(seed)
            t1 = time.perf_counter()

            latency = (t1 - t0) * 1000.0
            latencies_ms.append(latency)

            assert res is None, f"Iteration {i}: cycle was not preempted"

        total = len(latencies_ms)
        avg_latency = sum(latencies_ms) / total
        max_latency = max(latencies_ms)

        print(
            f"\n[Preemption Benchmark - SIGUSR1 Signal ({total} runs)]\n"
            f"  Avg: {avg_latency:.4f} ms (target < 20.0 ms)\n"
            f"  Max (worst-case): {max_latency:.4f} ms (hard limit < 50.0 ms)\n"
        )

        assert total >= 100
        assert avg_latency < 20.0, f"SIGUSR1 average latency {avg_latency:.2f}ms exceeded 20ms"
        assert max_latency < 50.0, f"SIGUSR1 max latency {max_latency:.2f}ms exceeded 50ms"


# ============================================================================
# Section 2: Stress Test SWR Memory Consolidation (50 Consecutive Calls)
# ============================================================================


class TestSWRConsolidationStress:
    """Stress testing SubconsciousConsolidator across 50 consecutive cycles with pruning."""

    def test_50_consecutive_consolidations_with_pruning(self, tmp_path: Path) -> None:
        """Executes 50 consecutive consolidate_insight calls with decaying scratchpads.

        Verifies:
        1. quanta_cognitive_state.json maintains valid JSON structure after every call.
        2. CSF dielectric shielding remains intact.
        3. Synaptic Homeostasis (SHY) pruning eliminates decayed items.
        4. Protected dream insights are never dropped.
        """
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(
            state_file=state_file,
            capacity=128,
            enable_csf_shielding=True,
        )

        assert consolidator.memory_manager.csf_env is not None
        assert isinstance(consolidator.memory_manager.csf_env, CSFShieldedEnvironment)

        consecutive_calls = 50
        created_dream_topics: list[str] = []

        for i in range(consecutive_calls):
            # 1. Inject an ephemeral low-salience scratchpad item destined to decay
            scratch_key = f"scratch_item_{i}"
            consolidator.memory_manager.record_decision(
                key=scratch_key,
                content=f"Temporary unreinforced memory trace {i}",
                salience=0.25,  # Low salience <= 0.50
                category="scratchpad",
            )

            # 2. Advance biological time by 1 turn
            consolidator.memory_manager.step(dt=1.0)

            # 3. Formulate and consolidate a subconscious dream insight
            topic = f"subconscious_architectural_pattern_{i}"
            created_dream_topics.append(topic)

            insight = DreamInsight(
                topic=topic,
                seed_question=f"How to optimize biomorphic cognitive pathway {i}?",
                synthesis=f"Consensus formulation {i}: Grounded via SWR replay and CSF shielding.",
                confidence=0.90 + (i % 10) * 0.01,
                turns_taken=3,
                tokens_used=400 + i * 5,
                anti_rumination_reset_occurred=False,
            )

            success = consolidator.consolidate_insight(insight)
            assert success is True, f"Consolidation failed at iteration {i}"

            # 4. Verify JSON validity on disk after every single write
            assert state_file.exists(), f"State file missing at iteration {i}"
            with open(state_file, encoding="utf-8") as f:
                data = json.load(f)

            assert "turn_count" in data
            assert data["turn_count"] == i + 1
            assert "engrams" in data
            assert isinstance(data["engrams"], list)

            # Sibling temp files must not leak
            temp_files = list(state_file.parent.glob(".tmp_*"))
            assert len(temp_files) == 0, f"Temp file leak detected at iteration {i}: {temp_files}"

        # Post-stress verification on final state
        final_state = consolidator.load_state()
        persisted_engrams = final_state.get("engrams", [])
        persisted_keys = {e["key"] for e in persisted_engrams}

        # Verify CSF shielding intact
        csf = consolidator.memory_manager.csf_env
        assert csf is not None
        attenuation = csf.compute_attenuation_factor()
        assert isinstance(attenuation, torch.Tensor)
        assert 0.0 < float(attenuation.item()) < 1.0

        # Verify SHY pruning removed decayed scratchpad items
        # Items older than max_age=20 turns must have been pruned by SHY
        # Thus, scratch items from iteration 0 to 28 must NOT exist
        for old_i in range(consecutive_calls - 22):
            assert f"scratch_item_{old_i}" not in persisted_keys, (
                f"Decayed scratchpad item {old_i} was not pruned by SHY"
            )

        # Verify protected dream insights are preserved without dropping
        dream_insights_in_state = [
            e for e in persisted_engrams if e.get("category") == "subconscious_dream"
        ]
        assert len(dream_insights_in_state) == consecutive_calls, (
            f"Expected {consecutive_calls} protected dream insights, "
            f"found {len(dream_insights_in_state)}"
        )

        for topic in created_dream_topics:
            key = f"insight_{topic}"
            assert key in persisted_keys, f"Protected dream insight '{key}' was incorrectly dropped"

        for dream in dream_insights_in_state:
            assert dream["salience"] >= 2.0, "Dream salience below protection threshold"
            assert dream["fidelity"] >= 0.95, "Dream fidelity unexpectedly low"
            assert "subconscious" in dream["tags"]
            assert "dream" in dream["tags"]

        # Reload verification into a fresh SubconsciousConsolidator instance
        fresh_consolidator = SubconsciousConsolidator(
            state_file=state_file,
            capacity=128,
            enable_csf_shielding=True,
        )
        loaded_insights = fresh_consolidator.get_insights()
        assert len(loaded_insights) == consecutive_calls


# ============================================================================
# Section 3: Subconscious Daemon PID Locking & Recovery Tests
# ============================================================================


class TestDaemonPIDLocking:
    """Verification of mutual exclusion and stale PID recovery."""

    def test_simultaneous_daemons_prevented_in_process(self, tmp_path: Path) -> None:
        """Verifies that a second daemon instance is prevented from starting simultaneously."""
        state_file = tmp_path / "lock_state.json"
        pid_file = tmp_path / "quanta_dream.pid"

        daemon1 = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)
        daemon2 = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)

        assert not daemon1.is_running()
        assert not daemon2.is_running()

        # Start daemon 1 in background worker
        daemon1.start(foreground=False)
        time.sleep(0.05)

        assert daemon1.is_running()
        assert daemon2.is_running()
        assert pid_file.exists()
        assert int(pid_file.read_text().strip()) == os.getpid()

        # Attempt to start daemon 2 — must be rejected and safely no-op
        daemon2.start(foreground=False)
        assert daemon2._worker_thread is None

        # Clean shutdown of daemon 1
        daemon1.stop(timeout=2.0)
        assert not daemon1.is_running()
        assert not daemon2.is_running()
        assert not pid_file.exists()

    def test_simultaneous_daemons_prevented_cross_process(self, tmp_path: Path) -> None:
        """Verifies cross-process mutual exclusion preventing two daemons running concurrently."""
        state_file = tmp_path / "cross_state.json"
        pid_file = tmp_path / "cross_daemon.pid"

        helper_script = tmp_path / "run_external_daemon.py"
        helper_script.write_text(
            "import time, os, sys\n"
            "pid_path = sys.argv[1]\n"
            "with open(pid_path, 'w') as f:\n"
            "    f.write(str(os.getpid()))\n"
            "sys.stdout.write('READY\\n')\n"
            "sys.stdout.flush()\n"
            "time.sleep(5.0)\n",
            encoding="utf-8",
        )

        proc = subprocess.Popen(
            [sys.executable, str(helper_script), str(pid_file)],
            stdout=subprocess.PIPE,
            text=True,
        )

        try:
            line = proc.stdout.readline() if proc.stdout else ""
            assert "READY" in line
            assert pid_file.exists()
            assert _is_pid_alive(proc.pid)

            # In current process, daemon must recognize running external PID
            daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)
            assert daemon.is_running() is True
            assert daemon._read_pid() == proc.pid

            # start() must recognize existing PID and return immediately without overwriting PID
            daemon.start(foreground=False)
            assert daemon._worker_thread is None
            assert int(pid_file.read_text().strip()) == proc.pid
        finally:
            proc.terminate()
            proc.wait(timeout=3.0)

        # After external process termination, daemon detects stale PID and recovers
        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)
        assert daemon.is_running() is False
        assert not pid_file.exists()

    def test_stale_pid_detection_and_automatic_recovery(self, tmp_path: Path) -> None:
        """Verifies stale PID files referencing dead processes are purged and recovered."""
        state_file = tmp_path / "stale_state.json"
        pid_file = tmp_path / "stale_daemon.pid"

        # Spawn a short-lived process to ensure process execution succeeds
        subprocess.run(
            [sys.executable, "-c", "import sys; sys.exit(0)"],
            check=True,
        )
        # Choose a guaranteed non-existent PID
        dead_pid = 99999999
        while _is_pid_alive(dead_pid):
            dead_pid -= 1

        pid_file.write_text(str(dead_pid))
        assert pid_file.exists()

        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)

        # is_running() must detect stale PID, remove the file, and return False
        assert daemon.is_running() is False
        assert not pid_file.exists()

        # Daemon must now be able to start normally
        daemon.start(foreground=False)
        time.sleep(0.05)
        try:
            assert daemon.is_running() is True
            assert pid_file.exists()
            assert int(pid_file.read_text().strip()) == os.getpid()
        finally:
            daemon.stop(timeout=2.0)

        assert not daemon.is_running()
        assert not pid_file.exists()
