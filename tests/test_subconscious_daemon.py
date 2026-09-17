"""tests/test_subconscious_daemon.py — Unit & Integration tests for SubconsciousDaemon.

Validates:
1. Daemon initialization, QoS configuration, and subsystem coordination.
2. PID lifecycle tracking (start creates PID, stop cleans up PID, stale PID recovery).
3. SubconsciousDaemon.status() reporting and SubconsciousDaemon.inspect() filtering.
4. Instant preemption latency (< 20ms) via thread event and SIGUSR1 signal.
5. Single-cycle execution and SWR memory consolidation.
6. Quanta CLI dream commands (start, stop, status, inspect).
"""

from __future__ import annotations

import os
import random
import signal
import time
from pathlib import Path

import pytest

from quanta.cli import main as cli_main
from quanta.cognitive.daemon import SubconsciousDaemon
from quanta.cognitive.mind_wander import DreamInsight
from quanta.cognitive.tom_analyzer import DreamSeed


@pytest.fixture(autouse=True, scope="module")
def settle_cpu_after_daemon_tests() -> None:
    """Settle CPU ticks and restore RNG state after running daemon lifecycle tests."""
    rng_state = random.getstate()
    yield
    time.sleep(0.08)
    random.setstate(rng_state)


class TestSubconsciousDaemon:
    """Comprehensive test suite for the background subconscious daemon."""

    def test_daemon_initialization(self, tmp_path: Path) -> None:
        """Verifies clean subsystem wiring and default parameters."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"

        daemon = SubconsciousDaemon(
            state_file=state_file,
            pid_file=pid_file,
            idle_min=10.0,
            lambda_0=0.2,
        )

        assert daemon.state_file == state_file
        assert daemon.pid_file == pid_file
        assert daemon.idle_min == 10.0
        assert daemon.lambda_0 == 0.2
        assert daemon.is_running() is False
        assert daemon.trigger.lambda_0 == 0.2
        assert daemon.consolidator.state_file == state_file

    def test_daemon_start_and_stop_lifecycle(self, tmp_path: Path) -> None:
        """Verifies start() writes current PID and stop() cleans up PID file."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"

        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)

        # Start background worker thread
        daemon.start(foreground=False)
        assert daemon.is_running() is True
        assert pid_file.exists()

        with open(pid_file, encoding="utf-8") as f:
            saved_pid = int(f.read().strip())
        assert saved_pid == os.getpid()

        # Stop daemon
        daemon.stop(timeout=2.0)
        assert daemon.is_running() is False
        assert not pid_file.exists()

    def test_daemon_status_contract(self, tmp_path: Path) -> None:
        """Verifies status dictionary contains all expected telemetry fields."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"

        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)
        stat = daemon.status()

        required_keys = {
            "running",
            "pid",
            "system_idle",
            "total_cycles",
            "consolidated_insights",
            "preempted_cycles",
            "last_dream_time",
            "uptime_seconds",
            "qos",
            "io_policy",
            "pid_file",
            "state_file",
        }
        assert required_keys.issubset(stat.keys())
        assert stat["running"] is False
        assert stat["pid"] is None
        assert "QOS_CLASS_BACKGROUND" in stat["qos"]
        assert "IOPOL_THROTTLE" in stat["io_policy"]

    def test_daemon_inspect_retrieves_dream_insights(self, tmp_path: Path) -> None:
        """Verifies inspect() extracts and orders subconscious dream engrams."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        daemon = SubconsciousDaemon(state_file=state_file)

        insight1 = DreamInsight(
            topic="architecture_a",
            seed_question="Question A?",
            synthesis="Synthesis A",
            confidence=0.85,
            turns_taken=2,
            tokens_used=200,
            anti_rumination_reset_occurred=False,
        )
        insight2 = DreamInsight(
            topic="architecture_b",
            seed_question="Question B?",
            synthesis="Synthesis B",
            confidence=0.99,
            turns_taken=3,
            tokens_used=350,
            anti_rumination_reset_occurred=False,
        )
        daemon.consolidator.consolidate_insight(insight1)
        daemon.consolidator.consolidate_insight(insight2)

        # Inspect top insights
        results = daemon.inspect(limit=10)
        assert len(results) == 2

        # Check sorting: highest confidence/salience first
        topics = [r["topic"] for r in results]
        assert topics[0] == "architecture_b"
        assert topics[1] == "architecture_a"

    def test_instant_preemption_thread_event_latency(self, tmp_path: Path) -> None:
        """Verifies interrupt_immediate() aborts active cycle in < 20ms."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        daemon = SubconsciousDaemon(state_file=state_file)

        seed = DreamSeed(
            topic="preemption_latency_test",
            speculative_question="Can it interrupt fast?",
            urgency=1.0,
            context_keys=[],
        )

        t0 = time.perf_counter()
        daemon.interrupt_immediate()
        insight = daemon.wander_engine.execute_dream_cycle(
            seed,
            preemption_check=lambda: daemon._preemption_event.is_set(),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert insight is None
        assert elapsed_ms < 20.0  # Strict < 20ms preemption requirement

    def test_instant_preemption_sigusr1_signal(self, tmp_path: Path) -> None:
        """Verifies SIGUSR1 signal triggers interrupt_immediate() and sets preemption event."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        daemon = SubconsciousDaemon(state_file=state_file)

        assert daemon._preemption_event.is_set() is False

        # Send SIGUSR1 to own process
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.01)  # allow signal dispatch

        assert daemon._preemption_event.is_set() is True

    def test_stale_pid_recovery(self, tmp_path: Path) -> None:
        """Verifies daemon recovers cleanly when a dead/stale PID exists in pid_file."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"

        # Write non-existent PID
        with open(pid_file, "w", encoding="utf-8") as f:
            f.write("99999999")

        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)
        assert daemon.is_running() is False
        assert not pid_file.exists()  # Stale PID file cleaned up automatically

    def test_run_single_cycle_executes_and_consolidates(self, tmp_path: Path) -> None:
        """Verifies run_single_cycle() generates and persists an insight on demand."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        daemon = SubconsciousDaemon(state_file=state_file)

        seed = DreamSeed(
            topic="single_cycle_test",
            speculative_question="Does manual cycle work?",
            urgency=2.0,
            context_keys=["quanta.cognitive.daemon"],
        )

        insight = daemon.run_single_cycle(seed=seed)
        assert insight is not None
        assert insight.topic == "single_cycle_test"
        assert daemon._consolidated_count == 1
        assert state_file.exists()

        data = daemon.consolidator.load_state()
        keys = [e["key"] for e in data.get("engrams", [])]
        assert "insight_single_cycle_test" in keys

    def test_cli_dream_commands(self, tmp_path: Path, capsys: object) -> None:
        """Verifies quanta dream CLI commands (status, inspect, start, stop)."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"

        # 1. status
        exit_code = cli_main([
            "dream",
            "status",
            "--pid-file",
            str(pid_file),
            "--state-file",
            str(state_file),
        ])
        assert exit_code == 0

        # 2. inspect (initially empty)
        exit_code = cli_main([
            "dream",
            "inspect",
            "--state-file",
            str(state_file),
        ])
        assert exit_code == 0

        # 3. start
        exit_code = cli_main([
            "dream",
            "start",
            "--pid-file",
            str(pid_file),
            "--state-file",
            str(state_file),
        ])
        assert exit_code == 0
        assert pid_file.exists()

        # 4. stop
        exit_code = cli_main([
            "dream",
            "stop",
            "--pid-file",
            str(pid_file),
        ])
        assert exit_code == 0
        assert not pid_file.exists()

    def test_daemon_signal_handlers_registered(self, tmp_path: Path) -> None:
        """Verifies that SIGUSR1, SIGTERM, and SIGINT handlers are registered on main thread."""
        state_file = tmp_path / "state.json"
        _ = SubconsciousDaemon(state_file=state_file)

        h_usr1 = signal.getsignal(signal.SIGUSR1)
        h_term = signal.getsignal(signal.SIGTERM)
        h_int = signal.getsignal(signal.SIGINT)

        assert callable(h_usr1)
        assert callable(h_term)
        assert callable(h_int)

    def test_daemon_stop_escalation_to_sigkill(self, tmp_path: Path) -> None:
        """Verifies stop() escalates to SIGKILL if external process ignores SIGTERM."""
        import subprocess
        import sys

        pid_file = tmp_path / "quanta_dream.pid"
        state_file = tmp_path / "state.json"

        # Spawn a python process that ignores SIGTERM and sleeps
        code = (
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "time.sleep(30)\n"
        )
        proc = subprocess.Popen([sys.executable, "-c", code])
        try:
            with open(pid_file, "w", encoding="utf-8") as f:
                f.write(str(proc.pid))

            daemon = SubconsciousDaemon(pid_file=pid_file, state_file=state_file)
            assert daemon.is_running() is True

            # Stop with small timeout so escalation occurs promptly
            daemon.stop(timeout=0.2)

            assert daemon.is_running() is False
            assert not pid_file.exists()
            assert proc.wait(timeout=1.0) is not None
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_daemon_coverage_edge_cases(self, tmp_path: Path) -> None:
        """Covers missing branches: _is_pid_alive edge cases, invalid pid contents, signals."""
        from unittest.mock import patch

        from quanta.cognitive.daemon import _is_pid_alive

        assert _is_pid_alive(0) is False
        assert _is_pid_alive(-1) is False

        with patch("os.kill", side_effect=PermissionError):
            assert _is_pid_alive(999999) is True

        with patch("os.kill", side_effect=OSError):
            assert _is_pid_alive(999999) is False

        pid_file = tmp_path / "corrupted.pid"
        pid_file.write_text("not_a_pid")
        daemon = SubconsciousDaemon(pid_file=pid_file, state_file=tmp_path / "state.json")
        assert daemon._read_pid() is None

        # Direct invocation of signal callbacks
        daemon._handle_sigterm_sigint(signal.SIGTERM, None)
        daemon._handle_sigusr1(signal.SIGUSR1, None)
        assert daemon._preemption_event.is_set()

    def test_daemon_run_cycle_branches(self, tmp_path: Path) -> None:
        """Covers run_single_cycle with seed=None, and preemption mid-cycle."""
        daemon = SubconsciousDaemon(
            state_file=tmp_path / "state.json",
            pid_file=tmp_path / "daemon.pid",
        )

        # 1. seed=None synthesizes seed
        insight = daemon.run_single_cycle(seed=None)
        assert insight is not None
        assert daemon._total_cycles == 1

        # 2. preemption flag set causes run_single_cycle to return None
        daemon._preemption_event.set()
        res = daemon.run_single_cycle(seed=None)
        assert res is None
        assert daemon._preempted_count == 1


