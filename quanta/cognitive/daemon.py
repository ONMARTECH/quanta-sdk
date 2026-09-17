"""quanta/cognitive/daemon.py — Autonomous Subconscious Mind-Wandering Daemon Manager.

Implements Pillar 3 of the Quanta Cognitive Architecture:
1. Low-level QoS pinning (QOS_CLASS_BACKGROUND = 0x09, IOPOL_THROTTLE) to Apple Silicon E-cores.
2. Stochastic coordination between Darwin quiescence, Poisson spindle triggers,
   Theory of Mind analysis, headless Antigravity dialectic, and SWR memory consolidation.
3. Sub-20ms instant preemption reflex via SIGUSR1 and thread event signaling.
4. PID lifecycle tracking (quanta_dream.pid) and inspection telemetry.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

from quanta.cognitive.consolidation import SubconsciousConsolidator
from quanta.cognitive.darwin_idle import (
    get_user_idle_seconds,
    is_system_idle,
    set_background_qos,
)
from quanta.cognitive.mind_wander import DreamInsight, MindWanderEngine
from quanta.cognitive.poisson_trigger import PoissonSpindleTrigger
from quanta.cognitive.tom_analyzer import DreamSeed, TheoryOfMindAnalyzer
from quanta.cognitive.workspace_harvester import WorkspaceContextHarvester

logger = logging.getLogger(__name__)


def _is_pid_alive(pid: int) -> bool:
    """Check if process with given PID is currently active and alive."""
    if pid <= 0:
        return False
    # Reap zombie if pid happens to be a child of the current process
    with contextlib.suppress(ChildProcessError, OSError):
        wpid, _ = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


class SubconsciousDaemon:
    """Coordinates autonomous biomorphic mind-wandering in background Apple Silicon E-cores.

    Enforces Darwin Mach background QoS (0x09) and IOPOL_THROTTLE to prevent
    foreground resource contention, guarantees instantaneous preemption (< 20ms)
    upon user arrival, and manages daemon PID lifecycle and status reporting.
    """

    def __init__(
        self,
        state_file: Path | str = "quanta_cognitive_state.json",
        pid_file: Path | str = "quanta_dream.pid",
        idle_min: float = 15.0,
        lambda_0: float = 0.1,
        tau: float = 5.0,
        refractory_sec: float = 10.0,
        idle_threshold: float = 0.70,
        max_thermal: int = 1,
        max_turns: int = 5,
        max_tokens: int = 2500,
    ) -> None:
        """Initialize SubconsciousDaemon with component subsystems and configuration."""
        self.state_file = Path(state_file)
        self.pid_file = Path(pid_file)
        self.idle_min = float(idle_min)
        self.lambda_0 = float(lambda_0)
        self.tau = float(tau)
        self.refractory_sec = float(refractory_sec)
        self.idle_threshold = float(idle_threshold)
        self.max_thermal = int(max_thermal)

        # Core cognitive subsystems
        self.trigger = PoissonSpindleTrigger(
            lambda_0=self.lambda_0,
            idle_min=self.idle_min,
            tau=self.tau,
            refractory_sec=self.refractory_sec,
        )
        self.tom_analyzer = TheoryOfMindAnalyzer()
        self.wander_engine = MindWanderEngine(
            max_turns=max_turns,
            max_tokens=max_tokens,
        )
        self.consolidator = SubconsciousConsolidator(state_file=self.state_file)
        self.workspace_harvester = WorkspaceContextHarvester(state_file=self.state_file)

        # Threading & Preemption primitives
        self._stop_event = threading.Event()
        self._preemption_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._is_running = False

        # Performance & telemetry metrics
        self._total_cycles = 0
        self._consolidated_count = 0
        self._preempted_count = 0
        self._last_dream_time: float | None = None
        self._start_time: float | None = None
        self._last_active_time: float = time.time()
        self._foreground: bool = False

        # Register SIGUSR1 signal handler for instant preemption if in main thread
        self._setup_signals()

    def _setup_signals(self) -> None:
        """Register OS signal handlers for SIGUSR1 instant preemption and graceful shutdown."""
        try:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGUSR1, self._handle_sigusr1)
                signal.signal(signal.SIGTERM, self._handle_sigterm_sigint)
                signal.signal(signal.SIGINT, self._handle_sigterm_sigint)
        except (ValueError, AttributeError):
            pass

    def _handle_sigusr1(self, signum: int, frame: Any) -> None:
        """Signal handler catching SIGUSR1 to abort active cycle in < 20ms."""
        self.interrupt_immediate()

    def _handle_sigterm_sigint(self, signum: int, frame: Any) -> None:
        """Signal handler catching SIGTERM and SIGINT for graceful daemon termination."""
        logger.info("Signal %s received; initiating clean daemon stop.", signum)
        self.stop()

    def interrupt_immediate(self) -> None:
        """Instantly pre-empt and abort active dream cycle (< 20ms latency budget)."""
        self._preemption_event.set()
        # If external process PID exists and is not self, send SIGUSR1
        ext_pid = self._read_pid()
        if ext_pid is not None and ext_pid != os.getpid() and _is_pid_alive(ext_pid):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(ext_pid, signal.SIGUSR1)

    def _read_pid(self) -> int | None:
        """Read recorded PID from pid_file if it exists."""
        if not self.pid_file.exists():
            return None
        try:
            with open(self.pid_file, encoding="utf-8") as f:
                content = f.read().strip()
                return int(content) if content else None
        except (ValueError, OSError):
            return None

    def _write_pid(self, pid: int) -> None:
        """Write current PID to pid_file atomically."""
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pid_file, "w", encoding="utf-8") as f:
            f.write(str(pid))

    def _remove_pid(self) -> None:
        """Remove pid_file from filesystem."""
        with contextlib.suppress(OSError):
            if self.pid_file.exists():
                self.pid_file.unlink()

    def is_running(self) -> bool:
        """Check if daemon is currently running locally or as external process."""
        if self._is_running and self._worker_thread is not None and self._worker_thread.is_alive():
            return True
        pid = self._read_pid()
        if pid is not None:
            if _is_pid_alive(pid):
                return True
            # Stale PID file found, clean up
            self._remove_pid()
        return False

    def start(self, foreground: bool = False) -> None:
        """Start the subconscious mind-wandering daemon.

        Enforces QOS_CLASS_BACKGROUND (0x09) and IOPOL_THROTTLE on current process,
        records PID to `self.pid_file`, and initiates execution either in
        the foreground or as a background worker thread.

        Args:
            foreground: If True, blocks on the main thread loop.
        """
        # 1. Check if already running
        if self.is_running():
            logger.info("SubconsciousDaemon is already running with PID %s", self._read_pid())
            return

        # 2. Enforce Darwin background QoS & I/O throttling
        set_background_qos()

        # 3. Write PID
        self._write_pid(os.getpid())

        # 4. Reset lifecycle events & state
        self._stop_event.clear()
        self._preemption_event.clear()
        self._is_running = True
        self._start_time = time.time()
        self._last_active_time = time.time()
        self._foreground = foreground

        if foreground:
            print("👁️  Canlı Telemetri Devrede: Terminalde rüya döngüleri izleniyor.", flush=True)
            print("   (Durdurmak için Ctrl + C tuşlarına basabilirsiniz)\n", flush=True)
            try:
                self._run_loop()
            finally:
                self.stop()
        else:
            self._worker_thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="SubconsciousDaemonWorker",
            )
            self._worker_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the subconscious daemon and cleanly release resources.

        Args:
            timeout: Maximum seconds to wait for worker thread or external process termination.
        """
        self._stop_event.set()
        self.interrupt_immediate()

        # Stop external PID if running
        ext_pid = self._read_pid()
        if ext_pid is not None and ext_pid != os.getpid() and _is_pid_alive(ext_pid):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(ext_pid, signal.SIGTERM)
            t_end = time.time() + timeout
            while time.time() < t_end and _is_pid_alive(ext_pid):
                time.sleep(0.05)

            # If external process is still alive after timeout, escalate to SIGKILL
            if _is_pid_alive(ext_pid):
                logger.warning(
                    "External PID %s did not terminate after SIGTERM; escalating to SIGKILL.",
                    ext_pid,
                )
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(ext_pid, signal.SIGKILL)
                # Sleep up to 0.5s to confirm death
                t_kill = time.time() + 0.5
                while time.time() < t_kill and _is_pid_alive(ext_pid):
                    time.sleep(0.05)

        # Join local worker thread if present
        if self._worker_thread is not None and self._worker_thread.is_alive():
            if threading.current_thread() != self._worker_thread:
                self._worker_thread.join(timeout=timeout)
            self._worker_thread = None

        # Only remove PID file once the process has truly terminated
        remaining_pid = self._read_pid()
        if (
            remaining_pid is not None
            and remaining_pid != os.getpid()
            and _is_pid_alive(remaining_pid)
        ):
            logger.error("External process %s remains alive; preserving PID file.", remaining_pid)
        else:
            self._remove_pid()

        self._is_running = False

    def run_single_cycle(self, seed: DreamSeed | None = None) -> DreamInsight | None:
        """Execute a single atomic dream cycle and consolidate any generated insight.

        Args:
            seed: Optional DreamSeed to deliberate on. If None, synthesizes default seed.

        Returns:
            DreamInsight if successful and not preempted; None otherwise.
        """
        self._total_cycles += 1

        if seed is None:
            # Dynamically harvest speculative seed across workspace projects
            seed = self.workspace_harvester.generate_next_seed()

        proj_label = (
            seed.context_keys[0]
            if seed.context_keys and seed.context_keys[0] != seed.topic
            else "quanta"
        )

        if self._foreground:
            print(f"\n🌙 [Rüya Başladı] Proje: '{proj_label}' | Konu: '{seed.topic}'", flush=True)
            print(f"   Soru: '{seed.speculative_question}'", flush=True)
            print("   DMN (T=0.85) ile Zeno (T=0.20) müzakere ediyor...", flush=True)

        start_user_idle = get_user_idle_seconds()

        def preemption_check() -> bool:
            if self._preemption_event.is_set() or self._stop_event.is_set():
                return True
            # Real-time physical user input check (< 20ms preemption reflex).
            # Fires if the system was idle when dream initiated (start_user_idle >= 1.0s)
            # and the user subsequently generated physical HID input (uidle < 1.0s).
            if start_user_idle is not None and start_user_idle >= 1.0:
                uidle = get_user_idle_seconds()
                if uidle is not None and uidle < 1.0:
                    self._preemption_event.set()
                    return True
            return False

        insight = self.wander_engine.execute_dream_cycle(seed, preemption_check=preemption_check)

        if insight is not None:
            self._last_dream_time = time.time()
            self.consolidator.consolidate_insight(insight)
            self._consolidated_count += 1
            if self._foreground:
                conf = insight.confidence * 100.0
                print(f"💡 [Uzlaşı Sağlandı - Güven: %{conf:.1f}]:", flush=True)
                print(f"   {insight.synthesis[:120]}...", flush=True)
                print(f"🧠 [SWR Mühürlendi]: insight_{seed.topic} (%100)\n", flush=True)
            return insight
        else:
            self._preempted_count += 1
            if self._foreground:
                print(
                    "⚡ [Uyanma Refleksi] Kullanıcı aktivitesi (klavye/fare) algılandı; "
                    "rüya anında kesildi.\n",
                    flush=True,
                )
            return None

    def _run_loop(self) -> None:
        """Internal main coordination loop executed under background QoS."""
        last_heartbeat = time.time()
        while not self._stop_event.is_set():
            try:
                self._preemption_event.clear()
                now = time.time()
                last_dream = self._last_dream_time if self._last_dream_time is not None else 0.0

                # 1. Physical user HID activity check (macOS CoreGraphics / Windows LastInput)
                user_idle = get_user_idle_seconds()
                if user_idle is not None:
                    self._last_active_time = now - user_idle
                    idle_dur = user_idle
                    user_is_idle = user_idle >= self.idle_min
                else:
                    idle_dur = max(0.0, now - self._last_active_time)
                    user_is_idle = idle_dur >= self.idle_min

                # 2. Hardware quiescence check (Mach CPU idle ticks + thermal pressure)
                hw_idle = is_system_idle(
                    idle_threshold=self.idle_threshold,
                    max_thermal=self.max_thermal,
                )

                # Subconscious mind-wandering proceeds only when user is idle AND hardware is quiet
                can_dream = user_is_idle and hw_idle

                if self._foreground and (now - last_heartbeat >= 5.0):
                    last_heartbeat = now
                    rate = self.trigger.compute_rate(
                        now, self._last_active_time, fatigue=0.0, tom_urgency=1.5
                    )
                    prob = 1.0 - math.exp(-rate * 1.0)
                    if user_idle is not None and user_idle < self.idle_min:
                        msg = (
                            f"⏳ [İzleme] Kullanıcı Aktif: {user_idle:.1f}s | "
                            f"Eşik: {self.idle_min:.0f}s | Donanım: {hw_idle} | Bekleniyor..."
                        )
                    else:
                        msg = (
                            f"⏳ [İzleme] Sessizlik: {idle_dur:.0f}s | "
                            f"Donanım: {hw_idle} | Poisson: %{prob * 100:.1f} | Bekleniyor..."
                        )
                    print(msg, flush=True)

                if can_dream:
                    # 3. Stochastic Poisson spindle evaluation
                    should_fire = self.trigger.should_trigger(
                        current_time=now,
                        last_active_time=self._last_active_time,
                        last_dream_time=last_dream,
                        fatigue=0.0,
                        tom_urgency=1.5,
                    )
                    if should_fire:
                        self.run_single_cycle()

                # Sleep brief interval with responsive stop check
                self._stop_event.wait(timeout=0.2)
            except Exception as e:
                logger.error("Error in subconscious daemon loop: %s", e)
                self._stop_event.wait(timeout=1.0)

    def status(self) -> dict[str, Any]:
        """Return comprehensive diagnostic status of the subconscious daemon."""
        active = self.is_running()
        current_pid = self._read_pid() if active else None
        now = time.time()
        uptime = (now - self._start_time) if (active and self._start_time) else 0.0

        return {
            "running": active,
            "pid": current_pid,
            "system_idle": is_system_idle(self.idle_threshold, self.max_thermal),
            "total_cycles": self._total_cycles,
            "consolidated_insights": self._consolidated_count,
            "preempted_cycles": self._preempted_count,
            "last_dream_time": self._last_dream_time,
            "uptime_seconds": round(uptime, 2),
            "qos": "QOS_CLASS_BACKGROUND (0x09)",
            "io_policy": "IOPOL_THROTTLE (3)",
            "pid_file": str(self.pid_file),
            "state_file": str(self.state_file),
        }

    def inspect(self, limit: int = 10) -> list[dict[str, Any]]:
        """Inspect recent consolidated subconscious dream insights from persistent storage.

        Args:
            limit: Maximum number of insights to return.

        Returns:
            List of dictionary summaries of dream engrams.
        """
        all_insights = self.consolidator.get_insights()
        # Sort by salience * fidelity descending
        all_insights.sort(
            key=lambda x: float(x.get("salience", 1.0)) * float(x.get("fidelity", 1.0)),
            reverse=True,
        )
        return all_insights[:limit]
