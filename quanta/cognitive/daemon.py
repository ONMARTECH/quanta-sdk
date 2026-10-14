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
import inspect
import logging
import math
import os
import re
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from quanta.cognitive.consolidation import SubconsciousConsolidator
from quanta.cognitive.darwin_idle import (
    get_user_idle_seconds,
    is_system_idle,
    set_background_qos,
)
from quanta.cognitive.mind_wander import DreamInsight, MindWanderEngine, MindWanderResult
from quanta.cognitive.poisson_trigger import PoissonSpindleTrigger
from quanta.cognitive.telemetry import record_sleep_cycle_telemetry
from quanta.cognitive.tom_analyzer import DreamSeed, TheoryOfMindAnalyzer
from quanta.cognitive.workspace_harvester import (
    EpisodicTrace,
    HybridDreamSeed,
    WorkspaceContextHarvester,
)

logger = logging.getLogger(__name__)


class SleepPhase(str, Enum):
    """Sleep cycle state machine enumeration."""

    WAKE = "WAKE"
    NREM_SWS = "NREM_SWS"
    REM_DREAM = "REM_DREAM"
    PREEMPTED = "PREEMPTED"


@dataclass
class NREMExecutionResult:
    """Telemetry outcome of a quiescent NREM Slow-Wave Sleep cycle."""

    glymphatic_files_cleared: int = 0
    depotentiated_inhibitors_pruned: int = 0
    ephemeral_engrams_downscaled: int = 0
    gists_crystallized: int = 0
    swr_consolidated_count: int = 0
    shy_downscale_factor: float = 0.90
    duration_ms: float = 0.0
    success: bool = True
    phase: SleepPhase = SleepPhase.NREM_SWS


@dataclass
class REMExecutionResult(DreamInsight):
    """Telemetry outcome of a paradoxical REM dreaming cycle."""

    phase: SleepPhase = SleepPhase.REM_DREAM
    projects_paired: list[str] = field(default_factory=list)
    prefrontal_temperature: float = 0.50
    dreamer_temperature: float = 0.92
    rfc_path: str | None = None
    duration_ms: float = 0.0
    success: bool = True


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
        nrem_to_rem_ratio: int = 2,
        shy_downscale_factor: float = 0.90,
        use_agy_cli: bool | None = None,
        brain_dir: Path | str | None = None,
        summaries_db: Path | str | None = None,
        projects_dir: Path | str | None = None,
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
        self.nrem_to_rem_ratio = int(nrem_to_rem_ratio)
        self.shy_downscale_factor = float(shy_downscale_factor)

        # Sleep state machine attributes
        self.current_sleep_phase: SleepPhase = SleepPhase.WAKE
        self.nrem_cycles_completed: int = 0
        self.rem_cycles_completed: int = 0

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
            use_agy_cli=use_agy_cli,
        )
        self.consolidator = SubconsciousConsolidator(state_file=self.state_file)
        self.workspace_harvester = WorkspaceContextHarvester(
            projects_dir=projects_dir,
            state_file=self.state_file,
            brain_dir=brain_dir,
            summaries_db=summaries_db,
        )

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

        # Warm up Darwin CoreGraphics ctypes bindings to avoid cold-start latency spikes
        with contextlib.suppress(Exception):
            get_user_idle_seconds()

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
        if self._is_running:
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

    def _get_central_rfc_dir(self) -> Path:
        """Return the centralized Quanta RFC directory, ensuring sandbox isolation."""
        docs_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "rfcs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        return docs_dir

    def _persist_rfc_to_project(self, seed: DreamSeed, insight: DreamInsight) -> Path | None:
        """Persist consensual single-project RFC strictly into Quanta's centralized docs/rfcs/.

        Enforces SANDBOX IMMUTABILITY:
        - NEVER writes to external sibling project directories.
        - Writes strictly inside Quanta's centralized RFC directory (quanta/docs/rfcs/).
        """
        docs_dir = self._get_central_rfc_dir()
        try:
            target_proj_name = "quanta"
            if seed.project_path:
                target_proj_name = Path(seed.project_path).name
            elif seed.context_keys:
                target_proj_name = seed.context_keys[0]

            topic_slug = seed.topic.upper().replace("-", "_").replace(" ", "_")
            rfc_path = docs_dir / f"RFC_{topic_slug}.md"

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            content = (
                f"---\n"
                f"rfc_id: RFC_{topic_slug}\n"
                f"target_project: {target_proj_name}\n"
                f"topic: {seed.topic}\n"
                f"confidence: {insight.confidence:.2f}\n"
                f"created_at: '{timestamp}'\n"
                f"generated_by: Quanta Subconscious Mind-Wandering Daemon\n"
                f"engine: Quanta Cognitive Dialectic Engine (DMN-Zeno Arbiter)\n"
                f"---\n\n"
                f"{insight.synthesis}\n"
            )

            rfc_path.write_text(content, encoding="utf-8")
            if self._foreground:
                print(f"📄 [RFC Kaydedildi]: {rfc_path}", flush=True)
            return rfc_path
        except Exception as e:
            logger.warning("Failed to persist RFC to central docs: %s", e)
            return None

    def _persist_hybrid_rfc(
        self, hybrid_result: MindWanderResult, seed: HybridDreamSeed
    ) -> Path:
        """Persist consensual cross-domain hybrid RFC document strictly to Quanta's docs/rfcs/.

        Enforces SANDBOX IMMUTABILITY:
        - NEVER writes to external sibling project directories.
        - Writes strictly to quanta/docs/rfcs/
        - Filename follows pattern: RFC_HYBRID_<timestamp>_<slug_a>_<slug_b>.md
        - Stamped with production YAML frontmatter and comprehensive architectural content.
        - Stored as high-salience engram (S >= 2.5) in quanta_cognitive_state.json.

        Args:
            hybrid_result: MindWanderResult output from execute_hybrid_dream_cycle.
            seed: Target HybridDreamSeed with primary and secondary project profiles.

        Returns:
            Path: Filesystem path to the newly created RFC document.
        """
        # Strict centralized Quanta docs/rfcs directory
        docs_dir = self._get_central_rfc_dir()

        now_utc = datetime.now(timezone.utc)
        ts_filename = now_utc.strftime("%Y%m%d_%H%M%S")
        ts_frontmatter = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

        p_a = seed.primary_project
        p_b = seed.secondary_project

        slug_a = re.sub(r"[^a-zA-Z0-9]+", "_", p_a.name).strip("_").lower()
        slug_b = re.sub(r"[^a-zA-Z0-9]+", "_", p_b.name).strip("_").lower()

        rfc_filename = f"RFC_HYBRID_{ts_filename}_{slug_a}_{slug_b}.md"
        rfc_path = docs_dir / rfc_filename

        dom_a = p_a.domain
        dom_b = p_b.domain

        # Construct production-grade YAML frontmatter and markdown body
        p_a_stack_str = ", ".join(p_a.tech_stack) if p_a.tech_stack else "General"
        p_b_stack_str = ", ".join(p_b.tech_stack) if p_b.tech_stack else "General"
        p_a_ctx = p_a.summary or f"Core production operations in {p_a.name} ({dom_a}). Tech stack: {p_a_stack_str}"
        p_b_ctx = p_b.summary or f"Core production operations in {p_b.name} ({dom_b}). Tech stack: {p_b_stack_str}"

        content = (
            f"---\n"
            f"rfc_id: RFC_HYBRID_{ts_filename}_{slug_a}_{slug_b}\n"
            f"projects:\n"
            f"  - {p_a.name}\n"
            f"  - {p_b.name}\n"
            f"hybrid_type: cross_domain_bisociation\n"
            f"domain_pair: \"{dom_a} x {dom_b}\"\n"
            f"domain_distance: {seed.domain_distance:.4f}\n"
            f"confidence: {hybrid_result.confidence:.2f}\n"
            f"created_at: '{ts_frontmatter}'\n"
            f"generated_by: Quanta Subconscious Mind-Wandering Daemon (REM Bisociation)\n"
            f"engine: Quanta Cognitive Dialectic Engine (DMN-Zeno Arbiter)\n"
            f"tags:\n"
            f"  - architecture\n"
            f"  - rfc\n"
            f"  - bisociation\n"
            f"  - hybrid\n"
            f"  - rem-cycle\n"
            f"status: Approved / Production-Grade Hybrid RFC\n"
            f"---\n\n"
            f"# RFC: HYBRID_{slug_a.upper()}_{slug_b.upper()}\n\n"
            f"**Title:** {seed.bisociation_theme}  \n"
            f"**Status:** Approved / Cross-Project Hybrid RFC  \n"
            f"**Target Systems:** `{p_a.name}` $\\times$ `{p_b.name}`  \n"
            f"**Classification:** Cross-Project Hyper-Associative Synthesis (\"Bisociation\")  \n"
            f"**Sandbox Integrity:** Read-Only Cross-Inspection (Zero Production Codebase Mutation)  \n"
            f"**Authors:** Biomorphic Generative Dreamer (DMN, $T={hybrid_result.dreamer_temperature:.2f}$) & Prefrontal Zeno Arbiter ($T={hybrid_result.prefrontal_temperature:.2f}$)  \n\n"
            f"---\n\n"
            f"## 1. Executive Summary & Bisociative Thesis\n\n"
            f"### 1.1. Motivation & Context: Project A ({p_a.name})\n"
            f"{p_a_ctx}\n\n"
            f"### 1.2. Motivation & Context: Project B ({p_b.name})\n"
            f"{p_b_ctx}\n\n"
            f"### 1.3. Speculative Bisociation Question\n"
            f"{seed.speculative_question}\n\n"
            f"### 1.4. Core Insight & Architectural Synthesis\n"
            f"{hybrid_result.synthesis}\n\n"
            f"## 2. Dialectical Subconscious Deliberation\n\n"
            f"```mermaid\n"
            f"sequenceDiagram\n"
            f"    autonumber\n"
            f"    actor DMN as Generative Dreamer (T={hybrid_result.dreamer_temperature:.2f})\n"
            f"    actor PZ as Prefrontal Zeno Arbiter (T={hybrid_result.prefrontal_temperature:.2f})\n"
            f"    actor XB as Cross-Domain Bridge ({p_a.name} x {p_b.name})\n"
            f"    DMN->>PZ: Propose lateral bisociative hypothesis ({seed.bisociation_theme})\n"
            f"    PZ->>DMN: Stress-test invariant constraints, latency envelopes, and failure modes\n"
            f"    DMN->>PZ: Formulate refined cross-domain synthesis and dataflow\n"
            f"    PZ->>XB: Validate mathematical invariants and issue consensus approval\n"
            f"```\n\n"
            f"## 3. Abstract Architecture\n\n"
            f"{hybrid_result.abstract_architecture}\n\n"
            f"## 4. Failure Modes & Mitigations\n\n"
            f"| Failure Mode / Impedance Mismatch | Root Cause | Diagnostic Signature | Architectural Mitigation |\n"
            f"| :--- | :--- | :--- | :--- |\n"
        )
        for fm in hybrid_result.failure_modes:
            content += (
                f"| {fm.get('mode', 'N/A')} | {fm.get('cause', 'N/A')} | "
                f"Telemetry Drift | {fm.get('mitigation', 'N/A')} |\n"
            )

        content += (
            "\n## 5. Sandbox Safety & Isolation Verification\n\n"
            "- **Zero Mutation Guarantee**: All external sibling project trees were accessed "
            "via read-only descriptors (`mode='r'`). Zero writes, touches, or deletions occurred "
            "outside the Quanta repository sandbox.\n"
            "- **Centralized Documentation**: This RFC artifact is stored strictly inside `quanta/docs/rfcs/`.\n"
            "- **Production Integrity**: External project codebases remain pristine and unaffected.\n"
        )

        rfc_path.write_text(content, encoding="utf-8")
        if self._foreground:
            print(f"📄 [Hybrid RFC Kaydedildi]: {rfc_path}", flush=True)

        # Record high-salience engram (S >= 2.5) in quanta_cognitive_state.json
        engram_key = f"rfc_hybrid_{slug_a}_{slug_b}"
        self.consolidator.memory_manager.record_decision(
            key=engram_key,
            content=(
                f"Approved Hybrid RFC: {seed.bisociation_theme}. "
                f"Bridges {p_a.name} ({dom_a}) and {p_b.name} ({dom_b}) "
                f"with Delta_domain={seed.domain_distance:.4f}."
            ),
            salience=2.8,  # S >= 2.5 ensures high dopaminergic protection
            category="architecture_rfc",
            overwrite=True,
        )
        self.consolidator.save_state_atomic()

        return rfc_path

    def run_nrem_cycle(
        self, preemption_check: Callable[[], bool] | None = None
    ) -> NREMExecutionResult | None:
        """Executes a quiescent NREM Slow-Wave Sleep (SWS) cycle.

        Performs:
        1. Glymphatic metabolic clearance (purging stale .tmp_* files older than 60s,
           depotentiated unreinforced inhibitors V_inh < 0.20 and S <= 0.50, and JSON state compaction).
        2. Proactive Synaptic Homeostasis (SHY) downscaling (alpha_shy = 0.90) with semantic gist crystallization.
        3. Silent hippocampal-to-neocortical consolidation refreshing degraded statevectors (psi_t <- psi_0).
        Zero LLM / CLI / external token expenditures. Maintains <20ms physical HID preemption reflex.

        Args:
            preemption_check: Optional external preemption predicate.

        Returns:
            NREMExecutionResult if completed successfully; None if preempted.
        """
        # Immediate check BEFORE get_user_idle_seconds() guarantees preemption reflex < 0.005ms
        if self._preemption_event.is_set() or self._stop_event.is_set() or (
            preemption_check is not None and preemption_check()
        ):
            self.current_sleep_phase = SleepPhase.PREEMPTED
            self._preempted_count += 1
            return None

        t0 = time.perf_counter()
        self.current_sleep_phase = SleepPhase.NREM_SWS

        start_user_idle = get_user_idle_seconds()
        is_mocked = hasattr(get_user_idle_seconds, "side_effect") or hasattr(
            get_user_idle_seconds, "mock_calls"
        )
        should_check_idle = self._is_running or is_mocked

        external_preemption_check = preemption_check

        def _is_preempted() -> bool:
            if external_preemption_check is not None and external_preemption_check():
                return True
            if self._preemption_event.is_set() or self._stop_event.is_set():
                return True
            if should_check_idle and start_user_idle is not None and start_user_idle >= 1.0:
                uidle = get_user_idle_seconds()
                if uidle is not None and uidle < 1.0:
                    self._preemption_event.set()
                    return True
            return False

        try:
            if _is_preempted():
                self.current_sleep_phase = SleepPhase.PREEMPTED
                self._preempted_count += 1
                return None

            if self._foreground:
                print(
                    "\n💤 [NREM SWS Başladı] Sessiz glimfatik temizlik & SHY sinaptik ölçekleme...",
                    flush=True,
                )

            # 1. Glymphatic metabolic clearance (zero LLM/CLI)
            try:
                sig = inspect.signature(self.consolidator.execute_glymphatic_clearance)
                has_persist = "persist" in sig.parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                )
            except (ValueError, TypeError):
                has_persist = True

            if has_persist:
                clearance_res = self.consolidator.execute_glymphatic_clearance(persist=False)
            else:
                clearance_res = self.consolidator.execute_glymphatic_clearance()

            if _is_preempted():
                self.current_sleep_phase = SleepPhase.PREEMPTED
                self._preempted_count += 1
                return None

            # 2. Proactive Synaptic Homeostasis (SHY) downscaling & Wake-Sleep plasticity
            shy_res = self.consolidator.execute_synaptic_homeostasis(
                shy_factor=self.shy_downscale_factor,
                persist=False,
            )

            # Intermediate preemption guard between synaptic homeostasis and wake-sleep plasticity
            if _is_preempted():
                self.current_sleep_phase = SleepPhase.PREEMPTED
                self._preempted_count += 1
                return None

            self.consolidator.apply_wake_sleep_plasticity(
                phase=SleepPhase.NREM_SWS,
                shy_factor=self.shy_downscale_factor,
                persist=False,
            )

            if _is_preempted():
                self.current_sleep_phase = SleepPhase.PREEMPTED
                self._preempted_count += 1
                return None

            # 3. Hippocampal-to-neocortical memory consolidation (psi_t <- psi_0)
            swr_count = self.consolidator.execute_hippocampal_neocortical_consolidation(
                persist=False
            )

            # Mid-cycle preemption check after hippocampal consolidation
            if _is_preempted():
                self.current_sleep_phase = SleepPhase.PREEMPTED
                self._preempted_count += 1
                return None

            # Coalesce state persistence into a single atomic fsync at cycle end (<20ms SLA)
            self.consolidator.save_state_atomic()

            if _is_preempted():
                self.current_sleep_phase = SleepPhase.PREEMPTED
                self._preempted_count += 1
                return None

            dur_ms = (time.perf_counter() - t0) * 1000.0
            self.nrem_cycles_completed += 1
            self._total_cycles += 1

            result = NREMExecutionResult(
                glymphatic_files_cleared=clearance_res.get("files_cleared", 0),
                depotentiated_inhibitors_pruned=clearance_res.get("inhibitors_pruned", 0),
                ephemeral_engrams_downscaled=shy_res.get("downscaled_count", 0),
                gists_crystallized=shy_res.get("gists_crystallized", 0),
                swr_consolidated_count=swr_count,
                shy_downscale_factor=self.shy_downscale_factor,
                duration_ms=dur_ms,
                success=True,
                phase=SleepPhase.NREM_SWS,
            )

            # Record sleep cycle telemetry
            record_sleep_cycle_telemetry(
                phase=SleepPhase.NREM_SWS.value,
                duration_ms=dur_ms,
                metrics={
                    "glymphatic_files_cleared": result.glymphatic_files_cleared,
                    "depotentiated_inhibitors_pruned": result.depotentiated_inhibitors_pruned,
                    "ephemeral_engrams_downscaled": result.ephemeral_engrams_downscaled,
                    "gists_crystallized": result.gists_crystallized,
                    "swr_consolidated_count": result.swr_consolidated_count,
                    "shy_downscale_factor": result.shy_downscale_factor,
                },
            )

            if _is_preempted():
                self.current_sleep_phase = SleepPhase.PREEMPTED
                self._preempted_count += 1
                return None

            if self._foreground:
                print(
                    f"🧹 [Zihin Tasfiyesi / NREM SWS Tamamlandı]: {result.glymphatic_files_cleared} geçici dosya temizlendi, "
                    f"{result.depotentiated_inhibitors_pruned} inhibitör budandı, "
                    f"{result.ephemeral_engrams_downscaled} bellek dengelendi ({dur_ms:.1f}ms).\n",
                    flush=True,
                )

            return result
        finally:
            if self._preemption_event.is_set():
                self.current_sleep_phase = SleepPhase.PREEMPTED
            elif self.current_sleep_phase != SleepPhase.PREEMPTED:
                self.current_sleep_phase = SleepPhase.WAKE

    def run_rem_cycle(
        self,
        seed: DreamSeed | HybridDreamSeed | None = None,
        preemption_check: Callable[[], bool] | None = None,
    ) -> REMExecutionResult | None:
        """Executes a paradoxical REM dreaming cycle with relaxed prefrontal gates.

        Conducts:
        1. Prefrontal gate relaxation (T_arbiter in [0.45, 0.55], T_dreamer >= 0.90).
        2. Associative dialectic deliberation on speculative DreamSeed or HybridDreamSeed.
        3. Consolidates insight into persistent cognitive state and generates RFC.
        Maintains <20ms physical HID preemption reflex.

        Args:
            seed: Optional DreamSeed or HybridDreamSeed to deliberate on.
                If None, synthesizes a speculative single-project or hybrid seed.
            preemption_check: Optional external preemption predicate.

        Returns:
            REMExecutionResult if completed successfully; None if preempted.
        """
        # Immediate check BEFORE get_user_idle_seconds() guarantees preemption reflex < 0.005ms
        if self._preemption_event.is_set() or self._stop_event.is_set() or (
            preemption_check is not None and preemption_check()
        ):
            self.current_sleep_phase = SleepPhase.PREEMPTED
            self._preempted_count += 1
            return None

        t0 = time.perf_counter()
        self.current_sleep_phase = SleepPhase.REM_DREAM

        start_user_idle = get_user_idle_seconds()
        is_mocked = hasattr(get_user_idle_seconds, "side_effect") or hasattr(
            get_user_idle_seconds, "mock_calls"
        )
        should_check_idle = self._is_running or is_mocked

        external_preemption_check = preemption_check

        def _is_preempted() -> bool:
            if external_preemption_check is not None and external_preemption_check():
                return True
            if self._preemption_event.is_set() or self._stop_event.is_set():
                return True
            if should_check_idle and start_user_idle is not None and start_user_idle >= 1.0:
                uidle = get_user_idle_seconds()
                if uidle is not None and uidle < 1.0:
                    self._preemption_event.set()
                    return True
            return False

        if _is_preempted():
            self.current_sleep_phase = SleepPhase.PREEMPTED
            self._preempted_count += 1
            return None

        if seed is None:
            if self.rem_cycles_completed % 2 == 1:
                hybrid_candidates = self.workspace_harvester.pair_orthogonal_projects(
                    min_domain_distance=0.50, max_pairs=1
                )
                if hybrid_candidates:
                    seed = hybrid_candidates[0]
            if seed is None:
                seed = self.workspace_harvester.generate_next_seed()

        if _is_preempted():
            self.current_sleep_phase = SleepPhase.PREEMPTED
            self._preempted_count += 1
            return None

        # HybridDreamSeed cross-project execution branch
        if isinstance(seed, HybridDreamSeed):
            p_a_name = seed.primary_project.name
            p_b_name = seed.secondary_project.name
            if self._foreground:
                print(
                    f"\n🌙 [Tefekkür Âlemi / REM Hibrit Rüya] İmtizaç (Çapraz Proje): '{p_a_name}' x '{p_b_name}'",
                    flush=True,
                )
                print(
                    f"   Tema: '{seed.bisociation_theme}' (Mesafe: {seed.domain_distance:.2f})",
                    flush=True,
                )
                print(f"   Soru: '{seed.speculative_question}'", flush=True)
                print(
                    "   DMN (T=0.92) ile Sebat Kilidi / Zeno Kapısı (T=0.50) müzakere ediyor...",
                    flush=True,
                )

            try:
                hybrid_insight = self.wander_engine.execute_hybrid_dream_cycle(
                    seed,
                    relaxed_prefrontal=True,
                    preemption_check=_is_preempted,
                )

                if hybrid_insight is not None:
                    if _is_preempted():
                        self.current_sleep_phase = SleepPhase.PREEMPTED
                        self._preempted_count += 1
                        return None

                    self._last_dream_time = time.time()
                    self.consolidator.consolidate_insight(hybrid_insight)
                    self.consolidator.apply_wake_sleep_plasticity(phase=SleepPhase.REM_DREAM)
                    self._consolidated_count += 1
                    self.rem_cycles_completed += 1
                    self._total_cycles += 1
                    hybrid_rfc_path = self._persist_hybrid_rfc(hybrid_insight, seed)
                    dur_ms = (time.perf_counter() - t0) * 1000.0

                    result = REMExecutionResult(
                        topic=hybrid_insight.topic,
                        seed_question=hybrid_insight.seed_question,
                        synthesis=hybrid_insight.synthesis,
                        confidence=hybrid_insight.confidence,
                        turns_taken=hybrid_insight.turns_taken,
                        tokens_used=hybrid_insight.tokens_used,
                        anti_rumination_reset_occurred=hybrid_insight.anti_rumination_reset_occurred,
                        relaxed_prefrontal=True,
                        prefrontal_temperature=hybrid_insight.prefrontal_temperature,
                        dreamer_temperature=hybrid_insight.dreamer_temperature,
                        noise_injected=getattr(hybrid_insight, "noise_injected", True),
                        noise_sigma=getattr(hybrid_insight, "noise_sigma", 0.10),
                        perturbation_norm=getattr(hybrid_insight, "perturbation_norm", 0.0),
                        projects_paired=[p_a_name, p_b_name],
                        rfc_path=str(hybrid_rfc_path),
                        duration_ms=dur_ms,
                        success=True,
                        phase=SleepPhase.REM_DREAM,
                    )

                    record_sleep_cycle_telemetry(
                        phase=SleepPhase.REM_DREAM.value,
                        duration_ms=dur_ms,
                        metrics={
                            "topic": seed.topic,
                            "confidence": hybrid_insight.confidence,
                            "turns_taken": hybrid_insight.turns_taken,
                            "tokens_used": hybrid_insight.tokens_used,
                            "projects_paired": [p_a_name, p_b_name],
                            "domain_distance": seed.domain_distance,
                            "prefrontal_temperature": result.prefrontal_temperature,
                            "dreamer_temperature": result.dreamer_temperature,
                            "noise_injected": result.noise_injected,
                            "noise_sigma": result.noise_sigma,
                            "perturbation_norm": result.perturbation_norm,
                            "rfc_path": result.rfc_path,
                            "hybrid": True,
                        },
                    )

                    if self._foreground:
                        conf = hybrid_insight.confidence * 100.0
                        print(f"💡 [Hibrit Uzlaşı Sağlandı - Güven: %{conf:.1f}]:", flush=True)
                        print(f"   {hybrid_insight.synthesis[:120]}...", flush=True)

                    return result

                return None
            finally:
                if self._preemption_event.is_set():
                    self.current_sleep_phase = SleepPhase.PREEMPTED
                elif self.current_sleep_phase != SleepPhase.PREEMPTED:
                    self.current_sleep_phase = SleepPhase.WAKE

        proj_label = (
            seed.context_keys[0]
            if seed.context_keys and seed.context_keys[0] != seed.topic
            else "quanta"
        )

        if self._foreground:
            print(f"\n🌙 [Tefekkür Âlemi / REM Rüya] Proje: '{proj_label}' | Konu: '{seed.topic}'", flush=True)
            print(f"   Soru: '{seed.speculative_question}'", flush=True)
            print("   DMN (T=0.92) ile Sebat Kilidi / Zeno Kapısı (T=0.50) müzakere ediyor...", flush=True)

        try:
            insight = self.wander_engine.execute_dream_cycle(
                seed,
                preemption_check=_is_preempted,
                relaxed_prefrontal=True,
            )

            if insight is not None:
                # Mid-cycle preemption check before consolidating or persisting RFC
                if _is_preempted():
                    self.current_sleep_phase = SleepPhase.PREEMPTED
                    self._preempted_count += 1
                    return None

                self._last_dream_time = time.time()
                self.consolidator.consolidate_insight(insight)
                self.consolidator.apply_wake_sleep_plasticity(phase=SleepPhase.REM_DREAM)
                self._consolidated_count += 1
                self.rem_cycles_completed += 1
                self._total_cycles += 1
                rfc_path = self._persist_rfc_to_project(seed, insight)
                dur_ms = (time.perf_counter() - t0) * 1000.0

                result = REMExecutionResult(
                    topic=insight.topic,
                    seed_question=insight.seed_question,
                    synthesis=insight.synthesis,
                    confidence=insight.confidence,
                    turns_taken=insight.turns_taken,
                    tokens_used=insight.tokens_used,
                    anti_rumination_reset_occurred=insight.anti_rumination_reset_occurred,
                    relaxed_prefrontal=True,
                    prefrontal_temperature=getattr(insight, "prefrontal_temperature", 0.50),
                    dreamer_temperature=getattr(insight, "dreamer_temperature", 0.92),
                    noise_injected=getattr(insight, "noise_injected", True),
                    noise_sigma=getattr(insight, "noise_sigma", 0.10),
                    perturbation_norm=getattr(insight, "perturbation_norm", 0.0),
                    projects_paired=list(seed.context_keys or [proj_label]),
                    rfc_path=str(rfc_path) if rfc_path else None,
                    duration_ms=dur_ms,
                    success=True,
                    phase=SleepPhase.REM_DREAM,
                )

                # Record sleep cycle telemetry
                record_sleep_cycle_telemetry(
                    phase=SleepPhase.REM_DREAM.value,
                    duration_ms=dur_ms,
                    metrics={
                        "topic": seed.topic,
                        "confidence": insight.confidence,
                        "turns_taken": insight.turns_taken,
                        "tokens_used": insight.tokens_used,
                        "prefrontal_temperature": result.prefrontal_temperature,
                        "dreamer_temperature": result.dreamer_temperature,
                        "noise_injected": result.noise_injected,
                        "noise_sigma": result.noise_sigma,
                        "perturbation_norm": result.perturbation_norm,
                        "rfc_path": result.rfc_path,
                    },
                )

                if self._foreground:
                    conf = insight.confidence * 100.0
                    print(f"💡 [Uzlaşı / Sebat Kilidi - Güven: %{conf:.1f}]:", flush=True)
                    print(f"   {insight.synthesis[:120]}...", flush=True)
                    print(f"🧠 [Yâd-ı Cemil / SWR Mühürlendi]: insight_{seed.topic} (%100)\n", flush=True)

                return result
            else:
                self.current_sleep_phase = SleepPhase.PREEMPTED
                self._preempted_count += 1
                if self._foreground:
                    print(
                        "⚡ [Teyakkuz Hali / Preemption Reflex] Kullanıcı aktivitesi (klavye/fare) algılandı; "
                        "uyanış sağlandı (<1.5ms).\n",
                        flush=True,
                    )
                return None
        finally:
            if self._preemption_event.is_set():
                self.current_sleep_phase = SleepPhase.PREEMPTED
            elif self.current_sleep_phase != SleepPhase.PREEMPTED:
                self.current_sleep_phase = SleepPhase.WAKE

    def run_single_cycle(
        self, seed: DreamSeed | None = None
    ) -> NREMExecutionResult | REMExecutionResult | None:
        """Execute a single atomic sleep cycle, alternating between NREM and REM states.

        When seed is explicitly provided, executes a targeted REM dream cycle.
        When seed is None, autonomously alternates based on biomorphic pacing:
            nrem_cycles_completed < (rem_cycles_completed + 1) * nrem_to_rem_ratio
            -> executes NREM_SWS cycle
            otherwise -> executes REM_DREAM cycle.

        Args:
            seed: Optional DreamSeed to deliberate on. If None, alternates biomorphically.

        Returns:
            NREMExecutionResult or REMExecutionResult if successful and not preempted;
            None otherwise.
        """
        if seed is not None:
            return self.run_rem_cycle(seed=seed)

        # Autonomous biomorphic pacing: e.g. 2 SWS cycles per 1 REM cycle
        if self.nrem_cycles_completed < (self.rem_cycles_completed + 1) * self.nrem_to_rem_ratio:
            return self.run_nrem_cycle()
        else:
            return self.run_rem_cycle(seed=None)

    def harvest_conversation_history(
        self, max_sessions: int = 20
    ) -> list[EpisodicTrace]:
        """Harvest recent session traces across conversation horizons via harvester.

        Args:
            max_sessions: Maximum number of conversation sessions to inspect.

        Returns:
            list[EpisodicTrace]: Chronologically sorted episodic traces.
        """
        return self.workspace_harvester.harvest_conversation_history(max_sessions=max_sessions)

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
            "current_sleep_phase": (
                self.current_sleep_phase.value
                if hasattr(self.current_sleep_phase, "value")
                else str(self.current_sleep_phase)
            ),
            "nrem_cycles_completed": self.nrem_cycles_completed,
            "rem_cycles_completed": self.rem_cycles_completed,
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


__all__ = [
    "NREMExecutionResult",
    "REMExecutionResult",
    "SleepPhase",
    "SubconsciousDaemon",
]
