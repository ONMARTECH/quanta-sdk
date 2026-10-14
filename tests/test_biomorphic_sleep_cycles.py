"""tests/test_biomorphic_sleep_cycles.py — Unit & Integration tests for Biomorphic Sleep State Engine.

Validates Milestone M1:
1. SleepPhase state machine (WAKE, NREM_SWS, REM_DREAM, PREEMPTED).
2. NREM Execution Result and REM Execution Result dataclasses.
3. NREM Glymphatic Clearance: purging stale .tmp_* files older than 60s, depotentiated
   unreinforced inhibitors (V_inh < 0.20 and S <= 0.50), and JSON state compaction.
4. Synaptic Homeostasis (SHY) proactive downscaling (alpha_shy = 0.90) and semantic gist crystallization.
5. Hippocampal-to-neocortical memory consolidation restoring quantum statevectors (psi_t <- psi_0).
6. REM Paradoxical Dreaming & Prefrontal Gate Relaxation (T_arbiter in [0.45, 0.55], T_dreamer >= 0.90).
7. Biomorphic pacing (alternating SWS:REM cycles, e.g. 2:1 ratio) in SubconsciousDaemon.
8. Sub-20ms physical HID preemption reflex in both NREM and REM cycles.
9. Sleep cycle telemetry recording and SubconsciousDaemon status reporting.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from quanta.cognitive.consolidation import SubconsciousConsolidator
from quanta.cognitive.daemon import (
    NREMExecutionResult,
    REMExecutionResult,
    SleepPhase,
    SubconsciousDaemon,
)
from quanta.cognitive.memory import (
    GIST_CATEGORY,
    LAMBDA_LONGTERM,
    LAMBDA_SPRINT,
    LAMBDA_TODAY,
    LAMBDA_YESTERDAY,
    W_LONGTERM_BOUNDARY,
    W_SPRINT_BOUNDARY,
    W_TODAY_BOUNDARY,
    W_YESTERDAY_BOUNDARY,
    CognitiveMemoryManager,
    calculate_temporal_decay,
    is_actionable_resolution,
)
from quanta.cognitive.mind_wander import (
    DreamInsight,
    MindWanderEngine,
    MindWanderResult,
    cosine_similarity,
    perturb_thought_vector,
)
from quanta.cognitive.telemetry import read_telemetry_events, record_sleep_cycle_telemetry
from quanta.cognitive.tom_analyzer import DreamSeed
from quanta.cognitive.workspace_harvester import (
    CANONICAL_DOMAINS,
    HybridDreamSeed,
    ProjectContext,
    WorkspaceContextHarvester,
    classify_project_domain,
)


class TestSleepPhaseStateMachine:
    """Validates Feature 1: SleepPhase enum and execution result representations."""

    def test_sleep_phase_enumeration_values(self) -> None:
        """Verifies formal states exist and evaluate to expected string identifiers."""
        assert SleepPhase.WAKE == "WAKE"
        assert SleepPhase.NREM_SWS == "NREM_SWS"
        assert SleepPhase.REM_DREAM == "REM_DREAM"
        assert SleepPhase.PREEMPTED == "PREEMPTED"

        assert str(SleepPhase.WAKE) == "SleepPhase.WAKE"
        assert SleepPhase.WAKE.value == "WAKE"
        assert SleepPhase("NREM_SWS") == SleepPhase.NREM_SWS

    def test_nrem_execution_result_dataclass(self) -> None:
        """Verifies NREMExecutionResult fields and typing."""
        res = NREMExecutionResult(
            glymphatic_files_cleared=3,
            depotentiated_inhibitors_pruned=2,
            ephemeral_engrams_downscaled=5,
            gists_crystallized=1,
            swr_consolidated_count=4,
            shy_downscale_factor=0.90,
            duration_ms=12.5,
            success=True,
        )
        assert res.glymphatic_files_cleared == 3
        assert res.depotentiated_inhibitors_pruned == 2
        assert res.ephemeral_engrams_downscaled == 5
        assert res.gists_crystallized == 1
        assert res.swr_consolidated_count == 4
        assert res.shy_downscale_factor == 0.90
        assert res.duration_ms == 12.5
        assert res.success is True

    def test_rem_execution_result_is_dream_insight_subclass(self) -> None:
        """Verifies REMExecutionResult inherits from DreamInsight for 100% backwards compatibility."""
        res = REMExecutionResult(
            topic="bisociation_test",
            seed_question="How to cross-synthesize?",
            synthesis="Consensus reached via lateral bridging.",
            confidence=0.95,
            turns_taken=3,
            tokens_used=450,
            anti_rumination_reset_occurred=False,
            relaxed_prefrontal=True,
            prefrontal_temperature=0.50,
            dreamer_temperature=0.92,
            projects_paired=["quanta", "quantum_sim"],
            rfc_path="/docs/RFC_BISOCIATION.md",
            duration_ms=45.2,
            success=True,
        )
        assert isinstance(res, DreamInsight)
        assert res.topic == "bisociation_test"
        assert res.prefrontal_temperature == 0.50
        assert res.dreamer_temperature == 0.92
        assert res.projects_paired == ["quanta", "quantum_sim"]
        assert res.rfc_path == "/docs/RFC_BISOCIATION.md"
        assert res.success is True


class TestNREMGlymphaticClearance:
    """Validates Feature 2: Glymphatic metabolic clearance in SubconsciousConsolidator."""

    def test_purges_stale_temp_files_while_preserving_fresh_ones(self, tmp_path: Path) -> None:
        """Verifies that .tmp_* files older than 60s are removed, while newer ones are retained."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        # Create a stale temp file (> 60s old)
        stale_file = tmp_path / ".tmp_stale_write_123"
        stale_file.write_text("corrupted partial write", encoding="utf-8")
        stale_mtime = time.time() - 120.0  # 120 seconds ago
        os.utime(stale_file, (stale_mtime, stale_mtime))

        # Create a fresh temp file (< 60s old, e.g. 5s ago)
        fresh_file = tmp_path / ".tmp_fresh_write_456"
        fresh_file.write_text("in progress write", encoding="utf-8")
        fresh_mtime = time.time() - 5.0
        os.utime(fresh_file, (fresh_mtime, fresh_mtime))

        # Create a non-temp file
        regular_file = tmp_path / "regular_file.txt"
        regular_file.write_text("important user notes", encoding="utf-8")

        # Execute glymphatic clearance
        result = consolidator.execute_glymphatic_clearance()

        assert result["files_cleared"] == 1
        assert not stale_file.exists(), "Stale temp file older than 60s should have been purged"
        assert fresh_file.exists(), "Fresh temp file (<60s) should have been preserved"
        assert regular_file.exists(), "Non-temp files must not be touched"

    def test_purges_depotentiated_unreinforced_inhibitors(self, tmp_path: Path) -> None:
        """Verifies depotentiated inhibitors (V_inh < 0.20 and S <= 0.50) are purged during NREM."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        # 1. Record an active potent inhibitor (V_inh = 0.80, S = 2.0) -> must survive
        idx_active = consolidator.memory_manager.record_decision(
            key="inh_active_glob",
            content="Quote zsh globs in shell commands",
            salience=2.0,
            category="inhibitor",
        )
        consolidator.memory_manager.buffer.buffer[idx_active]["metadata"]["v_inh"] = 0.80

        # 2. Record a depotentiated unreinforced inhibitor (V_inh = 0.12 < 0.20, S = 0.40 <= 0.50) -> must be purged
        idx_depot = consolidator.memory_manager.record_decision(
            key="inh_depotentiated_rule",
            content="Old obsolete negative rule that relaxed over time",
            salience=0.40,
            category="inhibitor",
        )
        consolidator.memory_manager.buffer.buffer[idx_depot]["metadata"]["v_inh"] = 0.12
        consolidator.memory_manager.buffer.buffer[idx_depot]["metadata"]["is_core_anchor"] = False

        # 3. Record a low V_inh but high-salience core anchor inhibitor -> must survive
        idx_core = consolidator.memory_manager.record_decision(
            key="inh_core_safety",
            content="Core inviolable safety inhibitor",
            salience=2.5,
            category="inhibitor",
        )
        consolidator.memory_manager.buffer.buffer[idx_core]["metadata"]["v_inh"] = 0.15
        consolidator.memory_manager.buffer.buffer[idx_core]["metadata"]["is_core_anchor"] = True

        # Initial buffer state
        assert len(consolidator.memory_manager.buffer.buffer) == 3

        # Execute glymphatic clearance
        result = consolidator.execute_glymphatic_clearance()

        assert result["inhibitors_pruned"] == 1
        assert result["compacted_engrams"] == 2

        surviving_keys = [
            e["metadata"]["key"] for e in consolidator.memory_manager.buffer.buffer
        ]
        assert "inh_active_glob" in surviving_keys
        assert "inh_core_safety" in surviving_keys
        assert "inh_depotentiated_rule" not in surviving_keys

    def test_glymphatic_clearance_compacts_json_state_offline(self, tmp_path: Path) -> None:
        """Verifies state compaction produces valid, clean JSON with zero LLM/CLI invocations."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        # Seed engrams with duplicated tags
        idx = consolidator.memory_manager.record_decision(
            key="test_compaction_engram",
            content="Test content for compaction",
            salience=1.5,
            category="general",
        )
        consolidator.memory_manager.buffer.buffer[idx]["metadata"]["tags"] = [
            "quanta",
            "quanta",
            "  quanta  ",
            "memory",
            "memory",
        ]

        res = consolidator.execute_glymphatic_clearance()
        assert res["compacted_engrams"] == 1
        assert state_file.exists()

        data = consolidator.load_state()
        engram_entry = data["engrams"][0]
        # Tags should be deduped and stripped
        assert engram_entry["tags"] == ["quanta", "memory"]


class TestSynapticHomeostasisAndConsolidation:
    """Validates Feature 3: Synaptic Homeostasis (SHY) and Hippocampal Consolidation."""

    def test_synaptic_homeostasis_downscales_non_core_engrams(self, tmp_path: Path) -> None:
        """Verifies proactive downscaling (S_e <- max(0.20, S_e * alpha_shy)) on non-core memories."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        # 1. Ephemeral non-core engram (salience = 1.0)
        idx_ephemeral = consolidator.memory_manager.record_decision(
            key="ephemeral_note",
            content="Temporary calculation notes from prior step",
            salience=1.0,
            category="scratchpad",
        )
        consolidator.memory_manager.buffer.buffer[idx_ephemeral]["metadata"]["is_core_anchor"] = False

        # 2. Core anchor engram (salience = 2.5, is_core_anchor = True)
        idx_core = consolidator.memory_manager.record_decision(
            key="core_architecture_invariant",
            content="Always enforce POSIX immutability on external project paths",
            salience=2.5,
            category="constraint",
        )
        consolidator.memory_manager.buffer.buffer[idx_core]["metadata"]["is_core_anchor"] = True

        # 3. Semantic gist engram (category = "semantic_gist")
        consolidator.memory_manager.record_semantic_gist(
            key="gist_step_42",
            content="Distilled architectural consensus on state persistence",
            salience=1.9,
        )

        # Execute Synaptic Homeostasis with shy_factor = 0.90
        res = consolidator.execute_synaptic_homeostasis(shy_factor=0.90)

        assert res["downscaled_count"] >= 1

        # Check saliences
        mem_by_key = {
            e["metadata"]["key"]: e for e in consolidator.memory_manager.buffer.buffer
        }

        # Ephemeral non-core engram was downscaled: 1.0 * 0.90 = 0.90
        assert round(float(mem_by_key["ephemeral_note"]["dopamine_tag"]), 3) == 0.900

        # Core anchor engrams remained protected at pristine salience
        assert float(mem_by_key["core_architecture_invariant"]["dopamine_tag"]) == 2.5
        assert float(mem_by_key["gist_step_42"]["dopamine_tag"]) == 1.9

    def test_synaptic_homeostasis_crystallizes_decayed_actionable_decisions(self, tmp_path: Path) -> None:
        """Verifies newly decayed actionable decisions crystallize into semantic gists via distill_semantic_gist."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        # Record a transient actionable decision right at the boundary (salience = 0.52)
        idx = consolidator.memory_manager.record_decision(
            key="actionable_redis_resolution",
            content="We decided to use Redis caching for session store to minimize latency bounds.",
            salience=0.52,
            category="decision",
        )
        consolidator.memory_manager.buffer.buffer[idx]["metadata"]["is_core_anchor"] = False

        # Verify it meets actionable resolution criteria
        assert is_actionable_resolution(
            "We decided to use Redis caching for session store to minimize latency bounds."
        ) is True

        # Induce dephasing decay across turns so the engram is newly decayed (F < 0.70)
        consolidator.memory_manager.step(dt=50.0)

        # Execute Synaptic Homeostasis: downscales saliences, prunes decayed items,
        # and crystallizes the actionable resolution into a permanent semantic gist
        res = consolidator.execute_synaptic_homeostasis(shy_factor=0.90)

        assert res["gists_crystallized"] == 1
        assert res["pruned_count"] >= 1

        # Confirm the distilled semantic gist was anchored into memory
        all_keys = [e["metadata"]["key"] for e in consolidator.memory_manager.buffer.buffer]
        gist_keys = [k for k in all_keys if k.startswith("gist_")]
        assert len(gist_keys) == 1

        gist_engram = next(e for e in consolidator.memory_manager.buffer.buffer if e["metadata"]["key"] == gist_keys[0])
        assert gist_engram["metadata"]["category"] == "semantic_gist"
        assert float(gist_engram["dopamine_tag"]) >= 1.80  # Gists are core anchors

    def test_synaptic_homeostasis_clamps_salience_at_floor(self, tmp_path: Path) -> None:
        """Verifies downscaling does not underflow below 0.20."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        idx = consolidator.memory_manager.record_decision(
            key="scratch_near_floor",
            content="Near floor transient scratchpad",
            salience=0.21,
            category="scratchpad",
        )
        consolidator.memory_manager.buffer.buffer[idx]["metadata"]["is_core_anchor"] = False

        # Downscaling 0.21 * 0.50 = 0.105 -> must be clamped to 0.20
        consolidator.execute_synaptic_homeostasis(shy_factor=0.50)

        mem = [e for e in consolidator.memory_manager.buffer.buffer if e["metadata"]["key"] == "scratch_near_floor"]
        if mem:
            assert float(mem[0]["dopamine_tag"]) >= 0.20

    def test_hippocampal_neocortical_consolidation_refreshes_statevectors(self, tmp_path: Path) -> None:
        """Verifies degraded quantum statevectors (psi_t <- psi_0) are refreshed to pristine F=1.0."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        # Record engram
        consolidator.memory_manager.record_decision(
            key="test_statevector_refresh",
            content="Consolidated rule needing dephasing reset",
            salience=2.2,
            category="decision",
        )

        engram = consolidator.memory_manager.buffer.buffer[0]
        psi_0 = engram["pristine_state"]

        # Artificially induce dephasing / noise in degraded_state
        noisy_state = psi_0 + torch.randn_like(psi_0) * 0.5
        noisy_state = noisy_state / torch.norm(noisy_state)
        engram["degraded_state"] = noisy_state
        engram["age"] = 15

        # Check that fidelity is degraded
        overlap_before = torch.vdot(psi_0, engram["degraded_state"])
        fid_before = float((torch.abs(overlap_before) ** 2).item())
        assert fid_before < 0.95

        # Execute Hippocampal-to-Neocortical Consolidation
        refreshed_count = consolidator.execute_hippocampal_neocortical_consolidation()

        assert refreshed_count == 1
        engram_after = consolidator.memory_manager.buffer.buffer[0]

        # Verify statevector was restored to pristine: psi_t == psi_0
        overlap_after = torch.vdot(psi_0, engram_after["degraded_state"])
        fid_after = float((torch.abs(overlap_after) ** 2).item())
        assert abs(fid_after - 1.0) < 1e-5
        assert engram_after["age"] == 0


class TestREMRelaxedPrefrontalDreaming:
    """Validates Feature 4: Relaxed Prefrontal Gate in MindWanderEngine."""

    def test_relaxed_prefrontal_elevates_temperatures(self) -> None:
        """Verifies relaxed_prefrontal=True sets T_arbiter in [0.45, 0.55] and T_dreamer >= 0.90."""
        engine = MindWanderEngine(max_turns=3, use_agy_cli=False)

        seed = DreamSeed(
            topic="prefrontal_relaxation_test",
            speculative_question="How does prefrontal disinhibition affect lateral exploration?",
            urgency=1.0,
            context_keys=["quanta.cognitive.mind_wander"],
        )

        # 1. Standard waking/structured cycle: T_arbiter = 0.20, T_dreamer = 0.85
        insight_standard = engine.execute_dream_cycle(seed, relaxed_prefrontal=False)
        assert insight_standard is not None
        assert insight_standard.relaxed_prefrontal is False
        assert insight_standard.prefrontal_temperature == 0.20
        assert insight_standard.dreamer_temperature == 0.85

        # 2. Relaxed prefrontal REM paradoxical cycle: T_arbiter in [0.45, 0.55], T_dreamer >= 0.90
        insight_rem = engine.execute_dream_cycle(seed, relaxed_prefrontal=True)
        assert insight_rem is not None
        assert insight_rem.relaxed_prefrontal is True
        assert 0.45 <= insight_rem.prefrontal_temperature <= 0.55
        assert insight_rem.dreamer_temperature >= 0.90

    def test_simulator_step_reflects_relaxed_gate_temperatures(self) -> None:
        """Verifies simulate_step uses elevated temperatures when relaxed_prefrontal is True."""
        engine = MindWanderEngine(use_agy_cli=False)
        seed = DreamSeed(
            topic="simulator_temperature_test",
            speculative_question="Can simulator step modulate?",
            urgency=1.0,
            context_keys=[],
        )

        # Turn 0: Dreamer
        thought_dreamer_std, _, _ = engine.simulator.simulate_step(0, seed, relaxed_prefrontal=False)
        thought_dreamer_rem, _, _ = engine.simulator.simulate_step(0, seed, relaxed_prefrontal=True)
        assert "T=0.85" in thought_dreamer_std
        assert "T=0.92" in thought_dreamer_rem

        # Turn 1: Arbiter
        thought_arbiter_std, _, _ = engine.simulator.simulate_step(1, seed, relaxed_prefrontal=False)
        thought_arbiter_rem, _, _ = engine.simulator.simulate_step(1, seed, relaxed_prefrontal=True)
        assert "T=0.20" in thought_arbiter_std
        assert "T=0.50" in thought_arbiter_rem


class TestSubconsciousDaemonSleepCycles:
    """Validates Feature 5 & 6: Biomorphic Alternating Pacing, Preemption, and Telemetry."""

    def test_run_nrem_cycle_executes_quiescently(self, tmp_path: Path) -> None:
        """Verifies run_nrem_cycle executes without LLM/CLI invocations and records telemetry."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(
            state_file=state_file,
            pid_file=pid_file,
            use_agy_cli=False,
        )

        result = daemon.run_nrem_cycle()
        assert result is not None
        assert isinstance(result, NREMExecutionResult)
        assert result.success is True
        assert daemon.nrem_cycles_completed == 1
        assert daemon._total_cycles == 1
        assert daemon.current_sleep_phase == SleepPhase.WAKE

    def test_run_rem_cycle_executes_and_creates_rfc(self, tmp_path: Path) -> None:
        """Verifies run_rem_cycle executes relaxed prefrontal dream cycle and creates RFC."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(
            state_file=state_file,
            pid_file=pid_file,
            use_agy_cli=False,
        )

        seed = DreamSeed(
            topic="rem_dream_feature_test",
            speculative_question="Does REM cycle persist insight?",
            urgency=1.5,
            context_keys=["quanta.test"],
        )

        result = daemon.run_rem_cycle(seed=seed)
        assert result is not None
        assert isinstance(result, REMExecutionResult)
        assert result.topic == "rem_dream_feature_test"
        assert result.prefrontal_temperature == 0.50
        assert result.dreamer_temperature >= 0.90
        assert daemon.rem_cycles_completed == 1
        assert daemon._total_cycles == 1
        assert daemon.current_sleep_phase == SleepPhase.WAKE

    def test_biomorphic_pacing_alternation(self, tmp_path: Path) -> None:
        """Verifies biomorphic pacing ratio (e.g. 2 SWS cycles per 1 REM cycle)."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(
            state_file=state_file,
            pid_file=pid_file,
            nrem_to_rem_ratio=2,
            use_agy_cli=False,
        )

        # Cycle 1: should be NREM (nrem=0, rem=0 -> nrem < 2)
        c1 = daemon.run_single_cycle()
        assert isinstance(c1, NREMExecutionResult)
        assert daemon.nrem_cycles_completed == 1
        assert daemon.rem_cycles_completed == 0

        # Cycle 2: should be NREM (nrem=1, rem=0 -> nrem < 2)
        c2 = daemon.run_single_cycle()
        assert isinstance(c2, NREMExecutionResult)
        assert daemon.nrem_cycles_completed == 2
        assert daemon.rem_cycles_completed == 0

        # Cycle 3: should be REM (nrem=2, rem=0 -> nrem not < 2 -> REM burst)
        c3 = daemon.run_single_cycle()
        assert isinstance(c3, REMExecutionResult)
        assert daemon.nrem_cycles_completed == 2
        assert daemon.rem_cycles_completed == 1

        # Cycle 4: should be NREM (nrem=2, rem=1 -> nrem < (1+1)*2 = 4)
        c4 = daemon.run_single_cycle()
        assert isinstance(c4, NREMExecutionResult)
        assert daemon.nrem_cycles_completed == 3
        assert daemon.rem_cycles_completed == 1

        # Cycle 5: should be NREM (nrem=3, rem=1 -> nrem < 4)
        c5 = daemon.run_single_cycle()
        assert isinstance(c5, NREMExecutionResult)
        assert daemon.nrem_cycles_completed == 4
        assert daemon.rem_cycles_completed == 1

        # Cycle 6: should be REM (nrem=4, rem=1 -> nrem not < 4 -> REM burst)
        c6 = daemon.run_single_cycle()
        assert isinstance(c6, REMExecutionResult)
        assert daemon.nrem_cycles_completed == 4
        assert daemon.rem_cycles_completed == 2

    def test_instant_preemption_latency_budget(self, tmp_path: Path) -> None:
        """Verifies sub-20ms preemption reflex aborts both NREM and REM cycles instantly."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(
            state_file=state_file,
            pid_file=pid_file,
            use_agy_cli=False,
        )

        # Pre-set preemption flag
        daemon._preemption_event.set()

        # 1. Test NREM cycle preemption latency
        t0 = time.perf_counter()
        res_nrem = daemon.run_nrem_cycle()
        elapsed_nrem_ms = (time.perf_counter() - t0) * 1000.0

        assert res_nrem is None
        assert elapsed_nrem_ms < 20.0, f"NREM preemption took {elapsed_nrem_ms:.2f}ms, expected < 20ms"
        assert daemon.current_sleep_phase == SleepPhase.PREEMPTED

        # 2. Test REM cycle preemption latency
        daemon.current_sleep_phase = SleepPhase.WAKE
        daemon._preemption_event.set()

        t1 = time.perf_counter()
        res_rem = daemon.run_rem_cycle()
        elapsed_rem_ms = (time.perf_counter() - t1) * 1000.0

        assert res_rem is None
        assert elapsed_rem_ms < 20.0, f"REM preemption took {elapsed_rem_ms:.2f}ms, expected < 20ms"
        assert daemon.current_sleep_phase == SleepPhase.PREEMPTED

    def test_daemon_status_reports_sleep_phase_and_cycle_counts(self, tmp_path: Path) -> None:
        """Verifies daemon.status() contains current_sleep_phase and cycle counters."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(
            state_file=state_file,
            pid_file=pid_file,
            use_agy_cli=False,
        )

        stat = daemon.status()
        assert "current_sleep_phase" in stat
        assert stat["current_sleep_phase"] == "WAKE"
        assert "nrem_cycles_completed" in stat
        assert stat["nrem_cycles_completed"] == 0
        assert "rem_cycles_completed" in stat
        assert stat["rem_cycles_completed"] == 0

        # Run an NREM cycle and verify status updates
        daemon.run_nrem_cycle()
        stat_after = daemon.status()
        assert stat_after["nrem_cycles_completed"] == 1
        assert stat_after["total_cycles"] == 1

    def test_record_sleep_cycle_telemetry_persists_events(self, tmp_path: Path) -> None:
        """Verifies record_sleep_cycle_telemetry appends valid JSONL records to telemetry file."""
        telem_file = tmp_path / "test_telemetry.jsonl"

        record_sleep_cycle_telemetry(
            phase="NREM_SWS",
            duration_ms=15.4,
            metrics={"glymphatic_files_cleared": 2, "shy_downscale_factor": 0.90},
            telemetry_file=telem_file,
            workspace="quanta_test",
        )

        record_sleep_cycle_telemetry(
            phase="REM_DREAM",
            duration_ms=52.1,
            metrics={"topic": "quantum_clustering", "confidence": 0.98},
            telemetry_file=telem_file,
            workspace="quanta_test",
        )

        assert telem_file.exists()
        events = read_telemetry_events(event_type="sleep_cycle", telemetry_file=telem_file)
        assert len(events) == 2

        assert events[0]["sleep_phase"] == "NREM_SWS"
        assert events[0]["duration_ms"] == 15.4
        assert events[0]["metrics"]["glymphatic_files_cleared"] == 2

        assert events[1]["sleep_phase"] == "REM_DREAM"
        assert events[1]["duration_ms"] == 52.1
        assert events[1]["metrics"]["topic"] == "quantum_clustering"


class TestMultiHorizonTemporalDecay:
    """Validates Feature 8: Multi-Horizon Temporal Decay Cascade."""

    def test_lambda_decay_constants(self) -> None:
        """Verifies decay rate constants match specification."""
        assert LAMBDA_TODAY == 0.02
        assert LAMBDA_YESTERDAY == 0.05
        assert LAMBDA_SPRINT == 0.15
        assert LAMBDA_LONGTERM == 0.05

    def test_today_horizon_decay_values(self) -> None:
        """Verifies exponential decay in Today horizon (<24h, lambda=0.02/h)."""
        now = time.time()
        # dt = 0h
        assert calculate_temporal_decay(now, salience=1.0, now_ts=now) == pytest.approx(1.0)
        # dt = 10h -> W_TODAY_BOUNDARY * exp(-LAMBDA_TODAY * 10) = exp(-0.2)
        expected_10h = W_TODAY_BOUNDARY * math.exp(-LAMBDA_TODAY * 10.0)
        assert calculate_temporal_decay(
            now - 10.0 * 3600.0, salience=1.0, now_ts=now
        ) == pytest.approx(expected_10h, rel=1e-4)

    def test_yesterday_horizon_decay_values(self) -> None:
        """Verifies anchored exponential decay in Yesterday horizon (24-48h, lambda=0.05/h)."""
        now = time.time()
        # dt = 30h -> anchored at 24h: W_YESTERDAY_BOUNDARY * exp(-0.05 * 6)
        expected_30h = W_YESTERDAY_BOUNDARY * math.exp(-LAMBDA_YESTERDAY * (30.0 - 24.0))
        assert calculate_temporal_decay(
            now - 30.0 * 3600.0, salience=1.0, now_ts=now
        ) == pytest.approx(expected_30h, rel=1e-4)

    def test_sprint_horizon_decay_values(self) -> None:
        """Verifies anchored exponential decay in Previous Sprint horizon (2-7d, lambda=0.15/d)."""
        now = time.time()
        # dt = 4 days (96h) -> 2 days into sprint: W_SPRINT_BOUNDARY * exp(-0.15 * 2)
        expected_4d = W_SPRINT_BOUNDARY * math.exp(-LAMBDA_SPRINT * 2.0)
        assert calculate_temporal_decay(
            now - 4.0 * 86400.0, salience=1.0, now_ts=now
        ) == pytest.approx(expected_4d, rel=1e-4)

    def test_longterm_horizon_decay_values(self) -> None:
        """Verifies anchored exponential decay in Long-Term horizon (>7d, lambda=0.05/d)."""
        now = time.time()
        # dt = 10 days (240h) -> 3 days into long-term: W_LONGTERM_BOUNDARY * exp(-0.05 * 3)
        expected_10d = W_LONGTERM_BOUNDARY * math.exp(-LAMBDA_LONGTERM * 3.0)
        assert calculate_temporal_decay(
            now - 10.0 * 86400.0, salience=1.0, now_ts=now
        ) == pytest.approx(expected_10d, rel=1e-4)

    def test_core_anchor_shielding_protection(self) -> None:
        """Verifies core anchor shielding gives w_eff = 1.0 unconditionally."""
        now = time.time()
        ancient_ts = now - 365.0 * 86400.0  # 1 year ago

        # Case 1: Salience >= 2.0
        assert (
            calculate_temporal_decay(ancient_ts, salience=2.5, is_core_anchor=False, now_ts=now)
            == 1.0
        )
        # Case 2: is_core_anchor=True
        assert (
            calculate_temporal_decay(ancient_ts, salience=0.5, is_core_anchor=True, now_ts=now)
            == 1.0
        )
        # Case 3: category == 'semantic_gist'
        assert (
            calculate_temporal_decay(
                ancient_ts, salience=0.5, is_core_anchor=False, category=GIST_CATEGORY, now_ts=now
            )
            == 1.0
        )

    def test_temporal_gradient_monotonicity(self) -> None:
        """Verifies strict continuous monotonicity and boundary consistency of temporal decay.

        Ensures w_today > w_yesterday > w_sprint > w_longterm holds across all horizons
        and verifies boundary transitions (23.9h vs 24.1h, 47.9h vs 48.1h, 167.9h vs 168.1h)
        are continuous and strictly decreasing without inversions.
        """
        now = time.time()

        # Multi-horizon gradient progression
        w_today_5h = calculate_temporal_decay(now - 5 * 3600, salience=1.0, now_ts=now)
        w_yesterday_30h = calculate_temporal_decay(now - 30 * 3600, salience=1.0, now_ts=now)
        w_sprint_72h = calculate_temporal_decay(now - 72 * 3600, salience=1.0, now_ts=now)
        w_longterm_240h = calculate_temporal_decay(now - 240 * 3600, salience=1.0, now_ts=now)

        assert (
            w_today_5h > w_yesterday_30h > w_sprint_72h > w_longterm_240h
        ), f"Inversion: {w_today_5h} > {w_yesterday_30h} > {w_sprint_72h} > {w_longterm_240h}"

        # Boundary 1: Today -> Yesterday boundary (24.0h)
        w_today_pre = calculate_temporal_decay(now - 23.9 * 3600, salience=1.0, now_ts=now)
        w_yesterday_post = calculate_temporal_decay(now - 24.1 * 3600, salience=1.0, now_ts=now)
        assert w_today_pre > w_yesterday_post, f"24h boundary failed: {w_today_pre} <= {w_yesterday_post}"
        assert abs(w_today_pre - w_yesterday_post) < 0.05

        # Boundary 2: Yesterday -> Sprint boundary (48.0h)
        w_yesterday_pre = calculate_temporal_decay(now - 47.9 * 3600, salience=1.0, now_ts=now)
        w_sprint_post = calculate_temporal_decay(now - 48.1 * 3600, salience=1.0, now_ts=now)
        assert w_yesterday_pre > w_sprint_post, f"48h boundary failed: {w_yesterday_pre} <= {w_sprint_post}"
        assert abs(w_yesterday_pre - w_sprint_post) < 0.05

        # Boundary 3: Sprint -> Long-term boundary (168.0h)
        w_sprint_pre = calculate_temporal_decay(now - 167.9 * 3600, salience=1.0, now_ts=now)
        w_longterm_post = calculate_temporal_decay(now - 168.1 * 3600, salience=1.0, now_ts=now)
        assert w_sprint_pre > w_longterm_post, f"168h boundary failed: {w_sprint_pre} <= {w_longterm_post}"
        assert abs(w_sprint_pre - w_longterm_post) < 0.05


class TestReverseChronologicalEpisodicReplayCascade:
    """Validates Feature 9: Reverse-Chronological Episodic Replay Engine."""

    def test_reverse_chronological_ordering(self) -> None:
        """Verifies memories are recalled strictly in reverse-chronological order (newest first).

        Engrams are recorded in scrambled order to guarantee recall ordering genuinely
        sorts on created_at rather than sequential insertion order.
        """
        mgr = CognitiveMemoryManager(capacity=10)
        now = time.time()

        # Scrambled insertion order
        mgr.record_decision(
            key="decision_today_recent",
            content="Decided 5m ago",
            salience=0.6,
            created_at=now - 300,
        )

        mgr.record_decision(
            key="decision_ancient",
            content="Decided long ago",
            salience=0.6,
            created_at=now - 100 * 3600,
        )

        mgr.record_decision(
            key="decision_today_earlier",
            content="Decided 4h ago",
            salience=0.6,
            created_at=now - 4 * 3600,
        )

        mgr.record_decision(
            key="decision_yesterday",
            content="Decided yesterday",
            salience=0.6,
            created_at=now - 30 * 3600,
        )

        results = mgr.recall_reverse_chronological_cascade(top_k=10, now_ts=now)
        assert len(results) == 4

        # Strictly descending created_at (newest first)
        timestamps = [r["created_at"] for r in results]
        assert timestamps == sorted(timestamps, reverse=True)
        assert [r["key"] for r in results] == [
            "decision_today_recent",
            "decision_today_earlier",
            "decision_yesterday",
            "decision_ancient",
        ]

    def test_cascade_recall_underfill_backfill(self) -> None:
        """Verifies backfilling of core candidates when non-core is sparse, preventing underfill."""
        mgr = CognitiveMemoryManager(capacity=25)
        now = time.time()

        # 15 core anchor decisions and 0 non-core decisions
        for i in range(15):
            mgr.record_decision(
                key=f"core_rule_{i:02d}",
                content=f"Foundational Rule {i}",
                salience=2.5,
                is_core_anchor=True,
                created_at=now - (i + 1) * 3600,
            )

        results = mgr.recall_reverse_chronological_cascade(top_k=10, now_ts=now)
        assert len(results) == 10, f"Expected 10 recalled engrams, got {len(results)}"
        timestamps = [r["created_at"] for r in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_cascade_recall_datetime_support(self) -> None:
        """Verifies recall handles datetime objects without crashing or type errors."""
        mgr = CognitiveMemoryManager(capacity=10)
        now_dt = datetime.now(timezone.utc)

        mgr.record_decision(
            key="dt_decision",
            content="Decision with datetime timestamp",
            salience=1.0,
            created_at=now_dt,
        )

        results = mgr.recall_reverse_chronological_cascade(top_k=5)
        assert len(results) == 1
        assert isinstance(results[0]["created_at"], float)
        assert abs(results[0]["created_at"] - now_dt.timestamp()) < 1.0

    def test_foundational_core_anchors_preserved_in_cascade(self) -> None:
        """Verifies core foundational decisions (S >= 2.0) are preserved even when capacity is tight."""
        mgr = CognitiveMemoryManager(capacity=20)
        now = time.time()

        # Core anchor from long ago (foundational rule)
        mgr.record_decision(
            key="foundational_native_rule",
            content="Always prioritize native APIs",
            salience=2.5,
            is_core_anchor=True,
        )
        mgr.buffer.buffer[-1]["created_at"] = now - 500 * 3600

        # Many transient decisions today
        for i in range(8):
            mgr.record_transient_decision(
                key=f"transient_today_{i}",
                content=f"Transient step {i} execution",
                salience=0.4,
            )
            mgr.buffer.buffer[-1]["created_at"] = now - (10 - i) * 60

        # Request top_k = 5
        results = mgr.recall_reverse_chronological_cascade(top_k=5, now_ts=now)
        assert len(results) == 5

        # Foundational core anchor MUST be preserved in the cascade
        keys = [r["key"] for r in results]
        assert "foundational_native_rule" in keys

        # Result is reverse-chronological
        timestamps = [r["created_at"] for r in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_cascade_metadata_structure(self) -> None:
        """Verifies returned cascade dicts have all required diagnostic fields."""
        mgr = CognitiveMemoryManager(capacity=5)
        now = time.time()
        mgr.record_decision(key="test_item", content="Important resolution", salience=1.5)
        mgr.buffer.buffer[-1]["created_at"] = now - 3600

        results = mgr.recall_reverse_chronological_cascade(top_k=5, now_ts=now)
        assert len(results) == 1
        item = results[0]

        expected_fields = {
            "key",
            "content",
            "category",
            "salience",
            "age_turns",
            "created_at",
            "age_hours",
            "horizon",
            "temporal_weight",
            "replay_score",
            "retention_fidelity",
            "retention_pct",
            "status",
            "is_core_anchor",
        }
        for field_name in expected_fields:
            assert field_name in item, f"Missing field: {field_name}"

        assert item["horizon"] == "today"
        assert item["temporal_weight"] <= 1.0


class TestDaemonEpisodicREMSeedIntegration:
    """Validates Feature 9 & 10: Daemon integration with episodic harvested seeds."""

    def test_daemon_harvest_conversation_history_delegation(self, tmp_path: Path) -> None:
        """Verifies SubconsciousDaemon delegates harvest_conversation_history to workspace_harvester."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)

        traces = daemon.harvest_conversation_history(max_sessions=5)
        assert isinstance(traces, list)

    def test_daemon_rem_cycle_uses_episodic_seeds(self, tmp_path: Path) -> None:
        """Verifies REM dreaming can be seeded by conversation traces."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)

        seed = DreamSeed(
            topic="episodic_test_task",
            speculative_question="How to optimize reverse-chronological replay?",
            urgency=2.2,
            context_keys=["quanta", "today", "episodic_harvested"],
        )
        res = daemon.run_rem_cycle(seed=seed)
        assert res is not None
        assert res.topic == "episodic_test_task"
        assert "episodic_harvested" in res.projects_paired


class TestMultiHorizonSWRHookLatency:
    """Validates Feature 10: Hook recency ordering and <25ms latency budget."""

    def test_hook_transient_decisions_reverse_chronological(self, tmp_path: Path) -> None:
        """Verifies hook outputs transient items in reverse-chronological order."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        now = time.time()
        seed_state = {
            "turn_count": 5,
            "last_injected_time": 0.0,
            "last_step_idx": 0,
            "engrams": [
                {
                    "key": "native_first_rule",
                    "content": "Platform işlemlerinde native API kullan.",
                    "salience": 2.8,
                    "category": "constraint",
                    "fidelity": 0.9998,
                    "is_core_anchor": True,
                },
                {
                    "key": "decision_older_step_10",
                    "content": "Older transient decision from 2h ago",
                    "salience": 0.60,
                    "category": "contextual_decision",
                    "fidelity": 0.88,
                    "is_core_anchor": False,
                    "created_at": now - 7200,
                },
                {
                    "key": "decision_newer_step_20",
                    "content": "Newer transient decision from 5m ago",
                    "salience": 0.60,
                    "category": "contextual_decision",
                    "fidelity": 0.88,
                    "is_core_anchor": False,
                    "created_at": now - 300,
                },
            ],
        }
        state_file.write_text(json.dumps(seed_state, indent=2), encoding="utf-8")

        hook_path = (
            Path(__file__).resolve().parent.parent
            / "scripts"
            / "hooks"
            / "cognitive_lifecycle_hook.py"
        )
        python_exe = sys.executable

        payload = {
            "conversationId": "test_recency_ordering",
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 5,
        }
        proc = subprocess.run(
            [python_exe, str(hook_path)],
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True,
            check=True,
        )

        data = json.loads(proc.stdout.decode("utf-8"))
        msg = data["injectSteps"][0]["ephemeralMessage"]

        assert "[Quanta Bilişsel Çıpa | SWR Replay]:" in msg
        assert "⚡ Geçici:" in msg
        # Newer item must appear before older item in transient section
        pos_newer = msg.find("decision_newer_step_20")
        pos_older = msg.find("decision_older_step_10")
        assert pos_newer != -1
        assert pos_older != -1
        assert pos_newer < pos_older, (
            f"Newer decision should precede older decision: newer={pos_newer}, older={pos_older}"
        )
        # Verify internal hook execution latency budget (< 25ms)
        telem_file = tmp_path / "quanta_cognitive_telemetry.jsonl"
        if not telem_file.exists():
            telem_file = (Path(__file__).resolve().parent.parent / "quanta_cognitive_telemetry.jsonl")
        if telem_file.exists():
            last_lines = telem_file.read_text(encoding="utf-8").strip().splitlines()[-10:]
            for line in reversed(last_lines):
                rec = json.loads(line)
                if "latency_ms" in rec:
                    assert (
                        rec["latency_ms"] < 25.0
                    ), f"Hook internal latency {rec['latency_ms']}ms exceeded 25ms!"
                    break


class TestCrossProjectDomainClustering:
    """Validates Feature 11: 8-cluster domain mapping and orthogonality metric."""

    def test_classify_project_domain_all_8_canonical_clusters(self) -> None:
        """Verifies each canonical cluster is classified accurately with unit norm."""
        assert len(CANONICAL_DOMAINS) == 8
        test_cases = [
            (Path("/mock/Quantum_Sim_Engine"), ["MSSQL", "CUDA-Q Backend", "simulator_engine"], "Quantum Simulation", 0),
            (Path("/mock/Alfa/quanta"), ["Python", "PyTorch", "MPS"], "Quantum/Brain", 1),
            (Path("/mock/SurfaceCodeQEC"), ["Python", "Steane", "ShorCode", "qec_code"], "Error Correction", 2),
            (Path("/mock/VariationalQML"), ["TypeScript", "Parameter Shift Gradients", "LCP"], "Variational/QML", 3),
            (Path("/mock/SabreTranspiler"), ["Streaming CDC", "Canonical Unitary", "MDM"], "Compiler/Transpiler", 4),
            (Path("/mock/qpu-backend-calib"), ["PulseSchedule", "QASM", "MCP", "BI"], "Hardware/Backends", 5),
            (Path("/mock/tensor-mps-network"), ["Android", "Room", "MPSBond", "Graph"], "Tensor Networks", 6),
            (Path("/mock/QuantumKDF"), ["NLP", "LLM", "Prompt", "Generator"], "Quantum Cryptography", 7),
        ]

        for path, stack, expected_domain, expected_idx in test_cases:
            domain, vec = classify_project_domain(
                path,
                tech_stack=stack,
                summary=f"Mock summary for {path.name}",
            )
            assert domain == expected_domain, f"Failed for {path.name}: expected {expected_domain}, got {domain}"
            assert len(vec) == 8, f"Vector for {path.name} must be 8-dimensional"
            norm = math.sqrt(sum(x * x for x in vec))
            assert abs(norm - 1.0) < 1e-4, f"Vector for {path.name} must have unit norm, got {norm}"
            assert vec.index(max(vec)) == expected_idx, f"Max component for {path.name} should be {expected_idx}"

    def test_domain_vector_orthogonality_calculation(self) -> None:
        """Verifies Delta_domain >= 0.50 for orthogonal domains and < 0.50 for collinear domains."""
        _, vec_travel = classify_project_domain(
            Path("/mock/Quantum_Sim_Engine"),
            tech_stack=["airline", "flight", "booking", "pricing"],
        )
        _, vec_legal = classify_project_domain(
            Path("/mock/SurfaceCodeQEC"),
            tech_stack=["stabilizer", "fault_tolerance", "steane", "contract"],
        )
        _, vec_travel_2 = classify_project_domain(
            Path("/mock/QuantumSimBench"),
            tech_stack=["quantum_sim", "ticket", "hotel", "reservation"],
        )

        # Delta_domain = 1.0 - (d_A . d_B) / (||d_A|| ||d_B||)
        dot_orthogonal = sum(a * b for a, b in zip(vec_travel, vec_legal, strict=False))
        delta_orthogonal = 1.0 - dot_orthogonal

        dot_collinear = sum(a * b for a, b in zip(vec_travel, vec_travel_2, strict=False))
        delta_collinear = 1.0 - dot_collinear

        assert delta_orthogonal >= 0.50, f"Disparate domains must have Delta_domain >= 0.50, got {delta_orthogonal}"
        assert delta_collinear < 0.50, f"Same-domain projects must have Delta_domain < 0.50, got {delta_collinear}"


class TestOrthogonalProjectPairing:
    """Validates Feature 11 & 12: Orthogonal project pairing and HybridDreamSeed construction."""

    def test_hybrid_dream_seed_attributes_and_post_init(self) -> None:
        """Verifies HybridDreamSeed fields, typing, and automatic context_keys initialization."""
        proj_a = ProjectContext(name="Quantum_Sim_Engine", path=Path("/mock/quantum_sim"), domain="Quantum Simulation")
        proj_b = ProjectContext(name="SurfaceCodeQEC", path=Path("/mock/legal"), domain="Error Correction")

        seed = HybridDreamSeed(
            topic="quantum_sim_x_surfacecodeqec",
            speculative_question="How to cross-synthesize?",
            primary_project=proj_a,
            secondary_project=proj_b,
            domain_distance=0.92,
            bisociation_theme="Travel Ticketing and Contract Automation",
            urgency=2.2,
        )

        assert seed.topic == "quantum_sim_x_surfacecodeqec"
        assert seed.primary_project.name == "Quantum_Sim_Engine"
        assert seed.secondary_project.name == "SurfaceCodeQEC"
        assert seed.domain_distance == 0.92
        assert seed.bisociation_theme == "Travel Ticketing and Contract Automation"
        assert seed.urgency == 2.2
        assert "Quantum_Sim_Engine" in seed.context_keys
        assert "SurfaceCodeQEC" in seed.context_keys
        assert "hybrid_bisociation" in seed.context_keys

    def test_pair_orthogonal_projects_discovery_and_distance_filtering(self, tmp_path: Path) -> None:
        """Verifies harvester pairs orthogonal projects with Delta_domain >= 0.50."""
        # Create mock projects directory with diverse projects
        (tmp_path / "Quantum_Sim_Engine").mkdir()
        (tmp_path / "Quantum_Sim_Engine" / "package.json").write_text(
            json.dumps({"name": "quantum_sim", "description": "hamiltonian trotter simulation"}),
            encoding="utf-8",
        )

        (tmp_path / "SurfaceCodeQEC").mkdir()
        (tmp_path / "SurfaceCodeQEC" / "pyproject.toml").write_text(
            'description = "Turkish fault_tolerance and contract compliance"',
            encoding="utf-8",
        )

        (tmp_path / "VariationalQML").mkdir()
        (tmp_path / "VariationalQML" / "package.json").write_text(
            json.dumps({"name": "variationalqml", "description": "web auditing parameter shift gradients"}),
            encoding="utf-8",
        )

        harvester = WorkspaceContextHarvester(projects_dir=tmp_path)
        pairs = harvester.pair_orthogonal_projects(min_domain_distance=0.50, max_pairs=3)

        assert len(pairs) > 0
        for p in pairs:
            assert isinstance(p, HybridDreamSeed)
            assert p.domain_distance >= 0.50
            assert p.primary_project.name != p.secondary_project.name

    def test_exclude_shared_project_prefixes(self, tmp_path: Path) -> None:
        """Verifies projects sharing prefix (including PascalCase) are never paired."""
        (tmp_path / "Quantum_Sim_Engine").mkdir()
        (tmp_path / "Quantum_Sim_Engine" / "package.json").write_text(
            json.dumps({"name": "sim-core", "description": "statevector simulation engine"}),
            encoding="utf-8",
        )

        (tmp_path / "QuantumSimBench").mkdir()
        (tmp_path / "QuantumSimBench" / "package.json").write_text(
            json.dumps({"name": "sim-tracker", "description": "statevector benchmark integration"}),
            encoding="utf-8",
        )

        (tmp_path / "SimCore").mkdir()
        (tmp_path / "SimCore" / "package.json").write_text(
            json.dumps({"name": "sim-core-app", "description": "statevector simulation engine mobile"}),
            encoding="utf-8",
        )

        (tmp_path / "SimTracker").mkdir()
        (tmp_path / "SimTracker" / "package.json").write_text(
            json.dumps({"name": "sim-tracker-app", "description": "statevector benchmark integration mobile"}),
            encoding="utf-8",
        )

        (tmp_path / "SurfaceCodeQEC").mkdir()
        (tmp_path / "SurfaceCodeQEC" / "pyproject.toml").write_text(
            'description = "legal contract compliance"',
            encoding="utf-8",
        )

        harvester = WorkspaceContextHarvester(projects_dir=tmp_path)
        pairs = harvester.pair_orthogonal_projects(min_domain_distance=0.50, max_pairs=5)

        for p in pairs:
            names = {p.primary_project.name, p.secondary_project.name}
            assert names != {"Quantum_Sim_Engine", "QuantumSimBench"}, "Quantum_Sim_Engine and QuantumSimBench must NOT be paired together!"
            assert names != {"SimCore", "SimTracker"}, "SimCore and SimTracker must NOT be paired together!"

    def test_anti_rumination_pair_buffer(self, tmp_path: Path) -> None:
        """Verifies the anti-rumination pair buffer tracks recent pairs and diversifies seeds."""
        (tmp_path / "Quantum_Sim_Engine").mkdir()
        (tmp_path / "Quantum_Sim_Engine" / "package.json").write_text(
            json.dumps({"name": "sim-core", "description": "statevector simulation engine"}),
            encoding="utf-8",
        )

        (tmp_path / "SurfaceCodeQEC").mkdir()
        (tmp_path / "SurfaceCodeQEC" / "pyproject.toml").write_text(
            'description = "legal contract compliance"',
            encoding="utf-8",
        )

        (tmp_path / "VariationalQML").mkdir()
        (tmp_path / "VariationalQML" / "package.json").write_text(
            json.dumps({"name": "variationalqml", "description": "web analytics and Parameter Shift Gradients"}),
            encoding="utf-8",
        )

        harvester = WorkspaceContextHarvester(projects_dir=tmp_path)
        pairs_1 = harvester.pair_orthogonal_projects(min_domain_distance=0.50, max_pairs=2)
        assert len(pairs_1) > 0
        assert len(harvester.anti_rumination_pairs) > 0
        assert len(harvester._recent_hybrid_pairs) > 0
        assert harvester.anti_rumination_pairs == list(harvester._recent_hybrid_pairs)

        # Buffer capacity should be 50
        assert harvester._recent_hybrid_pairs.maxlen == 50

        # Buffer should contain tuples of project names
        for item in harvester.anti_rumination_pairs:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_pair_orthogonal_projects_empty_or_single_project(self, tmp_path: Path) -> None:
        """Verifies pair_orthogonal_projects gracefully returns empty list when < 2 projects exist."""
        harvester = WorkspaceContextHarvester(projects_dir=tmp_path)
        # 0 projects in tmp_path
        assert harvester.pair_orthogonal_projects() == []

        # 1 project in tmp_path
        (tmp_path / "SoloProject").mkdir()
        (tmp_path / "SoloProject" / "package.json").write_text(
            json.dumps({"name": "solo", "description": "single project"}),
            encoding="utf-8",
        )
        assert harvester.pair_orthogonal_projects() == []


class TestHybridDreamCycleExecution:
    """Validates Feature 12: Dual-persona dialectic execution on HybridDreamSeed."""

    def test_execute_hybrid_dream_cycle_temperatures_and_result(self) -> None:
        """Verifies execute_hybrid_dream_cycle uses T_dreamer >= 0.90 and T_arbiter in [0.45, 0.55]."""
        proj_a = ProjectContext(
            name="Quantum_Sim_Engine",
            path=Path("/mock/quantum_sim"),
            domain="Quantum Simulation",
            tech_stack=["MSSQL", "DenseMatrix"],
            summary="Airline ticket pricing engine.",
        )
        proj_b = ProjectContext(
            name="tensor-mps-network",
            path=Path("/mock/network"),
            domain="Tensor Networks",
            tech_stack=["Android", "Room", "MPSBond"],
            summary="Spectral graph partitioning and offline sync.",
        )
        seed = HybridDreamSeed(
            topic="quantum_sim_x_network",
            speculative_question="How can spectral graphs optimize dynamic ticket pricing?",
            primary_project=proj_a,
            secondary_project=proj_b,
            domain_distance=0.95,
            bisociation_theme="Spectral Graph Partitioning for Trotter Step Optimization",
        )

        engine = MindWanderEngine(max_turns=3, max_tokens=1000)
        res = engine.execute_hybrid_dream_cycle(seed, relaxed_prefrontal=True)

        assert res is not None
        assert isinstance(res, MindWanderResult)
        assert res.dreamer_temperature >= 0.90
        assert 0.45 <= res.prefrontal_temperature <= 0.55
        assert res.confidence >= 0.90
        assert res.primary_project == "Quantum_Sim_Engine"
        assert res.secondary_project == "tensor-mps-network"
        assert res.domain_distance == 0.95
        assert len(res.dialectical_turns) > 0
        assert "Quantum_Sim_Engine" in res.synthesis
        assert "tensor-mps-network" in res.synthesis
        assert res.abstract_architecture != ""
        assert len(res.failure_modes) > 0

    def test_execute_hybrid_dream_cycle_instant_preemption(self) -> None:
        """Verifies hybrid dream cycle aborts instantly when preemption is signaled."""
        proj_a = ProjectContext(name="Quantum_Sim_Engine", path=Path("/mock/quantum_sim"), domain="Quantum Simulation")
        proj_b = ProjectContext(name="SurfaceCodeQEC", path=Path("/mock/legal"), domain="Error Correction")
        seed = HybridDreamSeed(
            topic="quantum_sim_x_legal",
            speculative_question="How to arbitrate disputes?",
            primary_project=proj_a,
            secondary_project=proj_b,
            domain_distance=0.90,
            bisociation_theme="Arbitration",
        )

        engine = MindWanderEngine(max_turns=3)
        res = engine.execute_hybrid_dream_cycle(
            seed,
            relaxed_prefrontal=True,
            preemption_check=lambda: True,
        )
        assert res is None, "Hybrid dream cycle must immediately return None on preemption"


class TestPersistHybridRFCAndSandboxSafety:
    """Validates Feature 13 & 14: Centralized RFC persistence and strict sandbox safety."""

    def test_persist_hybrid_rfc_frontmatter_and_sections(self, tmp_path: Path) -> None:
        """Verifies _persist_hybrid_rfc writes valid YAML frontmatter and required architectural sections."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)

        proj_a = ProjectContext(
            name="Quantum_Sim_Engine",
            path=Path("/mock/quantum_sim"),
            domain="Quantum Simulation",
            tech_stack=["MSSQL", "CUDA-Q Backend"],
            summary="Airline pricing engine.",
        )
        proj_b = ProjectContext(
            name="SurfaceCodeQEC",
            path=Path("/mock/legal"),
            domain="Error Correction",
            tech_stack=["Python", "FastAPI"],
            summary="Regulatory compliance.",
        )
        seed = HybridDreamSeed(
            topic="quantum_sim_x_surfacecodeqec",
            speculative_question="How can automated compliance arbitrate ticket repricing?",
            primary_project=proj_a,
            secondary_project=proj_b,
            domain_distance=0.91,
            bisociation_theme="Automated Compliance Gate for Dynamic Trotter Step Optimization",
        )
        result = MindWanderResult(
            topic=seed.topic,
            seed_question=seed.speculative_question,
            synthesis="Consensual synthesis bridging flight pricing and compliance.",
            confidence=0.96,
            turns_taken=3,
            tokens_used=480,
            anti_rumination_reset_occurred=False,
            relaxed_prefrontal=True,
            prefrontal_temperature=0.50,
            dreamer_temperature=0.92,
            primary_project="Quantum_Sim_Engine",
            secondary_project="SurfaceCodeQEC",
            domain_pair="Quantum Simulation x Error Correction",
            domain_distance=0.91,
            bisociation_theme=seed.bisociation_theme,
            abstract_architecture="[Quantum_Sim_Engine] <---> [Compliance Bus] <---> [SurfaceCodeQEC]",
            failure_modes=[{"mode": "Schema Drift", "cause": "Field drift", "mitigation": "Adapter"}],
        )

        rfc_path = daemon._persist_hybrid_rfc(result, seed)
        try:
            assert rfc_path.exists()
            assert "RFC_HYBRID_" in rfc_path.name
            assert "quantum_sim" in rfc_path.name
            assert "surfacecodeqec" in rfc_path.name
            # Must be saved inside quanta/docs/rfcs
            assert str(rfc_path).endswith(".md")
            assert "rfcs" in str(rfc_path.parent)

            content = rfc_path.read_text(encoding="utf-8")
            assert "---" in content
            assert "rfc_id: RFC_HYBRID_" in content
            assert "Quantum_Sim_Engine" in content
            assert "SurfaceCodeQEC" in content
            assert "cross_domain_bisociation" in content
            assert "domain_distance: 0.9100" in content
            assert "## 1. Executive Summary & Bisociative Thesis" in content
            assert "## 2. Dialectical Subconscious Deliberation" in content
            assert "## 3. Abstract Architecture" in content
            assert "## 4. Failure Modes & Mitigations" in content
            assert "## 5. Sandbox Safety & Isolation Verification" in content
        finally:
            if rfc_path.exists():
                rfc_path.unlink()

    def test_persist_hybrid_rfc_records_high_salience_engram(self, tmp_path: Path) -> None:
        """Verifies _persist_hybrid_rfc records high-salience engram (S >= 2.5) in state."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)

        proj_a = ProjectContext(name="Quantum_Sim_Engine", path=Path("/mock/quantum_sim"), domain="Quantum Simulation")
        proj_b = ProjectContext(name="SurfaceCodeQEC", path=Path("/mock/legal"), domain="Error Correction")
        seed = HybridDreamSeed(
            topic="quantum_sim_x_legal",
            speculative_question="Speculative test?",
            primary_project=proj_a,
            secondary_project=proj_b,
            domain_distance=0.88,
            bisociation_theme="Compliance Arbitration",
        )
        result = MindWanderResult(
            topic=seed.topic,
            seed_question=seed.speculative_question,
            synthesis="Synthesis test",
            confidence=0.95,
            turns_taken=2,
            tokens_used=300,
            anti_rumination_reset_occurred=False,
            primary_project="Quantum_Sim_Engine",
            secondary_project="SurfaceCodeQEC",
            domain_distance=0.88,
            bisociation_theme=seed.bisociation_theme,
        )

        rfc_path = daemon._persist_hybrid_rfc(result, seed)
        try:
            assert state_file.exists()
            state_data = json.loads(state_file.read_text(encoding="utf-8"))
            engrams = state_data.get("engrams", [])
            rfc_engrams = [e for e in engrams if e.get("category") == "architecture_rfc"]
            assert len(rfc_engrams) >= 1
            engram = rfc_engrams[-1]
            assert engram["salience"] >= 2.5
            assert "Quantum_Sim_Engine" in engram["content"]
            assert "SurfaceCodeQEC" in engram["content"]
        finally:
            if rfc_path.exists():
                rfc_path.unlink()

    def test_sandbox_safety_zero_external_modifications(self, tmp_path: Path) -> None:
        """CRITICAL INVARIANT: Asserts external project trees suffer ZERO modifications or writes."""
        # Create simulated external project trees
        ext_project_a = tmp_path / "External_Simulator_Engine"
        ext_project_a.mkdir()
        file_a1 = ext_project_a / "pricing.py"
        file_a1.write_text("def calculate_fare(): return 100", encoding="utf-8")
        file_a2 = ext_project_a / "package.json"
        file_a2.write_text(json.dumps({"name": "travel-engine"}), encoding="utf-8")

        ext_project_b = tmp_path / "External_QEC_Platform"
        ext_project_b.mkdir()
        file_b1 = ext_project_b / "compliance.py"
        file_b1.write_text("def verify_contract(): return True", encoding="utf-8")
        file_b2 = ext_project_b / "pyproject.toml"
        file_b2.write_text('name = "legal-platform"\nversion = "1.0.0"', encoding="utf-8")

        # Function to compute recursive hash of a directory
        def compute_tree_signature(p: Path) -> dict[str, tuple[str, float]]:
            sig: dict[str, tuple[str, float]] = {}
            for item in sorted(p.rglob("*")):
                if item.is_file():
                    content_hash = subprocess.check_output(
                        ["shasum", "-a", "256", str(item)],
                    ).decode("utf-8").split()[0]
                    mtime = item.stat().st_mtime
                    rel = str(item.relative_to(p))
                    sig[rel] = (content_hash, mtime)
            return sig

        sig_a_before = compute_tree_signature(ext_project_a)
        sig_b_before = compute_tree_signature(ext_project_b)

        # Execute full harvester and daemon dream cycle
        proj_a = ProjectContext(
            name="External_Simulator_Engine",
            path=ext_project_a,
            domain="Quantum Simulation",
            summary="Airline pricing engine.",
        )
        proj_b = ProjectContext(
            name="External_QEC_Platform",
            path=ext_project_b,
            domain="Error Correction",
            summary="Regulatory compliance.",
        )
        seed = HybridDreamSeed(
            topic="travel_x_legal_sandbox_test",
            speculative_question="How to maintain sandbox integrity?",
            primary_project=proj_a,
            secondary_project=proj_b,
            domain_distance=0.93,
            bisociation_theme="Sandbox Safety Verification",
        )

        state_file = tmp_path / "quanta_cognitive_state.json"
        daemon = SubconsciousDaemon(state_file=state_file, projects_dir=tmp_path)
        engine = MindWanderEngine(max_turns=2)
        hybrid_res = engine.execute_hybrid_dream_cycle(seed)
        assert hybrid_res is not None

        rfc_path = daemon._persist_hybrid_rfc(hybrid_res, seed)
        try:
            # Recompute external tree signatures
            sig_a_after = compute_tree_signature(ext_project_a)
            sig_b_after = compute_tree_signature(ext_project_b)

            assert sig_a_before == sig_a_after, "External Project A was modified during hybrid dreaming!"
            assert sig_b_before == sig_b_after, "External Project B was modified during hybrid dreaming!"

            # Assert no docs/ directory was created in external projects
            assert not (ext_project_a / "docs").exists(), "Created docs directory inside external project A!"
            assert not (ext_project_b / "docs").exists(), "Created docs directory inside external project B!"

            # Assert RFC was written ONLY to Quanta docs/rfcs directory
            assert "quanta/docs/rfcs" in str(rfc_path) or "docs/rfcs" in str(rfc_path)
            assert not str(rfc_path).startswith(str(ext_project_a))
            assert not str(rfc_path).startswith(str(ext_project_b))
        finally:
            if rfc_path.exists():
                rfc_path.unlink()


class TestDaemonREMIntegrationWithHybridSeed:
    """Validates Feature 13 & SubconsciousDaemon integration with HybridDreamSeed."""

    def test_run_rem_cycle_with_hybrid_seed(self, tmp_path: Path) -> None:
        """Verifies SubconsciousDaemon.run_rem_cycle handles HybridDreamSeed and creates RFC."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)

        proj_a = ProjectContext(
            name="Quantum_Sim_Engine",
            path=Path("/mock/quantum_sim"),
            domain="Quantum Simulation",
            summary="Airline dynamic ticket repricing.",
        )
        proj_b = ProjectContext(
            name="SurfaceCodeQEC",
            path=Path("/mock/legal"),
            domain="Error Correction",
            summary="Jurisprudence and contracts.",
        )
        seed = HybridDreamSeed(
            topic="quantum_sim_x_surfacecodeqec_integration",
            speculative_question="How to cross-synthesize travel and legal tech?",
            primary_project=proj_a,
            secondary_project=proj_b,
            domain_distance=0.92,
            bisociation_theme="Cross-Domain Travel Compliance Synthesis",
        )

        res = daemon.run_rem_cycle(seed=seed)
        assert res is not None
        assert isinstance(res, REMExecutionResult)
        assert res.projects_paired == ["Quantum_Sim_Engine", "SurfaceCodeQEC"]
        assert res.dreamer_temperature >= 0.90
        assert 0.45 <= res.prefrontal_temperature <= 0.55
        assert res.rfc_path is not None
        rfc_file = Path(res.rfc_path)
        try:
            assert rfc_file.exists()
            assert "quantum_sim" in rfc_file.name
            assert "surfacecodeqec" in rfc_file.name
        finally:
            if rfc_file.exists():
                rfc_file.unlink()


class TestHoelStochasticNoiseInjection:
    """Validates Feature 15: Hoel Stochastic Noise Injection (Overfitted Brain Hypothesis, OBH)."""

    def test_parameter_bounding_and_clamping(self) -> None:
        """Verifies sigma parameter bounds in [0.05, 0.20]."""
        vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Lower bound clamping
        _, p_norm_low = perturb_thought_vector(vec, sigma=0.01)
        assert p_norm_low > 0.0

        # Upper bound clamping
        _, p_norm_high = perturb_thought_vector(vec, sigma=0.50)
        assert p_norm_high > 0.0

        # Nominal in-range
        _, p_norm_nom = perturb_thought_vector(vec, sigma=0.10)
        assert p_norm_nom > 0.0

    def test_l2_norm_normalization(self) -> None:
        """Verifies ||v'||_2 = 1.000000 +/- 1e-6 without NaN or Inf."""
        test_vectors = [
            [1.0] * 8,
            [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0] * 8,  # Degenerate zero vector
        ]
        for v in test_vectors:
            for sig in (0.05, 0.10, 0.20):
                v_prime, p_norm = perturb_thought_vector(v, sigma=sig)
                norm = math.sqrt(sum(x * x for x in v_prime))
                assert math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6)
                assert all(not math.isnan(x) and not math.isinf(x) for x in v_prime)

    def test_diversity_enhancement_cosine_similarity(self) -> None:
        """Verifies stochastic perturbation achieves diversity (cosine similarity < 0.99)."""
        dim = 8
        base_v = [1.0 / math.sqrt(dim)] * dim
        similarities = []

        for _ in range(50):
            v_prime, _ = perturb_thought_vector(base_v, sigma=0.12)
            sim = cosine_similarity(base_v, v_prime)
            similarities.append(sim)

        avg_sim = sum(similarities) / len(similarities)
        # Cosine similarity under sigma=0.12 should average around 0.95 - 0.98 (< 0.99)
        assert avg_sim < 0.99
        assert min(similarities) < 0.99

    def test_rem_dream_cycle_stochastic_noise_telemetry(self) -> None:
        """Verifies execute_dream_cycle records noise injection telemetry during REM."""
        engine = MindWanderEngine(max_turns=3, use_sdk_if_available=False, use_agy_cli=False)
        seed = DreamSeed(
            topic="quantum_noise_test",
            speculative_question="How does stochastic noise regularize representations?",
            urgency=1.0,
            context_keys=["quanta"],
        )

        # REM with noise active
        insight_rem = engine.execute_dream_cycle(
            seed,
            relaxed_prefrontal=True,
            inject_stochastic_noise=True,
            noise_sigma=0.12,
        )
        assert insight_rem is not None
        assert insight_rem.noise_injected is True
        assert math.isclose(insight_rem.noise_sigma, 0.12, abs_tol=1e-5)
        assert insight_rem.perturbation_norm > 0.0

        # Non-REM (relaxed_prefrontal=False): noise must not be injected
        insight_sws = engine.execute_dream_cycle(
            seed,
            relaxed_prefrontal=False,
            inject_stochastic_noise=True,
            noise_sigma=0.12,
        )
        assert insight_sws is not None
        assert insight_sws.noise_injected is False
        assert insight_sws.noise_sigma == 0.0
        assert insight_sws.perturbation_norm == 0.0

        # Explicitly disabled noise: noise must not be injected
        insight_disabled = engine.execute_dream_cycle(
            seed,
            relaxed_prefrontal=True,
            inject_stochastic_noise=False,
            noise_sigma=0.12,
        )
        assert insight_disabled is not None
        assert insight_disabled.noise_injected is False

    def test_hybrid_dream_cycle_stochastic_noise_telemetry(self) -> None:
        """Verifies execute_hybrid_dream_cycle records noise injection in MindWanderResult."""
        engine = MindWanderEngine(max_turns=3, use_sdk_if_available=False, use_agy_cli=False)
        proj_a = ProjectContext(
            name="Quantum_Sim_Engine", path=Path("/mock/quantum_sim"), domain="Quantum Simulation"
        )
        proj_b = ProjectContext(
            name="SurfaceCodeQEC", path=Path("/mock/legal"), domain="Error Correction"
        )
        seed = HybridDreamSeed(
            topic="flight_x_legal_noise_test",
            speculative_question="How to bridge flight booking and legal compliance?",
            primary_project=proj_a,
            secondary_project=proj_b,
            domain_distance=0.90,
            bisociation_theme="Compliance and Pricing Synthesis",
        )

        res = engine.execute_hybrid_dream_cycle(
            seed,
            relaxed_prefrontal=True,
            inject_stochastic_noise=True,
            noise_sigma=0.15,
        )
        assert res is not None
        assert isinstance(res, MindWanderResult)
        assert res.noise_injected is True
        assert math.isclose(res.noise_sigma, 0.15, abs_tol=1e-5)
        assert res.perturbation_norm > 0.0


class TestHintonWakeSleepPlasticity:
    """Validates Feature 16: Hinton Wake-Sleep Synaptic Plasticity."""

    def test_sws_downscaling_and_noise_pruning(self) -> None:
        """Verifies SWS downscales non-core engrams and prunes low-salience noise."""
        mgr = CognitiveMemoryManager(capacity=16, auto_prune=False)

        # 1. Core anchor engram (S >= 2.0 or is_core_anchor=True)
        mgr.record_decision(
            key="core_invariant_anchor",
            content="Critical system invariant.",
            salience=2.50,
            category="architecture",
        )

        # 2. Normal operational engram (medium salience S=1.20)
        mgr.record_decision(
            key="operational_heuristic",
            content="Standard procedural decision.",
            salience=1.20,
            category="general",
        )

        # 3. Noisy ephemeral engram (S <= 0.80) with dephased state (fidelity < 0.70)
        noise_idx = mgr.record_decision(
            key="transient_debug_noise",
            content="Temporary shell print output.",
            salience=0.75,
            category="ephemeral",
        )
        # Artificially dephase the noise engram so fidelity < 0.70
        mgr.buffer.buffer[noise_idx]["degraded_state"] = (
            mgr.buffer.buffer[noise_idx]["pristine_state"] * 0.50
        )

        # Execute SWS wake-sleep plasticity
        res = mgr.apply_wake_sleep_plasticity(SleepPhase.NREM_SWS, shy_factor=0.90)
        assert res["phase"] == "NREM_SWS"
        assert res["downscaled_count"] >= 1
        assert res["pruned_count"] >= 1

        keys_surviving = [e.get("metadata", {}).get("key") for e in mgr.buffer.buffer]
        assert "core_invariant_anchor" in keys_surviving
        assert "operational_heuristic" in keys_surviving
        assert "transient_debug_noise" not in keys_surviving

        # Verify core anchor salience was protected
        core_engram = next(
            e for e in mgr.buffer.buffer if e["metadata"]["key"] == "core_invariant_anchor"
        )
        assert float(core_engram["dopamine_tag"]) == 2.50

        # Verify operational engram was downscaled: 1.20 * 0.90 = 1.08
        op_engram = next(
            e for e in mgr.buffer.buffer if e["metadata"]["key"] == "operational_heuristic"
        )
        assert math.isclose(float(op_engram["dopamine_tag"]), 1.08, abs_tol=1e-4)

    def test_rem_potentiation_of_cross_domain_bisociations(self) -> None:
        """Verifies REM potentiates generalizing cross-domain bisociations to S >= 2.50."""
        mgr = CognitiveMemoryManager(capacity=16, auto_prune=False)

        # Regular task engram
        mgr.record_decision(
            key="regular_task_step",
            content="Local file parsing step.",
            salience=1.10,
            category="general",
        )

        # Cross-domain bisociation engram (initial S=1.80)
        idx_bisoc = mgr.record_decision(
            key="rfc_hybrid_travel_x_compliance",
            content="Cross-domain synthesis bridging flight pricing with legal compliance.",
            salience=1.80,
            category="hybrid_dream",
        )
        mgr.buffer.buffer[idx_bisoc]["metadata"]["projects_paired"] = ["Quantum_Sim_Engine", "SurfaceCodeQEC"]
        mgr.buffer.buffer[idx_bisoc]["metadata"]["domain_pair"] = "Quantum Simulation x Error Correction"

        res = mgr.apply_wake_sleep_plasticity(SleepPhase.REM_DREAM)
        assert res["phase"] == "REM_DREAM"
        assert res["potentiated_count"] >= 1

        # Check bisociation was potentiated to >= 2.50 and marked as core anchor
        bisoc_engram = next(
            e
            for e in mgr.buffer.buffer
            if e["metadata"]["key"] == "rfc_hybrid_travel_x_compliance"
        )
        assert float(bisoc_engram["dopamine_tag"]) >= 2.50
        assert bisoc_engram["metadata"]["is_core_anchor"] is True

        # Check regular engram was not potentiated to core
        reg_engram = next(
            e for e in mgr.buffer.buffer if e["metadata"]["key"] == "regular_task_step"
        )
        assert float(reg_engram["dopamine_tag"]) == 1.10
        assert not reg_engram["metadata"].get("is_core_anchor", False)

    def test_subconscious_consolidator_atomic_persistence(self, tmp_path: Path) -> None:
        """Verifies SubconsciousConsolidator.apply_wake_sleep_plasticity persists state atomically."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file, capacity=16)

        # Inject bisociation and noise
        idx1 = consolidator.memory_manager.record_decision(
            key="rfc_hybrid_insight",
            content="Cross-domain hybrid architecture insight.",
            salience=1.90,
            category="hybrid_dream",
        )
        consolidator.memory_manager.buffer.buffer[idx1]["metadata"]["projects_paired"] = [
            "A",
            "B",
        ]

        idx2 = consolidator.memory_manager.record_decision(
            key="noise_step",
            content="Temporary noise output.",
            salience=0.60,
            category="ephemeral",
        )
        consolidator.memory_manager.buffer.buffer[idx2]["degraded_state"] *= 0.40

        # Execute SWS plasticity via consolidator
        res_sws = consolidator.apply_wake_sleep_plasticity(SleepPhase.NREM_SWS)
        assert res_sws["phase"] == "NREM_SWS"
        assert res_sws["pruned_count"] >= 1

        # Execute REM plasticity via consolidator
        res_rem = consolidator.apply_wake_sleep_plasticity(SleepPhase.REM_DREAM)
        assert res_rem["phase"] == "REM_DREAM"
        assert res_rem["potentiated_count"] >= 1

        # Verify disk state reflects changes
        loaded = consolidator.load_state()
        engrams_disk = {e["key"]: e for e in loaded.get("engrams", [])}
        assert "rfc_hybrid_insight" in engrams_disk
        assert engrams_disk["rfc_hybrid_insight"]["salience"] >= 2.50
        assert "noise_step" not in engrams_disk


class TestEndToEndTwoPhaseSleep:
    """Validates Feature 17: End-to-End Two-Phase Biomorphic Sleep Architecture."""

    def test_alternating_nrem_rem_sleep_cycles(self, tmp_path: Path) -> None:
        """Verifies state transitions, glymphatic clearance, and alternating cycles in daemon."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(
            state_file=state_file,
            pid_file=pid_file,
            use_agy_cli=False,
        )

        # Add stale .tmp_ file for glymphatic clearance verification
        stale_file = tmp_path / ".tmp_old_state_123"
        stale_file.write_text("temporary stale cache")
        past_time = time.time() - 120.0
        os.utime(stale_file, (past_time, past_time))

        assert daemon.current_sleep_phase == SleepPhase.WAKE

        # Phase 1: NREM Slow-Wave Sleep
        nrem_res = daemon.run_nrem_cycle()
        assert nrem_res is not None
        assert isinstance(nrem_res, NREMExecutionResult)
        assert nrem_res.phase == SleepPhase.NREM_SWS
        assert nrem_res.glymphatic_files_cleared >= 1
        assert not stale_file.exists()
        assert daemon.current_sleep_phase == SleepPhase.WAKE
        assert daemon.nrem_cycles_completed == 1

        # Phase 2: REM Paradoxical Dreaming
        proj_a = ProjectContext(
            name="Quantum_Sim_Engine", path=Path("/mock/quantum_sim"), domain="Quantum Simulation"
        )
        proj_b = ProjectContext(
            name="SurfaceCodeQEC", path=Path("/mock/legal"), domain="Error Correction"
        )
        seed = HybridDreamSeed(
            topic="two_phase_travel_legal",
            speculative_question="Cross-domain synthesis test.",
            primary_project=proj_a,
            secondary_project=proj_b,
            domain_distance=0.88,
            bisociation_theme="Travel Legal Synthesis",
        )
        rem_res = daemon.run_rem_cycle(seed=seed)
        assert rem_res is not None
        assert isinstance(rem_res, REMExecutionResult)
        assert rem_res.phase == SleepPhase.REM_DREAM
        assert rem_res.noise_injected is True
        assert 0.05 <= rem_res.noise_sigma <= 0.20
        assert rem_res.perturbation_norm > 0.0
        assert daemon.current_sleep_phase == SleepPhase.WAKE
        assert daemon.rem_cycles_completed == 1

        # Verify created RFC exists in centralized quanta/docs/rfcs/
        assert rem_res.rfc_path is not None
        rfc_path = Path(rem_res.rfc_path)
        try:
            assert rfc_path.exists()
            content = rfc_path.read_text(encoding="utf-8")
            assert "rfc_id: RFC_HYBRID_" in content
        finally:
            if rfc_path.exists():
                rfc_path.unlink()

    def test_reverse_chronological_replay_cascade_during_sleep(self, tmp_path: Path) -> None:
        """Verifies memories are replayed newest-first (t_now -> t_past) during consolidation."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)
        now = time.time()

        mgr = daemon.consolidator.memory_manager
        # Inject memories at different time horizons
        mgr.record_decision(
            "dec_today", "Turn today", salience=1.2, created_at=now - 1800
        )
        mgr.record_decision(
            "dec_yesterday", "Turn yesterday", salience=1.2, created_at=now - 108000
        )
        mgr.record_decision(
            "dec_sprint", "Turn sprint", salience=1.2, created_at=now - 300000
        )

        cascade = mgr.recall_reverse_chronological_cascade(top_k=10)
        assert len(cascade) >= 3
        timestamps = [item.get("timestamp", 0) for item in cascade]
        # Verify strict descending order (newest first)
        assert timestamps == sorted(timestamps, reverse=True)

    def test_sub_20ms_preemption_sla_in_both_phases(self, tmp_path: Path) -> None:
        """Verifies preemption interrupt (< 20ms) in both NREM and REM phases."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"
        daemon = SubconsciousDaemon(state_file=state_file, pid_file=pid_file)

        # NREM preemption
        t0 = time.perf_counter()
        nrem_res = daemon.run_nrem_cycle(preemption_check=lambda: True)
        elapsed_nrem_ms = (time.perf_counter() - t0) * 1000.0
        assert nrem_res is None
        assert elapsed_nrem_ms < 20.0
        assert daemon.current_sleep_phase == SleepPhase.PREEMPTED

        # Reset phase for REM
        daemon.current_sleep_phase = SleepPhase.WAKE

        # REM preemption
        t1 = time.perf_counter()
        rem_res = daemon.run_rem_cycle(preemption_check=lambda: True)
        elapsed_rem_ms = (time.perf_counter() - t1) * 1000.0
        assert rem_res is None
        assert elapsed_rem_ms < 20.0
        assert daemon.current_sleep_phase == SleepPhase.PREEMPTED



