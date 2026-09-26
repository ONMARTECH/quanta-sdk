"""tests/test_nrem_glymphatic_shy_stress.py — Empirical Challenger Stress Suite.

Adversarially tests and challenges:
1. Glymphatic Clearance:
   - Corrupt temp files (0-byte, random binary, corrupted content).
   - Locked temp files (flock advisory locks, non-file entries like .tmp_* directories).
   - Strict time boundary: only stale .tmp_* (>60.0s) are deleted; fresh (<=60.0s) and future mtime preserved.
   - Non-temp files and parent state files are strictly untouched.
2. Depotentiated Inhibitors:
   - Strict boundary partitioning for V_inh (< 0.20) and S (<= 0.50).
   - Active inhibitors (V_inh >= 0.20 or S > 0.50) strictly survive.
   - Core anchor inhibitors (is_core_anchor=True or S >= 2.0) are strictly immune even if V_inh < 0.20.
3. Synaptic Scaling (SHY):
   - Proactive downscaling by 0.90 down to 0.20 floor across multiple sleep cycles.
   - Core anchors (is_core_anchor=True, S >= 2.0, semantic_gist, subconscious_dream) are NEVER downscaled.
4. Semantic Gist Crystallization:
   - Decaying actionable decisions (S <= 0.8, F < 0.70) crystallize into semantic gists (category='semantic_gist', S >= 1.8).
   - Procedural/tool chatter is pruned without gist crystallization.
   - Inhibitors are pruned without gist crystallization.
   - Fresh actionable decisions (F >= 0.70) are retained and not prematurely crystallized.
5. End-to-End NREM Cycle Integration:
   - SubconsciousDaemon.run_nrem_cycle() executes quiescently and returns accurate telemetry.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path

import torch

from quanta.cognitive.consolidation import SubconsciousConsolidator
from quanta.cognitive.daemon import SleepPhase, SubconsciousDaemon
from quanta.cognitive.memory import (
    GIST_CATEGORY,
    is_actionable_resolution,
)


class TestAdversarialGlymphaticClearance:
    """Challenge dimension 1: Stale file clearance, corrupt files, locks, and time boundaries."""

    def test_corrupt_and_malformed_temp_files_handled(self, tmp_path: Path) -> None:
        """Adversarial scenario: Temp files with 0 bytes, binary junk, and empty content."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        now = time.time()
        # 1. 0-byte stale temp file
        f_zero = tmp_path / ".tmp_zero_bytes_corrupt"
        f_zero.touch()
        os.utime(f_zero, (now - 120.0, now - 120.0))

        # 2. Binary garbage stale temp file
        f_binary = tmp_path / ".tmp_binary_garbage"
        f_binary.write_bytes(b"\x00\xff\xfe\x00\x12\x34\x56\x78\xde\xad\xbe\xef" * 100)
        os.utime(f_binary, (now - 90.0, now - 90.0))

        # 3. Truncated invalid JSON stale temp file
        f_trunc = tmp_path / ".tmp_truncated_json"
        f_trunc.write_text('{"turn_count": 42, "engrams": [{"key": "unfi', encoding="utf-8")
        os.utime(f_trunc, (now - 75.0, now - 75.0))

        # 4. Fresh temp file (<60s)
        f_fresh = tmp_path / ".tmp_fresh_active_write"
        f_fresh.write_text('{"turn_count": 43}', encoding="utf-8")
        os.utime(f_fresh, (now - 10.0, now - 10.0))

        # 5. Non-temp file with stale mtime
        f_regular = tmp_path / "important_data.tmp"
        f_regular.write_text("regular data", encoding="utf-8")
        os.utime(f_regular, (now - 300.0, now - 300.0))

        # Run clearance
        res = consolidator.execute_glymphatic_clearance()

        assert res["files_cleared"] == 3, f"Expected 3 corrupt stale temp files cleared, got {res['files_cleared']}"
        assert not f_zero.exists(), "0-byte stale temp file must be unlinked"
        assert not f_binary.exists(), "Binary garbage stale temp file must be unlinked"
        assert not f_trunc.exists(), "Truncated JSON stale temp file must be unlinked"
        assert f_fresh.exists(), "Fresh temp file must be retained"
        assert f_regular.exists(), "Non-temp file must never be unlinked"

    def test_strict_temporal_boundary_and_future_mtimes(self, tmp_path: Path) -> None:
        """Adversarial scenario: files exactly at boundary, 59.9s, 60.1s, and future drift."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        now = time.time()
        # Stale: 60.5s old -> must be deleted
        f_stale = tmp_path / ".tmp_stale_60_5s"
        f_stale.write_text("stale", encoding="utf-8")
        os.utime(f_stale, (now - 60.5, now - 60.5))

        # Fresh: 59.5s old -> must be PRESERVED
        f_fresh = tmp_path / ".tmp_fresh_59_5s"
        f_fresh.write_text("fresh", encoding="utf-8")
        os.utime(f_fresh, (now - 59.5, now - 59.5))

        # Clock drift: Future mtime (+100s) -> must be PRESERVED
        f_future = tmp_path / ".tmp_future_drift"
        f_future.write_text("future", encoding="utf-8")
        os.utime(f_future, (now + 100.0, now + 100.0))

        # Normal recent file (1s old) -> must be PRESERVED
        f_recent = tmp_path / ".tmp_recent_1s"
        f_recent.write_text("recent", encoding="utf-8")
        os.utime(f_recent, (now - 1.0, now - 1.0))

        res = consolidator.execute_glymphatic_clearance()

        assert res["files_cleared"] == 1
        assert not f_stale.exists(), "File > 60s must be deleted"
        assert f_fresh.exists(), "File < 60s must be preserved"
        assert f_future.exists(), "Future file must be preserved"
        assert f_recent.exists(), "Recent file must be preserved"

    def test_directory_named_with_tmp_prefix_does_not_crash(self, tmp_path: Path) -> None:
        """Adversarial scenario: A subdirectory starting with .tmp_ must NOT crash or be deleted."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        now = time.time()
        # Create a directory named .tmp_subfolder
        d_tmp = tmp_path / ".tmp_scratch_directory"
        d_tmp.mkdir()
        # Put a file inside it
        (d_tmp / "nested_file.txt").write_text("nested", encoding="utf-8")
        os.utime(d_tmp, (now - 120.0, now - 120.0))

        res = consolidator.execute_glymphatic_clearance()

        assert res["files_cleared"] == 0
        assert d_tmp.exists(), "Directory with .tmp_ prefix must not be deleted by file unlinker"
        assert d_tmp.is_dir()
        assert (d_tmp / "nested_file.txt").exists()

    def test_locked_temp_file_advisory_flock_handling(self, tmp_path: Path) -> None:
        """Adversarial scenario: A stale temp file held open with an advisory fcntl flock."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        now = time.time()
        f_locked = tmp_path / ".tmp_locked_by_worker"
        f_locked.write_text("locked content", encoding="utf-8")
        os.utime(f_locked, (now - 120.0, now - 120.0))

        # Open and hold flock
        with open(f_locked, "r+") as fd:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
            try:
                # On POSIX / macOS, unlinking a locked open file removes directory entry while keeping fd open
                res = consolidator.execute_glymphatic_clearance()
                # Clearance should not throw an unhandled exception
                assert res["files_cleared"] >= 0
            finally:
                fcntl.flock(fd.fileno(), fcntl.LOCK_UN)


class TestAdversarialDepotentiatedInhibitors:
    """Challenge dimension 2: Depotentiated inhibitor pruning boundaries and active protection."""

    def test_strict_vinh_and_salience_boundary_matrix(self, tmp_path: Path) -> None:
        """Tests the full 2x2 boundary matrix for V_inh (< 0.20) and Salience (<= 0.50)."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        # 1. Below both: V_inh = 0.199 < 0.20, S = 0.500 <= 0.50 -> MUST BE PURGED
        idx1 = consolidator.memory_manager.record_decision(
            key="inh_purge_1",
            content="Depotentiated candidate 1",
            salience=0.50,
            category="inhibitor",
        )
        consolidator.memory_manager.buffer.buffer[idx1]["metadata"]["v_inh"] = 0.199
        consolidator.memory_manager.buffer.buffer[idx1]["metadata"]["is_core_anchor"] = False

        # 2. V_inh below, but S above: V_inh = 0.199, S = 0.501 > 0.50 -> MUST SURVIVE
        idx2 = consolidator.memory_manager.record_decision(
            key="inh_survive_high_salience",
            content="Active priority inhibitor",
            salience=0.501,
            category="inhibitor",
        )
        consolidator.memory_manager.buffer.buffer[idx2]["metadata"]["v_inh"] = 0.199
        consolidator.memory_manager.buffer.buffer[idx2]["metadata"]["is_core_anchor"] = False

        # 3. V_inh at boundary: V_inh = 0.200, S = 0.50 -> MUST SURVIVE (0.200 is not < 0.20)
        idx3 = consolidator.memory_manager.record_decision(
            key="inh_survive_boundary_vinh",
            content="Boundary V_inh inhibitor",
            salience=0.50,
            category="inhibitor",
        )
        consolidator.memory_manager.buffer.buffer[idx3]["metadata"]["v_inh"] = 0.200
        consolidator.memory_manager.buffer.buffer[idx3]["metadata"]["is_core_anchor"] = False

        # 4. V_inh above, S below: V_inh = 0.201, S = 0.30 -> MUST SURVIVE
        idx4 = consolidator.memory_manager.record_decision(
            key="inh_survive_active_vinh",
            content="Active potency low salience inhibitor",
            salience=0.30,
            category="inhibitor",
        )
        consolidator.memory_manager.buffer.buffer[idx4]["metadata"]["v_inh"] = 0.201
        consolidator.memory_manager.buffer.buffer[idx4]["metadata"]["is_core_anchor"] = False

        # 5. Deeply depotentiated: V_inh = 0.05, S = 0.20 -> MUST BE PURGED
        idx5 = consolidator.memory_manager.record_decision(
            key="inh_purge_deep",
            content="Deeply depotentiated negative rule",
            salience=0.20,
            category="inhibitor",
        )
        consolidator.memory_manager.buffer.buffer[idx5]["metadata"]["v_inh"] = 0.05
        consolidator.memory_manager.buffer.buffer[idx5]["metadata"]["is_core_anchor"] = False

        # 6. Anti-pattern category: V_inh = 0.15, S = 0.40 -> MUST BE PURGED
        idx6 = consolidator.memory_manager.record_decision(
            key="anti_pattern_purge",
            content="Obsolete anti-pattern candidate",
            salience=0.40,
            category="anti_pattern",
        )
        consolidator.memory_manager.buffer.buffer[idx6]["metadata"]["v_inh"] = 0.15
        consolidator.memory_manager.buffer.buffer[idx6]["metadata"]["is_core_anchor"] = False

        assert len(consolidator.memory_manager.buffer.buffer) == 6

        res = consolidator.execute_glymphatic_clearance()

        assert res["inhibitors_pruned"] == 3, f"Expected exactly 3 inhibitors pruned, got {res['inhibitors_pruned']}"
        assert res["compacted_engrams"] == 3

        survivors = {e["metadata"]["key"] for e in consolidator.memory_manager.buffer.buffer}
        assert "inh_purge_1" not in survivors
        assert "inh_purge_deep" not in survivors
        assert "anti_pattern_purge" not in survivors

        assert "inh_survive_high_salience" in survivors
        assert "inh_survive_boundary_vinh" in survivors
        assert "inh_survive_active_vinh" in survivors

    def test_core_anchor_inhibitors_are_strictly_immune(self, tmp_path: Path) -> None:
        """Verifies that is_core_anchor=True or S >= 2.0 grants absolute immunity to inhibitors."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        # Core anchor with is_core_anchor=True and deeply depotentiated V_inh = 0.05, S = 0.20
        idx1 = consolidator.memory_manager.record_decision(
            key="inh_core_explicit",
            content="Never modify production code in external projects",
            salience=0.20,
            category="inhibitor",
        )
        consolidator.memory_manager.buffer.buffer[idx1]["metadata"]["v_inh"] = 0.05
        consolidator.memory_manager.buffer.buffer[idx1]["metadata"]["is_core_anchor"] = True

        # Core anchor by virtue of salience >= 2.0 and low V_inh = 0.08
        idx2 = consolidator.memory_manager.record_decision(
            key="inh_core_high_salience",
            content="Never bypass sandbox without explicit user override",
            salience=2.20,
            category="inhibitor",
        )
        consolidator.memory_manager.buffer.buffer[idx2]["metadata"]["v_inh"] = 0.08
        consolidator.memory_manager.buffer.buffer[idx2]["metadata"]["is_core_anchor"] = False

        res = consolidator.execute_glymphatic_clearance()

        assert res["inhibitors_pruned"] == 0, "Core anchor inhibitors must NEVER be pruned"
        assert len(consolidator.memory_manager.buffer.buffer) == 2

    def test_non_inhibitor_engrams_not_purged_by_glymphatics(self, tmp_path: Path) -> None:
        """Verifies normal general decisions with low salience are NOT touched by glymphatics."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        idx = consolidator.memory_manager.record_decision(
            key="regular_scratchpad",
            content="Ordinary ephemeral note without v_inh",
            salience=0.25,
            category="scratchpad",
        )
        consolidator.memory_manager.buffer.buffer[idx]["metadata"]["is_core_anchor"] = False

        res = consolidator.execute_glymphatic_clearance()

        assert res["inhibitors_pruned"] == 0
        assert len(consolidator.memory_manager.buffer.buffer) == 1
        assert consolidator.memory_manager.buffer.buffer[0]["metadata"]["key"] == "regular_scratchpad"


class TestAdversarialSynapticScaling:
    """Challenge dimension 3: SHY downscaling rate, 0.20 floor clamping, core anchor inviolability."""

    def test_multi_cycle_downscaling_and_floor_clamping(self, tmp_path: Path) -> None:
        """Runs 15 consecutive SHY cycles to verify exponential decay and strict 0.20 floor."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        # 1. Non-core engram starting at S = 1.00
        idx1 = consolidator.memory_manager.record_decision(
            key="engram_1_0",
            content="Non-core engram starting at 1.0",
            salience=1.00,
            category="general",
        )
        consolidator.memory_manager.buffer.buffer[idx1]["metadata"]["is_core_anchor"] = False

        # 2. Non-core engram starting near floor at S = 0.22
        idx2 = consolidator.memory_manager.record_decision(
            key="engram_near_floor",
            content="Non-core engram near floor at 0.22",
            salience=0.22,
            category="general",
        )
        consolidator.memory_manager.buffer.buffer[idx2]["metadata"]["is_core_anchor"] = False

        # Cycle 1:
        # engram_1_0 -> 1.0 * 0.90 = 0.900
        # engram_near_floor -> max(0.20, 0.22 * 0.90) = max(0.20, 0.198) = 0.200
        consolidator.execute_synaptic_homeostasis(shy_factor=0.90)

        mem = {e["metadata"]["key"]: e for e in consolidator.memory_manager.buffer.buffer}
        assert round(float(mem["engram_1_0"]["dopamine_tag"]), 3) == 0.900
        assert round(float(mem["engram_near_floor"]["dopamine_tag"]), 3) == 0.200

        # Cycle 2:
        # engram_1_0 -> 0.90 * 0.90 = 0.810
        # engram_near_floor -> max(0.20, 0.20 * 0.90) = 0.200
        consolidator.execute_synaptic_homeostasis(shy_factor=0.90)

        mem = {e["metadata"]["key"]: e for e in consolidator.memory_manager.buffer.buffer}
        assert round(float(mem["engram_1_0"]["dopamine_tag"]), 3) == 0.810
        assert round(float(mem["engram_near_floor"]["dopamine_tag"]), 3) == 0.200

        # Run 15 more cycles
        for _ in range(15):
            consolidator.execute_synaptic_homeostasis(shy_factor=0.90)

        mem = {e["metadata"]["key"]: e for e in consolidator.memory_manager.buffer.buffer}
        # engram_1_0 must now have hit the 0.20 floor and never dip below 0.20
        assert round(float(mem["engram_1_0"]["dopamine_tag"]), 3) == 0.200
        assert round(float(mem["engram_near_floor"]["dopamine_tag"]), 3) == 0.200

    def test_core_anchors_never_downscaled_across_20_cycles(self, tmp_path: Path) -> None:
        """Verifies all types of core anchors are 100% immune to synaptic downscaling."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        # Type A: Explicit is_core_anchor=True with low salience (S=0.85)
        idx_a = consolidator.memory_manager.record_decision(
            key="core_explicit_low_sal",
            content="Core policy low salience",
            salience=0.85,
            category="policy",
        )
        consolidator.memory_manager.buffer.buffer[idx_a]["metadata"]["is_core_anchor"] = True

        # Type B: Salience >= 2.0 (S=2.40)
        consolidator.memory_manager.record_decision(
            key="core_high_sal",
            content="Core rule high salience",
            salience=2.40,
            category="rule",
        )

        # Type C: Semantic gist (category='semantic_gist', S=1.85)
        consolidator.memory_manager.record_semantic_gist(
            key="core_gist",
            content="Distilled semantic gist",
            salience=1.85,
        )

        # Type D: Subconscious dream insight (category='subconscious_dream', S=2.70)
        consolidator.memory_manager.record_decision(
            key="core_dream",
            content="Consolidated dream insight",
            salience=2.70,
            category="subconscious_dream",
        )

        # Run 20 consecutive SHY cycles
        for _ in range(20):
            consolidator.execute_synaptic_homeostasis(shy_factor=0.90)

        mem = {e["metadata"]["key"]: e for e in consolidator.memory_manager.buffer.buffer}
        assert float(mem["core_explicit_low_sal"]["dopamine_tag"]) == 0.85
        assert float(mem["core_high_sal"]["dopamine_tag"]) == 2.40
        assert float(mem["core_gist"]["dopamine_tag"]) == 1.85
        assert float(mem["core_dream"]["dopamine_tag"]) == 2.70


class TestAdversarialGistCrystallization:
    """Challenge dimension 4: Decaying actionable decisions vs non-actionable chatter."""

    def test_decaying_actionable_resolution_crystallizes_and_protects(self, tmp_path: Path) -> None:
        """Verifies actionable transient decisions decay into permanent semantic gists."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        # 1. Actionable architectural resolution (salience <= 0.8, is_core_anchor=False)
        content_text = (
            "We decided to adopt atomic POSIX replace with fsync on SQLite state persistence "
            "as the universal standard to prevent file corruption across concurrent tasks."
        )
        assert is_actionable_resolution(content_text) is True

        idx = consolidator.memory_manager.record_decision(
            key="decision_step_55",
            content=content_text,
            salience=0.50,
            category="decision",
        )
        consolidator.memory_manager.buffer.buffer[idx]["metadata"]["is_core_anchor"] = False

        # Induce dephasing decay so fidelity < 0.70
        psi_0 = consolidator.memory_manager.buffer.buffer[idx]["pristine_state"]
        noisy = psi_0 + torch.randn_like(psi_0) * 0.8
        noisy = noisy / torch.norm(noisy)
        consolidator.memory_manager.buffer.buffer[idx]["degraded_state"] = noisy

        # Execute Synaptic Homeostasis
        res = consolidator.execute_synaptic_homeostasis(shy_factor=0.90)

        assert res["gists_crystallized"] == 1
        assert res["pruned_count"] >= 1

        all_keys = [e["metadata"]["key"] for e in consolidator.memory_manager.buffer.buffer]
        assert "decision_step_55" not in all_keys, "Original verbatim transient decision must be evicted"

        gist_keys = [k for k in all_keys if k.startswith("gist_")]
        assert len(gist_keys) == 1, f"Expected 1 crystallized gist, found {gist_keys}"

        gist_engram = next(e for e in consolidator.memory_manager.buffer.buffer if e["metadata"]["key"] == gist_keys[0])
        assert gist_engram["metadata"]["category"] == GIST_CATEGORY
        assert float(gist_engram["dopamine_tag"]) >= 1.80
        assert gist_engram["metadata"]["is_core_anchor"] is True

        # Verify quantum fidelity of new gist is pristine (F >= 0.99)
        overlap = torch.vdot(gist_engram["pristine_state"], gist_engram["degraded_state"])
        fid = float((torch.abs(overlap) ** 2).item())
        assert fid >= 0.99

        # Verify that running 5 subsequent SHY cycles NEVER downscales or evicts the crystallized gist
        for _ in range(5):
            consolidator.execute_synaptic_homeostasis(shy_factor=0.90)

        remaining_gists = [e["metadata"]["key"] for e in consolidator.memory_manager.buffer.buffer if e["metadata"]["key"] == gist_keys[0]]
        assert len(remaining_gists) == 1
        assert float(gist_engram["dopamine_tag"]) >= 1.80

    def test_decaying_procedural_tool_chatter_pruned_without_crystallization(self, tmp_path: Path) -> None:
        """Verifies procedural chatter (ls, cat, view_file) is pruned with zero gist crystallization."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        # Procedural tool noise
        chatter_content = "view_file on /path/to/source.py checked line 42 and listing directory contents"
        assert is_actionable_resolution(chatter_content) is False

        idx = consolidator.memory_manager.record_decision(
            key="chatter_step_12",
            content=chatter_content,
            salience=0.45,
            category="scratchpad",
        )
        consolidator.memory_manager.buffer.buffer[idx]["metadata"]["is_core_anchor"] = False

        # Induce dephasing
        psi_0 = consolidator.memory_manager.buffer.buffer[idx]["pristine_state"]
        noisy = psi_0 + torch.randn_like(psi_0) * 0.8
        noisy = noisy / torch.norm(noisy)
        consolidator.memory_manager.buffer.buffer[idx]["degraded_state"] = noisy

        res = consolidator.execute_synaptic_homeostasis(shy_factor=0.90)

        assert res["gists_crystallized"] == 0, "Tool chatter must NOT crystallize into a semantic gist"
        assert res["pruned_count"] == 1

        all_keys = [e["metadata"]["key"] for e in consolidator.memory_manager.buffer.buffer]
        assert "chatter_step_12" not in all_keys
        assert len(all_keys) == 0

    def test_fresh_actionable_decisions_retained_until_decay(self, tmp_path: Path) -> None:
        """Verifies that an actionable decision with high fidelity (F >= 0.70) is downscaled but NOT crystallized."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        idx = consolidator.memory_manager.record_decision(
            key="fresh_decision",
            content="We decided to adopt Redis caching as our permanent architectural standard.",
            salience=0.70,
            category="decision",
        )
        consolidator.memory_manager.buffer.buffer[idx]["metadata"]["is_core_anchor"] = False

        # Pristine fidelity (F = 1.0 >= 0.70)
        res = consolidator.execute_synaptic_homeostasis(shy_factor=0.90)

        assert res["gists_crystallized"] == 0, "Fresh decisions must not be prematurely crystallized"
        assert res["downscaled_count"] == 1
        assert res["pruned_count"] == 0

        # Engram is still present, downscaled to 0.70 * 0.90 = 0.63
        all_keys = [e["metadata"]["key"] for e in consolidator.memory_manager.buffer.buffer]
        assert "fresh_decision" in all_keys
        assert round(float(consolidator.memory_manager.buffer.buffer[0]["dopamine_tag"]), 3) == 0.630


class TestAdversarialEndToEndNREMCycle:
    """Challenge dimension 5: Full daemon NREM cycle execution with mixed stress state."""

    def test_daemon_nrem_cycle_mixed_stress_workload(self, tmp_path: Path) -> None:
        """Executes SubconsciousDaemon.run_nrem_cycle on a state containing corrupt files,
        depotentiated inhibitors, core anchors, and decaying actionable memories."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        daemon = SubconsciousDaemon(
            state_file=state_file,
            nrem_to_rem_ratio=2,
            use_agy_cli=False,
        )

        now = time.time()
        # Seed directory with stale, corrupt, and fresh temp files
        (tmp_path / ".tmp_corrupt_empty").touch()
        os.utime(tmp_path / ".tmp_corrupt_empty", (now - 100.0, now - 100.0))

        (tmp_path / ".tmp_corrupt_bin").write_bytes(b"\xde\xad\xbe\xef" * 50)
        os.utime(tmp_path / ".tmp_corrupt_bin", (now - 120.0, now - 120.0))

        (tmp_path / ".tmp_fresh").write_text("fresh", encoding="utf-8")
        os.utime(tmp_path / ".tmp_fresh", (now - 15.0, now - 15.0))

        # Seed engrams in consolidator
        c = daemon.consolidator

        # 1. Depotentiated inhibitor (must be purged)
        idx_dep = c.memory_manager.record_decision(
            key="inh_depot",
            content="Old negative constraint",
            salience=0.35,
            category="inhibitor",
        )
        c.memory_manager.buffer.buffer[idx_dep]["metadata"]["v_inh"] = 0.12
        c.memory_manager.buffer.buffer[idx_dep]["metadata"]["is_core_anchor"] = False

        # 2. Active inhibitor (must survive)
        idx_act = c.memory_manager.record_decision(
            key="inh_act",
            content="Active critical inhibitor",
            salience=1.50,
            category="inhibitor",
        )
        c.memory_manager.buffer.buffer[idx_act]["metadata"]["v_inh"] = 0.85

        # 3. Core anchor invariant (must survive and NEVER downscale)
        c.memory_manager.record_decision(
            key="core_invariant",
            content="Core architecture rule",
            salience=2.50,
            category="rule",
        )

        # 4. Decaying actionable decision (must crystallize)
        idx_act_dec = c.memory_manager.record_decision(
            key="actionable_redis_rule",
            content="We decided to always adopt Redis caching protocol for fast session indexing.",
            salience=0.45,
            category="decision",
        )
        c.memory_manager.buffer.buffer[idx_act_dec]["metadata"]["is_core_anchor"] = False
        psi = c.memory_manager.buffer.buffer[idx_act_dec]["pristine_state"]
        noisy = psi + torch.randn_like(psi) * 0.9
        c.memory_manager.buffer.buffer[idx_act_dec]["degraded_state"] = noisy / torch.norm(noisy)

        # Run NREM cycle
        result = daemon.run_nrem_cycle()

        assert result is not None
        assert result.success is True
        assert daemon.current_sleep_phase == SleepPhase.WAKE
        assert result.glymphatic_files_cleared == 2
        assert result.depotentiated_inhibitors_pruned == 1
        assert result.gists_crystallized == 1
        assert result.duration_ms >= 0.0

        # Verify state persistence on disk
        assert state_file.exists()
        disk_data = json.loads(state_file.read_text(encoding="utf-8"))
        disk_keys = [e["key"] for e in disk_data["engrams"]]

        assert "inh_depot" not in disk_keys
        assert "inh_act" in disk_keys
        assert "core_invariant" in disk_keys
        assert "actionable_redis_rule" not in disk_keys

        gist_keys = [k for k in disk_keys if k.startswith("gist_")]
        assert len(gist_keys) == 1

        # Confirm fresh temp file remains
        assert (tmp_path / ".tmp_fresh").exists()
        assert not (tmp_path / ".tmp_corrupt_empty").exists()
        assert not (tmp_path / ".tmp_corrupt_bin").exists()
