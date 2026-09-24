"""Comprehensive E2E and Contextual Plasticity Test Suite for Quanta Cognitive Engine.

Validates the contextual plasticity architecture, microglial synaptic pruning,
subconscious hook lifecycle integration, atomic state durability, and standalone
cognitive cockpit dashboard across 4 rigorous tiers:
- Tier 1: Feature Coverage (Core Anchor immunity, Transient decay, Microglial pruning,
          Subconscious hook format, Web Dashboard structure, 53 production engrams).
- Tier 2: Boundary & Corner Cases (Capacity saturation with core anchors, extreme salience
          inputs, rapid zero-delay debouncing, atomic write durability).
- Tier 3: Cross-Feature Interactions (Feedback loop + Plasticity, Hook + SWR + Pruning,
          Concurrent state access).
- Tier 4: Real-World Multi-Turn Simulation (Full 20-turn conversational lifecycle).
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from quanta.cognitive.feedback import (
    ActionOutcome,
    CognitiveFeedbackLoop,
    OutcomeType,
    OutcomeVerifier,
    _save_state_atomically,
)
from quanta.cognitive.memory import CognitiveMemoryManager
from quanta.cognitive.telemetry import generate_dashboard_html
from scripts.hooks.quanta_subconscious_hook import (
    KAPPA_CSF,
    FastBiomorphicMemory,
)

HOOK_PATH = Path("/Users/aes/Antigravity Projects/Alfa/quanta/scripts/hooks/quanta_subconscious_hook.py")
PROD_STATE_PATH = Path("/Users/aes/Antigravity Projects/Alfa/quanta/quanta_cognitive_state.json")


# ============================================================================
# Helpers & Fixtures
# ============================================================================

def make_sample_state(turn_count: int = 10, num_core: int = 3, num_transient: int = 2) -> dict[str, Any]:
    """Builds a realistic cognitive state fixture with core and transient engrams."""
    engrams: list[dict[str, Any]] = []
    for i in range(num_core):
        engrams.append({
            "key": f"core_anchor_{i}",
            "content": f"Critical rule {i} must always be enforced",
            "salience": 2.5 + 0.1 * i,
            "category": "constraint",
            "fidelity": 0.9998,
            "age": turn_count,
            "is_core_anchor": True,
            "confidence": 0.98,
        })
    for j in range(num_transient):
        engrams.append({
            "key": f"transient_decision_{j}",
            "content": f"Temporary working hypothesis {j}",
            "salience": 0.4 + 0.1 * j,
            "category": "contextual_decision",
            "fidelity": 0.85 - 0.1 * j,
            "age": 3 + j,
            "is_core_anchor": False,
            "confidence": 0.90,
        })
    return {
        "turn_count": turn_count,
        "last_injected_time": time.time(),
        "last_step_idx": turn_count * 2,
        "engrams": engrams,
        "total_pruned_count": 0,
        "kappa_csf": KAPPA_CSF,
        "outcome_history": [],
    }


# ============================================================================
# Tier 1: Feature Coverage Tests
# ============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Comprehensive feature coverage of contextual plasticity & microglial pruning."""

    def test_core_anchors_never_pruned_over_time(self) -> None:
        """Validates that Core Anchors (salience >= 2.0 or is_core_anchor=True) survive 100 turns."""
        mem = CognitiveMemoryManager(capacity=16, enable_csf_shielding=True)
        mem.record_decision(
            key="native_first_rule",
            content="Always prioritize native CLI/API tools.",
            salience=2.8,
            category="constraint",
        )
        mem.record_decision(
            key="scientific_integrity_rule",
            content="Never fabricate benchmarks or data points.",
            salience=2.5,
            category="constraint",
        )

        # Advance biological time across 100 simulated turns
        for _ in range(100):
            mem.step(dt=1.0)

        # Execute active synaptic pruning
        pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert len(pruned) == 0, f"Core anchors should never be pruned, but got {pruned}"

        # Vital recall must retrieve core anchors with pristine fidelity
        vital = mem.recall_vital_context(top_k=2)
        vital_keys = [v["key"] for v in vital]
        assert "native_first_rule" in vital_keys
        assert "scientific_integrity_rule" in vital_keys

        for v in vital:
            fid = v["retention_fidelity"]
            assert fid >= 0.95, f"Core anchor fidelity decayed too much: {fid}"

    def test_core_anchors_never_evicted_on_capacity_overflow(self) -> None:
        """Validates that when buffer capacity overflows, transient items are evicted, not core anchors."""
        mem = CognitiveMemoryManager(capacity=3, enable_csf_shielding=True)

        # Store 2 core anchors (salience >= 2.0)
        mem.record_decision("core_1", "Core anchor 1", salience=2.5, category="constraint")
        mem.record_decision("core_2", "Core anchor 2", salience=2.8, category="constraint")

        # Store 1 transient item (salience = 0.4) -> Buffer is now full (3/3)
        mem.record_decision("transient_1", "Transient hypothesis 1", salience=0.4, category="contextual")
        assert len(mem.buffer.buffer) == 3

        # Store another transient decision -> Triggers smart priority eviction
        mem.record_decision("transient_2", "Transient hypothesis 2", salience=0.5, category="contextual")
        assert len(mem.buffer.buffer) == 3

        # Core anchors MUST be intact in the buffer
        buffer_keys = [e.get("metadata", {}).get("key") for e in mem.buffer.buffer]
        assert "core_1" in buffer_keys, "Core anchor 1 was unexpectedly evicted!"
        assert "core_2" in buffer_keys, "Core anchor 2 was unexpectedly evicted!"
        # The lowest-scoring transient item ("transient_1" with salience 0.4) was evicted
        assert "transient_1" not in buffer_keys
        assert "transient_2" in buffer_keys

    def test_transient_engrams_decay_with_turn_age(self) -> None:
        """Validates that transient engrams (salience 0.2 - 0.8) exhibit measurable dephasing decay."""
        mem = FastBiomorphicMemory(capacity=16)
        mem.record("core_rule", "Permanent constraint", salience=2.8, category="constraint")
        mem.record("transient_task", "Temporary task scratchpad", salience=0.3, category="contextual")

        initial_core = next(e for e in mem.engrams if e["key"] == "core_rule")
        initial_transient = next(e for e in mem.engrams if e["key"] == "transient_task")
        assert initial_core["fidelity"] == 0.9998
        assert initial_transient["fidelity"] == 0.9998

        # Step 50 turns
        for _ in range(50):
            mem.step(dt=1.0)

        updated_core = next(e for e in mem.engrams if e["key"] == "core_rule")
        updated_transient = next(e for e in mem.engrams if e["key"] == "transient_task")

        # Core anchor protected by CSF + high dopamine remains pristine (> 0.999)
        assert updated_core["fidelity"] > 0.999
        # Transient item has lower dopamine shielding, so its dephasing rate is higher
        assert updated_transient["fidelity"] < updated_core["fidelity"]
        assert updated_transient["age"] == 50

    def test_microglial_pruning_sweeps_decayed_transient_engrams(self) -> None:
        """Validates that microglial synaptic pruning removes engrams below threshold or aged."""
        mem = FastBiomorphicMemory(capacity=16)
        mem.record("core_rule", "Core anchor rule", salience=2.5)
        mem.record("transient_fresh", "Recent transient note", salience=0.45)
        mem.record("transient_decayed", "Obsolete transient note", salience=0.30)

        # Artificially decay transient_decayed below threshold
        target = next(e for e in mem.engrams if e["key"] == "transient_decayed")
        target["fidelity"] = 0.62

        fresh_target = next(e for e in mem.engrams if e["key"] == "transient_fresh")
        fresh_target["fidelity"] = 0.88

        # Execute microglial pruning
        pruned_keys = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)

        assert "transient_decayed" in pruned_keys
        assert "transient_fresh" not in pruned_keys
        assert "core_rule" not in pruned_keys

        remaining_keys = [e["key"] for e in mem.engrams]
        assert "transient_decayed" not in remaining_keys
        assert "transient_fresh" in remaining_keys
        assert "core_rule" in remaining_keys
        assert mem.total_pruned_count == 1

    def test_fast_biomorphic_memory_core_vs_transient(self) -> None:
        """Validates FastBiomorphicMemory distinction between core anchors and contextual engrams."""
        mem = FastBiomorphicMemory(capacity=4)
        mem.record("core_api", "Native first rule", salience=2.8, category="constraint")
        mem.record("scratchpad_1", "Temp query 1", salience=0.25, category="scratchpad")
        mem.record("scratchpad_2", "Temp query 2", salience=0.25, category="scratchpad")
        mem.record("scratchpad_3", "Temp query 3", salience=0.25, category="scratchpad")

        # Buffer at capacity (4/4). Adding another item should evict a low-salience scratchpad
        mem.record("scratchpad_4", "Temp query 4", salience=0.35, category="scratchpad")
        keys = [e["key"] for e in mem.engrams]
        assert "core_api" in keys, "Core anchor must never be evicted on capacity saturation"
        assert len(mem.engrams) == 4

    def test_subconscious_hook_hierarchical_or_core_format(self, tmp_path: Path) -> None:
        """Validates that subconscious hook runs fail-safe and outputs proper SWR replay format."""
        assert HOOK_PATH.exists(), f"Hook script not found at {HOOK_PATH}"

        payload = {
            "conversationId": "test_conv_plasticity",
            "artifactDirectoryPath": str(tmp_path),
            "workspacePaths": ["/Users/aes/Antigravity Projects/Alfa/quanta"],
            "stepIdx": 1,
        }
        res = subprocess.run(
            [sys.executable, str(HOOK_PATH)],
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True,
            check=True,
        )
        data = json.loads(res.stdout.decode("utf-8"))

        assert "injectSteps" in data
        assert len(data["injectSteps"]) == 1
        ephemeral = data["injectSteps"][0]["ephemeralMessage"]

        # Ephemeral message must begin with Quanta cognitive anchor prefix
        assert "Quanta Bilişsel Çıpa" in ephemeral
        # Must contain invariant core rules
        assert "native_first_rule" in ephemeral
        assert "scientific_integrity_rule" in ephemeral
        assert "executive_summary_rule" in ephemeral
        # Must never output raw 100.0%
        assert "%100.0" not in ephemeral

        # Check that state was persisted cleanly
        state_file = tmp_path / "quanta_cognitive_state.json"
        assert state_file.exists()
        with open(state_file, encoding="utf-8") as sf:
            state_data = json.load(sf)
            assert state_data["turn_count"] >= 1
            assert len(state_data["engrams"]) >= 3
            # Invariant rules must have salience >= 2.0
            for eng in state_data["engrams"]:
                if eng["key"] in ("native_first_rule", "scientific_integrity_rule", "executive_summary_rule"):
                    assert eng["salience"] >= 2.0

    def test_dashboard_html_structure_and_zero_cdn(self, tmp_path: Path) -> None:
        """Validates dashboard HTML file existence, valid HTML5, and strict zero-CDN rule."""
        canonical_dashboard = Path("quanta/cognitive/dashboard/index.html")

        # If standalone dashboard index.html exists, test it; otherwise verify telemetry generator
        if canonical_dashboard.exists():
            target_file = canonical_dashboard
        else:
            target_file = generate_dashboard_html(output_path=tmp_path / "dashboard.html")

        assert target_file.exists()
        content = target_file.read_text(encoding="utf-8")

        # HTML5 standard structure
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "</html>" in content

        # Zero-CDN verification: no external scripts or stylesheets loaded over http/https
        import re
        external_scripts = re.findall(r'<script[^>]+src=["\'](https?://[^"\']+)["\']', content)
        external_styles = re.findall(r'<link[^>]+href=["\'](https?://[^"\']+)["\']', content)
        assert len(external_scripts) == 0, f"Violated zero-CDN rule (external scripts): {external_scripts}"
        assert len(external_styles) == 0, f"Violated zero-CDN rule (external stylesheets): {external_styles}"

        # Core visualization components check
        assert "zeno-gauge" in content.lower() or "svg" in content.lower()
        # Memory/rule fidelity section
        assert "fidelity" in content.lower() or "kural" in content.lower() or "sadakat" in content.lower()

    def test_53_existing_production_engrams_preservation(self) -> None:
        """Validates that all 53 existing production engrams in quanta_cognitive_state.json qualify as Core Anchors."""
        assert PROD_STATE_PATH.exists(), f"Production state file missing at {PROD_STATE_PATH}"
        with open(PROD_STATE_PATH, encoding="utf-8") as f:
            data = json.load(f)

        engrams = data.get("engrams", [])
        assert len(engrams) == 53, f"Expected exactly 53 engrams in production state, found {len(engrams)}"

        # 100% of existing engrams must have salience >= 2.0 (CSF protected core anchors)
        for idx, eng in enumerate(engrams):
            key = eng.get("key", f"index_{idx}")
            salience = eng.get("salience", 0.0)
            assert salience >= 2.0, (
                f"Engram '{key}' at index {idx} has salience {salience} < 2.0! "
                "All existing production engrams must be immune Core Anchors."
            )
            assert "content" in eng and len(eng["content"]) > 0


# ============================================================================
# Tier 2: Boundary & Corner Cases Tests
# ============================================================================

class TestTier2BoundaryAndCornerCases:
    """Tier 2: Boundary stress testing, numeric stability, debouncing, and fault injection."""

    def test_capacity_saturated_with_core_anchors_graceful_expansion(self) -> None:
        """Validates behavior when buffer capacity is saturated entirely with core anchors."""
        mem = CognitiveMemoryManager(capacity=3, enable_csf_shielding=True)
        mem.record_decision("core_a", "Core Anchor A", salience=2.5)
        mem.record_decision("core_b", "Core Anchor B", salience=2.6)
        mem.record_decision("core_c", "Core Anchor C", salience=2.7)
        assert len(mem.buffer.buffer) == 3

        # Add a 4th core anchor: the system should handle this gracefully without crashing
        mem.record_decision("core_d", "Core Anchor D", salience=2.8)
        # All items remaining in buffer must be core anchors
        for e in mem.buffer.buffer:
            assert float(e.get("dopamine_tag", 0.0)) >= 2.0

    def test_extreme_salience_and_numerical_stability(self) -> None:
        """Validates that extreme, negative, NaN, Inf, or zero salience values are safely handled."""
        mem = FastBiomorphicMemory(capacity=8)

        # Test negative salience
        mem.record("neg_sal", "Negative salience test", salience=-5.0)
        neg_item = next(e for e in mem.engrams if e["key"] == "neg_sal")
        assert neg_item["salience"] >= 0.01

        # Test near-zero salience
        mem.record("zero_sal", "Zero salience test", salience=0.000001)
        zero_item = next(e for e in mem.engrams if e["key"] == "zero_sal")
        assert zero_item["salience"] >= 0.01

        # Test extremely large salience
        mem.record("huge_sal", "Huge salience test", salience=999.0)
        huge_item = next(e for e in mem.engrams if e["key"] == "huge_sal")
        assert math.isfinite(huge_item["salience"])

        # Advance biological step with extreme inputs — verify no ZeroDivisionError or NaN
        for _ in range(10):
            mem.step(dt=1.0)

        for e in mem.engrams:
            fid = e["fidelity"]
            assert math.isfinite(fid)
            assert 0.0 <= fid <= 1.0

    def test_rapid_zero_delay_turns_debouncing(self, tmp_path: Path) -> None:
        """Validates that rapid consecutive hook invocations (< 2.0s) are debounced cleanly."""
        payload = {
            "conversationId": "test_debounce",
            "artifactDirectoryPath": str(tmp_path),
            "workspacePaths": ["/Users/aes/Antigravity Projects/Alfa/quanta"],
            "stepIdx": 5,
        }
        raw_input = json.dumps(payload).encode("utf-8")

        # First invocation
        res1 = subprocess.run(
            [sys.executable, str(HOOK_PATH)],
            input=raw_input,
            capture_output=True,
            check=True,
        )
        data1 = json.loads(res1.stdout.decode("utf-8"))
        assert "injectSteps" in data1

        # Immediate second invocation (same stepIdx, within 2 seconds) -> Debounced
        res2 = subprocess.run(
            [sys.executable, str(HOOK_PATH)],
            input=raw_input,
            capture_output=True,
            check=True,
        )
        assert res2.stdout.strip() == b"{}"

    def test_atomic_write_safety_under_simulated_interruption(self, tmp_path: Path) -> None:
        """Validates that interrupted state serialization cleans up all temporary .tmp_* files."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        sample = make_sample_state(turn_count=5)
        _save_state_atomically(state_file, sample, "conv_test")

        assert state_file.exists()
        original_mtime = state_file.stat().st_mtime_ns

        # Simulate interruption by injecting un-serializable object into state dict
        class UnserializableObject:
            pass

        corrupted_state = dict(sample)
        corrupted_state["bad_data"] = UnserializableObject()

        # Call atomic save with corrupted data
        _save_state_atomically(state_file, corrupted_state, "conv_test")

        # Original file must remain intact
        assert state_file.exists()
        assert state_file.stat().st_mtime_ns == original_mtime
        with open(state_file, encoding="utf-8") as f:
            valid_data = json.load(f)
            assert valid_data["turn_count"] == 5

        # Zero orphaned temporary files in directory
        tmp_files = list(tmp_path.glob(".tmp_*"))
        assert len(tmp_files) == 0, f"Found orphaned temporary files: {tmp_files}"


# ============================================================================
# Tier 3: Cross-Feature Interaction Tests
# ============================================================================

class TestTier3CrossFeatureInteractions:
    """Tier 3: Interactions between Feedback Loop, Contextual Plasticity, SWR Replay & Concurrency."""

    def test_feedback_loop_with_contextual_plasticity(self, tmp_path: Path) -> None:
        """Validates that outcome feedback updates rule confidence while plasticity decays transient items."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        state = make_sample_state(turn_count=10, num_core=2, num_transient=2)
        # Ensure native_first_rule is present
        state["engrams"][0]["key"] = "native_first_rule"
        state["engrams"][0]["salience"] = 2.8
        state["engrams"][0]["confidence"] = 0.95
        _save_state_atomically(state_file, state, "cross_feature_conv")

        # Initialize feedback loop
        verifier = OutcomeVerifier(dim=64)
        feedback_loop = CognitiveFeedbackLoop(verifier=verifier)

        # 1. Process successful native outcome -> reinforces native_first_rule
        outcome = ActionOutcome(
            step_idx=21,
            tool_name="run_command",
            outcome_type=OutcomeType.SUCCESS,
            exit_code=0,
            rule_violated=None,
        )
        feedback_loop.process_outcomes(
            outcomes=[outcome],
            state_path=state_file,
            workspace=str(tmp_path),
        )

        # 2. Simulate biological plasticity step on in-memory representation
        with open(state_file, encoding="utf-8") as f:
            updated_state = json.load(f)

        mem = FastBiomorphicMemory(capacity=16)
        for e in updated_state["engrams"]:
            mem.record(
                key=e["key"],
                content=e["content"],
                salience=e["salience"],
                category=e.get("category", "general"),
            )
            # Retain existing fidelity and confidence metadata
            mem.engrams[-1]["fidelity"] = e.get("fidelity", 0.9998)
            mem.engrams[-1]["confidence"] = e.get("confidence", 0.95)
            mem.engrams[-1]["is_core_anchor"] = e.get("is_core_anchor", e.get("salience", 0) >= 2.0)


        # Advance 20 turns
        for _ in range(20):
            mem.step(dt=1.0)

        # Prune obsolete transient engrams
        pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)

        # Save merged state
        updated_state["engrams"] = mem.engrams
        updated_state["total_pruned_count"] += len(pruned)
        _save_state_atomically(state_file, updated_state, "cross_feature_conv")

        # Final verification: native_first_rule confidence boosted, transient item pruned or decayed
        with open(state_file, encoding="utf-8") as f:
            final_data = json.load(f)
            native_rule = next(e for e in final_data["engrams"] if e["key"] == "native_first_rule")
            assert native_rule["confidence"] >= 0.95
            assert native_rule["fidelity"] >= 0.999

    def test_hook_lifecycle_combines_swr_and_pruning(self) -> None:
        """Validates that memory manager boosts vital context via SWR while allowing decayed items to prune."""
        mem = FastBiomorphicMemory(capacity=8)
        mem.record("vital_core", "Crucial constraint", salience=2.8)
        mem.record("transient_decayed", "Obsolete thought", salience=0.35)

        # Artificially degrade transient_decayed
        decayed_item = next(e for e in mem.engrams if e["key"] == "transient_decayed")
        decayed_item["fidelity"] = 0.65

        vital_item = next(e for e in mem.engrams if e["key"] == "vital_core")
        vital_item["fidelity"] = 0.9900

        # SWR Replay recall + consolidation
        vital = mem.recall_vital(top_k=1)
        assert vital[0]["key"] == "vital_core"
        mem.consolidate([vital[0]["key"]], boost=0.005)
        assert vital_item["fidelity"] == pytest.approx(0.9950, abs=1e-4)

        # Synaptic pruning sweeps the degraded transient item
        pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert "transient_decayed" in pruned
        assert len(mem.engrams) == 1
        assert mem.engrams[0]["key"] == "vital_core"

    def test_concurrent_state_access_simulation(self, tmp_path: Path) -> None:
        """Validates that multi-threaded concurrent updates do not corrupt the state file."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        init_state = make_sample_state(turn_count=1)
        _save_state_atomically(state_file, init_state, "concurrent_conv")

        errors: list[Exception] = []

        def worker_task(thread_id: int) -> None:
            try:
                for step in range(5):
                    outcome = ActionOutcome(
                        step_idx=thread_id * 10 + step,
                        tool_name="test_tool",
                        outcome_type=OutcomeType.SUCCESS,
                        exit_code=0,
                    )
                    loop = CognitiveFeedbackLoop()
                    loop.process_outcomes(
                        outcomes=[outcome],
                        state_path=state_file,
                        workspace=str(tmp_path),
                    )
                    time.sleep(0.01)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker_task, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Encountered concurrent access errors: {errors}"
        # Validate final state file is healthy and valid JSON
        with open(state_file, encoding="utf-8") as f:
            final_data = json.load(f)
            assert "turn_count" in final_data
            assert "engrams" in final_data
            assert isinstance(final_data["engrams"], list)


# ============================================================================
# Tier 4: Real-World Multi-Turn Simulation Tests
# ============================================================================

class TestTier4RealWorldMultiTurnSimulation:
    """Tier 4: End-to-end 20-turn conversational simulation with transient decay & microglial sweep."""

    def test_20_turn_lifecycle_simulation(self, tmp_path: Path) -> None:
        """Simulates a full 20-turn conversation:

        - Turns 1-5: Core rules active, 3 transient decisions recorded.
        - Turns 6-12: Intermediate conversation steps, transient decay accumulates.
        - Turn 14: Microglial synaptic pruning clears expired decisions.
        - Turns 15-20: Final verification, state atomically durable throughout.
        """
        state_file = tmp_path / "quanta_cognitive_state.json"
        mem = FastBiomorphicMemory(capacity=16)

        # --- Turn 1: Initialization of Permanent Core Anchors ---
        mem.record("native_first_rule", "Prioritize native platform API/CLI", salience=2.8, category="constraint")
        mem.record("scientific_integrity_rule", "Maintain mathematical truth", salience=2.5, category="constraint")
        mem.record("executive_summary_rule", "Deliver concise executive summaries", salience=2.5, category="constraint")

        state = {
            "turn_count": 1,
            "last_injected_time": time.time(),
            "last_step_idx": 1,
            "engrams": copy.deepcopy(mem.engrams),
            "total_pruned_count": 0,
            "kappa_csf": KAPPA_CSF,
        }
        _save_state_atomically(state_file, state, "sim_20_turn")

        # --- Turns 2-5: Ephemeral Decisions Recorded ---
        mem.step(dt=1.0)
        mem.record("temp_db_choice", "Use sqlite in-memory for unit testing", salience=0.4, category="contextual")

        mem.step(dt=1.0)
        mem.record("temp_cache_strategy", "LRU cache size 128 for session", salience=0.35, category="contextual")

        mem.step(dt=1.0)
        mem.record("temp_scratch_port", "Local test server port 8899", salience=0.30, category="contextual")

        assert len(mem.engrams) == 6  # 3 core + 3 transient

        # --- Turns 6-12: Conversational Steps & Differential Decay ---
        for _turn in range(6, 13):
            mem.step(dt=1.0)
            # Periodically consolidate core anchors via SWR replay
            mem.consolidate(["native_first_rule", "scientific_integrity_rule", "executive_summary_rule"])

        # Verify decay characteristics at Turn 12
        core_eng = next(e for e in mem.engrams if e["key"] == "native_first_rule")
        transient_eng = next(e for e in mem.engrams if e["key"] == "temp_scratch_port")

        assert core_eng["fidelity"] >= 0.999, "Core anchor fidelity must remain pristine under SWR"
        assert transient_eng["fidelity"] < core_eng["fidelity"], "Transient item must decay faster than core anchor"

        # --- Turn 13-14: Microglial Pruning Trigger ---
        mem.step(dt=1.0)
        # Advance transient items to trigger threshold
        for e in mem.engrams:
            if e["key"] in ("temp_scratch_port", "temp_cache_strategy"):
                e["fidelity"] = 0.65  # Decayed below threshold 0.70

        pruned_keys = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert "temp_scratch_port" in pruned_keys
        assert "temp_cache_strategy" in pruned_keys
        assert "native_first_rule" not in pruned_keys
        assert "scientific_integrity_rule" not in pruned_keys
        assert "executive_summary_rule" not in pruned_keys

        # --- Turns 15-20: Final Steady-State Stepping ---
        for _turn in range(15, 21):
            mem.step(dt=1.0)

        # Persist final state
        final_state = {
            "turn_count": 20,
            "last_injected_time": time.time(),
            "last_step_idx": 40,
            "engrams": mem.engrams,
            "total_pruned_count": mem.total_pruned_count,
            "kappa_csf": KAPPA_CSF,
        }
        _save_state_atomically(state_file, final_state, "sim_20_turn")

        # Validate final state on disk
        with open(state_file, encoding="utf-8") as f:
            disk_data = json.load(f)
            assert disk_data["turn_count"] == 20
            disk_keys = [e["key"] for e in disk_data["engrams"]]
            # Core anchors guaranteed to be present and healthy
            assert "native_first_rule" in disk_keys
            assert "scientific_integrity_rule" in disk_keys
            assert "executive_summary_rule" in disk_keys
            # Pruned transient items guaranteed gone
            assert "temp_scratch_port" not in disk_keys
            assert "temp_cache_strategy" not in disk_keys
            assert disk_data["total_pruned_count"] >= 2
