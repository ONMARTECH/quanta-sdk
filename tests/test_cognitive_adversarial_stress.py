"""Adversarial Empirical Stress Tests for Quanta Contextual Plasticity & Cognitive State.

Authored by empirical challenger to stress-test:
1. Immunity of core anchors under 100+ and 500+ turns of biological stepping
   (F >= 0.999 in FastBiomorphicMemory).
2. Capacity overflow eviction protecting core anchors under high-volume bombardment.
3. Transient engram decay and microglial pruning behavior across salience range [0.2, 0.8].
4. Verification of FastBiomorphicMemory pruning boundary and natural dephasing remediation.
5. Bit-for-bit invariance of 53 existing production engrams in quanta_cognitive_state.json.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import torch

from quanta.cognitive.feedback import (
    ActionOutcome,
    CognitiveFeedbackLoop,
    OutcomeType,
)
from quanta.cognitive.memory import CognitiveMemoryManager
from scripts.hooks.quanta_subconscious_hook import (
    FastBiomorphicMemory,
)

PROD_STATE_PATH = Path("/Users/aes/Antigravity Projects/Alfa/quanta/quanta_cognitive_state.json")


class TestCoreAnchorImmunityStress:
    """Stress tests verifying immunity of core anchors under extended biological turns."""

    def test_core_anchors_immune_100_turns_cognitive_memory_manager(self) -> None:
        """Verifies core anchors survive 100 continuous turns in CognitiveMemoryManager
        without pruning.
        """
        torch.manual_seed(42)
        mem = CognitiveMemoryManager(capacity=16, enable_csf_shielding=True)
        mem.record_decision(
            key="native_first_rule",
            content="Prioritize native CLI and platform tools.",
            salience=2.8,
            is_core_anchor=True,
        )
        mem.record_decision(
            key="scientific_integrity_rule",
            content="Never fabricate benchmarks or data points.",
            salience=2.5,
            is_core_anchor=True,
        )
        mem.record_decision(
            key="executive_summary_rule",
            content="Provide concise executive summaries by default.",
            salience=2.0,
            is_core_anchor=True,
        )

        for _ in range(100):
            mem.step(dt=1.0)

        pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert len(pruned) == 0, f"Core anchors were pruned: {pruned}"

        # Vital recall must retrieve all 3 core anchors
        vital = mem.recall_vital_context(top_k=5)
        vital_keys = [v["key"] for v in vital]
        assert "native_first_rule" in vital_keys
        assert "scientific_integrity_rule" in vital_keys
        assert "executive_summary_rule" in vital_keys

        for v in vital:
            assert v["is_core_anchor"] is True
            assert v["retention_fidelity"] >= 0.90

    def test_core_anchors_immune_100_turns_fast_biomorphic_memory(self) -> None:
        """Verifies core anchors in FastBiomorphicMemory maintain F >= 0.999 across 100 turns."""
        mem = FastBiomorphicMemory(capacity=16)
        mem.record("core_anchor_1", "Invariant rule 1", salience=2.8, is_core_anchor=True)
        mem.record("core_anchor_2", "Invariant rule 2", salience=2.0, is_core_anchor=True)

        for _ in range(100):
            mem.step(dt=1.0)

        for e in mem.engrams:
            assert e["fidelity"] >= 0.999, f"Fidelity dropped below 0.999: {e['fidelity']}"
            assert e["age"] == 100

        pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert len(pruned) == 0

        remaining_keys = [e["key"] for e in mem.engrams]
        assert "core_anchor_1" in remaining_keys
        assert "core_anchor_2" in remaining_keys

    def test_core_anchors_extreme_500_turns(self) -> None:
        """Adversarial stress: 500 biological turns on both memory managers."""
        # FastBiomorphicMemory
        fmem = FastBiomorphicMemory(capacity=8)
        fmem.record("core_extreme", "Deep core constraint", salience=3.0, is_core_anchor=True)
        for _ in range(500):
            fmem.step(dt=1.0)
        f_pruned = fmem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert len(f_pruned) == 0
        assert fmem.engrams[0]["key"] == "core_extreme"
        assert fmem.engrams[0]["fidelity"] >= 0.999, (
            f"Fidelity dropped below 0.999 after 500 turns: {fmem.engrams[0]['fidelity']}"
        )

        # CognitiveMemoryManager
        cmem = CognitiveMemoryManager(capacity=8, enable_csf_shielding=True)
        cmem.record_decision(
            "core_extreme", "Deep core constraint", salience=3.0, is_core_anchor=True
        )
        for _ in range(500):
            cmem.step(dt=1.0)
        c_pruned = cmem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert len(c_pruned) == 0
        assert len(cmem.buffer.buffer) == 1
        assert cmem.buffer.buffer[0].get("metadata", {}).get("key") == "core_extreme"


class TestCapacityOverflowAdversarial:
    """Stress tests validating capacity overflow eviction protecting core anchors."""

    def test_capacity_5_saturated_with_core_anchors_then_10_transients(self) -> None:
        """Capacity=5 saturated with 5 core anchors, then bombarded with 10 transient items."""
        mem = CognitiveMemoryManager(capacity=5, enable_csf_shielding=True)

        for i in range(5):
            mem.record_decision(
                f"core_{i}", f"Core constraint {i}", salience=2.5 + 0.1 * i, is_core_anchor=True
            )

        assert len(mem.buffer.buffer) == 5

        # Bombard with 10 transient decisions
        for j in range(10):
            mem.record_transient_decision(f"trans_{j}", f"Transient hypothesis {j}", salience=0.4)
            # Verify at every single step that all 5 core anchors are intact
            current_keys = [e.get("metadata", {}).get("key") for e in mem.buffer.buffer]
            for i in range(5):
                assert f"core_{i}" in current_keys, (
                    f"Core anchor core_{i} was evicted at transient step {j}!"
                )

        # Final buffer size must be 6 (5 core anchors + 1 latest transient)
        final_keys = [e.get("metadata", {}).get("key") for e in mem.buffer.buffer]
        for i in range(5):
            assert f"core_{i}" in final_keys
        assert "trans_9" in final_keys

    def test_fast_biomorphic_memory_capacity_overflow_bombardment(self) -> None:
        """FastBiomorphicMemory capacity=5 saturated with 5 core anchors,
        bombarded with 50 transients.
        """
        fmem = FastBiomorphicMemory(capacity=5)
        for i in range(5):
            fmem.record(f"core_{i}", f"Core anchor {i}", salience=2.5, is_core_anchor=True)

        for j in range(50):
            fmem.record_transient(f"trans_{j}", f"Transient {j}", salience=0.45)
            core_count = sum(
                1 for e in fmem.engrams if e.get("is_core_anchor") or e.get("salience", 0) >= 2.0
            )
            assert core_count == 5, f"Core anchor evicted during transient step {j}!"

        keys = [e["key"] for e in fmem.engrams]
        for i in range(5):
            assert f"core_{i}" in keys
        assert "trans_49" in keys


class TestTransientDecayAndPruningBehavior:
    """Stress tests evaluating transient engram decay and microglial pruning across both engines."""

    def test_transient_engrams_decay_and_pruned_in_cognitive_memory_manager(self) -> None:
        """In CognitiveMemoryManager, transient engrams (0.2 - 0.8) decay below 0.70
        and are 100% pruned.
        """
        mem = CognitiveMemoryManager(capacity=16, enable_csf_shielding=True)

        # Store core anchor
        mem.record_decision("core_anchor", "Permanent rule", salience=2.8, is_core_anchor=True)

        # Store transient items across salience spectrum [0.2, 0.4, 0.5, 0.6, 0.8]
        test_saliences = [0.2, 0.4, 0.5, 0.6, 0.8]
        for s in test_saliences:
            mem.record_transient_decision(f"trans_{s}", f"Transient hypothesis {s}", salience=s)

        # Biological progression for 50 turns to allow higher salience items (0.6, 0.8)
        # to naturally dephase below 0.70
        for _ in range(50):
            mem.step(dt=1.0)

        # Execute microglial synaptic pruning
        pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        pruned_keys = [p["key"] for p in pruned]

        # All transient engrams should have decayed below 0.70 and been pruned
        for s in test_saliences:
            assert f"trans_{s}" in pruned_keys, f"trans_{s} failed to be pruned: {pruned_keys}"

        # Core anchor must remain
        remaining_keys = [e.get("metadata", {}).get("key") for e in mem.buffer.buffer]
        assert "core_anchor" in remaining_keys
        assert len(remaining_keys) == 1

    def test_fast_biomorphic_memory_pruning_remediated_boundary_and_salience(self) -> None:
        """Verifies remediation of Defect 1:

        Asserts transient decisions (is_core_anchor=False and salience <= 0.8,
        including default hook salience 0.5 and boundary saliences 0.2, 0.6, 0.8)
        ARE 100% pruned when fidelity drops below 0.70, while items above 0.70
        and core anchors remain immune.
        """
        mem = FastBiomorphicMemory(capacity=16)

        # Record transient decisions with salience in [0.2, 0.5, 0.6, 0.8]
        mem.record_transient("trans_0.2", "Low salience transient", salience=0.2)
        mem.record_transient("trans_0.5_hook", "Default hook transient", salience=0.5)
        mem.record_transient("trans_0.6", "Mid salience transient", salience=0.6)
        mem.record_transient("trans_0.8", "High transient salience", salience=0.8)

        # Record transient decision that stays above threshold
        mem.record_transient("trans_above_threshold", "Fresh transient", salience=0.5)

        # Record core anchor with depressed fidelity to stress test immunity
        mem.record(
            "core_anchor_protected", "Permanent core anchor", salience=2.5, is_core_anchor=True
        )

        # Simulate sub-threshold dephasing for target transient items
        for e in mem.engrams:
            if e["key"] in ("trans_0.2", "trans_0.5_hook", "trans_0.6", "trans_0.8"):
                e["fidelity"] = 0.69  # Just below 0.70 threshold
            elif e["key"] == "trans_above_threshold":
                e["fidelity"] = 0.75  # Above threshold
            elif e["key"] == "core_anchor_protected":
                e["fidelity"] = 0.50  # Artificially depressed core anchor

        # Execute microglial pruning using exact arguments called by hook at line 739
        pruned_keys = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)

        # Assert transient decisions below 0.70 ARE cleanly pruned
        assert "trans_0.2" in pruned_keys
        assert "trans_0.5_hook" in pruned_keys
        assert "trans_0.6" in pruned_keys
        assert "trans_0.8" in pruned_keys
        assert len(pruned_keys) == 4

        # Assert above-threshold transient and core anchor survive
        survivor_keys = [e["key"] for e in mem.engrams]
        assert "trans_above_threshold" in survivor_keys
        assert "core_anchor_protected" in survivor_keys
        assert len(survivor_keys) == 2

    def test_fast_biomorphic_memory_transient_natural_decay_under_step(self) -> None:
        """Verifies remediation of Defect 2:

        Asserts transient decisions naturally dephase and decay under FastBiomorphicMemory.step()
        across [0.2, 0.8] without unconditional CSF over-shielding, falling below 0.70
        and getting 100% pruned, while core anchors remain pristine (F >= 0.999).
        """
        mem = FastBiomorphicMemory(capacity=16)

        # Store core anchor
        mem.record("core_anchor", "Permanent invariant", salience=2.8, is_core_anchor=True)

        # Store transient items across salience range
        test_saliences = [0.2, 0.4, 0.5, 0.6, 0.8]
        for s in test_saliences:
            mem.record_transient(f"trans_{s}", f"Transient item {s}", salience=s)

        # Step 100 conversational turns
        for _turn in range(1, 101):
            mem.step(dt=1.0)

        # Check core anchor retained pristine fidelity F >= 0.999
        core_engram = next(e for e in mem.engrams if e["key"] == "core_anchor")
        assert core_engram["fidelity"] >= 0.999, (
            f"Core anchor fidelity decayed: {core_engram['fidelity']}"
        )

        # Check all transient items naturally decayed below 0.70 threshold
        for s in test_saliences:
            e = next(eng for eng in mem.engrams if eng["key"] == f"trans_{s}")
            assert e["fidelity"] < 0.70, (
                f"trans_{s} failed to decay below 0.70: fidelity={e['fidelity']}"
            )

        # Execute microglial synaptic pruning
        pruned_keys = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)

        # 100% of transient engrams must be pruned
        for s in test_saliences:
            assert f"trans_{s}" in pruned_keys, f"trans_{s} was not pruned: {pruned_keys}"
        assert "core_anchor" not in pruned_keys

        # Final survivors: only the core anchor
        survivors = [e["key"] for e in mem.engrams]
        assert survivors == ["core_anchor"]


class TestProductionEngramBitForBitInvariance:
    """Verifies that all 53 existing production engrams retain bit-for-bit invariance."""

    def test_53_production_engrams_integrity_and_invariance(self, tmp_path: Path) -> None:
        """Verifies 53 engrams exist, all have salience >= 2.0, and retain bit-for-bit
        fields under feedback.
        """
        assert PROD_STATE_PATH.exists()
        with open(PROD_STATE_PATH, encoding="utf-8") as f:
            state = json.load(f)

        engrams = state.get("engrams", [])
        assert len(engrams) == 53, f"Expected exactly 53 engrams, found {len(engrams)}"

        # Verify all qualify as Core Anchors
        for _idx, eng in enumerate(engrams):
            assert "key" in eng and len(eng["key"]) > 0
            assert "content" in eng and len(eng["content"]) > 0
            salience = float(eng.get("salience", 0.0))
            assert salience >= 2.0, f"Engram '{eng['key']}' has salience {salience} < 2.0!"

        # Create temporary working copy to test feedback loops
        temp_state = tmp_path / "quanta_cognitive_state.json"
        shutil.copy(PROD_STATE_PATH, temp_state)

        initial_dict = {e["key"]: copy.deepcopy(e) for e in engrams}
        feedback_loop = CognitiveFeedbackLoop(state_path=temp_state)

        # Run 30 feedback updates
        for step in range(1, 31):
            rule = "native_first_rule" if step % 2 == 0 else "executive_summary_rule"
            ot = OutcomeType.SUCCESS if step % 3 != 0 else OutcomeType.TOOL_FAILURE
            outcome = ActionOutcome(
                step_idx=step,
                tool_name="run_command",
                outcome_type=ot,
                rule_violated=rule if ot != OutcomeType.SUCCESS else None,
                details={"rule_key": rule},
            )
            feedback_loop.process_outcomes([outcome])

        # Reload state and verify all 53 original engrams are 100% preserved
        with open(temp_state, encoding="utf-8") as f:
            updated_state = json.load(f)

        updated_engrams = updated_state.get("engrams", [])
        # All 53 original engrams must exist (any additional are dynamic feedback rule records)
        assert len(updated_engrams) >= 53
        updated_dict = {e["key"]: e for e in updated_engrams}

        for k, orig in initial_dict.items():
            assert k in updated_dict, f"Production engram {k} was lost!"
            upd = updated_dict[k]
            # Bit-for-bit check on immutable semantic fields
            assert orig["content"] == upd["content"]
            assert orig.get("category") == upd.get("category")
            assert orig.get("topic") == upd.get("topic")
            assert orig.get("tags") == upd.get("tags")
            assert orig.get("salience") == upd.get("salience")
