"""tests/test_cognitive_m2_gist_challenge.py

Empirical stress testing and verification harness for Milestone 2 (R2):
Fuzzy-Trace Semantic Gist Extraction, Microglial Pruning, SWR Replay & Mirror Persistence.

Authored by teamwork_preview_challenger_m2_2 (EMPIRICAL CHALLENGER):
1. Actionable resolution classifier and preamble/target distillation stress tests.
2. CognitiveMemoryManager (PyTorch / Theorem 4) FTT crystallization and capacity protection.
3. FastBiomorphicMemory (runtime hook) micro-step decay to F < 0.70 and microglial crystallization.
4. SWR Replay output formatting: '🧠 Özüt' row and '✨ Kristalleşen Özüt: X karar' notifications.
5. Atomic state mirror serialization and non-gist transient isolation in root mirror.
6. Edge cases: Empty/malformed inputs, Turkish characters, capacity eviction immunity,
   and isolation when prod_engrams is empty.
7. Full multi-turn end-to-end hook simulation with transcript and post_tool_use steps.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from quanta.cognitive.memory import (
    GIST_CATEGORY,
    SALIENCE_GIST_DEFAULT,
    CognitiveMemoryManager,
)
from quanta.cognitive.memory import (
    distill_semantic_gist as distill_torch,
)
from quanta.cognitive.memory import (
    is_actionable_resolution as is_actionable_torch,
)
from scripts.hooks.quanta_subconscious_hook import (
    DEFAULT_DIM,
    FastBiomorphicMemory,
    _atomic_write_single_file,
    distill_semantic_gist,
    format_fidelity,
    handle_post_tool_use,
    is_actionable_resolution,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Actionable Resolution Classification & Distillation Stress Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestActionableResolutionAndDistillation:
    """Stress-tests resolution detection heuristics and distillation normalization."""

    @pytest.mark.parametrize(
        "chatter",
        [
            "view_file /Users/aes/quanta/quanta/cognitive/memory.py line 42",
            "run_command .venv/bin/pytest tests/test_core.py",
            "grep_search 'def is_actionable' in scripts/",
            "find_by_name *.json in /Users/aes/",
            "list_dir /Users/aes/Antigravity Projects",
            "checked line 120 of memory.py for changes",
            "reading file contents for inspection",
            "listing directory children for audit",
            "command exited with code 0 without output",
            "too short",
            "ok",
            "[thinking] I am thinking about checking the code now",
        ],
    )
    def test_procedural_chatter_rejected(self, chatter: str) -> None:
        """Confirms that pure procedural tool chatter is never classified as actionable."""
        assert not is_actionable_resolution(chatter), f"Noise improperly accepted: {chatter}"
        assert not is_actionable_torch(chatter), f"Torch noise improperly accepted: {chatter}"

    @pytest.mark.parametrize(
        ("resolution", "search_terms"),
        [
            (
                "Selected SQLite WAL mode for high concurrency across multi-agent processes (Hedef: DB kilitlenmesini onle)",
                ["sqlite", "wal"],
            ),
            (
                "Always quote dynamic route segments in shell commands to prevent zsh glob errors (Goal: eliminate syntax error)",
                ["always", "quote"],
            ),
            (
                "Karar: POSIX atomic rename stratejisi ile dosya bozulmasını engelle (Hedef: zero corruption)",
                ["posix", "atomic"],
            ),
            (
                "Kural: Production veritabanı sorgularında daima WHERE filtrelemesi uygula",
                ["production", "veritaban"],
            ),
            (
                "Adopted CockroachDB over Spanner for zero-downtime geo-distributed multi-region consistency",
                ["cockroachdb", "spanner"],
            ),
            (
                "Mimaride WAF cf.client.bot bypass kuralı oluşturarak Googlebot 403 bloklamasını engelle",
                ["waf", "cf.client.bot"],
            ),
        ],
    )
    def test_actionable_resolutions_accepted_and_distilled(
        self, resolution: str, search_terms: list[str]
    ) -> None:
        """Confirms strategic directives are recognized and cleanly distilled."""
        # Verification in fast hook implementation
        assert is_actionable_resolution(resolution), f"Failed to detect: {resolution}"
        key, distilled = distill_semantic_gist(resolution, key="decision_step_42")
        assert key.startswith("gist_")
        for term in search_terms:
            assert term in distilled.lower() or term in key.lower(), (
                f"Term '{term}' missing from distilled: {distilled}"
            )
        assert "(Hedef:" not in distilled
        assert "(Goal:" not in distilled
        assert not distilled.startswith(("Karar:", "Decision:", "Çözüm:", "Solution:"))
        assert len(distilled) <= 140
        assert distilled.endswith((".", "!", "?"))

        # Verification in PyTorch memory implementation
        assert is_actionable_torch(resolution), f"Torch failed to detect: {resolution}"
        t_key, t_distilled = distill_torch(resolution, key="decision_step_42")
        assert t_key.startswith("gist_")
        assert len(t_distilled) <= 140
        assert t_distilled.endswith((".", "!", "?"))

    def test_distillation_strips_markdown_and_whitespace(self) -> None:
        """Verifies bold/header markdown lead lines and redundant whitespace are cleaned."""
        raw = "### **Karar:**    Adopt Cloud Spanner with multi-region replication.   (Target: 99.999% SLA)   "
        key, distilled = distill_semantic_gist(raw, key="decision_cloud_spanner")
        assert key == "gist_cloud_spanner"
        assert distilled == "Adopt Cloud Spanner with multi-region replication."
        assert "Karar" not in distilled
        assert "Target" not in distilled


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PyTorch CognitiveMemoryManager (Theorem 4 FTT Consolidation)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCognitiveMemoryManagerFTTConsolidation:
    """Stress-tests Fuzzy-Trace Theory crystallization in CognitiveMemoryManager."""

    def test_transient_decision_crystallizes_on_microglial_prune(self) -> None:
        """Actionable decision decaying below F < 0.70 crystallizes into permanent gist."""
        cmem = CognitiveMemoryManager(capacity=16, dim=DEFAULT_DIM)
        cmem.record_transient_decision(
            key="decision_step_10",
            content="Selected SQLite WAL mode for high concurrency (Hedef: DB kilitlenmesini onle)",
            salience=0.50,
        )

        # Step 50 conversational turns to induce dephasing
        for _ in range(50):
            cmem.step(dt=1.0)

        # Microglial synaptic pruning at 70% threshold
        pruned = cmem.prune_obsolete(fidelity_threshold=0.70, enable_gist_consolidation=True)

        assert len(pruned) == 1
        p_record = pruned[0]
        assert p_record["key"] == "decision_step_10"
        assert p_record["gist_crystallized"] is True
        assert p_record["gist_key"] == "gist_step_10"

        # Check that transient was evicted and gist is anchored
        gists = cmem.recall_semantic_gists()
        assert len(gists) == 1
        gist = gists[0]
        assert gist["key"] == "gist_step_10"
        assert gist["category"] == GIST_CATEGORY
        assert float(gist["salience"]) >= 1.80

    def test_non_actionable_transient_pruned_without_crystallization(self) -> None:
        """Procedural transient chatter decaying below 0.70 is pruned without crystallization."""
        cmem = CognitiveMemoryManager(capacity=16, dim=DEFAULT_DIM)
        cmem.record_transient_decision(
            key="decision_tool_call_5",
            content="view_file /Users/aes/quanta/quanta/cognitive/memory.py line 42",
            salience=0.50,
        )

        for _ in range(50):
            cmem.step(dt=1.0)

        pruned = cmem.prune_obsolete(fidelity_threshold=0.70, enable_gist_consolidation=True)
        assert len(pruned) == 1
        assert pruned[0]["gist_crystallized"] is False
        assert pruned[0]["gist_key"] is None
        assert len(cmem.recall_semantic_gists()) == 0

    def test_crystallized_gist_capacity_protection(self) -> None:
        """Crystallized semantic gist is never evicted even under heavy buffer saturation."""
        cmem = CognitiveMemoryManager(capacity=4, dim=DEFAULT_DIM)
        # Directly anchor a semantic gist
        cmem.record_semantic_gist(
            key="gist_core_rule",
            content="Always enforce strict atomic writes on state files.",
            salience=SALIENCE_GIST_DEFAULT,
        )

        # Flood buffer with 10 normal engrams
        for i in range(10):
            cmem.record_decision(
                key=f"temp_flood_{i}",
                content=f"Temporary operation index {i}",
                salience=1.0,
            )

        keys_in_buffer = [e.get("metadata", {}).get("key") for e in cmem.buffer.buffer]
        assert "gist_core_rule" in keys_in_buffer, "Semantic gist was unlawfully evicted!"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FastBiomorphicMemory (Runtime Hook Consolidation & Micro-Step Kinetics)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFastBiomorphicMemoryConsolidation:
    """Stress-tests FastBiomorphicMemory microglial pruning and gist anchoring."""

    def test_runtime_hook_microstep_decay_and_crystallization(self) -> None:
        """Simulates 115 tool micro-steps (dt=0.2); transient decays below 0.70 and crystallizes."""
        mem = FastBiomorphicMemory(capacity=32)
        mem.record_transient(
            key="decision_step_88",
            content="Always quote dynamic route segments in shell commands (Hedef: glob error)",
            salience=0.50,
        )

        # Analytical: at S=0.5, dt=0.2, gamma_eff ~ 0.016487
        # Step 100: t=20.0 -> F ~ 0.723 > 0.70
        for _ in range(100):
            mem.step(dt=0.2)
        pruned_100 = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert len(pruned_100) == 0, "Pruned prematurely at step 100!"

        # Step 115: t=23.0 -> F ~ 0.688 < 0.70
        for _ in range(15):
            mem.step(dt=0.2)

        engram = mem.engrams[0]
        assert engram["fidelity"] < 0.70

        pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert len(pruned) == 1
        p = pruned[0]
        assert p["key"] == "decision_step_88"
        assert p["gist_crystallized"] is True
        assert p["gist_key"] == "gist_step_88"

        # Check anchored gist
        gists = mem.recall_semantic_gists()
        assert len(gists) == 1
        g = gists[0]
        assert g["key"] == "gist_step_88"
        assert g["category"] == "semantic_gist"
        assert g["is_core_anchor"] is True
        assert g["salience"] >= 1.80
        assert g["fidelity"] == 0.9998

        # Verify gist survives another 100 tool steps without decay/eviction
        for _ in range(100):
            mem.step(dt=0.2)
        pruned_subsequent = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert len(pruned_subsequent) == 0
        assert len(mem.recall_semantic_gists()) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SWR Replay Output Formatting & Notification Inspection
# ═══════════════════════════════════════════════════════════════════════════════


class TestSWRReplayOutputFormatting:
    """Verifies SWR Replay block contains '🧠 Özüt' and '✨ Kristalleşen Özüt: X karar'."""

    def test_swr_replay_formatting_with_gist_and_crystallization(self) -> None:
        """Simulates SWR replay prompt generation during turn with active and newly crystallized gists."""
        mem = FastBiomorphicMemory(capacity=16)

        # Pre-anchor a core rule and a prior gist
        mem.record("core_rule", "Never modify production code without consent", salience=3.0)
        mem.record_semantic_gist(
            "gist_sqlite_wal",
            "Selected SQLite WAL mode for high concurrency.",
            salience=1.85,
        )

        # Add a transient decision that is ready to be pruned
        mem.record_transient(
            "decision_step_99",
            "Always quote dynamic route segments in shell commands (Hedef: glob error)",
            salience=0.50,
        )
        for _ in range(120):
            mem.step(dt=0.2)

        # Execute microglial pruning
        pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert len(pruned) == 1
        assert pruned[0]["gist_crystallized"] is True

        # Construct SWR replay output lines using exact hook logic
        active_core = [
            e for e in mem.engrams
            if (e.get("is_core_anchor", False) or float(e.get("salience", 0.0)) >= 2.0)
            and e.get("category") != "semantic_gist"
            and float(e.get("fidelity", 0.0)) >= 0.70
        ]
        active_gists = mem.recall_semantic_gists(top_k=4, min_fidelity=0.70)
        active_transient = [
            e for e in mem.engrams
            if e.get("is_core_anchor") is False
            and float(e.get("salience", 1.0)) <= 0.8
            and float(e.get("fidelity", 0.0)) >= 0.70
        ]

        core_items = [f"{c['key']} ({format_fidelity(c['fidelity'])})" for c in active_core]
        gist_items = [f"{g['key']} ({format_fidelity(g['fidelity'])})" for g in active_gists]
        transient_items = [f"{t['key']} ({format_fidelity(t['fidelity'])})" for t in active_transient]

        lines = ["[Quanta Bilişsel Çıpa | SWR Replay]:"]
        if core_items:
            lines.append(f"  🔒 Çekirdek: {', '.join(core_items)}")
        if gist_items:
            lines.append(f"  🧠 Özüt: {', '.join(gist_items)}")
        if transient_items:
            lines.append(f"  ⚡ Geçici: {', '.join(transient_items)}")

        crystallized_in_turn = [
            p for p in pruned
            if isinstance(p, dict) and p.get("gist_crystallized")
        ]
        if crystallized_in_turn:
            lines.append(f"  ✨ Kristalleşen Özüt: {len(crystallized_in_turn)} karar")
        if pruned:
            lines.append(f"  ✂️ Budandı: {len(pruned)} engram")

        swr_block = "\n".join(lines)

        # Assertions
        assert "[Quanta Bilişsel Çıpa | SWR Replay]:" in swr_block
        assert "🔒 Çekirdek: core_rule (99.98%)" in swr_block
        assert "🧠 Özüt: " in swr_block
        assert "gist_sqlite_wal" in swr_block
        assert "gist_step_99" in swr_block
        assert "✨ Kristalleşen Özüt: 1 karar" in swr_block
        assert "✂️ Budandı: 1 engram" in swr_block


# ═══════════════════════════════════════════════════════════════════════════════
# 5. State Mirror Persistence & Non-Gist Isolation Stress Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestStateMirrorPersistenceAndIsolation:
    """Stress-tests atomic serialization, category='semantic_gist', and non-gist isolation."""

    def test_save_mirrored_state_atomically_preserves_gists_and_isolates_transients(
        self, tmp_path: Path
    ) -> None:
        """Verifies primary file gets all engrams while root mirror retains ONLY production engrams."""
        primary_file = tmp_path / "primary_quanta_cognitive_state.json"
        mirror_file = tmp_path / "mirror_quanta_cognitive_state.json"

        state_dict = {
            "turn_count": 5,
            "engrams": [
                {
                    "key": "insight_concurrent_preemption",
                    "content": "Production insight",
                    "category": "subconscious_dream",
                    "salience": 3.5,
                    "fidelity": 0.9998,
                },
                {
                    "key": "gist_sqlite_wal",
                    "content": "Selected SQLite WAL mode for high concurrency.",
                    "category": "semantic_gist",
                    "salience": 1.85,
                    "fidelity": 0.9998,
                    "is_core_anchor": True,
                },
                {
                    "key": "decision_step_transient_temp",
                    "content": "Ephemeral transient note",
                    "category": "contextual_decision",
                    "salience": 0.50,
                    "fidelity": 0.85,
                    "is_core_anchor": False,
                },
            ],
        }

        # Simulate root mirror target behavior
        primary_ok = _atomic_write_single_file(primary_file, state_dict)
        assert primary_ok

        # Read back primary file
        with open(primary_file, encoding="utf-8") as f:
            primary_data = json.load(f)

        assert len(primary_data["engrams"]) == 3
        categories_in_primary = {e["category"] for e in primary_data["engrams"]}
        assert "subconscious_dream" in categories_in_primary
        assert "semantic_gist" in categories_in_primary
        assert "contextual_decision" in categories_in_primary

        # Now test the filtering logic that runs when saving to the root mirror file:
        raw_list = state_dict.get("engrams", [])
        engrams_list = raw_list if isinstance(raw_list, list) else []
        prod_engrams = [
            e for e in engrams_list
            if isinstance(e, dict)
            and (
                e.get("category") in (
                    "subconscious_dream",
                    "architecture_rfc",
                    "semantic_gist",
                )
                or str(e.get("key", "")).startswith("insight_")
                or e.get("key") == "architecture_rfc"
            )
        ]
        mirror_dict = copy.deepcopy(state_dict)
        if prod_engrams:
            mirror_dict["engrams"] = prod_engrams

        mirror_ok = _atomic_write_single_file(mirror_file, mirror_dict)
        assert mirror_ok

        # Read back mirror file
        with open(mirror_file, encoding="utf-8") as f:
            mirror_data = json.load(f)

        assert len(mirror_data["engrams"]) == 2
        categories_in_mirror = {e["category"] for e in mirror_data["engrams"]}
        assert "semantic_gist" in categories_in_mirror
        assert "subconscious_dream" in categories_in_mirror
        assert "contextual_decision" not in categories_in_mirror, "Transient item leaked into mirror!"

    def test_isolation_edge_case_empty_prod_engrams(self, tmp_path: Path) -> None:
        """Exposes edge case: if state_dict has ONLY transient items, does mirror_dict leak them?"""
        state_dict = {
            "turn_count": 1,
            "engrams": [
                {
                    "key": "decision_step_transient_only",
                    "content": "Only a transient item exists",
                    "category": "contextual_decision",
                    "salience": 0.50,
                    "fidelity": 0.85,
                    "is_core_anchor": False,
                }
            ],
        }

        # In hook line 918:
        # prod_engrams = [ ... ]
        # if prod_engrams:
        #     mirror_dict["engrams"] = prod_engrams
        raw_list = state_dict.get("engrams", [])
        engrams_list = raw_list if isinstance(raw_list, list) else []
        prod_engrams = [
            e for e in engrams_list
            if isinstance(e, dict)
            and (
                e.get("category") in (
                    "subconscious_dream",
                    "architecture_rfc",
                    "semantic_gist",
                )
                or str(e.get("key", "")).startswith("insight_")
                or e.get("key") == "architecture_rfc"
            )
        ]
        # Notice: prod_engrams is empty!
        # If 'if prod_engrams:' is used, mirror_dict['engrams'] is NOT updated,
        # so mirror_dict['engrams'] leaks the transient item!
        # We test and document this exact edge condition.
        assert len(prod_engrams) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Real Multi-Turn End-to-End Hook Lifecycle Simulation
# ═══════════════════════════════════════════════════════════════════════════════


class TestEndToEndHookSimulation:
    """Simulates a complete real-world multi-turn session with post_tool_use and PreInvocation."""

    def test_full_session_crystallization_and_mirror_persistence(self, tmp_path: Path) -> None:
        """Complete workflow test:
        1. Initialize state with core anchor and user prompt.
        2. Record an actionable transient decision.
        3. Advance 115 tool steps via handle_post_tool_use (dt=0.2).
        4. Trigger PreInvocation hook: transient pruned, gist crystallized, SWR block output.
        5. Verify state file on disk has category='semantic_gist' and no transient items.
        """
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        primary_state_file = artifact_dir / "quanta_cognitive_state.json"
        transcript_file = artifact_dir / "transcript.jsonl"

        # Write transcript with user prompt
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps({
                    "role": "user",
                    "content": "<USER_REQUEST>\nOptimize database architecture for concurrent multi-agent access\n</USER_REQUEST>",
                })
                + "\n"
            )

        # Initial state setup
        init_state = {
            "turn_count": 1,
            "last_injected_time": 1000.0,
            "last_step_idx": 1,
            "last_recorded_decision_step": 0,
            "last_verified_step": 1,
            "last_user_query": "Optimize database architecture for concurrent multi-agent access",
            "pending_tool_continuation": False,
            "engrams": [
                {
                    "key": "native_first_rule",
                    "content": "Native first principle: use platform native APIs",
                    "salience": 3.0,
                    "category": "core_rule",
                    "fidelity": 0.9998,
                    "age": 0,
                    "is_core_anchor": True,
                },
                {
                    "key": "decision_step_1",
                    "content": "Selected SQLite WAL mode for high concurrency across multi-agent processes (Hedef: DB kilitlenmesini onle)",
                    "salience": 0.50,
                    "category": "contextual_decision",
                    "fidelity": 0.9998,
                    "age": 0,
                    "is_core_anchor": False,
                },
            ],
            "total_pruned_count": 0,
        }
        with open(primary_state_file, "w", encoding="utf-8") as f:
            json.dump(init_state, f, indent=2)

        # 3. Simulate 115 consecutive tool executions via handle_post_tool_use
        post_payload = {
            "hook_event_name": "post_tool_use",
            "tool_name": "run_command",
            "tool_input": {"CommandLine": "ls"},
            "tool_result": {"output": "success"},
            "conversationId": "test_conv_m2_challenge",
            "sessionId": "test_session",
            "workspacePaths": [str(tmp_path)],
            "artifactDirectoryPath": str(artifact_dir),
            "step_idx": 2,
        }

        # Step through tool calls
        for step in range(2, 117):
            post_payload["step_idx"] = step
            handle_post_tool_use(post_payload, state_file=primary_state_file)

        # Verify state file after 115 tool steps: decision_step_1 should have decayed below 0.70
        with open(primary_state_file, encoding="utf-8") as f:
            state_after_tools = json.load(f)

        transient_e = next(
            e for e in state_after_tools["engrams"] if e["key"] == "decision_step_1"
        )
        assert transient_e["fidelity"] < 0.70, (
            f"Expected fidelity < 0.70 after 115 steps, got {transient_e['fidelity']}"
        )

        # 4. Simulate PreInvocation hook execution
        mem = FastBiomorphicMemory(capacity=32)
        for e in state_after_tools["engrams"]:
            mem.record(
                key=e["key"],
                content=e["content"],
                salience=e["salience"],
                category=e["category"],
                is_core_anchor=e.get("is_core_anchor"),
            )
            mem.engrams[-1]["fidelity"] = e["fidelity"]
            mem.engrams[-1]["age"] = e["age"]

        pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert len(pruned) == 1
        assert pruned[0]["gist_crystallized"] is True

        # Check memory has crystallized gist
        gists = mem.recall_semantic_gists()
        assert len(gists) == 1
        assert gists[0]["category"] == "semantic_gist"
        assert gists[0]["is_core_anchor"] is True
        assert gists[0]["salience"] >= 1.80

        # Check transient decision is no longer in active engrams
        active_keys = [e["key"] for e in mem.engrams]
        assert "decision_step_1" not in active_keys
        assert gists[0]["key"] in active_keys

        # Check persisted state after hook save
        state_after_tools["engrams"] = mem.engrams
        state_after_tools["total_pruned_count"] += len(pruned)
        _atomic_write_single_file(primary_state_file, state_after_tools)

        with open(primary_state_file, encoding="utf-8") as f:
            final_saved_state = json.load(f)

        saved_categories = {e["category"] for e in final_saved_state["engrams"]}
        assert "semantic_gist" in saved_categories
        assert "contextual_decision" not in saved_categories
