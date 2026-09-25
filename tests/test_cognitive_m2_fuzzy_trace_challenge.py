"""tests/test_cognitive_m2_fuzzy_trace_challenge.py

Empirical stress testing and verification harness for Milestone 2 (R2):
Fuzzy-Trace Semantic Gist Extraction & Consolidation.

Authored by teamwork_preview_challenger_m2_1 (EMPIRICAL CHALLENGER):
1. Procedural noise discrimination: tests that procedural commands ("view_file ...",
   "cat file", "ls -la", "grep ...") are NEVER crystallized and are properly evicted at 70%.
2. Actionable architectural decisions: tests that decisions (SQLite WAL, dynamic route quote,
   POSIX atomic replace, WAF rules) are crystallized into category="semantic_gist",
   S >= 1.8, F=0.9998, is_core_anchor=True.
3. Multi-decision edge cases: tests complex mixtures of actionable decisions and noise in buffer.
4. Longevity & pruning immunity: tests survival across 50-100 biological turns without decaying
   below threshold or being swept by microglia or capacity overflow.
5. Dual-memory parity: compares CognitiveMemoryManager (PyTorch) and FastBiomorphicMemory (stdlib).
"""

from __future__ import annotations

import pytest

from quanta.cognitive.memory import (
    GIST_CATEGORY,
    CognitiveMemoryManager,
    distill_semantic_gist,
)
from quanta.cognitive.memory import (
    is_actionable_resolution as cmem_is_actionable,
)
from scripts.hooks.quanta_subconscious_hook import (
    FastBiomorphicMemory,
)
from scripts.hooks.quanta_subconscious_hook import (
    distill_semantic_gist as hook_distill_gist,
)
from scripts.hooks.quanta_subconscious_hook import (
    is_actionable_resolution as hook_is_actionable,
)


class TestProceduralNoiseDiscrimination:
    """Empirical challenge 1: Procedural commands must NEVER be crystallized."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "view_file src/app/routes/api/page.tsx for checking handlers",
            "view_file /Users/aes/app/routes/api.py",
            "cat src/app/routes/sqlite_manager.py line 40",
            "cat /etc/hosts to check local domain mapping",
            "ls -la src/app/routes/api/ venv and config",
            "ls -la /Users/aes/quanta/src/db/",
            "grep_search for route in src/app",
            "list_dir /Users/aes/Antigravity Projects/Alfa/quanta",
            "find_by_name sqlite in project",
            "[DEBUG] command exited with code 0 while checking rule",
            "checked line 45 in route configuration",
            "reading file src/app/routes.py for inspection",
        ],
    )
    def test_standard_procedural_commands_not_actionable(self, cmd: str) -> None:
        """Verifies standard tool noise commands are rejected by both classifiers."""
        assert not cmem_is_actionable(cmd), f"CognitiveMemoryManager classified as actionable: {cmd}"
        assert not hook_is_actionable(cmd), f"FastBiomorphicMemory classified as actionable: {cmd}"

    @pytest.mark.parametrize(
        "grep_cmd",
        [
            "grep -rn API_KEY src/config.py",
            "grep -rn pattern src/app/routes/",
            "grep -i sqlite src/db/schema.sql",
            "grep -E 'waf|cf.client.bot' nginx.conf",
            "grep -rn posix src/fs/manager.py",
        ],
    )
    def test_grep_procedural_commands_must_never_be_actionable(self, grep_cmd: str) -> None:
        """Tests that 'grep ...' commands containing keyword matches are NEVER crystallized.

        Bug detection: If noise_patterns omits '^grep\\b' while containing keywords like
        'api', 'route', 'sqlite', 'waf', or 'posix', the classifier will incorrectly mark
        procedural search commands as permanent actionable resolutions.
        """
        is_cmem_actionable = cmem_is_actionable(grep_cmd)
        is_hook_actionable = hook_is_actionable(grep_cmd)

        assert not is_cmem_actionable, (
            f"BUG DETECTED: CognitiveMemoryManager classified procedural grep command "
            f"'{grep_cmd}' as actionable resolution!"
        )
        assert not is_hook_actionable, (
            f"BUG DETECTED: FastBiomorphicMemory classified procedural grep command "
            f"'{grep_cmd}' as actionable resolution!"
        )

    def test_procedural_commands_evicted_at_70_percent_without_crystallization(self) -> None:
        """Verifies procedural commands decay past 70% and are evicted with gist_crystallized=False."""
        fmem = FastBiomorphicMemory(capacity=16)
        fmem.record_transient("cmd_view", "view_file src/app/routes/api/page.tsx", salience=0.5)
        fmem.record_transient("cmd_cat", "cat src/app/routes/sqlite_manager.py line 40", salience=0.5)
        fmem.record_transient("cmd_ls", "ls -la src/app/routes/api/ venv config", salience=0.5)

        # Decay past 70% threshold
        for _ in range(150):
            fmem.step(dt=0.2)

        pruned = fmem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert len(pruned) == 3, f"Expected 3 pruned commands, got {len(pruned)}"

        for p in pruned:
            assert not p.get("gist_crystallized"), (
                f"Procedural command {p.get('key')} was incorrectly crystallized!"
            )
            assert p.get("gist_key") is None

        # Verify no gists were recorded
        assert len(fmem.recall_semantic_gists()) == 0

    def test_grep_procedural_command_falsely_crystallized_at_70_percent(self) -> None:
        """Demonstrates end-to-end bug: grep command is falsely crystallized into permanent core memory.

        Per Requirement R2 and challenger objective:
        Procedural commands ("view_file ...", "cat file", "ls -la", "grep ...") must NEVER
        be crystallized and must be evicted at 70% fidelity.
        """
        fmem = FastBiomorphicMemory(capacity=16)
        fmem.record_transient("cmd_grep", "grep -rn API_KEY src/config.py", salience=0.5)

        for _ in range(150):
            fmem.step(dt=0.2)

        pruned = fmem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert len(pruned) == 1
        p = pruned[0]

        # CHALLENGE ASSERTION: This command must NOT be crystallized
        assert not p.get("gist_crystallized"), (
            f"BUG DETECTED: Procedural grep command '{p.get('key')}' was crystallized into "
            f"gist '{p.get('gist_key')}' instead of being evicted!"
        )
        assert len(fmem.recall_semantic_gists()) == 0


class TestActionableArchitecturalResolutionCrystallization:
    """Empirical challenge 2: Actionable architectural decisions must crystallize."""

    ACTIONABLE_CASES = [
        (
            "decision_wal",
            "Selected SQLite WAL mode for high concurrency (Hedef: DB kilitlenmesini onle)",
            "gist_wal",
            "Selected SQLite WAL mode for high concurrency.",
        ),
        (
            "decision_route",
            "Always quote dynamic route segments in shell commands (Hedef: zsh glob error)",
            "gist_route",
            "Always quote dynamic route segments in shell commands.",
        ),
        (
            "decision_posix",
            "POSIX atomic replace with fsync for state persistence (Goal: prevent state corruption)",
            "gist_posix",
            "POSIX atomic replace with fsync for state persistence.",
        ),
        (
            "decision_waf",
            "Bypass managed WAF rules for cf.client.bot using custom skip rule (Target: avoid 403)",
            "gist_waf",
            "Bypass managed WAF rules for cf.client.bot using custom skip rule.",
        ),
    ]

    def test_actionable_resolution_classifier_and_distiller(self) -> None:
        """Verifies actionable decisions are recognized and cleanly distilled."""
        for orig_key, text, exp_gist_key, exp_content in self.ACTIONABLE_CASES:
            assert cmem_is_actionable(text, orig_key)
            assert hook_is_actionable(text, orig_key)

            c_key, c_content = distill_semantic_gist(text, orig_key)
            h_key, h_content = hook_distill_gist(text, orig_key)

            assert c_key == exp_gist_key
            assert c_content == exp_content
            assert h_key == exp_gist_key
            assert h_content == exp_content

    def test_crystallization_in_cognitive_memory_manager(self) -> None:
        """Verifies CognitiveMemoryManager crystallizes transient decisions at 70% threshold."""
        cmem = CognitiveMemoryManager(capacity=16)

        for orig_key, text, _, _ in self.ACTIONABLE_CASES:
            cmem.record_transient_decision(orig_key, text, salience=0.5)

        # Step until fidelity decays below 0.70
        for _ in range(50):
            cmem.step(dt=1.0)

        pruned = cmem.prune_obsolete(fidelity_threshold=0.70)
        assert len(pruned) == len(self.ACTIONABLE_CASES)

        for p in pruned:
            assert p["gist_crystallized"] is True, f"Engram {p['key']} failed to crystallize"
            assert p["gist_key"].startswith("gist_")

        # Verify crystallized semantic gists in memory
        gists = cmem.recall_semantic_gists(top_k=10)
        assert len(gists) == len(self.ACTIONABLE_CASES)

        for g in gists:
            assert g["category"] == GIST_CATEGORY
            assert g["salience"] >= 1.80
            assert g["is_core_anchor"] is True
            assert g["retention_fidelity"] >= 0.999

    def test_crystallization_in_fast_biomorphic_memory(self) -> None:
        """Verifies FastBiomorphicMemory crystallizes transient decisions at 70% threshold."""
        fmem = FastBiomorphicMemory(capacity=16)

        for orig_key, text, _, _ in self.ACTIONABLE_CASES:
            fmem.record_transient(orig_key, text, salience=0.5)

        # Micro-step until decay past 0.70
        for _ in range(150):
            fmem.step(dt=0.2)

        pruned = fmem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert len(pruned) == len(self.ACTIONABLE_CASES)

        for p in pruned:
            assert p.get("gist_crystallized") is True, f"Engram {p.get('key')} failed to crystallize"
            assert str(p.get("gist_key")).startswith("gist_")

        # Verify crystallized semantic gists in memory
        gists = fmem.recall_semantic_gists(top_k=10)
        assert len(gists) == len(self.ACTIONABLE_CASES)

        for g in gists:
            assert g.get("category") == "semantic_gist"
            assert float(g.get("salience", 0)) >= 1.80
            assert g.get("is_core_anchor") is True
            assert abs(float(g.get("fidelity", 0)) - 0.9998) < 1e-4


class TestFuzzyTraceEdgeCaseCrystallization:
    """Empirical challenge 3: Edge cases with multiple transient decisions & noise."""

    def test_mixed_transient_buffer_selective_crystallization(self) -> None:
        """Tests buffer containing a mix of actionable resolutions, commands, and noise."""
        mixed_items = [
            ("act_wal", "Selected SQLite WAL mode for high concurrency", 0.5, True),
            ("act_route", "Always quote dynamic route segments in shell commands", 0.5, True),
            ("act_posix", "POSIX atomic replace with fsync for state persistence", 0.5, True),
            ("act_waf", "Bypass managed WAF rules for cf.client.bot using custom skip rule", 0.5, True),
            ("proc_view", "view_file src/app/routes/api/page.tsx for checking handlers", 0.5, False),
            ("proc_cat", "cat src/app/routes/sqlite_manager.py line 40", 0.5, False),
            ("proc_ls", "ls -la src/app/routes/api/ venv and config", 0.5, False),
            ("noise_short", "temporary note", 0.5, False),
            ("noise_debug", "[DEBUG] command exited with code 0 while checking rule", 0.5, False),
        ]

        fmem = FastBiomorphicMemory(capacity=20)
        for k, c, s, _ in mixed_items:
            fmem.record_transient(k, c, salience=s)

        for _ in range(150):
            fmem.step(dt=0.2)

        pruned = fmem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert len(pruned) == len(mixed_items)

        crystallized_keys = {p["key"] for p in pruned if p.get("gist_crystallized")}
        expected_crystallized = {"act_wal", "act_route", "act_posix", "act_waf"}

        assert crystallized_keys == expected_crystallized, (
            f"Mismatch in crystallized items! Expected {expected_crystallized}, got {crystallized_keys}"
        )

        # Active gists count must match expected
        active_gists = fmem.recall_semantic_gists(top_k=10)
        assert len(active_gists) == 4

    def test_disabled_gist_consolidation_flag(self) -> None:
        """Tests that enable_gist_consolidation=False suppresses crystallization completely."""
        fmem = FastBiomorphicMemory(capacity=16)
        fmem.record_transient(
            "decision_wal",
            "Selected SQLite WAL mode for high concurrency (Hedef: DB kilitlenmesini onle)",
            salience=0.5,
        )

        for _ in range(150):
            fmem.step(dt=0.2)

        pruned = fmem.prune_obsolete(
            fidelity_threshold=0.70, min_salience=0.50, enable_gist_consolidation=False
        )
        assert len(pruned) == 1
        assert pruned[0].get("gist_crystallized") is False
        assert len(fmem.recall_semantic_gists()) == 0


class TestCrystallizedGistLongevityAndPruningImmunity:
    """Empirical challenge 4: Newly crystallized gists survive 50-100 turns."""

    def test_gists_survive_100_turns_in_fast_biomorphic_memory(self) -> None:
        """Verifies crystallized gists maintain F >= 0.999 across 100 subsequent turns."""
        fmem = FastBiomorphicMemory(capacity=16)
        fmem.record_transient(
            "decision_wal",
            "Selected SQLite WAL mode for high concurrency",
            salience=0.5,
        )

        for _ in range(150):
            fmem.step(dt=0.2)

        fmem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        gists = fmem.recall_semantic_gists()
        assert len(gists) == 1
        assert gists[0]["key"] == "gist_wal"

        # Advance 100 subsequent biological conversational turns (dt=1.0)
        for _ in range(100):
            fmem.step(dt=1.0)
            # Periodic pruning sweeps must never evict the gist
            pruned = fmem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
            assert len(pruned) == 0

        gist_after_100 = fmem.recall_semantic_gists()[0]
        fid = float(gist_after_100["fidelity"])
        assert fid >= 0.999, f"Fidelity {fid:.6f} dropped unexpectedly after 100 turns!"
        assert gist_after_100["age"] == 100

    def test_gists_survive_100_turns_in_cognitive_memory_manager(self) -> None:
        """Verifies crystallized gists maintain F >= 0.90 across 100 subsequent turns in PyTorch."""
        cmem = CognitiveMemoryManager(capacity=16)
        cmem.record_transient_decision(
            "decision_wal",
            "Selected SQLite WAL mode for high concurrency",
            salience=0.5,
        )

        for _ in range(50):
            cmem.step(dt=1.0)

        cmem.prune_obsolete(fidelity_threshold=0.70)
        gists = cmem.recall_semantic_gists()
        assert len(gists) == 1
        assert gists[0]["key"] == "gist_wal"

        # Advance 100 subsequent turns
        for _ in range(100):
            cmem.step(dt=1.0)
            pruned = cmem.prune_obsolete(fidelity_threshold=0.70)
            assert len(pruned) == 0

        gist_after_100 = cmem.recall_semantic_gists()[0]
        fid = float(gist_after_100["retention_fidelity"])
        assert fid >= 0.90, f"Fidelity {fid:.6f} dropped unexpectedly in PyTorch manager!"

    def test_gists_immune_to_capacity_overflow_eviction(self) -> None:
        """Verifies that capacity overflow sweeps never evict crystallized semantic gists."""
        fmem = FastBiomorphicMemory(capacity=5)
        fmem.record_semantic_gist(
            "gist_protected", "Always quote dynamic route segments in shell commands", salience=1.85
        )

        # Flood buffer with 10 new transient decisions
        for i in range(10):
            fmem.record_transient(f"transient_{i}", f"Ephemeral item {i} filling the buffer", salience=0.4)

        # The gist must remain intact in memory
        gists = fmem.recall_semantic_gists()
        assert len(gists) == 1
        assert gists[0]["key"] == "gist_protected"


class TestDualMemoryParity:
    """Empirical challenge 5: Parity between PyTorch CognitiveMemoryManager & FastBiomorphicMemory."""

    def test_classification_and_distillation_parity(self) -> None:
        """Verifies identical classification and distillation across both implementations."""
        test_strings = [
            ("Karar: SQLite WAL modunu etkinlestir (Hedef: kilitlenmeyi onle)", "decision_step_1"),
            ("Always quote dynamic route segments in shell commands", "step_quote"),
            ("cat /etc/passwd", "read_file"),
            ("view_file src/quanta/cognitive/memory.py", "tool_step"),
            ("POSIX atomic replace with fsync for state persistence", "decision_fsync"),
            ("ls -la /tmp", "tool_step"),
        ]

        for text, key in test_strings:
            c_act = cmem_is_actionable(text, key)
            h_act = hook_is_actionable(text, key)
            assert c_act == h_act, f"Classification divergence on '{text}': cmem={c_act}, hook={h_act}"

            if c_act:
                c_k, c_c = distill_semantic_gist(text, key)
                h_k, h_c = hook_distill_gist(text, key)
                assert c_k == h_k, f"Key distillation divergence: {c_k} vs {h_k}"
                assert c_c == h_c, f"Content distillation divergence: {c_c} vs {h_c}"
