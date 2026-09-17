"""tests/test_consolidation.py — Unit & Integration tests for SWR memory consolidation.

Validates:
1. SubconsciousConsolidator initialization, buffer wiring, and preexisting state loading.
2. Memory consolidation storing category='subconscious_dream', fidelity >= 0.99,
   salience >= 2.0, and topic tags.
3. Synaptic homeostasis (SHY microglial downscaling & pruning of obsolete engrams).
4. Atomic state persistence to disk without temporary file leakage.
5. Updating existing insights and handling corrupt state files gracefully.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from quanta.cognitive.consolidation import SubconsciousConsolidator
from quanta.cognitive.mind_wander import DreamInsight


class TestSubconsciousConsolidation:
    """Comprehensive test suite for SubconsciousConsolidator."""

    def test_initialization_with_new_state_file(self, tmp_path: Path) -> None:
        """Verifies clean initialization when backing state file does not exist yet."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file, capacity=32)

        assert consolidator.state_file == state_file
        assert consolidator.memory_manager.buffer.capacity == 32
        assert len(consolidator.memory_manager.buffer.buffer) == 0
        assert not state_file.exists()

    def test_consolidate_insight_stores_category_and_fidelity(self, tmp_path: Path) -> None:
        """Verifies insight storage with category='subconscious_dream' and fidelity >= 0.99."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        insight = DreamInsight(
            topic="quantum_zeno_attention",
            seed_question="How to stabilize phase kickback during mind-wandering?",
            synthesis="Projective Zeno observation pins working memory to initial state.",
            confidence=0.98,
            turns_taken=3,
            tokens_used=450,
            anti_rumination_reset_occurred=False,
        )

        success = consolidator.consolidate_insight(insight)
        assert success is True
        assert state_file.exists()

        with open(state_file, encoding="utf-8") as f:
            data = json.load(f)

        assert "engrams" in data
        assert len(data["engrams"]) == 1

        engram = data["engrams"][0]
        assert engram["key"] == "insight_quantum_zeno_attention"
        assert engram["category"] == "subconscious_dream"
        assert engram["description"] == insight.synthesis
        assert engram["content"] == insight.synthesis
        assert engram["fidelity"] >= 0.99
        assert engram["salience"] >= 2.0
        assert engram["tags"] == ["subconscious", "dream", "quantum_zeno_attention"]
        assert engram["topic"] == "quantum_zeno_attention"
        assert engram["confidence"] == 0.98

    def test_atomic_save_leaves_no_temporary_files(self, tmp_path: Path) -> None:
        """Verifies that atomic write commits directly and cleans up all temporary files."""
        state_file = tmp_path / "subfolder" / "state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        insight = DreamInsight(
            topic="atomic_test",
            seed_question="Will this write cleanly?",
            synthesis="Atomic rename guarantees durability.",
            confidence=0.95,
            turns_taken=2,
            tokens_used=200,
            anti_rumination_reset_occurred=False,
        )

        assert consolidator.consolidate_insight(insight) is True
        assert state_file.exists()

        # Check that no sibling .tmp files exist in the parent folder
        tmp_files = list(state_file.parent.glob(".tmp_*"))
        assert len(tmp_files) == 0

    def test_synaptic_homeostasis_prunes_obsolete_engrams(self, tmp_path: Path) -> None:
        """Verifies prune_obsolete() clears decayed memories while preserving dreams."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        # 1. Record an ephemeral low-salience scratchpad item
        consolidator.memory_manager.record_decision(
            key="temp_scratchpad",
            content="Temporary debug note.",
            salience=0.2,  # Low salience <= 0.5
            category="scratchpad",
        )

        # Force biological decay and age progression on the scratchpad engram (exceeding max_age=20)
        consolidator.memory_manager.step(dt=25.0)

        # 2. Consolidate a high-salience subconscious dream insight
        dream = DreamInsight(
            topic="vital_architecture",
            seed_question="What is the core constraint?",
            synthesis="Maintain CSF dielectric shielding around memory buffer.",
            confidence=0.99,
            turns_taken=4,
            tokens_used=600,
            anti_rumination_reset_occurred=False,
        )
        assert consolidator.consolidate_insight(dream) is True

        # Check saved state: temp_scratchpad should be pruned, dream must survive
        data = consolidator.load_state()
        keys = [e["key"] for e in data.get("engrams", [])]

        assert "insight_vital_architecture" in keys
        assert "temp_scratchpad" not in keys

    def test_multiple_insights_consolidation(self, tmp_path: Path) -> None:
        """Verifies multiple different topic insights are all retained in state."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        topics = ["mach_qos", "poisson_renewal", "tom_sociology"]
        for topic in topics:
            insight = DreamInsight(
                topic=topic,
                seed_question=f"Question about {topic}",
                synthesis=f"Synthesis for {topic}",
                confidence=0.92,
                turns_taken=2,
                tokens_used=300,
                anti_rumination_reset_occurred=False,
            )
            assert consolidator.consolidate_insight(insight) is True

        data = consolidator.load_state()
        saved_topics = [e["topic"] for e in data.get("engrams", [])]
        for topic in topics:
            assert topic in saved_topics

    def test_update_existing_topic_insight_supersedes_cleanly(self, tmp_path: Path) -> None:
        """Verifies that consolidating an updated insight with same topic supersedes old content."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file)

        v1 = DreamInsight(
            topic="cache_policy",
            seed_question="Which cache policy to use?",
            synthesis="Initial hypothesis: LRU eviction.",
            confidence=0.70,
            turns_taken=1,
            tokens_used=150,
            anti_rumination_reset_occurred=False,
        )
        consolidator.consolidate_insight(v1)

        v2 = DreamInsight(
            topic="cache_policy",
            seed_question="Which cache policy to use?",
            synthesis="Dialectical resolution: Microglial phagocytic eviction.",
            confidence=0.97,
            turns_taken=4,
            tokens_used=500,
            anti_rumination_reset_occurred=False,
        )
        consolidator.consolidate_insight(v2)

        data = consolidator.load_state()
        engrams = data.get("engrams", [])
        cache_engrams = [e for e in engrams if e["key"] == "insight_cache_policy"]

        # Only one entry should exist (overwritten cleanly without duplicates)
        assert len(cache_engrams) == 1
        expected_desc = "Dialectical resolution: Microglial phagocytic eviction."
        assert cache_engrams[0]["description"] == expected_desc
        assert cache_engrams[0]["confidence"] == 0.97

    def test_state_persistence_and_reload(self, tmp_path: Path) -> None:
        """Verifies that a newly instantiated SubconsciousConsolidator reloads prior state."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        c1 = SubconsciousConsolidator(state_file=state_file)

        insight = DreamInsight(
            topic="persistence_check",
            seed_question="Does state reload?",
            synthesis="Confirmed durable across instances.",
            confidence=0.96,
            turns_taken=3,
            tokens_used=400,
            anti_rumination_reset_occurred=False,
        )
        c1.consolidate_insight(insight)

        # Instantiate second consolidator on same file
        c2 = SubconsciousConsolidator(state_file=state_file)
        insights = c2.get_insights()

        assert len(insights) == 1
        assert insights[0]["topic"] == "persistence_check"
        assert insights[0]["category"] == "subconscious_dream"

    def test_load_corrupted_state_file_handled_gracefully(self, tmp_path: Path) -> None:
        """Verifies resilience against corrupted JSON on disk without uncaught exceptions."""
        state_file = tmp_path / "corrupt_state.json"
        with open(state_file, "w", encoding="utf-8") as f:
            f.write("{corrupt json syntax ...")

        # Must not crash; should initialize with fresh memory
        consolidator = SubconsciousConsolidator(state_file=state_file)
        assert len(consolidator.memory_manager.buffer.buffer) == 0

        insight = DreamInsight(
            topic="recovery_test",
            seed_question="Can we recover?",
            synthesis="State recovered and overwritten with valid JSON.",
            confidence=0.95,
            turns_taken=2,
            tokens_used=250,
            anti_rumination_reset_occurred=False,
        )
        assert consolidator.consolidate_insight(insight) is True
        assert consolidator.load_state()["engrams"][0]["key"] == "insight_recovery_test"

    def test_consolidation_thread_safety_mutex(self, tmp_path: Path) -> None:
        """Verifies internal mutex prevents race conditions under concurrent consolidation."""
        state_file = tmp_path / "concurrent_state.json"
        consolidator = SubconsciousConsolidator(state_file=state_file, capacity=64)

        num_threads = 8
        errors: list[Exception] = []

        def worker(idx: int) -> None:
            try:
                insight = DreamInsight(
                    topic=f"concurrent_topic_{idx}",
                    seed_question=f"Question {idx}?",
                    synthesis=f"Synthesis result from thread {idx}.",
                    confidence=0.95,
                    turns_taken=2,
                    tokens_used=200,
                    anti_rumination_reset_occurred=False,
                )
                res = consolidator.consolidate_insight(insight)
                assert res is True
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        persisted = consolidator.load_state()
        persisted_keys = {e["key"] for e in persisted.get("engrams", [])}
        for i in range(num_threads):
            assert f"insight_concurrent_topic_{i}" in persisted_keys

    def test_consolidation_external_state_merge(self, tmp_path: Path) -> None:
        """Verifies consolidator reloads state to merge external engrams.

        Prevents lost updates when external processes modify the state file.
        """
        state_file = tmp_path / "quanta_cognitive_state.json"
        c1 = SubconsciousConsolidator(state_file=state_file)

        # First consolidate one insight
        in1 = DreamInsight(
            topic="alpha",
            seed_question="Seed A?",
            synthesis="Synthesis Alpha",
            confidence=0.9,
            turns_taken=1,
            tokens_used=100,
            anti_rumination_reset_occurred=False,
        )
        c1.consolidate_insight(in1)

        # External process modifies the state file directly (e.g. hook adding a constraint)
        with open(state_file, encoding="utf-8") as f:
            data = json.load(f)

        data["turn_count"] = 5
        data["engrams"].append({
            "key": "external_rule_spanner",
            "content": "Always use Spanner in production",
            "salience": 2.5,
            "category": "constraint",
            "fidelity": 0.9998,
            "age": 0,
            "tags": ["constraint", "external"],
        })
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # c1 consolidates a second insight — must merge external_rule_spanner without losing it
        in2 = DreamInsight(
            topic="beta",
            seed_question="Seed B?",
            synthesis="Synthesis Beta",
            confidence=0.92,
            turns_taken=2,
            tokens_used=150,
            anti_rumination_reset_occurred=False,
        )
        c1.consolidate_insight(in2)

        final_data = c1.load_state()
        final_keys = {e["key"] for e in final_data.get("engrams", [])}
        assert "insight_alpha" in final_keys
        assert "external_rule_spanner" in final_keys
        assert "insight_beta" in final_keys
        assert final_data["turn_count"] >= 6

    def test_consolidation_coverage_edge_cases(self, tmp_path: Path) -> None:
        """Covers missing branches in consolidation: disk updates, existing item updates."""
        non_existent_file = tmp_path / "non_existent.json"
        c = SubconsciousConsolidator(state_file=non_existent_file)
        loaded = c.load_state()
        assert loaded["turn_count"] == 0
        assert loaded["engrams"] == []

        # Create state file with last_injected_time, last_step_idx, empty item, and confidence
        state_file = tmp_path / "edge_state.json"
        data = {
            "turn_count": 1,
            "last_injected_time": 1000.0,
            "last_step_idx": 42,
            "engrams": [
                {"key": "", "content": ""},  # empty item skipped
                {
                    "key": "rule1",
                    "content": "Initial Rule",
                    "salience": 2.0,
                    "confidence": 0.95,
                },
            ],
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        c2 = SubconsciousConsolidator(state_file=state_file)
        assert c2.last_injected_time == 1000.0
        assert c2.last_step_idx == 42

        # Now simulate external state updating rule1 content and salience
        data["engrams"] = [
            {
                "key": "rule1",
                "content": "Modified Rule",
                "salience": 1.5,
                "confidence": 0.99,
                "tags": ["updated"],
            },
        ]
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        c2._merge_external_state()
        persisted = c2.get_engrams_payload()
        assert any(e["key"] == "rule1" for e in persisted)

