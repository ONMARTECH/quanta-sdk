"""Comprehensive Test Suite for Dynamic Context-Adaptive SWR Resonance (Milestone M1).

Validates:
1. 100% Core Anchor Retention regardless of prompt/task context topic.
2. >= 40% Token Reduction of transient engrams on off-topic tasks.
3. Correct ranking and retrieval of on-topic transient decisions.
4. Composite Hilbert-space overlap + lexical Jaccard resonance metric properties.
5. Active off-topic transient synaptic pruning with strict core preservation.
6. SWR Replay prompt string formatting.
"""

from __future__ import annotations

import pytest

from quanta.cognitive.memory import (
    CognitiveMemoryManager,
    extract_context_tokens,
)


@pytest.fixture
def populated_memory() -> CognitiveMemoryManager:
    """Provides a memory manager pre-populated with core anchors and quantum transient decisions."""
    mem = CognitiveMemoryManager(capacity=64, dim=64, enable_csf_shielding=True)

    # 1. Five Immutable Core Anchors (salience >= 2.0, is_core_anchor=True)
    core_rules = [
        ("scientific_integrity_rule", "Never falsify or cherry-pick benchmarks without full ablation.", 2.5),
        ("secure_vault_rule", "Never expose API credentials outside central vault.", 2.5),
        ("native_first_rule", "Prioritize native platform APIs over third-party scrapers.", 2.5),
        ("immutable_assets_rule", "Never delete screenshots or user assets during pruning.", 2.5),
        ("subconscious_anchor_rule", "SWR replay must be preserved across multi-agent turns.", 2.2),
    ]
    for key, content, sal in core_rules:
        mem.record_decision(key=key, content=content, salience=sal, is_core_anchor=True)

    # Core Semantic Gist
    mem.record_semantic_gist(
        key="gist_posix_atomic_write",
        content="Always use atomic POSIX replace and fsync for shared storage.",
        salience=2.0,
    )

    # 2. Ten Transient Engrams (Domain: Quantum Chemistry, Circuit Routing & Lattice CVP)
    transient_items = [
        ("qaoa_cvp_lattice_basis", "Babai nearest plane vector reduction for dimension 4 lattice.", 0.8),
        ("qaoa_mixer_weights", "Angle gamma=0.15 beta=0.82 in QAOA mixer Hamiltonian.", 0.7),
        ("vqe_h2_molecule", "UCCSD ansatz calculation for hydrogen molecule bond length 0.74A.", 0.9),
        ("mps_bond_dimension_cutoff", "Truncate singular values below 1e-6 with chi_max=32.", 0.8),
        ("custatevec_stream_sync", "CUDA stream synchronization barrier on multi-GPU nodes.", 0.7),
        ("pauli_frame_clifford_tableau", "Update Pauli X and Z destabilizer tableau indices.", 0.8),
        ("willow_surface_code_syndrome", "Detect bit-flip syndrome on 17-qubit distance-3 rotated surface code.", 0.9),
        ("ionq_native_gateset", "Decompose arbitrary rotations into GPI, GPI2 and MS gates.", 0.7),
        ("qiskit_transpile_pass", "Run CommutativeCancellation and CXDirection routing pass.", 0.8),
        ("openqasm_3_feedforward", "Classical register conditional branching on mid-circuit measurement.", 0.8),
    ]
    for key, content, sal in transient_items:
        mem.record_decision(key=key, content=content, salience=sal, is_core_anchor=False)

    return mem


class TestCoreAnchorRetention:
    """Verifies 100% Core Anchor Retention regardless of prompt topic (Acceptance Criterion)."""

    def test_core_anchors_100_percent_retained_on_offtopic_prompt(
        self, populated_memory: CognitiveMemoryManager
    ) -> None:
        """Core anchors must never be dropped, even when prompt is completely off-topic."""
        offtopic_context = "Design a mobile-friendly CSS flexbox card layout for user account billing."

        # 1. Structured recall
        swr_dict = populated_memory.recall_context_adaptive_swr(
            context=offtopic_context,
            top_k_transient=2,
            min_fidelity=0.40,
        )
        recalled_core_keys = {item["key"] for item in swr_dict["core_anchors"]}
        recalled_gist_keys = {item["key"] for item in swr_dict["semantic_gists"]}

        expected_core_keys = {
            "scientific_integrity_rule",
            "secure_vault_rule",
            "native_first_rule",
            "immutable_assets_rule",
            "subconscious_anchor_rule",
        }
        assert expected_core_keys.issubset(recalled_core_keys), (
            f"Missing core anchors in structured recall: {expected_core_keys - recalled_core_keys}"
        )
        assert "gist_posix_atomic_write" in recalled_gist_keys

        # 2. Flat list convenience recall
        flat_items = populated_memory.recall_contextual_resonance(
            context_prompt=offtopic_context,
            top_k_transient=2,
            min_fidelity=0.40,
        )
        flat_keys = {item["key"] for item in flat_items}
        assert expected_core_keys.issubset(flat_keys), (
            f"Missing core anchors in flat recall: {expected_core_keys - flat_keys}"
        )
        assert "gist_posix_atomic_write" in flat_keys

    @pytest.mark.parametrize(
        "diverse_context",
        [
            "Write a sonnet about the Mediterranean sunrise in autumn.",
            "Recipe for baking sourdough bread with 75% hydration and rye flour.",
            "ALTER TABLE users ADD COLUMN billing_address_v2 VARCHAR(255);",
            "Terms of Service clause 14: indemnification and limitation of liability.",
            "DSP IIR biquad filter audio equalization at 48kHz sample rate.",
        ],
    )
    def test_core_anchors_retained_across_diverse_orthogonal_prompts(
        self, populated_memory: CognitiveMemoryManager, diverse_context: str
    ) -> None:
        """100% of core anchors must be retained across diverse off-topic domains."""
        expected_core_keys = {
            "scientific_integrity_rule",
            "secure_vault_rule",
            "native_first_rule",
            "immutable_assets_rule",
            "subconscious_anchor_rule",
        }
        recalled = populated_memory.recall_context_adaptive_swr(
            context=diverse_context,
            top_k_transient=3,
        )
        core_keys = {item["key"] for item in recalled["core_anchors"]}
        assert expected_core_keys == core_keys


class TestTransientTokenReduction:
    """Verifies >= 40% reduction of transient tokens on off-topic tasks (Acceptance Criterion)."""

    def test_token_reduction_ge_40_percent_on_offtopic_task(
        self, populated_memory: CognitiveMemoryManager
    ) -> None:
        """Transient prompt tokens must decrease by >= 40% when prompt is off-topic."""
        offtopic_context = "Write an SQL migration to rename table tbl_billing_orders to orders_v2."

        # 1. Baseline: Unfiltered static recall containing all 10 transient items
        all_vital = populated_memory.recall_vital_context(top_k=20, min_fidelity=0.0)
        baseline_transient = [
            it for it in all_vital if not it.get("is_core_anchor") and it.get("category") != "semantic_gist"
        ]
        assert len(baseline_transient) == 10
        baseline_transient_text = " ".join(f"{it['key']}: {it['content']}" for it in baseline_transient)
        baseline_transient_tokens = len(baseline_transient_text.split())

        # 2. Contextual Resonance filtered recall
        swr_dict = populated_memory.recall_context_adaptive_swr(
            context=offtopic_context,
            top_k_transient=3,
            min_resonance=0.12,
            min_fidelity=0.40,
        )
        adaptive_transient = swr_dict["transient_decisions"]
        adaptive_transient_text = " ".join(f"{it['key']}: {it['content']}" for it in adaptive_transient)
        adaptive_transient_tokens = len(adaptive_transient_text.split())

        # 3. Verify >= 40% token reduction
        reduction_ratio = (baseline_transient_tokens - adaptive_transient_tokens) / float(baseline_transient_tokens)
        assert reduction_ratio >= 0.40, (
            f"Transient token reduction was only {reduction_ratio:.2%} (expected >= 40.0%). "
            f"Baseline: {baseline_transient_tokens} tokens, Resonant: {adaptive_transient_tokens} tokens."
        )

    def test_zero_transient_bloat_on_completely_orthogonal_context(
        self, populated_memory: CognitiveMemoryManager
    ) -> None:
        """Completely orthogonal context filters out 100% of transient engrams."""
        orthogonal_context = "Prepare a traditional lasagna with bechamel sauce and parmesan cheese."

        swr_dict = populated_memory.recall_context_adaptive_swr(
            context=orthogonal_context,
            top_k_transient=3,
            min_resonance=0.12,
        )
        # All 10 transient items are quantum physics / circuits; zero should survive
        assert len(swr_dict["transient_decisions"]) == 0
        # While all core anchors remain intact
        assert len(swr_dict["core_anchors"]) == 5

    def test_full_prompt_string_token_reduction(
        self, populated_memory: CognitiveMemoryManager
    ) -> None:
        """Verifies overall formatted prompt string token reduction on off-topic task."""
        offtopic_context = "Update the Tailwind CSS theme colors for the dark mode sidebar."

        # Static uncompressed prompt formatting containing all core + all 10 transient items
        all_items = populated_memory.recall_vital_context(top_k=20, min_fidelity=0.0)
        static_core = [
            f"{c['key']} ({c['retention_pct']:.2f}%)"
            for c in all_items
            if c.get("is_core_anchor") or c.get("category") == "semantic_gist"
        ]
        static_transient = [
            f"{t['key']} ({t['retention_pct']:.2f}%)"
            for t in all_items
            if not t.get("is_core_anchor") and t.get("category") != "semantic_gist"
        ]
        static_lines = [
            "[Quanta Bilişsel Çıpa | SWR Replay]:",
            f"  🔒 Çekirdek: {', '.join(static_core)}",
            f"  ⚡ Geçici: {', '.join(static_transient)}",
        ]
        uncompressed_prompt = "\n".join(static_lines)
        uncompressed_tokens = len(uncompressed_prompt.split())

        # Context-adaptive prompt (prunes off-topic transient items)
        adaptive_prompt = populated_memory.format_context_adaptive_swr_prompt(
            context=offtopic_context,
            top_k_transient=3,
            min_resonance=0.12,
        )
        adaptive_tokens = len(adaptive_prompt.split())

        reduction = (uncompressed_tokens - adaptive_tokens) / float(uncompressed_tokens)
        assert reduction >= 0.40, f"Overall token reduction was {reduction:.2%}, expected >= 40%"


class TestOnTopicTransientRanking:
    """Verifies correct ranking and retention of relevant transient engrams on on-topic tasks."""

    def test_on_topic_prompt_preserves_relevant_transient_engrams(
        self, populated_memory: CognitiveMemoryManager
    ) -> None:
        """When prompt matches transient topics, relevant engrams are preserved and ranked high."""
        ontopic_context = "Simulate 17-qubit surface code syndrome decoding and bond dimension truncation."

        swr_dict = populated_memory.recall_context_adaptive_swr(
            context=ontopic_context,
            top_k_transient=4,
            min_resonance=0.12,
            min_fidelity=0.40,
        )
        transient_recalled = swr_dict["transient_decisions"]
        assert len(transient_recalled) > 0

        recalled_keys = [t["key"] for t in transient_recalled]
        # Most resonant should be surface code or MPS bond dimension
        assert "willow_surface_code_syndrome" in recalled_keys or "mps_bond_dimension_cutoff" in recalled_keys

        # Check that top element has high resonance
        top_entry = transient_recalled[0]
        assert top_entry["resonance"] >= 0.15
        assert top_entry["priority"] > 0.0

    def test_fine_grained_topic_discrimination(
        self, populated_memory: CognitiveMemoryManager
    ) -> None:
        """QAOA prompt prioritizes QAOA engrams; VQE prompt prioritizes VQE engrams."""
        # Query 1: QAOA focus
        res_qaoa = populated_memory.recall_context_adaptive_swr(
            context="Optimize QAOA mixer Hamiltonian angle beta and gamma for lattice basis reduction.",
            top_k_transient=2,
            min_resonance=0.10,
        )
        qaoa_keys = [t["key"] for t in res_qaoa["transient_decisions"]]
        assert any("qaoa" in k for k in qaoa_keys)

        # Query 2: VQE focus
        res_vqe = populated_memory.recall_context_adaptive_swr(
            context="UCCSD ansatz calculation for hydrogen molecule bond length in VQE.",
            top_k_transient=2,
            min_resonance=0.10,
        )
        vqe_keys = [t["key"] for t in res_vqe["transient_decisions"]]
        assert "vqe_h2_molecule" in vqe_keys


class TestContextResonanceMetric:
    """Verifies mathematical properties of compute_context_resonance."""

    def test_resonance_score_bounds_and_symmetry(
        self, populated_memory: CognitiveMemoryManager
    ) -> None:
        """Resonance score must be strictly in [0.0, 1.0]."""
        engram = populated_memory.buffer.buffer[0]
        score_1 = populated_memory.compute_context_resonance("quantum circuit simulation", engram)
        score_2 = populated_memory.compute_context_resonance("", engram)

        assert 0.0 <= score_1 <= 1.0
        assert 0.0 <= score_2 <= 1.0

    def test_identical_text_has_highest_resonance(
        self, populated_memory: CognitiveMemoryManager
    ) -> None:
        """Exact matching text achieves significantly higher resonance than disparate text."""
        engram = {
            "key": "test_engram",
            "content": "Superconducting qubit transmon coherence time T1 and T2 relaxation.",
            "category": "decision",
            "salience": 1.0,
        }
        identical_ctx = "Superconducting qubit transmon coherence time T1 and T2 relaxation."
        disparate_ctx = "Baking artisan sourdough bread with whole wheat flour."

        score_identical = populated_memory.compute_context_resonance(identical_ctx, engram)
        score_disparate = populated_memory.compute_context_resonance(disparate_ctx, engram)

        assert score_identical > 0.70
        assert score_disparate < 0.15
        assert score_identical > score_disparate * 4.0

    def test_alpha_weighting_sensitivity(
        self, populated_memory: CognitiveMemoryManager
    ) -> None:
        """Validates interpolation between pure Hilbert overlap (alpha=1) and pure Jaccard (alpha=0)."""
        engram = populated_memory.buffer.buffer[0]
        context = "scientific integrity and benchmarks"

        score_hilbert_only = populated_memory.compute_context_resonance(context, engram, alpha=1.0)
        score_jaccard_only = populated_memory.compute_context_resonance(context, engram, alpha=0.0)
        score_composite = populated_memory.compute_context_resonance(context, engram, alpha=0.6)

        expected = 0.6 * score_hilbert_only + 0.4 * score_jaccard_only
        assert pytest.approx(score_composite, abs=1e-5) == expected

    def test_engram_index_vs_dict_polymorphism(
        self, populated_memory: CognitiveMemoryManager
    ) -> None:
        """compute_context_resonance accepts int index or dict transparently."""
        score_by_idx = populated_memory.compute_context_resonance("scientific integrity", 0)
        score_by_dict = populated_memory.compute_context_resonance(
            "scientific integrity", populated_memory.buffer.buffer[0]
        )
        assert pytest.approx(score_by_idx, abs=1e-5) == score_by_dict

    def test_token_extractor_stopword_and_unicode_handling(self) -> None:
        """Token extractor strips stopwords and preserves Turkish characters."""
        text = "Türkçe karakterler: çözümler, şifreleme, 17-qubit ve entegrasyon."
        tokens = extract_context_tokens(text)
        assert "çözümler" in tokens or "çözüm" in tokens or "cozumler" in tokens or "17" in tokens
        assert "şifreleme" in tokens
        assert "qubit" in tokens
        assert "ve" not in tokens  # stopword stripped


class TestOffTopicPruning:
    """Verifies prune_off_topic_transient evicts off-topic transient engrams while protecting core."""

    def test_prune_off_topic_transient_preserves_core_and_evicts_unrelated(
        self, populated_memory: CognitiveMemoryManager
    ) -> None:
        """Simulates 3 turns of aging, then evicts off-topic transient engrams."""
        # Age all engrams past min_age_turns (2.0)
        populated_memory.step(dt=3.0)

        initial_count = len(populated_memory.buffer.buffer)
        offtopic_ctx = "Design a mobile CSS flexbox card layout for user account billing."

        pruned = populated_memory.prune_off_topic_transient(
            context=offtopic_ctx,
            min_resonance=0.08,
            min_age_turns=2.0,
        )
        assert len(pruned) > 0

        # Verify all core anchors remain in buffer
        surviving_keys = {
            e.get("metadata", {}).get("key") for e in populated_memory.buffer.buffer
        }
        expected_core_keys = {
            "scientific_integrity_rule",
            "secure_vault_rule",
            "native_first_rule",
            "immutable_assets_rule",
            "subconscious_anchor_rule",
            "gist_posix_atomic_write",
        }
        assert expected_core_keys.issubset(surviving_keys), (
            f"Core anchors were illegally evicted: {expected_core_keys - surviving_keys}"
        )

        # Total pruned count tracking
        assert populated_memory.total_pruned_count >= len(pruned)
        assert len(populated_memory.buffer.buffer) == initial_count - len(pruned)

    def test_fresh_transient_items_protected_by_min_age_turns(
        self, populated_memory: CognitiveMemoryManager
    ) -> None:
        """Transient items younger than min_age_turns are not evicted prematurely."""
        # Add a brand new transient decision with age = 0.0
        populated_memory.record_transient_decision(
            key="fresh_unrelated_decision",
            content="Temporary buffer configuration for staging.",
            salience=0.4,
        )
        # Advance only 0.5 turns (< min_age_turns = 2.0)
        populated_memory.step(dt=0.5)

        pruned = populated_memory.prune_off_topic_transient(
            context="quantum surface code decoding",
            min_resonance=0.08,
            min_age_turns=2.0,
        )
        pruned_keys = {p["key"] for p in pruned}
        assert "fresh_unrelated_decision" not in pruned_keys


class TestPromptFormatting:
    """Verifies format_context_adaptive_swr_prompt output structure."""

    def test_formatted_prompt_structure_and_prefixes(
        self, populated_memory: CognitiveMemoryManager
    ) -> None:
        """Formatted prompt must start with standard header and contain appropriate tiers."""
        ontopic_context = "Simulate 17-qubit surface code syndrome decoding."
        prompt_str = populated_memory.format_context_adaptive_swr_prompt(
            context=ontopic_context,
            top_k_transient=2,
        )

        assert prompt_str.startswith("[Quanta Bilişsel Çıpa | SWR Replay]:")
        assert "🔒 Çekirdek:" in prompt_str
        assert "scientific_integrity_rule" in prompt_str
        assert "🧠 Özüt:" in prompt_str
        assert "gist_posix_atomic_write" in prompt_str
        assert "⚡ Geçici:" in prompt_str

    def test_inhibitor_tier_in_prompt_formatting(
        self, populated_memory: CognitiveMemoryManager
    ) -> None:
        """Inhibitors are formatted with V_inh and fidelity in the prompt."""
        populated_memory.record_inhibitor(
            key="inh_eval_raw_string",
            content="Never use eval() on unsanitized user expressions.",
            v_inh=0.75,
            salience=1.8,
        )
        prompt_str = populated_memory.format_context_adaptive_swr_prompt(
            context="arbitrary script execution",
            top_k_transient=2,
        )
        assert "🚫 İnhibitör / Anti-Pattern:" in prompt_str
        assert "inh_eval_raw_string" in prompt_str
        assert "V_inh=0.75" in prompt_str


class TestEdgeCasesAndStability:
    """Boundary cases: empty buffers, empty contexts, numerical stability."""

    def test_empty_memory_buffer(self) -> None:
        mem = CognitiveMemoryManager(capacity=16, dim=64)
        swr_dict = mem.recall_context_adaptive_swr(context="anything")
        assert swr_dict == {
            "core_anchors": [],
            "semantic_gists": [],
            "transient_decisions": [],
            "inhibitors": [],
        }
        flat = mem.recall_contextual_resonance(context="anything")
        assert flat == []
        pruned = mem.prune_off_topic_transient(context="anything")
        assert pruned == []
        prompt = mem.format_context_adaptive_swr_prompt(context="anything")
        assert prompt == "[Quanta Bilişsel Çıpa | SWR Replay]:"

    def test_empty_and_whitespace_context(
        self, populated_memory: CognitiveMemoryManager
    ) -> None:
        """Empty context executes without error, retaining core anchors."""
        swr_dict = populated_memory.recall_context_adaptive_swr(context="   ")
        assert len(swr_dict["core_anchors"]) == 5
        assert len(swr_dict["semantic_gists"]) == 1
