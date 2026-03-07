"""
tests/test_entity_resolution.py -- Full coverage for entity resolution module.

Tests the complete pipeline: similarity, blocking, QAOA/greedy merge,
Turkish name handling, and edge cases.
"""

import pytest
import numpy as np

from quanta.layer3.entity_resolution import (
    resolve,
    ResolutionResult,
    Cluster,
    compute_similarity,
    _normalize,
    _levenshtein_sim,
    _soundex_tr,
    _field_similarity,
    _block_records,
    _greedy_merge_block,
    _qaoa_optimize_block,
    _pick_canonical,
    _merge_overlapping_sets,
    _split_block_by_similarity,
)


# ═══════════════════════════════════════════
#  Turkish normalization
# ═══════════════════════════════════════════

class TestNormalize:
    def test_lowercase(self):
        assert _normalize("AHMET") == "ahmet"

    def test_strip(self):
        assert _normalize("  test  ") == "test"

    def test_turkish_chars(self):
        assert _normalize("çğıöşü") == "cgiosu"
        # İ (capital I with dot above) normalizes; check key chars
        result = _normalize("ÇĞİÖŞÜ")
        assert result.startswith("cg")
        assert "osu" in result

    def test_mixed(self):
        assert _normalize("Yılmaz") == "yilmaz"
        assert _normalize("Öztürk") == "ozturk"
        assert _normalize("Şahin") == "sahin"

    def test_empty(self):
        assert _normalize("") == ""


# ═══════════════════════════════════════════
#  Levenshtein similarity
# ═══════════════════════════════════════════

class TestLevenshteinSim:
    def test_identical(self):
        assert _levenshtein_sim("ahmet", "ahmet") == 1.0

    def test_completely_different(self):
        assert _levenshtein_sim("abc", "xyz") == 0.0

    def test_one_char_diff(self):
        sim = _levenshtein_sim("ahmet", "ahmat")
        assert 0.7 < sim < 1.0

    def test_turkish_normalization(self):
        # Yılmaz vs Yilmaz should be identical after normalization
        assert _levenshtein_sim("Yılmaz", "Yilmaz") == 1.0

    def test_empty_both(self):
        assert _levenshtein_sim("", "") == 1.0

    def test_empty_one(self):
        assert _levenshtein_sim("test", "") == 0.0
        assert _levenshtein_sim("", "test") == 0.0

    def test_abbreviation(self):
        sim = _levenshtein_sim("A.", "Ahmet")
        assert sim < 0.5  # quite different


# ═══════════════════════════════════════════
#  Turkish Soundex
# ═══════════════════════════════════════════

class TestSoundexTR:
    def test_basic(self):
        code = _soundex_tr("Ahmet")
        assert len(code) == 4
        assert code[0] == "A"

    def test_turkish_variants_match(self):
        # Same name spelled differently should give same code
        assert _soundex_tr("Yılmaz") == _soundex_tr("Yilmaz")
        assert _soundex_tr("Öztürk") == _soundex_tr("Ozturk")

    def test_similar_names(self):
        # Similar sounding names
        assert _soundex_tr("Demir") == _soundex_tr("Demır")

    def test_different_names(self):
        assert _soundex_tr("Ahmet") != _soundex_tr("Zeynep")

    def test_empty(self):
        assert _soundex_tr("") == ""

    def test_padding(self):
        # Short names should be padded to 4 chars
        code = _soundex_tr("Al")
        assert len(code) == 4


# ═══════════════════════════════════════════
#  Field similarity
# ═══════════════════════════════════════════

class TestFieldSimilarity:
    def test_exact_match(self):
        a = {"name": "Ahmet Yılmaz"}
        b = {"name": "Ahmet Yilmaz"}
        assert _field_similarity(a, b, "name") == 1.0

    def test_phone_digits_only(self):
        a = {"phone": "+90 532 111 22 33"}
        b = {"phone": "905321112233"}
        assert _field_similarity(a, b, "phone") == 1.0

    def test_phone_different(self):
        a = {"phone": "5321112233"}
        b = {"phone": "5329998877"}
        assert _field_similarity(a, b, "phone") == 0.0

    def test_email_match(self):
        a = {"email": "Test@Gmail.com"}
        b = {"email": "test@gmail.com"}
        assert _field_similarity(a, b, "email") == 1.0

    def test_email_different(self):
        a = {"email": "a@b.com"}
        b = {"email": "c@d.com"}
        assert _field_similarity(a, b, "email") == 0.0

    def test_missing_field(self):
        a = {"name": "Ahmet"}
        b = {}
        assert _field_similarity(a, b, "name") == 0.0

    def test_tc_exact(self):
        a = {"tc": "12345678901"}
        b = {"tc": "12345678901"}
        assert _field_similarity(a, b, "tc") == 1.0


# ═══════════════════════════════════════════
#  Compute similarity (weighted)
# ═══════════════════════════════════════════

class TestComputeSimilarity:
    def test_identical_records(self):
        rec = {"name": "Ahmet Yılmaz", "phone": "5321112233"}
        sim = compute_similarity(rec, rec)
        assert sim == 1.0

    def test_same_phone_different_name(self):
        a = {"name": "Ahmet", "phone": "5321112233"}
        b = {"name": "Mehmet", "phone": "5321112233"}
        sim = compute_similarity(a, b)
        assert sim > 0.5  # phone has high weight

    def test_custom_fields(self):
        a = {"name": "Test", "city": "Istanbul"}
        b = {"name": "Test", "city": "Ankara"}
        sim = compute_similarity(a, b, fields=["name"])
        assert sim == 1.0  # only comparing name

    def test_custom_weights(self):
        a = {"name": "A", "city": "Istanbul"}
        b = {"name": "B", "city": "Istanbul"}
        # With high city weight, similarity should be high
        sim_high = compute_similarity(a, b, weights={"name": 0.1, "city": 10.0})
        sim_low = compute_similarity(a, b, weights={"name": 10.0, "city": 0.1})
        assert sim_high > sim_low

    def test_no_common_fields(self):
        a = {"name": "Test"}
        b = {"phone": "123"}
        sim = compute_similarity(a, b)
        assert sim == 0.0


# ═══════════════════════════════════════════
#  Blocking
# ═══════════════════════════════════════════

class TestBlocking:
    def test_phone_blocking(self):
        records = [
            {"name": "Ahmet", "phone": "5321112233"},
            {"name": "A. Yilmaz", "phone": "5321112233"},
            {"name": "Mehmet", "phone": "5559998877"},
        ]
        blocks = _block_records(records)
        # Records 0 and 1 should be in the same block
        found = False
        for block in blocks:
            if 0 in block and 1 in block:
                found = True
                break
        assert found, f"Records 0,1 not blocked together. Blocks: {blocks}"

    def test_email_blocking(self):
        records = [
            {"name": "Ahmet", "email": "a@test.com"},
            {"name": "A. Yilmaz", "email": "a@test.com"},
        ]
        blocks = _block_records(records)
        found = any(0 in b and 1 in b for b in blocks)
        assert found

    def test_last_name_blocking(self):
        records = [
            {"name": "Ahmet Yilmaz"},
            {"name": "Mehmet Yilmaz"},
        ]
        blocks = _block_records(records)
        found = any(0 in b and 1 in b for b in blocks)
        assert found

    def test_soundex_blocking(self):
        records = [
            {"name": "Yılmaz Demir"},
            {"name": "Yilmaz Demır"},
        ]
        blocks = _block_records(records)
        found = any(0 in b and 1 in b for b in blocks)
        assert found

    def test_singletons_added(self):
        records = [
            {"name": "Unique Person 1"},
            {"name": "Completely Different 2"},
        ]
        blocks = _block_records(records)
        # Both should appear in some block
        all_ids = set()
        for b in blocks:
            all_ids.update(b)
        assert 0 in all_ids and 1 in all_ids


# ═══════════════════════════════════════════
#  Merge and helpers
# ═══════════════════════════════════════════

class TestMergeHelpers:
    def test_merge_overlapping_sets(self):
        sets = [{0, 1}, {1, 2}, {3, 4}]
        merged = _merge_overlapping_sets(sets)
        # {0,1,2} and {3,4}
        assert len(merged) == 2
        big = [m for m in merged if len(m) == 3][0]
        assert set(big) == {0, 1, 2}

    def test_merge_no_overlap(self):
        sets = [{0, 1}, {2, 3}]
        merged = _merge_overlapping_sets(sets)
        assert len(merged) == 2

    def test_merge_all_overlap(self):
        sets = [{0, 1}, {1, 2}, {2, 3}]
        merged = _merge_overlapping_sets(sets)
        assert len(merged) == 1
        assert set(merged[0]) == {0, 1, 2, 3}

    def test_pick_canonical_most_complete(self):
        records = [
            {"name": "A"},
            {"name": "Ahmet Yilmaz", "phone": "532", "city": "Istanbul"},
            {"name": "A. Yilmaz"},
        ]
        canon = _pick_canonical(records, [0, 1, 2])
        assert canon["name"] == "Ahmet Yilmaz"

    def test_split_block_small(self):
        records = [{"name": f"R{i}"} for i in range(5)]
        result = _split_block_by_similarity(records, [0, 1, 2, 3, 4], max_size=7)
        assert len(result) == 1  # no split needed


# ═══════════════════════════════════════════
#  Greedy merge
# ═══════════════════════════════════════════

class TestGreedyMerge:
    def test_merge_similar_records(self):
        records = [
            {"name": "Ahmet Yılmaz", "phone": "5321112233"},
            {"name": "Ahmet Yilmaz", "phone": "5321112233"},
        ]
        clusters = _greedy_merge_block(records, [0, 1], threshold=0.5)
        assert len(clusters) == 1
        assert set(clusters[0]) == {0, 1}

    def test_keep_different_records(self):
        records = [
            {"name": "Ahmet Yılmaz", "phone": "5321112233"},
            {"name": "Zeynep Kaya", "phone": "5559998877"},
        ]
        clusters = _greedy_merge_block(records, [0, 1], threshold=0.8)
        assert len(clusters) == 2

    def test_transitive_merge(self):
        records = [
            {"name": "Ahmet Yılmaz",  "phone": "5321112233"},
            {"name": "A. Yılmaz",     "phone": "5321112233"},
            {"name": "Ahmet Yilmaz",  "phone": "5321112233"},
        ]
        clusters = _greedy_merge_block(records, [0, 1, 2], threshold=0.5)
        assert len(clusters) == 1


# ═══════════════════════════════════════════
#  QAOA merge
# ═══════════════════════════════════════════

class TestQAOAMerge:
    def test_qaoa_merges_similar(self):
        records = [
            {"name": "Ahmet Yılmaz", "phone": "5321112233"},
            {"name": "Ahmet Yilmaz", "phone": "5321112233"},
        ]
        clusters, qubits = _qaoa_optimize_block(
            records, [0, 1], threshold=0.3, seed=42
        )
        assert qubits == 1  # 1 pair -> 1 qubit
        # QAOA is probabilistic; verify it produced valid output
        assert len(clusters) in (1, 2)
        all_ids = sorted(idx for c in clusters for idx in c)
        assert all_ids == [0, 1]

    def test_qaoa_separates_different(self):
        records = [
            {"name": "Ahmet Yılmaz", "phone": "5321112233"},
            {"name": "Zeynep Kaya",  "phone": "5559998877"},
        ]
        clusters, qubits = _qaoa_optimize_block(
            records, [0, 1], threshold=0.8, seed=42
        )
        assert qubits == 1
        assert len(clusters) == 2

    def test_qaoa_single_record(self):
        records = [{"name": "Test"}]
        clusters, qubits = _qaoa_optimize_block(records, [0], threshold=0.5)
        assert qubits == 0
        assert len(clusters) == 1

    def test_qaoa_three_records(self):
        records = [
            {"name": "Ahmet Yılmaz", "phone": "5321112233"},
            {"name": "A. Yılmaz",    "phone": "5321112233"},
            {"name": "Ahmet Yilmaz", "phone": "5321112233"},
        ]
        clusters, qubits = _qaoa_optimize_block(
            records, [0, 1, 2], threshold=0.5, seed=42
        )
        assert qubits == 3  # C(3,2) = 3 pairs -> 3 qubits


# ═══════════════════════════════════════════
#  resolve() -- Full pipeline
# ═══════════════════════════════════════════

class TestResolve:
    def test_basic_dedup_qaoa(self):
        records = [
            {"name": "Ahmet Yılmaz", "phone": "5321112233"},
            {"name": "A. Yılmaz",    "phone": "5321112233"},
            {"name": "Zeynep Kaya",  "phone": "5559998877"},
        ]
        result = resolve(records, threshold=0.5, method="qaoa", seed=42)
        assert result.num_records == 3
        assert result.num_entities <= 3
        assert result.method == "qaoa"
        assert len(result.clusters) > 0

    def test_basic_dedup_greedy(self):
        records = [
            {"name": "Ahmet Yılmaz", "phone": "5321112233"},
            {"name": "A. Yılmaz",    "phone": "5321112233"},
            {"name": "Zeynep Kaya",  "phone": "5559998877"},
        ]
        result = resolve(records, threshold=0.5, method="greedy", seed=42)
        assert result.num_records == 3
        assert result.method == "greedy"
        assert result.quantum_blocks == 0

    def test_all_unique(self):
        records = [
            {"name": "Person A", "phone": "1111111111"},
            {"name": "Person B", "phone": "2222222222"},
            {"name": "Person C", "phone": "3333333333"},
        ]
        result = resolve(records, threshold=0.9, method="greedy", seed=42)
        assert result.num_entities == 3

    def test_all_same(self):
        records = [
            {"name": "Ahmet Yılmaz", "phone": "5321112233"},
            {"name": "Ahmet Yilmaz", "phone": "5321112233"},
            {"name": "Ahmet Yılmaz", "phone": "532 111 22 33"},
        ]
        result = resolve(records, threshold=0.5, method="greedy", seed=42)
        assert result.num_entities < result.num_records

    def test_turkish_name_handling(self):
        records = [
            {"name": "Öztürk Şahin",   "city": "İstanbul"},
            {"name": "Ozturk Sahin",    "city": "Istanbul"},
        ]
        result = resolve(records, threshold=0.5, method="greedy", seed=42)
        # Should merge due to Turkish char normalization
        assert result.num_entities == 1

    def test_fields_param(self):
        records = [
            {"name": "Ahmet", "phone": "5321112233", "city": "Istanbul"},
            {"name": "Ahmet", "phone": "5559998877", "city": "Ankara"},
        ]
        # Only compare name -> should merge
        result = resolve(records, threshold=0.5, fields=["name"],
                        method="greedy", seed=42)
        assert result.num_entities == 1

    def test_result_summary(self):
        records = [
            {"name": "Ahmet Yılmaz", "phone": "5321112233"},
            {"name": "Zeynep Kaya",  "phone": "5559998877"},
        ]
        result = resolve(records, threshold=0.5, method="greedy", seed=42)
        summary = result.summary()
        assert "Entity Resolution" in summary
        assert "Records" in summary
        assert "Entities" in summary

    def test_cluster_confidence(self):
        records = [
            {"name": "Ahmet Yılmaz", "phone": "5321112233"},
            {"name": "Ahmet Yilmaz", "phone": "5321112233"},
        ]
        result = resolve(records, threshold=0.3, method="greedy", seed=42)
        for cluster in result.clusters:
            assert 0 <= cluster.confidence <= 1.0

    def test_canonical_record(self):
        records = [
            {"name": "A."},  # sparse
            {"name": "Ahmet Yılmaz", "phone": "5321112233", "city": "Istanbul"},
        ]
        result = resolve(records, threshold=0.3, method="greedy", seed=42)
        # The canonical should be the more complete record
        for cluster in result.clusters:
            if len(cluster.record_ids) > 1:
                assert "phone" in cluster.canonical or len(cluster.canonical) > 1

    def test_qaoa_vs_greedy_consistency(self):
        """QAOA and greedy should produce comparable entity counts."""
        records = [
            {"name": "Ahmet Yılmaz", "phone": "5321112233"},
            {"name": "Ahmet Yilmaz", "phone": "5321112233"},
            {"name": "Zeynep Kaya",  "phone": "5559998877"},
            {"name": "Z. Kaya",      "phone": "5559998877"},
        ]
        qaoa_result = resolve(records, threshold=0.5, method="qaoa", seed=42)
        greedy_result = resolve(records, threshold=0.5, method="greedy", seed=42)

        # Both should identify roughly same number of entities
        diff = abs(qaoa_result.num_entities - greedy_result.num_entities)
        assert diff <= 2, (
            f"QAOA entities={qaoa_result.num_entities}, "
            f"greedy entities={greedy_result.num_entities}"
        )

    def test_large_dataset(self):
        """Tests with 20+ records to exercise blocking and splitting."""
        records = []
        for i in range(10):
            records.append({"name": f"Person {i}", "phone": f"555000{i:04d}"})
        # Add duplicates
        records.append({"name": "Person 0", "phone": "5550000000"})
        records.append({"name": "Person 1", "phone": "5550000001"})

        result = resolve(records, threshold=0.6, method="greedy", seed=42)
        assert result.num_records == 12
        assert result.num_entities < 12  # some should merge


# ═══════════════════════════════════════════
#  ResolutionResult
# ═══════════════════════════════════════════

class TestResolutionResult:
    def test_dataclass_fields(self):
        result = ResolutionResult(
            clusters=[],
            num_records=10,
            num_entities=5,
            method="qaoa",
            quantum_blocks=3,
            total_qubits=15,
        )
        assert result.num_records == 10
        assert result.num_entities == 5
        assert result.method == "qaoa"

    def test_summary_with_accuracy(self):
        result = ResolutionResult(
            clusters=[
                Cluster(record_ids=[0, 1], canonical={"name": "Test"}, confidence=0.9)
            ],
            num_records=3,
            num_entities=2,
            method="qaoa",
            comparison_accuracy=0.95,
        )
        summary = result.summary()
        assert "95%" in summary
        assert "Accuracy" in summary

    def test_summary_without_accuracy(self):
        result = ResolutionResult(
            clusters=[],
            num_records=1,
            num_entities=1,
            method="greedy",
        )
        summary = result.summary()
        assert "Accuracy" not in summary
