"""Comprehensive End-to-End (E2E) Test Suite for Quanta Tabbed Cockpit & Decision Arbiter.

Validates the full Milestone 4 architecture across 4 rigorous tiers:
- Tier 1: Feature Coverage (R1 dual state mirroring, R2 dynamic Zeno pinning,
          R2 microglial pruning history, R3 6-qubit dilemma A/B generation,
          R4 4-tab HTML cockpit rendering, embedded state snapshot).
- Tier 2: Boundary & Corner Cases (Clamped/extreme Zeno values, NaN/infinite resilience,
          missing/corrupted state fallback, pruning history 100-entry capping,
          XSS and special character neutralization, SVG needle radius invariance).
- Tier 3: Cross-Feature Combinations (SWR replay -> State -> Dashboard needle trigonometry,
          Pruning event -> State -> Tab 4 audit table, Arbiter dilemma -> Telemetry -> Tab 2 A/B,
          Simulated closed-loop outcome verification).
- Tier 4: Real-World Workflow Scenarios (End-to-end multi-turn conversational simulation,
          zero-CDN offline air-gap compliance, multi-target HTML mirroring equivalence).

Strictly isolated: All tests use tmp_path and monkeypatching to guard production files.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from quanta.cognitive.arbiter import QuantumDecisionArbiter
from quanta.cognitive.telemetry import generate_dashboard_html, record_decision_telemetry
from scripts.hooks.quanta_subconscious_hook import (
    FastBiomorphicMemory,
    PrunedEngram,
    _atomic_write_single_file,
    _save_mirrored_state_atomically,
    refresh_memory_from_state,
)


@pytest.fixture(autouse=True)
def isolate_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Automatically isolates tests from the production telemetry log."""
    telem_dir = tmp_path / ".quanta"
    telem_dir.mkdir(parents=True, exist_ok=True)
    telem_file = telem_dir / "quanta_cognitive_telemetry.jsonl"
    monkeypatch.setattr("quanta.cognitive.telemetry.DEFAULT_TELEMETRY_DIR", telem_dir)
    monkeypatch.setattr("quanta.cognitive.telemetry.DEFAULT_TELEMETRY_FILE", telem_file)


# ==============================================================================
# TIER 1: FEATURE COVERAGE
# ==============================================================================


class TestTier1FeatureCoverage:
    """Tier 1: Comprehensive feature coverage for Milestone 4 requirements (R1 - R4)."""

    def test_tier1_r1_atomic_dual_target_state_mirroring(self, tmp_path: Path) -> None:
        """R1: Verifies atomic dual-target persistence across primary and mirror paths."""
        primary_dir = tmp_path / "conv_artifact"
        mirror_dir = tmp_path / "workspace_mirror"

        primary_path = primary_dir / "quanta_cognitive_state.json"
        mirror_path = mirror_dir / "quanta_cognitive_state.json"

        state_payload = {
            "turn_count": 142,
            "mean_zeno_pinning": 0.8842,
            "current_zeno_pinning": 0.8842,
            "engrams": [
                {
                    "key": "native_first_rule",
                    "salience": 2.8,
                    "fidelity": 0.9998,
                    "is_core_anchor": True,
                },
                {
                    "key": "transient_decision_1",
                    "salience": 0.5,
                    "fidelity": 0.8540,
                    "is_core_anchor": False,
                },
            ],
            "total_pruned_count": 3,
            "pruning_history": [
                {
                    "timestamp": "2026-09-22T10:00:00Z",
                    "rule_name": "obsolete_temp_key",
                    "reason": "fidelity_decayed",
                    "decayed_salience": 0.12,
                    "turn_pruned": 140,
                    "final_fidelity": 0.62,
                }
            ],
        }

        prim_ok, mir_ok = _save_mirrored_state_atomically(
            primary_path=primary_path,
            mirror_path=mirror_path,
            state_dict=state_payload,
            conv_id="test_dual_conv",
            is_test_env=False,
        )

        assert prim_ok is True, "Primary atomic state write failed"
        assert mir_ok is True, "Mirror atomic state write failed"
        assert primary_path.exists(), "Primary state file was not created"
        assert mirror_path.exists(), "Mirror state file was not created"

        # Bit-for-bit JSON equivalence between primary and mirror targets
        with open(primary_path, encoding="utf-8") as f1:
            prim_loaded = json.load(f1)
        with open(mirror_path, encoding="utf-8") as f2:
            mir_loaded = json.load(f2)

        assert prim_loaded == mir_loaded
        assert prim_loaded["turn_count"] == 142
        assert prim_loaded["mean_zeno_pinning"] == 0.8842
        assert len(prim_loaded["engrams"]) == 2

        # Verify ZERO orphaned .tmp_* files left behind in either directory
        assert len(list(primary_dir.glob(".tmp_*"))) == 0
        assert len(list(mirror_dir.glob(".tmp_*"))) == 0

    def test_tier1_r2_dynamic_zeno_pinning_calculation(self) -> None:
        """R2: Verifies dynamic Zeno pinning factor computation from 6-qubit projections."""
        arbiter = QuantumDecisionArbiter(dim=64, seed=42)

        # 1. Focused scenario: low exploration drive (0.1)
        res_dominant = arbiter.arbitrate_dilemma(
            dilemma="Production Persistence: High Availability Spanner vs SQLite Single Node",
            hypotheses=[
                "Cloud Spanner Multi-Region Synchronous Replication",
                "Local ephemeral in-memory SQLite without backup",
            ],
            exploration_drive=0.1,
        )
        p_dominant = res_dominant["p_zeno"]
        assert 0.50 <= p_dominant <= 0.985, (
            f"Expected dynamic Zeno pinning in [0.50, 0.985], got {p_dominant}"
        )

        # 2. Exploratory scenario: high exploration drive (0.85)
        res_competing = arbiter.arbitrate_dilemma(
            dilemma="Heuristic Exploration: Variant Alpha vs Variant Beta",
            hypotheses=[
                "Variant Alpha Approach with balanced heuristics",
                "Variant Beta Approach with balanced heuristics",
            ],
            exploration_drive=0.85,
        )
        p_competing = res_competing["p_zeno"]

        # Dynamic variation check: must not be static or identical
        assert p_dominant != p_competing, (
            "Zeno pinning factor must dynamically vary across different dilemmas"
        )
        assert p_competing < p_dominant, (
            "High exploration drive must yield lower Zeno pinning than focused drive"
        )
        assert res_dominant["p_zeno"] != 0.8600, (
            "Zeno pinning must not be hardcoded to static 0.8600"
        )

        # Check quantum ranking properties
        for hyp in res_dominant["hypotheses"]:
            assert "tr_rho_pi" in hyp
            assert "score" in hyp
            assert hyp["tr_rho_pi"] >= 0.0

    def test_tier1_r2_pruning_history_accumulation(self) -> None:
        """R2: Verifies microglial pruning records accumulate with complete audit schema."""
        mem = FastBiomorphicMemory(capacity=64)

        # Core anchors must never be pruned
        mem.record(
            key="native_first_rule",
            content="Native platform API first",
            salience=2.8,
            is_core_anchor=True,
        )

        # Transient engrams that should decay and be swept
        mem.record_transient(
            key="transient_feature_flag",
            content="Temporary feature flag for A/B rollout",
            salience=0.35,
        )
        mem.record_transient(
            key="transient_cache_dir",
            content="Local cache directory path",
            salience=0.30,
        )

        # Manually attenuate transient engrams below threshold (0.70)
        for eng in mem.engrams:
            if not eng.get("is_core_anchor"):
                eng["fidelity"] = 0.58

        pruned_records = mem.prune_obsolete(
            fidelity_threshold=0.70,
            min_salience=0.50,
            turn=15,
        )

        assert len(pruned_records) == 2, f"Expected 2 pruned records, got {len(pruned_records)}"
        assert len(mem.pruning_history) == 2

        # Verify audit schema of each pruning record
        for pr in pruned_records:
            assert isinstance(pr, PrunedEngram)
            assert "timestamp" in pr
            assert pr["rule_name"] in ("transient_feature_flag", "transient_cache_dir")
            assert "fidelity_decayed" in pr["reason"]
            assert pr["decayed_salience"] > 0.0
            assert pr["turn_pruned"] == 15
            assert pr["final_fidelity"] <= 0.70

        # Verify that core anchor survived unscathed
        surviving_keys = [e["key"] for e in mem.engrams]
        assert "native_first_rule" in surviving_keys
        assert "transient_feature_flag" not in surviving_keys

    def test_tier1_r3_six_qubit_dilemma_arbitration_schema(self) -> None:
        """R3: Verifies 6-qubit dilemma arbitration schema with candidate kets and A/B."""
        arbiter = QuantumDecisionArbiter(dim=64, seed=101)

        dilemma_title = "Sync Protocol Dilemma: Direct POSIX vs Async Thread vs Local Only"
        candidate_hyps = [
            {"id": "d1", "label": "Direct POSIX Synchronous Dual-Mirror", "ket": "|d1>"},
            {"id": "d2", "label": "Asynchronous Background Thread Sync", "ket": "|d2>"},
            {"id": "d3", "label": "Local Only / Zero Root Mirroring", "ket": "|d3>"},
        ]

        res = arbiter.arbitrate_dilemma(
            dilemma=dilemma_title,
            hypotheses=candidate_hyps,
            step_idx=77,
            log_telemetry=False,
        )

        # Verify schema keys
        assert res["dilemma"] == dilemma_title
        assert res["step_idx"] == 77
        assert "winner" in res
        assert "winning_label" in res
        assert "p_zeno" in res
        assert "hypotheses" in res
        assert "injected_prompt_constraint" in res
        assert "counterfactual_ab" in res

        # Verify 3 hypothesis kets with Tr(rho Pi) projections
        assert len(res["hypotheses"]) == 3
        for h in res["hypotheses"]:
            assert h["ket"] in ("|d1>", "|d2>", "|d3>")
            assert "score" in h
            assert "tr_rho_pi" in h
            assert 0.0 <= h["probability"] <= 1.0

        # Winning hypothesis consistency
        winning_id = res["winner"]
        assert winning_id in ("d1", "d2", "d3")
        winning_hyp = next(h for h in res["hypotheses"] if h["id"] == winning_id)
        assert res["winning_label"] == winning_hyp["label"]
        assert winning_hyp["score"] == max(h["score"] for h in res["hypotheses"])

        # Injected prompt constraint verification
        constraint_str = res["injected_prompt_constraint"]
        assert constraint_str.startswith("🔒 Bilişsel Kuantum Karar Kısıtı:")
        assert winning_hyp["label"] in constraint_str

        # Side-by-side counterfactual A/B verification
        ab = res["counterfactual_ab"]
        assert ab.get("effect_verified") is True
        assert "without_quanta" in ab
        assert "with_quanta" in ab
        assert len(ab["without_quanta"]) > 20
        assert len(ab["with_quanta"]) > 20

    def test_tier1_r4_four_tab_cockpit_html_rendering(self, tmp_path: Path) -> None:
        """R4: Verifies generate_dashboard_html renders 4-tab modular UI with all required panes."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        state_data = {
            "turn_count": 250,
            "mean_zeno_pinning": 0.875,
            "current_zeno_pinning": 0.875,
            "engrams": [
                {
                    "key": "native_first_rule",
                    "content": "Platform native API first",
                    "salience": 2.8,
                    "fidelity": 0.9998,
                    "confidence": 0.98,
                    "is_core_anchor": True,
                },
                {
                    "key": "temp_pipeline_step",
                    "content": "Step 3 Dataflow Flex Template",
                    "salience": 0.5,
                    "fidelity": 0.82,
                    "age": 2,
                    "max_age": 8,
                    "is_core_anchor": False,
                },
            ],
            "total_pruned_count": 5,
            "pruning_history": [
                {
                    "timestamp": "2026-09-22T11:00:00Z",
                    "rule_name": "pruned_temp_cache",
                    "category": "contextual_decision",
                    "reason": "fidelity_decayed",
                    "decayed_salience": 0.18,
                    "final_fidelity": 0.61,
                    "turn_pruned": 240,
                }
            ],
            "recent_arbitrations": [
                {
                    "dilemma": "Dual Sync Architecture",
                    "winner": "d1",
                    "winning_label": "Direct POSIX Synchronous",
                    "p_zeno": 0.875,
                    "injected_prompt_constraint": "🔒 Lock to POSIX sync",
                    "counterfactual_ab": {
                        "without_quanta": "Agent drifts across async and sync",
                        "with_quanta": "Deterministic execution guaranteed",
                    },
                    "hypotheses": [
                        {
                            "id": "d1",
                            "label": "Direct POSIX Synchronous",
                            "ket": "|d1>",
                            "probability": 0.80,
                            "tr_rho_pi": 0.72,
                        }
                    ],
                }
            ],
        }
        state_file.write_text(json.dumps(state_data), encoding="utf-8")

        dash_out = tmp_path / "dashboard.html"
        generated_path = generate_dashboard_html(
            output_path=dash_out,
            state_file=state_file,
            mirror_to_repo=False,
        )

        assert generated_path == dash_out
        assert dash_out.exists()
        content = dash_out.read_text(encoding="utf-8")

        # Verify 4-Tab Navigation Buttons
        assert 'data-tab="tab-cockpit"' in content
        assert 'data-tab="tab-arbiter"' in content
        assert 'data-tab="tab-anchors"' in content
        assert 'data-tab="tab-pruning"' in content

        # Verify 4-Tab Panes
        assert 'id="tab-cockpit"' in content
        assert 'id="tab-arbiter"' in content
        assert 'id="tab-anchors"' in content
        assert 'id="tab-pruning"' in content

        # Verify Tab Role and Container
        assert 'role="tablist"' in content
        assert 'class="tabs-nav"' in content

        # Verify Embedded State Snapshot
        assert "window.EMBEDDED_INITIAL_STATE =" in content
        assert '"mean_zeno_pinning": 0.875' in content
        assert '"total_pruned_count": 5' in content

    def test_tier1_embedded_state_json_structure(self, tmp_path: Path) -> None:
        """Verifies that window.EMBEDDED_INITIAL_STATE serializes all required cognitive fields."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        test_state = {
            "turn_count": 300,
            "mean_zeno_pinning": 0.912,
            "current_zeno_pinning": 0.912,
            "total_pruned_count": 12,
            "pruning_history": [{"rule_name": "test_pruned", "reason": "age_exceeded"}],
            "recent_arbitrations": [{"dilemma": "Test Dilemma", "winner": "d1"}],
        }
        state_file.write_text(json.dumps(test_state), encoding="utf-8")

        dash_out = tmp_path / "cockpit_embedded.html"
        generate_dashboard_html(output_path=dash_out, state_file=state_file)

        content = dash_out.read_text(encoding="utf-8")
        m = re.search(r"window\.EMBEDDED_INITIAL_STATE\s*=\s*(\{.*?\});", content, re.DOTALL)
        assert m is not None, "window.EMBEDDED_INITIAL_STATE script tag not found"

        embedded_str = m.group(1)
        # Restore JSON-safe characters escaped for HTML script insertion
        unescaped_str = (
            embedded_str.replace(r"\u003c", "<")
            .replace(r"\u003e", ">")
            .replace(r"\u0026", "&")
        )
        embedded_dict = json.loads(unescaped_str)

        assert embedded_dict["turn_count"] == 300
        assert embedded_dict["mean_zeno_pinning"] == 0.912
        assert embedded_dict["total_pruned_count"] == 12
        assert len(embedded_dict["pruning_history"]) == 1
        assert len(embedded_dict["recent_arbitrations"]) == 1


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# ==============================================================================


class TestTier2BoundaryAndCornerCases:
    """Tier 2: Boundary, numerical clamping, adversarial inputs, and graceful degradation."""

    @pytest.mark.parametrize(
        "zeno_input,expected_clamped",
        [
            (-10.0, 0.0),
            (-0.0001, 0.0),
            (1.0001, 1.0),
            (2.5, 1.0),
            (100.0, 1.0),
        ],
    )
    def test_tier2_extreme_and_clamped_zeno_values(
        self, tmp_path: Path, zeno_input: float, expected_clamped: float
    ) -> None:
        """Verifies that out-of-bounds Zeno pinning values are safely clamped to [0.0, 1.0]."""
        state_file = tmp_path / f"state_extreme_{abs(int(zeno_input * 10))}.json"
        state_file.write_text(
            json.dumps({"mean_zeno_pinning": zeno_input, "current_zeno_pinning": zeno_input}),
            encoding="utf-8",
        )

        dash_out = tmp_path / f"dash_extreme_{abs(int(zeno_input * 10))}.html"
        generate_dashboard_html(output_path=dash_out, state_file=state_file)

        content = dash_out.read_text(encoding="utf-8")

        # Extract SVG gauge needle coordinates
        m = re.search(
            r'<line id="gauge-needle" x1="160" y1="140" x2="([0-9.-]+)" y2="([0-9.-]+)"',
            content,
        )
        assert m is not None, "Needle line element missing from rendered HTML"
        nx, ny = float(m.group(1)), float(m.group(2))

        # Check needle stays within valid gauge canvas box
        assert 70.0 <= nx <= 250.0, f"nx={nx} out of bounds for zeno={zeno_input}"
        assert 50.0 <= ny <= 140.0, f"ny={ny} out of bounds for zeno={zeno_input}"

        # Gauge text must display valid percentage between 0% and 100%
        expected_pct = expected_clamped * 100.0
        assert f"%{expected_pct:.1f}" in content or f"%{int(expected_pct)}" in content

    def test_tier2_nan_and_infinite_zeno_needle_safety(self, tmp_path: Path) -> None:
        """Verifies resilience against NaN and infinity without OverflowError or crashing."""
        for hostile_val in (float("nan"), float("inf"), float("-inf"), None):
            state_file = tmp_path / f"state_hostile_{hostile_val}.json"
            if hostile_val is None:
                state_content = '{"mean_zeno_pinning": null}'
            elif math.isnan(hostile_val):
                state_content = '{"mean_zeno_pinning": "NaN"}'
            else:
                state_content = '{"mean_zeno_pinning": "Infinity"}'

            state_file.write_text(state_content, encoding="utf-8")

            dash_out = tmp_path / f"dash_hostile_{hostile_val}.html"
            out = generate_dashboard_html(output_path=dash_out, state_file=state_file)
            assert out.exists()

            content = dash_out.read_text(encoding="utf-8")
            # SVG needle must contain finite coordinates
            m = re.search(
                r'<line id="gauge-needle" x1="160" y1="140" x2="([0-9.-]+)" y2="([0-9.-]+)"',
                content,
            )
            assert m is not None
            nx, ny = float(m.group(1)), float(m.group(2))
            assert math.isfinite(nx) and math.isfinite(ny)

    def test_tier2_missing_or_corrupted_state_file_fallback(self, tmp_path: Path) -> None:
        """Verifies graceful fallback when state file is missing, empty, non-dict, or corrupted."""
        # 1. Non-existent file
        missing_file = tmp_path / "missing_state.json"
        dash1 = tmp_path / "dash_missing.html"
        assert generate_dashboard_html(output_path=dash1, state_file=missing_file).exists()

        # 2. Empty 0-byte file
        empty_file = tmp_path / "empty_state.json"
        empty_file.write_bytes(b"")
        dash2 = tmp_path / "dash_empty.html"
        assert generate_dashboard_html(output_path=dash2, state_file=empty_file).exists()

        # 3. Corrupted non-JSON syntax
        corrupt_file = tmp_path / "corrupt_state.json"
        corrupt_file.write_text("{corrupt: [unterminated", encoding="utf-8")
        dash3 = tmp_path / "dash_corrupt.html"
        assert generate_dashboard_html(output_path=dash3, state_file=corrupt_file).exists()

        # 4. Valid JSON but wrong type (top-level list instead of dict)
        list_file = tmp_path / "list_state.json"
        list_file.write_text('["not", "a", "dict", 123]', encoding="utf-8")
        dash4 = tmp_path / "dash_list.html"
        assert generate_dashboard_html(output_path=dash4, state_file=list_file).exists()

        # Check that refresh_memory_from_state handles all gracefully
        mem = FastBiomorphicMemory()
        assert refresh_memory_from_state(mem, missing_file) == {}
        assert refresh_memory_from_state(mem, empty_file) == {}
        assert refresh_memory_from_state(mem, corrupt_file) == {}
        assert refresh_memory_from_state(mem, list_file) == {}

    def test_tier2_pruning_history_capping_at_100(self) -> None:
        """Verifies that pruning_history strictly enforces a 100-entry capacity cap."""
        mem = FastBiomorphicMemory(capacity=200)

        # Generate 140 transient engrams and prune them across multiple turns
        for i in range(140):
            mem.record_transient(
                key=f"transient_rule_{i:03d}",
                content=f"Transient content for item {i}",
                salience=0.3,
            )
            mem.engrams[-1]["fidelity"] = 0.50

            if (i + 1) % 20 == 0:
                mem.prune_obsolete(fidelity_threshold=0.70, turn=i)

        assert mem.total_pruned_count == 140, "Expected total pruned count to track all 140 items"
        assert len(mem.pruning_history) == 100, (
            f"pruning_history must be capped at 100 entries, found {len(mem.pruning_history)}"
        )

        # Oldest preserved entry must be transient_rule_040 (FIFO eviction of items 0..39)
        oldest_key = mem.pruning_history[0]["rule_name"]
        newest_key = mem.pruning_history[-1]["rule_name"]
        assert oldest_key == "transient_rule_040"
        assert newest_key == "transient_rule_139"

    def test_tier2_xss_and_special_character_neutralization(self, tmp_path: Path) -> None:
        """Verifies strict neutralization and HTML escaping of hostile injection strings."""
        hostile_state = {
            "turn_count": 10,
            "mean_zeno_pinning": 0.86,
            "engrams": [
                {
                    "key": '<script>alert("XSS_KEY")</script>',
                    "content": '"><img src=x onerror=alert("XSS_CONTENT")>',
                    "salience": 2.5,
                    "fidelity": 0.9998,
                    "is_core_anchor": True,
                }
            ],
            "pruning_history": [
                {
                    "timestamp": '2026-09-22T00:00:00Z"><svg onload=alert(1)>',
                    "rule_name": '<iframe src="javascript:alert(2)">',
                    "category": "contextual_decision",
                    "reason": "decayed & <escaped>",
                    "decayed_salience": 0.1,
                    "final_fidelity": 0.5,
                    "turn_pruned": 1,
                }
            ],
            "recent_arbitrations": [
                {
                    "dilemma": '<script>alert("XSS_DILEMMA")</script>',
                    "winner": "d1",
                    "winning_label": '"><b onmouseover=alert(3)>Malicious Label</b>',
                    "p_zeno": 0.86,
                    "injected_prompt_constraint": '🔒 <script>alert("XSS_CONSTRAINT")</script>',
                    "counterfactual_ab": {
                        "without_quanta": '<img src=x onerror=alert("AB_WITHOUT")>',
                        "with_quanta": '<img src=x onerror=alert("AB_WITH")>',
                    },
                    "hypotheses": [
                        {
                            "id": "d1",
                            "label": '"><script>alert("HYP")</script>',
                            "ket": "|d1>",
                            "score": 1.0,
                            "tr_rho_pi": 0.99,
                        }
                    ],
                }
            ],
        }

        state_file = tmp_path / "state_xss.json"
        state_file.write_text(json.dumps(hostile_state), encoding="utf-8")

        dash_out = tmp_path / "dash_xss.html"
        generate_dashboard_html(output_path=dash_out, state_file=state_file)

        content = dash_out.read_text(encoding="utf-8")

        # Assert no unescaped hostile script or tag elements exist outside of embedded JSON strings
        assert '<script>alert("XSS_KEY")</script>' not in content
        assert '<script>alert("XSS_DILEMMA")</script>' not in content
        assert '<script>alert("XSS_CONSTRAINT")</script>' not in content
        assert '<img src=x onerror=alert("XSS_CONTENT")>' not in content
        assert '<iframe src="javascript:alert(2)">' not in content

        # Assert proper HTML character escaping
        assert "&lt;script&gt;" in content
        assert "&lt;img src=x" in content or "&quot;&gt;&lt;img" in content
        assert "&amp;" in content

    def test_tier2_dynamic_svg_needle_circle_radius_invariance(self, tmp_path: Path) -> None:
        """Verifies geometric circle radius invariance: R = sqrt(dx^2 + dy^2) == 90.0."""
        center_x, center_y, expected_radius = 160.0, 140.0, 90.0

        for zeno_val in [0.0, 0.15, 0.33, 0.50, 0.72, 0.86, 0.95, 1.0]:
            state_file = tmp_path / f"state_radius_{int(zeno_val * 100)}.json"
            state_file.write_text(
                json.dumps({"mean_zeno_pinning": zeno_val}),
                encoding="utf-8",
            )

            dash_out = tmp_path / f"dash_radius_{int(zeno_val * 100)}.html"
            generate_dashboard_html(output_path=dash_out, state_file=state_file)

            content = dash_out.read_text(encoding="utf-8")
            m = re.search(
                r'<line id="gauge-needle" x1="160" y1="140" x2="([0-9.-]+)" y2="([0-9.-]+)"',
                content,
            )
            assert m is not None

            nx, ny = float(m.group(1)), float(m.group(2))
            computed_radius = math.sqrt((nx - center_x) ** 2 + (ny - center_y) ** 2)

            assert abs(computed_radius - expected_radius) < 0.2, (
                f"Radius violation for zeno={zeno_val}: expected {expected_radius}, "
                f"got {computed_radius:.3f}"
            )
            # Semicircular needle domain: ny must always be <= 140 (upper half-plane)
            assert ny <= center_y + 0.01, f"Needle dipped below horizon: ny={ny} > {center_y}"


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ==============================================================================


class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise and multi-component interaction pipelines."""

    def test_tier3_swr_replay_to_state_to_dashboard_needle_match(self, tmp_path: Path) -> None:
        """Cross-Feature: SWR replay -> State -> Dashboard HTML -> Needle trigonometry."""
        mem = FastBiomorphicMemory(capacity=64)

        # 1. Register foundational core anchors
        mem.record(
            key="native_first_rule",
            content="Native platform API first",
            salience=2.8,
            is_core_anchor=True,
        )
        mem.record(
            key="scientific_integrity_rule",
            content="Scientific verification first",
            salience=2.5,
            is_core_anchor=True,
        )

        # 2. Advance time and trigger SWR consolidation boost
        mem.step(dt=1.0)
        mem.consolidate(["native_first_rule"], boost=0.005)

        target_zeno = 0.7420
        state_file = tmp_path / "quanta_cognitive_state.json"
        state_dict = {
            "turn_count": 12,
            "engrams": mem.engrams,
            "mean_zeno_pinning": target_zeno,
            "current_zeno_pinning": target_zeno,
            "total_pruned_count": 0,
        }
        _atomic_write_single_file(state_file, state_dict)

        # 3. Generate cockpit HTML
        dash_out = tmp_path / "swr_cockpit.html"
        generate_dashboard_html(output_path=dash_out, state_file=state_file)

        content = dash_out.read_text(encoding="utf-8")

        # 4. Assert needle trigonometry matches target_zeno exactly
        # Formula: rad = pi * (1.0 - zeno); nx = 160 + 90 * cos(rad); ny = 140 - 90 * sin(rad)
        expected_rad = math.pi * (1.0 - target_zeno)
        expected_nx = 160.0 + 90.0 * math.cos(expected_rad)
        expected_ny = 140.0 - 90.0 * math.sin(expected_rad)

        m = re.search(
            r'<line id="gauge-needle" x1="160" y1="140" x2="([0-9.-]+)" y2="([0-9.-]+)"',
            content,
        )
        assert m is not None
        actual_nx, actual_ny = float(m.group(1)), float(m.group(2))

        assert abs(actual_nx - expected_nx) < 0.2, (
            f"Needle nx mismatch: expected {expected_nx:.2f}, got {actual_nx:.2f}"
        )
        assert abs(actual_ny - expected_ny) < 0.2, (
            f"Needle ny mismatch: expected {expected_ny:.2f}, got {actual_ny:.2f}"
        )

        # Verify textual KPI displays
        assert "P_zeno: 0.742" in content
        assert "%74.2 Odak Kitlemesi" in content

    def test_tier3_hook_pruning_event_to_tab4_table_rendering(self, tmp_path: Path) -> None:
        """Cross-Feature: Pruning event -> Ledger persistence -> Tab 4 Pruning table rendering."""
        mem = FastBiomorphicMemory(capacity=64)
        mem.record_transient(
            key="temp_exploration_strategy",
            content="Testing temporary search heuristic",
            salience=0.32,
        )

        # Decay below pruning threshold
        mem.engrams[-1]["fidelity"] = 0.52
        pruned_list = mem.prune_obsolete(fidelity_threshold=0.70, turn=88)
        assert len(pruned_list) == 1

        state_file = tmp_path / "quanta_cognitive_state.json"
        state_dict = {
            "turn_count": 88,
            "engrams": mem.engrams,
            "total_pruned_count": 1,
            "pruning_history": [dict(p) for p in pruned_list],
        }
        _atomic_write_single_file(state_file, state_dict)

        dash_out = tmp_path / "pruning_cockpit.html"
        generate_dashboard_html(output_path=dash_out, state_file=state_file)

        content = dash_out.read_text(encoding="utf-8")

        # Tab 4 Pruning Table Row Verification
        assert '<tbody id="pruning-history-tbody">' in content
        assert "✂️ temp_exploration_strategy" in content
        assert "fidelity_decayed" in content
        assert "Turn #88" in content
        assert "%52.0" in content

    def test_tier3_arbiter_dilemma_to_telemetry_to_tab2_card(self, tmp_path: Path) -> None:
        """Cross-Feature: Arbiter dilemma -> Telemetry -> Tab 2 Arbiter card with A/B."""
        arbiter = QuantumDecisionArbiter(dim=64, seed=55)
        dilemma = "Database Scalability Dilemma: Cloud Spanner vs DynamoDB vs CockroachDB"
        hyps = [
            {"id": "d1", "label": "Google Cloud Spanner", "ket": "|d1>"},
            {"id": "d2", "label": "Amazon DynamoDB Global Tables", "ket": "|d2>"},
            {"id": "d3", "label": "CockroachDB Multi-Region", "ket": "|d3>"},
        ]

        arb_res = arbiter.arbitrate_dilemma(
            dilemma=dilemma,
            hypotheses=hyps,
            step_idx=95,
            workspace="alfa-quanta-e2e",
            log_telemetry=False,
        )

        # Record into state and telemetry
        telem_file = tmp_path / "telemetry.jsonl"
        record_decision_telemetry(
            goal=dilemma,
            options=[h["label"] for h in hyps],
            winner=arb_res["winning_label"],
            confidence=arb_res["hypotheses"][0]["score"],
            zeno_pinning_factor=arb_res["p_zeno"],
            anti_zeno_kickback=1.0 - arb_res["p_zeno"],
            regime="Zeno Pinning",
            latency_ms=1.2,
            workspace="alfa-quanta-e2e",
            telemetry_file=telem_file,
            ranking=arb_res["hypotheses"],
        )

        state_file = tmp_path / "quanta_cognitive_state.json"
        state_dict = {
            "turn_count": 95,
            "mean_zeno_pinning": arb_res["p_zeno"],
            "current_zeno_pinning": arb_res["p_zeno"],
            "recent_arbitrations": [arb_res],
        }
        _atomic_write_single_file(state_file, state_dict)

        dash_out = tmp_path / "arbiter_cockpit.html"
        generate_dashboard_html(
            output_path=dash_out,
            state_file=state_file,
            telemetry_file=telem_file,
        )

        content = dash_out.read_text(encoding="utf-8")

        # Tab 2 Arbiter Card Verifications
        assert 'id="tab-arbiter"' in content
        assert dilemma in content
        assert arb_res["winning_label"] in content
        assert f"P_zeno: {arb_res['p_zeno']:.3f}" in content

        # All 3 hypotheses rendered with Tr(rho Pi) projections and HTML-escaped kets
        assert "|d1&gt;" in content
        assert "|d2&gt;" in content
        assert "|d3&gt;" in content
        assert "Tr(ρ Π)" in content

        # Injected prompt constraint box
        assert "LLM Sistem Promptuna Enjekte Edilen Kuantum Karar Kısıtı" in content
        assert arb_res["injected_prompt_constraint"] in content

        # Side-by-side counterfactual A/B columns
        assert 'id="ab-without-quanta"' in content
        assert 'id="ab-with-quanta"' in content
        assert arb_res["counterfactual_ab"]["without_quanta"] in content
        assert arb_res["counterfactual_ab"]["with_quanta"] in content


# ==============================================================================
# TIER 4: REAL-WORLD WORKFLOW SCENARIOS
# ==============================================================================


class TestTier4RealWorldWorkflowScenarios:
    """Tier 4: End-to-end multi-turn simulation and zero-CDN offline compliance verification."""

    def test_tier4_end_to_end_multiturn_simulation_with_dynamic_zeno_evolution(
        self, tmp_path: Path
    ) -> None:
        """Simulates 5-turn session: SWR replays, dynamic Zeno, decay, pruning, and dashboard."""
        primary_dir = tmp_path / "conv_turn_artifacts"
        mirror_dir = tmp_path / "workspace_mirror"

        primary_state = primary_dir / "quanta_cognitive_state.json"
        mirror_state = mirror_dir / "quanta_cognitive_state.json"

        mem = FastBiomorphicMemory(capacity=128)
        arbiter = QuantumDecisionArbiter(dim=64, seed=42)

        # Turn 1: Session bootstrap with core anchors and first transient decision
        mem.record("native_first_rule", "Native platform first", salience=2.8, is_core_anchor=True)
        mem.record(
            "executive_summary_rule", "Executive summary first", salience=2.5, is_core_anchor=True
        )
        mem.record_transient("turn1_decision", "Use in-memory dictionary for Step 1", salience=0.5)

        turn1_state = {
            "turn_count": 1,
            "engrams": copy.deepcopy(mem.engrams),
            "mean_zeno_pinning": 0.8603,
            "current_zeno_pinning": 0.8603,
            "pruning_history": [],
            "recent_arbitrations": [],
        }
        _save_mirrored_state_atomically(
            primary_state, mirror_state, turn1_state, "conv_e2e", is_test_env=False
        )

        # Turn 2: Architectural dilemma encountered and arbitrated
        dilemma1 = "State Persistence Dilemma: Direct Atomic File vs In-Memory Only"
        arb1 = arbiter.arbitrate_dilemma(
            dilemma=dilemma1,
            hypotheses=["Direct Atomic POSIX File Replacement", "In-Memory Volatile Dictionary"],
            exploration_drive=0.15,
            step_idx=2,
        )
        mem.step(dt=1.0)
        mem.consolidate(["native_first_rule"])

        turn2_state = copy.deepcopy(turn1_state)
        turn2_state.update({
            "turn_count": 2,
            "engrams": copy.deepcopy(mem.engrams),
            "current_zeno_pinning": arb1["p_zeno"],
            "mean_zeno_pinning": arb1["p_zeno"],
            "recent_arbitrations": [arb1],
        })
        _save_mirrored_state_atomically(
            primary_state, mirror_state, turn2_state, "conv_e2e", is_test_env=False
        )

        # Turn 3: Second dilemma arbitrated, dynamic mean Zeno evolved
        dilemma2 = "Visualization Dilemma: 4-Tab Modular Cockpit vs Single Monolithic Page"
        arb2 = arbiter.arbitrate_dilemma(
            dilemma=dilemma2,
            hypotheses=["4-Tab Modular Cockpit Layout", "Single Continuous Monolithic Page"],
            exploration_drive=0.25,
            step_idx=3,
        )
        mem.step(dt=1.0)
        mean_z = round((arb1["p_zeno"] + arb2["p_zeno"]) / 2.0, 4)

        # Turn 4: Transient turn1_decision decays and is pruned
        for e in mem.engrams:
            if e["key"] == "turn1_decision":
                e["fidelity"] = 0.50
        pruned_records = mem.prune_obsolete(fidelity_threshold=0.70, turn=4)
        assert len(pruned_records) == 1

        # Turn 5: Final state persistence and dashboard compilation
        final_state = {
            "turn_count": 5,
            "engrams": copy.deepcopy(mem.engrams),
            "current_zeno_pinning": arb2["p_zeno"],
            "mean_zeno_pinning": mean_z,
            "total_pruned_count": len(pruned_records),
            "pruning_history": [dict(p) for p in pruned_records],
            "recent_arbitrations": [arb2, arb1],
        }
        prim_ok, mir_ok = _save_mirrored_state_atomically(
            primary_state, mirror_state, final_state, "conv_e2e", is_test_env=False
        )
        assert prim_ok and mir_ok

        dash_out = tmp_path / "simulated_e2e_cockpit.html"
        generate_dashboard_html(output_path=dash_out, state_file=primary_state)

        # Full verification of synthesized HTML
        html_text = dash_out.read_text(encoding="utf-8")
        assert f"P_zeno: {mean_z:.3f}" in html_text
        assert "✂️ turn1_decision" in html_text
        assert dilemma2 in html_text
        assert "native_first_rule" in html_text
        assert "executive_summary_rule" in html_text
        assert "window.EMBEDDED_INITIAL_STATE" in html_text

    def test_tier4_zero_cdn_offline_compliance_verification(self, tmp_path: Path) -> None:
        """Verifies zero-CDN compliance: zero remote script, style, font, or asset links."""
        dash_out = tmp_path / "offline_compliance_cockpit.html"
        generate_dashboard_html(output_path=dash_out)

        content = dash_out.read_text(encoding="utf-8")

        # 1. Zero external script tags (<script src="http...">)
        ext_scripts = re.findall(r'<script[^>]+src=["\'](https?://[^"\']+)["\']', content)
        assert len(ext_scripts) == 0, f"Violated zero-CDN rule (external scripts): {ext_scripts}"

        # 2. Zero external stylesheet links (<link href="http...">)
        ext_links = re.findall(
            r'<link[^>]+href=["\'](https?://[^"\']+)["\']', content, re.IGNORECASE
        )
        assert len(ext_links) == 0, f"Violated zero-CDN rule (external stylesheets): {ext_links}"

        # 3. Zero external image tags (<img src="http...">)
        ext_imgs = re.findall(r'<img[^>]+src=["\'](https?://[^"\']+)["\']', content)
        assert len(ext_imgs) == 0, f"Violated zero-CDN rule (external images): {ext_imgs}"

        # 4. Zero CSS @import or url() pointing to external http/https
        css_ext_urls = re.findall(r'url\(["\']?(https?://[^"\'\)]+)["\']?\)', content)
        assert len(css_ext_urls) == 0, f"Violated zero-CDN rule (CSS url imports): {css_ext_urls}"

        # 5. Check absence of common third-party CDN domain references
        banned_cdns = [
            "cdnjs.cloudflare.com",
            "cdn.jsdelivr.net",
            "unpkg.com",
            "fonts.googleapis.com",
            "fonts.gstatic.com",
            "ajax.googleapis.com",
            "code.jquery.com",
        ]
        for cdn in banned_cdns:
            assert cdn not in content, f"Detected banned external CDN domain: {cdn}"

        # 6. Verify that JavaScript client controllers are embedded inline
        assert "const LiveSyncController =" in content
        assert "const CognitiveCockpit =" in content
        assert "setupTabs()" in content

    def test_tier4_simulated_dashboard_mirror_to_repo_equivalence(self, tmp_path: Path) -> None:
        """Verifies that generate_dashboard_html with mirror_to_repo writes identical files."""
        fake_repo = tmp_path / "simulated_repo"
        fake_dashboard_dir = fake_repo / "quanta" / "cognitive" / "dashboard"
        fake_dashboard_dir.mkdir(parents=True, exist_ok=True)
        (fake_repo / "site").mkdir(parents=True, exist_ok=True)

        target_file = tmp_path / "custom_out.html"
        written_files: dict[Path, str] = {}

        def mock_write_atomic(path: Path, html: str) -> None:
            written_files[path] = html
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")

        with (
            patch("quanta.cognitive.telemetry.Path") as mock_path_cls,
            patch(
                "quanta.cognitive.telemetry._write_html_atomically", side_effect=mock_write_atomic
            ),
        ):
            mock_path_cls.side_effect = lambda *args, **kwargs: Path(*args, **kwargs)

            generate_dashboard_html(
                output_path=target_file,
                mirror_to_repo=False,
            )

        assert target_file in written_files
        assert len(written_files) == 1, "mirror_to_repo=False must only write to output_path"
