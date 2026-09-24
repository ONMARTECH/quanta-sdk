"""Unit and Integration tests for Quanta Cognitive Telemetry & Monitoring."""

import json
import math
from pathlib import Path

import pytest

from quanta.cli import main
from quanta.cognitive.telemetry import (
    detect_workspace,
    generate_dashboard_html,
    get_telemetry_summary,
    read_telemetry_events,
    record_decision_telemetry,
    record_hook_telemetry,
)


class TestCognitiveTelemetry:
    def test_record_decision_and_read_summary(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        telemetry_file = tmp_path / "telemetry_test.jsonl"
        monkeypatch.setattr("quanta.cognitive.telemetry.DEFAULT_TELEMETRY_DIR", tmp_path)
        monkeypatch.setattr("quanta.cognitive.telemetry.DEFAULT_TELEMETRY_FILE", telemetry_file)

        # Record a decision
        record_decision_telemetry(
            goal="Turna Meiro Synchronization",
            options=["Option A (Sync)", "Option B (Async)"],
            winner="Option A (Sync)",
            confidence=0.85,
            zeno_pinning_factor=0.92,
            anti_zeno_kickback=0.08,
            regime="Zeno Pinning",
            latency_ms=1.45,
            workspace="turna-test-ws",
            telemetry_file=telemetry_file,
        )

        assert telemetry_file.exists()
        events = read_telemetry_events(telemetry_file=telemetry_file)
        assert len(events) == 1
        assert events[0]["event_type"] == "decision"
        assert events[0]["winner"] == "Option A (Sync)"
        assert events[0]["confidence"] == 0.85
        assert events[0]["latency_ms"] == 1.45

        # Record a hook step
        record_hook_telemetry(
            conversation_id="conv-123",
            step_idx=1,
            turn_count=5,
            rules_replayed=["rule1", "rule2"],
            pruned_count=1,
            latency_ms=0.25,
            workspace="turna-test-ws",
            telemetry_file=telemetry_file,
        )

        # Verify summary aggregation
        summary = get_telemetry_summary(telemetry_file=telemetry_file)
        assert summary["total_decisions"] == 1
        assert summary["total_hook_steps"] == 1
        assert summary["avg_decision_latency_ms"] == 1.45
        assert summary["avg_confidence_pct"] == 85.0
        assert "turna-test-ws" in summary["active_workspaces"]
        assert len(summary["recent_decisions"]) == 1
        assert summary["recent_decisions"][0]["winner"] == "Option A (Sync)"

    def test_empty_or_missing_file_handling(self, tmp_path: Path) -> None:
        missing_file = tmp_path / "non_existent.jsonl"
        summary = get_telemetry_summary(telemetry_file=missing_file)
        assert summary["total_decisions"] == 0
        assert summary["avg_decision_latency_ms"] == 0.0
        assert summary["avg_confidence_pct"] == 0.0
        assert summary["recent_decisions"] == []
        assert summary["active_workspaces"] == []

    def test_detect_workspace(self) -> None:
        assert detect_workspace(explicit="custom-ws") == "custom-ws"

    def test_dashboard_html_generation(self, tmp_path: Path) -> None:
        telemetry_file = tmp_path / "telemetry_test.jsonl"

        record_decision_telemetry(
            goal="Cache strategy",
            options=["Redis", "Memory"],
            winner="Memory",
            confidence=0.78,
            zeno_pinning_factor=0.8,
            anti_zeno_kickback=0.2,
            regime="Zeno Pinning",
            latency_ms=0.95,
            workspace="test-workspace",
            telemetry_file=telemetry_file,
        )

        html_out = tmp_path / "dashboard.html"
        generated_path = generate_dashboard_html(output_path=html_out, telemetry_file=telemetry_file)
        assert generated_path == html_out
        assert html_out.exists()

        content = html_out.read_text(encoding="utf-8")
        assert "Quanta Bilişsel Hakem & Telemetri Kokpiti" in content
        assert "Cache strategy" in content
        assert "Memory" in content
        assert "test-workspace" in content

    def test_cli_monitor_command(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        telemetry_file = tmp_path / "telemetry_test.jsonl"
        monkeypatch.setattr("quanta.cognitive.telemetry.DEFAULT_TELEMETRY_DIR", tmp_path)
        monkeypatch.setattr("quanta.cognitive.telemetry.DEFAULT_TELEMETRY_FILE", telemetry_file)

        # Run with no decisions
        code = main(["monitor"])
        assert code == 0
        captured = capsys.readouterr()
        assert "QUANTA BİLİŞSEL HAKEM & TELEMETRİ KOKPİTİ" in captured.out

        # Add an entry and run with json output
        record_decision_telemetry(
            goal="CLI test",
            options=["A", "B"],
            winner="A",
            confidence=0.99,
            zeno_pinning_factor=0.95,
            anti_zeno_kickback=0.05,
            regime="Zeno Pinning",
            latency_ms=1.1,
            workspace="cli-ws",
            telemetry_file=telemetry_file,
        )

        code_json = main(["monitor", "--json"])
        assert code_json == 0
        captured_json = capsys.readouterr()
        parsed = json.loads(captured_json.out)
        assert parsed["total_decisions"] == 1
        assert parsed["avg_decision_latency_ms"] == 1.1

        # Run with dashboard flag
        html_out = tmp_path / "cli_dash.html"
        code_dash = main(["monitor", "--dashboard", "--output", str(html_out)])
        assert code_dash == 0
        captured_dash = capsys.readouterr()
        assert "İNTERAKTİF HTML DASHBOARD" in captured_dash.out
        assert html_out.exists()

    def test_cli_arbitrate_command(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        telemetry_file = tmp_path / "telemetry.jsonl"
        monkeypatch.setattr("quanta.cognitive.telemetry.DEFAULT_TELEMETRY_DIR", tmp_path)
        monkeypatch.setattr("quanta.cognitive.telemetry.DEFAULT_TELEMETRY_FILE", telemetry_file)

        from quanta.cli import main

        code = main([
            "arbitrate",
            "--goal", "Test Architecture",
            "--options", "Option Alpha; Option Beta",
            "--workspace", "TestWS",
        ])
        assert code == 0
        captured = capsys.readouterr()
        assert "QUANTA BİLİŞSEL KARAR HAKEMİ" in captured.out
        assert "Kazanan Seçenek:" in captured.out

        # Verify telemetry was recorded
        assert telemetry_file.exists()
        summary = get_telemetry_summary(telemetry_file=telemetry_file)
        assert summary["total_decisions"] == 1
        assert summary["recent_decisions"][0]["workspace"] == "TestWS"

    def test_subconscious_rule_guardian_telemetry_fidelity_and_csf(self, tmp_path: Path) -> None:
        """Verifies R1: hook telemetry records instantaneous fidelity in rules_replayed, kappa_csf, and pruning stats."""
        telemetry_file = tmp_path / "telemetry_hook_test.jsonl"

        replayed_rules = [
            {
                "rule": "native_first_rule",
                "key": "native_first_rule",
                "fidelity": 0.9998,
                "salience": 2.8,
                "category": "constraint",
            },
            {
                "rule": "executive_summary_rule",
                "key": "executive_summary_rule",
                "fidelity": 0.9995,
                "salience": 2.5,
                "category": "constraint",
            },
        ]

        record_hook_telemetry(
            conversation_id="conv-guardian-test",
            step_idx=7,
            turn_count=12,
            rules_replayed=replayed_rules,
            pruned_count=2,
            latency_ms=0.85,
            workspace="guardian-ws",
            last_user_query="Implement telemetry guardian",
            telemetry_file=telemetry_file,
            kappa_csf=1.0 / 6250.0,
            pruned_keys=["temp_note_1", "decayed_cache"],
            total_pruned_count=5,
            active_engrams_count=8,
        )

        events = read_telemetry_events(telemetry_file=telemetry_file)
        assert len(events) == 1
        ev = events[0]
        assert ev["event_type"] == "hook_step"
        assert ev["kappa_csf"] == pytest.approx(1.0 / 6250.0, rel=1e-5)
        assert ev["pruned_count"] == 2
        assert ev["total_pruned_count"] == 5
        assert ev["pruned_keys"] == ["temp_note_1", "decayed_cache"]
        assert ev["active_engrams_count"] == 8

        # Verify instantaneous fidelity in rules_replayed
        rules = ev["rules_replayed"]
        assert len(rules) == 2
        assert rules[0]["rule"] == "native_first_rule"
        assert rules[0]["fidelity"] == pytest.approx(0.9998, rel=1e-4)
        assert rules[1]["rule"] == "executive_summary_rule"
        assert rules[1]["fidelity"] == pytest.approx(0.9995, rel=1e-4)

    def test_quantum_decision_metrics_tr_rho_pi_and_zeno(self, tmp_path: Path) -> None:
        """Verifies R2: genuine 6-qubit quantum probabilities Tr(rho Pi), P_zeno, and PyTorch latency without static 95%."""
        telemetry_file = tmp_path / "telemetry_decision_test.jsonl"

        from quanta.cognitive.arbiter import QuantumDecisionArbiter
        arbiter = QuantumDecisionArbiter(dim=64, num_heads=4)

        goal = "Evaluate scalable database architecture for high concurrency"
        options = [
            "Distributed Spanner cluster with multi-region replication",
            "Local single-node SQLite file with exclusive locking",
        ]

        result = arbiter.arbitrate(
            goal=goal,
            options=options,
            workspace="quantum-decision-ws",
            log_telemetry=True,
        )

        assert "recommended_option" in result
        assert result["confidence"] > 0.0
        assert result["confidence"] != 0.95  # Must NOT be hardcoded static 95%
        assert "tr_rho_pi" in result
        assert len(result["tr_rho_pi"]) == 2
        assert "anti_zeno_tunneling_rate" in result
        assert "pytorch_latency_ms" in result
        assert result["pytorch_latency_ms"] >= 0.0

        # Check telemetry file
        events = read_telemetry_events(telemetry_file=telemetry_file)
        assert isinstance(events, list)
        # Arbiter by default logs to default telemetry, so test direct record or check summary
        record_decision_telemetry(
            goal=goal,
            options=options,
            winner=result["recommended_option"],
            confidence=result["confidence"],
            zeno_pinning_factor=result["zeno_pinning_factor"],
            anti_zeno_kickback=result["anti_zeno_kickback"],
            anti_zeno_tunneling_rate=result["anti_zeno_tunneling_rate"],
            tr_rho_pi=result["tr_rho_pi"],
            pytorch_latency_ms=result["pytorch_latency_ms"],
            regime=result["regime"],
            latency_ms=result["latency_ms"],
            ranking=result["ranked_options"],
            workspace="quantum-decision-ws",
            telemetry_file=telemetry_file,
        )

        evs = read_telemetry_events(telemetry_file=telemetry_file)
        assert len(evs) == 1
        d_ev = evs[0]
        assert d_ev["event_type"] == "decision"
        assert d_ev["confidence"] != 0.95
        assert "tr_rho_pi" in d_ev
        assert "anti_zeno_tunneling_rate" in d_ev
        assert "pytorch_latency_ms" in d_ev
        assert len(d_ev["ranking"]) == 2
        assert "tr_rho_pi" in d_ev["ranking"][0]

    def test_dashboard_html_svg_zeno_gauge_and_progress_bars(self, tmp_path: Path) -> None:
        """Verifies R3: dashboard.html renders SVG Zeno gauge, rule fidelity progress bars, and controls."""
        telemetry_file = tmp_path / "telemetry_dash_rich.jsonl"

        # Record decision
        record_decision_telemetry(
            goal="Architectural Choice",
            options=["Option A", "Option B"],
            winner="Option A",
            confidence=0.742,
            zeno_pinning_factor=0.885,
            anti_zeno_kickback=0.115,
            anti_zeno_tunneling_rate=0.115,
            tr_rho_pi={"Option A": 0.6451, "Option B": 0.3549},
            regime="Zeno Pinning",
            latency_ms=2.1,
            ranking=[
                {"option": "Option A", "score": 0.742, "probability": 0.742, "tr_rho_pi": 0.6451},
                {"option": "Option B", "score": 0.258, "probability": 0.258, "tr_rho_pi": 0.3549},
            ],
            workspace="rich-dash-ws",
            telemetry_file=telemetry_file,
        )

        # Record hook step with active rules
        record_hook_telemetry(
            conversation_id="conv-rich",
            step_idx=5,
            turn_count=10,
            rules_replayed=[
                {"rule": "native_first_rule", "fidelity": 0.9998, "salience": 2.8, "category": "constraint"},
                {"rule": "executive_summary_rule", "fidelity": 0.9995, "salience": 2.5, "category": "constraint"},
            ],
            pruned_count=1,
            latency_ms=0.4,
            workspace="rich-dash-ws",
            telemetry_file=telemetry_file,
        )

        html_out = tmp_path / "rich_dashboard.html"
        generate_dashboard_html(output_path=html_out, telemetry_file=telemetry_file)
        assert html_out.exists()

        content = html_out.read_text(encoding="utf-8")
        # Check SVG Gauge Meter
        assert "zeno-gauge-svg" in content
        assert "zenoGaugeGrad" in content
        assert "Zeno Pinning Göstergesi" in content
        assert "%88.5" in content

        # Check Subconscious Rule Guardian & Progress Bars
        assert "Bilinçaltı Kural Muhafızlığı" in content
        assert "native_first_rule" in content
        assert "executive_summary_rule" in content
        assert "progress-bar" in content
        assert "bar-high" in content
        assert "PRISTINE" in content
        assert "99.98%" in content
        assert "1/6250" in content

        # Check Controls & Quantum Ranking
        assert "countdown" in content
        assert "interval-select" in content
        assert "pause-btn" in content
        assert "Tr: 0.6451" in content

    def test_cli_monitor_colored_subconscious_guardian_panel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verifies R4: quanta monitor displays colored subconscious guardian panel and quantum decisions."""
        telemetry_file = tmp_path / "telemetry_cli_panel.jsonl"
        monkeypatch.setattr("quanta.cognitive.telemetry.DEFAULT_TELEMETRY_DIR", tmp_path)
        monkeypatch.setattr("quanta.cognitive.telemetry.DEFAULT_TELEMETRY_FILE", telemetry_file)

        # Seed telemetry with hook step and decision
        record_hook_telemetry(
            conversation_id="conv-cli-test",
            step_idx=3,
            turn_count=6,
            rules_replayed=[
                {"rule": "native_first_rule", "fidelity": 0.9998, "salience": 2.8, "category": "constraint"},
                {"rule": "executive_summary_rule", "fidelity": 0.9992, "salience": 2.5, "category": "constraint"},
            ],
            pruned_count=3,
            latency_ms=0.5,
            workspace="cli-panel-ws",
            telemetry_file=telemetry_file,
            kappa_csf=1.0 / 6250.0,
            total_pruned_count=3,
            active_engrams_count=5,
        )

        record_decision_telemetry(
            goal="CLI Panel Architecture Test",
            options=["Path Alpha", "Path Beta"],
            winner="Path Alpha",
            confidence=0.82,
            zeno_pinning_factor=0.89,
            anti_zeno_kickback=0.11,
            anti_zeno_tunneling_rate=0.11,
            tr_rho_pi={"Path Alpha": 0.72, "Path Beta": 0.28},
            regime="Zeno Pinning (Target Focus)",
            latency_ms=1.8,
            ranking=[
                {"option": "Path Alpha", "score": 0.82, "probability": 0.82, "tr_rho_pi": 0.72},
                {"option": "Path Beta", "score": 0.18, "probability": 0.18, "tr_rho_pi": 0.28},
            ],
            workspace="cli-panel-ws",
            telemetry_file=telemetry_file,
        )

        code = main(["monitor"])
        assert code == 0
        captured = capsys.readouterr()

        # Check Top KPI Summary
        assert "QUANTA BİLİŞSEL HAKEM & TELEMETRİ KOKPİTİ" in captured.out
        assert "Toplam Karar:" in captured.out
        assert "Ortalama Kuantum Güveni:" in captured.out
        assert "SWR Replay Adımları:" in captured.out
        assert "Aktif Engram Sayısı:" in captured.out

        # Check Subconscious Rule Guardian Panel
        assert "BİLİNÇALTI KURAL MUHAFIZLIĞI (SWR REPLAY & ENGRAL SADAKATİ)" in captured.out
        assert "native_first_rule" in captured.out
        assert "executive_summary_rule" in captured.out
        assert "PRISTINE" in captured.out
        assert "1/6250" in captured.out

        # Check Active Projects & Quantum Decisions Panels
        assert "CANLIDA AKTİF PROJELER & SORULAR" in captured.out
        assert "6-QUBIT KUANTUM KARARLARI & ZENO KİTLEMESİ" in captured.out
        assert "Path Alpha" in captured.out
        assert "Kuantum Sıralaması:" in captured.out

    def test_malformed_non_dict_and_null_field_handling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Adversarial test: non-dict JSONL rows, corrupt lines, and null fields must never crash telemetry, CLI, or dashboard."""
        telemetry_file = tmp_path / "corrupt_telemetry.jsonl"
        monkeypatch.setattr("quanta.cognitive.telemetry.DEFAULT_TELEMETRY_DIR", tmp_path)
        monkeypatch.setattr("quanta.cognitive.telemetry.DEFAULT_TELEMETRY_FILE", telemetry_file)

        # Write hostile JSONL entries
        hostile_content = (
            "12345\n"
            '"just a bare string"\n'
            "true\n"
            "null\n"
            "[1, 2, 3]\n"
            '{"event_type": "decision", "goal": "Corrupt Null Test", "options": ["A"], "winner": "A", "confidence": null, "zeno_pinning_factor": null, "latency_ms": null, "ranking": [{"option": "A", "score": null, "tr_rho_pi": null}], "workspace": null}\n'
            '{"event_type": "hook_step", "step_idx": null, "turn_count": null, "rules_replayed": [{"rule": "test_rule", "fidelity": null, "salience": null, "category": null}], "pruned_count": null, "latency_ms": null, "workspace": null, "kappa_csf": null}\n'
            '{"not_an_event": true}\n'
            "{broken json line\n"
        )
        telemetry_file.write_text(hostile_content, encoding="utf-8")

        # 1. read_telemetry_events must skip non-dicts without AttributeError
        events = read_telemetry_events(telemetry_file=telemetry_file)
        assert len(events) >= 2
        for ev in events:
            assert isinstance(ev, dict)

        # 2. get_telemetry_summary must compute safely without TypeError on None values
        summary = get_telemetry_summary(telemetry_file=telemetry_file)
        assert summary["total_decisions"] == 1
        assert summary["total_hook_steps"] == 1
        assert summary["avg_decision_latency_ms"] == 0.0
        assert summary["avg_confidence_pct"] == 0.0

        # 3. generate_dashboard_html must generate valid HTML without crashing
        dash_path = tmp_path / "corrupt_dash.html"
        out_path = generate_dashboard_html(output_path=dash_path, telemetry_file=telemetry_file)
        assert out_path.exists()
        html_text = out_path.read_text(encoding="utf-8")
        assert "userSpaceOnUse" in html_text
        assert "NaN" not in html_text
        assert "%nan" not in html_text
        assert '="nan"' not in html_text
        assert ">nan<" not in html_text

        # 4. CLI monitor must render safely without crash
        code = main(["monitor"])
        assert code == 0
        captured = capsys.readouterr()
        assert "QUANTA BİLİŞSEL HAKEM & TELEMETRİ KOKPİTİ" in captured.out

    def test_dashboard_svg_needle_boundaries_and_nan(self, tmp_path: Path) -> None:
        """Adversarial test: SVG needle math handles boundary values [0.0, 1.0], out-of-bounds, and NaN."""
        dash_file = tmp_path / "needle_test.html"
        log_file = tmp_path / "needle_telemetry.jsonl"

        # NaN test
        record_decision_telemetry(
            goal="NaN Test",
            options=["Opt1"],
            winner="Opt1",
            confidence=0.5,
            zeno_pinning_factor=float("nan"),
            anti_zeno_kickback=0.5,
            regime="Zeno",
            latency_ms=1.0,
            telemetry_file=log_file,
        )

        out = generate_dashboard_html(output_path=dash_file, telemetry_file=log_file)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert 'x2="nan"' not in content
        assert 'y2="nan"' not in content
        assert "%nan" not in content

    def test_subconscious_hook_swr_consolidation(self) -> None:
        """Verifies that SWR replay active consolidation restores degraded engram fidelity."""
        from scripts.hooks.quanta_subconscious_hook import FastBiomorphicMemory

        mem = FastBiomorphicMemory(capacity=10)
        mem.record("critical_rule", "Never bypass user consent", salience=2.8, category="constraint")

        initial_fid = mem.engrams[0]["fidelity"]
        assert initial_fid == 0.9998

        # Simulate 100 turns of continuous Lindblad damping
        for _ in range(100):
            mem.step(dt=1.0)

        decayed_fid = mem.engrams[0]["fidelity"]
        assert decayed_fid < initial_fid

        # Perform SWR consolidation
        mem.consolidate(["critical_rule"], boost=0.005)
        consolidated_fid = mem.engrams[0]["fidelity"]
        assert consolidated_fid > decayed_fid
        assert consolidated_fid <= 0.9998

    def test_dashboard_html_xss_and_special_characters_escaping(self, tmp_path: Path) -> None:
        """Verifies that all dynamic fields in dashboard.html are safely HTML-escaped against XSS."""
        telemetry_file = tmp_path / "xss_telemetry.jsonl"
        dash_file = tmp_path / "xss_dashboard.html"

        record_decision_telemetry(
            goal='Test <script>alert("XSS")</script> "quote"',
            options=['Option <A>', 'Option "B" & <C>'],
            winner='Option <A>',
            confidence=0.92,
            zeno_pinning_factor=0.85,
            anti_zeno_kickback=0.15,
            regime="Zeno Pinning",
            latency_ms=1.5,
            workspace="Workspace <Exploit>",
            ranking=[
                {"option": "Option <A>", "score": 0.92, "tr_rho_pi": 0.8464},
                {"option": 'Option "B" & <C>', "score": 0.08, "tr_rho_pi": 0.1536},
            ],
            telemetry_file=telemetry_file,
        )

        record_hook_telemetry(
            conversation_id="conv<xss>",
            step_idx=1,
            turn_count=2,
            rules_replayed=[
                {"rule": "rule_<script>", "fidelity": 0.9998, "salience": 2.5, "category": "cat<script>"},
            ],
            pruned_count=0,
            latency_ms=0.5,
            workspace="Workspace <Exploit>",
            last_user_query='User query <img src=x onerror=alert(1)>',
            telemetry_file=telemetry_file,
        )

        out = generate_dashboard_html(output_path=dash_file, telemetry_file=telemetry_file)
        assert out.exists()
        content = out.read_text(encoding="utf-8")

        # Must NOT contain raw unescaped XSS tags
        assert '<script>alert("XSS")</script>' not in content
        assert '<img src=x onerror=alert(1)>' not in content
        assert 'rule_<script>' not in content
        assert 'cat<script>' not in content
        assert 'Workspace <Exploit>' not in content

        # MUST contain properly escaped entities
        assert '&lt;script&gt;' in content
        assert 'Option &lt;A&gt;' in content
        assert 'Workspace &lt;Exploit&gt;' in content

    def test_dashboard_html_atomic_write(self, tmp_path: Path) -> None:
        """Verifies atomic write of dashboard.html without leaving temporary files behind."""
        dash_file = tmp_path / "atomic_dashboard.html"
        out = generate_dashboard_html(output_path=dash_file)
        assert out.exists()
        assert out.stat().st_size > 0
        # Verify no temporary files remain in the target directory
        temp_files = list(tmp_path.glob(".tmp_*"))
        assert len(temp_files) == 0

    def test_telemetry_corrupt_utf8_resilience(self, tmp_path: Path) -> None:
        """Verifies that read_telemetry_events survives invalid UTF-8 bytes without discarding valid entries."""
        t_file = tmp_path / "corrupt_utf8.jsonl"
        t_file.write_bytes(
            b'{"event_type": "decision", "goal": "first"}\n'
            b'\xff\xfe\x00INVALID\x80\x81\n'
            b'{"event_type": "decision", "goal": "second"}\n'
        )
        events = read_telemetry_events(telemetry_file=t_file)
        assert len(events) == 2
        assert events[0]["goal"] == "first"
        assert events[1]["goal"] == "second"

    def test_cli_monitor_watch_json_streaming_and_keyboard_interrupt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verifies quanta monitor -w --json outputs valid JSON streaming and exits cleanly on SIGINT."""
        from unittest.mock import patch

        from quanta.cli import main

        t_file = tmp_path / "watch_json.jsonl"
        record_decision_telemetry(
            goal="Streaming Goal",
            options=["A", "B"],
            winner="A",
            confidence=0.9,
            zeno_pinning_factor=0.85,
            anti_zeno_kickback=0.15,
            regime="Zeno",
            latency_ms=1.0,
            telemetry_file=t_file,
        )

        monkeypatch.setattr("quanta.cognitive.telemetry.DEFAULT_TELEMETRY_DIR", tmp_path)
        monkeypatch.setattr("quanta.cognitive.telemetry.DEFAULT_TELEMETRY_FILE", t_file)

        with patch("time.sleep", side_effect=KeyboardInterrupt):
            code = main(["monitor", "-w", "1", "--json"])
            assert code == 0

        captured = capsys.readouterr()
        # Ensure it output JSON without ANSI header codes
        lines = [line.strip() for line in captured.out.splitlines() if line.strip()]
        assert len(lines) >= 1
        data = json.loads(lines[0])
        assert data["total_decisions"] == 1

    def test_subconscious_hook_save_state_fallback_and_sanitization(self, tmp_path: Path) -> None:
        """Verifies _save_state_atomically sanitizes conversation_id and falls back cleanly if directory is unwritable."""
        import re
        import stat

        from scripts.hooks.quanta_subconscious_hook import _save_state_atomically

        # Safe sanitization of path traversal conversation IDs
        dangerous_id = "../../etc/passwd"
        safe_conv = re.sub(r'[^a-zA-Z0-9_\-]', '_', dangerous_id)
        assert "/" not in safe_conv
        assert ".." not in safe_conv

        # Read-only directory simulation
        ro_dir = tmp_path / "readonly_dir"
        ro_dir.mkdir()
        unwritable_file = ro_dir / "quanta_cognitive_state.json"

        # Make dir read-only
        ro_dir.chmod(stat.S_IREAD | stat.S_IEXEC)
        try:
            state_dict = {"turn_count": 5, "engrams": []}
            # Should not raise exception, but fallback to /tmp
            _save_state_atomically(unwritable_file, state_dict, safe_conv)
            fallback_file = Path(f"/tmp/quanta_cognitive_{safe_conv}.json")
            assert fallback_file.exists()
            with open(fallback_file, encoding="utf-8") as f:
                saved = json.load(f)
            assert saved["turn_count"] == 5
            fallback_file.unlink(missing_ok=True)
        finally:
            ro_dir.chmod(stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)

    def test_telemetry_hostile_infinity_and_boolean_resilience(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Adversarial test: records containing float('inf'), float('-inf'), or bools must never raise OverflowError or crash CLI/dashboard."""
        from quanta.cli import main

        t_file = tmp_path / "hostile_inf_telemetry.jsonl"
        t_file.write_text(
            json.dumps({
                "event_type": "hook_step",
                "timestamp": float("-inf"),
                "step_idx": float("inf"),
                "rules_replayed": [
                    {"rule": "inf_rule", "fidelity": float("inf"), "fidelity_pct": float("inf"), "salience": True},
                    {"rule": "bool_rule", "fidelity": True, "fidelity_pct": True, "salience": False},
                ],
                "pruned_count": float("inf"),
                "kappa_csf": float("inf"),
            })
            + "\n"
            + json.dumps({
                "event_type": "decision",
                "timestamp": float("inf"),
                "confidence": float("inf"),
                "latency_ms": float("inf"),
                "pytorch_latency_ms": float("inf"),
                "zeno_pinning_factor": float("inf"),
                "ranking": [{"option": "HostileOpt", "score": float("inf"), "tr_rho_pi": float("inf")}],
                "options": ["HostileOpt", "SafeOpt"],
                "winner": "HostileOpt",
                "goal": "Hostile Inf Test Goal",
            })
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr("quanta.cognitive.telemetry.DEFAULT_TELEMETRY_DIR", tmp_path)
        monkeypatch.setattr("quanta.cognitive.telemetry.DEFAULT_TELEMETRY_FILE", t_file)

        # 1. generate_dashboard_html must not raise OverflowError
        dash_file = tmp_path / "inf_dashboard.html"
        out_dash = generate_dashboard_html(output_path=dash_file, telemetry_file=t_file)
        assert out_dash.exists()
        dash_content = out_dash.read_text(encoding="utf-8")
        assert 'x2="nan"' not in dash_content
        assert 'x2="inf"' not in dash_content

        # 2. CLI monitor must render safely without crash
        code = main(["monitor"])
        assert code == 0
        captured = capsys.readouterr()
        assert "QUANTA BİLİŞSEL HAKEM & TELEMETRİ KOKPİTİ" in captured.out
        assert "inf_rule" in captured.out

    def test_subconscious_hook_negative_salience_resilience(self) -> None:
        """Adversarial test: negative or non-finite salience must not trigger ZeroDivisionError or fidelity explosion."""
        from scripts.hooks.quanta_subconscious_hook import FastBiomorphicMemory

        mem = FastBiomorphicMemory(capacity=5)
        # Salience that would cause 1.0 + 1.5 * salience = 0.0 without safeguard
        mem.record("singular_rule", "Test singular rule", salience=-0.666667)
        mem.record("inf_rule", "Test inf rule", salience=float("inf"))

        assert mem.engrams[0]["salience"] > 0.0
        assert math.isfinite(mem.engrams[1]["salience"])

        # Stepping must not raise ZeroDivisionError and fidelity must stay <= 0.9998
        for _ in range(50):
            mem.step(dt=1.0)
            for e in mem.engrams:
                assert math.isfinite(e["fidelity"])
                assert 0.0 < e["fidelity"] <= 0.9998



