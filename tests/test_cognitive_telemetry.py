"""Unit and Integration tests for Quanta Cognitive Telemetry & Monitoring."""

import io
import json
from pathlib import Path
import pytest

from quanta.cognitive.telemetry import (
    record_decision_telemetry,
    record_hook_telemetry,
    read_telemetry_events,
    get_telemetry_summary,
    generate_dashboard_html,
    detect_workspace,
)
from quanta.cli import main


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
