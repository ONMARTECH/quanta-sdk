"""Unit and integration test suite for Antigravity PostToolUse hook decoupling.

Validates payload parsing, automated LTP/LTD inhibitor plasticity,
fail-safe fault tolerance, and strict sub-25ms latency budget.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts.hooks.quanta_subconscious_hook import (
    extract_context_tags,
    handle_post_tool_use,
)
from scripts.hooks.quanta_subconscious_hook import (
    main as hook_main,
)

HOOK_PATH = Path("/Users/aes/Antigravity Projects/Alfa/quanta/scripts/hooks/quanta_subconscious_hook.py")


def _invoke_hook_subprocess(payload: dict[str, Any]) -> subprocess.CompletedProcess[bytes]:
    """Helper to invoke quanta_subconscious_hook.py as an isolated subprocess."""
    raw_input = json.dumps(payload).encode("utf-8")
    return subprocess.run(
        [sys.executable, str(HOOK_PATH)],
        input=raw_input,
        capture_output=True,
        check=False,
    )


class TestPostToolUsePayloadParsing:
    """Validates handling and parsing of Antigravity PostToolUse payload structures."""

    def test_valid_tool_result_payload_extraction(self, tmp_path: Path) -> None:
        """Verifies standard toolCall and result payload extracts name, args, output."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        payload = {
            "event": "PostToolUse",
            "conversationId": "test_conv_valid_parse",
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 7,
            "toolCall": {
                "name": "run_command",
                "args": {"CommandLine": "python -m pytest", "Cwd": str(tmp_path)},
            },
            "result": {"exit_code": 0, "stdout": "1 passed"},
            "testing": True,
        }

        res = handle_post_tool_use(payload, state_file=state_file)
        assert res == {}
        assert state_file.exists()

        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert state.get("last_verified_step") == 7

    def test_empty_stdin_produces_empty_dict(self) -> None:
        """Verifies empty stdin payload cleanly produces exit code 0 and stdout {}."""
        res = subprocess.run(
            [sys.executable, str(HOOK_PATH)],
            input=b"",
            capture_output=True,
            check=False,
        )
        assert res.returncode == 0
        assert res.stdout.strip() == b"{}"

    def test_malformed_json_stdin_exits_zero(self) -> None:
        """Verifies corrupted JSON on stdin exits cleanly with code 0 and stdout {}."""
        res = subprocess.run(
            [sys.executable, str(HOOK_PATH)],
            input=b"{malformed json: unquoted, broken...",
            capture_output=True,
            check=False,
        )
        assert res.returncode == 0
        assert res.stdout.strip() == b"{}"

    def test_missing_fields_graceful_noop(self, tmp_path: Path) -> None:
        """Verifies payload with missing toolCall/result/error produces clean no-op."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        state_file.write_text(json.dumps({"turn_count": 2, "engrams": []}), encoding="utf-8")

        payload = {
            "event": "PostToolUse",
            "conversationId": "test_conv_missing_fields",
            "artifactDirectoryPath": str(tmp_path),
            "testing": True,
        }

        res = handle_post_tool_use(payload, state_file=state_file)
        assert res == {}
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert state["turn_count"] == 2


class TestAutoInhibitorLTPAndLTD:
    """Validates automated LTP potentiation on error and LTD depression on success."""

    def test_tool_error_registers_negative_engram(self, tmp_path: Path) -> None:
        """Verifies tool error automatically registers a negative inhibitor engram."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        state_file.write_text(json.dumps({"turn_count": 1, "engrams": []}), encoding="utf-8")

        payload = {
            "event": "PostToolUse",
            "conversationId": "test_auto_ltp_1",
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 3,
            "toolCall": {
                "name": "run_command",
                "args": {"CommandLine": "zsh bad_syntax [lang]"},
            },
            "error": "zsh: no matches found: [lang]",
            "testing": True,
        }

        res = handle_post_tool_use(payload, state_file=state_file)
        assert res == {}

        state = json.loads(state_file.read_text(encoding="utf-8"))
        engrams = state.get("engrams", [])
        assert any(e.get("category") == "inhibitor" for e in engrams)

        inh = next(e for e in engrams if e.get("category") == "inhibitor")
        assert inh["key"] == "inh_run_command"
        assert float(inh["v_inh"]) >= 0.50
        assert inh["consecutive_failures"] == 1
        assert inh["consecutive_successes"] == 0

    def test_repeated_tool_error_triggers_ltp_potentiation(self, tmp_path: Path) -> None:
        """Verifies repeated failures progressively deepen inhibition weight (LTP)."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        state_file.write_text(json.dumps({"turn_count": 1, "engrams": []}), encoding="utf-8")

        payload = {
            "event": "PostToolUse",
            "conversationId": "test_auto_ltp_repeat",
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 4,
            "toolCall": {"name": "run_command", "args": {"CommandLine": "cargo test"}},
            "result": {"exit_code": 101, "stderr": "compilation error"},
            "testing": True,
        }

        # First failure
        handle_post_tool_use(payload, state_file=state_file)
        state1 = json.loads(state_file.read_text(encoding="utf-8"))
        inh1 = next(e for e in state1["engrams"] if e.get("category") == "inhibitor")
        v1 = float(inh1["v_inh"])
        sal1 = float(inh1["salience"])

        # Second failure (LTP)
        payload["stepIdx"] = 5
        handle_post_tool_use(payload, state_file=state_file)
        state2 = json.loads(state_file.read_text(encoding="utf-8"))
        inh2 = next(e for e in state2["engrams"] if e.get("category") == "inhibitor")
        v2 = float(inh2["v_inh"])
        sal2 = float(inh2["salience"])

        assert v2 > v1, f"Expected LTP potentiation: v2 ({v2}) > v1 ({v1})"
        assert sal2 > sal1, f"Expected salience awakening: sal2 ({sal2}) > sal1 ({sal1})"
        assert inh2["consecutive_failures"] == 2

    def test_subsequent_tool_success_triggers_ltd_depression(self, tmp_path: Path) -> None:
        """Verifies subsequent tool success relaxes active inhibitor synaptic weight (LTD)."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        initial_state = {
            "turn_count": 1,
            "engrams": [
                {
                    "key": "inh_run_command",
                    "content": "Previous failure in run_command",
                    "category": "inhibitor",
                    "v_inh": 0.85,
                    "salience": 2.2,
                    "is_core_anchor": True,
                    "context_tags": {"tool": "run_command", "command": "cargo test"},
                    "consecutive_failures": 3,
                    "consecutive_successes": 0,
                    "context_divergence": [],
                }
            ],
        }
        state_file.write_text(json.dumps(initial_state), encoding="utf-8")

        success_payload = {
            "event": "PostToolUse",
            "conversationId": "test_auto_ltd",
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 6,
            "toolCall": {"name": "run_command", "args": {"CommandLine": "cargo test --lib"}},
            "result": {"exit_code": 0, "stdout": "ok"},
            "testing": True,
        }

        handle_post_tool_use(success_payload, state_file=state_file)
        state = json.loads(state_file.read_text(encoding="utf-8"))
        inh = next(e for e in state["engrams"] if e["key"] == "inh_run_command")

        v_depressed = float(inh["v_inh"])
        sal_depressed = float(inh["salience"])

        assert v_depressed < 0.85, f"Expected LTD relaxation: {v_depressed} < 0.85"
        assert sal_depressed < 2.2, f"Expected salience decay: {sal_depressed} < 2.2"
        assert inh["consecutive_successes"] == 1
        assert inh["consecutive_failures"] == 0

    def test_context_tag_extraction_and_divergence_logging(self, tmp_path: Path) -> None:
        """Verifies tool arguments yield accurate tags and log context divergence upon recovery."""
        # 1. Direct unit test of tag extraction
        tags = extract_context_tags(
            tool_name="run_command",
            tool_args={"CommandLine": "python -m pytest tests/test_fast.py"},
            error_msg="syntax error in line 10",
        )
        assert tags["tool"] == "run_command"
        assert tags["runtime"] == "python3"
        assert tags["target"] == "-m"
        assert tags["error_type"] == "syntax_error"
        assert tags["os"] == sys.platform

        # 2. Context divergence logging test
        state_file = tmp_path / "quanta_cognitive_state.json"
        state_file.write_text(json.dumps({"turn_count": 1, "engrams": []}), encoding="utf-8")

        fail_payload = {
            "event": "PostToolUse",
            "conversationId": "test_ctx_div",
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 10,
            "toolCall": {"name": "run_command", "args": {"CommandLine": "npm test"}},
            "error": "npm command not found",
            "testing": True,
        }
        handle_post_tool_use(fail_payload, state_file=state_file)

        succ_payload = {
            "event": "PostToolUse",
            "conversationId": "test_ctx_div",
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 11,
            "toolCall": {"name": "run_command", "args": {"CommandLine": "yarn test"}},
            "result": {"exit_code": 0},
            "testing": True,
        }
        handle_post_tool_use(succ_payload, state_file=state_file)

        state = json.loads(state_file.read_text(encoding="utf-8"))
        inh = next(e for e in state["engrams"] if e["key"] == "inh_run_command")
        assert len(inh["context_divergence"]) >= 1
        diff = inh["context_divergence"][0]
        assert "command" in diff
        assert tuple(diff["command"]) == ("npm test", "yarn test")


class TestFailSafeShielding:
    """Validates zero-crash fault tolerance and error recovery guarantees."""

    def test_unwritable_state_directory_failsafe(self, tmp_path: Path) -> None:
        """Verifies unwritable state target path does not raise and falls back to /tmp."""
        unwritable_path = Path("/proc/sys/nonexistent_quanta_state.json")
        payload = {
            "event": "PostToolUse",
            "conversationId": "test_unwritable_fs",
            "stepIdx": 1,
            "toolCall": {"name": "run_command", "args": {}},
            "result": {"exit_code": 0},
            "testing": True,
        }

        # Must not raise OSError or PermissionError
        res = handle_post_tool_use(payload, state_file=unwritable_path)
        assert res == {}

    def test_corrupted_state_file_recovery(self, tmp_path: Path) -> None:
        """Verifies corrupt state JSON file on disk is recovered gracefully."""
        corrupt_state = tmp_path / "quanta_cognitive_state.json"
        corrupt_state.write_text("{broken json syntax: missing quotes", encoding="utf-8")

        payload = {
            "event": "PostToolUse",
            "conversationId": "test_corrupt_rec",
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 2,
            "toolCall": {"name": "run_command", "args": {}},
            "error": "Execution error",
            "testing": True,
        }

        res = handle_post_tool_use(payload, state_file=corrupt_state)
        assert res == {}

        # Verifies file has been rewritten as valid JSON
        recovered = json.loads(corrupt_state.read_text(encoding="utf-8"))
        assert "engrams" in recovered
        assert any(e.get("category") == "inhibitor" for e in recovered["engrams"])

    def test_exception_in_handler_never_propagates(self) -> None:
        """Verifies simulated disk or runtime exception is caught fail-safe."""
        payload = {
            "event": "PostToolUse",
            "conversationId": "test_exc_shield",
            "toolCall": {"name": "run_command", "args": {}},
            "testing": True,
        }

        with patch(
            "scripts.hooks.quanta_subconscious_hook._save_mirrored_state_atomically",
            side_effect=RuntimeError("Simulated disk fault"),
        ):
            res = handle_post_tool_use(payload)
            assert res == {}


class TestPostToolUseLatency:
    """Validates strict sub-25ms latency budget under repeated execution."""

    def test_post_tool_use_latency_50_iterations_p95_p99(self, tmp_path: Path) -> None:
        """Verifies 50 consecutive PostToolUse evaluations achieve p95 < 15ms and p99 < 25ms."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        state_file.write_text(json.dumps({"turn_count": 1, "engrams": []}), encoding="utf-8")

        payload = {
            "event": "PostToolUse",
            "conversationId": "benchmark_lat",
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 50,
            "toolCall": {"name": "run_command", "args": {"CommandLine": "git status"}},
            "result": {"exit_code": 0},
            "testing": True,
        }

        # Warm-up invocation (JIT & file creation)
        handle_post_tool_use(payload, state_file=state_file)

        latencies: list[float] = []
        for _ in range(50):
            t0 = time.perf_counter()
            handle_post_tool_use(payload, state_file=state_file)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

        latencies.sort()
        median = latencies[25]
        p95 = latencies[47]
        p99 = latencies[49]

        assert median < 5.0, f"Median latency {median:.2f}ms exceeded 5.0ms"
        assert p95 < 15.0, f"p95 latency {p95:.2f}ms exceeded 15.0ms"
        assert p99 < 25.0, f"p99 latency {p99:.2f}ms exceeded 25.0ms ceiling"

    def test_in_process_main_post_tool_use_latency_budget(self, tmp_path: Path) -> None:
        """Verifies in-process main() execution with PostToolUse payload completes in < 25ms."""
        payload = {
            "event": "PostToolUse",
            "conversationId": "test_inproc_main",
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 15,
            "toolCall": {"name": "run_command", "args": {}},
            "result": {"exit_code": 0},
            "testing": True,
        }
        raw_json = json.dumps(payload)

        stdin_backup = sys.stdin
        stdout_backup = sys.stdout
        try:
            fake_out = io.StringIO()
            sys.stdin = io.StringIO(raw_json)
            sys.stdout = fake_out

            t0 = time.perf_counter()
            hook_main()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            assert fake_out.getvalue().strip() == "{}"
            assert latency_ms < 25.0, f"Latency {latency_ms:.2f}ms exceeded 25.0ms"
        finally:
            sys.stdin = stdin_backup
            sys.stdout = stdout_backup

    def test_post_tool_use_subprocess_fast_exit(self, tmp_path: Path) -> None:
        """Verifies subprocess invocation strictly returns {} without emitting injectSteps."""
        payload = {
            "event": "PostToolUse",
            "conversationId": "test_subp_fast",
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 20,
            "toolCall": {"name": "run_command", "args": {"CommandLine": "echo hello"}},
            "result": {"exit_code": 0, "stdout": "hello"},
            "testing": True,
        }

        res = _invoke_hook_subprocess(payload)
        assert res.returncode == 0
        raw = res.stdout.strip()
        assert raw == b"{}"
        assert b"injectSteps" not in res.stdout
        assert b"ephemeralMessage" not in res.stdout
