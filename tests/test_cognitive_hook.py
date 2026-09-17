"""Unit tests for Quanta Autonomous Subconscious Hook (quanta_subconscious_hook.py)."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HOOK_DIR = Path("/Users/aes/Antigravity Projects/Alfa/quanta/scripts/hooks")
HOOK_PATH = HOOK_DIR / "quanta_subconscious_hook.py"


class TestQuantaSubconsciousHook:
    def test_hook_execution_and_ephemeral_injection(self) -> None:
        assert HOOK_PATH.exists()

        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = {
                "conversationId": "test_conv_auto",
                "artifactDirectoryPath": tmp_dir,
                "workspacePaths": ["/Users/aes/Antigravity Projects/Alfa/quanta"],
            }
            raw_input = json.dumps(payload).encode("utf-8")

            # First invocation
            res = subprocess.run(
                [sys.executable, str(HOOK_PATH)],
                input=raw_input,
                capture_output=True,
                check=True,
            )
            data = json.loads(res.stdout.decode("utf-8"))

            assert "injectSteps" in data
            assert len(data["injectSteps"]) == 1
            ephemeral = data["injectSteps"][0]["ephemeralMessage"]
            assert "Quanta Bilişsel Çıpa" in ephemeral
            assert "executive_summary_rule" in ephemeral
            assert "scientific_integrity_rule" in ephemeral
            assert "%100.0" not in ephemeral  # Must never output flat 100.0%

            # Check that state file was created
            state_file = Path(tmp_dir) / "quanta_cognitive_state.json"
            assert state_file.exists()

            with open(state_file) as sf:
                state_data = json.load(sf)
                assert state_data["turn_count"] == 1
                assert len(state_data["engrams"]) >= 2

            # Immediate re-invocation within 2 seconds should be debounced
            res_debounced = subprocess.run(
                [sys.executable, str(HOOK_PATH)],
                input=raw_input,
                capture_output=True,
                check=True,
            )
            assert res_debounced.stdout.strip() == b"{}"

    def test_hook_empty_input_failsafe(self) -> None:
        res = subprocess.run(
            [sys.executable, str(HOOK_PATH)],
            input=b"",
            capture_output=True,
            check=True,
        )
        assert res.stdout.strip() == b"{}"
