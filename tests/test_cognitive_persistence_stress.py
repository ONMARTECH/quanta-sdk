"""Adversarial Stress Test Suite for Quanta Cognitive Feedback Persistence.

Covers:
1. Concurrency: Multithreaded and multiprocess concurrent writes to state file.
2. Schema integrity: Truncated JSON, empty files, malformed engrams, type mismatches.
3. Permission errors: Read-only directories, read-only files, fallback to /tmp.
4. Atomic replacement: Interrupted writes, disk full simulation, zero orphaned .tmp files.
5. Engram preservation: Exact non-target engram preservation across 53 real production engrams.
"""

import contextlib
import copy
import json
import shutil
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from quanta.cognitive.feedback import (
    ActionOutcome,
    CognitiveFeedbackLoop,
    OutcomeType,
    RuleDriftTracker,
    _save_state_atomically,
)


@pytest.fixture
def real_state_path() -> Path:
    p = Path(__file__).parent.parent / "quanta_cognitive_state.json"
    assert p.exists(), f"quanta_cognitive_state.json not found at {p}"
    return p


@pytest.fixture
def real_state_dict(real_state_path: Path) -> dict[str, Any]:
    with open(real_state_path, encoding="utf-8") as f:
        return json.load(f)


class TestConcurrentUpdatesStress:
    """Stress-tests concurrent access from threads and separate processes."""

    def test_multithreaded_concurrent_feedback_updates(
        self, real_state_path: Path, tmp_path: Path
    ) -> None:
        state_file = tmp_path / "quanta_cognitive_state.json"
        shutil.copy(real_state_path, state_file)

        num_threads = 16
        iters_per_thread = 10

        def thread_task(worker_id: int) -> bool:
            loop = CognitiveFeedbackLoop(state_path=state_file)
            for i in range(iters_per_thread):
                ot = OutcomeType.SUCCESS if (i + worker_id) % 2 == 0 else OutcomeType.TOOL_FAILURE
                outcome = ActionOutcome(
                    step_idx=worker_id * 100 + i,
                    tool_name="run_command",
                    outcome_type=ot,
                    rule_violated="native_first_rule" if ot != OutcomeType.SUCCESS else None,
                )
                loop.process_outcomes([outcome])
            return True

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(thread_task, range(num_threads)))

        assert all(results)

        # Verify state integrity
        with open(state_file, encoding="utf-8") as f:
            saved = json.load(f)

        assert isinstance(saved, dict)
        assert "engrams" in saved
        assert len(saved["engrams"]) >= 53
        assert "outcome_history" in saved or "feedback_history" in saved

        # Verify ZERO orphaned temp files
        orphaned = list(tmp_path.glob(".tmp_*"))
        assert len(orphaned) == 0, f"Found orphaned temp files: {orphaned}"

    def test_multiprocess_concurrent_subprocess_updates(
        self, real_state_path: Path, tmp_path: Path
    ) -> None:
        state_file = tmp_path / "quanta_cognitive_state.json"
        shutil.copy(real_state_path, state_file)

        # Worker script executed as separate OS processes
        worker_code = """
import sys
from pathlib import Path
from quanta.cognitive.feedback import CognitiveFeedbackLoop, ActionOutcome, OutcomeType

state_file = Path(sys.argv[1])
worker_id = int(sys.argv[2])
iters = int(sys.argv[3])

loop = CognitiveFeedbackLoop(state_path=state_file)
for i in range(iters):
    ot = OutcomeType.SUCCESS if (i + worker_id) % 2 == 0 else OutcomeType.TOOL_FAILURE
    outcome = ActionOutcome(
        step_idx=worker_id * 100 + i,
        tool_name="run_command",
        outcome_type=ot,
        rule_violated="native_first_rule" if ot != OutcomeType.SUCCESS else None,
    )
    loop.process_outcomes([outcome])
sys.exit(0)
"""
        worker_script = tmp_path / "proc_worker.py"
        worker_script.write_text(worker_code, encoding="utf-8")

        num_procs = 6
        iters_per_proc = 5
        procs = []

        python_bin = sys.executable
        for wid in range(num_procs):
            p = subprocess.Popen(
                [python_bin, str(worker_script), str(state_file), str(wid), str(iters_per_proc)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            procs.append(p)

        for p in procs:
            stdout, stderr = p.communicate(timeout=30)
            assert p.returncode == 0, f"Process failed with code {p.returncode}: {stderr.decode()}"

        # Check final state file is valid and uncorrupted
        with open(state_file, encoding="utf-8") as f:
            saved = json.load(f)

        assert isinstance(saved, dict)
        assert len(saved.get("engrams", [])) >= 53

        orphaned = list(tmp_path.glob(".tmp_*"))
        assert len(orphaned) == 0, f"Orphaned files after multiprocess: {orphaned}"

    def test_high_frequency_burst_writes(self, tmp_path: Path) -> None:
        state_file = tmp_path / "burst_state.json"
        loop = CognitiveFeedbackLoop(state_path=state_file)

        # Rapid 100 sequential burst updates
        for step in range(100):
            ot = OutcomeType.SUCCESS if step % 3 != 0 else OutcomeType.TOOL_FAILURE
            outcome = ActionOutcome(
                step_idx=step,
                tool_name="run_command",
                outcome_type=ot,
                rule_violated="native_first_rule" if ot != OutcomeType.SUCCESS else None,
            )
            res = loop.process_outcomes([outcome])
            assert res["processed_count"] == 1

        with open(state_file, encoding="utf-8") as f:
            saved = json.load(f)

        assert saved["turn_count"] >= 0
        orphaned = list(tmp_path.glob(".tmp_*"))
        assert len(orphaned) == 0


class TestCorruptedStateResilience:
    """Stress-tests resilience against corrupted, malformed, or missing state data."""

    def test_truncated_json_file_recovery(self, tmp_path: Path) -> None:
        state_file = tmp_path / "corrupted_truncated.json"
        state_file.write_text("{\"turn_count\": 1073, \"engrams\": [ {\"key\": \"rule1\"", encoding="utf-8")

        loop = CognitiveFeedbackLoop(state_path=state_file)
        outcome = ActionOutcome(
            step_idx=1,
            tool_name="run_command",
            outcome_type=OutcomeType.SUCCESS,
            rule_violated="native_first_rule",
        )
        res = loop.process_outcomes([outcome])

        assert res["processed_count"] == 1
        assert state_file.exists()

        # The state file must now be valid JSON
        with open(state_file, encoding="utf-8") as f:
            saved = json.load(f)

        assert isinstance(saved, dict)
        assert "engrams" in saved
        assert any(e.get("key") == "native_first_rule" for e in saved["engrams"])

    def test_empty_zero_byte_file_recovery(self, tmp_path: Path) -> None:
        state_file = tmp_path / "zero_bytes.json"
        state_file.write_text("", encoding="utf-8")

        loop = CognitiveFeedbackLoop(state_path=state_file)
        outcome = ActionOutcome(
            step_idx=1,
            tool_name="run_command",
            outcome_type=OutcomeType.SUCCESS,
            rule_violated="native_first_rule",
        )
        res = loop.process_outcomes([outcome])
        assert res["processed_count"] == 1

        with open(state_file, encoding="utf-8") as f:
            saved = json.load(f)

        assert isinstance(saved, dict)
        assert "turn_count" in saved
        assert isinstance(saved["engrams"], list)

    def test_invalid_json_types_top_level_array(self, tmp_path: Path) -> None:
        state_file = tmp_path / "array_state.json"
        state_file.write_text(json.dumps([1, 2, 3, {"unexpected": True}]), encoding="utf-8")

        loop = CognitiveFeedbackLoop(state_path=state_file)
        outcome = ActionOutcome(
            step_idx=1,
            tool_name="run_command",
            outcome_type=OutcomeType.SUCCESS,
            rule_violated="native_first_rule",
        )
        res = loop.process_outcomes([outcome])
        assert res["processed_count"] == 1

        with open(state_file, encoding="utf-8") as f:
            saved = json.load(f)
        assert isinstance(saved, dict)
        assert "engrams" in saved

    def test_malformed_engrams_key_not_list(self, tmp_path: Path) -> None:
        state_file = tmp_path / "bad_engrams.json"
        state_file.write_text(
            json.dumps({"turn_count": 10, "engrams": "NOT_A_LIST_CORRUPTED"}), encoding="utf-8"
        )

        loop = CognitiveFeedbackLoop(state_path=state_file)
        outcome = ActionOutcome(
            step_idx=1,
            tool_name="run_command",
            outcome_type=OutcomeType.SUCCESS,
            rule_violated="native_first_rule",
        )
        # Should gracefully reset or handle engrams without raising TypeError
        res = loop.process_outcomes([outcome])
        assert res["processed_count"] == 1

        with open(state_file, encoding="utf-8") as f:
            saved = json.load(f)
        assert isinstance(saved["engrams"], list)

    def test_engrams_list_containing_non_dict_elements(self, tmp_path: Path) -> None:
        state_file = tmp_path / "bad_elements.json"
        state_file.write_text(
            json.dumps({"turn_count": 10, "engrams": [None, "string_item", 42, {"key": "valid_rule"}]}),
            encoding="utf-8",
        )

        loop = CognitiveFeedbackLoop(state_path=state_file)
        outcome = ActionOutcome(
            step_idx=1,
            tool_name="run_command",
            outcome_type=OutcomeType.SUCCESS,
            rule_violated="valid_rule",
        )
        res = loop.process_outcomes([outcome])
        assert res["processed_count"] == 1

        with open(state_file, encoding="utf-8") as f:
            saved = json.load(f)
        assert isinstance(saved["engrams"], list)


class TestUnwritableAndPermissionErrors:
    """Stress-tests filesystem permissions and graceful /tmp fallback."""

    def test_readonly_state_directory_fallback_to_tmp(self, tmp_path: Path) -> None:
        ro_dir = tmp_path / "readonly_dir"
        ro_dir.mkdir()
        state_file = ro_dir / "quanta_cognitive_state.json"
        state_file.write_text(json.dumps({"turn_count": 1, "engrams": []}), encoding="utf-8")

        # Make directory read-only
        ro_dir.chmod(stat.S_IREAD | stat.S_IEXEC)
        try:
            loop = CognitiveFeedbackLoop(state_path=state_file)
            outcome = ActionOutcome(
                step_idx=1,
                tool_name="run_command",
                outcome_type=OutcomeType.SUCCESS,
                rule_violated="native_first_rule",
            )
            # Must not raise PermissionError, falls back to /tmp
            res = loop.process_outcomes([outcome])
            assert res["processed_count"] == 1

            # Check fallback in /tmp
            fallback_path = Path("/tmp/quanta_cognitive_state.json")
            assert fallback_path.exists()
            with open(fallback_path, encoding="utf-8") as f:
                saved = json.load(f)
            assert any(e.get("key") == "native_first_rule" for e in saved.get("engrams", []))
        finally:
            ro_dir.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)

    def test_unwritable_fallback_failsafe(self, tmp_path: Path) -> None:
        """When both target and fallback fail, save returns False without crashing."""
        state_file = tmp_path / "unwritable.json"
        state_dict = {"turn_count": 1, "engrams": []}

        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            success = _save_state_atomically(state_file, state_dict, safe_conv_id="test_failsafe")
            assert success is False

            loop = CognitiveFeedbackLoop(state_path=state_file)
            outcome = ActionOutcome(
                step_idx=1,
                tool_name="run_command",
                outcome_type=OutcomeType.SUCCESS,
                rule_violated="native_first_rule",
            )
            # process_outcomes must not raise
            res = loop.process_outcomes([outcome])
            assert res["processed_count"] == 1


class TestInterruptedAtomicReplacementAndNoOrphanedTmp:
    """Stress-tests atomic replacement interruptions and verifies zero orphaned temp files."""

    def test_simulated_crash_during_os_replace_cleans_tmp(self, tmp_path: Path) -> None:
        state_file = tmp_path / "target_state.json"
        state_file.write_text(json.dumps({"turn_count": 100, "engrams": [{"key": "original"}]}), encoding="utf-8")

        state_dict = {"turn_count": 101, "engrams": [{"key": "new_state"}]}

        # Simulate sudden crash / disk error during os.replace
        with patch("os.replace", side_effect=OSError("Simulated ENOSPC Disk Full")):
            _save_state_atomically(state_file, state_dict, safe_conv_id="crash_test")
            # Should fail to save to primary, try fallback or fail cleanly
            orphaned = list(tmp_path.glob(".tmp_*"))
            assert len(orphaned) == 0, f"Found orphaned temp files after crash: {orphaned}"

        # Original state file must remain untouched and uncorrupted
        with open(state_file, encoding="utf-8") as f:
            original = json.load(f)
        assert original["turn_count"] == 100
        assert original["engrams"][0]["key"] == "original"

    def test_simulated_json_dump_serialization_error_cleans_tmp(self, tmp_path: Path) -> None:
        state_file = tmp_path / "serialization_err.json"
        state_file.write_text(json.dumps({"turn_count": 50, "engrams": []}), encoding="utf-8")

        state_dict = {"turn_count": 51, "engrams": [], "bad_object": object()}

        success = _save_state_atomically(state_file, state_dict, safe_conv_id="serial_err")
        assert success is False

        orphaned = list(tmp_path.glob(".tmp_*"))
        assert len(orphaned) == 0, f"Found orphaned temp files: {orphaned}"

    def test_adversarial_intermittent_errors_zero_orphaned_files(self, tmp_path: Path) -> None:
        state_file = tmp_path / "intermittent.json"
        state_file.write_text(json.dumps({"turn_count": 0, "engrams": []}), encoding="utf-8")

        loop = CognitiveFeedbackLoop(state_path=state_file)
        call_count = 0

        original_open = open

        def flaky_open(file, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0 and ".tmp_" in str(file):
                raise OSError("Simulated intermittent disk I/O error")
            return original_open(file, *args, **kwargs)

        with patch("builtins.open", side_effect=flaky_open):
            for i in range(30):
                outcome = ActionOutcome(
                    step_idx=i,
                    tool_name="run_command",
                    outcome_type=OutcomeType.SUCCESS if i % 2 == 0 else OutcomeType.TOOL_FAILURE,
                    rule_violated="native_first_rule",
                )
                with contextlib.suppress(Exception):
                    loop.process_outcomes([outcome])

        orphaned = list(tmp_path.glob(".tmp_*"))
        assert len(orphaned) == 0, f"Found orphaned temp files after intermittent errors: {orphaned}"


class TestExistingEngramPreservationAdversarial:
    """Stress-tests preservation of all 53 real production engrams bit-for-bit."""

    def test_53_existing_engrams_bit_for_bit_preservation(
        self, real_state_path: Path, real_state_dict: dict[str, Any], tmp_path: Path
    ) -> None:
        state_file = tmp_path / "quanta_cognitive_state.json"
        shutil.copy(real_state_path, state_file)

        initial_engrams = copy.deepcopy(real_state_dict.get("engrams", []))
        assert len(initial_engrams) == 53
        initial_keys = {e["key"]: e for e in initial_engrams}

        loop = CognitiveFeedbackLoop(state_path=state_file)

        # Run 50 updates on target rules
        target_rules = ["native_first_rule", "executive_summary_rule", "scientific_integrity_rule"]
        for step in range(1, 51):
            rule = target_rules[step % len(target_rules)]
            ot = (
                OutcomeType.SUCCESS
                if step % 4 != 0
                else (OutcomeType.TOOL_FAILURE if step % 2 == 0 else OutcomeType.RULE_VIOLATION)
            )
            outcome = ActionOutcome(
                step_idx=step,
                tool_name="run_command",
                outcome_type=ot,
                rule_violated=rule if ot != OutcomeType.SUCCESS else None,
                details={"rule_key": rule},
            )
            loop.process_outcomes([outcome])

        with open(state_file, encoding="utf-8") as f:
            final_state = json.load(f)

        final_engrams = final_state.get("engrams", [])
        final_keys = {e["key"]: e for e in final_engrams}

        # Check all 53 original engrams exist
        for orig_key, orig_engram in initial_keys.items():
            assert orig_key in final_keys, f"Engram {orig_key} was lost during feedback updates!"
            final_eng = final_keys[orig_key]

            # Verify non-target fields were not altered
            for field_name in ("content", "description", "category", "tags", "topic"):
                assert orig_engram.get(field_name) == final_eng.get(field_name), (
                    f"Field {field_name} for engram {orig_key} corrupted! "
                    f"Original: {orig_engram.get(field_name)}, Final: {final_eng.get(field_name)}"
                )

            # Salience, fidelity, age should be preserved for untouched insight engrams
            assert orig_engram.get("salience") == final_eng.get("salience")
            assert orig_engram.get("fidelity") == final_eng.get("fidelity")
            assert orig_engram.get("age") == final_eng.get("age")

        # Check that target rules were created/updated
        for tr in target_rules:
            assert tr in final_keys
            tr_eng = final_keys[tr]
            assert "confidence" in tr_eng
            assert 0.05 <= tr_eng["confidence"] <= 0.9998

    def test_repeated_rule_violation_quarantine_preserves_non_target_engrams(
        self, real_state_path: Path, real_state_dict: dict[str, Any], tmp_path: Path
    ) -> None:
        state_file = tmp_path / "quanta_cognitive_state.json"
        shutil.copy(real_state_path, state_file)

        initial_engrams = copy.deepcopy(real_state_dict.get("engrams", []))
        initial_keys = {e["key"] for e in initial_engrams}

        loop = CognitiveFeedbackLoop(state_path=state_file)

        # 10 consecutive severe violations of native_first_rule to trigger quarantine
        for step in range(1, 11):
            outcome = ActionOutcome(
                step_idx=step,
                tool_name="dengage",
                outcome_type=OutcomeType.RULE_VIOLATION,
                rule_violated="native_first_rule",
            )
            loop.process_outcomes([outcome])

        with open(state_file, encoding="utf-8") as f:
            saved = json.load(f)

        saved_keys = {e["key"]: e for e in saved.get("engrams", [])}

        # The violated rule must be quarantined
        native_rule = saved_keys.get("native_first_rule")
        assert native_rule is not None
        assert native_rule.get("is_quarantined") is True
        assert native_rule.get("confidence") < 0.20

        # All 53 original engrams must NOT be quarantined
        for k in initial_keys:
            eng = saved_keys[k]
            assert eng.get("is_quarantined") is not True, f"Non-target engram {k} was erroneously quarantined!"


class TestDefectReproduction:
    """Explicit reproduction tests for identified persistence and schema integrity defects."""

    def test_defect_1_top_level_array_json_attribute_error(self, tmp_path: Path) -> None:
        state_file = tmp_path / "top_level_array.json"
        state_file.write_text(json.dumps([{"corrupt": 1}]), encoding="utf-8")
        loop = CognitiveFeedbackLoop(state_path=state_file)
        outcome = ActionOutcome(step_idx=1, tool_name="cmd", outcome_type=OutcomeType.SUCCESS)
        loop.process_outcomes([outcome])

    def test_defect_2_malformed_engrams_string_attribute_error(self, tmp_path: Path) -> None:
        state_file = tmp_path / "bad_engrams_str.json"
        state_file.write_text(json.dumps({"turn_count": 1, "engrams": "corrupted_string"}), encoding="utf-8")
        loop = CognitiveFeedbackLoop(state_path=state_file)
        outcome = ActionOutcome(step_idx=1, tool_name="cmd", outcome_type=OutcomeType.SUCCESS)
        loop.process_outcomes([outcome])

    def test_defect_3_non_dict_elements_in_engrams_list(self, tmp_path: Path) -> None:
        state_file = tmp_path / "bad_elements.json"
        state_file.write_text(json.dumps({"turn_count": 1, "engrams": [None, 42, "foo"]}), encoding="utf-8")
        loop = CognitiveFeedbackLoop(state_path=state_file)
        outcome = ActionOutcome(step_idx=1, tool_name="cmd", outcome_type=OutcomeType.SUCCESS)
        loop.process_outcomes([outcome])

    def test_defect_4_orphaned_tmp_file_on_keyboard_interrupt(self, tmp_path: Path) -> None:
        state_file = tmp_path / "target.json"
        try:
            with patch("json.dump", side_effect=KeyboardInterrupt("Simulated Interrupt")):
                _save_state_atomically(state_file, {"turn_count": 1}, safe_conv_id="ki_test")
        except KeyboardInterrupt:
            pass
        orphaned = list(tmp_path.glob(".tmp_*"))
        assert len(orphaned) == 0, f"Found orphaned temp files: {orphaned}"

    def test_defect_5_key_error_on_missing_meta_keys(self) -> None:
        tracker = RuleDriftTracker()
        state_dict = {"engrams": []}
        outcome = ActionOutcome(step_idx=1, tool_name="cmd", outcome_type=OutcomeType.SUCCESS)
        # Empty meta must not raise KeyError
        tracker.update_engram_in_state(
            state_dict=state_dict,
            rule_key="rule_x",
            new_confidence=0.96,
            meta={},
            outcome=outcome,
        )
