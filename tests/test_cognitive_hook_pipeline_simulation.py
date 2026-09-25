"""Empirical Hook Pipeline & Turn Invariant Simulation Test Suite.

Authored by empirical challenger to stress-test:
1. Realistic 25-step multi-tool task (interleaving handle_post_tool_use and PreInvocation).
2. Invariant: turn_count remains 1 across all 25 sub-turns and only increments on new user query.
3. Invariant: transient decisions recorded at step 1 maintain F >= 0.885 and are STILL active
   and injected in SWR replay at step 20 and step 25.
4. Robustness: Re-entrancy, debouncing, rapid concurrent atomic state saves, and subprocess isolation.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts.hooks.quanta_subconscious_hook import (
    FastBiomorphicMemory,
    _save_mirrored_state_atomically,
    handle_post_tool_use,
)
from scripts.hooks.quanta_subconscious_hook import (
    main as hook_main,
)

HOOK_PATH = Path("/Users/aes/Antigravity Projects/Alfa/quanta/scripts/hooks/quanta_subconscious_hook.py")


def _run_inprocess_hook(payload: dict[str, Any], sim_time: float | None = None) -> dict[str, Any]:
    """Helper to invoke hook_main() in-process with mocked stdin, stdout, and time."""
    raw_json = json.dumps(payload)
    stdin_backup = sys.stdin
    stdout_backup = sys.stdout
    fake_out = io.StringIO()
    try:
        sys.stdin = io.StringIO(raw_json)
        sys.stdout = fake_out
        if sim_time is not None:
            with patch("time.time", return_value=sim_time):
                hook_main()
        else:
            hook_main()
        out_str = fake_out.getvalue().strip()
        if not out_str:
            return {}
        res: dict[str, Any] = json.loads(out_str)
        return res
    finally:
        sys.stdin = stdin_backup
        sys.stdout = stdout_backup


def _invoke_hook_subprocess(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Helper to invoke hook via subprocess over stdin/stdout."""
    raw_input = json.dumps(payload).encode("utf-8")
    res = subprocess.run(
        [sys.executable, str(HOOK_PATH)],
        input=raw_input,
        capture_output=True,
        check=False,
    )
    out_str = res.stdout.decode("utf-8", errors="replace").strip()
    data: dict[str, Any] = json.loads(out_str) if out_str else {}
    return res.returncode, data


class TestHookPipeline25StepSimulation:
    """Simulates a realistic 25-step multi-tool task and verifies invariants."""

    def test_25_step_multitool_task_end_to_end_simulation(self, tmp_path: Path) -> None:
        """Empirical Challenge 1:
        Simulate a realistic 25-step multi-tool task:
        1. Step 0: User query arrives -> PreInvocation sets turn_count = 1.
        2. Step 1: Transient decision recorded with salience=0.5. Tool 1 executes.
        3. Steps 1-25: Interleaved PostToolUse (dt=0.2) and PreInvocation continuation sub-turns.
        4. Verify turn_count == 1 throughout all 25 sub-turns.
        5. Verify transient decision is STILL active and injected in SWR Replay at step 20 (F >= 0.885).
        6. Step 26: New user query arrives -> PreInvocation increments turn_count to 2.
        """
        state_file = tmp_path / "quanta_cognitive_state.json"
        conv_id = "test_conv_empirical_25_steps"
        base_time = 1700000000.0

        # --- STEP 0: Initial User Query ---
        query_1 = "Execute 25-step cognitive migration pipeline"
        payload_turn1 = {
            "event": "PreInvocation",
            "conversationId": conv_id,
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 0,
            "userPrompt": query_1,
            "workspacePaths": [str(tmp_path)],
            "testing": True,
        }
        res_step0 = _run_inprocess_hook(payload_turn1, sim_time=base_time)
        assert state_file.exists()
        assert "injectSteps" in res_step0
        msg_step0 = res_step0["injectSteps"][0]["ephemeralMessage"]
        assert "Quanta Bilişsel Çıpa" in msg_step0

        # State check after Step 0
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert state["turn_count"] == 1
        assert state["last_user_query"] == query_1.lower()
        assert state.get("pending_tool_continuation", False) is False

        # --- STEP 1: Ingest transient decision into state ---
        # Simulate agent deciding on architectural strategy
        mem = FastBiomorphicMemory(capacity=128)
        for e in state.get("engrams", []):
            mem.engrams.append(e)
        mem.record_transient(
            key="decision_step_1",
            content="Use streaming chunking strategy for pipeline migration",
            salience=0.5,
            category="contextual_decision",
        )
        state["engrams"] = mem.engrams
        state["last_step_idx"] = 0
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

        # Track history across the 25 steps
        turn_counts: list[int] = []
        decision_fidelities: list[float] = []
        swr_injections: list[bool] = []

        # --- STEPS 1 to 25: Interleaved PostToolUse and PreInvocation Continuation ---
        for step in range(1, 26):
            step_time = base_time + (step * 5.0)  # Advance 5s per step (exceeds 2.0s debounce)

            # A. Tool Execution -> PostToolUse Hook
            post_payload = {
                "event": "PostToolUse",
                "conversationId": conv_id,
                "artifactDirectoryPath": str(tmp_path),
                "stepIdx": step,
                "toolCall": {
                    "name": "run_command",
                    "args": {"CommandLine": f"process_chunk_{step}.sh"},
                },
                "result": {"exit_code": 0, "stdout": f"chunk {step} done"},
                "testing": True,
            }
            post_res = handle_post_tool_use(post_payload, state_file=state_file)
            assert post_res == {}

            # Verify post-tool state: dt=0.2 advanced, turn_count unchanged, pending set
            post_state = json.loads(state_file.read_text(encoding="utf-8"))
            assert post_state["turn_count"] == 1, (
                f"PostToolUse illegally incremented turn_count to {post_state['turn_count']} at step {step}"
            )
            assert post_state["pending_tool_continuation"] is True

            # B. PreInvocation Continuation Sub-Turn
            continuation_payload = {
                "event": "PreInvocation",
                "conversationId": conv_id,
                "artifactDirectoryPath": str(tmp_path),
                "stepIdx": step,
                "userPrompt": query_1,  # Same user query active during continuation
                "workspacePaths": [str(tmp_path)],
                "testing": True,
            }
            pre_res = _run_inprocess_hook(continuation_payload, sim_time=step_time)
            assert "injectSteps" in pre_res, f"PreInvocation failed to emit injectSteps at step {step}"

            ephemeral_msg = pre_res["injectSteps"][0]["ephemeralMessage"]
            has_decision_injected = "decision_step_1" in ephemeral_msg
            swr_injections.append(has_decision_injected)

            # Reload state after PreInvocation subturn
            cur_state = json.loads(state_file.read_text(encoding="utf-8"))
            turn_counts.append(cur_state["turn_count"])
            assert cur_state["turn_count"] == 1, (
                f"PreInvocation sub-turn illegally incremented turn_count to {cur_state['turn_count']} at step {step}"
            )
            assert cur_state["pending_tool_continuation"] is False, (
                f"PreInvocation failed to reset pending_tool_continuation at step {step}"
            )

            # Check transient decision retention
            d_eng = next((e for e in cur_state["engrams"] if e["key"] == "decision_step_1"), None)
            assert d_eng is not None, f"decision_step_1 was prematurely evicted at step {step}!"
            decision_fidelities.append(float(d_eng["fidelity"]))

        # --- VERIFICATION AT STEP 20 ---
        # At step 20 (index 19): 20 tool steps completed (total dt = 20 * 0.2 = 4.0)
        fid_at_step_20 = decision_fidelities[19]
        assert fid_at_step_20 >= 0.885, (
            f"Transient fidelity at step 20 ({fid_at_step_20:.6f}) fell below 0.885 floor!"
        )
        assert swr_injections[19] is True, (
            "Transient decision_step_1 was NOT injected in SWR replay at step 20!"
        )

        # --- VERIFICATION AT STEP 25 ---
        fid_at_step_25 = decision_fidelities[24]
        assert fid_at_step_25 >= 0.885, (
            f"Transient fidelity at step 25 ({fid_at_step_25:.6f}) fell below 0.885 floor!"
        )
        assert swr_injections[24] is True, (
            "Transient decision_step_1 was NOT injected in SWR replay at step 25!"
        )

        # Turn count invariant: MUST BE 1 throughout all 25 subturns
        assert all(tc == 1 for tc in turn_counts), (
            f"turn_count did not remain 1 throughout all sub-turns: {turn_counts}"
        )

        # --- STEP 26: New User Query Arrives ---
        query_2 = "Now verify the completed migration and generate report"
        step26_time = base_time + 150.0
        payload_turn2 = {
            "event": "PreInvocation",
            "conversationId": conv_id,
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 26,
            "userPrompt": query_2,
            "workspacePaths": [str(tmp_path)],
            "testing": True,
        }
        res_step26 = _run_inprocess_hook(payload_turn2, sim_time=step26_time)
        assert "injectSteps" in res_step26

        final_state = json.loads(state_file.read_text(encoding="utf-8"))
        assert final_state["turn_count"] == 2, (
            f"New user query failed to increment turn_count to 2! Got: {final_state['turn_count']}"
        )
        assert final_state["last_user_query"] == query_2.lower()


class TestTransientRetentionBoundaryStress:
    """Stress tests verifying retention boundaries for low-salience transient engrams."""

    def test_low_salience_transient_0_2_retention_over_20_tool_steps(self, tmp_path: Path) -> None:
        """Verifies that even the lowest transient salience (S=0.2) survives 20 tool steps."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        conv_id = "test_conv_s02_boundary"
        base_time = 1700000000.0

        # Initialize state with S=0.2 transient decision
        mem = FastBiomorphicMemory(capacity=16)
        mem.record_transient("ephemeral_choice", "Temporary cache folder /tmp/data", salience=0.2)
        state = {
            "turn_count": 1,
            "engrams": mem.engrams,
            "pending_tool_continuation": False,
            "last_user_query": "batch process",
            "last_injected_time": base_time,
            "last_step_idx": 0,
        }
        state_file.write_text(json.dumps(state), encoding="utf-8")

        for step in range(1, 21):
            s_time = base_time + (step * 3.0)
            handle_post_tool_use(
                {
                    "event": "PostToolUse",
                    "conversationId": conv_id,
                    "artifactDirectoryPath": str(tmp_path),
                    "stepIdx": step,
                    "toolCall": {"name": "run_command", "args": {}},
                    "result": {"exit_code": 0},
                    "testing": True,
                },
                state_file=state_file,
            )
            pre_res = _run_inprocess_hook(
                {
                    "event": "PreInvocation",
                    "conversationId": conv_id,
                    "artifactDirectoryPath": str(tmp_path),
                    "stepIdx": step,
                    "userPrompt": "batch process",
                    "workspacePaths": [str(tmp_path)],
                    "testing": True,
                },
                sim_time=s_time,
            )
            assert "injectSteps" in pre_res

        # Check final state after 20 tool steps
        final_state = json.loads(state_file.read_text(encoding="utf-8"))
        eng = next((e for e in final_state["engrams"] if e["key"] == "ephemeral_choice"), None)
        assert eng is not None, "S=0.2 transient decision was prematurely pruned before 20 steps!"
        assert eng["age"] == 4.0, f"Expected cumulative age 4.0, got {eng['age']}"
        assert eng["fidelity"] >= 0.885, (
            f"Fidelity {eng['fidelity']:.6f} dropped below 0.885 floor after 20 tool steps!"
        )
        assert final_state["turn_count"] == 1


class TestAdversarialTurnTransitionAttacks:
    """Attacks turn transition logic with counter-examples and edge cases."""

    def test_identical_user_prompt_repetition_advances_turn_count(self, tmp_path: Path) -> None:
        """Adversarial Attack: User repeats the EXACT same prompt on turn 2.
        Verifies that turn_count still advances because pending_tool_continuation is False.
        """
        state_file = tmp_path / "quanta_cognitive_state.json"
        conv_id = "test_conv_repeat_prompt"
        base_time = 1700000000.0

        query = "status check"
        # Turn 1
        _run_inprocess_hook(
            {
                "event": "PreInvocation",
                "conversationId": conv_id,
                "artifactDirectoryPath": str(tmp_path),
                "stepIdx": 0,
                "userPrompt": query,
                "workspacePaths": [str(tmp_path)],
                "testing": True,
            },
            sim_time=base_time,
        )
        s1 = json.loads(state_file.read_text(encoding="utf-8"))
        assert s1["turn_count"] == 1
        assert s1["pending_tool_continuation"] is False

        # Turn 2: User repeats identical prompt, after 10 seconds, no tool continuation pending
        _run_inprocess_hook(
            {
                "event": "PreInvocation",
                "conversationId": conv_id,
                "artifactDirectoryPath": str(tmp_path),
                "stepIdx": 1,
                "userPrompt": query,
                "workspacePaths": [str(tmp_path)],
                "testing": True,
            },
            sim_time=base_time + 10.0,
        )
        s2 = json.loads(state_file.read_text(encoding="utf-8"))
        assert s2["turn_count"] == 2, (
            f"Turn count failed to advance on repeated prompt without tool continuation! Got: {s2['turn_count']}"
        )

    def test_empty_user_prompt_continuation_preserves_turn_count(self, tmp_path: Path) -> None:
        """Verifies sub-turn continuation payload with omitted/empty userPrompt maintains turn_count."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        conv_id = "test_conv_empty_prompt"
        base_time = 1700000000.0

        # Turn 1 start
        _run_inprocess_hook(
            {
                "event": "PreInvocation",
                "conversationId": conv_id,
                "artifactDirectoryPath": str(tmp_path),
                "stepIdx": 0,
                "userPrompt": "Original prompt",
                "workspacePaths": [str(tmp_path)],
                "testing": True,
            },
            sim_time=base_time,
        )

        # Tool 1
        handle_post_tool_use(
            {
                "event": "PostToolUse",
                "conversationId": conv_id,
                "artifactDirectoryPath": str(tmp_path),
                "stepIdx": 1,
                "toolCall": {"name": "run_command", "args": {}},
                "result": {"exit_code": 0},
                "testing": True,
            },
            state_file=state_file,
        )

        # Continuation with empty prompt
        _run_inprocess_hook(
            {
                "event": "PreInvocation",
                "conversationId": conv_id,
                "artifactDirectoryPath": str(tmp_path),
                "stepIdx": 1,
                "userPrompt": "",  # Empty prompt
                "workspacePaths": [str(tmp_path)],
                "testing": True,
            },
            sim_time=base_time + 5.0,
        )
        s_cont = json.loads(state_file.read_text(encoding="utf-8"))
        assert s_cont["turn_count"] == 1


class TestConcurrencyAndReentrancyStress:
    """Stress tests concurrency, re-entrancy debouncing, and atomic file stability."""

    def test_rapid_reentrancy_debounce_protection(self, tmp_path: Path) -> None:
        """Verifies rapid re-entrancy within 2.0s or for identical step index cleanly returns {}."""
        conv_id = "test_conv_reentrancy"
        now = time.time()

        payload = {
            "event": "PreInvocation",
            "conversationId": conv_id,
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 5,
            "userPrompt": "Quick call",
            "workspacePaths": [str(tmp_path)],
            "testing": True,
        }

        # First call succeeds
        res1 = _run_inprocess_hook(payload, sim_time=now)
        assert "injectSteps" in res1

        # Second call immediately (0.01s later, same stepIdx 5) must debounce
        res2 = _run_inprocess_hook(payload, sim_time=now + 0.01)
        assert res2 == {}

        # Third call (1.0s later, new stepIdx 6, but still < 2.0s window) must debounce
        payload_step6 = dict(payload, stepIdx=6)
        res3 = _run_inprocess_hook(payload_step6, sim_time=now + 1.0)
        assert res3 == {}

        # Fourth call (2.1s later, new stepIdx 6) must succeed
        res4 = _run_inprocess_hook(payload_step6, sim_time=now + 2.1)
        assert "injectSteps" in res4

    def test_multithreaded_rapid_atomic_state_saves_zero_corruption(self, tmp_path: Path) -> None:
        """Verifies high-concurrency state saves from multiple workers produce zero corruption."""
        primary_file = tmp_path / "quanta_cognitive_state.json"
        mirror_file = tmp_path / "mirror" / "quanta_cognitive_state.json"

        errors: list[Exception] = []

        def worker_task(worker_id: int) -> None:
            try:
                for cycle in range(15):
                    st = {
                        "turn_count": 1,
                        "worker_id": worker_id,
                        "cycle": cycle,
                        "engrams": [
                            {"key": f"w{worker_id}_c{cycle}", "fidelity": 0.9998, "salience": 2.0}
                        ],
                    }
                    _save_mirrored_state_atomically(
                        primary_path=primary_file,
                        mirror_path=mirror_file,
                        state_dict=st,
                        conv_id="stress_concurrency",
                        is_test_env=True,
                    )
                    # Immediate read check
                    with open(primary_file, encoding="utf-8") as pf:
                        data = json.load(pf)
                        assert "turn_count" in data
            except Exception as ex:
                errors.append(ex)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_task, i) for i in range(10)]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Encountered concurrency errors: {errors}"
        # Verify no orphan .tmp_* files remain
        orphans = list(tmp_path.glob(".tmp_*")) + list((tmp_path / "mirror").glob(".tmp_*"))
        assert len(orphans) == 0, f"Found leaked orphan temp files: {orphans}"

    def test_tool_failure_interleaved_in_25_steps_preserves_turn_count_and_inhibitor_ltp(
        self, tmp_path: Path
    ) -> None:
        """Verifies tool failures during multi-tool continuation do NOT inflate turn_count,
        properly potentiate inhibitors via LTP, and transient decisions remain protected.
        """
        state_file = tmp_path / "quanta_cognitive_state.json"
        conv_id = "test_conv_tool_failure"
        base_time = 1700000000.0

        # Step 0: User query arrives
        _run_inprocess_hook(
            {
                "event": "PreInvocation",
                "conversationId": conv_id,
                "artifactDirectoryPath": str(tmp_path),
                "stepIdx": 0,
                "userPrompt": "Execute pipeline with fault injection",
                "workspacePaths": [str(tmp_path)],
                "testing": True,
            },
            sim_time=base_time,
        )

        # Ingest transient decision
        mem = FastBiomorphicMemory(capacity=16)
        mem.record_transient("transient_task_plan", "Plan A execution", salience=0.5)
        st = json.loads(state_file.read_text(encoding="utf-8"))
        st["engrams"] = mem.engrams
        state_file.write_text(json.dumps(st), encoding="utf-8")

        # Step 1-9: Successful tools
        for step in range(1, 10):
            handle_post_tool_use(
                {
                    "event": "PostToolUse",
                    "conversationId": conv_id,
                    "artifactDirectoryPath": str(tmp_path),
                    "stepIdx": step,
                    "toolCall": {"name": "run_command", "args": {}},
                    "result": {"exit_code": 0},
                    "testing": True,
                },
                state_file=state_file,
            )
            _run_inprocess_hook(
                {
                    "event": "PreInvocation",
                    "conversationId": conv_id,
                    "artifactDirectoryPath": str(tmp_path),
                    "stepIdx": step,
                    "userPrompt": "Execute pipeline with fault injection",
                    "workspacePaths": [str(tmp_path)],
                    "testing": True,
                },
                sim_time=base_time + (step * 3.0),
            )

        # Step 10: Tool failure injected!
        handle_post_tool_use(
            {
                "event": "PostToolUse",
                "conversationId": conv_id,
                "artifactDirectoryPath": str(tmp_path),
                "stepIdx": 10,
                "toolCall": {"name": "faulty_service", "args": {}},
                "error": "Connection refused to database",
                "testing": True,
            },
            state_file=state_file,
        )

        # PreInvocation after failure
        res_post_err = _run_inprocess_hook(
            {
                "event": "PreInvocation",
                "conversationId": conv_id,
                "artifactDirectoryPath": str(tmp_path),
                "stepIdx": 10,
                "userPrompt": "Execute pipeline with fault injection",
                "workspacePaths": [str(tmp_path)],
                "testing": True,
            },
            sim_time=base_time + 35.0,
        )
        assert "injectSteps" in res_post_err
        ephemeral = res_post_err["injectSteps"][0]["ephemeralMessage"]
        # Verify inhibitor potentiated
        assert "inh_faulty_service" in ephemeral
        assert "transient_task_plan" in ephemeral

        st_after_err = json.loads(state_file.read_text(encoding="utf-8"))
        assert st_after_err["turn_count"] == 1, "Tool failure must NOT advance turn_count!"

        # Step 11: Tool retries and succeeds -> LTD depression
        handle_post_tool_use(
            {
                "event": "PostToolUse",
                "conversationId": conv_id,
                "artifactDirectoryPath": str(tmp_path),
                "stepIdx": 11,
                "toolCall": {"name": "faulty_service", "args": {}},
                "result": {"exit_code": 0, "stdout": "reconnected"},
                "testing": True,
            },
            state_file=state_file,
        )

        st_after_recovery = json.loads(state_file.read_text(encoding="utf-8"))
        inh_eng = next(e for e in st_after_recovery["engrams"] if e["key"] == "inh_faulty_service")
        assert inh_eng["v_inh"] < 0.60, f"Expected LTD depression, got V_inh={inh_eng['v_inh']}"
        assert st_after_recovery["turn_count"] == 1

