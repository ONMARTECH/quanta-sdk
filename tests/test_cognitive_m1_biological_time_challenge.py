"""tests/test_cognitive_m1_biological_time_challenge.py

Empirical stress testing and verification harness for Milestone 1 (R1):
Calibrated Biological Time & Calmed Decay Kinetics.

Authored by teamwork_preview_challenger_m1_1 (EMPIRICAL CHALLENGER):
1. Transient decision retention trajectory at S=0.5 across 20, 30, 40, 50, and 111 tool steps.
2. Salience sensitivity comparison: lowest transient S=0.2 vs highest transient S=0.8.
3. Core anchor dielectric shielding (S >= 2.0 or is_core_anchor=True) across 1,000 steps.
4. Inhibitor exponential decay smoothness, floor saturation, and microglial depotentiation.
5. End-to-end subconscious hook lifecycle (handle_post_tool_use + PreInvocation decoupling).
6. Adversarial boundaries: dt=0, float accumulation precision over 1,000 steps.
"""

from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

from scripts.hooks.quanta_subconscious_hook import (
    FastBiomorphicMemory,
    format_fidelity,
    handle_post_tool_use,
)
from scripts.hooks.quanta_subconscious_hook import (
    main as hook_main,
)


class TestTransientDecayMicrostepTrajectory:
    """Empirical challenge 1: Transient decision retention trajectory at dt=0.2."""

    def test_transient_0_5_fidelity_after_20_tool_steps(self) -> None:
        """Verifies transient decision (S=0.5) maintains F >= 0.885 across 20 tool steps (dt=0.2)."""
        mem = FastBiomorphicMemory(capacity=16)
        mem.record_transient("trans_0.5", "Standard ephemeral tool decision", salience=0.5)

        for _ in range(20):
            mem.step(dt=0.2)

        engram = mem.engrams[0]
        fid = engram["fidelity"]
        age = engram["age"]

        # Analytical expectation:
        # kappa_eff = KAPPA_CSF + (1 - KAPPA_CSF)*exp(-2.2*0.5) = 0.333014
        # eff_gamma = 0.05 * sqrt(0.333014) / (1 + 1.5*0.5) = 0.016487
        # F(4.0) = 1/64 + (0.9998 - 1/64) * exp(-0.016487 * 4.0) = 0.936990
        assert age == 4 or age == 4.0
        assert isinstance(age, int)
        assert fid >= 0.885, f"Fidelity {fid:.6f} dropped below required 0.885 after 20 steps!"
        assert abs(fid - 0.936990) < 1e-4, f"Fidelity {fid:.6f} deviates from analytical 0.936990"

        # Microglial pruning must NOT evict this engram
        pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert len(pruned) == 0, f"Transient engram was prematurely pruned after 20 steps: {pruned}"

    def test_transient_0_5_lifespan_30_40_50_steps(self) -> None:
        """Verifies transient decision (S=0.5) survives steps 30, 40, and 50."""
        mem = FastBiomorphicMemory(capacity=16)
        mem.record_transient("trans_0.5", "Standard ephemeral tool decision", salience=0.5)

        step_snapshots: dict[int, float] = {}
        for step in range(1, 51):
            mem.step(dt=0.2)
            if step in (20, 30, 40, 50):
                fid = mem.engrams[0]["fidelity"]
                step_snapshots[step] = fid
                pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
                assert len(pruned) == 0, f"Pruned prematurely at step {step}!"

        # Step 30: t=6.0 -> F ≈ 0.9073 >= 0.70
        assert step_snapshots[30] >= 0.90
        # Step 40: t=8.0 -> F ≈ 0.8785 >= 0.70
        assert step_snapshots[40] >= 0.87
        # Step 50: t=10.0 -> F ≈ 0.8507 >= 0.70
        assert step_snapshots[50] >= 0.85

    def test_transient_0_5_critical_pruning_step(self) -> None:
        """Verifies transient decision (S=0.5) prunes exactly at step 111 (derived t_crit ≈ 22.04)."""
        mem = FastBiomorphicMemory(capacity=16)
        mem.record_transient("trans_0.5", "Standard ephemeral tool decision", salience=0.5)

        pruned_step = None
        for step in range(1, 200):
            mem.step(dt=0.2)
            pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
            if pruned:
                pruned_step = step
                break

        # Analytical derivation:
        # t_crit = -ln((0.70 - 1/64)/(0.9998 - 1/64)) / 0.016487 = 22.04
        # step_crit = ceil(22.04 / 0.2) = 111
        assert pruned_step == 111, f"Expected pruning at step 111, but occurred at {pruned_step}"

    def test_lowest_transient_0_2_trajectory(self) -> None:
        """Verifies lowest transient (S=0.2) maintains F >= 0.885 at step 20 and prunes at 59."""
        mem = FastBiomorphicMemory(capacity=16)
        mem.record_transient("trans_0.2", "Lowest transient decision", salience=0.2)

        # Step 20
        for _ in range(20):
            mem.step(dt=0.2)
        fid_20 = mem.engrams[0]["fidelity"]
        # Analytical: eff_gamma = 0.030867 -> F(4.0) = 0.885486 >= 0.885
        assert fid_20 >= 0.885, f"Lowest transient F={fid_20:.6f} dropped below 0.885 at step 20!"
        assert abs(fid_20 - 0.885486) < 1e-4

        # Survives across 20-30 tool steps
        for _ in range(10):
            mem.step(dt=0.2)
        fid_30 = mem.engrams[0]["fidelity"]
        assert fid_30 >= 0.830
        pruned_30 = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert len(pruned_30) == 0, "S=0.2 should still be alive at step 30"

        # Steps until pruning (derived t_crit ≈ 11.77 -> step 59)
        pruned_step = None
        for step in range(31, 100):
            mem.step(dt=0.2)
            pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
            if pruned:
                pruned_step = step
                break
        assert pruned_step == 59, f"Expected S=0.2 to prune at step 59, got {pruned_step}"

    def test_highest_transient_0_8_trajectory(self) -> None:
        """Verifies highest transient (S=0.8) maintains F >= 0.96 at step 20 and prunes at 193."""
        mem = FastBiomorphicMemory(capacity=16)
        mem.record_transient("trans_0.8", "Highest transient decision", salience=0.8)

        for _ in range(20):
            mem.step(dt=0.2)
        fid_20 = mem.engrams[0]["fidelity"]
        assert fid_20 >= 0.96, f"Fidelity {fid_20:.6f} below 0.96 at step 20"
        assert abs(fid_20 - 0.963366) < 1e-4

        # Prunes at step 193 (derived t_crit ≈ 38.52 -> step 193)
        pruned_step = None
        for step in range(21, 300):
            mem.step(dt=0.2)
            pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
            if pruned:
                pruned_step = step
                break
        assert pruned_step == 193, f"Expected S=0.8 to prune at step 193, got {pruned_step}"


class TestCoreAnchorDielectricShieldingStress:
    """Empirical challenge 2: Core anchor dielectric shielding."""

    def test_core_anchor_shielding_fidelity_and_formatting(self) -> None:
        """Verifies core anchors (S >= 2.0 or is_core_anchor=True) retain 99.98% formatted fidelity."""
        mem = FastBiomorphicMemory(capacity=16)
        mem.record("core_explicit", "Flagged core anchor", salience=1.0, is_core_anchor=True)
        mem.record("core_sal_2.0", "Boundary core anchor S=2.0", salience=2.0)
        mem.record("core_sal_2.8", "High core anchor S=2.8", salience=2.8)

        # 50 tool steps (dt=0.2 => t = 10.0)
        for _ in range(50):
            mem.step(dt=0.2)

        for e in mem.engrams:
            fid = e["fidelity"]
            fmt = format_fidelity(fid)
            assert fmt == "99.98%", f"Engram {e['key']} formatted fidelity deviated: {fmt}"
            assert fid >= 0.9997, f"Engram {e['key']} fidelity {fid:.8f} dropped below 0.9997"

    def test_core_anchor_extreme_1000_steps_retention(self) -> None:
        """Verifies core anchors under 1,000 tool steps (t = 200.0) remain > 0.999 and never prune."""
        mem = FastBiomorphicMemory(capacity=16)
        mem.record("core_rule", "Critical invariant rule", salience=2.5, is_core_anchor=True)

        for _ in range(1000):
            mem.step(dt=0.2)

        e = mem.engrams[0]
        # Mathematical verification:
        # eff_gamma = (0.05 * 0.00016) / (1 + 1.5 * 2.5) = 1.6842e-6
        # F(200) = 1/64 + (0.9998 - 1/64) * exp(-1.6842e-6 * 200) = 0.999468 >= 0.999
        assert e["fidelity"] >= 0.9990
        assert e["age"] == 200 or e["age"] == 200.0
        assert isinstance(e["age"], int)

        # 100% immune from microglial pruning
        pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert len(pruned) == 0


class TestInhibitorExponentialDecayKinetics:
    """Empirical challenge 3: Inhibitor exponential decay smoothness and depotentiation."""

    def test_inhibitor_exponential_smoothness_no_cliff_drops(self) -> None:
        """Verifies unreinforced inhibitor decays strictly exponentially: V(t+dt)/V(t) = exp(-0.02*dt)."""
        mem = FastBiomorphicMemory(capacity=16)
        mem.record_inhibitor("inh_rule", "Forbidden action", v_inh=0.50, salience=0.50)

        v_history = [float(mem.engrams[0]["v_inh"])]
        dts = [0.2] * 50 + [0.1] * 20 + [0.5] * 10

        for dt in dts:
            mem.step(dt=dt)
            v_curr = float(mem.engrams[0]["v_inh"])
            v_prev = v_history[-1]
            v_history.append(v_curr)

            expected_ratio = math.exp(-0.02 * dt)
            actual_ratio = v_curr / v_prev
            assert abs(actual_ratio - expected_ratio) < 1e-12, (
                f"Decay not smooth at dt={dt}: actual {actual_ratio:.8f} vs expected {expected_ratio:.8f}"
            )

    def test_inhibitor_floor_saturation(self) -> None:
        """Verifies inhibitor weight saturates at V_inh_min = 0.05 without underflowing."""
        mem = FastBiomorphicMemory(capacity=16)
        mem.record_inhibitor("inh_rule", "Forbidden action", v_inh=0.50, salience=0.50)

        # 3,000 steps of dt=0.2 (t=600.0)
        for _ in range(3000):
            mem.step(dt=0.2)

        v_final = mem.engrams[0]["v_inh"]
        assert v_final == 0.05

    def test_unreinforced_inhibitor_pruned_via_fidelity_at_step_111(self) -> None:
        """Verifies unreinforced inhibitor (S=0.50) is pruned via fidelity decay at step 111."""
        mem = FastBiomorphicMemory(capacity=16)
        mem.record_inhibitor("inh_ephemeral", "Bad tool pattern", v_inh=0.50, salience=0.50)

        pruned_step = None
        prune_reason = None
        for step in range(1, 300):
            mem.step(dt=0.2)
            pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
            if pruned:
                pruned_step = step
                prune_reason = pruned[0].get("reason", "")
                break

        # S=0.50 reaches F < 0.70 at step 111 before V_inh reaches 0.20 at step 230
        assert pruned_step == 111
        assert "fidelity_decayed" in prune_reason

    def test_active_ltd_inhibitor_pruned_via_depotentiation(self) -> None:
        """Verifies inhibitor with active LTD depression is pruned via depotentiated_inhibitor."""
        mem = FastBiomorphicMemory(capacity=16)
        mem.record_inhibitor("inh_ephemeral", "Bad tool pattern", v_inh=0.50, salience=0.50)

        # Simulate 3 successful tool runs triggering LTD depression: V <- V * 0.70
        for _ in range(3):
            mem.step(dt=0.2)
            mem.engrams[0]["v_inh"] = max(0.05, float(mem.engrams[0]["v_inh"]) * 0.70)

        v_now = mem.engrams[0]["v_inh"]
        fid_now = mem.engrams[0]["fidelity"]
        # v_now = 0.5 * 0.7^3 ≈ 0.1715 < 0.20 while fid_now ≈ 0.99 > 0.70
        assert v_now < 0.20
        assert fid_now > 0.95

        pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)
        assert len(pruned) == 1
        assert "depotentiated_inhibitor" in pruned[0].get("reason", "")


class TestPostToolUseSubconsciousHookEndToEnd:
    """Empirical challenge 4: Subconscious hook runtime execution and turn decoupling."""

    def test_post_tool_use_mutates_state_dt_0_2(self, tmp_path: Path) -> None:
        """Verifies handle_post_tool_use steps engram ages by 0.2 and decays fidelity."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        initial_state = {
            "turn_count": 5,
            "last_verified_step": 10,
            "engrams": [
                {
                    "key": "transient_test",
                    "content": "Temporary plan",
                    "salience": 0.5,
                    "category": "general",
                    "fidelity": 0.9998,
                    "age": 0,
                    "is_core_anchor": False,
                }
            ],
        }
        state_file.write_text(json.dumps(initial_state), encoding="utf-8")

        payload = {
            "event": "PostToolUse",
            "conversationId": "test_m1_post_hook",
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 11,
            "toolCall": {"name": "run_command", "args": {"CommandLine": "ls"}},
            "result": {"exit_code": 0, "stdout": "ok"},
            "testing": True,
        }

        handle_post_tool_use(payload, state_file=state_file)

        saved = json.loads(state_file.read_text(encoding="utf-8"))
        assert saved["turn_count"] == 5, "turn_count must NOT advance on tool execution!"
        assert saved.get("pending_tool_continuation") is True
        e = saved["engrams"][0]
        assert e["age"] == 0.2
        assert e["fidelity"] < 0.9998
        assert e["fidelity"] >= 0.995

    def test_pre_invocation_subturn_decoupling_avoids_double_step(self, tmp_path: Path) -> None:
        """Verifies PreInvocation does not double-step or increment turn_count on subturn continuation."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        initial_state = {
            "turn_count": 5,
            "last_step_idx": 11,
            "last_user_query": "please investigate bug",
            "pending_tool_continuation": True,
            "engrams": [
                {
                    "key": "transient_test",
                    "content": "Temporary plan",
                    "salience": 0.5,
                    "category": "general",
                    "fidelity": 0.936990,
                    "age": 4.0,
                    "is_core_anchor": False,
                }
            ],
        }
        state_file.write_text(json.dumps(initial_state), encoding="utf-8")

        pre_payload = {
            "conversationId": "test_conv_decoupling",
            "artifactDirectoryPath": str(tmp_path),
            "workspacePaths": [str(tmp_path)],
            "stepIdx": 12,
            "prompt": "Please investigate bug",  # Same user query (continuation)
            "testing": True,
        }

        old_stdin = sys.stdin
        old_stdout = sys.stdout
        try:
            sys.stdin = io.StringIO(json.dumps(pre_payload))
            sys.stdout = io.StringIO()
            hook_main()
        finally:
            sys.stdin = old_stdin
            sys.stdout = old_stdout

        after = json.loads(state_file.read_text(encoding="utf-8"))
        # turn_count remains 5 (not incremented to 6)
        assert after["turn_count"] == 5
        # age remains 4 (not incremented by 1.0)
        assert after["engrams"][0]["age"] == 4
        # pending_tool_continuation is reset
        assert after.get("pending_tool_continuation") is False

    def test_new_user_conversational_turn_advances_dt_1(self, tmp_path: Path) -> None:
        """Verifies genuine new user prompt advances turn_count by 1 and steps dt=1.0."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        initial_state = {
            "turn_count": 5,
            "last_step_idx": 11,
            "last_user_query": "old prompt",
            "pending_tool_continuation": False,
            "engrams": [
                {
                    "key": "transient_test",
                    "content": "Temporary plan",
                    "salience": 0.5,
                    "category": "general",
                    "fidelity": 0.936990,
                    "age": 4.0,
                    "is_core_anchor": False,
                }
            ],
        }
        state_file.write_text(json.dumps(initial_state), encoding="utf-8")

        pre_payload = {
            "conversationId": "test_conv_turn_advance",
            "artifactDirectoryPath": str(tmp_path),
            "workspacePaths": [str(tmp_path)],
            "stepIdx": 12,
            "prompt": "Brand new user prompt",
            "testing": True,
        }

        old_stdin = sys.stdin
        old_stdout = sys.stdout
        try:
            sys.stdin = io.StringIO(json.dumps(pre_payload))
            sys.stdout = io.StringIO()
            hook_main()
        finally:
            sys.stdin = old_stdin
            sys.stdout = old_stdout

        after = json.loads(state_file.read_text(encoding="utf-8"))
        assert after["turn_count"] == 6
        assert after["engrams"][0]["age"] == 5
        assert after["last_user_query"] == "brand new user prompt"


class TestAdversarialBoundaryKinetics:
    """Empirical challenge 5: Numerical edge cases and float precision."""

    def test_zero_dt_step_is_idempotent(self) -> None:
        """Verifies dt=0.0 causes zero age progression and zero fidelity decay."""
        mem = FastBiomorphicMemory(capacity=16)
        mem.record_transient("trans", "Ephemeral", salience=0.5)

        initial_fid = mem.engrams[0]["fidelity"]
        initial_age = mem.engrams[0]["age"]

        mem.step(dt=0.0)

        assert mem.engrams[0]["fidelity"] == initial_fid
        assert mem.engrams[0]["age"] == initial_age

    def test_float_accumulation_rounding_precision_1000_steps(self) -> None:
        """Verifies 1,000 steps of dt=0.2 accumulates cleanly into int(200) without float drift."""
        mem = FastBiomorphicMemory(capacity=16)
        mem.record("core", "Rule", salience=2.5, is_core_anchor=True)

        for _ in range(1000):
            mem.step(dt=0.2)

        age = mem.engrams[0]["age"]
        assert age == 200
        assert isinstance(age, int)
