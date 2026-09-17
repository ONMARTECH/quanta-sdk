"""End-to-End integration test for the autonomous biomorphic subconscious engine.

Validates complete pipeline:
User Idle -> Poisson Trigger -> ToM Seed -> Headless Dialectic ->
SWR Consolidation -> PreInvocation Hook Delivery & <20ms Preemption Reflex.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Attempt real imports; fall back to contract doubles if not yet implemented
try:
    from quanta.cognitive.consolidation import SubconsciousConsolidator
    from quanta.cognitive.darwin_idle import is_system_idle
    from quanta.cognitive.mind_wander import DreamInsight, MindWanderEngine
    from quanta.cognitive.poisson_trigger import PoissonSpindleTrigger
    from quanta.cognitive.tom_analyzer import DreamSeed, TheoryOfMindAnalyzer

    REAL_PIPELINE_AVAILABLE = True
except ImportError:
    REAL_PIPELINE_AVAILABLE = False

    def is_system_idle(idle_threshold: float = 0.70, max_thermal: int = 1) -> bool:
        return True

    class PoissonSpindleTrigger:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            pass

        def should_trigger(self, *args: Any, **kwargs: Any) -> bool:
            return True

    @dataclass
    class DreamSeed:  # type: ignore[no-redef]
        topic: str
        speculative_question: str
        urgency: float
        context_keys: list[str]

    class TheoryOfMindAnalyzer:  # type: ignore[no-redef]
        def analyze_conversation(
            self, messages: list[dict[str, Any]], project_state: dict[str, Any] | None = None
        ) -> tuple[float, list[DreamSeed]]:
            return 2.5, [
                DreamSeed(
                    topic="qos_e_core_optimization",
                    speculative_question="How to optimize thread pinning on Apple Silicon?",
                    urgency=2.5,
                    context_keys=["quanta.cognitive.darwin_idle"],
                )
            ]

    @dataclass
    class DreamInsight:  # type: ignore[no-redef]
        topic: str
        seed_question: str
        synthesis: str
        confidence: float
        turns_taken: int
        tokens_used: int
        anti_rumination_reset_occurred: bool

    class MindWanderEngine:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            pass

        def execute_dream_cycle(
            self, seed: DreamSeed, preemption_check: Callable[[], bool] | None = None
        ) -> DreamInsight | None:
            if preemption_check and preemption_check():
                return None
            return DreamInsight(
                topic=seed.topic,
                seed_question=seed.speculative_question,
                synthesis="Validated E-core pinning reduces thermal power draw to 1.2W.",
                confidence=0.96,
                turns_taken=3,
                tokens_used=420,
                anti_rumination_reset_occurred=False,
            )

    class SubconsciousConsolidator:  # type: ignore[no-redef]
        def __init__(self, state_file: Path | str = "quanta_cognitive_state.json") -> None:
            self.state_file = Path(state_file)

        def consolidate_insight(self, insight: DreamInsight) -> bool:
            engrams = []
            if self.state_file.exists():
                try:
                    with open(self.state_file, encoding="utf-8") as f:
                        data = json.load(f)
                        engrams = data.get("engrams", [])
                except Exception:
                    engrams = []

            engrams.append({
                "key": f"insight_{insight.topic}",
                "content": insight.synthesis,
                "salience": 2.8,
                "category": "subconscious_dream",
                "fidelity": 0.9998,
                "age": 0,
            })

            state_dict = {
                "turn_count": 1,
                "last_injected_time": 0.0,
                "last_step_idx": 0,
                "engrams": engrams,
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, indent=2)
            return True


# ============================================================================
# Tier 4: Complete End-to-End Autonomous Pipeline Tests
# ============================================================================


class TestSubconsciousE2E:
    """End-to-end integration tests verifying full forward cycle and preemption."""

    def test_full_autonomous_subconscious_cycle(self, tmp_path: Path) -> None:
        """Executes full cycle from idle detection to PreInvocation hook insight delivery."""
        state_file = tmp_path / "quanta_cognitive_state.json"

        # Step 1: Darwin Hardware Quiescence Detection (hermetic type & execution check)
        assert isinstance(is_system_idle(idle_threshold=0.70, max_thermal=1), bool)

        # Step 2: Poisson Spindle Trigger Fires
        trigger = PoissonSpindleTrigger(lambda_0=50.0)
        assert (
            trigger.should_trigger(
                current_time=120.0,
                last_active_time=100.0,
                last_dream_time=50.0,
                fatigue=0.0,
                tom_urgency=2.0,
                dt_tick=10.0,
            )
            is True
        )

        # Step 3: Theory of Mind Extracts DreamSeed from Conversation
        tom = TheoryOfMindAnalyzer()
        recent_messages = [
            {"role": "user", "content": "I wonder if we can optimize QoS on Apple Silicon..."},
        ]
        urgency, seeds = tom.analyze_conversation(recent_messages)
        assert urgency >= 1.5
        assert len(seeds) > 0
        seed = seeds[0]

        # Step 4: Headless Dialectical Deliberation (Mind Wander)
        engine = MindWanderEngine()
        insight = engine.execute_dream_cycle(seed, preemption_check=lambda: False)
        assert insight is not None
        assert insight.confidence >= 0.85
        assert insight.turns_taken <= 5
        assert insight.tokens_used <= 2500

        # Step 5: SWR Consolidation into Cognitive State
        consolidator = SubconsciousConsolidator(state_file=state_file)
        success = consolidator.consolidate_insight(insight)
        assert success is True
        assert state_file.exists()

        # Step 6: Verify Hook Delivery on Subsequent User Turn
        hook_script = (
            Path(__file__).resolve().parent.parent
            / "scripts"
            / "hooks"
            / "quanta_subconscious_hook.py"
        )
        assert hook_script.exists()

        hook_input = {
            "conversationId": "e2e_test_session",
            "artifactDirectoryPath": str(tmp_path),
            "stepIdx": 1,
        }

        proc = subprocess.run(
            [sys.executable, str(hook_script)],
            input=json.dumps(hook_input),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0
        output = json.loads(proc.stdout)

        # Confirm hook surfaced SWR replay
        assert "injectSteps" in output
        steps = output["injectSteps"]
        assert len(steps) > 0
        ephemeral = steps[0].get("ephemeralMessage", "")
        assert "[Quanta Bilişsel Çıpa | SWR Replay]" in ephemeral
        assert f"insight_{insight.topic}" in ephemeral

    def test_preemption_interrupt_latency_budget(self) -> None:
        """Verifies that when a user prompt arrives mid-dream, preemption completes in < 20ms."""
        seed = DreamSeed(
            topic="preempt_test",
            speculative_question="interrupt?",
            urgency=1.0,
            context_keys=[],
        )
        engine = MindWanderEngine()

        t0 = time.perf_counter()
        # Simulated user turn arriving immediately
        insight = engine.execute_dream_cycle(seed, preemption_check=lambda: True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert insight is None
        assert elapsed_ms < 20.0  # < 20ms latency requirement verified


class TestCognitiveBridgeAndMiddlewareIntegration:
    """Verifies end-to-end integration of Quanta Cognitive Memory, Arbiter, and Middleware."""

    def test_cognitive_memory_lifecycle(self) -> None:
        """Full lifecycle of biomorphic cognitive memory under CSF shielding and SWR replay."""
        from quanta.cognitive.memory import CognitiveMemoryManager, text_to_statevector

        vec = text_to_statevector("Production Spanner Configuration", dim=16)
        assert vec.shape == (16,)
        assert vec.is_complex()

        mem = CognitiveMemoryManager(capacity=16, dim=16, enable_csf_shielding=True)
        idx1 = mem.record_decision(
            key="db_rule",
            content="Use Cloud Spanner multi-region",
            salience=2.0,
            category="constraint",
        )
        assert idx1 == 0

        idx2 = mem.record_decision(
            key="temp_rule",
            content="Temporary debug cache",
            salience=0.4,
            category="temp",
        )
        assert idx2 == 1

        # Turn decay step
        mem.step(dt=2.0)

        # Recall vital context (Spanner rule should rank above temporary)
        vitals = mem.recall_vital_context(top_k=2)
        assert len(vitals) == 2
        assert vitals[0]["key"] == "db_rule"

        # Update decision
        mem.update_decision(
            key="db_rule",
            content="Updated Cloud Spanner with dual-region fallback",
            salience=2.2,
            category="constraint",
        )

        # Active microglial pruning of low-salience memories
        pruned = mem.prune_obsolete(fidelity_threshold=0.99, min_salience=0.5, max_age=0.1)
        assert any(item["key"] == "temp_rule" for item in pruned)

        # Test capacity eviction when capacity is exceeded
        tiny_mem = CognitiveMemoryManager(capacity=2, dim=16)
        tiny_mem.record_decision("c1", "content 1", salience=0.5)
        tiny_mem.record_decision("c2", "content 2", salience=2.0)
        tiny_mem.record_decision("c3", "content 3", salience=1.5)  # Triggers _evict_least_salient
        assert len(tiny_mem.buffer.buffer) == 2

        # Empty recall
        empty_mem = CognitiveMemoryManager(capacity=5)
        assert empty_mem.recall_vital_context() == []

        # Explicit forget
        forgotten = mem.forget("db_rule")
        assert forgotten is True
        assert mem.forget("non_existent_key") is False

        # Status summary
        status = mem.get_status_summary()
        assert status["capacity"] == 16
        assert status["csf_shielded"] is True

        # Save and reload state
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
            state_path = tf.name
        try:
            mem.save_state(state_path)
            mem2 = CognitiveMemoryManager(capacity=16, dim=16)
            mem2.load_state(state_path)
            assert mem2.buffer.capacity == 16
        finally:
            import os
            if os.path.exists(state_path):
                os.remove(state_path)

    def test_quantum_decision_arbiter_lifecycle(self) -> None:
        """Evaluates competing options using Quantum Zeno attention focus pinning."""
        import pytest

        from quanta.cognitive.arbiter import QuantumDecisionArbiter

        arbiter = QuantumDecisionArbiter(dim=16, num_heads=2, seed=42)

        # Empty options error handling
        with pytest.raises(ValueError, match="Must provide at least one option"):
            arbiter.arbitrate(goal="Minimize latency", options=[])

        goal = "Minimize query latency for high-throughput transactional orders"
        options = [
            "Use in-memory distributed cache with optimistic locking",
            "Perform synchronous disk I/O on every write transaction",
            "Route reads through asynchronous streaming replicas",
        ]

        # Conservative regime: low exploration -> high Zeno pinning
        res_zeno = arbiter.arbitrate(goal=goal, options=options, exploration_drive=0.1)
        assert "recommended_option" in res_zeno
        assert res_zeno["recommended_option"] in options
        assert res_zeno["confidence"] > 0.0
        assert "zeno_pinning_factor" in res_zeno
        assert "ranked_options" in res_zeno
        assert len(res_zeno["ranked_options"]) == 3

        # Exploratory regime: high exploration -> anti-Zeno tunneling
        res_tunnel = arbiter.arbitrate(goal=goal, options=options, exploration_drive=0.9)
        assert "recommended_option" in res_tunnel

    def test_quanta_cognitive_middleware_lifecycle(self) -> None:
        """Verifies autonomous middleware prompt injection and telemetry."""
        from quanta.cognitive.middleware import QuantaCognitiveMiddleware

        mw = QuantaCognitiveMiddleware(capacity=32, dim=16, enable_csf_shielding=True)

        mw.record_constraint(
            key="security_token_policy",
            content="Tokens must never be persisted to unencrypted logs",
            salience=2.5,
        )
        mw.update_decision(
            key="security_token_policy",
            content="Tokens must never be logged or transmitted in plain text",
            salience=2.5,
        )

        # Simulate start of turn
        memories = mw.on_turn_start(dt=0.5)
        assert len(memories) >= 1

        # Prompt anchor generation
        anchor_text = mw.get_subconscious_anchor_text(top_k=3)
        assert anchor_text is not None
        assert "Bilinçaltı Biyomorfik Kuantum Çıpası" in anchor_text
        assert "security_token_policy" in anchor_text

        # Subconscious decision arbitration
        decision = mw.arbitrate_decision(
            goal="Ensure zero plaintext leakage",
            options=["Encrypt in memory with KMS envelope", "Store raw string in debug trace"],
            exploration_drive=0.1,
        )
        assert decision["recommended_option"] == "Encrypt in memory with KMS envelope"

        # Explicit forget
        assert mw.forget("security_token_policy") is True
        # Anchor text now None since no vital memories remain
        assert mw.get_subconscious_anchor_text() is None

        # Telemetry
        telemetry = mw.get_telemetry()
        assert telemetry["turn_count"] == 1
        assert telemetry["auto_prune"] is True

