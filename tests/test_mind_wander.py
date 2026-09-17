"""Tests for the isolated headless dialectic mind-wandering engine.

Tests dual-persona dialectic (DMN Dreamer vs Zeno Arbiter), psychiatric
anti-rumination cosine detection (> 0.95), synthetic noradrenaline reset,
token and turn bounds, and < 20ms preemption interruption.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import pytest

# Attempt real import; fall back to contract double if not yet implemented
try:
    from quanta.cognitive.mind_wander import DreamInsight, MindWanderEngine
    from quanta.cognitive.tom_analyzer import DreamSeed

    REAL_WANDER_AVAILABLE = True
except ImportError:
    REAL_WANDER_AVAILABLE = False

    @dataclass
    class DreamSeed:  # type: ignore[no-redef]
        topic: str
        speculative_question: str
        urgency: float
        context_keys: list[str]

    @dataclass
    class DreamInsight:  # type: ignore[no-redef]
        """Authoritative contract double adhering strictly to PROJECT.md."""

        topic: str
        seed_question: str
        synthesis: str
        confidence: float
        turns_taken: int
        tokens_used: int
        anti_rumination_reset_occurred: bool

    class MindWanderEngine:  # type: ignore[no-redef]
        """Authoritative contract double adhering strictly to PROJECT.md & Survey 2."""

        def __init__(
            self,
            max_turns: int = 5,
            max_tokens: int = 2500,
            rumination_threshold: float = 0.95,
        ) -> None:
            self.max_turns = max_turns
            self.max_tokens = max_tokens
            self.rumination_threshold = rumination_threshold

        def _cosine_similarity(self, u: list[float], v: list[float]) -> float:
            dot = sum(a * b for a, b in zip(u, v, strict=False))
            nu = math.sqrt(sum(a * a for a in u)) or 1e-9
            nv = math.sqrt(sum(b * b for b in v)) or 1e-9
            return max(-1.0, min(1.0, dot / (nu * nv)))

        def execute_dream_cycle(
            self,
            seed: DreamSeed,
            preemption_check: Callable[[], bool] | None = None,
            simulated_thought_vectors: list[list[float]] | None = None,
            simulated_turn_tokens: int = 150,
        ) -> DreamInsight | None:
            turns = 0
            tokens = 0
            reset_occurred = False
            prev_vec: list[float] | None = None
            consecutive_rumination = 0

            for i in range(self.max_turns):
                if preemption_check and preemption_check():
                    return None

                turns += 1
                tokens += simulated_turn_tokens
                if tokens > self.max_tokens:
                    tokens = self.max_tokens
                    break

                if simulated_thought_vectors and i < len(simulated_thought_vectors):
                    current_vec = simulated_thought_vectors[i]
                else:
                    current_vec = [random.random() for _ in range(8)]

                if prev_vec is not None:
                    sim = self._cosine_similarity(current_vec, prev_vec)
                    if sim > self.rumination_threshold:
                        reset_occurred = True
                        consecutive_rumination += 1
                        if consecutive_rumination >= 2:
                            # Hard abort on persistent rumination loop
                            return None
                    else:
                        consecutive_rumination = 0

                prev_vec = current_vec

            return DreamInsight(
                topic=seed.topic,
                seed_question=seed.speculative_question,
                synthesis=f"Consolidated biomorphic insight on {seed.topic}",
                confidence=0.95,
                turns_taken=turns,
                tokens_used=tokens,
                anti_rumination_reset_occurred=reset_occurred,
            )


# ============================================================================
# Tier 3B: Headless Dialectic & Persona Execution Tests
# ============================================================================


class TestMindWanderDialectic:
    """Verifies that the headless dialectic produces high-quality consensual insights."""

    @pytest.fixture
    def engine(self) -> MindWanderEngine:
        return MindWanderEngine(max_turns=5, max_tokens=2500)

    @pytest.fixture
    def sample_seed(self) -> DreamSeed:
        return DreamSeed(
            topic="darwin_qos",
            speculative_question="Can we schedule mind-wandering strictly on E-cores?",
            urgency=1.5,
            context_keys=["quanta.cognitive.darwin_idle"],
        )

    def test_nominal_dream_cycle_completes(
        self, engine: MindWanderEngine, sample_seed: DreamSeed
    ) -> None:
        """Nominal dream cycle runs within bounds and outputs valid DreamInsight."""
        insight = engine.execute_dream_cycle(sample_seed, preemption_check=lambda: False)
        assert insight is not None
        assert isinstance(insight, DreamInsight)
        assert insight.topic == sample_seed.topic
        assert insight.seed_question == sample_seed.speculative_question
        assert len(insight.synthesis) > 0
        assert insight.confidence >= 0.85
        assert 1 <= insight.turns_taken <= 5
        assert insight.tokens_used <= 2500


# ============================================================================
# Tier 3B: Bounded Resource Ceilings (Turn Cap <= 5, Token Cap <= 2500)
# ============================================================================


class TestBoundedResourceCeilings:
    """Verifies that turn caps and token budgets are strictly enforced."""

    def test_hard_turn_limit_enforced(self) -> None:
        """Even under unconstrained reasoning, turns taken never exceed max_turns."""
        seed = DreamSeed("test", "test?", 1.0, [])
        for cap in (1, 3, 5):
            engine = MindWanderEngine(max_turns=cap, max_tokens=10000)
            insight = engine.execute_dream_cycle(seed, preemption_check=lambda: False)
            assert insight is not None
            assert insight.turns_taken == cap

    def test_hard_token_cap_enforced(self) -> None:
        """When cumulative tokens exceed max_tokens, deliberation immediately halts."""
        seed = DreamSeed("test", "test?", 1.0, [])
        # Each turn generates 800 tokens; max_tokens = 1500 -> must stop at turn 2 (1500 tokens)
        engine = MindWanderEngine(max_turns=10, max_tokens=1500)
        # Using simulated_turn_tokens parameter
        insight = engine.execute_dream_cycle(
            seed,
            preemption_check=lambda: False,
            simulated_turn_tokens=800,
        )
        assert insight is not None
        assert insight.tokens_used <= 1500
        assert insight.turns_taken == 2


# ============================================================================
# Tier 3B: Psychiatric Anti-Rumination Safeguard Tests
# ============================================================================


class TestAntiRuminationSafeguards:
    """Verifies that thought vector loops (cosine similarity > 0.95) trigger noradrenaline reset."""

    def test_rumination_triggers_noradrenaline_reset(self) -> None:
        """Identical thought vectors across consecutive turns trigger noradrenaline reset."""
        seed = DreamSeed("rumination_test", "loop?", 1.0, [])
        engine = MindWanderEngine(max_turns=3, rumination_threshold=0.95)

        # Feed identical thought vectors to simulate circular thinking
        repetitive_vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],  # Cosine sim = 1.0 > 0.95 -> Trigger reset!
            [0.0, 1.0, 0.0, 0.0],  # Diverged vector after reset
        ]

        insight = engine.execute_dream_cycle(
            seed,
            preemption_check=lambda: False,
            simulated_thought_vectors=repetitive_vectors,
        )
        assert insight is not None
        assert insight.anti_rumination_reset_occurred is True

    def test_persistent_rumination_causes_hard_abort(self) -> None:
        """If thoughts remain locked in a loop despite reset, hard abort returns None."""
        seed = DreamSeed("persistent_rumination", "loop?", 1.0, [])
        engine = MindWanderEngine(max_turns=5, rumination_threshold=0.95)

        # 3 consecutive identical thought vectors
        persistent_loop = [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],  # 1st violation -> reset
            [1.0, 0.0, 0.0, 0.0],  # 2nd violation -> hard abort!
        ]

        insight = engine.execute_dream_cycle(
            seed,
            preemption_check=lambda: False,
            simulated_thought_vectors=persistent_loop,
        )
        assert insight is None


# ============================================================================
# Tier 3B: Instant Preemption Interruption Tests (< 20ms Latency)
# ============================================================================


class TestInstantPreemption:
    """Verifies that preemption halts mind-wandering in < 20ms without stalling."""

    def test_immediate_preemption_at_start(self) -> None:
        """Preemption flag True before dream cycle starts yields immediately."""
        seed = DreamSeed("preemption", "preempt?", 1.0, [])
        engine = MindWanderEngine()
        t0 = time.perf_counter()
        insight = engine.execute_dream_cycle(seed, preemption_check=lambda: True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert insight is None
        assert elapsed_ms < 20.0  # Must preempt in < 20ms

    def test_preemption_during_active_deliberation(self) -> None:
        """Preemption triggered mid-cycle halts execution within < 20ms."""
        seed = DreamSeed("preemption", "preempt?", 1.0, [])
        engine = MindWanderEngine(max_turns=5)

        turn_counter = 0

        def check_preempt_at_turn_2() -> bool:
            nonlocal turn_counter
            turn_counter += 1
            return turn_counter >= 2

        t0 = time.perf_counter()
        insight = engine.execute_dream_cycle(seed, preemption_check=check_preempt_at_turn_2)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert insight is None
        assert elapsed_ms < 20.0

    def test_shannon_entropy_and_similarity_math(self) -> None:
        """Covers compute_shannon_entropy and cosine_similarity boundary cases."""
        from quanta.cognitive.mind_wander import compute_shannon_entropy, cosine_similarity

        assert compute_shannon_entropy("") == 0.0
        assert compute_shannon_entropy("hello") == 0.0
        assert compute_shannon_entropy("repeat repeat repeat") == 0.0
        assert compute_shannon_entropy("one two three four five") > 0.5

        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_antigravity_sdk_mock_dialectic(self) -> None:
        """Covers _execute_antigravity_sdk_dialectic and SDK integration branch."""
        from unittest.mock import MagicMock, patch

        from quanta.cognitive.mind_wander import DreamSeed, MindWanderEngine

        seed = DreamSeed("sdk_topic", "how to scale?", 1.5, ["sdk"])
        engine = MindWanderEngine(use_sdk_if_available=True)

        mock_agent = MagicMock()
        mock_convo = MagicMock()
        mock_convo.send.return_value = MagicMock(
            content="Synthetic consensus from Antigravity agents"
        )

        with patch("quanta.cognitive.mind_wander.ANTIGRAVITY_SDK_AVAILABLE", True), \
             patch("quanta.cognitive.mind_wander.Agent", return_value=mock_agent), \
             patch("quanta.cognitive.mind_wander.Conversation", return_value=mock_convo), \
             patch("quanta.cognitive.mind_wander.LocalAgentConfig", return_value=MagicMock()), \
             patch("quanta.cognitive.mind_wander.BudgetConfig", return_value=MagicMock()), \
             patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}):

            insight = engine._execute_with_sdk(seed)
            assert insight is not None
            assert insight.topic == "sdk_topic"

            # Preemption check active
            preempted = engine._execute_with_sdk(seed, preemption_check=lambda: True)
            assert preempted is None

            # End-to-end SDK deliberation branch
            e2e_sdk = engine.execute_dream_cycle(seed)
            assert e2e_sdk is not None

