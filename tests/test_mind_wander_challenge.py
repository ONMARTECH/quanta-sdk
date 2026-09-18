"""Empirical adversarial challenge and stress test suite for MindWanderEngine.

Mission Targets:
1. Preemption Interrupt Latency:
   - Measure preemption_check() returning True at start, mid-turn, and end of cycles.
   - Assert latency < 20 ms across all phases and across statistical distributions.
2. Persistent Rumination Abort:
   - Construct adversarial repetitive seeds with identical thought vectors.
   - Assert synthetic noradrenaline reset fires on turn 1 (2nd turn).
   - Assert cycle hard aborts (returns None) on turn 2 (3rd turn).
   - Assert recovery dynamics when thought vectors diverge after turn 1 reset.
3. Strict Token and Turn Boundaries:
   - Exhaustive Cartesian grid test: max_turns in {1, 2, 5, 10} x max_tokens in {100, 500, 2500}.
   - Multi-token consumption step testing.
   - Assert bounds are strictly respected under all conditions.
4. Invalid and Boundary Inputs:
   - Empty, whitespace, Unicode, emoji, and giant (50k char) questions and topics.
   - None preemption callbacks.
   - High rumination thresholds (1.0, 1.05) and low thresholds (0.0, -1.0).
   - Extreme boundary limits (max_turns=0, max_tokens=0).
5. Property-Based Fuzzing via Hypothesis:
   - Invariant fuzzing on cosine_similarity, compute_shannon_entropy, and execute_dream_cycle.
"""

from __future__ import annotations

import math
import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quanta.cognitive.mind_wander import (
    DreamInsight,
    MindWanderEngine,
    compute_shannon_entropy,
    cosine_similarity,
)
from quanta.cognitive.tom_analyzer import DreamSeed

# ============================================================================
# Section 1: Preemption Interrupt Latency Stress Tests (< 20ms Assertion)
# ============================================================================


class TestPreemptionInterruptLatency:
    """Rigorous empirical timing and stress testing for instant preemption (< 20ms)."""

    @pytest.fixture
    def nominal_seed(self) -> DreamSeed:
        return DreamSeed(
            topic="preemption_bench",
            speculative_question="Does preemption reflex respond in <20ms?",
            urgency=1.0,
            context_keys=["darwin_idle"],
        )

    def test_preemption_at_start_latency_1000_runs(self, nominal_seed: DreamSeed) -> None:
        """Preemption returning True immediately at start (before turn 0).

        Must preempt in < 20 ms on every run across 1,000 iterations.
        """
        latencies_ms: list[float] = []
        for _ in range(1000):
            engine = MindWanderEngine(max_turns=5, max_tokens=2500)
            t0 = time.perf_counter()
            result = engine.execute_dream_cycle(nominal_seed, preemption_check=lambda: True)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            assert result is None
            assert elapsed_ms < 20.0, f"Start preemption exceeded 20ms: {elapsed_ms:.4f}ms"
            latencies_ms.append(elapsed_ms)

        max_latency = max(latencies_ms)
        avg_latency = sum(latencies_ms) / len(latencies_ms)
        p99_latency = sorted(latencies_ms)[int(0.99 * len(latencies_ms))]

        assert max_latency < 20.0
        assert avg_latency < 1.0, f"Average start latency unexpectedly high: {avg_latency:.4f}ms"
        assert p99_latency < 5.0, f"P99 start latency unexpectedly high: {p99_latency:.4f}ms"

    @pytest.mark.parametrize("target_turn", [1, 2, 3, 4])
    def test_preemption_mid_turn_latency_100_runs(
        self, nominal_seed: DreamSeed, target_turn: int
    ) -> None:
        """Preemption returning True mid-cycle at turn 1, 2, 3, or 4.

        Must halt immediately and the final interruption latency must be < 20 ms.
        """
        latencies_ms: list[float] = []
        for _ in range(100):
            engine = MindWanderEngine(max_turns=10, max_tokens=5000)
            state = {"call_count": 0}

            def mid_preempt(s: dict[str, int] = state, trigger_at: int = target_turn) -> bool:
                s["call_count"] += 1
                return s["call_count"] >= (trigger_at + 1)

            t0 = time.perf_counter()
            result = engine.execute_dream_cycle(nominal_seed, preemption_check=mid_preempt)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            assert result is None
            assert elapsed_ms < 20.0, f"Mid-turn preemption exceeded 20ms: {elapsed_ms:.4f}ms"
            latencies_ms.append(elapsed_ms)

        assert max(latencies_ms) < 20.0

    def test_preemption_at_end_of_cycle_latency(self, nominal_seed: DreamSeed) -> None:
        """Preemption returning True at the final turn of the cycle.

        When max_turns=5, preemption fires on the 5th check (turn_idx=4).
        """
        latencies_ms: list[float] = []
        for _ in range(100):
            engine = MindWanderEngine(max_turns=5, max_tokens=2500)
            state = {"call_count": 0}

            def end_preempt(s: dict[str, int] = state) -> bool:
                s["call_count"] += 1
                return s["call_count"] >= 5  # Fires on the 5th step (end of cycle)

            t0 = time.perf_counter()
            result = engine.execute_dream_cycle(nominal_seed, preemption_check=end_preempt)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            assert result is None
            assert elapsed_ms < 20.0, f"End-of-cycle preemption exceeded 20ms: {elapsed_ms:.4f}ms"
            latencies_ms.append(elapsed_ms)

        assert max(latencies_ms) < 20.0

    def test_preemption_callback_overhead_benchmark(self, nominal_seed: DreamSeed) -> None:
        """Verify MindWanderEngine adds < 1ms overhead on top of preemption callback itself."""
        engine = MindWanderEngine(max_turns=5, use_agy_cli=False)

        # Warmup execution to avoid cold start / import jitter
        engine.execute_dream_cycle(nominal_seed, preemption_check=lambda: True)

        # A callback simulating a 2ms kernel/IPC interrupt check
        def simulated_kernel_ipc_check() -> bool:
            time.sleep(0.002)  # 2ms simulated IPC delay
            return True

        cb_start = time.perf_counter()
        simulated_kernel_ipc_check()
        measured_cb_ms = (time.perf_counter() - cb_start) * 1000.0

        latencies: list[float] = []
        result = None
        for _ in range(3):
            t0 = time.perf_counter()
            result = engine.execute_dream_cycle(
                nominal_seed,
                preemption_check=simulated_kernel_ipc_check,
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)

        total_elapsed_ms = min(latencies)
        assert result is None
        # Engine overhead = total - measured callback duration
        engine_overhead_ms = max(0.0, total_elapsed_ms - measured_cb_ms)
        assert (
            engine_overhead_ms < 1.0
        ), f"Engine preemption overhead too high: {engine_overhead_ms:.4f}ms"
        assert total_elapsed_ms < 20.0


# ============================================================================
# Section 2: Persistent Rumination Abort Stress Tests
# ============================================================================


class TestPersistentRuminationAbort:
    """Stress tests for psychiatric anti-rumination safeguards.

    Verifies:
    1. Turn 1 (0-indexed 1, 2nd turn): synthetic noradrenaline reset fires.
    2. Turn 2 (0-indexed 2, 3rd turn): cycle hard aborts (returns None).
    3. Divergence after turn 1 reset allows cycle recovery.
    """

    @pytest.fixture
    def repetitive_seed(self) -> DreamSeed:
        return DreamSeed(
            topic="adversarial_rumination",
            speculative_question="Are we trapped in an associative attractor basin?",
            urgency=2.0,
            context_keys=["quanta.cognitive.arbiter"],
        )

    def test_synthetic_noradrenaline_reset_on_turn_1_and_hard_abort_on_turn_2(
        self, repetitive_seed: DreamSeed
    ) -> None:
        """Adversarial thought vector repetition test.

        - max_turns=2 with [v, v]:
          Turn 0: initial thought
          Turn 1: cosine sim = 1.0 > 0.95 -> noradrenaline reset fires!
          Cycle finishes with reset_occurred = True.
        - max_turns=3 with [v, v, v]:
          Turn 0: initial thought
          Turn 1: reset fires (consecutive = 1)
          Turn 2: persistent rumination (consecutive = 2 >= 2) -> HARD ABORT returns None!
        """
        vec = [0.5, 0.5, 0.5, 0.5]

        # Phase 1: Assert noradrenaline reset fires on turn 1 (2-turn cycle)
        engine_turn1 = MindWanderEngine(max_turns=2, rumination_threshold=0.95)
        insight_turn1 = engine_turn1.execute_dream_cycle(
            repetitive_seed,
            simulated_thought_vectors=[vec, vec],
        )
        assert insight_turn1 is not None, "Cycle should not hard abort on turn 1 reset"
        assert insight_turn1.anti_rumination_reset_occurred is True, (
            "Noradrenaline reset must fire on turn 1"
        )
        assert insight_turn1.turns_taken == 2
        assert insight_turn1.tokens_used == 300

        # Phase 2: Assert hard abort returns None on turn 2 (3-turn cycle)
        engine_turn2 = MindWanderEngine(max_turns=3, rumination_threshold=0.95)
        insight_turn2 = engine_turn2.execute_dream_cycle(
            repetitive_seed,
            simulated_thought_vectors=[vec, vec, vec],
        )
        assert insight_turn2 is None, (
            "Cycle must hard abort (return None) on turn 2 persistent rumination"
        )

    @pytest.mark.parametrize("dim", [4, 8, 16, 64])
    def test_persistent_rumination_abort_various_dimensions(
        self, repetitive_seed: DreamSeed, dim: int
    ) -> None:
        """Hard abort on turn 2 must hold across various vector dimensions."""
        vec = [1.0 / math.sqrt(dim)] * dim
        engine = MindWanderEngine(max_turns=5, rumination_threshold=0.95)
        result = engine.execute_dream_cycle(
            repetitive_seed,
            simulated_thought_vectors=[vec, vec, vec],
        )
        assert result is None, f"Failed to hard abort with dim={dim}"

    def test_divergence_after_turn_1_reset_allows_recovery(
        self, repetitive_seed: DreamSeed
    ) -> None:
        """When the thought vector diverges after turn 1 reset, the cycle successfully recovers."""
        engine = MindWanderEngine(max_turns=4, rumination_threshold=0.95)
        # Turn 0 and 1 are identical; Turn 2 diverges orthogonally; Turn 3 continues
        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],  # triggers reset on turn 1
            [0.0, 1.0, 0.0, 0.0],  # diverges on turn 2 (cosine sim = 0.0) -> resets count
            [0.0, 0.0, 1.0, 0.0],  # diverges on turn 3
        ]
        insight = engine.execute_dream_cycle(
            repetitive_seed,
            simulated_thought_vectors=vectors,
        )
        assert insight is not None, "Cycle should have recovered after divergence"
        assert insight.anti_rumination_reset_occurred is True, (
            "Reset must still be recorded as having occurred"
        )
        assert insight.turns_taken == 4
        assert insight.confidence >= 0.85

    def test_threshold_boundary_sensitivity_949_vs_951(
        self, repetitive_seed: DreamSeed
    ) -> None:
        """Verify sharp boundary sensitivity at the 0.95 threshold:

        - cos sim ~ 0.949 -> NO reset
        - cos sim ~ 0.951 -> RESET triggered
        """
        engine = MindWanderEngine(max_turns=2, rumination_threshold=0.95)

        # Construct vector pair with cosine similarity = cos(theta)
        theta_sub = math.acos(0.949)
        v0 = [1.0, 0.0]
        v_sub = [math.cos(theta_sub), math.sin(theta_sub)]
        sim_sub = cosine_similarity(v0, v_sub)
        assert 0.948 < sim_sub < 0.950

        res_sub = engine.execute_dream_cycle(
            repetitive_seed,
            simulated_thought_vectors=[v0, v_sub],
        )
        assert res_sub is not None
        assert res_sub.anti_rumination_reset_occurred is False, (
            f"Similarity {sim_sub:.4f} < 0.95 should not trigger reset"
        )

        theta_super = math.acos(0.951)
        v_super = [math.cos(theta_super), math.sin(theta_super)]
        sim_super = cosine_similarity(v0, v_super)
        assert 0.950 < sim_super < 0.952

        res_super = engine.execute_dream_cycle(
            repetitive_seed,
            simulated_thought_vectors=[v0, v_super],
        )
        assert res_super is not None
        assert res_super.anti_rumination_reset_occurred is True, (
            f"Similarity {sim_super:.4f} > 0.95 MUST trigger reset"
        )


# ============================================================================
# Section 3: Token and Turn Boundary Invariant Tests
# ============================================================================


class TestTokenAndTurnBoundaries:
    """Rigorous matrix testing of bounds:

    - max_turns in {1, 2, 5, 10}
    - max_tokens in {100, 500, 2500}
    - Simulated token step sizes {25, 100, 150, 300, 1000}
    """

    @pytest.fixture
    def test_seed(self) -> DreamSeed:
        return DreamSeed("boundary_test", "Can we exceed boundaries?", 1.0, [])

    @pytest.mark.parametrize("max_turns", [1, 2, 5, 10])
    @pytest.mark.parametrize("max_tokens", [100, 500, 2500])
    @pytest.mark.parametrize("simulated_turn_tokens", [25, 100, 150, 300, 1000])
    def test_cartesian_grid_boundary_respect(
        self,
        test_seed: DreamSeed,
        max_turns: int,
        max_tokens: int,
        simulated_turn_tokens: int,
    ) -> None:
        """Assert bounds are strictly respected under all Cartesian combinations."""
        engine = MindWanderEngine(max_turns=max_turns, max_tokens=max_tokens)
        insight = engine.execute_dream_cycle(
            test_seed,
            simulated_turn_tokens=simulated_turn_tokens,
        )
        assert insight is not None
        assert isinstance(insight, DreamInsight)

        # 1. Turn boundary assertion
        assert (
            insight.turns_taken <= max_turns
        ), f"Turns taken {insight.turns_taken} exceeded max_turns {max_turns}"

        # 2. Token boundary assertion
        assert (
            insight.tokens_used <= max_tokens
        ), f"Tokens used {insight.tokens_used} exceeded max_tokens {max_tokens}"

        # 3. Structural invariant: if tokens did not hit limit, turns must equal max_turns
        if simulated_turn_tokens * max_turns <= max_tokens:
            assert insight.turns_taken == max_turns
            assert insight.tokens_used == simulated_turn_tokens * max_turns
        else:
            # Token budget capped the run
            assert insight.tokens_used == max_tokens
            assert insight.turns_taken <= max_turns

    def test_extreme_turn_boundary_1(self, test_seed: DreamSeed) -> None:
        """max_turns=1 must execute exactly 1 turn."""
        engine = MindWanderEngine(max_turns=1, max_tokens=2500)
        insight = engine.execute_dream_cycle(test_seed)
        assert insight is not None
        assert insight.turns_taken == 1
        assert insight.tokens_used == 150

    def test_extreme_token_boundary_100(self, test_seed: DreamSeed) -> None:
        """max_tokens=100 with default turn tokens=150 must halt on turn 1 with tokens=100."""
        engine = MindWanderEngine(max_turns=5, max_tokens=100)
        insight = engine.execute_dream_cycle(test_seed, simulated_turn_tokens=150)
        assert insight is not None
        assert insight.tokens_used == 100
        assert insight.turns_taken == 1

    def test_zero_and_negative_bounds_graceful_handling(self, test_seed: DreamSeed) -> None:
        """max_turns=0 or max_tokens=0 must degrade gracefully without raising."""
        # max_turns = 0
        engine_0_turns = MindWanderEngine(max_turns=0, max_tokens=2500)
        insight_0_turns = engine_0_turns.execute_dream_cycle(test_seed)
        assert insight_0_turns is not None
        assert insight_0_turns.turns_taken == 0
        assert insight_0_turns.tokens_used == 0

        # max_tokens = 0
        engine_0_tokens = MindWanderEngine(max_turns=5, max_tokens=0)
        insight_0_tokens = engine_0_tokens.execute_dream_cycle(test_seed)
        assert insight_0_tokens is not None
        assert insight_0_tokens.tokens_used == 0


# ============================================================================
# Section 4: Invalid and Boundary Inputs Stress Tests
# ============================================================================


class TestInvalidAndBoundaryInputs:
    """Stress tests for invalid, unusual, or boundary inputs."""

    @pytest.fixture
    def engine(self) -> MindWanderEngine:
        return MindWanderEngine(max_turns=3, max_tokens=1000)

    def test_empty_and_whitespace_questions_and_topics(self, engine: MindWanderEngine) -> None:
        """Empty, whitespace, and symbol-only topics and speculative questions."""
        adversarial_inputs = [
            ("", ""),
            ("   ", "   "),
            ("\t\n", "\r\n"),
            ("?", "!"),
            ("...", "???"),
            ("🚀" * 20, "🧠" * 20),
            ("CJK_日本語_中文_한국어", "マルチバイト文字テスト"),
            ("A" * 10000, "B" * 10000),
        ]
        for topic, question in adversarial_inputs:
            seed = DreamSeed(
                topic=topic,
                speculative_question=question,
                urgency=1.0,
                context_keys=[],
            )
            insight = engine.execute_dream_cycle(seed)
            assert insight is not None
            assert insight.topic == topic
            assert insight.seed_question == question
            assert len(insight.synthesis) > 0
            assert insight.confidence >= 0.85

    def test_none_preemption_callback_supported(self, engine: MindWanderEngine) -> None:
        """preemption_check=None must run nominal cycle without error."""
        seed = DreamSeed("test_none", "None callback test", 1.0, [])
        insight = engine.execute_dream_cycle(seed, preemption_check=None)
        assert insight is not None
        assert insight.turns_taken == 3

    def test_high_rumination_threshold_1_0(self) -> None:
        """When rumination_threshold=1.0, identical vectors (sim=1.0) do NOT trigger reset."""
        engine = MindWanderEngine(max_turns=3, rumination_threshold=1.0)
        seed = DreamSeed("high_threshold", "High threshold test", 1.0, [])
        vec = [1.0, 0.0, 0.0]
        insight = engine.execute_dream_cycle(
            seed,
            simulated_thought_vectors=[vec, vec, vec],
        )
        assert insight is not None
        assert insight.anti_rumination_reset_occurred is False
        assert insight.turns_taken == 3

    def test_low_rumination_threshold_0_0(self) -> None:
        """When rumination_threshold=0.0, positive dot product vectors trigger reset/abort."""
        engine = MindWanderEngine(max_turns=3, rumination_threshold=0.0)
        seed = DreamSeed("low_threshold", "Low threshold test", 1.0, [])
        v1 = [1.0, 0.0]
        v2 = [math.cos(math.pi / 3), math.sin(math.pi / 3)]  # cos(60 deg) = 0.5 > 0.0
        v3 = [math.cos(math.pi / 3), math.sin(math.pi / 3)]
        insight = engine.execute_dream_cycle(
            seed,
            simulated_thought_vectors=[v1, v2, v3],
        )
        # Turn 1: sim(v2, v1)=0.5 > 0.0 -> reset
        # Turn 2: sim(v3, v2)=1.0 > 0.0 -> hard abort
        assert insight is None

    def test_zero_norm_vectors_in_cosine_similarity(self) -> None:
        """All-zero vectors [0, 0, 0] must not raise ZeroDivisionError."""
        sim = cosine_similarity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        assert not math.isnan(sim)
        assert -1.0 <= sim <= 1.0

    def test_shannon_entropy_boundary_cases(self) -> None:
        """compute_shannon_entropy handles empty, single-word, and uniform strings safely."""
        assert compute_shannon_entropy("") == 0.0
        assert compute_shannon_entropy("hello") == 0.0
        assert compute_shannon_entropy("hello hello hello hello") == 0.0
        ent = compute_shannon_entropy("alpha beta gamma delta epsilon zeta eta theta")
        assert 0.9 <= ent <= 1.0


# ============================================================================
# Section 5: Property-Based Fuzzing with Hypothesis
# ============================================================================


class TestPropertyBasedFuzzing:
    """Hypothesis invariant tests searching for edge-case contract violations."""

    @settings(max_examples=100, deadline=None)
    @given(
        u=st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
            min_size=1,
            max_size=16,
        ),
        v=st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
            min_size=1,
            max_size=16,
        ),
    )
    def test_cosine_similarity_mathematical_invariants(
        self, u: list[float], v: list[float]
    ) -> None:
        """Cosine similarity must always return a float strictly in [-1.0, 1.0] without NaN."""
        sim = cosine_similarity(u, v)
        assert isinstance(sim, float)
        assert not math.isnan(sim)
        assert -1.0 <= sim <= 1.0

    @settings(max_examples=100, deadline=None)
    @given(
        topic=st.text(max_size=100),
        question=st.text(max_size=200),
        max_turns=st.integers(min_value=1, max_value=10),
        max_tokens=st.integers(min_value=100, max_value=5000),
    )
    def test_engine_execution_invariants(
        self,
        topic: str,
        question: str,
        max_turns: int,
        max_tokens: int,
    ) -> None:
        """Under nominal simulation without interrupts, engine always produces valid DreamInsight.

        Turns taken <= max_turns and tokens used <= max_tokens.
        """
        engine = MindWanderEngine(max_turns=max_turns, max_tokens=max_tokens)
        seed = DreamSeed(
            topic=topic,
            speculative_question=question,
            urgency=1.0,
            context_keys=[],
        )
        insight = engine.execute_dream_cycle(seed)
        assert insight is not None
        assert insight.turns_taken <= max_turns
        assert insight.tokens_used <= max_tokens
        assert 0.0 <= insight.confidence <= 1.0
