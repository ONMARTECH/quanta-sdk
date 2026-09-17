"""Empirical adversarial stress test suite for TheoryOfMindAnalyzer.

Challenges:
1. 500+ generated conversation patterns:
   - Empty conversations
   - Single-word messages
   - Giant transcripts (10,000+ tokens)
   - Rapid bursts (< 0.1s latency)
   - Prolonged silence (> 300s latency)
   - Adversarial timestamps (decreasing, negative, NaN, inf, invalid types)
   - Adversarial project state matrices
2. Unconditional urgency clamping: S_ToM in [0.2, 5.0].
3. DreamSeed structural integrity: non-empty topic, non-empty speculative_question,
   urgency in [0.2, 5.0], non-empty context_keys list of non-empty strings.
4. Property-based fuzzing via Hypothesis.
"""

from __future__ import annotations

import math
import random
import string
import time
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quanta.cognitive.tom_analyzer import DreamSeed, TheoryOfMindAnalyzer


# Helper validation oracle
def assert_valid_analysis_result(
    urgency: Any,
    seeds: Any,
    context_label: str = "",
) -> None:
    """Assert mathematical and structural integrity contracts on analyzer output."""
    # 1. Type and clamping validation for urgency
    assert isinstance(
        urgency, (float, int)
    ), f"[{context_label}] Urgency must be float or int, got {type(urgency)}"
    assert not math.isnan(urgency), f"[{context_label}] Urgency must not be NaN"
    assert not math.isinf(urgency), f"[{context_label}] Urgency must not be infinite"
    assert (
        0.2 <= urgency <= 5.0
    ), f"[{context_label}] Urgency {urgency} outside strict bounds [0.2, 5.0]"

    # 2. DreamSeed structural validation
    assert isinstance(
        seeds, list
    ), f"[{context_label}] Seeds must be a list, got {type(seeds)}"
    for idx, seed in enumerate(seeds):
        assert isinstance(
            seed, DreamSeed
        ), f"[{context_label}] Element {idx} is not a DreamSeed: {type(seed)}"
        assert isinstance(
            seed.topic, str
        ), f"[{context_label}] Seed {idx} topic not str: {type(seed.topic)}"
        assert (
            len(seed.topic.strip()) > 0
        ), f"[{context_label}] Seed {idx} topic is empty"
        assert isinstance(
            seed.speculative_question, str
        ), f"[{context_label}] Seed {idx} question not str: {type(seed.speculative_question)}"
        assert (
            len(seed.speculative_question.strip()) > 0
        ), f"[{context_label}] Seed {idx} question is empty"
        assert isinstance(
            seed.urgency, (float, int)
        ), f"[{context_label}] Seed {idx} urgency not float/int: {type(seed.urgency)}"
        assert not math.isnan(
            seed.urgency
        ), f"[{context_label}] Seed {idx} urgency is NaN"
        assert not math.isinf(
            seed.urgency
        ), f"[{context_label}] Seed {idx} urgency is infinite"
        assert (
            0.2 <= seed.urgency <= 5.0
        ), f"[{context_label}] Seed {idx} urgency {seed.urgency} outside [0.2, 5.0]"
        assert isinstance(
            seed.context_keys, list
        ), f"[{context_label}] Seed {idx} context_keys not list: {type(seed.context_keys)}"
        assert (
            len(seed.context_keys) > 0
        ), f"[{context_label}] Seed {idx} context_keys is empty"
        for k_idx, key in enumerate(seed.context_keys):
            assert isinstance(
                key, str
            ), f"[{context_label}] Seed {idx} key {k_idx} not str: {type(key)}"
            assert (
                len(key.strip()) > 0
            ), f"[{context_label}] Seed {idx} key {k_idx} is empty"


# ============================================================================
# Category 1: Empty and Edge-Case Conversations (50 patterns)
# ============================================================================


class TestEmptyAndEdgeConversations:
    """Evaluates analyzer behavior on empty, null, and degenerate message structures."""

    @pytest.fixture
    def analyzer(self) -> TheoryOfMindAnalyzer:
        return TheoryOfMindAnalyzer()

    def test_empty_conversations_suite(self, analyzer: TheoryOfMindAnalyzer) -> None:
        """50 variations of empty and degenerate message collections."""
        patterns: list[list[dict[str, Any]]] = [
            [],  # Strict empty list
        ]
        # Degenerate dict messages with empty or missing content
        for _ in range(49):
            k = random.randint(1, 5)
            msgs: list[dict[str, Any]] = []
            for _ in range(k):
                variant: dict[str, Any] = random.choice(
                    [
                        {},
                        {"content": ""},
                        {"content": "   "},
                        {"content": None},
                        {"content": 0},
                        {"content": False},
                        {"role": "user"},
                        {"timestamp": None},
                    ]
                )
                msgs.append(variant)
            patterns.append(msgs)

        assert len(patterns) == 50
        for i, pat in enumerate(patterns):
            urgency, seeds = analyzer.analyze_conversation(pat)
            assert_valid_analysis_result(urgency, seeds, f"EmptyPattern_{i}")


# ============================================================================
# Category 2: Single Words & Minimal Turns (100 patterns)
# ============================================================================


class TestSingleWordConversations:
    """Evaluates single-word inputs: hesitation, urgent, domain, and random vocabulary."""

    @pytest.fixture
    def analyzer(self) -> TheoryOfMindAnalyzer:
        return TheoryOfMindAnalyzer()

    def test_single_word_patterns_suite(self, analyzer: TheoryOfMindAnalyzer) -> None:
        """100 single-word conversation patterns."""
        patterns: list[list[dict[str, Any]]] = []

        # Hesitation words
        for w in sorted(analyzer.HESITATION_WORDS):
            patterns.append([{"content": w}])

        # Urgent words
        for w in sorted(analyzer.URGENT_WORDS):
            patterns.append([{"content": w}])

        # Domain topics
        for t in analyzer.DOMAIN_TOPICS:
            patterns.append([{"content": t}])

        # Punctuation markers
        patterns.append([{"content": "..."}])
        patterns.append([{"content": "???"}])
        patterns.append([{"content": "!?!"}])
        patterns.append([{"content": "......."}])

        # Fill remaining up to 100 with random ascii, unicode, numbers, noise
        while len(patterns) < 100:
            rand_word = "".join(
                random.choices(string.ascii_letters + string.digits, k=random.randint(1, 15))
            )
            patterns.append([{"content": rand_word}])

        assert len(patterns) >= 100
        for i, pat in enumerate(patterns[:100]):
            urgency, seeds = analyzer.analyze_conversation(pat)
            assert_valid_analysis_result(urgency, seeds, f"SingleWordPattern_{i}")


# ============================================================================
# Category 3: Giant Transcripts (10,000+ Tokens) (50 patterns)
# ============================================================================


class TestGiantTranscripts:
    """Evaluates performance, memory stability, and correctness on 10k-20k token texts."""

    @pytest.fixture
    def analyzer(self) -> TheoryOfMindAnalyzer:
        return TheoryOfMindAnalyzer()

    def test_giant_transcripts_suite(self, analyzer: TheoryOfMindAnalyzer) -> None:
        """50 giant conversation patterns with 10,000+ words/tokens."""
        vocabulary = [
            "the", "quantum", "darwin", "qos", "poisson", "maybe", "confused",
            "deploy", "urgent", "broken", "swr", "hippocampus", "consolidation",
            "memory", "state", "system", "kernel", "efficiency", "zeno", "preemption",
            "variable", "function", "execution", "thread", "resonance", "dialectic",
        ]

        patterns: list[list[dict[str, Any]]] = []
        for p in range(50):
            token_count = random.randint(10000, 15000)
            # Create either 1 huge message or multiple large turns totaling 10k+ tokens
            num_turns = random.choice([1, 5, 20])
            tokens_per_turn = token_count // num_turns

            msgs: list[dict[str, Any]] = []
            for t in range(num_turns):
                words = [random.choice(vocabulary) for _ in range(tokens_per_turn)]
                # Inject specific trigger tokens deterministically into some patterns
                if p % 2 == 0:
                    words.insert(100, "critical")
                    words.insert(500, "darwin")
                if p % 3 == 0:
                    words.insert(250, "unsure")
                    words.insert(800, "qos")
                content = " ".join(words)
                msgs.append({"role": "user", "content": content, "timestamp": float(t * 10)})
            patterns.append(msgs)

        assert len(patterns) == 50
        for i, pat in enumerate(patterns):
            t0 = time.perf_counter()
            urgency, seeds = analyzer.analyze_conversation(pat)
            elapsed = time.perf_counter() - t0
            assert_valid_analysis_result(urgency, seeds, f"GiantTranscript_{i}")
            # Scalability check: must process 10k tokens in < 250ms
            assert elapsed < 0.25, f"Giant transcript {i} took too long: {elapsed:.3f}s"


# ============================================================================
# Category 4: Rapid Bursts (< 0.1s Latency) (100 patterns)
# ============================================================================


class TestRapidBursts:
    """Evaluates rapid message firing with timestamps < 0.1s apart (or identical)."""

    @pytest.fixture
    def analyzer(self) -> TheoryOfMindAnalyzer:
        return TheoryOfMindAnalyzer()

    def test_rapid_bursts_suite(self, analyzer: TheoryOfMindAnalyzer) -> None:
        """100 rapid burst patterns."""
        patterns: list[list[dict[str, Any]]] = []

        for p in range(100):
            num_messages = random.randint(2, 25)
            start_time = 1000.0 + p * 10
            # Latency delta < 0.1s, down to 0.00001s or 0.0s
            delta = random.choice([0.0, 0.0001, 0.001, 0.01, 0.05, 0.099])
            msgs: list[dict[str, Any]] = []
            curr_time = start_time
            for m_idx in range(num_messages):
                msgs.append(
                    {
                        "role": "user",
                        "content": f"rapid burst turn {m_idx} qos preemption",
                        "timestamp": curr_time,
                    }
                )
                curr_time += delta
            patterns.append(msgs)

        assert len(patterns) == 100
        for i, pat in enumerate(patterns):
            urgency, seeds = analyzer.analyze_conversation(pat)
            assert_valid_analysis_result(urgency, seeds, f"RapidBurst_{i}")


# ============================================================================
# Category 5: Prolonged Silence (> 300s Latency) (100 patterns)
# ============================================================================


class TestProlongedSilence:
    """Evaluates conversational stagnation and prolonged silence (> 300s)."""

    @pytest.fixture
    def analyzer(self) -> TheoryOfMindAnalyzer:
        return TheoryOfMindAnalyzer()

    def test_prolonged_silence_suite(self, analyzer: TheoryOfMindAnalyzer) -> None:
        """100 patterns with silence gaps > 300s."""
        patterns: list[list[dict[str, Any]]] = []

        for _p in range(100):
            num_messages = random.randint(2, 10)
            start_time = 5000.0
            gap = random.uniform(301.0, 86400.0)  # Between 5 mins and 24 hours
            msgs: list[dict[str, Any]] = []
            curr_time = start_time
            for m_idx in range(num_messages):
                msgs.append(
                    {
                        "role": "user",
                        "content": f"prolonged silence turn {m_idx} stuck with swr memory",
                        "timestamp": curr_time,
                    }
                )
                curr_time += gap
            patterns.append(msgs)

        assert len(patterns) == 100
        for i, pat in enumerate(patterns):
            urgency, seeds = analyzer.analyze_conversation(pat)
            assert_valid_analysis_result(urgency, seeds, f"ProlongedSilence_{i}")
            # Silence > 60s triggers the 1.2x cognitive impasse multiplier
            assert urgency >= 1.2, f"Expected impasse boost for pattern {i}"


# ============================================================================
# Category 6: Adversarial Timestamps & Cadence Anomalies (100 patterns)
# ============================================================================


class TestAdversarialTimestamps:
    """Evaluates resilience against non-monotonic, negative, NaN, inf, or malformed timestamps."""

    @pytest.fixture
    def analyzer(self) -> TheoryOfMindAnalyzer:
        return TheoryOfMindAnalyzer()

    def test_adversarial_timestamps_suite(self, analyzer: TheoryOfMindAnalyzer) -> None:
        """100 adversarial timestamp patterns."""
        patterns: list[list[dict[str, Any]]] = []

        # 1. Monotonically decreasing timestamps (time-travel)
        for _p in range(20):
            msgs = [
                {"content": "reverse time", "timestamp": 1000.0 - i * 50.0}
                for i in range(random.randint(2, 8))
            ]
            patterns.append(msgs)

        # 2. Negative timestamps
        for _p in range(20):
            msgs = [
                {"content": "negative time", "timestamp": -500.0 + i * 20.0}
                for i in range(random.randint(2, 8))
            ]
            patterns.append(msgs)

        # 3. String representations, scientific notation, and invalid types
        for _p in range(20):
            msgs = [
                {"content": "str time 1", "timestamp": "1234567.89"},
                {"content": "str time 2", "timestamp": "1.234e6"},
                {"content": "invalid time", "timestamp": "not_a_number"},
                {"content": "none time", "timestamp": None},
                {"content": "dict time", "timestamp": {"nested": 123}},
                {"content": "list time", "timestamp": [1, 2, 3]},
            ]
            patterns.append(msgs)

        # 4. NaN and Infinity float values
        for _p in range(20):
            msgs = [
                {"content": "nan time", "timestamp": float("nan")},
                {"content": "inf time", "timestamp": float("inf")},
                {"content": "-inf time", "timestamp": float("-inf")},
                {"content": "normal time", "timestamp": 100.0},
            ]
            patterns.append(msgs)

        # 5. Jumbled / alternating timestamps
        for _p in range(20):
            msgs = [
                {
                    "content": f"jumbled {i}",
                    "timestamp": random.choice([10.0, 500.0, 5.0, 900.0, 0.0]),
                }
                for i in range(5)
            ]
            patterns.append(msgs)

        assert len(patterns) == 100
        for i, pat in enumerate(patterns):
            urgency, seeds = analyzer.analyze_conversation(pat)
            assert_valid_analysis_result(urgency, seeds, f"AdversarialTimestamp_{i}")


# ============================================================================
# Category 7: Adversarial Project State Fuzzing (50 patterns)
# ============================================================================


class TestAdversarialProjectState:
    """Evaluates resilience against malformed, boundary, or extreme project_state dictionaries."""

    @pytest.fixture
    def analyzer(self) -> TheoryOfMindAnalyzer:
        return TheoryOfMindAnalyzer()

    def test_adversarial_project_state_suite(self, analyzer: TheoryOfMindAnalyzer) -> None:
        """50 adversarial project_state matrices."""
        base_messages = [
            {"role": "user", "content": "Checking project status and darwin qos"}
        ]

        states: list[dict[str, Any] | None] = [
            None,
            {},
            {"test_failures": 0},
            {"test_failures": -10},
            {"test_failures": 1000000},
            {"test_failures": 1e9},
            {"test_failures": float("nan")},
            {"test_failures": float("inf")},
            {"test_failures": "three"},
            {"test_failures": None},
            {"milestone_urgency": True},
            {"milestone_urgency": False},
            {"milestone_urgency": "very_urgent"},
            {"milestone_urgency": [1, 2, 3]},
            {"blocker": True, "deadline": True},
            {"blocker": False, "deadline": False},
            {"blocker": "yes", "deadline": 1},
        ]

        while len(states) < 50:
            rand_state: dict[str, Any] = {
                "test_failures": random.choice(
                    [-5, 0, 1, 9999, 1e12, float("inf"), "err", None]
                ),
                "milestone_urgency": random.choice([True, False, 1, 0, "yes", None, []]),
                "blocker": random.choice([True, False, "critical", None]),
                "deadline": random.choice([True, False, "asap", None]),
                "extra_noise": "random_value",
            }
            states.append(rand_state)

        assert len(states) >= 50
        for i, st_dict in enumerate(states[:50]):
            urgency, seeds = analyzer.analyze_conversation(base_messages, project_state=st_dict)
            assert_valid_analysis_result(urgency, seeds, f"AdversarialProjectState_{i}")


# ============================================================================
# Category 8: Property-Based Testing via Hypothesis (150+ generated patterns)
# ============================================================================


class TestHypothesisPropertyBasedFuzzing:
    """Uses Hypothesis generator to search for arbitrary inputs that could violate contracts."""

    @settings(max_examples=150, deadline=None)
    @given(
        messages=st.lists(
            st.dictionaries(
                keys=st.sampled_from(["content", "role", "timestamp"]),
                values=st.one_of(
                    st.text(max_size=200),
                    st.floats(allow_nan=True, allow_infinity=True),
                    st.integers(min_value=-10000, max_value=10000),
                    st.none(),
                ),
            ),
            max_size=30,
        ),
        project_state=st.one_of(
            st.none(),
            st.dictionaries(
                keys=st.sampled_from(
                    ["test_failures", "milestone_urgency", "blocker", "deadline", "extra"]
                ),
                values=st.one_of(
                    st.integers(min_value=-100, max_value=1000),
                    st.booleans(),
                    st.floats(allow_nan=True, allow_infinity=True),
                    st.text(max_size=30),
                    st.none(),
                ),
            ),
        ),
    )
    def test_hypothesis_invariants(
        self,
        messages: list[dict[str, Any]],
        project_state: dict[str, Any] | None,
    ) -> None:
        """Invariant: analyze_conversation must never raise unhandled exceptions,

        urgency must always be clamped in [0.2, 5.0], and all seeds must be structurally sound.
        """
        analyzer = TheoryOfMindAnalyzer()
        urgency, seeds = analyzer.analyze_conversation(messages, project_state=project_state)
        assert_valid_analysis_result(urgency, seeds, "HypothesisFuzz")


# ============================================================================
# Category 9: Quantitative Clamping Invariance Stress Test
# ============================================================================


class TestClampingInvariance:
    """Verifies that mathematical boundaries [0.2, 5.0] are strictly respected."""

    @pytest.fixture
    def analyzer(self) -> TheoryOfMindAnalyzer:
        return TheoryOfMindAnalyzer()

    def test_maximum_urgency_compounding_does_not_exceed_5(
        self, analyzer: TheoryOfMindAnalyzer
    ) -> None:
        """Every single multiplier active simultaneously still clamps strictly to 5.0."""
        # Multipliers:
        # hesitation: 1.8
        # urgent: 2.0
        # latency > 60: 1.2
        # test_failures > 0: 1.5
        # milestone_urgency: 1.3
        # blocker/deadline: 1.4
        # Total product = 1.0 * 1.8 * 2.0 * 1.2 * 1.5 * 1.3 * 1.4 = 11.7936 -> Clamped to 5.0
        messages = [
            {"content": "stuck maybe", "timestamp": 0.0},
            {"content": "critical blocker bug deadline deploy", "timestamp": 500.0},
        ]
        state = {
            "test_failures": 100,
            "milestone_urgency": True,
            "blocker": True,
            "deadline": True,
        }
        urgency, seeds = analyzer.analyze_conversation(messages, project_state=state)
        assert urgency == 5.0
        assert len(seeds) > 0
        for s in seeds:
            assert s.urgency == 5.0

    def test_nominal_without_triggers_is_valid(
        self, analyzer: TheoryOfMindAnalyzer
    ) -> None:
        """Nominal conversation with zero triggers yields urgency 1.0."""
        messages = [{"content": "hello world", "timestamp": 10.0}]
        urgency, seeds = analyzer.analyze_conversation(messages)
        assert urgency == 1.0
        assert seeds == []
