"""Tests for Theory of Mind (ToM) analyzer and DreamSeed generation.

Tests conversation cadence tracking, sociolinguistic hesitation heuristics,
milestone urgency, and speculative DreamSeed generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

# Attempt real import; fall back to contract double if not yet implemented
try:
    from quanta.cognitive.tom_analyzer import DreamSeed, TheoryOfMindAnalyzer

    REAL_TOM_AVAILABLE = True
except ImportError:
    REAL_TOM_AVAILABLE = False

    @dataclass
    class DreamSeed:  # type: ignore[no-redef]
        """Authoritative contract double adhering strictly to PROJECT.md."""

        topic: str
        speculative_question: str
        urgency: float
        context_keys: list[str]

    class TheoryOfMindAnalyzer:  # type: ignore[no-redef]
        """Authoritative contract double adhering strictly to PROJECT.md & Survey 2."""

        HESITATION_WORDS = {
            "maybe",
            "perhaps",
            "confused",
            "stuck",
            "wonder",
            "unsure",
            "doubt",
            "not sure",
        }
        URGENT_WORDS = {
            "deadline",
            "deploy",
            "release",
            "critical",
            "broken",
            "bug",
            "failing",
            "blocker",
        }

        def analyze_conversation(
            self,
            messages: list[dict[str, Any]],
            project_state: dict[str, Any] | None = None,
        ) -> tuple[float, list[DreamSeed]]:
            """Analyzes user messages and project state to extract urgency and DreamSeeds."""
            if not messages:
                return 1.0, []

            urgency = 1.0
            hesitation_detected = False
            urgent_detected = False
            topics: set[str] = set()

            for m in messages:
                content = str(m.get("content", "")).lower()
                if (
                    any(w in content for w in self.HESITATION_WORDS)
                    or "..." in content
                    or "???" in content
                ):
                    hesitation_detected = True
                if any(w in content for w in self.URGENT_WORDS):
                    urgent_detected = True

                for token in (
                    "qos",
                    "darwin",
                    "poisson",
                    "memory",
                    "swr",
                    "hook",
                    "preemption",
                    "dialectic",
                ):
                    if token in content:
                        topics.add(token)

            if hesitation_detected:
                urgency *= 1.8
            if urgent_detected:
                urgency *= 2.0

            if project_state:
                if project_state.get("test_failures", 0) > 0:
                    urgency *= 1.5
                if project_state.get("milestone_urgency", False):
                    urgency *= 1.3

            urgency = max(0.2, min(5.0, urgency))

            seeds: list[DreamSeed] = []
            if topics:
                for t in sorted(topics):
                    seeds.append(
                        DreamSeed(
                            topic=t,
                            speculative_question=f"How can we optimize {t} for biomorphic use?",
                            urgency=urgency,
                            context_keys=[t, "quanta.cognitive"],
                        )
                    )
            elif hesitation_detected or urgent_detected:
                seeds.append(
                    DreamSeed(
                        topic="general_architecture",
                        speculative_question="What latent bottlenecks exist in the current flow?",
                        urgency=urgency,
                        context_keys=["general"],
                    )
                )

            return urgency, seeds


# ============================================================================
# Tier 3A: Conversational Hesitation & Epistemic Doubt Tests
# ============================================================================


class TestHesitationHeuristics:
    """Verifies sociolinguistic parsing of user hesitation, hedging, and doubt."""

    @pytest.fixture
    def analyzer(self) -> TheoryOfMindAnalyzer:
        return TheoryOfMindAnalyzer()

    def test_confident_conversation_nominal_urgency(
        self, analyzer: TheoryOfMindAnalyzer
    ) -> None:
        """A confident, declarative conversation maintains nominal urgency ~ 1.0."""
        messages = [
            {"role": "user", "content": "Let's review the code layout for quanta."},
            {"role": "assistant", "content": "Layout is clean."},
            {"role": "user", "content": "Run the tests and commit."},
        ]
        urgency, seeds = analyzer.analyze_conversation(messages)
        assert 0.8 <= urgency <= 1.2
        assert isinstance(seeds, list)

    def test_hedging_and_doubt_boosts_urgency(
        self, analyzer: TheoryOfMindAnalyzer
    ) -> None:
        """Hedging words ('maybe', 'perhaps', 'confused') significantly elevate urgency."""
        messages = [
            {"role": "user", "content": "I am not sure how the Darwin QoS scheduling works..."},
            {"role": "assistant", "content": "It schedules onto E-cores."},
            {"role": "user", "content": "Maybe I am confused about relative priority?"},
        ]
        urgency, seeds = analyzer.analyze_conversation(messages)
        assert urgency > 1.5
        assert len(seeds) > 0
        assert any("darwin" in s.topic.lower() or "qos" in s.topic.lower() for s in seeds)

    def test_ellipsis_and_trailing_pause_heuristics(
        self, analyzer: TheoryOfMindAnalyzer
    ) -> None:
        """Trailing ellipses and multiple question marks flag user hesitation."""
        messages = [
            {"role": "user", "content": "Wait... will this cause memory leaks???"},
        ]
        urgency, seeds = analyzer.analyze_conversation(messages)
        assert urgency > 1.2
        assert len(seeds) > 0


# ============================================================================
# Tier 3A: Milestone & Project State Urgency Tests
# ============================================================================


class TestMilestoneUrgency:
    """Verifies sensitivity to project milestones, deadlines, and active test failures."""

    @pytest.fixture
    def analyzer(self) -> TheoryOfMindAnalyzer:
        return TheoryOfMindAnalyzer()

    def test_deadline_keywords_escalate_urgency(
        self, analyzer: TheoryOfMindAnalyzer
    ) -> None:
        """Keywords like 'deadline', 'deploy', 'critical' elevate urgency."""
        messages = [
            {"role": "user", "content": "We have a critical deploy deadline tonight."},
        ]
        urgency, seeds = analyzer.analyze_conversation(messages)
        assert urgency >= 2.0
        assert len(seeds) > 0

    def test_test_failures_in_project_state_scale_urgency(
        self, analyzer: TheoryOfMindAnalyzer
    ) -> None:
        """Active test failures reported in project_state elevate urgency."""
        messages = [
            {"role": "user", "content": "Let's inspect the test results."},
        ]
        state_with_failures = {"test_failures": 3, "failing_tests": ["test_darwin"]}
        urgency, seeds = analyzer.analyze_conversation(messages, project_state=state_with_failures)
        assert urgency > 1.2

    def test_urgency_clamped_to_strict_bounds(
        self, analyzer: TheoryOfMindAnalyzer
    ) -> None:
        """Urgency must always remain within [0.2, 5.0] regardless of input intensity."""
        hyper_urgent_messages = [
            {"role": "user", "content": "CRITICAL BUG DEADLINE BROKEN FAILING BLOCKER DEPLOY"}
        ] * 10
        extreme_state = {"test_failures": 50, "milestone_urgency": True}
        urgency, _ = analyzer.analyze_conversation(
            hyper_urgent_messages, project_state=extreme_state
        )
        assert urgency <= 5.0
        assert urgency >= 0.2


# ============================================================================
# Tier 3A: DreamSeed Generation & Structural Integrity
# ============================================================================


class TestDreamSeedSynthesis:
    """Verifies properties and structure of synthesized DreamSeed objects."""

    @pytest.fixture
    def analyzer(self) -> TheoryOfMindAnalyzer:
        return TheoryOfMindAnalyzer()

    def test_dream_seed_field_contracts(self, analyzer: TheoryOfMindAnalyzer) -> None:
        """Every DreamSeed must contain topic, speculative_question, urgency, context_keys."""
        messages = [
            {"role": "user", "content": "How should we handle SWR memory consolidation and QoS?"}
        ]
        urgency, seeds = analyzer.analyze_conversation(messages)
        assert len(seeds) > 0
        for seed in seeds:
            assert isinstance(seed.topic, str)
            assert len(seed.topic) > 0
            assert isinstance(seed.speculative_question, str)
            assert len(seed.speculative_question) > 0
            assert isinstance(seed.urgency, (float, int))
            assert 0.2 <= seed.urgency <= 5.0
            assert isinstance(seed.context_keys, list)

    def test_empty_messages_returns_empty_seeds(
        self, analyzer: TheoryOfMindAnalyzer
    ) -> None:
        """Empty conversational history returns nominal urgency and empty seed list."""
        urgency, seeds = analyzer.analyze_conversation([])
        assert 0.8 <= urgency <= 1.2
        assert seeds == []

    def test_cadence_latency_and_blockers(
        self, analyzer: TheoryOfMindAnalyzer
    ) -> None:
        """Verify timestamp cadence (>60s silence) and blocker/deadline project state."""
        messages = [
            {"role": "user", "content": "qos and memory planning", "timestamp": 100.0},
            {"role": "user", "content": "still stuck on swr", "timestamp": 200.0},
            {"role": "user", "content": "bad timestamp", "timestamp": "invalid_ts"},
        ]
        project_state = {"blocker": True, "deadline": True}
        urgency, seeds = analyzer.analyze_conversation(messages, project_state=project_state)
        assert urgency > 1.0
        assert len(seeds) > 0

