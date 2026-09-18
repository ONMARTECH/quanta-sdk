"""Theory of Mind (ToM) and Sociolinguistic Conversational Analyzer.

Extracts latent developer needs, conversational cadence shifts, and epistemic doubt
to compute sociological urgency S_ToM in [0.2, 5.0] and synthesize high-utility DreamSeeds.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DreamSeed:
    """Structured cognitive seed tuple driving DMN incubation cycles.

    Attributes:
        topic: Focus domain or subsystem identifier.
        speculative_question: Dialectical prompt for DMN/Zeno deliberation.
        urgency: Sociological urgency multiplier S_ToM associated with this seed.
        context_keys: Subsystem context keys or module paths for working memory retrieval.
        project_path: Optional filesystem path to the target workspace project.
        project_summary: Optional summarized architecture / problem context.
        tech_stack: Optional list of identified tools and frameworks.
    """

    topic: str
    speculative_question: str
    urgency: float
    context_keys: list[str]
    project_path: str = ""
    project_summary: str = ""
    tech_stack: list[str] = field(default_factory=list)


class TheoryOfMindAnalyzer:
    """Sociological Theory of Mind conversational analyzer.

    Grounded in Gricean conversational implicature, sociolinguistic hesitation heuristics,
    and reflective silence latency metrics to extract latent technical anxieties and
    synthesize structured DreamSeed tuples.
    """

    HESITATION_WORDS: set[str] = {
        "maybe",
        "perhaps",
        "confused",
        "stuck",
        "wonder",
        "unsure",
        "doubt",
        "not sure",
        "fails",
        "fail",
        "might",
        "seems",
        "unexpected",
        "weird",
    }

    URGENT_WORDS: set[str] = {
        "deadline",
        "deploy",
        "release",
        "critical",
        "broken",
        "bug",
        "failing",
        "blocker",
        "urgent",
        "prod",
        "staging",
        "ship",
    }

    DOMAIN_TOPICS: tuple[str, ...] = (
        "qos",
        "darwin",
        "poisson",
        "memory",
        "swr",
        "hook",
        "preemption",
        "dialectic",
        "consolidation",
        "zeno",
        "dmn",
        "thermal",
        "quiescence",
    )

    def analyze_conversation(
        self,
        messages: list[dict[str, Any]],
        project_state: dict[str, Any] | None = None,
    ) -> tuple[float, list[DreamSeed]]:
        """Analyze conversational history and project state to extract urgency and DreamSeeds.

        Args:
            messages: List of conversation turn dicts containing at least 'content'
                and optionally 'timestamp' or 'role'.
            project_state: Optional dictionary containing project health signals
                (e.g., 'test_failures', 'milestone_urgency', 'blocker').

        Returns:
            Tuple of (tom_urgency, dream_seeds), where tom_urgency is clamped in [0.2, 5.0].
        """
        if not messages:
            return 1.0, []

        urgency = 1.0
        hesitation_detected = False
        urgent_detected = False
        topics: set[str] = set()
        timestamps: list[float] = []

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

            for token in self.DOMAIN_TOPICS:
                if token in content:
                    topics.add(token)

            ts = m.get("timestamp")
            if ts is not None:
                with contextlib.suppress(ValueError, TypeError):
                    timestamps.append(float(ts))

        # Cadence & latency analysis if timestamps are present
        if len(timestamps) >= 2:
            latencies = [
                timestamps[i] - timestamps[i - 1]
                for i in range(1, len(timestamps))
                if timestamps[i] >= timestamps[i - 1]
            ]
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                # Prolonged reflective silence indicates cognitive impasse
                if avg_latency > 60.0:
                    urgency *= 1.2

        if hesitation_detected:
            urgency *= 1.8
        if urgent_detected:
            urgency *= 2.0

        if project_state:
            test_failures = project_state.get("test_failures", 0)
            if isinstance(test_failures, (int, float)) and test_failures > 0:
                urgency *= 1.5
            if project_state.get("milestone_urgency", False):
                urgency *= 1.3
            if project_state.get("blocker", False) or project_state.get("deadline", False):
                urgency *= 1.4

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
