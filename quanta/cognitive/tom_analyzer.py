"""Theory of Mind (ToM) and Sociolinguistic Conversational Analyzer.

Extracts latent developer needs, conversational cadence shifts, and epistemic doubt
to compute sociological urgency S_ToM in [0.2, 5.0] and synthesize high-utility DreamSeeds.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any

from quanta.cognitive.arbiter import (
    CognitivePanelScore,
    NeurobiologicalEvaluation,
    PsychiatricEvaluation,
    SociologicalEvaluation,
)

__all__ = [
    "DreamSeed",
    "TheoryOfMindAnalyzer",
    "_clamp01",
    "_safe_float",
    "_safe_int",
]


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts value to float with default fallback."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely converts value to int with default fallback."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return int(f)
    except (ValueError, TypeError, OverflowError):
        return default


def _clamp01(val: Any, default: float = 0.0) -> float:
    """Safely converts value to float clamped in [0.0, 1.0] with NaN/Inf handling."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return max(0.0, min(1.0, f))
    except (ValueError, TypeError):
        return default


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

    CATASTROPHIC_WORDS: set[str] = {
        "disaster",
        "catastrophe",
        "ruined",
        "panic",
        "emergency",
        "crisis",
        "catastrophic",
        "fatal",
        "impossible",
    }

    MISALIGNMENT_WORDS: set[str] = {
        "stop doing that",
        "why did you",
        "annoying",
        "listen to me",
        "bad agent",
        "shut up",
        "wrong",
        "terrible",
        "useless",
        "stupid",
    }

    COLLABORATIVE_WORDS: set[str] = {
        "thanks",
        "thank you",
        "please",
        "great",
        "nice",
        "let's",
        "team",
        "together",
        "awesome",
        "good job",
        "appreciate",
    }

    FRICTION_WORDS: set[str] = {
        "conflict",
        "desync",
        "handshake failed",
        "misunderstanding",
        "coordination error",
        "deadlock",
    }

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
            test_failures = _safe_float(project_state.get("test_failures", 0))
            if test_failures > 0:
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

        # Interdisciplinary DreamSeed synthesis
        # 1. Synaptic homeostasis downscaling when turns > 20
        if len(messages) > 20:
            seeds.append(
                DreamSeed(
                    topic="synaptic_homeostasis",
                    speculative_question=(
                        "How can we downscale synaptic weights and consolidate memory "
                        "after prolonged conversation?"
                    ),
                    urgency=min(5.0, max(urgency, 2.5)),
                    context_keys=["synaptic_homeostasis", "quanta.cognitive.consolidation"],
                )
            )

        # 2. Anti-rumination when circular topics repeat
        topic_counts: dict[str, int] = {}
        for m in messages:
            c = str(m.get("content", "")).lower()
            for tok in self.DOMAIN_TOPICS:
                if tok in c:
                    topic_counts[tok] = topic_counts.get(tok, 0) + 1

        circular_detected = any(cnt >= 3 for cnt in topic_counts.values())
        if project_state and (
            _safe_int(project_state.get("repetition_count", 0)) >= 3
            or project_state.get("rumination", False)
        ):
            circular_detected = True

        if circular_detected:
            seeds.append(
                DreamSeed(
                    topic="anti_rumination",
                    speculative_question=(
                        "How can the agent break out of repetitive circular deliberation "
                        "and restore cognitive flexibility?"
                    ),
                    urgency=min(5.0, max(urgency, 2.8)),
                    context_keys=["anti_rumination", "quanta.cognitive.arbiter"],
                )
            )

        # 3. User fatigue reduction when terse user turns dominate
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs and messages:
            user_msgs = messages

        terse_count = sum(
            1 for m in user_msgs if 0 < len(str(m.get("content", "")).strip().split()) <= 3
        )
        terse_ratio = terse_count / len(user_msgs) if user_msgs else 0.0
        if (len(user_msgs) >= 2 and terse_ratio >= 0.5) or (
            project_state and project_state.get("user_fatigue", False)
        ):
            seeds.append(
                DreamSeed(
                    topic="user_fatigue_reduction",
                    speculative_question=(
                        "How can response density be streamlined to minimize human "
                        "cognitive fatigue?"
                    ),
                    urgency=min(5.0, max(urgency, 2.2)),
                    context_keys=["user_fatigue_reduction", "quanta.cognitive.tom_analyzer"],
                )
            )

        return urgency, seeds

    def evaluate_cognitive_panel(
        self,
        messages: list[dict[str, Any]],
        project_state: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> CognitivePanelScore:
        """Computes multi-disciplinary cognitive panel metrics grounded in conversational signals.

        Evaluates neurobiological (synaptic saturation, metabolic energy expenditure, sleep
        consolidation affinity), psychiatric (rumination risk, perseveration penalty, threat
        distortion), and sociological (social misalignment, user fatigue impact, coordination
        friction) dimensions.

        Args:
            messages: List of conversation turn dicts with content and optional metadata.
            project_state: Optional dictionary containing project health signals.
            context: Optional dictionary containing runtime execution context.

        Returns:
            CognitivePanelScore with neurobiological, psychiatric, and sociological evaluations.
        """
        # --- 1. Neurobiological Evaluation ---
        base_sat = len(messages) / 30.0
        if context:
            ctx_density = _clamp01(context.get("buffer_density") or context.get("density"))
            ctx_turns = _safe_float(context.get("turn_count", 0)) / 30.0
            base_sat = max(base_sat, ctx_density, ctx_turns)

        tf = _safe_float(project_state.get("test_failures", 0)) if project_state else 0.0
        fail_stress = min(0.30, tf * 0.10)
        synaptic_saturation = _clamp01(base_sat + fail_stress)

        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        base_exp = total_chars / 4000.0
        if context:
            ctx_exp = _clamp01(context.get("energy_expenditure") or context.get("metabolic_cost"))
            base_exp = max(base_exp, ctx_exp)
        tool_count = sum(
            1 for m in messages if "tool" in m or "tool_call" in m or "tool_calls" in m
        )
        base_exp += tool_count * 0.05
        energy_expenditure = _clamp01(base_exp)

        base_affinity = 0.50 * synaptic_saturation
        if len(messages) > 20:
            base_affinity = max(
                base_affinity, 0.70 + 0.30 * min(1.0, (len(messages) - 20) / 10.0)
            )
        if project_state:
            if (
                project_state.get("quiescent")
                or project_state.get("quiescence")
                or project_state.get("resolved")
            ):
                base_affinity = max(base_affinity, 0.85)
            elif _safe_float(project_state.get("test_failures", 0)) == 0.0 and len(messages) >= 5:
                base_affinity = max(base_affinity, 0.60)
        if context and ("sleep_affinity" in context or "sleep_consolidation_affinity" in context):
            ctx_sleep = _clamp01(
                context.get("sleep_affinity") or context.get("sleep_consolidation_affinity")
            )
            base_affinity = max(base_affinity, ctx_sleep)
        sleep_consolidation_affinity = _clamp01(base_affinity)

        neurobiology = NeurobiologicalEvaluation(
            synaptic_saturation=synaptic_saturation,
            energy_expenditure=energy_expenditure,
            sleep_consolidation_affinity=sleep_consolidation_affinity,
        )

        # --- 2. Psychiatric Evaluation ---
        topic_counts: dict[str, int] = {}
        for m in messages:
            c = str(m.get("content", "")).lower()
            for tok in self.DOMAIN_TOPICS:
                if tok in c:
                    topic_counts[tok] = topic_counts.get(tok, 0) + 1

        max_reps = max(topic_counts.values()) if topic_counts else 0
        base_rum = 0.30 + 0.20 * (max_reps - 2) if max_reps >= 3 else 0.0
        if context:
            raw_c_reps = context.get("repetition_count")
            c_reps = _safe_int(
                raw_c_reps if raw_c_reps is not None else context.get("circular_depth", 0)
            )
            if c_reps >= 3:
                base_rum = max(base_rum, 0.30 + 0.20 * (c_reps - 2))
            if "rumination_risk" in context:
                base_rum = max(base_rum, _clamp01(context.get("rumination_risk")))
        if project_state:
            p_reps = _safe_int(project_state.get("repetition_count", 0))
            if p_reps >= 3:
                base_rum = max(base_rum, 0.30 + 0.20 * (p_reps - 2))
            if project_state.get("rumination"):
                base_rum = max(base_rum, 0.75)
        rumination_risk = _clamp01(base_rum)

        fail_terms = (
            "fail",
            "fails",
            "failing",
            "error",
            "traceback",
            "exception",
            "broken",
            "bug",
        )
        fail_turn_count = sum(
            1 for m in messages if any(w in str(m.get("content", "")).lower() for w in fail_terms)
        )
        if tf > 1 and fail_turn_count >= 2:
            base_pers = min(1.0, 0.35 * (fail_turn_count - 1))
        elif fail_turn_count >= 3:
            base_pers = min(1.0, 0.30 * (fail_turn_count - 2))
        else:
            base_pers = 0.0
        if context:
            if "perseveration" in context or "perseveration_penalty" in context:
                base_pers = max(
                    base_pers,
                    _clamp01(
                        context.get("perseveration_penalty") or context.get("perseveration")
                    ),
                )
            if _safe_int(context.get("repeated_tool_failures", 0)) >= 2:
                base_pers = max(base_pers, 0.65)
        perseveration_penalty = _clamp01(base_pers)

        alarm_count = sum(
            1
            for m in messages
            if any(w in str(m.get("content", "")).lower() for w in self.CATASTROPHIC_WORDS)
        )
        is_blocker = bool(
            project_state.get("blocker", False) or project_state.get("critical", False)
        ) if project_state else False
        if alarm_count > 0:
            if tf == 0 and not is_blocker:
                base_dist = min(1.0, 0.45 + 0.25 * alarm_count)
            elif tf <= 1:
                base_dist = min(1.0, 0.30 + 0.15 * alarm_count)
            else:
                base_dist = min(0.25, 0.08 * alarm_count)
        else:
            base_dist = 0.0
        if context and "threat_distortion" in context:
            base_dist = max(base_dist, _clamp01(context.get("threat_distortion")))
        threat_distortion = _clamp01(base_dist)

        psychiatry = PsychiatricEvaluation(
            rumination_risk=rumination_risk,
            perseveration_penalty=perseveration_penalty,
            threat_distortion=threat_distortion,
        )

        # --- 3. Sociological Evaluation ---
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs and messages:
            user_msgs = messages

        terse_count = sum(
            1 for m in user_msgs if 0 < len(str(m.get("content", "")).strip().split()) <= 3
        )
        terse_ratio = terse_count / len(user_msgs) if user_msgs else 0.0
        if len(user_msgs) >= 2 and terse_ratio >= 0.5:
            base_fatigue = 0.55 + 0.40 * (terse_ratio - 0.5)
        elif terse_ratio > 0.0:
            base_fatigue = 0.25 * terse_ratio
        else:
            base_fatigue = 0.0
        base_fatigue += min(0.25, len(messages) / 40.0)
        if context and ("user_fatigue" in context or "user_fatigue_impact" in context):
            base_fatigue = max(
                base_fatigue,
                _clamp01(
                    context.get("user_fatigue_impact") or context.get("user_fatigue")
                ),
            )
        if project_state and project_state.get("user_fatigue"):
            base_fatigue = max(base_fatigue, 0.70)
        user_fatigue_impact = _clamp01(base_fatigue)

        misalign_count = sum(
            1
            for m in messages
            if any(w in str(m.get("content", "")).lower() for w in self.MISALIGNMENT_WORDS)
        )
        collab_count = sum(
            1
            for m in messages
            if any(w in str(m.get("content", "")).lower() for w in self.COLLABORATIVE_WORDS)
        )
        base_misalign = max(0.0, misalign_count * 0.40 - collab_count * 0.15)
        if context and "social_misalignment" in context:
            base_misalign = max(base_misalign, _clamp01(context.get("social_misalignment")))
        social_misalignment = _clamp01(base_misalign)

        fric_count = sum(
            1
            for m in messages
            if any(w in str(m.get("content", "")).lower() for w in self.FRICTION_WORDS)
        )
        base_fric = fric_count * 0.35
        if context and "coordination_friction" in context:
            base_fric = max(base_fric, _clamp01(context.get("coordination_friction")))
        if project_state and "coordination_friction" in project_state:
            base_fric = max(base_fric, _clamp01(project_state.get("coordination_friction")))
        coordination_friction = _clamp01(base_fric)

        sociology = SociologicalEvaluation(
            social_misalignment=social_misalignment,
            user_fatigue_impact=user_fatigue_impact,
            coordination_friction=coordination_friction,
        )

        return CognitivePanelScore(
            neurobiology=neurobiology,
            psychiatry=psychiatry,
            sociology=sociology,
        )

    def analyze_interdisciplinary(
        self,
        messages: list[dict[str, Any]],
        project_state: dict[str, Any] | None = None,
    ) -> tuple[float, list[DreamSeed], CognitivePanelScore]:
        """Analyzes conversational history through an interdisciplinary cognitive lens.

        Args:
            messages: List of conversation turn dicts with content and optional metadata.
            project_state: Optional dictionary containing project health signals.

        Returns:
            Tuple of (tom_urgency, dream_seeds, cognitive_panel_score).
        """
        urgency, seeds = self.analyze_conversation(messages, project_state=project_state)
        panel = self.evaluate_cognitive_panel(messages, project_state=project_state)
        return urgency, seeds, panel
