"""Quanta Cognitive Middleware for Antigravity and Autonomous AI Agents.

Provides seamless, subconscious, and automatic biological memory tracking,
Lindblad phase dephasing attenuation (CSF shielding), active microglial pruning,
and Quantum Zeno decision arbitration without manual overhead.
"""

from __future__ import annotations

from typing import Any

from quanta.cognitive.arbiter import QuantumDecisionArbiter
from quanta.cognitive.memory import CognitiveMemoryManager


class QuantaCognitiveMiddleware:
    """Autonomous biomorphic cognitive middleware for AI agent loops.

    Operates silently in the agent's subconscious:
    - Automatically steps memory decay on every turn.
    - Consciously prunes obsolete/decayed notes via microglial clearance.
    - Formats and injects high-fidelity vital constraints into prompt context.
    - Resolves architectural branching via Quantum Zeno Focus Pinning.
    """

    def __init__(
        self,
        capacity: int = 64,
        dim: int = 16,
        enable_csf_shielding: bool = True,
        auto_prune: bool = True,
        fidelity_threshold: float = 0.70,
        min_salience: float = 0.50,
        default_exploration: float = 0.20,
        seed: int = 42,
    ) -> None:
        self.capacity = capacity
        self.dim = dim
        self.enable_csf_shielding = enable_csf_shielding
        self.auto_prune = auto_prune
        self.fidelity_threshold = fidelity_threshold
        self.min_salience = min_salience
        self.default_exploration = default_exploration

        self.memory = CognitiveMemoryManager(
            capacity=capacity,
            dim=dim,
            enable_csf_shielding=enable_csf_shielding,
        )
        self.arbiter = QuantumDecisionArbiter(dim=dim, num_heads=2, seed=seed)
        self.turn_count = 0

    def record_constraint(
        self,
        key: str,
        content: str,
        salience: float = 2.0,
        category: str = "constraint",
    ) -> int:
        """Records an invariant rule or constraint with dopaminergic protection."""
        return self.memory.record_decision(
            key=key,
            content=content,
            salience=salience,
            category=category,
        )

    def update_decision(
        self,
        key: str,
        content: str,
        salience: float = 2.0,
        category: str = "decision",
    ) -> int:
        """Updates or supersedes an existing decision, eliminating conflicting old memories."""
        return self.memory.update_decision(
            key=key,
            content=content,
            salience=salience,
            category=category,
        )

    def forget(self, key: str) -> bool:
        """Explicitly forgets a specific obsolete or invalidated key."""
        return self.memory.forget(key)

    def on_turn_start(self, dt: float = 1.0) -> list[dict[str, Any]]:
        """Invoked silently at the start of each conversational turn.

        Advances decay, executes microglial pruning, and returns top vital memories.
        """
        self.turn_count += 1
        self.memory.step(dt=dt, auto_prune=self.auto_prune)
        return self.memory.recall_vital_context(top_k=5)

    def get_subconscious_anchor_text(self, top_k: int = 3) -> str | None:
        """Generates a concise, high-salience prompt anchor for subconscious injection.

        Returns None if no high-salience memories exist.
        """
        vitals = self.memory.recall_vital_context(top_k=top_k)
        # Filter for high-retention vital memories (salience >= 1.5, fidelity >= 0.85)
        anchors = [
            v for v in vitals
            if v.get("salience", 1.0) >= 1.5 and v.get("retention_fidelity", 0.0) >= 0.85
        ]
        if not anchors:
            return None

        lines = ["[Bilinçaltı Biyomorfik Kuantum Çıpası - Hayati Kurallar]:"]
        for a in anchors:
            pct = a.get("retention_pct", "100.0%")
            lines.append(f" • [{a['key']}] (Sadakat: {pct}): {a['content']}")
        return "\n".join(lines)

    def arbitrate_decision(
        self,
        goal: str,
        options: list[str],
        exploration_drive: float | None = None,
    ) -> dict[str, Any]:
        """Runs subconscious Quantum Zeno decision arbitration between competing paths.

        Returns recommended option, confidence, and Zeno pinning metrics.
        """
        exp = exploration_drive if exploration_drive is not None else self.default_exploration
        return self.arbiter.arbitrate(goal=goal, options=options, exploration_drive=exp)

    def get_telemetry(self) -> dict[str, Any]:
        """Returns diagnostic telemetry of the subconscious engine."""
        status = self.memory.get_status_summary()
        status["turn_count"] = self.turn_count
        status["auto_prune"] = self.auto_prune
        return status
