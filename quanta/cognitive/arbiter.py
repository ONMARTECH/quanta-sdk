"""Quantum Decision Arbiter using Quantum Zeno & Anti-Zeno Attention dynamics."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from quanta.cognitive.memory import text_to_statevector
from quanta.cognitive.telemetry import record_decision_telemetry
from quanta.torch.brain import QuantumZenoAttention


class QuantumDecisionArbiter:
    """Quantum-inspired 6-qubit decision arbiter (dim=64, 4-head attention) for Antigravity AI agents.

    Employs Quantum Zeno Attention to balance attentional focus pinning
    (Zeno effect: sticking firmly to established goals) and divergent
    exploratory tunneling (Anti-Zeno effect: breaking out of local optima).
    """

    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 4,
        seed: int | None = 42,
        device: torch.device | str | None = "cpu",
    ) -> None:
        self.dim = dim
        if seed is not None:
            torch.manual_seed(seed)
        self.zeno_attention = QuantumZenoAttention(
            dim=dim,
            observation_frequency=12.0,
            dopamine_coupling=0.6,
            num_heads=num_heads,
            device=device,
            dtype=torch.float32,
        )

    def arbitrate(
        self,
        goal: str,
        options: list[str],
        exploration_drive: float = 0.2,
        workspace: str | None = None,
        log_telemetry: bool = True,
    ) -> dict[str, Any]:
        """Evaluates decision options against a goal using Zeno/Anti-Zeno attention.

        Args:
            goal: Target objective or strategic constraint.
            options: List of proposed actions or architectural choices.
            exploration_drive: Float in [0.0, 1.0].
                Lower values (0.0 - 0.3) -> High Zeno pinning (conservative, goal-aligned).
                Higher values (0.7 - 1.0) -> Anti-Zeno tunneling (divergent, exploratory).
            workspace: Optional explicit workspace name for multi-project telemetry audit.
            log_telemetry: When True, logs decision telemetry to central ledger.

        Returns:
            Dictionary containing recommended option, scores, latency, and quantum attention diagnostics.
        """
        if not options:
            raise ValueError("Must provide at least one option to arbitrate.")

        t_start = time.perf_counter()

        goal_c = text_to_statevector(goal, dim=self.dim)
        goal_vec = goal_c.real

        option_vecs = []
        for opt in options:
            opt_c = text_to_statevector(f"{goal} -> {opt}", dim=self.dim)
            option_vecs.append(opt_c.real)

        # Batch tensor: [num_options, dim]
        x = torch.stack(option_vecs, dim=0)

        dopamine_val = max(0.01, 1.0 - exploration_drive)
        dopamine_tensor = torch.tensor(dopamine_val, dtype=torch.float32)

        with torch.no_grad():
            res = self.zeno_attention(x, dopamine=dopamine_tensor, return_diagnostics=True)

        out_tensor = res["output"]  # [num_options, dim]
        zeno_pin = float(torch.mean(res["P_zeno"]).item())
        explore_mag = float(torch.norm(res["h_explore"]).item())

        alignments = F.cosine_similarity(out_tensor, goal_vec.unsqueeze(0), dim=-1)
        probs = F.softmax(alignments * (1.0 + zeno_pin), dim=-1).tolist()

        ranking: list[dict[str, Any]] = []
        for i, opt in enumerate(options):
            ranking.append(
                {
                    "option": opt,
                    "score": round(float(probs[i]), 4),
                    "raw_alignment": round(float(alignments[i].item()), 4),
                }
            )

        ranking.sort(key=lambda item: float(item["score"]), reverse=True)
        recommended = ranking[0]
        regime = (
            "Zeno Pinning (Target Focus)"
            if zeno_pin >= 0.5
            else "Anti-Zeno Tunneling (Exploration)"
        )
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if log_telemetry:
            record_decision_telemetry(
                goal=goal,
                options=options,
                winner=recommended["option"],
                confidence=recommended["score"],
                zeno_pinning_factor=zeno_pin,
                anti_zeno_kickback=explore_mag,
                regime=regime,
                latency_ms=latency_ms,
                ranking=ranking,
                workspace=workspace,
            )

        return {
            "recommended_option": recommended["option"],
            "confidence": recommended["score"],
            "zeno_pinning_factor": round(zeno_pin, 4),
            "anti_zeno_kickback": round(explore_mag, 4),
            "latency_ms": round(latency_ms, 2),
            "regime": regime,
            "ranked_options": ranking,
        }
