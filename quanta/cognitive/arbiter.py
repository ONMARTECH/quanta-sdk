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
        options: list[str] | list[dict[str, Any]],
        criteria: list[str] | None = None,
        exploration_drive: float = 0.2,
        workspace: str | None = None,
        log_telemetry: bool = True,
    ) -> dict[str, Any]:
        """Evaluates decision options against a goal and criteria using Zeno/Anti-Zeno attention.

        Supports arbitrary N-option branching and structured options with multi-criteria/impact analysis.

        Args:
            goal: Target objective or strategic constraint.
            options: List of proposed choices (as strings or structured dicts with impact/pros/cons).
            criteria: Optional list of specific constraints, trade-off targets, or impact criteria.
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

        effective_goal = goal
        if criteria:
            valid_criteria = [c.strip() for c in criteria if isinstance(c, str) and c.strip()]
            if valid_criteria:
                effective_goal = f"{goal} [Kriterler/Kısıtlar: {'; '.join(valid_criteria)}]"

        goal_c = text_to_statevector(effective_goal, dim=self.dim)
        goal_vec = goal_c.real

        parsed_options: list[tuple[str, str, dict[str, Any] | None]] = []
        option_vecs = []
        for opt in options:
            if isinstance(opt, dict):
                name = str(opt.get("name") or opt.get("title") or opt.get("option") or "Option")
                parts = [f"{k}: {v}" for k, v in opt.items() if k not in ("name", "title", "option")]
                desc = " | ".join(parts)
                full_repr = f"{name} ({desc})" if desc else name
                parsed_options.append((name, full_repr, opt))
            else:
                opt_str = str(opt)
                parsed_options.append((opt_str, opt_str, None))

        for _, full_repr, _ in parsed_options:
            opt_c = text_to_statevector(f"{effective_goal} -> {full_repr}", dim=self.dim)
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
        for i, (name, full_repr, details) in enumerate(parsed_options):
            item: dict[str, Any] = {
                "option": name,
                "score": round(float(probs[i]), 4),
                "raw_alignment": round(float(alignments[i].item()), 4),
            }
            if details is not None:
                item["details"] = details
            elif full_repr != name:
                item["full_representation"] = full_repr
            ranking.append(item)

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
                goal=effective_goal,
                options=[p[0] for p in parsed_options],
                winner=recommended["option"],
                confidence=recommended["score"],
                zeno_pinning_factor=zeno_pin,
                anti_zeno_kickback=explore_mag,
                regime=regime,
                latency_ms=latency_ms,
                ranking=ranking,
                workspace=workspace,
            )

        result: dict[str, Any] = {
            "recommended_option": recommended["option"],
            "confidence": recommended["score"],
            "zeno_pinning_factor": round(zeno_pin, 4),
            "anti_zeno_kickback": round(explore_mag, 4),
            "latency_ms": round(latency_ms, 2),
            "regime": regime,
            "ranked_options": ranking,
        }
        if "details" in recommended:
            result["recommended_details"] = recommended["details"]

        return result
