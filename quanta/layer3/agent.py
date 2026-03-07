"""
quanta.layer3.agent — Multi-agent quantum modeling.


Felsefe:

Example:
    >>> from quanta.layer3.agent import MultiAgentSystem, Agent
    >>> system = MultiAgentSystem([
    ...     Agent("rakip",   choices=["indirim", "fiyat_koru"]),
    ... ])
    >>> result = system.simulate(shots=1024)
    >>> print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quanta.simulator.statevector import StateVectorSimulator

# ── Public API ──
__all__ = ["Agent", "MultiAgentSystem"]

@dataclass
class Agent:
    """Bir karar verici ajan.


    Attributes:
        name: Agent name (unique).
    """

    name: str
    choices: list[str]
    bias: list[float] | None = None

    def __post_init__(self) -> None:
        if len(self.choices) < 2:
            raise ValueError(f"Agent '{self.name}': at least 2 choices required.")
        if len(self.choices) != 2:
            raise ValueError(
                f"Verilen: {len(self.choices)}"
            )

@dataclass
class Interaction:
    """Interaction between two agents.

    Attributes:
    """

    agent_a: str
    agent_b: str
    strength: float

class MultiAgentSystem:
    """Multi-agent quantum decision system.


    Args:
        agents: Ajan listesi.

    Example:
        >>> system = MultiAgentSystem([
        ... ])
        >>> system.interact("A", "B", strength=0.8)
        >>> result = system.simulate(shots=1024)
    """

    def __init__(self, agents: list[Agent]) -> None:
        self._agents = {a.name: a for a in agents}
        self._agent_order = [a.name for a in agents]
        self._interactions: list[Interaction] = []
        self._qubit_map: dict[str, int] = {
            name: i for i, name in enumerate(self._agent_order)
        }

    def interact(self, agent_a: str, agent_b: str, strength: float = 0.5) -> None:
        """Defines interaction between two agents.

        Args:
        """
        if agent_a not in self._agents:
            raise ValueError(f"Bilinmeyen ajan: {agent_a}")
        if agent_b not in self._agents:
            raise ValueError(f"Bilinmeyen ajan: {agent_b}")
        if not 0 <= strength <= 1:
            raise ValueError(f"Strength must be in [0,1]: {strength}")

        self._interactions.append(Interaction(agent_a, agent_b, strength))

    def simulate(
        self, shots: int = 1024, seed: int | None = None
    ) -> AgentResult:
        """Quantum simulates the system.


        Args:
            seed: Random seed.

        Returns:
        """
        n = len(self._agents)
        sim = StateVectorSimulator(n, seed=seed)

        for name, agent in self._agents.items():
            q = self._qubit_map[name]
            if agent.bias:
                theta = 2 * np.arccos(np.sqrt(agent.bias[0]))
                sim.apply("RY", (q,), (theta,))
            else:
                sim.apply("H", (q,))

        for inter in self._interactions:
            qa = self._qubit_map[inter.agent_a]
            qb = self._qubit_map[inter.agent_b]

            if inter.strength > 0.01:
                angle = inter.strength * np.pi / 2
                sim.apply("RY", (qb,), (angle,))
                sim.apply("CX", (qa, qb))
                sim.apply("RY", (qb,), (-angle / 2,))

        counts = sim.sample(shots)

        return AgentResult(
            agents=self._agent_order,
            agent_choices={n: a.choices for n, a in self._agents.items()},
            counts=counts,
            shots=shots,
            interactions=self._interactions,
        )

    def __repr__(self) -> str:
        return (
            f"MultiAgentSystem(agents={self._agent_order}, "
            f"interactions={len(self._interactions)})"
        )

@dataclass
class AgentResult:
    """Multi-agent simulation result.


    Attributes:
    """

    agents: list[str]
    agent_choices: dict[str, list[str]]
    counts: dict[str, int]
    shots: int
    interactions: list[Interaction] = field(default_factory=list)

    def agent_probabilities(self, agent_name: str) -> dict[str, float]:
        """Marginal probabilities for a single agent.

        Args:
            agent_name: Agent name.

        Returns:
        """
        idx = self.agents.index(agent_name)
        choices = self.agent_choices[agent_name]

        probs = {c: 0.0 for c in choices}
        for bitstring, count in self.counts.items():
            bit = int(bitstring[idx])
            probs[choices[bit]] += count / self.shots

        return probs

    def correlation(self, agent_a: str, agent_b: str) -> float:
        """Correlation between two agents [-1, 1].

        """
        ia = self.agents.index(agent_a)
        ib = self.agents.index(agent_b)

        same = 0
        total = 0
        for bitstring, count in self.counts.items():
            if bitstring[ia] == bitstring[ib]:
                same += count
            total += count

        return (2 * same / total) - 1 if total > 0 else 0.0

    def summary(self) -> str:
        """Human-readable result summary."""
        lines = ["=== Multi-Agent Result ==="]

        for name in self.agents:
            probs = self.agent_probabilities(name)
            lines.append(f"╠─── {name} ───")
            for choice, prob in probs.items():
                bar = "█" * int(prob * 25)
                lines.append(f"║   {choice}: {prob:.1%}  {bar}")

        # Korelasyonlar
        if len(self.agents) > 1:
            lines.append("╠═══ Korelasyonlar ═══")
            for inter in self.interactions:
                corr = self.correlation(inter.agent_a, inter.agent_b)
                symbol = "↑↑" if corr > 0.1 else "↑↓" if corr < -0.1 else "──"
                lines.append(
                    f"║   {inter.agent_a} ↔ {inter.agent_b}: "
                    f"{corr:+.2f} {symbol} (strength: {inter.strength:.1f})"
                )

        lines.append("╚" + "═" * 40)
        return "\n".join(lines)
