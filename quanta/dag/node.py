"""
quanta.dag.node — DAG node types.


    InputNode(q0) → OpNode(H, q0) → OpNode(CX, q0,q1) → OutputNode(q0)
    InputNode(q1) ─────────────────→ OpNode(CX, q0,q1) → OutputNode(q1)
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Public API ──
__all__ = ["DAGNode", "InputNode", "OpNode", "OutputNode"]

@dataclass(frozen=True, slots=True)
class DAGNode:
    """Base class for DAG nodes.

    Attributes:
    """

    node_id: int

@dataclass(frozen=True, slots=True)
class InputNode(DAGNode):
    """Qubit input node. Start point of a DAG.

    Attributes:
    """

    qubit: int

    def __repr__(self) -> str:
        return f"In(q{self.qubit})"

@dataclass(frozen=True, slots=True)
class OpNode(DAGNode):
    """Gate operation node. Represents an instruction.

    Attributes:
    """

    gate_name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()

    def __repr__(self) -> str:
        q_str = ",".join(f"q{q}" for q in self.qubits)
        return f"Op({self.gate_name}[{q_str}])"

@dataclass(frozen=True, slots=True)
class OutputNode(DAGNode):
    """Qubit output node. End point of a DAG.

    Attributes:
    """

    qubit: int

    def __repr__(self) -> str:
        return f"Out(q{self.qubit})"
