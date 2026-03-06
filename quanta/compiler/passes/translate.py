"""
quanta.compiler.passes.translate — Target gate set transpilation.


  - IBM Heron:   {CX, RZ, SX, X}
  - Google Sycamore: {CZ, √iSWAP, PhasedXZ}
  - Quantinuum:  {RZZ, Rz, U1q}


Example:
    >>> from quanta.compiler.passes.translate import TranslateToTarget
    >>> pass_ = TranslateToTarget(target_gates={"CX", "RZ", "SX", "X"})
    >>> optimized = pass_.run(dag)
"""

from __future__ import annotations

import numpy as np

from quanta.core.circuit import CircuitBuilder
from quanta.core.types import Instruction
from quanta.dag.dag_circuit import DAGCircuit
from quanta.dag.node import OpNode

# ── Public API ──
__all__ = ["TranslateToTarget", "GATE_SETS"]

GATE_SETS: dict[str, frozenset[str]] = {
    "ibm":        frozenset({"CX", "RZ", "SX", "X", "I"}),
    "google":     frozenset({"CZ", "RZ", "RX", "RY", "I"}),
    "quantinuum": frozenset({"CX", "RZ", "RY", "RX", "I"}),
    "universal":  frozenset({"CX", "RZ", "RY", "H", "X", "I"}),
}

# Her kural: (gate_name, qubit_offset, params)
_DECOMPOSITION_RULES: dict[str, list[tuple[str, list[int], tuple[float, ...]]]] = {
    # H = RZ(π) · RY(π/2) = RZ(π) RY(π/2)
    "H": [
        ("RZ", [0], (np.pi,)),
        ("RY", [0], (np.pi / 2,)),
    ],
    # S = RZ(π/2)
    "S": [
        ("RZ", [0], (np.pi / 2,)),
    ],
    # T = RZ(π/4)
    "T": [
        ("RZ", [0], (np.pi / 4,)),
    ],
    # Z = RZ(π)
    "Z": [
        ("RZ", [0], (np.pi,)),
    ],
    # Y = RY(π) (global faza kadar)
    "Y": [
        ("RY", [0], (np.pi,)),
    ],
    # X = RX(π) (global faza kadar)
    "X": [
        ("RX", [0], (np.pi,)),
    ],
    # SWAP = 3 CX
    "SWAP": [
        ("CX", [0, 1], ()),
        ("CX", [1, 0], ()),
        ("CX", [0, 1], ()),
    ],
    # CZ = H·CX·H (hedef qubit'e)
    "CZ": [
        ("H", [1], ()),
        ("CX", [0, 1], ()),
        ("H", [1], ()),
    ],
    # CY = S†·CX·S (hedef qubit'e)
    "CY": [
        ("RZ", [1], (-np.pi / 2,)),
        ("CX", [0, 1], ()),
        ("RZ", [1], (np.pi / 2,)),
    ],
    "CCX": [
        ("H", [2], ()),
        ("CX", [1, 2], ()),
        ("RZ", [2], (-np.pi / 4,)),
        ("CX", [0, 2], ()),
        ("RZ", [2], (np.pi / 4,)),
        ("CX", [1, 2], ()),
        ("RZ", [2], (-np.pi / 4,)),
        ("CX", [0, 2], ()),
        ("RZ", [1], (np.pi / 4,)),
        ("RZ", [2], (np.pi / 4,)),
        ("H", [2], ()),
        ("CX", [0, 1], ()),
        ("RZ", [0], (np.pi / 4,)),
        ("RZ", [1], (-np.pi / 4,)),
        ("CX", [0, 1], ()),
    ],
}

class TranslateToTarget:
    """Circuityi hedef gate set'e transpile eder.


    Args:
        target_gates: Hedef gate set ismi ("ibm", "google") veya set.
    """

    name = "TranslateToTarget"

    def __init__(self, target_gates: str | set[str] = "universal") -> None:
        if isinstance(target_gates, str):
            if target_gates not in GATE_SETS:
                raise ValueError(
                    f"Bilinmeyen gate set: {target_gates}. "
                    f"Bilinen: {list(GATE_SETS.keys())}"
                )
            self._target = GATE_SETS[target_gates]
        else:
            self._target = frozenset(target_gates)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Translates gates to target set.

        Args:

        Returns:
        """
        ops = dag.op_nodes()
        needs_translation = any(op.gate_name not in self._target for op in ops)

        if not needs_translation:
            return dag  # Zaten hedef sette

        builder = CircuitBuilder(dag.num_qubits)
        builder.measurement = dag.measurement

        for op in ops:
            if op.gate_name in self._target:
                builder.record(Instruction(op.gate_name, op.qubits, op.params))
            elif op.gate_name in ("RX", "RY", "RZ") and op.gate_name in self._target:
                builder.record(Instruction(op.gate_name, op.qubits, op.params))
            elif op.gate_name in _DECOMPOSITION_RULES:
                # Decompose
                self._decompose(builder, op)
            else:
                builder.record(Instruction(op.gate_name, op.qubits, op.params))

        return DAGCircuit.from_builder(builder)

    def _decompose(self, builder: CircuitBuilder, op: OpNode) -> None:
        rules = _DECOMPOSITION_RULES[op.gate_name]

        for gate_name, qubit_offsets, params in rules:
            actual_qubits = tuple(op.qubits[i] for i in qubit_offsets)

            if gate_name in self._target:
                builder.record(Instruction(gate_name, actual_qubits, params))
            elif gate_name in _DECOMPOSITION_RULES:
                sub_op = OpNode(0, gate_name=gate_name, qubits=actual_qubits, params=params)
                self._decompose(builder, sub_op)
            else:
                builder.record(Instruction(gate_name, actual_qubits, params))

    def __repr__(self) -> str:
        return f"TranslateToTarget(gates={sorted(self._target)})"
