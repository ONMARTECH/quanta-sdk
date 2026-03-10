"""
quanta.export.qasm — OpenQASM 3.0 circuit export.


Example:
    >>> from quanta.export.qasm import to_qasm
    >>> print(to_qasm(bell_state))
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[2] c;
    h q[0];
    cx q[0], q[1];
    c = measure q;
"""

from __future__ import annotations

from quanta.core.circuit import CircuitDefinition
from quanta.dag.dag_circuit import DAGCircuit
from quanta.dag.node import OpNode

# ── Public API ──
__all__ = ["to_qasm"]

_QASM_GATE_MAP: dict[str, str] = {
    "H": "h",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "S": "s",
    "T": "t",
    "I": "id",
    "SDG": "sdg",
    "TDG": "tdg",
    "SX": "sx",
    "SXdg": "sxdg",
    "P": "p",
    "U": "u",
    "CX": "cx",
    "CZ": "cz",
    "CY": "cy",
    "SWAP": "swap",
    "CCX": "ccx",
    "RCCX": "rccx",
    "RC3X": "rc3x",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "RXX": "rxx",
    "RZZ": "rzz",
}

def to_qasm(circuit: CircuitDefinition) -> str:
    """Converts circuit to OpenQASM 3.0 format.

    Args:
        circuit: Circuit defined with @circuit.

    Returns:
        OpenQASM 3.0 string'i.

    Raises:
    """
    builder = circuit.build()
    dag = DAGCircuit.from_builder(builder)

    lines: list[str] = []

    lines.append("OPENQASM 3.0;")
    lines.append('include "stdgates.inc";')
    lines.append("")

    lines.append(f"qubit[{dag.num_qubits}] q;")

    has_measure = builder.measurement is not None
    if has_measure:
        measured = (
            builder.measurement.qubits
            if builder.measurement.qubits
            else tuple(range(dag.num_qubits))
        )
        lines.append(f"bit[{len(measured)}] c;")

    lines.append("")

    for op in dag.op_nodes():
        qasm_line = _op_to_qasm(op, dag.num_qubits)
        lines.append(qasm_line)

    if has_measure:
        lines.append("")
        measured = (
            builder.measurement.qubits
            if builder.measurement.qubits
            else tuple(range(dag.num_qubits))
        )
        for i, q in enumerate(measured):
            lines.append(f"c[{i}] = measure q[{q}];")

    lines.append("")
    return "\n".join(lines)

def _op_to_qasm(op: OpNode, num_qubits: int) -> str:
    qasm_name = _QASM_GATE_MAP.get(op.gate_name)

    if qasm_name is None:
        raise ValueError(
            f"Desteklenenler: {list(_QASM_GATE_MAP.keys())}"
        )

    qubit_args = ", ".join(f"q[{q}]" for q in op.qubits)

    if op.params:
        param_str = ", ".join(f"{p:.6f}" for p in op.params)
        return f"{qasm_name}({param_str}) {qubit_args};"

    return f"{qasm_name} {qubit_args};"

def from_qasm_gates(qasm_str: str) -> list[tuple[str, tuple[int, ...]]]:
    """Extracts gate list from QASM string (simple parser).


    Args:
        qasm_str: OpenQASM 3.0 string'i.

    Returns:
    """
    reverse_map = {v: k for k, v in _QASM_GATE_MAP.items()}
    gates: list[tuple[str, tuple[int, ...]]] = []

    for line in qasm_str.strip().split("\n"):
        line = line.strip().rstrip(";")

        if not line or line.startswith(("OPENQASM", "include", "qubit", "bit", "//")):
            continue
        if line.startswith("c[") or line.startswith("measure"):
            continue

        parts = line.split()
        if not parts:
            continue

        gate_name_raw = parts[0].split("(")[0]  # clean parametric gates
        quanta_name = reverse_map.get(gate_name_raw, gate_name_raw.upper())

        qubit_str = " ".join(parts[1:])
        import re
        qubit_indices = re.findall(r"q\[(\d+)\]", qubit_str)
        qubits = tuple(int(idx) for idx in qubit_indices)

        gates.append((quanta_name, qubits))

    return gates
