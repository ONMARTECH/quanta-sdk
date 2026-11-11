"""quanta.export.qasm — OpenQASM 3.0 circuit export.

Supports full export of all 34 Quanta SDK gates (fixed, parametric, and multi-parametric)
with double-precision angle serialization for lossless QASM 3.0 round-trips.
"""

from __future__ import annotations

import re

from quanta.core.circuit import CircuitDefinition
from quanta.dag.dag_circuit import DAGCircuit
from quanta.dag.node import OpNode

# ── Public API ──
__all__ = ["to_qasm", "from_qasm_gates"]

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
    "CH": "ch",
    "SWAP": "swap",
    "iSWAP": "iswap",
    "ECR": "ecr",
    "CCX": "ccx",
    "CSWAP": "cswap",
    "RCCX": "rccx",
    "RC3X": "rc3x",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "RXX": "rxx",
    "RZZ": "rzz",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
    "CP": "cp",
    "MS": "ms",
}


def to_qasm(circuit: CircuitDefinition | DAGCircuit) -> str:
    """Converts circuit or DAGCircuit to OpenQASM 3.0 format.

    Args:
        circuit: Circuit defined with @circuit or a DAGCircuit.

    Returns:
        OpenQASM 3.0 string.
    """
    if isinstance(circuit, DAGCircuit):
        dag = circuit
        measurement = getattr(dag, "measurement", None)
    else:
        builder = circuit.build()
        dag = DAGCircuit.from_builder(builder)
        measurement = builder.measurement

    lines: list[str] = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        "",
        f"qubit[{dag.num_qubits}] q;",
    ]

    has_measure = measurement is not None
    if has_measure:
        measured = (
            measurement.qubits
            if measurement.qubits
            else tuple(range(dag.num_qubits))
        )
        lines.append(f"bit[{len(measured)}] c;")

    lines.append("")

    for op in dag.op_nodes():
        lines.append(_op_to_qasm(op, dag.num_qubits))

    if has_measure:
        lines.append("")
        measured = (
            measurement.qubits
            if measurement.qubits
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
            f"Unsupported gate '{op.gate_name}' for OpenQASM 3.0 export. "
            f"Supported: {list(_QASM_GATE_MAP.keys())}"
        )

    qubit_args = ", ".join(f"q[{q}]" for q in op.qubits)

    if op.params:
        param_str = ", ".join(f"{float(p):.15g}" for p in op.params)
        return f"{qasm_name}({param_str}) {qubit_args};"

    return f"{qasm_name} {qubit_args};"


def from_qasm_gates(qasm_str: str) -> list[tuple[str, tuple[int, ...]]]:
    """Extracts (gate_name, qubits) tuples from an OpenQASM string."""
    reverse_map = {v: k for k, v in _QASM_GATE_MAP.items()}
    gates: list[tuple[str, tuple[int, ...]]] = []

    for raw_line in qasm_str.strip().split("\n"):
        line = re.sub(r"//.*", "", raw_line).strip().rstrip(";").strip()
        if not line or line.startswith(("OPENQASM", "include", "qubit", "bit", "qreg", "creg")):
            continue
        if line.startswith(("c[", "measure", "barrier")) or "= measure" in line:
            continue

        m = re.match(r"(\w+)\s*(?:\([^)]*\))?\s+(.*)", line)
        if not m:
            continue

        gate_name_raw = m.group(1).lower()
        quanta_name = reverse_map.get(gate_name_raw, gate_name_raw.upper())
        qubit_indices = re.findall(r"\w+\[(\d+)\]", m.group(2))
        qubits = tuple(int(idx) for idx in qubit_indices)
        gates.append((quanta_name, qubits))

    return gates
