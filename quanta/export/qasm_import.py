"""
quanta.export.qasm_import -- QASM 2.0/3.0 to Quanta DAG import pipeline.

Parses OpenQASM files and converts them to Quanta's internal DAG
representation. Essential for QASMBench integration and Benchpress.

Supports:
  - QASM 2.0 (qelib1.inc gates) and QASM 3.0 (stdgates.inc)
  - Standard gates: h, x, y, z, s, t, cx, cz, swap, ccx, rx, ry, rz
  - Qubit/bit register declarations
  - Parametric gates with float params
  - Measurement statements

Example:
    >>> from quanta.export.qasm_import import from_qasm
    >>> dag = from_qasm('''
    ...     OPENQASM 2.0;
    ...     include "qelib1.inc";
    ...     qreg q[2];
    ...     creg c[2];
    ...     h q[0];
    ...     cx q[0],q[1];
    ...     measure q -> c;
    ... ''')
    >>> print(dag.gate_count())
    2
"""

from __future__ import annotations

import re

from quanta.core.circuit import CircuitBuilder
from quanta.core.types import Instruction, MeasureSpec
from quanta.dag.dag_circuit import DAGCircuit

__all__ = ["from_qasm", "from_qasm_file"]


# QASM gate name -> Quanta gate name
_QASM_TO_QUANTA: dict[str, str] = {
    "h": "H", "x": "X", "y": "Y", "z": "Z",
    "s": "S", "t": "T", "sdg": "S", "tdg": "T",
    "cx": "CX", "cnot": "CX", "CX": "CX",
    "cz": "CZ", "cy": "CY",
    "swap": "SWAP",
    "ccx": "CCX", "toffoli": "CCX",
    "rx": "RX", "ry": "RY", "rz": "RZ",
    "u1": "RZ", "p": "RZ",  # u1 and p are equivalent to RZ
    "id": "I", "i": "I",
}


def from_qasm(qasm_str: str) -> DAGCircuit:
    """Parses QASM 2.0/3.0 string and returns a Quanta DAGCircuit.

    Args:
        qasm_str: OpenQASM string (2.0 or 3.0).

    Returns:
        DAGCircuit ready for simulation or compilation.
    """
    lines = qasm_str.strip().split("\n")

    num_qubits = 0
    num_cbits = 0
    instructions: list[Instruction] = []
    has_measure = False
    qubit_offset: dict[str, int] = {}  # register_name -> starting index

    for line in lines:
        line = line.strip()
        if not line or line.startswith("//"):
            continue

        # Remove trailing semicolons and comments
        line = line.split("//")[0].strip().rstrip(";").strip()

        # Skip headers
        if line.startswith(("OPENQASM", "include")):
            continue

        # QASM 2.0: qreg q[N];
        m = re.match(r"qreg\s+(\w+)\s*\[\s*(\d+)\s*\]", line)
        if m:
            reg_name, size = m.group(1), int(m.group(2))
            qubit_offset[reg_name] = num_qubits
            num_qubits += size
            continue

        # QASM 3.0: qubit[N] q;
        m = re.match(r"qubit\s*\[\s*(\d+)\s*\]\s+(\w+)", line)
        if m:
            size, reg_name = int(m.group(1)), m.group(2)
            qubit_offset[reg_name] = num_qubits
            num_qubits += size
            continue

        # Classical registers
        m = re.match(r"(?:creg|bit)\s*\[?\s*(\d+)\s*\]?\s*(\w*)", line)
        if m and (line.startswith("creg") or line.startswith("bit")):
            num_cbits = max(num_cbits, int(m.group(1)))
            continue

        # Measurement (QASM 2.0: measure q -> c; QASM 3.0: c = measure q;)
        if "measure" in line:
            has_measure = True
            continue

        # Barrier (skip)
        if line.startswith("barrier"):
            continue

        # Gate application
        parsed = _parse_gate_line(line, qubit_offset)
        if parsed:
            instructions.append(parsed)

    if num_qubits == 0:
        num_qubits = _infer_qubits(instructions)

    # Build DAG
    builder = CircuitBuilder(num_qubits)
    for inst in instructions:
        builder.record(inst)

    if has_measure:
        builder.measurement = MeasureSpec(qubits=tuple(range(num_qubits)))

    return DAGCircuit.from_builder(builder)


def from_qasm_file(filepath: str) -> DAGCircuit:
    """Loads a QASM file and returns a DAGCircuit."""
    with open(filepath) as f:
        return from_qasm(f.read())


def _parse_gate_line(
    line: str, qubit_offset: dict[str, int]
) -> Instruction | None:
    """Parses a single gate line into an Instruction."""

    # Match: gate_name(params) qubit_args
    m = re.match(
        r"(\w+)\s*(?:\(([^)]*)\))?\s+(.*)",
        line,
    )
    if not m:
        return None

    gate_raw = m.group(1).lower()
    param_str = m.group(2)
    qubit_str = m.group(3)

    # Map gate name
    quanta_gate = _QASM_TO_QUANTA.get(gate_raw)
    if quanta_gate is None:
        # Unknown gate — skip silently
        return None

    if quanta_gate == "I":
        return None  # Identity gate — no-op

    # Parse parameters
    params: tuple[float, ...] = ()
    if param_str:
        try:
            # Handle pi expressions
            param_parts = param_str.split(",")
            parsed_params = []
            for p in param_parts:
                p = p.strip()
                p = p.replace("pi", str(3.141592653589793))
                parsed_params.append(float(eval(p)))
            params = tuple(parsed_params)
        except Exception:
            params = ()

    # Parse qubits
    qubit_matches = re.findall(r"(\w+)\[(\d+)\]", qubit_str)
    if qubit_matches:
        qubits = []
        for reg_name, idx in qubit_matches:
            offset = qubit_offset.get(reg_name, 0)
            qubits.append(offset + int(idx))
        return Instruction(
            gate_name=quanta_gate,
            qubits=tuple(qubits),
            params=params,
        )

    return None


def _infer_qubits(instructions: list[Instruction]) -> int:
    """Infers qubit count from instructions."""
    if not instructions:
        return 1
    max_q = max(max(inst.qubits) for inst in instructions if inst.qubits)
    return max_q + 1
