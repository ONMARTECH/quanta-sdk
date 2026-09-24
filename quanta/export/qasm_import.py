"""
quanta.export.qasm_import -- QASM 2.0/3.0 to Quanta DAG import pipeline.

Parses OpenQASM files and converts them to Quanta's internal DAG
representation. Supports OpenQASM 3.0 dynamic circuits including
mid-circuit measurement and classical feedforward condition execution.

Supports:
  - QASM 2.0 (qelib1.inc gates) and QASM 3.0 (stdgates.inc)
  - Standard gates: h, x, y, z, s, t, cx, cz, swap, ccx, rx, ry, rz
  - Qubit and bit register declarations (qubit[N], bit[N], qreg, creg)
  - Parametric gates with float params
  - Mid-circuit measurements: c[i] = measure q[j]; and measure q[j] -> c[i];
  - Conditional gate execution: if (c[i] == 1) { gate q; } and if (c == 1) gate q;

Example:
    >>> from quanta.export.qasm_import import from_qasm
    >>> dag = from_qasm('''
    ...     OPENQASM 3.0;
    ...     include "stdgates.inc";
    ...     qubit[2] q;
    ...     bit[1] c;
    ...     h q[0];
    ...     c[0] = measure q[0];
    ...     if (c[0] == 1) { x q[1]; }
    ... ''')
    >>> print(dag.gate_count())
    3
"""

from __future__ import annotations

import math
import re

from quanta.core.circuit import CircuitBuilder
from quanta.core.types import Instruction, MeasureSpec
from quanta.dag.dag_circuit import DAGCircuit
from quanta.dag.node import OpNode, OutputNode

__all__ = ["ConditionalOpNode", "DynamicDAGCircuit", "from_qasm", "from_qasm_file"]


# QASM gate name -> Quanta gate name
_QASM_TO_QUANTA: dict[str, str] = {
    "h": "H", "x": "X", "y": "Y", "z": "Z",
    "s": "S", "t": "T",
    "sdg": "RZ", "tdg": "RZ",  # S† = RZ(-π/2), T† = RZ(-π/4)
    "cx": "CX", "cnot": "CX", "CX": "CX",
    "cz": "CZ", "cy": "CY",
    "swap": "SWAP",
    "ccx": "CCX", "toffoli": "CCX",
    "rx": "RX", "ry": "RY", "rz": "RZ",
    "u1": "RZ", "p": "RZ",  # u1 and p are equivalent to RZ
    "id": "I", "i": "I",
}

# sdg/tdg implicit parameter values (Hermitian conjugates)
_IMPLICIT_PARAMS: dict[str, tuple[float, ...]] = {
    "sdg": (-math.pi / 2,),  # S† = RZ(-π/2)
    "tdg": (-math.pi / 4,),  # T† = RZ(-π/4)
}


class ConditionalOpNode(OpNode):
    """OpNode extended with condition and cbit fields for dynamic circuits."""
    __slots__ = ("condition", "cbit")

    def __init__(
        self,
        node_id: int,
        gate_name: str,
        qubits: tuple[int, ...],
        params: tuple[float, ...] = (),
        condition: tuple[int, int] | None = None,
        cbit: int | None = None,
    ) -> None:
        super().__init__(node_id=node_id, gate_name=gate_name, qubits=qubits, params=params)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "cbit", cbit)


class DynamicDAGCircuit(DAGCircuit):
    """DAGCircuit supporting dynamic circuits with classical conditions and registers."""
    __slots__ = ("num_clbits", "conditions", "instructions")

    def __init__(self, num_qubits: int, num_clbits: int = 0) -> None:
        super().__init__(num_qubits)
        self.num_clbits = num_clbits
        self.conditions: dict[int, tuple[int, int]] = {}
        self.instructions: list[Instruction] = []


def _safe_parse_param(expr: str) -> float:
    """Safely parse a QASM parameter expression using AST.

    Only allows: numbers, pi, +, -, *, /, parentheses.
    Uses ast.parse instead of eval() — zero code execution risk.
    """
    import ast
    import operator

    expr = expr.strip()

    if not re.match(r"^[\d\s.+\-*/()pieE]+$", expr):
        raise ValueError(f"Unsafe QASM parameter expression: {expr!r}")

    safe_expr = expr.replace("pi", str(math.pi))

    _OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval_node(node.operand))
        raise ValueError(f"Disallowed AST node: {type(node).__name__}")

    try:
        tree = ast.parse(safe_expr, mode="eval")
        return float(_eval_node(tree))
    except (SyntaxError, TypeError, ZeroDivisionError) as e:
        raise ValueError(f"Cannot parse QASM parameter: {expr!r}") from e


def _parse_measure_line(
    line: str,
    qubit_offset: dict[str, int],
    cbit_offset: dict[str, int],
) -> tuple[tuple[int, ...], int] | None:
    """Parses single-qubit measurement targeting a classical bit.

    Matches:
      - QASM 3.0: c[i] = measure q[j]
      - QASM 2.0: measure q[j] -> c[i]
    """
    # QASM 3.0: c[i] = measure q[j]
    m3 = re.match(r"(\w+)\[(\d+)\]\s*=\s*measure\s+(\w+)\[(\d+)\]", line)
    if m3:
        c_reg, c_i, q_reg, q_j = m3.groups()
        q_idx = qubit_offset.get(q_reg, 0) + int(q_j)
        c_idx = cbit_offset.get(c_reg, 0) + int(c_i)
        return ((q_idx,), c_idx)

    # QASM 2.0: measure q[j] -> c[i]
    m2 = re.match(r"measure\s+(\w+)\[(\d+)\]\s*->\s*(\w+)\[(\d+)\]", line)
    if m2:
        q_reg, q_j, c_reg, c_i = m2.groups()
        q_idx = qubit_offset.get(q_reg, 0) + int(q_j)
        c_idx = cbit_offset.get(c_reg, 0) + int(c_i)
        return ((q_idx,), c_idx)

    return None


def _split_qasm_statements(qasm: str) -> list[str]:
    """Tokenizes QASM statements delimited by ';', '{', or '}'."""
    qasm_clean = re.sub(r"//.*", "", qasm)
    tokens = re.split(r"([;{}])", qasm_clean)
    stmts: list[str] = []
    curr = ""
    for tok in tokens:
        if tok in (";", "{", "}"):
            full = (curr + tok).strip()
            if full and full != ";":
                stmts.append(full)
            curr = ""
        else:
            curr += tok
    if curr.strip():
        stmts.append(curr.strip())
    return stmts


def from_qasm(qasm_str: str) -> DAGCircuit:
    """Parses QASM 2.0/3.0 string and returns a Quanta DAGCircuit (or DynamicDAGCircuit).

    Args:
        qasm_str: OpenQASM string (2.0 or 3.0).

    Returns:
        DAGCircuit ready for simulation or compilation.
    """
    statements = _split_qasm_statements(qasm_str)

    num_qubits = 0
    num_cbits = 0
    raw_ops: list[tuple] = []
    active_condition: tuple[int, int] | None = None
    has_final_measure = False
    qubit_offset: dict[str, int] = {}
    cbit_offset: dict[str, int] = {}

    for stmt in statements:
        if stmt == "}":
            active_condition = None
            continue

        clean = stmt.rstrip(";").strip()
        if not clean:
            continue

        # Skip headers
        if clean.startswith(("OPENQASM", "include")):
            continue

        # QASM 2.0: qreg q[N];
        m_q2 = re.match(r"qreg\s+(\w+)\s*\[\s*(\d+)\s*\]", clean)
        if m_q2:
            reg_name, size = m_q2.group(1), int(m_q2.group(2))
            qubit_offset[reg_name] = num_qubits
            num_qubits += size
            continue

        # QASM 3.0: qubit[N] q;
        m_q3 = re.match(r"qubit\s*\[\s*(\d+)\s*\]\s+(\w+)", clean)
        if m_q3:
            size, reg_name = int(m_q3.group(1)), m_q3.group(2)
            qubit_offset[reg_name] = num_qubits
            num_qubits += size
            continue

        # Classical registers: creg c[N]; or bit[N] c; or bit c;
        m_c = re.match(r"(?:creg|bit)\s*(?:\[\s*(\d+)\s*\])?\s*(\w+)", clean)
        if m_c and (clean.startswith("creg") or clean.startswith("bit")):
            size_str, reg_name = m_c.group(1), m_c.group(2)
            size = int(size_str) if size_str is not None else 1
            cbit_offset[reg_name] = num_cbits
            num_cbits += size
            continue

        # Barrier (skip)
        if clean.startswith("barrier"):
            continue

        # Conditional branch: if (c[i] == val) { ... } or if (c == val) ...
        m_if = re.match(
            r"if\s*\(\s*(\w+)(?:\[(\d+)\])?\s*==\s*(\d+)\s*\)\s*(\{?)\s*(.*)",
            clean,
        )
        if m_if:
            reg, idx_str, val_str, has_brace, raw_body = m_if.groups()
            c_idx = cbit_offset.get(reg, 0) + (int(idx_str) if idx_str is not None else 0)
            cond = (c_idx, int(val_str))
            has_closing = "}" in raw_body
            active_condition = cond if (has_brace and not has_closing) else None

            body = raw_body.rstrip("};").strip()
            if body:
                pm = _parse_measure_line(body, qubit_offset, cbit_offset)
                if pm:
                    raw_ops.append(("measure", pm[0], pm[1], cond))
                else:
                    pg = _parse_gate_line(body, qubit_offset)
                    if pg:
                        raw_ops.append((
                            "gate",
                            Instruction(
                                gate_name=pg.gate_name,
                                qubits=pg.qubits,
                                params=pg.params,
                                condition=cond,
                            ),
                        ))
            continue

        # If inside a multi-line conditional block
        if active_condition is not None:
            pm = _parse_measure_line(clean, qubit_offset, cbit_offset)
            if pm:
                raw_ops.append(("measure", pm[0], pm[1], active_condition))
            else:
                pg = _parse_gate_line(clean, qubit_offset)
                if pg:
                    raw_ops.append((
                        "gate",
                        Instruction(
                            gate_name=pg.gate_name,
                            qubits=pg.qubits,
                            params=pg.params,
                            condition=active_condition,
                        ),
                    ))
            continue

        # Mid-circuit measurement targeting specific classical bit
        pm = _parse_measure_line(clean, qubit_offset, cbit_offset)
        if pm:
            raw_ops.append(("measure", pm[0], pm[1], None))
            continue

        # Full register measurement (e.g., measure q -> c; or c = measure q;)
        if "measure" in clean:
            raw_ops.append(("final_measure", None))
            continue

        # Regular gate line
        pg = _parse_gate_line(clean, qubit_offset)
        if pg:
            raw_ops.append((
                "gate",
                Instruction(
                    gate_name=pg.gate_name,
                    qubits=pg.qubits,
                    params=pg.params,
                    condition=None,
                ),
            ))

    has_dynamic_features = any(
        (op[0] == "measure" and any(raw_ops[j][0] == "gate" for j in range(idx + 1, len(raw_ops))))
        or (len(op) > 3 and op[3] is not None)
        for idx, op in enumerate(raw_ops)
    )

    # Second pass: classify measurements as mid-circuit vs terminal
    instructions: list[Instruction] = []
    for _i, op in enumerate(raw_ops):
        op_type = op[0]
        if op_type == "gate":
            instructions.append(op[1])
        elif op_type == "final_measure":
            has_final_measure = True
        elif op_type == "measure":
            qubits, cbit, cond = op[1], op[2], op[3]
            if has_dynamic_features:
                instructions.append(
                    Instruction(gate_name="measure", qubits=qubits, condition=cond, cbit=cbit)
                )
            else:
                has_final_measure = True

    if num_qubits == 0:
        num_qubits = _infer_qubits(instructions)

    # Check if this circuit uses dynamic features (conditions or mid-circuit measurements)
    is_dynamic = any(inst.condition is not None or inst.cbit is not None for inst in instructions)

    if not is_dynamic:
        # Static circuit: use standard CircuitBuilder & DAGCircuit for 100% backward compatibility
        builder = CircuitBuilder(num_qubits)
        for inst in instructions:
            builder.record(inst)
        if has_final_measure:
            builder.measurement = MeasureSpec(qubits=tuple(range(num_qubits)))
        return DAGCircuit.from_builder(builder)

    # Dynamic circuit: build DynamicDAGCircuit with causal dependency tracking
    dag = DynamicDAGCircuit(num_qubits, num_clbits=num_cbits)
    dag.instructions = instructions

    last_on_qubit: dict[int, int] = {
        q: dag._input_nodes[q].node_id
        for q in range(num_qubits)
    }
    last_on_cbit: dict[int, int] = {}

    for inst in instructions:
        node_id = dag._node_counter
        op = ConditionalOpNode(
            node_id=node_id,
            gate_name=inst.gate_name,
            qubits=inst.qubits,
            params=inst.params,
            condition=inst.condition,
            cbit=inst.cbit,
        )
        dag._nodes[node_id] = op
        dag._node_counter += 1

        if inst.condition is not None:
            dag.conditions[node_id] = inst.condition
            cbit_idx = inst.condition[0]
            if cbit_idx in last_on_cbit:
                dag._add_edge(last_on_cbit[cbit_idx], node_id)

        if inst.cbit is not None:
            last_on_cbit[inst.cbit] = node_id

        for q in inst.qubits:
            if q in last_on_qubit:
                dag._add_edge(last_on_qubit[q], node_id)
                last_on_qubit[q] = node_id

    for q in range(num_qubits):
        out = dag._add_node(OutputNode(node_id=0, qubit=q))
        dag._output_nodes[q] = out
        dag._add_edge(last_on_qubit[q], out.node_id)

    if has_final_measure:
        dag.measurement = MeasureSpec(qubits=tuple(range(num_qubits)))

    return dag


def from_qasm_file(filepath: str) -> DAGCircuit:
    """Loads a QASM file and returns a DAGCircuit."""
    with open(filepath) as f:
        return from_qasm(f.read())


def _parse_gate_line(
    line: str, qubit_offset: dict[str, int]
) -> Instruction | None:
    """Parses a single gate line into an Instruction."""
    m = re.match(r"(\w+)\s*(?:\(([^)]*)\))?\s+(.*)", line)
    if not m:
        return None

    gate_raw = m.group(1).lower()
    param_str = m.group(2)
    qubit_str = m.group(3)

    quanta_gate = _QASM_TO_QUANTA.get(gate_raw)
    if quanta_gate is None:
        return None

    if quanta_gate == "I":
        return None  # Identity gate — no-op

    params: tuple[float, ...] = ()
    if gate_raw in _IMPLICIT_PARAMS:
        params = _IMPLICIT_PARAMS[gate_raw]
    elif param_str:
        try:
            param_parts = param_str.split(",")
            parsed_params = [_safe_parse_param(p) for p in param_parts]
            params = tuple(parsed_params)
        except (ValueError, Exception):
            params = ()

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
