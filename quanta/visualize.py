"""
quanta.visualize -- ASCII circuit drawing with moment boundaries.

Cirq-inspired circuit display with moment separators and clean gate alignment.

Output style:
    q[0]: ───H───●───M───
                 │
    q[1]: ───────X───M───

Example:
    >>> from quanta import circuit, H, CX, measure
    >>> @circuit(qubits=2)
    ... def bell(q):
    ...     H(q[0])
    ...     CX(q[0], q[1])
    ...     return measure(q)
    >>> print(draw(bell))
"""

from __future__ import annotations

from quanta.core.circuit import CircuitDefinition
from quanta.dag.dag_circuit import DAGCircuit

# -- Public API --
__all__ = ["draw"]

_GATE_SYMBOLS: dict[str, str] = {
    "H": "H", "X": "X", "Y": "Y", "Z": "Z",
    "S": "S", "T": "T", "I": "I",
    "SDG": "S†", "TDG": "T†",
    "SX": "√X", "SXdg": "√X†",
    "CX": "@", "CZ": "@", "CY": "@",
    "SWAP": "x", "CCX": "@", "RCCX": "@", "RC3X": "@",
    "RX": "Rx", "RY": "Ry", "RZ": "Rz",
    "P": "P", "U": "U",
    "RXX": "Rxx", "RZZ": "Rzz",
}

_TARGET_SYMBOLS: dict[str, str] = {
    "CX": "X", "CY": "Y", "CZ": "Z",
    "CCX": "X", "RCCX": "X", "RC3X": "X",
}


def draw(circuit: CircuitDefinition) -> str:
    """Draws circuit in Cirq-style ASCII format.

    Features:
      - Clean gate alignment per moment
      - Vertical connection lines for multi-qubit gates
      - Measure symbols at the end
      - Proper width padding per column

    Args:
        circuit: Circuit defined with @circuit.

    Returns:
        Multi-line ASCII string representing the circuit.
    """
    builder = circuit.build()
    dag = DAGCircuit.from_builder(builder)
    layers = dag.layers()

    n = dag.num_qubits
    if n == 0:
        return "(empty circuit)"

    # Build column-by-column
    # Each column = one moment (layer)
    max_label = max(len(f"q[{i}]") for i in range(n))

    # Grid: columns[col_idx][qubit_idx] = cell content
    columns: list[list[str]] = []
    connect_columns: list[set[tuple[int, int]]] = []

    for layer in layers:
        col: dict[int, str] = {}
        connections: set[tuple[int, int]] = set()

        for op in layer:
            if op.gate_name in ("CX", "CZ", "CY"):
                col[op.qubits[0]] = "@"
                col[op.qubits[1]] = _TARGET_SYMBOLS.get(op.gate_name, "X")
                connections.add((min(op.qubits), max(op.qubits)))
            elif op.gate_name == "CCX":
                col[op.qubits[0]] = "@"
                col[op.qubits[1]] = "@"
                col[op.qubits[2]] = "X"
                connections.add((min(op.qubits), max(op.qubits)))
            elif op.gate_name == "SWAP":
                col[op.qubits[0]] = "x"
                col[op.qubits[1]] = "x"
                connections.add((min(op.qubits), max(op.qubits)))
            else:
                symbol = _GATE_SYMBOLS.get(op.gate_name, op.gate_name)
                for q in op.qubits:
                    col[q] = symbol

        # Determine connected qubits
        connected = set()
        for lo, hi in connections:
            for q in range(lo, hi + 1):
                connected.add(q)

        # Build column cells
        cells = []
        for i in range(n):
            if i in col:
                cells.append(col[i])
            elif i in connected:
                cells.append("|")
            else:
                cells.append("")
        columns.append(cells)
        connect_columns.append(connections)

    # Add measurement column
    if builder.measurement:
        measured = (
            set(builder.measurement.qubits)
            if builder.measurement.qubits
            else set(range(n))
        )
        cells = []
        for i in range(n):
            cells.append("M" if i in measured else "")
        columns.append(cells)

    # Calculate column widths (max cell width per column + padding)
    col_widths = []
    for col_cells in columns:
        w = max((len(c) for c in col_cells), default=1)
        col_widths.append(max(w, 1))

    # Render
    lines = []
    for i in range(n):
        label = f"q[{i}]"
        parts = [f"{label:>{max_label}}: "]

        for col_idx, col_cells in enumerate(columns):
            cell = col_cells[i]
            w = col_widths[col_idx]

            if cell == "|":
                # Vertical connection line
                pad_l = w // 2
                pad_r = w - pad_l - 1
                parts.append("─" * pad_l + "┼" + "─" * pad_r)
            elif cell:
                # Gate or measure
                pad_total = w - len(cell)
                pad_l = pad_total // 2
                pad_r = pad_total - pad_l
                parts.append("─" * pad_l + cell + "─" * pad_r)
            else:
                # Empty wire
                parts.append("─" * w)

            # Wire between columns
            if col_idx < len(columns) - 1:
                parts.append("──")

        parts.append("─")
        lines.append("".join(parts))

    return "\n".join(lines)
