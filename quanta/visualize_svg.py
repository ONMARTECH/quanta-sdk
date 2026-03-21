"""
quanta.visualize_svg -- SVG circuit diagram generator.

Creates publication-quality circuit diagrams as HTML/SVG,
similar to IBM Quantum's visual circuit editor.

Features:
  - Color-coded gates by category
  - Multi-qubit gate connections with vertical lines
  - Measurement symbols
  - Responsive SVG output
  - Standalone HTML with embedded CSS

Example:
    >>> from quanta.visualize_svg import to_svg, to_html
    >>> svg = to_svg(bell)
    >>> html = to_html(bell, title="Bell State")
"""

from __future__ import annotations

from quanta.core.circuit import CircuitDefinition
from quanta.dag.dag_circuit import DAGCircuit

__all__ = ["to_svg", "to_html"]

# ── Gate Colors (IBM-inspired) ──

_GATE_COLORS: dict[str, tuple[str, str]] = {
    # (background, text_color)
    # Hadamard — coral/red
    "H": ("#E8384F", "#FFF"),
    # Pauli — blue
    "X": ("#4A90D9", "#FFF"),
    "Y": ("#9B59B6", "#FFF"),
    "Z": ("#4A90D9", "#FFF"),
    # Phase — light blue
    "S": ("#5DADE2", "#FFF"),
    "T": ("#5DADE2", "#FFF"),
    "SDG": ("#5DADE2", "#FFF"),
    "TDG": ("#5DADE2", "#FFF"),
    "P": ("#5DADE2", "#FFF"),
    # Identity
    "I": ("#BDC3C7", "#333"),
    # SX — magenta/pink
    "SX": ("#C0392B", "#FFF"),
    "SXdg": ("#C0392B", "#FFF"),
    # Rotation — deep magenta
    "RX": ("#8E44AD", "#FFF"),
    "RY": ("#8E44AD", "#FFF"),
    "RZ": ("#2E86C1", "#FFF"),
    "RXX": ("#8E44AD", "#FFF"),
    "RZZ": ("#2E86C1", "#FFF"),
    # Universal
    "U": ("#E67E22", "#FFF"),
    # Multi-qubit — dark blue
    "CX": ("#2C3E50", "#FFF"),
    "CY": ("#2C3E50", "#FFF"),
    "CZ": ("#2C3E50", "#FFF"),
    "CCX": ("#2C3E50", "#FFF"),
    "RCCX": ("#7F8C8D", "#FFF"),
    "RC3X": ("#7F8C8D", "#FFF"),
    "SWAP": ("#2C3E50", "#FFF"),
}

_DEFAULT_COLOR = ("#95A5A6", "#FFF")

# Gate display labels
_GATE_LABELS: dict[str, str] = {
    "H": "H", "X": "X", "Y": "Y", "Z": "Z",
    "S": "S", "T": "T", "I": "I",
    "SDG": "S†", "TDG": "T†",
    "SX": "√X", "SXdg": "√X†",
    "P": "P", "U": "U",
    "RX": "RX", "RY": "RY", "RZ": "RZ",
    "RXX": "RXX", "RZZ": "RZZ",
    "CX": "⊕", "CY": "⊕", "CZ": "●",
    "CCX": "⊕", "RCCX": "⊕", "RC3X": "⊕",
    "SWAP": "×",
}

# ── Layout Constants ──

_QUBIT_SPACING = 50
_GATE_WIDTH = 44
_GATE_HEIGHT = 36
_COL_SPACING = 60
_LEFT_MARGIN = 80
_TOP_MARGIN = 30
_CTRL_RADIUS = 6
_TARGET_RADIUS = 14
_WIRE_COLOR = "#555"
_BG_COLOR = "#FAFBFC"
_MEASURE_COLOR = "#2C3E50"


def _build_layers(dag: DAGCircuit):
    """Returns layers of operations for column placement."""
    return dag.layers()


def to_svg(
    circuit_def: CircuitDefinition,
    *,
    show_params: bool = True,
) -> str:
    """Generates an SVG circuit diagram.

    Args:
        circuit_def: Circuit defined with @circuit.
        show_params: Show rotation parameters.

    Returns:
        SVG string.
    """
    builder = circuit_def.build()
    dag = DAGCircuit.from_builder(builder)
    layers = _build_layers(dag)
    n = dag.num_qubits

    if n == 0:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="40">' \
               '<text x="10" y="25" font-size="14">(empty circuit)</text></svg>'

    # Add measurement column
    has_measure = builder.measurement is not None
    num_cols = len(layers) + (1 if has_measure else 0)

    # SVG dimensions
    width = _LEFT_MARGIN + num_cols * _COL_SPACING + 40
    height = _TOP_MARGIN + n * _QUBIT_SPACING + 20

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" '
        f'style="font-family: \'Inter\', \'SF Pro\', -apple-system, '
        f'sans-serif; background: {_BG_COLOR};">'
    )

    # Defs for shadows and gradients
    parts.append("""
    <defs>
      <filter id="shadow" x="-10%" y="-10%" width="130%" height="130%">
        <feDropShadow dx="1" dy="1" stdDeviation="1.5"
         flood-color="#000" flood-opacity="0.15"/>
      </filter>
      <filter id="measure-shadow" x="-20%" y="-20%"
       width="140%" height="140%">
        <feDropShadow dx="0.5" dy="0.5" stdDeviation="1"
         flood-color="#000" flood-opacity="0.2"/>
      </filter>
    </defs>
    """)

    # ── Qubit labels and wires ──
    for i in range(n):
        y = _TOP_MARGIN + i * _QUBIT_SPACING
        wire_end = _LEFT_MARGIN + num_cols * _COL_SPACING + 10

        # Qubit label
        parts.append(
            f'<text x="{_LEFT_MARGIN - 10}" y="{y + 5}" '
            f'text-anchor="end" font-size="13" fill="#333" '
            f'font-weight="500">q[{i}]</text>'
        )

        # Wire line
        parts.append(
            f'<line x1="{_LEFT_MARGIN - 5}" y1="{y}" '
            f'x2="{wire_end}" y2="{y}" '
            f'stroke="{_WIRE_COLOR}" stroke-width="1.5" '
            f'stroke-linecap="round"/>'
        )

    # ── Gate rendering ──
    for col_idx, layer in enumerate(layers):
        cx = _LEFT_MARGIN + col_idx * _COL_SPACING + _COL_SPACING // 2

        for op in layer:
            gate = op.gate_name
            qubits = op.qubits
            params = op.params

            if gate in ("CX", "CY", "CCX", "RCCX", "RC3X"):
                # Control-target gate
                controls = list(qubits[:-1])
                target = qubits[-1]

                # Vertical line
                min_q = min(qubits)
                max_q = max(qubits)
                y_min = _TOP_MARGIN + min_q * _QUBIT_SPACING
                y_max = _TOP_MARGIN + max_q * _QUBIT_SPACING
                parts.append(
                    f'<line x1="{cx}" y1="{y_min}" '
                    f'x2="{cx}" y2="{y_max}" '
                    f'stroke="{_WIRE_COLOR}" stroke-width="2"/>'
                )

                # Control dots
                for c in controls:
                    cy_pos = _TOP_MARGIN + c * _QUBIT_SPACING
                    parts.append(
                        f'<circle cx="{cx}" cy="{cy_pos}" '
                        f'r="{_CTRL_RADIUS}" fill="#2C3E50"/>'
                    )

                # Target circle with ⊕
                ty = _TOP_MARGIN + target * _QUBIT_SPACING
                parts.append(
                    f'<circle cx="{cx}" cy="{ty}" '
                    f'r="{_TARGET_RADIUS}" fill="none" '
                    f'stroke="#2C3E50" stroke-width="2"/>'
                )
                parts.append(
                    f'<line x1="{cx}" y1="{ty - _TARGET_RADIUS}" '
                    f'x2="{cx}" y2="{ty + _TARGET_RADIUS}" '
                    f'stroke="#2C3E50" stroke-width="2"/>'
                )
                parts.append(
                    f'<line x1="{cx - _TARGET_RADIUS}" y1="{ty}" '
                    f'x2="{cx + _TARGET_RADIUS}" y2="{ty}" '
                    f'stroke="#2C3E50" stroke-width="2"/>'
                )

            elif gate == "CZ":
                # Two control dots
                y0 = _TOP_MARGIN + qubits[0] * _QUBIT_SPACING
                y1 = _TOP_MARGIN + qubits[1] * _QUBIT_SPACING
                parts.append(
                    f'<line x1="{cx}" y1="{y0}" '
                    f'x2="{cx}" y2="{y1}" '
                    f'stroke="{_WIRE_COLOR}" stroke-width="2"/>'
                )
                for q in qubits:
                    qy = _TOP_MARGIN + q * _QUBIT_SPACING
                    parts.append(
                        f'<circle cx="{cx}" cy="{qy}" '
                        f'r="{_CTRL_RADIUS}" fill="#2C3E50"/>'
                    )

            elif gate == "SWAP":
                y0 = _TOP_MARGIN + qubits[0] * _QUBIT_SPACING
                y1 = _TOP_MARGIN + qubits[1] * _QUBIT_SPACING
                parts.append(
                    f'<line x1="{cx}" y1="{y0}" '
                    f'x2="{cx}" y2="{y1}" '
                    f'stroke="{_WIRE_COLOR}" stroke-width="2"/>'
                )
                # × symbols
                size = 8
                for q in qubits:
                    qy = _TOP_MARGIN + q * _QUBIT_SPACING
                    parts.append(
                        f'<line x1="{cx - size}" y1="{qy - size}" '
                        f'x2="{cx + size}" y2="{qy + size}" '
                        f'stroke="#2C3E50" stroke-width="2.5"/>'
                    )
                    parts.append(
                        f'<line x1="{cx + size}" y1="{qy - size}" '
                        f'x2="{cx - size}" y2="{qy + size}" '
                        f'stroke="#2C3E50" stroke-width="2.5"/>'
                    )

            elif gate in ("RXX", "RZZ"):
                # 2-qubit parametric: boxes on both qubits
                y0 = _TOP_MARGIN + qubits[0] * _QUBIT_SPACING
                y1 = _TOP_MARGIN + qubits[1] * _QUBIT_SPACING

                # Vertical line
                parts.append(
                    f'<line x1="{cx}" y1="{y0}" '
                    f'x2="{cx}" y2="{y1}" '
                    f'stroke="{_WIRE_COLOR}" stroke-width="2"/>'
                )

                bg, fg = _GATE_COLORS.get(gate, _DEFAULT_COLOR)
                label = _GATE_LABELS.get(gate, gate)

                # Box on both qubits
                for q in qubits:
                    qy = _TOP_MARGIN + q * _QUBIT_SPACING
                    gw = _GATE_WIDTH + 4
                    parts.append(
                        f'<rect x="{cx - gw // 2}" '
                        f'y="{qy - _GATE_HEIGHT // 2}" '
                        f'width="{gw}" height="{_GATE_HEIGHT}" '
                        f'rx="6" fill="{bg}" filter="url(#shadow)"/>'
                    )
                    parts.append(
                        f'<text x="{cx}" y="{qy + 5}" '
                        f'text-anchor="middle" font-size="12" '
                        f'font-weight="600" fill="{fg}">{label}</text>'
                    )

            else:
                # Single-qubit gate: colored box
                for q in qubits:
                    qy = _TOP_MARGIN + q * _QUBIT_SPACING
                    bg, fg = _GATE_COLORS.get(gate, _DEFAULT_COLOR)
                    label = _GATE_LABELS.get(gate, gate)

                    # Wider box for long labels
                    gw = _GATE_WIDTH
                    if len(label) > 2:
                        gw = _GATE_WIDTH + 12

                    parts.append(
                        f'<rect x="{cx - gw // 2}" '
                        f'y="{qy - _GATE_HEIGHT // 2}" '
                        f'width="{gw}" height="{_GATE_HEIGHT}" '
                        f'rx="6" fill="{bg}" filter="url(#shadow)"/>'
                    )

                    # Label
                    parts.append(
                        f'<text x="{cx}" y="{qy + 5}" '
                        f'text-anchor="middle" font-size="13" '
                        f'font-weight="600" fill="{fg}">{label}</text>'
                    )

                    # Param value below
                    if show_params and params:
                        import math
                        p_val = params[0]
                        # Show as fraction of π if close
                        if abs(p_val - math.pi) < 0.01:
                            p_str = "π"
                        elif abs(p_val - math.pi / 2) < 0.01:
                            p_str = "π/2"
                        elif abs(p_val - math.pi / 4) < 0.01:
                            p_str = "π/4"
                        elif abs(p_val + math.pi) < 0.01:
                            p_str = "-π"
                        elif abs(p_val + math.pi / 2) < 0.01:
                            p_str = "-π/2"
                        else:
                            p_str = f"{p_val:.2f}"
                        parts.append(
                            f'<text x="{cx}" '
                            f'y="{qy + _GATE_HEIGHT // 2 + 12}" '
                            f'text-anchor="middle" font-size="9" '
                            f'fill="#888">{p_str}</text>'
                        )

    # ── Measurement ──
    if has_measure:
        col_idx = len(layers)
        cx = _LEFT_MARGIN + col_idx * _COL_SPACING + _COL_SPACING // 2

        measured = set(range(n))
        if builder.measurement and builder.measurement.qubits:
            measured = set(builder.measurement.qubits)

        for q in range(n):
            if q not in measured:
                continue
            qy = _TOP_MARGIN + q * _QUBIT_SPACING

            # Measurement box
            mw, mh = 32, 32
            parts.append(
                f'<rect x="{cx - mw // 2}" y="{qy - mh // 2}" '
                f'width="{mw}" height="{mh}" '
                f'rx="4" fill="#F8F9FA" stroke="{_MEASURE_COLOR}" '
                f'stroke-width="1.5" filter="url(#measure-shadow)"/>'
            )

            # Meter arc
            arc_r = 8
            parts.append(
                f'<path d="M {cx - arc_r} {qy + 2} '
                f'A {arc_r} {arc_r} 0 0 1 {cx + arc_r} {qy + 2}" '
                f'fill="none" stroke="{_MEASURE_COLOR}" '
                f'stroke-width="1.5"/>'
            )

            # Meter needle
            parts.append(
                f'<line x1="{cx}" y1="{qy + 2}" '
                f'x2="{cx + 6}" y2="{qy - 7}" '
                f'stroke="{_MEASURE_COLOR}" stroke-width="1.5" '
                f'stroke-linecap="round"/>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def to_html(
    circuit_def: CircuitDefinition,
    *,
    title: str = "Quantum Circuit",
    dark_mode: bool = False,
    show_params: bool = True,
) -> str:
    """Generates a standalone HTML page with the circuit diagram.

    Args:
        circuit_def: Circuit defined with @circuit.
        title: Page title.
        dark_mode: Use dark background.
        show_params: Show rotation parameters.

    Returns:
        HTML string.
    """
    svg = to_svg(circuit_def, show_params=show_params)

    bg = "#1a1a2e" if dark_mode else "#FAFBFC"
    fg = "#e0e0e0" if dark_mode else "#333"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — Quanta SDK</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap"
 rel="stylesheet">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, sans-serif;
    background: {bg};
    color: {fg};
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem;
    min-height: 100vh;
  }}
  .header {{
    text-align: center;
    margin-bottom: 2rem;
  }}
  .header h1 {{
    font-size: 1.5rem;
    font-weight: 600;
    letter-spacing: -0.02em;
  }}
  .header .subtitle {{
    font-size: 0.85rem;
    color: #888;
    margin-top: 0.25rem;
  }}
  .circuit-container {{
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    overflow-x: auto;
    max-width: 95vw;
  }}
  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-top: 1.5rem;
    justify-content: center;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    color: #666;
  }}
  .legend-dot {{
    width: 12px;
    height: 12px;
    border-radius: 3px;
  }}
  .footer {{
    margin-top: 2rem;
    font-size: 0.7rem;
    color: #aaa;
  }}
</style>
</head>
<body>
  <div class="header">
    <h1>{title}</h1>
    <div class="subtitle">Generated by Quanta SDK</div>
  </div>
  <div class="circuit-container">
    {svg}
  </div>
  <div class="legend">
    <div class="legend-item">
      <div class="legend-dot" style="background:#E8384F"></div>Hadamard
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#4A90D9"></div>Pauli
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#5DADE2"></div>Phase
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#8E44AD"></div>Rotation
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#2C3E50"></div>Multi-Qubit
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#C0392B"></div>√X
    </div>
  </div>
  <div class="footer">
    Quanta SDK v0.8 — 31 gates, IBM Quantum compatible
  </div>
</body>
</html>"""

