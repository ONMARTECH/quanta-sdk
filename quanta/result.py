"""
quanta.result -- Circuit execution result with rich display.

Inspired by Cirq's result display: Dirac notation, histogram, summary.

Example:
    >>> result = run(bell_state, shots=1024)
    >>> print(result)               # Pretty summary with histogram
    >>> result.dirac_notation()     # |00> + |11>
    >>> result.histogram()          # ASCII bar chart
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# -- Public API --
__all__ = ["Result"]


@dataclass
class Result:
    """Quantum circuit execution result.

    Provides multiple display formats inspired by Cirq:
      - summary()         -> pretty box with stats + histogram
      - dirac_notation()  -> Dirac ket representation of statevector
      - histogram()       -> ASCII bar chart of measurement results
      - print(result)     -> combines all for quick display

    Attributes:
        counts: Measurement results {bitstring: count}.
        shots: Number of measurement repetitions.
        num_qubits: Number of qubits in the circuit.
        circuit_name: Name of the circuit function.
        gate_count: Total number of gates.
        depth: Circuit depth (critical path).
        statevector: Final state vector (if available).
    """

    counts: dict[str, int]
    shots: int
    num_qubits: int
    circuit_name: str = ""
    gate_count: int = 0
    depth: int = 0
    statevector: np.ndarray | None = field(default=None, repr=False)

    @property
    def probabilities(self) -> dict[str, float]:
        """Measurement probabilities (count / shots)."""
        return {
            state: count / self.shots
            for state, count in sorted(self.counts.items())
        }

    @property
    def most_frequent(self) -> str:
        """Most frequently observed result."""
        return max(self.counts, key=self.counts.get)  # type: ignore[arg-type]

    def dirac_notation(self, decimals: int = 3) -> str:
        """Returns statevector in Dirac bra-ket notation.

        Like Cirq's `result.dirac_notation()`.

        Example:
            >>> result.dirac_notation()
            '0.707|00> + 0.707|11>'
        """
        if self.statevector is None:
            # Fall back to measurement probabilities
            terms = []
            for state, prob in sorted(
                self.probabilities.items(), key=lambda x: -x[1]
            ):
                if prob < 0.001:
                    continue
                amp = np.sqrt(prob)
                terms.append(f"{amp:.{decimals}f}|{state}>")
            return " + ".join(terms) if terms else "|0>"

        terms = []
        for i, amp in enumerate(self.statevector):
            if abs(amp) < 1e-8:
                continue
            bitstring = format(i, f"0{self.num_qubits}b")

            # Format amplitude
            real, imag = amp.real, amp.imag
            if abs(imag) < 1e-8:
                coeff = f"{real:.{decimals}f}"
            elif abs(real) < 1e-8:
                coeff = f"{imag:.{decimals}f}j"
            else:
                coeff = f"({real:.{decimals}f}{imag:+.{decimals}f}j)"

            terms.append(f"{coeff}|{bitstring}>")

        return " + ".join(terms) if terms else "|" + "0" * self.num_qubits + ">"

    def histogram(self, width: int = 40) -> str:
        """Returns an ASCII histogram of measurement results.

        Like Cirq's `plot_state_histogram()` but in text.

        Args:
            width: Maximum bar width in characters.

        Example:
            >>> print(result.histogram())
            |00> ████████████████████ 512 (50.0%)
            |11> ████████████████████ 512 (50.0%)
        """
        if not self.counts:
            return "No measurement results."

        max_count = max(self.counts.values())
        sorted_items = sorted(self.counts.items(), key=lambda x: -x[1])
        lines = []

        for state, count in sorted_items:
            prob = count / self.shots
            bar_len = int(count / max_count * width)
            bar = "█" * bar_len
            lines.append(f"  |{state}>  {bar}  {count} ({prob:.1%})")

        return "\n".join(lines)

    def summary(self) -> str:
        """Pretty result summary with circuit info and histogram.

        Combines stats + histogram + Dirac notation in a nice box.
        """
        w = 50
        name = self.circuit_name or "circuit"
        lines = [
            f"╔{'═' * w}╗",
            f"║  Quanta Result: {name:<{w - 18}}║",
            f"╠{'═' * w}╣",
            f"║  Qubits: {self.num_qubits:<6} Gates: {self.gate_count:<6} Depth: {self.depth:<6}║",
            f"║  Shots:  {self.shots:<{w - 11}}║",
            f"╠{'─' * w}╣",
        ]

        # Histogram
        max_count = max(self.counts.values()) if self.counts else 1
        sorted_items = sorted(self.counts.items(), key=lambda x: -x[1])
        for state, count in sorted_items[:16]:  # Top 16 results
            prob = count / self.shots
            bar_len = int(count / max_count * 20)
            bar = "█" * bar_len
            entry = f"  |{state}>  {bar}  {prob:.1%}"
            lines.append(f"║{entry:<{w}}║")

        if len(sorted_items) > 16:
            lines.append(f"║  ... +{len(sorted_items) - 16} more states{' ' * (w - 22)}║")

        # Dirac notation (if statevector available)
        if self.statevector is not None:
            dirac = self.dirac_notation(2)
            if len(dirac) <= w - 4:
                lines.append(f"╠{'─' * w}╣")
                lines.append(f"║  {dirac:<{w - 2}}║")

        lines.append(f"╚{'═' * w}╝")
        return "\n".join(lines)

    def __str__(self) -> str:
        """print(result) shows the pretty summary."""
        return self.summary()

    def __repr__(self) -> str:
        top = sorted(self.counts.items(), key=lambda x: -x[1])[:5]
        counts_str = ", ".join(f"'{s}': {c}" for s, c in top)
        return (
            f"Result(circuit='{self.circuit_name}', "
            f"shots={self.shots}, "
            f"counts={{{counts_str}}})"
        )

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks.

        Renders a styled card with circuit stats, color-coded histogram,
        and Dirac notation. Automatically used by Jupyter when displaying
        a Result object.
        """
        name = self.circuit_name or "circuit"
        sorted_items = sorted(self.counts.items(), key=lambda x: -x[1])
        max_count = max(self.counts.values()) if self.counts else 1

        # Color palette for histogram bars
        colors = [
            "#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd",
            "#818cf8", "#93c5fd", "#7dd3fc", "#67e8f9",
        ]

        rows = []
        for i, (state, count) in enumerate(sorted_items[:16]):
            prob = count / self.shots
            width = int(count / max_count * 100)
            color = colors[i % len(colors)]
            rows.append(
                f'<tr>'
                f'<td style="font-family:monospace;font-weight:600;padding:4px 8px">'
                f'|{state}⟩</td>'
                f'<td style="padding:4px 8px;width:100%">'
                f'<div style="background:{color};height:20px;border-radius:4px;'
                f'width:{width}%"></div></td>'
                f'<td style="padding:4px 8px;white-space:nowrap;'
                f'font-family:monospace">{prob:.1%}</td>'
                f'<td style="padding:4px 8px;color:#666;'
                f'font-family:monospace">{count}</td>'
                f'</tr>'
            )
        table = "".join(rows)

        overflow = ""
        if len(sorted_items) > 16:
            overflow = (
                f'<p style="color:#888;font-size:12px;margin:4px 0 0 0">'
                f'... +{len(sorted_items) - 16} more states</p>'
            )

        dirac_html = ""
        dirac = self.dirac_notation(3)
        if len(dirac) < 120:
            dirac_html = (
                f'<div style="margin-top:8px;padding:8px 12px;'
                f'background:#f0f0ff;border-radius:6px;'
                f'font-family:monospace;font-size:13px">'
                f'{dirac}</div>'
            )

        return (
            f'<div style="font-family:system-ui,-apple-system,sans-serif;'
            f'border:1px solid #e0e0e0;border-radius:10px;padding:16px;'
            f'max-width:600px;background:#fafafa">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;margin-bottom:12px">'
            f'<h3 style="margin:0;color:#1a1a2e">⚛️ {name}</h3>'
            f'<span style="background:#6366f1;color:white;'
            f'padding:2px 10px;border-radius:12px;font-size:12px">'
            f'Quanta</span></div>'
            f'<div style="display:flex;gap:16px;margin-bottom:12px;'
            f'font-size:13px;color:#555">'
            f'<span>🔮 {self.num_qubits} qubits</span>'
            f'<span>⚡ {self.gate_count} gates</span>'
            f'<span>📊 {self.shots:,} shots</span></div>'
            f'<table style="width:100%;border-collapse:collapse">'
            f'{table}</table>'
            f'{overflow}{dirac_html}</div>'
        )
