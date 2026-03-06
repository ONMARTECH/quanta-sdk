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
