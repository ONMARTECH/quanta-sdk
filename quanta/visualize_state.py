"""
quanta.visualize_state — Statevector and probability visualization.


Example:
    >>> from quanta.visualize_state import show_probabilities, show_statevector
    >>> result = run(bell, shots=1024, seed=42)
    >>> print(show_probabilities(result))
    >>> print(show_statevector(result.statevector, num_qubits=2))
"""

from __future__ import annotations

import numpy as np

from quanta.result import Result

# ── Public API ──
__all__ = ["show_probabilities", "show_statevector", "show_phases"]

def show_probabilities(result: Result, max_states: int = 16) -> str:
    """Shows measurement probabilities as bar chart.

    Args:
        result: Execution sonucu.

    Returns:
        ASCII bar chart string'i.
    """
    probs = sorted(result.probabilities.items(), key=lambda x: -x[1])
    if len(probs) > max_states:
        probs = probs[:max_states]

    if not probs:
        return "No results."

    max_prob = max(p for _, p in probs)
    bar_width = 40

    lines = [f"╔═══ Probability Distribution ({result.circuit_name}) ═══"]

    for state, prob in probs:
        bar_len = int(prob / max_prob * bar_width) if max_prob > 0 else 0
        bar = "█" * bar_len
        lines.append(f"║ |{state}⟩  {prob:6.3f}  {bar}")

    lines.append("╚" + "═" * 50)
    return "\n".join(lines)

def show_statevector(
    statevector: np.ndarray,
    num_qubits: int,
    threshold: float = 0.01,
) -> str:
    """Shows statevector as a table.


    Args:

    Returns:
        Durum tablosu string'i.
    """
    lines = [
        "╠" + "─" * 55,
    ]

    for i, amp in enumerate(statevector):
        prob = abs(amp) ** 2
        if prob < threshold:
            continue

        bitstring = format(i, f"0{num_qubits}b")
        phase = np.angle(amp)
        phase_deg = np.degrees(phase)

        phase_symbol = _phase_to_symbol(phase)

        lines.append(
            f"║  |{bitstring}⟩    "
            f"{amp.real:+.4f}{amp.imag:+.4f}i    "
            f"{prob:.4f}      "
            f"{phase_symbol} {phase_deg:+.1f}°"
        )

    lines.append("╚" + "═" * 55)
    return "\n".join(lines)

def show_phases(statevector: np.ndarray, num_qubits: int) -> str:
    """Shows phase diagram in ASCII.


    Args:

    Returns:
    """

    lines = ["=== Phase Diagram ==="]

    for i, amp in enumerate(statevector):
        prob = abs(amp) ** 2
        if prob < 0.001:
            continue

        bitstring = format(i, f"0{num_qubits}b")
        phase = np.angle(amp)

        bar_len = int(np.sqrt(prob) * 20)
        bar = "█" * bar_len

        # Faz oku / Phase arrow
        arrow = _phase_to_arrow(phase)

        lines.append(f"║ |{bitstring}⟩  {arrow}  {bar}  ({prob:.3f})")

    lines.append("╚" + "═" * 40)
    return "\n".join(lines)

def _phase_to_symbol(phase: float) -> str:
    deg = np.degrees(phase) % 360
    if deg < 22.5 or deg >= 337.5:
        return "→"   # 0°
    elif deg < 67.5:
        return "↗"   # 45°
    elif deg < 112.5:
        return "↑"   # 90°
    elif deg < 157.5:
        return "↖"   # 135°
    elif deg < 202.5:
        return "←"   # 180°
    elif deg < 247.5:
        return "↙"   # 225°
    elif deg < 292.5:
        return "↓"   # 270°
    else:
        return "↘"   # 315°

def _phase_to_arrow(phase: float) -> str:
    deg = np.degrees(phase) % 360
    arrows = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]
    idx = int((deg + 22.5) / 45) % 8
    return arrows[idx]
