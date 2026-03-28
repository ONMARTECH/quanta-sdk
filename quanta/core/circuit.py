"""

The @circuit decorator marks a function as a quantum circuit.

Example:
    >>> @circuit(qubits=2)
    ... def bell(q):
    ...     H(q[0])
    ...     CX(q[0], q[1])
    ...     return measure(q)
    >>>
    >>> bell.num_qubits
    2
    >>> bell.build()
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from quanta.core.types import (
    CircuitError,
    Instruction,
    MeasureSpec,
    QubitRegister,
)

# ── Public API ──
__all__ = ["circuit", "CircuitBuilder", "CircuitDefinition"]

# ═══════════════════════════════════════════
#  CircuitBuilder — Instruction Recorder
# ═══════════════════════════════════════════

class CircuitBuilder:
    """Context manager that collects circuit instructions.


    Attributes:
        measurement: Measurement specification (if any).
    """

    __slots__ = ("instructions", "measurement", "num_qubits")

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits
        self.instructions: list[Instruction] = []
        self.measurement: MeasureSpec | None = None

    def record(self, instruction: Instruction) -> None:
        """Records an instruction to the circuit.

        Args:
            instruction: Gate operation to append (gate name, qubits, params).

        Raises:
            QubitIndexError: If any qubit index is out of range [0, num_qubits).
        """
        from quanta.core.types import QubitIndexError

        for q in instruction.qubits:
            if q < 0 or q >= self.num_qubits:
                raise QubitIndexError(
                    f"Invalid qubit index. "
                    f"Circuit contains N qubits [0, {self.num_qubits - 1}]."
                )
        self.instructions.append(instruction)

    def set_measurement(self, spec: MeasureSpec) -> None:
        """Sets the measurement specification."""
        self.measurement = spec

    def __enter__(self) -> CircuitBuilder:
        """Pushes builder onto the active builder stack."""
        from quanta.core.gates import _active_builders
        _active_builders.append(self)
        return self

    def __exit__(self, *args: Any) -> None:
        """Pops builder from the stack."""
        from quanta.core.gates import _active_builders
        _active_builders.pop()

# ═══════════════════════════════════════════

class CircuitDefinition:
    """A circuit defined with @circuit.


    Attributes:
        name: Circuit function name.
    """

    def __init__(self, fn: Callable, num_qubits: int) -> None:
        self._fn = fn
        self.num_qubits = num_qubits
        self.name = fn.__name__
        self.__doc__ = fn.__doc__
        self.__wrapped__ = fn

    def build(self, **kwargs: float) -> CircuitBuilder:
        """Runs the circuit function and collects instructions.

        Args:
            **kwargs: Circuit parameters (e.g., theta=0.5). Forwarded
                to the circuit function if it accepts them.

        Returns:
            CircuitBuilder containing all recorded instructions and measurements.

        Raises:
            CircuitError: If the circuit function raises an exception.
        """
        builder = CircuitBuilder(self.num_qubits)
        register = QubitRegister(self.num_qubits)

        with builder:
            try:
                # Cache signature to avoid repeated inspect calls
                if not hasattr(self, "_param_count"):
                    import inspect
                    sig = inspect.signature(self._fn)
                    self._param_count = len(sig.parameters)
                if self._param_count > 1 and kwargs:
                    self._fn(register, **kwargs)
                else:
                    self._fn(register)
            except Exception as exc:
                if isinstance(exc, CircuitError):
                    raise
                raise CircuitError(
                    f"Circuit '{self.name}' raised an error during build: {exc}"
                ) from exc

        return builder

    def __repr__(self) -> str:
        return f"CircuitDefinition(name='{self.name}', qubits={self.num_qubits})"

    def _repr_html_(self) -> str:
        """Rich SVG display for Jupyter notebooks.

        Renders the circuit as an inline SVG diagram with color-coded gates.
        Automatically used by Jupyter when displaying a CircuitDefinition.
        """
        try:
            from quanta.visualize_svg import to_svg
            svg = to_svg(self)
            return (
                f'<div style="font-family:system-ui,-apple-system,sans-serif;'
                f'border:1px solid #e0e0e0;border-radius:10px;padding:16px;'
                f'background:#fafafa;display:inline-block">'
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;margin-bottom:8px">'
                f'<span style="font-weight:600;color:#1a1a2e">'
                f'⚛️ {self.name}</span>'
                f'<span style="background:#6366f1;color:white;'
                f'padding:2px 10px;border-radius:12px;font-size:12px">'
                f'{self.num_qubits}q</span></div>{svg}</div>'
            )
        except Exception:
            return f"<pre>{self!r}</pre>"

# ═══════════════════════════════════════════

def circuit(qubits: int) -> Callable[[Callable], CircuitDefinition]:
    """Marks a function as a quantum circuit.

    Args:
        qubits: Number of qubits in the circuit (must be ≥ 1).

    Returns:
        A decorator that wraps the function in a CircuitDefinition.

    Raises:
        CircuitError: qubits < 1 ise.

    Example:
        >>> @circuit(qubits=3)
        ... def ghz(q):
        ...     H(q[0])
        ...     CX(q[0], q[1])
        ...     CX(q[1], q[2])
        ...     return measure(q)
    """
    if qubits < 1:
        raise CircuitError(f"Qubit count must be positive, given: {qubits}")

    def decorator(fn: Callable) -> CircuitDefinition:
        return CircuitDefinition(fn, qubits)

    return decorator
