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

        Raises:
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

    def build(self) -> CircuitBuilder:
        """Runs the circuit function and collects instructions.

        Returns:

        Raises:
            CircuitError: If the circuit function raises.
        """
        builder = CircuitBuilder(self.num_qubits)
        register = QubitRegister(self.num_qubits)

        with builder:
            try:
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

# ═══════════════════════════════════════════

def circuit(qubits: int) -> Callable[[Callable], CircuitDefinition]:
    """Marks a function as a quantum circuit.

    Args:

    Returns:

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
