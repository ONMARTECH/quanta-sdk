"""
quanta.core.types -- Core quantum types.

This module defines all fundamental data structures of the SDK:
  - QubitRef: Indexed reference to a qubit
  - Instruction: Single circuit instruction (gate + qubits)
  - QubitRegister: N-qubit register

These types are the foundation of the core layer with zero external dependencies.

Example:
    >>> reg = QubitRegister(3)
    >>> reg[0]
    QubitRef(index=0)
    >>> len(reg)
    3
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# -- Public API --
__all__ = [
    "QubitRef",
    "Instruction",
    "QubitRegister",
    "MeasureSpec",
    "CircuitSpec",
    "QuantaError",
    "CircuitError",
    "QubitIndexError",
]


# ===============================================
#  Error Hierarchy
# ===============================================

class QuantaError(Exception):
    """Base error class for the Quanta SDK."""


class CircuitError(QuantaError):
    """Circuit creation or validation error."""


class QubitIndexError(CircuitError):
    """Invalid qubit index error."""


# ===============================================
#  Core Types
# ===============================================

@dataclass(frozen=True, slots=True)
class QubitRef:
    """Immutable reference to a qubit.

    Created as `q[0]`, `q[1]` within a circuit.
    Being frozen, it is hashable and can be used in sets/dicts.

    Attributes:
        index: Zero-based index of the qubit within its register.
    """

    index: int

    def __repr__(self) -> str:
        return f"q[{self.index}]"


@dataclass(frozen=True, slots=True)
class Instruction:
    """A single circuit instruction: gate + target qubits + parameters.

    Created during circuit construction (lazy evaluation)
    and recorded into the CircuitBuilder.

    Attributes:
        gate_name: Gate name (e.g., "H", "CX", "RY").
        qubits: Target qubit indices.
        params: Angle values for parametric gates (in radians).
    """

    gate_name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class MeasureSpec:
    """Measurement specification: which qubits to measure.

    Attributes:
        qubits: Qubit indices to measure. If empty, all are measured.
    """

    qubits: tuple[int, ...] = ()


# ===============================================
#  QubitRegister
# ===============================================

class QubitRegister:
    """An N-qubit register with indexed access.

    Automatically created by `@circuit(qubits=N)`.
    User accesses via `q[0]`, `q[1]` or `for qubit in q:`.

    Args:
        size: Number of qubits. Must be a positive integer.

    Raises:
        CircuitError: If size < 1.

    Example:
        >>> reg = QubitRegister(3)
        >>> reg[0], reg[1], reg[2]
        (q[0], q[1], q[2])
    """

    __slots__ = ("_size", "_qubits")

    def __init__(self, size: int) -> None:
        if size < 1:
            raise CircuitError(f"Qubit count must be positive, given: {size}")
        self._size = size
        self._qubits = tuple(QubitRef(i) for i in range(size))

    def __getitem__(self, index: int) -> QubitRef:
        """Access a single qubit. Supports negative indexing."""
        if index < -self._size or index >= self._size:
            raise QubitIndexError(
                f"Invalid qubit index. "
                f"Valid range: [0, {self._size - 1}]"
            )
        return self._qubits[index]

    def __iter__(self) -> Iterator[QubitRef]:
        """Iterate over all qubits."""
        return iter(self._qubits)

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"QubitRegister(size={self._size})"


