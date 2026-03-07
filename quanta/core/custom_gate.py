"""
quanta.core.custom_gate -- User-defined quantum gates.

Allows creating custom gates from unitary matrices, fitting seamlessly
into the existing Gate architecture and circuit decorator.

Example:
    >>> from quanta.core.custom_gate import custom_gate
    >>> SqrtX = custom_gate("SqrtX", [[0.5+0.5j, 0.5-0.5j],
    ...                                [0.5-0.5j, 0.5+0.5j]])
    >>> @circuit(qubits=2)
    ... def my_circuit(q):
    ...     SqrtX(q[0])
    ...     CX(q[0], q[1])
    ...     return measure(q)
"""

from __future__ import annotations

import numpy as np

from quanta.core.gates import GATE_REGISTRY, Gate
from quanta.core.types import QuantaError

__all__ = ["custom_gate"]


class CustomGateError(QuantaError):
    """Error in custom gate definition."""


class CustomGate(Gate):
    """A user-defined gate backed by a unitary matrix.

    Inherits from Gate so it works everywhere built-in gates work:
    circuit decorator, simulator, QASM export, compiler, etc.

    Args:
        name: Unique name for this gate.
        matrix: Unitary matrix as a numpy array.
    """

    def __init__(self, name: str, matrix: np.ndarray) -> None:
        self._name = name
        self._matrix = np.asarray(matrix, dtype=complex)
        self._num_qubits = int(np.log2(self._matrix.shape[0]))

    @property
    def name(self) -> str:
        return self._name

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    def __repr__(self) -> str:
        return f"CustomGate('{self._name}', {self._num_qubits}q)"


def custom_gate(
    name: str,
    matrix: list | np.ndarray,
) -> CustomGate:
    """Creates a custom quantum gate from a unitary matrix.

    The gate is automatically registered and can be used like any
    built-in gate (H, X, CX, etc.) in @circuit functions.

    Args:
        name: Unique gate name (e.g., "SqrtX", "MyGate").
        matrix: Unitary matrix. Must be 2^n x 2^n.

    Returns:
        CustomGate instance callable on qubits.

    Raises:
        CustomGateError: If matrix is not square, not power-of-2, or not unitary.

    Example:
        >>> SqrtZ = custom_gate("SqrtZ", [[1, 0], [0, 1j]])
        >>> @circuit(qubits=1)
        ... def test(q):
        ...     SqrtZ(q[0])
        ...     return measure(q)
    """
    mat = np.asarray(matrix, dtype=complex)

    # Validate shape
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise CustomGateError(
            f"Gate matrix must be square. Got shape {mat.shape}"
        )

    dim = mat.shape[0]
    if dim == 0 or (dim & (dim - 1)) != 0:
        raise CustomGateError(
            f"Gate dimension must be power of 2. Got {dim}"
        )

    # Validate unitarity: U @ U.H should be identity
    product = mat @ mat.conj().T
    if not np.allclose(product, np.eye(dim), atol=1e-8):
        raise CustomGateError(
            f"Gate '{name}' is not unitary. U @ U.H != I"
        )

    # Check name uniqueness
    if name in GATE_REGISTRY:
        raise CustomGateError(
            f"Gate name '{name}' already registered. Choose a unique name."
        )

    gate = CustomGate(name, mat)

    # Register so simulator and QASM can find it
    GATE_REGISTRY[name] = gate

    return gate
