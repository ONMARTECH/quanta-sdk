"""

  - Bir ismi (name)
  - Unitär matris temsili (matrix)


Example:
    >>> from quanta.core.gates import H, CX
    >>> H.matrix.shape
    (2, 2)
    >>> CX.num_qubits
    2
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quanta.core.types import Instruction, QubitRef

if TYPE_CHECKING:
    from collections.abc import Iterable

# ── Public API ──
__all__ = [
    "Gate", "ParametricGate",
    "H", "X", "Y", "Z", "S", "T",
    "CX", "CZ", "CY", "SWAP", "CCX",
    "RX", "RY", "RZ",
]

# ── Sabitler ──
_SQRT2_INV = 1 / np.sqrt(2)
_I = np.eye(2, dtype=complex)

# ═══════════════════════════════════════════

_active_builders: list = []

def _get_active_builder():
    """Returns the active CircuitBuilder. Raises error if none."""
    if not _active_builders:
        from quanta.core.types import CircuitError
        raise CircuitError(
        )
    return _active_builders[-1]

# ═══════════════════════════════════════════

class Gate:
    """Base class for a quantum gate.


    Attributes:
    """

    name: str = ""
    num_qubits: int = 1

    @property
    def matrix(self) -> np.ndarray:
        """Unitary matrix representation of the gate."""
        return self._build_matrix()

    def _build_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, *args: QubitRef | Iterable[QubitRef]) -> None:
        """Records gate to active circuit. Supports broadcast.

            H(q[0])           → tek qubit
        """
        qubits = _flatten_qubits(args)

        if self.num_qubits == 1 and len(qubits) > 1:
            # Broadcast: H(q) = H(q[0]), H(q[1]), ...
            for qubit in qubits:
                _get_active_builder().record(
                    Instruction(self.name, (qubit,))
                )
        else:
            if len(qubits) != self.num_qubits:
                from quanta.core.types import CircuitError
                raise CircuitError(
                    f"ama {len(qubits)} verildi."
                )
            _get_active_builder().record(
                Instruction(self.name, tuple(qubits))
            )

    def __repr__(self) -> str:
        return f"Gate({self.name})"

class ParametricGate:
    """Parametric gate factory. Like RX(θ), RY(θ), RZ(θ).


    Example:
        >>> RY(np.pi/4)(q[0])  # θ=π/4 ile RY uygula
    """

    def __init__(self, name: str, matrix_fn) -> None:
        self.name = name
        self._matrix_fn = matrix_fn

    def __call__(self, theta: float) -> _BoundParametricGate:
        """Returns a gate bound with an angle."""
        return _BoundParametricGate(self.name, theta, self._matrix_fn)

    def __repr__(self) -> str:
        return f"ParametricGate({self.name})"

class _BoundParametricGate:
    """Parametric gate bound with an angle value."""

    num_qubits = 1

    def __init__(self, name: str, theta: float, matrix_fn) -> None:
        self.name = name
        self.theta = theta
        self._matrix_fn = matrix_fn

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix_fn(self.theta)

    def __call__(self, *args: QubitRef | Iterable[QubitRef]) -> None:
        qubits = _flatten_qubits(args)
        for qubit in qubits:
            _get_active_builder().record(
                Instruction(self.name, (qubit,), (self.theta,))
            )

# ═══════════════════════════════════════════

def _flatten_qubits(args) -> list[int]:
    result: list[int] = []
    for arg in args:
        if isinstance(arg, QubitRef):
            result.append(arg.index)
        elif hasattr(arg, "__iter__"):
            for item in arg:
                if isinstance(item, QubitRef):
                    result.append(item.index)
                else:
                    result.append(int(item))
        else:
            result.append(int(arg))
    return result

# ═══════════════════════════════════════════

# ── Tek-Qubit Sabit Gates ──

class _H(Gate):
    name = "H"
    def _build_matrix(self) -> np.ndarray:
        return np.array([[1, 1], [1, -1]], dtype=complex) * _SQRT2_INV

class _X(Gate):
    name = "X"
    def _build_matrix(self) -> np.ndarray:
        return np.array([[0, 1], [1, 0]], dtype=complex)

class _Y(Gate):
    name = "Y"
    def _build_matrix(self) -> np.ndarray:
        return np.array([[0, -1j], [1j, 0]], dtype=complex)

class _Z(Gate):
    name = "Z"
    def _build_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, -1]], dtype=complex)

class _S(Gate):
    name = "S"
    def _build_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, 1j]], dtype=complex)

class _T(Gate):
    name = "T"
    def _build_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

class _CX(Gate):
    name = "CX"
    num_qubits = 2
    def _build_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=complex)

class _CZ(Gate):
    name = "CZ"
    num_qubits = 2
    def _build_matrix(self) -> np.ndarray:
        return np.diag([1, 1, 1, -1]).astype(complex)

class _CY(Gate):
    name = "CY"
    num_qubits = 2
    def _build_matrix(self) -> np.ndarray:
        m = np.eye(4, dtype=complex)
        m[2:, 2:] = np.array([[0, -1j], [1j, 0]])
        return m

class _SWAP(Gate):
    name = "SWAP"
    num_qubits = 2
    def _build_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=complex)

class _CCX(Gate):
    name = "CCX"
    num_qubits = 3
    def _build_matrix(self) -> np.ndarray:
        m = np.eye(8, dtype=complex)
        m[6, 6], m[6, 7] = 0, 1
        m[7, 6], m[7, 7] = 1, 0
        return m

RX = ParametricGate("RX", lambda t: np.array([
    [np.cos(t / 2), -1j * np.sin(t / 2)],
    [-1j * np.sin(t / 2), np.cos(t / 2)],
], dtype=complex))

RY = ParametricGate("RY", lambda t: np.array([
    [np.cos(t / 2), -np.sin(t / 2)],
    [np.sin(t / 2), np.cos(t / 2)],
], dtype=complex))

RZ = ParametricGate("RZ", lambda t: np.array([
    [np.exp(-1j * t / 2), 0],
    [0, np.exp(1j * t / 2)],
], dtype=complex))

# ── Singleton Instances ──
H = _H()
X = _X()
Y = _Y()
Z = _Z()
S = _S()
T = _T()
CX = _CX()
CZ = _CZ()
CY = _CY()
SWAP = _SWAP()
CCX = _CCX()


GATE_REGISTRY: dict[str, Gate | ParametricGate] = {
    "H": H, "X": X, "Y": Y, "Z": Z, "S": S, "T": T,
    "CX": CX, "CZ": CZ, "CY": CY, "SWAP": SWAP, "CCX": CCX,
}
