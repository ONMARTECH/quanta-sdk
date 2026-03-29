"""

  - A name (name)
  - Unitary matrix representation (matrix)


Example:
    >>> from quanta.core.gates import H, CX
    >>> H.matrix.shape
    (2, 2)
    >>> CX.num_qubits
    2
"""

from __future__ import annotations

import threading
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
    # IBM-parity gates
    "I", "SDG", "TDG", "P", "SX", "SXdg",
    "U", "RXX", "RZZ", "RCCX", "RC3X",
    # v0.9 new gates
    "ECR", "iSWAP", "CSWAP", "CH", "CP", "MS",
]

# ── Constants ──
_SQRT2_INV = 1 / np.sqrt(2)
_I = np.eye(2, dtype=complex)


_thread_local = threading.local()


def _get_builders_stack() -> list:
    """Returns the per-thread builder stack, creating it if needed."""
    if not hasattr(_thread_local, "builders"):
        _thread_local.builders = []
    return _thread_local.builders


# Public alias for backward compatibility (used by circuit.py)
# Now returns the thread-local stack instead of a global list.
_active_builders = type("_BuilderProxy", (), {
    "append": staticmethod(lambda b: _get_builders_stack().append(b)),
    "pop": staticmethod(lambda: _get_builders_stack().pop()),
    "__bool__": staticmethod(lambda: bool(_get_builders_stack())),
})()


def _get_active_builder():
    """Returns the active CircuitBuilder for the current thread.

    Each thread has its own isolated builder stack, so concurrent
    circuit builds never interfere with each other.

    Raises:
        CircuitError: If no builder is active in the current thread.
    """
    stack = _get_builders_stack()
    if not stack:
        from quanta.core.types import CircuitError
        raise CircuitError(
            "No active circuit builder. Use gates inside a @circuit function."
        )
    return stack[-1]

# ═══════════════════════════════════════════

class Gate:
    """Base class for a quantum gate.

    Attributes:
        name: Gate name (e.g. "H", "CX").
        num_qubits: Number of qubits this gate acts on.
    """

    name: str = ""
    num_qubits: int = 1

    @property
    def matrix(self) -> np.ndarray:
        """Unitary matrix representation of the gate."""
        return self._build_matrix()

    def _build_matrix(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def inverse(self) -> Gate:
        """Returns the inverse (adjoint) gate: U†.

        For self-inverse gates (H, X, Y, Z, CX, CZ, SWAP) returns self.
        For known pairs (S↔SDG, T↔TDG, SX↔SXdg) returns the partner.
        Otherwise computes U† from the matrix.

        Example:
            >>> S.inverse.name
            'SDG'
            >>> H.inverse is H
            True
        """
        # Check known inverse pairs
        if self.name in _INVERSE_MAP:
            return GATE_REGISTRY[_INVERSE_MAP[self.name]]
        # Self-inverse gates
        if self.name in _SELF_INVERSE_GATES:
            return self
        # Fallback: compute U† from matrix
        return _MatrixGate(
            f"{self.name}†", self.matrix.conj().T, self.num_qubits,
        )

    def controlled(self, num_ctrl: int = 1) -> Gate:
        """Returns a controlled version of this gate.

        Builds a (num_ctrl + num_qubits)-qubit controlled-U gate.
        The first num_ctrl qubits are controls, the rest are targets.

        Args:
            num_ctrl: Number of control qubits (default: 1).

        Returns:
            A new Gate with the controlled-U matrix.

        Example:
            >>> X.controlled().name
            'CX'
            >>> H.controlled().num_qubits
            2
        """
        if num_ctrl < 1:
            from quanta.core.types import GateError
            raise GateError(f"num_ctrl must be >= 1, got {num_ctrl}")

        total_qubits = self.num_qubits + num_ctrl
        dim = 2 ** total_qubits
        target_dim = 2 ** self.num_qubits

        # Build controlled-U: identity for all states except
        # when all controls are |1⟩
        cu = np.eye(dim, dtype=complex)
        # Replace bottom-right block with U
        u_mat = self.matrix
        start = dim - target_dim
        cu[start:, start:] = u_mat

        prefix = "C" * num_ctrl
        return _MatrixGate(f"{prefix}{self.name}", cu, total_qubits)

    def __call__(self, *args: QubitRef | Iterable[QubitRef]) -> None:
        """Records gate to active circuit. Supports broadcast.

            H(q[0])           → single qubit
            H(q)              → broadcast to all qubits
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
                from quanta.core.types import GateError
                raise GateError(
                    f"Gate '{self.name}' expects {self.num_qubits} "
                    f"qubit(s), got {len(qubits)}."
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

    def __init__(self, name: str, matrix_fn, num_qubits: int = 1) -> None:
        self.name = name
        self._matrix_fn = matrix_fn
        self.num_qubits = num_qubits

    def __call__(self, theta: float) -> _BoundParametricGate:
        """Returns a gate bound with an angle."""
        return _BoundParametricGate(
            self.name, theta, self._matrix_fn, self.num_qubits,
        )

    def __repr__(self) -> str:
        return f"ParametricGate({self.name})"

class _BoundParametricGate:
    """Parametric gate bound with an angle value."""

    def __init__(
        self,
        name: str,
        theta: float,
        matrix_fn,
        num_qubits: int = 1,
    ) -> None:
        self.name = name
        self.theta = theta
        self._matrix_fn = matrix_fn
        self.num_qubits = num_qubits

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix_fn(self.theta)

    def __call__(self, *args: QubitRef | Iterable[QubitRef]) -> None:
        qubits = _flatten_qubits(args)
        if self.num_qubits == 1:
            for qubit in qubits:
                _get_active_builder().record(
                    Instruction(self.name, (qubit,), (self.theta,))
                )
        else:
            if len(qubits) != self.num_qubits:
                from quanta.core.types import CircuitError
                raise CircuitError(
                    f"{self.name} requires {self.num_qubits} "
                    f"qubits, got {len(qubits)}."
                )
            _get_active_builder().record(
                Instruction(self.name, tuple(qubits), (self.theta,))
            )


class MultiParametricGate:
    """Multi-parameter gate factory. Like U(θ, φ, λ)."""

    def __init__(self, name: str, matrix_fn, num_params: int = 3) -> None:
        self.name = name
        self._matrix_fn = matrix_fn
        self.num_params = num_params

    def __call__(self, *params: float) -> _BoundMultiParametricGate:
        return _BoundMultiParametricGate(
            self.name, params, self._matrix_fn,
        )

    def __repr__(self) -> str:
        return f"MultiParametricGate({self.name})"


class _BoundMultiParametricGate:
    """Multi-parameter gate bound with values."""

    num_qubits = 1

    def __init__(self, name: str, params: tuple, matrix_fn) -> None:
        self.name = name
        self.params = params
        self._matrix_fn = matrix_fn

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix_fn(*self.params)

    def __call__(self, *args: QubitRef | Iterable[QubitRef]) -> None:
        qubits = _flatten_qubits(args)
        for qubit in qubits:
            _get_active_builder().record(
                Instruction(self.name, (qubit,), self.params)
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

# ── Single-Qubit Fixed Gates ──

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

# ── New: Identity ──

class _Identity(Gate):
    name = "I"
    def _build_matrix(self) -> np.ndarray:
        return np.eye(2, dtype=complex)

# ── New: S†, T† (conjugate transpose) ──

class _SDG(Gate):
    name = "SDG"
    def _build_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, -1j]], dtype=complex)

class _TDG(Gate):
    name = "TDG"
    def _build_matrix(self) -> np.ndarray:
        return np.array(
            [[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex,
        )

# ── New: SX (√X), SXdg (√X†) ──

class _SX(Gate):
    """Square root of X gate. Native Heron gate."""
    name = "SX"
    def _build_matrix(self) -> np.ndarray:
        return 0.5 * np.array([
            [1 + 1j, 1 - 1j],
            [1 - 1j, 1 + 1j],
        ], dtype=complex)

class _SXdg(Gate):
    """Conjugate transpose of SX."""
    name = "SXdg"
    def _build_matrix(self) -> np.ndarray:
        return 0.5 * np.array([
            [1 - 1j, 1 + 1j],
            [1 + 1j, 1 - 1j],
        ], dtype=complex)

# ── Multi-Qubit Fixed Gates ──

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

# ── New: RCCX (Relative-phase Toffoli) ──

class _RCCX(Gate):
    """Relative-phase Toffoli (simplified CCX, up to phase)."""
    name = "RCCX"
    num_qubits = 3
    def _build_matrix(self) -> np.ndarray:
        m = np.eye(8, dtype=complex)
        m[6, 6], m[6, 7] = 0, -1j
        m[7, 6], m[7, 7] = -1j, 0
        return m

# ── New: RC3X (Relative-phase 3-controlled X) ──

class _RC3X(Gate):
    """Relative-phase 3-controlled X gate."""
    name = "RC3X"
    num_qubits = 4
    def _build_matrix(self) -> np.ndarray:
        m = np.eye(16, dtype=complex)
        m[14, 14], m[14, 15] = 0, -1j
        m[15, 14], m[15, 15] = -1j, 0
        return m

# ── Parametric Gates ──

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

# ── New: P (Phase gate) ──

P = ParametricGate("P", lambda t: np.array([
    [1, 0],
    [0, np.exp(1j * t)],
], dtype=complex))

# ── New: RXX (2-qubit XX rotation) ──

RXX = ParametricGate("RXX", lambda t: np.array([
    [np.cos(t / 2), 0, 0, -1j * np.sin(t / 2)],
    [0, np.cos(t / 2), -1j * np.sin(t / 2), 0],
    [0, -1j * np.sin(t / 2), np.cos(t / 2), 0],
    [-1j * np.sin(t / 2), 0, 0, np.cos(t / 2)],
], dtype=complex), num_qubits=2)

# ── New: RZZ (2-qubit ZZ rotation) ──

RZZ = ParametricGate("RZZ", lambda t: np.diag([
    np.exp(-1j * t / 2),
    np.exp(1j * t / 2),
    np.exp(1j * t / 2),
    np.exp(-1j * t / 2),
]).astype(complex), num_qubits=2)

# ── New: U (Universal 1-qubit gate) ──

U = MultiParametricGate("U", lambda t, p, lam: np.array([
    [np.cos(t / 2), -np.exp(1j * lam) * np.sin(t / 2)],
    [np.exp(1j * p) * np.sin(t / 2),
     np.exp(1j * (p + lam)) * np.cos(t / 2)],
], dtype=complex), num_params=3)

# ── New v0.9: ECR (Echoed Cross-Resonance) ──
# IBM Heron native 2-qubit gate

class _ECR(Gate):
    """Echoed cross-resonance gate. IBM Heron native 2-qubit gate."""
    name = "ECR"
    num_qubits = 2
    def _build_matrix(self) -> np.ndarray:
        return np.array([
            [0, 0, 1, 1j],
            [0, 0, 1j, 1],
            [1, -1j, 0, 0],
            [-1j, 1, 0, 0],
        ], dtype=complex) * _SQRT2_INV

# ── New v0.9: iSWAP ──
# Google Sycamore native gate

class _iSWAP(Gate):
    """Imaginary SWAP gate. Google Sycamore native 2-qubit gate."""
    name = "iSWAP"
    num_qubits = 2
    def _build_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1],
        ], dtype=complex)

# ── New v0.9: CSWAP (Fredkin) ──

class _CSWAP(Gate):
    """Controlled-SWAP (Fredkin) gate. 3-qubit."""
    name = "CSWAP"
    num_qubits = 3
    def _build_matrix(self) -> np.ndarray:
        m = np.eye(8, dtype=complex)
        # Swap |101⟩ ↔ |110⟩
        m[5, 5], m[5, 6] = 0, 1
        m[6, 5], m[6, 6] = 1, 0
        return m

# ── New v0.9: CH (Controlled-Hadamard) ──

class _CH(Gate):
    """Controlled-Hadamard gate."""
    name = "CH"
    num_qubits = 2
    def _build_matrix(self) -> np.ndarray:
        m = np.eye(4, dtype=complex)
        m[2, 2] = _SQRT2_INV
        m[2, 3] = _SQRT2_INV
        m[3, 2] = _SQRT2_INV
        m[3, 3] = -_SQRT2_INV
        return m

# ── New v0.9: CP (Controlled-Phase) ──

CP = ParametricGate("CP", lambda t: np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, np.exp(1j * t)],
], dtype=complex), num_qubits=2)

# ── New v0.9: MS (Mølmer-Sørensen) ──
# IonQ / trapped-ion native gate

MS = ParametricGate("MS", lambda t: np.array([
    [np.cos(t / 2), 0, 0, -1j * np.sin(t / 2)],
    [0, np.cos(t / 2), -1j * np.sin(t / 2), 0],
    [0, -1j * np.sin(t / 2), np.cos(t / 2), 0],
    [-1j * np.sin(t / 2), 0, 0, np.cos(t / 2)],
], dtype=complex), num_qubits=2)

# ── Singleton Instances ──
H = _H()
X = _X()
Y = _Y()
Z = _Z()
S = _S()
T = _T()
I = _Identity()  # noqa: E741
SDG = _SDG()
TDG = _TDG()
SX = _SX()
SXdg = _SXdg()
CX = _CX()
CZ = _CZ()
CY = _CY()
SWAP = _SWAP()
CCX = _CCX()
RCCX = _RCCX()
RC3X = _RC3X()
ECR = _ECR()
iSWAP = _iSWAP()
CSWAP = _CSWAP()
CH = _CH()


GATE_REGISTRY: dict[str, Gate | ParametricGate] = {
    "H": H, "X": X, "Y": Y, "Z": Z, "S": S, "T": T,
    "I": I, "SDG": SDG, "TDG": TDG, "SX": SX, "SXdg": SXdg,
    "CX": CX, "CZ": CZ, "CY": CY, "SWAP": SWAP,
    "CCX": CCX, "RCCX": RCCX, "RC3X": RC3X,
    # New v0.9 fixed gates
    "ECR": ECR, "iSWAP": iSWAP, "CSWAP": CSWAP, "CH": CH,
    # Parametric gates
    "RX": RX, "RY": RY, "RZ": RZ, "P": P, "U": U,
    "RXX": RXX, "RZZ": RZZ,
    # New v0.9 parametric gates
    "CP": CP, "MS": MS,
}


# ── Inverse Pairs ──

_INVERSE_MAP: dict[str, str] = {
    "S": "SDG", "SDG": "S",
    "T": "TDG", "TDG": "T",
    "SX": "SXdg", "SXdg": "SX",
}

_SELF_INVERSE_GATES = frozenset({
    "H", "X", "Y", "Z", "I",
    "CX", "CZ", "CY", "SWAP", "CCX", "CSWAP",
    "ECR",
})


# ── MatrixGate (for computed inverse/controlled) ──

class _MatrixGate(Gate):
    """Gate constructed from an explicit matrix.

    Used internally by .inverse and .controlled() when no named
    gate exists for the result.
    """

    def __init__(self, name: str, matrix: np.ndarray, num_qubits: int) -> None:
        self.name = name
        self.num_qubits = num_qubits
        self._matrix = matrix

    def _build_matrix(self) -> np.ndarray:
        return self._matrix


