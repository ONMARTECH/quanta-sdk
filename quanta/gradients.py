"""
quanta.gradients — Differentiable Quantum Computing.

Standalone exact and numerical gradient engine for parameterized quantum circuits:
  - 2-term & Anselmetti 4-term exact parameter-shift rules (including controlled rotations CRX/CRY/CRZ)
  - Jones & Gacon (arXiv:2009.02823) O(G * 2^n) reverse-mode statevector adjoint differentiation
  - Mari–Bromley–Killoran (Phys. Rev. A 103, 012405, 2021) gauge-invariant overlap QFIM & natural gradient
  - Central and forward finite differences
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from quanta.core.circuit import CircuitBuilder, CircuitDefinition
from quanta.core.gates import GATE_REGISTRY, Gate, MultiParametricGate, ParametricGate
from quanta.core.types import Instruction
from quanta.dag.dag_circuit import DAGCircuit

__all__ = [
    "parameter_shift",
    "finite_diff",
    "expectation",
    "multi_expectation",
    "quantum_fisher_information",
    "natural_gradient",
    "adjoint_gradient",
    "GradientResult",
]


# ── Pauli matrices ──
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}

# Controlled rotations with 3 generator eigenvalues {-1/2, 0, +1/2} requiring 4-term shift
_FOUR_TERM_GATES = frozenset({"CRX", "CRY", "CRZ"})


@dataclass
class GradientResult:
    """Result of gradient computation.

    Attributes:
        gradients: Gradient vector, one entry per parameter.
        function_value: The objective function value at the given params.
        num_evaluations: Number of circuit evaluations performed.
        method: Gradient method used.
        qfim: Optional Quantum Fisher Information Matrix (populated by natural_gradient).
    """

    gradients: np.ndarray
    function_value: float
    num_evaluations: int
    method: str
    qfim: np.ndarray | None = None

    def __repr__(self) -> str:
        return (
            f"GradientResult(method={self.method!r}, "
            f"value={self.function_value:.6f}, "
            f"grad_norm={np.linalg.norm(self.gradients):.6f})"
        )


# ═══════════════════════════════════════════
#  Fast O(n * 2^n) Pauli & Expectation Helpers
# ═══════════════════════════════════════════

def _apply_pauli_string(psi: np.ndarray, pauli_str: str, num_qubits: int) -> np.ndarray:
    """Applies a Pauli tensor string P to statevector psi in O(n * 2^n) without 4^n matrices."""
    padded = pauli_str.upper().ljust(num_qubits, "I")
    if len(padded) > num_qubits:
        raise ValueError(
            f"Observable '{pauli_str}' length exceeds num_qubits={num_qubits}."
        )
    tensor = np.asarray(psi, dtype=np.complex128).reshape((2,) * num_qubits).copy()
    for q, ch in enumerate(padded):
        if ch == "I":
            continue
        mat = _PAULI[ch]
        contracted = np.tensordot(mat, tensor, axes=([1], [q]))
        tensor = np.moveaxis(contracted, 0, q)
    return tensor.reshape(-1)


def expectation(
    statevector: np.ndarray,
    observable: str,
    num_qubits: int,
) -> float:
    """Compute expectation value <ψ|O|ψ> for a Pauli observable in O(n * 2^n)."""
    psi = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    o_psi = _apply_pauli_string(psi, observable, num_qubits)
    return float(np.real(np.vdot(psi, o_psi)))


def multi_expectation(
    statevector: np.ndarray,
    observables: list[tuple[str, float]],
    num_qubits: int,
) -> float:
    """Compute expectation of a Hamiltonian = Σ cᵢ Oᵢ."""
    total = 0.0
    for obs, coeff in observables:
        total += float(coeff) * expectation(statevector, obs, num_qubits)
    return total


# ═══════════════════════════════════════════
#  Parameter-Shift Rule (2-Term & 4-Term)
# ═══════════════════════════════════════════

def parameter_shift(
    circuit_fn: Callable[[np.ndarray], float],
    params: np.ndarray | Sequence[float],
    shift: float = np.pi / 2,
    rule: str = "auto",
    gate_types: Sequence[str] | None = None,
) -> GradientResult:
    """Compute exact analytical gradients via 2-term or Anselmetti 4-term parameter-shift.

    - 2-term rule (for 2-eigenvalue Pauli generators RX, RY, RZ, RXX, RZZ, MS, P, CP):
        ∂f/∂θ = [f(θ + s) - f(θ - s)] / (2 sin(s))
    - 4-term rule (Anselmetti et al., New J. Phys. 23, 113010, 2021, for 3-eigenvalue
      controlled rotations CRX, CRY, CRZ, and also exact for all 2-eigenvalue gates):
        ∂f/∂θ = c₁ [f(θ + π/2) - f(θ - π/2)] - c₂ [f(θ + 3π/2) - f(θ - 3π/2)]
      where c₁ = (√2 + 1) / (4√2) and c₂ = (√2 - 1) / (4√2).

    Args:
        circuit_fn: Function mapping parameters → scalar expectation value.
        params: Parameter vector.
        shift: Shift angle for the 2-term rule (default π/2).
        rule: "auto" (uses 4-term when gate_types[i] is CRX/CRY/CRZ, else 2-term),
            "two-term", or "four-term".
        gate_types: Optional per-parameter gate names enabling automatic 4-term dispatch.

    Returns:
        GradientResult with exact analytical gradients.
    """
    if rule not in {"auto", "two-term", "four-term"}:
        raise ValueError(
            f"Unsupported parameter-shift rule '{rule}'. "
            "Expected 'auto', 'two-term', or 'four-term'."
        )

    params_arr = np.asarray(params, dtype=float)
    num_params = len(params_arr)
    if gate_types is not None and len(gate_types) != num_params:
        raise ValueError(
            f"gate_types length ({len(gate_types)}) must match params length ({num_params})."
        )

    gradients = np.zeros(num_params, dtype=float)
    f_0 = float(circuit_fn(params_arr))
    num_evals = 1

    scale_2 = 2.0 * np.sin(shift)
    sqrt2 = math.sqrt(2.0)
    c1 = (sqrt2 + 1.0) / (4.0 * sqrt2)
    c2 = (sqrt2 - 1.0) / (4.0 * sqrt2)
    s1 = math.pi / 2.0
    s2 = 3.0 * math.pi / 2.0

    for i in range(num_params):
        use_four_term = rule == "four-term" or (
            rule == "auto"
            and gate_types is not None
            and gate_types[i].upper() in _FOUR_TERM_GATES
        )

        if use_four_term:
            p_p1 = params_arr.copy()
            p_p1[i] += s1
            p_m1 = params_arr.copy()
            p_m1[i] -= s1
            p_p2 = params_arr.copy()
            p_p2[i] += s2
            p_m2 = params_arr.copy()
            p_m2[i] -= s2

            f_p1 = float(circuit_fn(p_p1))
            f_m1 = float(circuit_fn(p_m1))
            f_p2 = float(circuit_fn(p_p2))
            f_m2 = float(circuit_fn(p_m2))
            gradients[i] = c1 * (f_p1 - f_m1) - c2 * (f_p2 - f_m2)
            num_evals += 4
        else:
            params_plus = params_arr.copy()
            params_plus[i] += shift
            f_plus = float(circuit_fn(params_plus))

            params_minus = params_arr.copy()
            params_minus[i] -= shift
            f_minus = float(circuit_fn(params_minus))

            gradients[i] = (f_plus - f_minus) / scale_2
            num_evals += 2

    method_label = (
        "parameter-shift-4term" if rule == "four-term" else "parameter-shift"
    )
    return GradientResult(
        gradients=gradients,
        function_value=f_0,
        num_evaluations=num_evals,
        method=method_label,
    )


# ═══════════════════════════════════════════
#  Finite Difference
# ═══════════════════════════════════════════

def finite_diff(
    circuit_fn: Callable[[np.ndarray], float],
    params: np.ndarray | Sequence[float],
    epsilon: float = 1e-7,
    method: str = "central",
) -> GradientResult:
    """Compute gradients using finite differences."""
    params_arr = np.asarray(params, dtype=float)
    num_params = len(params_arr)
    gradients = np.zeros(num_params, dtype=float)
    f_0 = float(circuit_fn(params_arr))

    if method == "central":
        for i in range(num_params):
            params_plus = params_arr.copy()
            params_plus[i] += epsilon
            f_plus = float(circuit_fn(params_plus))

            params_minus = params_arr.copy()
            params_minus[i] -= epsilon
            f_minus = float(circuit_fn(params_minus))

            gradients[i] = (f_plus - f_minus) / (2.0 * epsilon)

        num_evals = 2 * num_params + 1
    else:  # forward
        for i in range(num_params):
            params_plus = params_arr.copy()
            params_plus[i] += epsilon
            f_plus = float(circuit_fn(params_plus))
            gradients[i] = (f_plus - f_0) / epsilon

        num_evals = num_params + 1

    return GradientResult(
        gradients=gradients,
        function_value=f_0,
        num_evaluations=num_evals,
        method=f"finite-diff-{method}",
    )


# ═══════════════════════════════════════════
#  Exact Overlap QFIM & Natural Gradient
# ═══════════════════════════════════════════

def _compute_qfim_mari_bromley_killoran(
    state_fn: Callable[[np.ndarray], np.ndarray],
    params_arr: np.ndarray,
    shift: float = np.pi / 2,
) -> tuple[np.ndarray, int]:
    """Computes the exact gauge-invariant QFIM via Mari–Bromley–Killoran overlap shifts.

    Reference:
        Mari, Bromley, Killoran, Phys. Rev. A 103, 012405 (2021).
        Diagonal:     F_ii = [2 - |<ψ₀|ψ_{+i}>|² - |<ψ₀|ψ_{-i}>|²] / (2 sin²(s/2))
        Off-diagonal: F_ij = -[|<ψ₀|ψ_{+i,+j}>|² - |<ψ₀|ψ_{+i,-j}>|²
                               - |<ψ₀|ψ_{-i,+j}>|² + |<ψ₀|ψ_{-i,-j}>|²] / (2 sin²(s))
    """
    num_params = len(params_arr)
    psi_0 = np.asarray(state_fn(params_arr), dtype=np.complex128).reshape(-1)
    evals = 1

    # Precompute single-shifted states ONCE: 2 * num_params evaluations
    psi_plus: list[np.ndarray] = []
    psi_minus: list[np.ndarray] = []
    for i in range(num_params):
        p_plus = params_arr.copy()
        p_plus[i] += shift
        p_minus = params_arr.copy()
        p_minus[i] -= shift
        psi_plus.append(np.asarray(state_fn(p_plus), dtype=np.complex128).reshape(-1))
        psi_minus.append(np.asarray(state_fn(p_minus), dtype=np.complex128).reshape(-1))
        evals += 2

    F = np.zeros((num_params, num_params), dtype=float)
    denom_diag = 2.0 * (math.sin(shift / 2.0) ** 2)
    denom_off = 2.0 * (math.sin(shift) ** 2)

    for i in range(num_params):
        ov_p = abs(np.vdot(psi_0, psi_plus[i])) ** 2
        ov_m = abs(np.vdot(psi_0, psi_minus[i])) ** 2
        F[i, i] = float((2.0 - ov_p - ov_m) / denom_diag)

        for j in range(i + 1, num_params):
            p_pp = params_arr.copy()
            p_pp[i] += shift
            p_pp[j] += shift

            p_pm = params_arr.copy()
            p_pm[i] += shift
            p_pm[j] -= shift

            p_mp = params_arr.copy()
            p_mp[i] -= shift
            p_mp[j] += shift

            p_mm = params_arr.copy()
            p_mm[i] -= shift
            p_mm[j] -= shift

            f_pp = abs(np.vdot(psi_0, np.asarray(state_fn(p_pp), dtype=np.complex128).reshape(-1))) ** 2
            f_pm = abs(np.vdot(psi_0, np.asarray(state_fn(p_pm), dtype=np.complex128).reshape(-1))) ** 2
            f_mp = abs(np.vdot(psi_0, np.asarray(state_fn(p_mp), dtype=np.complex128).reshape(-1))) ** 2
            f_mm = abs(np.vdot(psi_0, np.asarray(state_fn(p_mm), dtype=np.complex128).reshape(-1))) ** 2
            evals += 4

            f_ij = -float((f_pp - f_pm - f_mp + f_mm) / denom_off)
            F[i, j] = f_ij
            F[j, i] = f_ij

    return F, evals


def quantum_fisher_information(
    state_fn: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray | Sequence[float],
    shift: float = np.pi / 2,
) -> np.ndarray:
    """Computes the exact Fubini-Study Quantum Fisher Information Matrix (QFIM)."""
    params_arr = np.asarray(params, dtype=float)
    F, _ = _compute_qfim_mari_bromley_killoran(state_fn, params_arr, shift=shift)
    return F


def natural_gradient(
    circuit_fn: Callable[[np.ndarray], float],
    state_fn: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray | Sequence[float],
    shift: float = np.pi / 2,
    regularization: float = 1e-4,
) -> GradientResult:
    """Compute natural gradients using the exact gauge-invariant overlap QFIM.

    Solves (F + λ I) g_nat = g where F is the Fubini-Study QFIM computed via
    Mari–Bromley–Killoran (Phys. Rev. A 103, 012405, 2021) overlap parameter shifts.
    """
    params_arr = np.asarray(params, dtype=float)
    num_params = len(params_arr)

    ps_result = parameter_shift(circuit_fn, params_arr, shift=shift)
    grad = ps_result.gradients

    F, qfim_evals = _compute_qfim_mari_bromley_killoran(state_fn, params_arr, shift=shift)

    F_reg = F + regularization * np.eye(num_params, dtype=float)
    nat_grad = np.linalg.solve(F_reg, grad)

    return GradientResult(
        gradients=nat_grad,
        function_value=ps_result.function_value,
        num_evaluations=ps_result.num_evaluations + qfim_evals,
        method="natural-gradient",
        qfim=F,
    )


# ═══════════════════════════════════════════
#  O(G * 2^n) Reverse-Mode Adjoint Differentiation
# ═══════════════════════════════════════════

class _TrackedParam(float):
    """Lightweight forward-tangent scalar recording ∂θ/∂p_i during circuit construction."""

    _sens: dict[int, float]

    def __new__(cls, val: float, sens: dict[int, float] | None = None) -> _TrackedParam:
        obj = super().__new__(cls, float(val))
        obj._sens = dict(sens) if sens else {}
        return obj

    def __add__(self, other: Any) -> _TrackedParam:
        if isinstance(other, _TrackedParam):
            merged = dict(self._sens)
            for k, v in other._sens.items():
                merged[k] = merged.get(k, 0.0) + v
            return _TrackedParam(float(self) + float(other), merged)
        if isinstance(other, (int, float, np.floating)):
            return _TrackedParam(float(self) + float(other), self._sens)
        return NotImplemented

    def __radd__(self, other: Any) -> _TrackedParam:
        return self.__add__(other)

    def __neg__(self) -> _TrackedParam:
        return _TrackedParam(-float(self), {k: -v for k, v in self._sens.items()})

    def __pos__(self) -> _TrackedParam:
        return self

    def __sub__(self, other: Any) -> _TrackedParam:
        if isinstance(other, _TrackedParam):
            return self.__add__(-other)
        if isinstance(other, (int, float, np.floating)):
            return _TrackedParam(float(self) - float(other), self._sens)
        return NotImplemented

    def __rsub__(self, other: Any) -> _TrackedParam:
        if isinstance(other, (int, float, np.floating)):
            return _TrackedParam(float(other) - float(self), {k: -v for k, v in self._sens.items()})
        return NotImplemented

    def __mul__(self, other: Any) -> _TrackedParam:
        if isinstance(other, _TrackedParam):
            v1, v2 = float(self), float(other)
            merged: dict[int, float] = {}
            for k, g in self._sens.items():
                merged[k] = merged.get(k, 0.0) + g * v2
            for k, g in other._sens.items():
                merged[k] = merged.get(k, 0.0) + v1 * g
            return _TrackedParam(v1 * v2, merged)
        if isinstance(other, (int, float, np.floating)):
            s = float(other)
            return _TrackedParam(float(self) * s, {k: v * s for k, v in self._sens.items()})
        return NotImplemented

    def __rmul__(self, other: Any) -> _TrackedParam:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> _TrackedParam:
        if isinstance(other, _TrackedParam):
            v1, v2 = float(self), float(other)
            merged: dict[int, float] = {}
            for k, g in self._sens.items():
                merged[k] = merged.get(k, 0.0) + g / v2
            for k, g in other._sens.items():
                merged[k] = merged.get(k, 0.0) - (v1 * g) / (v2 * v2)
            return _TrackedParam(v1 / v2, merged)
        if isinstance(other, (int, float, np.floating)):
            s = float(other)
            return _TrackedParam(float(self) / s, {k: v / s for k, v in self._sens.items()})
        return NotImplemented


def _gate_matrix_and_derivatives(
    gate_name: str,
    raw_params: tuple[float, ...],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Returns exact unitary matrix U(θ) and ∂U/∂θ_m for every parameter slot m."""
    entry = GATE_REGISTRY.get(gate_name)
    if entry is None:
        raise ValueError(f"Unknown gate '{gate_name}' in adjoint_gradient.")

    if isinstance(entry, Gate):
        return np.asarray(entry.matrix, dtype=np.complex128), []

    if isinstance(entry, ParametricGate):
        t = float(raw_params[0])
        u_mat = np.asarray(entry(t).matrix, dtype=np.complex128)

        # Analytical generator derivatives ∂U/∂t
        c, s = math.cos(t / 2.0), math.sin(t / 2.0)
        if gate_name == "RX":
            du = 0.5 * np.array([[-s, -1j * c], [-1j * c, -s]], dtype=np.complex128)
        elif gate_name == "RY":
            du = 0.5 * np.array([[-s, -c], [c, -s]], dtype=np.complex128)
        elif gate_name == "RZ":
            du = np.array(
                [[-0.5j * np.exp(-0.5j * t), 0.0], [0.0, 0.5j * np.exp(0.5j * t)]],
                dtype=np.complex128,
            )
        elif gate_name == "P":
            du = np.array([[0.0, 0.0], [0.0, 1j * np.exp(1j * t)]], dtype=np.complex128)
        elif gate_name in ("RXX", "MS"):
            du = 0.5 * np.array(
                [
                    [-s, 0.0, 0.0, -1j * c],
                    [0.0, -s, -1j * c, 0.0],
                    [0.0, -1j * c, -s, 0.0],
                    [-1j * c, 0.0, 0.0, -s],
                ],
                dtype=np.complex128,
            )
        elif gate_name == "RZZ":
            ep = np.exp(0.5j * t)
            em = np.exp(-0.5j * t)
            du = np.diag([-0.5j * em, 0.5j * ep, 0.5j * ep, -0.5j * em]).astype(np.complex128)
        elif gate_name == "CRX":
            du = np.zeros((4, 4), dtype=np.complex128)
            du[2:, 2:] = 0.5 * np.array([[-s, -1j * c], [-1j * c, -s]], dtype=np.complex128)
        elif gate_name == "CRY":
            du = np.zeros((4, 4), dtype=np.complex128)
            du[2:, 2:] = 0.5 * np.array([[-s, -c], [c, -s]], dtype=np.complex128)
        elif gate_name == "CRZ":
            du = np.zeros((4, 4), dtype=np.complex128)
            du[2, 2] = -0.5j * np.exp(-0.5j * t)
            du[3, 3] = 0.5j * np.exp(0.5j * t)
        elif gate_name == "CP":
            du = np.zeros((4, 4), dtype=np.complex128)
            du[3, 3] = 1j * np.exp(1j * t)
        else:
            # High-accuracy 4-point central difference on the tiny 2^k x 2^k gate matrix
            h = 1e-6
            m_p1 = np.asarray(entry(t + h).matrix, dtype=np.complex128)
            m_m1 = np.asarray(entry(t - h).matrix, dtype=np.complex128)
            m_p2 = np.asarray(entry(t + 2.0 * h).matrix, dtype=np.complex128)
            m_m2 = np.asarray(entry(t - 2.0 * h).matrix, dtype=np.complex128)
            du = (8.0 * (m_p1 - m_m1) - (m_p2 - m_m2)) / (12.0 * h)
        return u_mat, [du]

    if isinstance(entry, MultiParametricGate):
        vals = tuple(float(x) for x in raw_params)
        u_mat = np.asarray(entry(*vals).matrix, dtype=np.complex128)
        if gate_name == "U" and len(vals) == 3:
            t, p, lam = vals
            ct, st = math.cos(t / 2.0), math.sin(t / 2.0)
            ep = np.exp(1j * p)
            el = np.exp(1j * lam)
            epl = np.exp(1j * (p + lam))
            du_dt = 0.5 * np.array(
                [[-st, -el * ct], [ep * ct, -epl * st]], dtype=np.complex128
            )
            du_dp = np.array(
                [[0.0, 0.0], [1j * ep * st, 1j * epl * ct]], dtype=np.complex128
            )
            du_dlam = np.array(
                [[0.0, -1j * el * st], [0.0, 1j * epl * ct]], dtype=np.complex128
            )
            return u_mat, [du_dt, du_dp, du_dlam]

        derivs: list[np.ndarray] = []
        h = 1e-6
        for idx in range(len(vals)):
            vp1 = list(vals)
            vp1[idx] += h
            vm1 = list(vals)
            vm1[idx] -= h
            vp2 = list(vals)
            vp2[idx] += 2.0 * h
            vm2 = list(vals)
            vm2[idx] -= 2.0 * h
            du_m = (
                8.0 * (np.asarray(entry(*vp1).matrix) - np.asarray(entry(*vm1).matrix))
                - (np.asarray(entry(*vp2).matrix) - np.asarray(entry(*vm2).matrix))
            ) / (12.0 * h)
            derivs.append(np.asarray(du_m, dtype=np.complex128))
        return u_mat, derivs

    raise TypeError(f"Unsupported gate entry type for '{gate_name}': {type(entry)}")


def _apply_local_operator(
    state_tensor: np.ndarray,
    op_matrix: np.ndarray,
    qubits: tuple[int, ...],
) -> np.ndarray:
    """Applies a k-qubit matrix (2^k x 2^k) to an n-qubit state tensor (2,)*n in O(2^n)."""
    k = len(qubits)
    op_tensor = op_matrix.reshape((2,) * (2 * k))
    contracted = np.tensordot(op_tensor, state_tensor, axes=(list(range(k, 2 * k)), list(qubits)))
    return np.moveaxis(contracted, list(range(k)), list(qubits))


def _extract_tracked_ops(
    circuit_or_ops: CircuitDefinition | Callable[..., Any] | DAGCircuit | Sequence[Any],
    tracked_params: list[_TrackedParam],
    param_names: list[str] | None,
    num_qubits: int | None,
) -> tuple[int, list[tuple[str, tuple[int, ...], tuple[Any, ...]]]]:
    """Builds the gate list once with _TrackedParam scalars to capture parameter map."""
    if isinstance(circuit_or_ops, CircuitDefinition):
        sig = inspect.signature(circuit_or_ops._fn)
        fn_params = list(sig.parameters.keys())[1:]  # skip register `q`
        builder = CircuitBuilder(circuit_or_ops.num_qubits)
        from quanta.core.types import QubitRegister

        reg = QubitRegister(circuit_or_ops.num_qubits)
        with builder:
            if not fn_params:
                circuit_or_ops._fn(reg)
            elif param_names is not None:
                kw = dict(zip(param_names, tracked_params, strict=True))
                circuit_or_ops._fn(reg, **kw)
            elif len(fn_params) == 1 and len(tracked_params) != 1:
                circuit_or_ops._fn(reg, tracked_params)
            elif len(fn_params) == len(tracked_params):
                # Support both `def circ(q, a, b)` and `def circ(q, params)` when len==1
                first_param = sig.parameters[fn_params[0]]
                if len(fn_params) == 1 and first_param.name in ("params", "p", "weights", "thetas"):
                    try:
                        circuit_or_ops._fn(reg, tracked_params)
                    except (TypeError, IndexError):
                        builder.instructions.clear()
                        circuit_or_ops._fn(reg, tracked_params[0])
                else:
                    circuit_or_ops._fn(reg, *tracked_params)
            else:
                circuit_or_ops._fn(reg, tracked_params)
        ops = [(inst.gate_name, inst.qubits, inst.params) for inst in builder.instructions]
        return circuit_or_ops.num_qubits, ops

    if isinstance(circuit_or_ops, DAGCircuit):
        ops = [(op.gate_name, op.qubits, op.params) for op in circuit_or_ops.op_nodes()]
        return circuit_or_ops.num_qubits, ops

    if callable(circuit_or_ops):
        built = circuit_or_ops(tracked_params)
        if isinstance(built, DAGCircuit):
            ops = [(op.gate_name, op.qubits, op.params) for op in built.op_nodes()]
            return built.num_qubits, ops
        if isinstance(built, CircuitBuilder):
            ops = [(inst.gate_name, inst.qubits, inst.params) for inst in built.instructions]
            return built.num_qubits, ops
        circuit_or_ops = built

    # Declarative sequence of Instructions or (gate_name, qubits[, params_or_param_idx]) tuples
    raw_ops: list[tuple[str, tuple[int, ...], tuple[Any, ...]]] = []
    max_q = -1
    for item in circuit_or_ops:
        if isinstance(item, Instruction):
            g_name, qs, ps = item.gate_name, item.qubits, item.params
        elif isinstance(item, tuple):
            g_name = item[0]
            qs = (int(item[1]),) if isinstance(item[1], int) else tuple(int(q) for q in item[1])
            if len(item) == 2 or item[2] is None:
                ps = ()
            elif isinstance(item[2], int):
                # Parameter index shorthand: ("RY", (0,), 0) -> tracked_params[0]
                ps = (tracked_params[item[2]],)
            elif isinstance(item[2], (tuple, list)):
                ps = tuple(item[2])
            else:
                ps = (item[2],)
        else:
            raise TypeError(f"Unsupported operation specification in adjoint_gradient: {item!r}")
        for q in qs:
            max_q = max(max_q, q)
        raw_ops.append((g_name, qs, ps))

    n_q = num_qubits if num_qubits is not None else (max_q + 1 if max_q >= 0 else 1)
    return n_q, raw_ops


def adjoint_gradient(
    circuit: CircuitDefinition | Callable[..., Any] | DAGCircuit | Sequence[Any],
    params: np.ndarray | Sequence[float] | dict[str, float],
    observable: str | Sequence[tuple[str, float]] | np.ndarray,
    num_qubits: int | None = None,
) -> GradientResult:
    """Computes exact O(G * 2^n) reverse-mode statevector gradients via Adjoint Differentiation.

    Implements Algorithm 1 of Jones & Gacon (2020, arXiv:2009.02823):
      1. Forward pass: evolve |ψ⟩ = U_{G-1} ... U_0 |0...0⟩ in O(G * 2^n).
      2. Co-state initialization: |λ⟩ = O |ψ⟩ and f(θ) = Re⟨ψ|λ⟩.
      3. Reverse sweep (k = G-1 ... 0):
         - Uncompute |ψ_{k-1}⟩ = U_k† |ψ_k⟩
         - Accumulate ∂f/∂θ_k = 2 Re ⟨λ_k | (∂U_k/∂θ_k) | ψ_{k-1}⟩
         - Backpropagate |λ_{k-1}⟩ = U_k† |λ_k⟩
    Total runtime is O(G * 2^n) — completely independent of parameter count P!

    Args:
        circuit: A `@circuit` `CircuitDefinition`, a callable `fn(params)` returning a
            `DAGCircuit`/`CircuitBuilder`/instruction list, or a sequence of gate specs.
        params: Parameter vector (1D array/list) or `{param_name: value}` dict.
        observable: Pauli string (e.g. `"ZZ"`), list of `(pauli_str, coeff)` terms,
            or explicit `(2^n, 2^n)` Hermitian matrix.
        num_qubits: Optional qubit count when `circuit` is a raw operation list.

    Returns:
        GradientResult with exact analytical gradients (`method="adjoint"`, `num_evaluations=1`).
    """
    if isinstance(params, dict):
        param_names: list[str] | None = list(params.keys())
        params_arr = np.asarray([float(v) for v in params.values()], dtype=float)
    else:
        param_names = None
        params_arr = np.asarray(params, dtype=float).reshape(-1)

    num_params = len(params_arr)
    tracked = [_TrackedParam(float(val), {i: 1.0}) for i, val in enumerate(params_arr)]

    n_qubits, ops = _extract_tracked_ops(circuit, tracked, param_names, num_qubits)
    dim = 1 << n_qubits

    # 1. Forward pass: |ψ_G⟩ = U_{G-1} ... U_0 |0...0⟩
    psi_tensor = np.zeros((2,) * n_qubits, dtype=np.complex128)
    psi_tensor[(0,) * n_qubits] = 1.0

    compiled_ops: list[
        tuple[tuple[int, ...], np.ndarray, list[tuple[np.ndarray, dict[int, float]]]]
    ] = []

    for gate_name, qubits, raw_params in ops:
        if gate_name in ("I", "ID"):
            continue
        float_params = tuple(float(p) for p in raw_params)
        u_mat, du_list = _gate_matrix_and_derivatives(gate_name, float_params)

        active_derivs: list[tuple[np.ndarray, dict[int, float]]] = []
        for slot_idx, p_obj in enumerate(raw_params):
            if isinstance(p_obj, _TrackedParam) and p_obj._sens:
                active_derivs.append((du_list[slot_idx], p_obj._sens))

        psi_tensor = _apply_local_operator(psi_tensor, u_mat, qubits)
        compiled_ops.append((qubits, u_mat, active_derivs))

    # 2. Initialize adjoint state |λ⟩ = O |ψ⟩ and objective value f_0 = Re⟨ψ|λ⟩
    psi_flat = psi_tensor.reshape(dim)
    if isinstance(observable, str):
        lam_flat = _apply_pauli_string(psi_flat, observable, n_qubits)
    elif isinstance(observable, np.ndarray):
        if observable.shape != (dim, dim):
            raise ValueError(
                f"Observable matrix shape {observable.shape} does not match ({dim}, {dim})."
            )
        lam_flat = np.asarray(observable, dtype=np.complex128) @ psi_flat
    else:
        lam_flat = np.zeros(dim, dtype=np.complex128)
        for p_str, coeff in observable:
            lam_flat += float(coeff) * _apply_pauli_string(psi_flat, p_str, n_qubits)

    f_0 = float(np.real(np.vdot(psi_flat, lam_flat)))
    lam_tensor = lam_flat.reshape((2,) * n_qubits)

    # 3. Reverse sweep: O(G * 2^n)
    gradients = np.zeros(num_params, dtype=float)
    for qubits, u_mat, active_derivs in reversed(compiled_ops):
        u_dag = u_mat.conj().T
        # Uncompute state prior to gate k: |ψ_{k-1}⟩ = U_k† |ψ_k⟩
        psi_tensor = _apply_local_operator(psi_tensor, u_dag, qubits)

        if active_derivs:
            for du_mat, sens_map in active_derivs:
                phi_tensor = _apply_local_operator(psi_tensor, du_mat, qubits)
                grad_slot = 2.0 * float(np.real(np.vdot(lam_tensor, phi_tensor)))
                for p_idx, chain_scale in sens_map.items():
                    gradients[p_idx] += chain_scale * grad_slot

        # Backpropagate co-state: |λ_{k-1}⟩ = U_k† |λ_k⟩
        lam_tensor = _apply_local_operator(lam_tensor, u_dag, qubits)

    return GradientResult(
        gradients=gradients,
        function_value=f_0,
        num_evaluations=1,
        method="adjoint",
    )
