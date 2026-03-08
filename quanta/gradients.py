"""
quanta.gradients — Differentiable Quantum Computing.

Standalone gradient computation for parameterized quantum circuits.
Supports parameter-shift rule and finite-difference methods.

This module enables:
  - Gradient-based optimization of variational circuits
  - Hybrid quantum-classical machine learning workflows
  - Sensitivity analysis of quantum parameters

PennyLane-inspired API with Quanta-native integration.

Example:
    >>> from quanta.gradients import parameter_shift, finite_diff
    >>> from quanta.simulator.statevector import StateVectorSimulator
    >>>
    >>> def circuit_fn(params):
    ...     sim = StateVectorSimulator(2)
    ...     sim.apply("RY", (0,), (params[0],))
    ...     sim.apply("CX", (0, 1))
    ...     sim.apply("RZ", (1,), (params[1],))
    ...     return expectation(sim.state, "ZZ", 2)
    >>>
    >>> grads = parameter_shift(circuit_fn, [0.5, 1.2])
    >>> print(grads)  # [dE/d0, dE/d1]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "parameter_shift",
    "finite_diff",
    "expectation",
    "natural_gradient",
    "GradientResult",
]


# ── Pauli matrices ──
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


@dataclass
class GradientResult:
    """Result of gradient computation.

    Attributes:
        gradients: Gradient vector, one entry per parameter.
        function_value: The objective function value at the given params.
        num_evaluations: Number of circuit evaluations performed.
        method: Gradient method used.
    """

    gradients: np.ndarray
    function_value: float
    num_evaluations: int
    method: str

    def __repr__(self) -> str:
        return (
            f"GradientResult(method={self.method!r}, "
            f"value={self.function_value:.6f}, "
            f"grad_norm={np.linalg.norm(self.gradients):.6f})"
        )


# ═══════════════════════════════════════════
#  Expectation Value
# ═══════════════════════════════════════════

def expectation(
    statevector: np.ndarray,
    observable: str,
    num_qubits: int,
) -> float:
    """Compute expectation value <ψ|O|ψ> for a Pauli observable.

    Args:
        statevector: State vector (length 2^n).
        observable: Pauli string, e.g. "ZZ", "XI", "ZIZ".
        num_qubits: Number of qubits.

    Returns:
        Real-valued expectation <ψ|O|ψ>.

    Example:
        >>> sim = StateVectorSimulator(2)
        >>> sim.apply("H", (0,))
        >>> sim.apply("CX", (0, 1))
        >>> expectation(sim.state, "ZZ", 2)  # ≈ 1.0 for Bell state
    """
    padded = observable.ljust(num_qubits, "I")

    op = np.array([[1.0]], dtype=complex)
    for ch in padded:
        op = np.kron(op, _PAULI[ch])

    return float(np.real(statevector.conj() @ op @ statevector))


def multi_expectation(
    statevector: np.ndarray,
    observables: list[tuple[str, float]],
    num_qubits: int,
) -> float:
    """Compute expectation of a Hamiltonian = Σ cᵢ Oᵢ.

    Args:
        statevector: State vector.
        observables: List of (pauli_string, coefficient).
        num_qubits: Number of qubits.

    Returns:
        Σ cᵢ <ψ|Oᵢ|ψ>.
    """
    total = 0.0
    for obs, coeff in observables:
        total += coeff * expectation(statevector, obs, num_qubits)
    return total


# ═══════════════════════════════════════════
#  Parameter-Shift Rule
# ═══════════════════════════════════════════

def parameter_shift(
    circuit_fn: Callable[[np.ndarray], float],
    params: np.ndarray | list[float],
    shift: float = np.pi / 2,
) -> GradientResult:
    """Compute exact gradients using the parameter-shift rule.

    For quantum gates of the form exp(-iθG/2), the gradient is:
        ∂f/∂θ = [f(θ + s) - f(θ - s)] / (2 sin(s))

    With the standard shift s = π/2, this simplifies to:
        ∂f/∂θ = [f(θ + π/2) - f(θ - π/2)] / 2

    This gives **exact** gradients (not approximations) for
    parameterized rotation gates (RX, RY, RZ).

    Args:
        circuit_fn: Function mapping parameters → scalar expectation value.
        params: Parameter vector.
        shift: Shift value (default π/2 for exact gradients).

    Returns:
        GradientResult with exact gradients.

    Example:
        >>> def cost(p):
        ...     sim = StateVectorSimulator(1)
        ...     sim.apply("RY", (0,), (p[0],))
        ...     return expectation(sim.state, "Z", 1)
        >>> result = parameter_shift(cost, [0.5])
        >>> print(result.gradients)  # [-sin(0.5)]
    """
    params = np.asarray(params, dtype=float)
    num_params = len(params)
    gradients = np.zeros(num_params)

    f_0 = circuit_fn(params)
    scale = 2.0 * np.sin(shift)

    for i in range(num_params):
        params_plus = params.copy()
        params_plus[i] += shift
        f_plus = circuit_fn(params_plus)

        params_minus = params.copy()
        params_minus[i] -= shift
        f_minus = circuit_fn(params_minus)

        gradients[i] = (f_plus - f_minus) / scale

    return GradientResult(
        gradients=gradients,
        function_value=f_0,
        num_evaluations=2 * num_params + 1,
        method="parameter-shift",
    )


# ═══════════════════════════════════════════
#  Finite Difference
# ═══════════════════════════════════════════

def finite_diff(
    circuit_fn: Callable[[np.ndarray], float],
    params: np.ndarray | list[float],
    epsilon: float = 1e-7,
    method: str = "central",
) -> GradientResult:
    """Compute gradients using finite differences.

    Less accurate than parameter-shift but works for any
    differentiable function (not just rotation gates).

    Args:
        circuit_fn: Function mapping parameters → scalar value.
        params: Parameter vector.
        epsilon: Step size for finite differences.
        method: "central" (default, O(ε²)) or "forward" (O(ε)).

    Returns:
        GradientResult with approximate gradients.

    Example:
        >>> result = finite_diff(cost_fn, params, epsilon=1e-5)
    """
    params = np.asarray(params, dtype=float)
    num_params = len(params)
    gradients = np.zeros(num_params)
    f_0 = circuit_fn(params)

    if method == "central":
        for i in range(num_params):
            params_plus = params.copy()
            params_plus[i] += epsilon
            f_plus = circuit_fn(params_plus)

            params_minus = params.copy()
            params_minus[i] -= epsilon
            f_minus = circuit_fn(params_minus)

            gradients[i] = (f_plus - f_minus) / (2 * epsilon)

        num_evals = 2 * num_params + 1
    else:  # forward
        for i in range(num_params):
            params_plus = params.copy()
            params_plus[i] += epsilon
            f_plus = circuit_fn(params_plus)
            gradients[i] = (f_plus - f_0) / epsilon

        num_evals = num_params + 1

    return GradientResult(
        gradients=gradients,
        function_value=f_0,
        num_evaluations=num_evals,
        method=f"finite-diff-{method}",
    )


# ═══════════════════════════════════════════
#  Natural Gradient (Quantum Fisher Information)
# ═══════════════════════════════════════════

def natural_gradient(
    circuit_fn: Callable[[np.ndarray], float],
    state_fn: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray | list[float],
    shift: float = np.pi / 2,
    regularization: float = 1e-4,
) -> GradientResult:
    """Compute natural gradients using the Quantum Fisher Information Matrix.

    Natural gradient descent follows the steepest descent direction
    on the parameter manifold, accounting for the geometry of the
    quantum state space. This typically converges faster than
    vanilla gradient descent.

    g_nat = F⁻¹ · g

    Where F is the Fubini-Study metric (QFIM) and g is the
    parameter-shift gradient.

    Args:
        circuit_fn: Function mapping params → scalar expectation.
        state_fn: Function mapping params → statevector.
        params: Parameter vector.
        shift: Shift for parameter-shift gradient.
        regularization: Tikhonov regularization for QFIM inversion.

    Returns:
        GradientResult with natural gradients.
    """
    params = np.asarray(params, dtype=float)
    num_params = len(params)

    # Step 1: Compute standard parameter-shift gradient
    ps_result = parameter_shift(circuit_fn, params, shift)
    grad = ps_result.gradients

    # Step 2: Estimate Quantum Fisher Information Matrix (QFIM)
    # Using parameter-shift rule on state overlaps
    psi_0 = state_fn(params)
    F = np.zeros((num_params, num_params))

    for i in range(num_params):
        params_plus_i = params.copy()
        params_plus_i[i] += shift

        params_minus_i = params.copy()
        params_minus_i[i] -= shift

        psi_plus_i = state_fn(params_plus_i)
        psi_minus_i = state_fn(params_minus_i)

        # d|ψ⟩/dθᵢ ≈ (|ψ(θ+s)⟩ - |ψ(θ-s)⟩) / (2 sin(s))
        dpsi_i = (psi_plus_i - psi_minus_i) / (2 * np.sin(shift))

        for j in range(i, num_params):
            if i == j:
                # Diagonal: F_ii = 4(⟨∂ᵢψ|∂ᵢψ⟩ - |⟨ψ|∂ᵢψ⟩|²)
                overlap = np.vdot(dpsi_i, dpsi_i)
                cross = np.vdot(psi_0, dpsi_i)
                F[i, i] = float(np.real(4 * (overlap - abs(cross) ** 2)))
            else:
                params_plus_j = params.copy()
                params_plus_j[j] += shift
                params_minus_j = params.copy()
                params_minus_j[j] -= shift

                psi_plus_j = state_fn(params_plus_j)
                psi_minus_j = state_fn(params_minus_j)
                dpsi_j = (psi_plus_j - psi_minus_j) / (2 * np.sin(shift))

                # Off-diagonal: F_ij = 4 Re(⟨∂ᵢψ|∂ⱼψ⟩ - ⟨∂ᵢψ|ψ⟩⟨ψ|∂ⱼψ⟩)
                overlap = np.vdot(dpsi_i, dpsi_j)
                cross_i = np.vdot(dpsi_i, psi_0)
                cross_j = np.vdot(psi_0, dpsi_j)
                F[i, j] = float(np.real(4 * (overlap - cross_i * cross_j)))
                F[j, i] = F[i, j]

    # Step 3: Regularize and invert QFIM
    F_reg = F + regularization * np.eye(num_params)
    nat_grad = np.linalg.solve(F_reg, grad)

    return GradientResult(
        gradients=nat_grad,
        function_value=ps_result.function_value,
        num_evaluations=(
            ps_result.num_evaluations
            + 4 * num_params
            + 2 * num_params * (num_params - 1)
        ),
        method="natural-gradient",
    )
