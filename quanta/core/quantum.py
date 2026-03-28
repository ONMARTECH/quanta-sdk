"""
quanta.core.quantum -- @quantum decorator for auto-differentiable circuits.

PennyLane @qml.qnode equivalent: wraps a circuit function to return
expectation values, supports parameter-shift gradients, and enables
both sync and async execution.

Example:
    >>> from quanta.core.quantum import quantum
    >>> from quanta import H, CX, RY, measure
    >>>
    >>> @quantum(qubits=2, diff_method="parameter-shift")
    ... def circuit(q, theta=0.0):
    ...     RY(theta)(q[0])
    ...     CX(q[0], q[1])
    ...     return measure(q)
    >>>
    >>> result = circuit(theta=0.5)
    >>> grad = circuit.gradient(theta=0.5)
"""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from quanta.core.circuit import circuit as _circuit_decorator
from quanta.result import Result
from quanta.runner import run

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["quantum"]


@dataclass
class GradientResult:
    """Result of gradient computation.

    Attributes:
        gradients: Dict of parameter name → gradient value.
        cost: The cost function value at the evaluation point.
    """

    gradients: dict[str, float]
    cost: float

    def __repr__(self) -> str:
        g = {k: round(v, 6) for k, v in self.gradients.items()}
        return f"GradientResult(cost={self.cost:.6f}, gradients={g})"


class QuantumFunction:
    """Wrapper for a quantum circuit with gradient support.

    Created by the @quantum decorator. Provides:
    - __call__: Execute circuit with parameters
    - gradient: Compute parameter-shift gradients
    - run_async: Async execution
    """

    def __init__(
        self,
        fn: Callable,
        qubits: int,
        shots: int,
        diff_method: str,
        observable: list[tuple[str, float]] | None,
        seed: int | None,
    ) -> None:
        self._fn = fn
        self._qubits = qubits
        self._shots = shots
        self._diff_method = diff_method
        self._observable = observable
        self._seed = seed
        self._circuit_def = _circuit_decorator(qubits=qubits)(fn)
        functools.update_wrapper(self, fn)

    def __call__(self, **kwargs: float) -> Result:
        """Execute the quantum circuit with given parameters.

        Args:
            **kwargs: Parameter values (e.g., theta=0.5).

        Returns:
            Result: Measurement results.
        """
        return run(
            self._circuit_def,
            shots=self._shots,
            seed=self._seed,
            **kwargs,
        )

    def expectation(
        self,
        observable: list[tuple[str, float]] | None = None,
        **kwargs: float,
    ) -> float:
        """Compute ⟨ψ|O|ψ⟩ for a Pauli observable.

        Args:
            observable: Pauli observable [(pauli_str, coeff), ...].
                Uses default if not provided.
            **kwargs: Circuit parameters.

        Returns:
            Expectation value as float.
        """
        from quanta.primitives import Estimator

        obs = observable or self._observable
        if obs is None:
            raise ValueError(
                "No observable provided. Pass observable= to @quantum "
                "or to expectation()."
            )
        estimator = Estimator(seed=self._seed)
        result = estimator.run(self._circuit_def, observables=[obs], **kwargs)
        return result.value

    def gradient(
        self,
        observable: list[tuple[str, float]] | None = None,
        shift: float = np.pi / 2,
        **kwargs: float,
    ) -> GradientResult:
        """Compute gradients via parameter-shift rule.

        Args:
            observable: Pauli observable for cost function.
                If None, uses the one set in @quantum.
            shift: Shift parameter (default π/2 for Pauli rotations).
            **kwargs: Current parameter values.

        Returns:
            GradientResult with gradients dict and cost value.
        """
        obs = observable or self._observable
        if obs is None:
            raise ValueError(
                "gradient() requires an observable. Pass observable= "
                "to @quantum or to gradient()."
            )

        cost_val = self.expectation(observable=obs, **kwargs)

        gradients = {}
        for param_name, param_val in kwargs.items():
            # f(θ+s) - f(θ-s) / (2 sin(s))
            kwargs_plus = {**kwargs, param_name: param_val + shift}
            kwargs_minus = {**kwargs, param_name: param_val - shift}

            f_plus = self.expectation(observable=obs, **kwargs_plus)
            f_minus = self.expectation(observable=obs, **kwargs_minus)

            gradients[param_name] = (f_plus - f_minus) / (2 * np.sin(shift))

        return GradientResult(gradients=gradients, cost=cost_val)

    async def run_async(self, **kwargs: float) -> Result:
        """Async circuit execution.

        Example:
            >>> result = await circuit.run_async(theta=0.5)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self(**kwargs))

    @property
    def num_qubits(self) -> int:
        """Number of qubits."""
        return self._qubits

    @property
    def diff_method(self) -> str:
        """Differentiation method."""
        return self._diff_method

    def __repr__(self) -> str:
        return (
            f"QuantumFunction({self._fn.__name__}, "
            f"qubits={self._qubits}, diff={self._diff_method})"
        )


def quantum(
    qubits: int,
    shots: int = 1024,
    diff_method: str = "parameter-shift",
    observable: list[tuple[str, float]] | None = None,
    seed: int | None = None,
) -> Callable:
    """Decorator to create an auto-differentiable quantum circuit.

    PennyLane @qml.qnode equivalent.

    Args:
        qubits: Number of qubits.
        shots: Default shot count.
        diff_method: Gradient method ("parameter-shift" or "finite-diff").
        observable: Default Pauli observable for expectation/gradient.
        seed: Random seed for reproducibility.

    Returns:
        QuantumFunction with __call__, gradient, expectation, run_async.

    Example:
        >>> @quantum(qubits=2, observable=[("ZZ", 1.0)])
        ... def vqe_ansatz(q, theta=0.0, phi=0.0):
        ...     RY(theta)(q[0])
        ...     RY(phi)(q[1])
        ...     CX(q[0], q[1])
        ...     return measure(q)
        >>>
        >>> result = vqe_ansatz(theta=0.5, phi=0.3)
        >>> grad = vqe_ansatz.gradient(theta=0.5, phi=0.3)
        >>> print(grad)
        GradientResult(cost=..., gradients={'theta': ..., 'phi': ...})
    """
    def decorator(fn: Callable) -> QuantumFunction:
        return QuantumFunction(
            fn=fn,
            qubits=qubits,
            shots=shots,
            diff_method=diff_method,
            observable=observable,
            seed=seed,
        )
    return decorator
