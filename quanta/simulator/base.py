"""
quanta.simulator.base -- Abstract base class for all simulators.

All simulator backends (StateVector, Sparse, MPS, PauliFrame)
implement this interface. This ensures that Layer 3 algorithms
can work with any simulator without direct coupling.

Design:
    SimulatorBackend defines the minimal contract:
    - apply(gate, qubits, params) → mutate internal state
    - probabilities() → measurement probabilities
    - sample(shots) → measurement counts
    - state property → raw state access (backend-specific)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

__all__ = ["SimulatorBackend"]


class SimulatorBackend(ABC):
    """Abstract base class for quantum simulators.

    All simulator implementations must inherit from this class
    to be usable in the Quanta execution pipeline.

    Attributes:
        num_qubits: Number of qubits being simulated.
    """

    num_qubits: int

    @abstractmethod
    def apply(
        self,
        gate_name: str,
        qubits: tuple[int, ...],
        params: tuple[float, ...] = (),
    ) -> None:
        """Applies a quantum gate to the simulator state.

        Args:
            gate_name: Gate identifier (e.g., "H", "CX", "RZ").
            qubits: Target qubit indices.
            params: Gate parameters (angles for parametric gates).
        """

    @abstractmethod
    def probabilities(self) -> np.ndarray:
        """Returns measurement probabilities for all basis states.

        Returns:
            Array of shape (2^n,) with P(|i⟩) = |α_i|².
        """

    @abstractmethod
    def sample(self, shots: int) -> dict[str, int]:
        """Performs measurement sampling.

        Args:
            shots: Number of measurement repetitions.

        Returns:
            Dict mapping bitstring → count.
        """

    @property
    @abstractmethod
    def state(self) -> Any:
        """Returns the current quantum state (backend-specific type).

        - StateVector: np.ndarray of shape (2^n,)
        - Sparse: dict[int, complex]
        - MPS: list of tensors
        - PauliFrame: stabilizer tableau
        """

    @state.setter
    @abstractmethod
    def state(self, value: Any) -> None:
        """Sets the quantum state (backend-specific type)."""

    @property
    def max_qubits(self) -> int:
        """Maximum supported qubit count for this backend.

        Override in subclasses. Default: no explicit limit.
        """
        return 1000

    def reset(self) -> None:
        """Resets the simulator to |00...0⟩ state.

        Override in subclasses for optimized reset.
        """
        self.__init__(self.num_qubits)  # type: ignore[misc]

    def apply_phase(self, index: int, phase: complex) -> None:
        """Applies a phase factor to a specific basis state.

        Used by Grover oracle and similar algorithms.
        Default implementation modifies the state array directly.

        Args:
            index: Basis state index (0 to 2^n - 1).
            phase: Phase factor (e.g., -1 for phase flip).

        Raises:
            NotImplementedError: If backend doesn't support direct state access.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support apply_phase(). "
            f"Use a dense or sparse simulator for Grover's algorithm."
        )

    def apply_noise(
        self,
        noise_model: object,
        qubits: tuple[int, ...],
        rng: np.random.Generator,
    ) -> None:
        """Applies a noise model after a gate operation.

        Args:
            noise_model: NoiseModel with apply_noise() method.
            qubits: Qubits the gate acted on.
            rng: Random number generator.

        Raises:
            NotImplementedError: If backend doesn't support noise.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support noise simulation."
        )
