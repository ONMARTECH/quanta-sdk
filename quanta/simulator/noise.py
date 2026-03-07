"""
quanta.simulator.noise — Quantum noise models.


Example:
    >>> from quanta.simulator.noise import NoiseModel, Depolarizing
    >>> noise = NoiseModel()
    >>> result = run(bell, noise=noise)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# ── Public API ──
__all__ = [
    "NoiseChannel", "NoiseModel",
    "Depolarizing", "BitFlip", "PhaseFlip", "AmplitudeDamping",
]

class NoiseChannel(ABC):
    """Abstract interface for a single noise channel.

    """

    @abstractmethod
    def apply(
        self, state: np.ndarray, qubit: int, num_qubits: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Applies noise to the statevector.

        Args:
            qubit: Target qubit index.
            num_qubits: Total qubit count.

        Returns:
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Channel name."""
        ...

@dataclass
class Depolarizing(NoiseChannel):
    """Depolarizing channel: random Pauli (X, Y, Z) error.


    Attributes:
    """

    probability: float = 0.01

    @property
    def name(self) -> str:
        return f"Depolarizing(p={self.probability})"

    def apply(
        self, state: np.ndarray, qubit: int, num_qubits: int, rng: np.random.Generator
    ) -> np.ndarray:
        if rng.random() > self.probability:
            return state  # No error

        pauli = rng.integers(0, 3)
        return _apply_single_qubit_error(state, qubit, num_qubits, pauli)

@dataclass
class BitFlip(NoiseChannel):
    """Bit-flip channel: random |0⟩↔|1⟩ flip (X error).

    Attributes:
    """

    probability: float = 0.01

    @property
    def name(self) -> str:
        return f"BitFlip(p={self.probability})"

    def apply(
        self, state: np.ndarray, qubit: int, num_qubits: int, rng: np.random.Generator
    ) -> np.ndarray:
        if rng.random() > self.probability:
            return state
        return _apply_single_qubit_error(state, qubit, num_qubits, pauli=0)

@dataclass
class PhaseFlip(NoiseChannel):
    """Phase-flip channel: random phase error (Z error).

    Attributes:
    """

    probability: float = 0.01

    @property
    def name(self) -> str:
        return f"PhaseFlip(p={self.probability})"

    def apply(
        self, state: np.ndarray, qubit: int, num_qubits: int, rng: np.random.Generator
    ) -> np.ndarray:
        if rng.random() > self.probability:
            return state
        return _apply_single_qubit_error(state, qubit, num_qubits, pauli=2)

@dataclass
class AmplitudeDamping(NoiseChannel):
    """Amplitude damping: energy loss (T1 decay).


    Attributes:
    """

    gamma: float = 0.01

    @property
    def name(self) -> str:
        return f"AmplitudeDamping(γ={self.gamma})"

    def apply(
        self, state: np.ndarray, qubit: int, num_qubits: int, rng: np.random.Generator
    ) -> np.ndarray:
        n = num_qubits
        dim = len(state)
        new_state = state.copy()

        for i in range(dim):
            if (i >> (n - 1 - qubit)) & 1:  # qubit in |1> state
                j = i ^ (1 << (n - 1 - qubit))  # paired |0> index
                if rng.random() < self.gamma:
                    new_state[j] += new_state[i]
                    new_state[i] = 0

        # Normalize
        norm = np.linalg.norm(new_state)
        if norm > 1e-15:
            new_state /= norm

        return new_state

class NoiseModel:
    """Noise model: manages multiple channels.


    Example:
        >>> model = NoiseModel()
        >>> model.add(Depolarizing(0.01))
        >>> model.add(AmplitudeDamping(0.005))
    """

    def __init__(self) -> None:
        self._channels: list[NoiseChannel] = []

    def add(self, channel: NoiseChannel) -> NoiseModel:
        """Adds a channel. Chainable API."""
        self._channels.append(channel)
        return self

    def apply_noise(
        self,
        state: np.ndarray,
        qubits: tuple[int, ...],
        num_qubits: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Applies all channels to the specified qubits."""
        for channel in self._channels:
            for qubit in qubits:
                state = channel.apply(state, qubit, num_qubits, rng)
        return state

    def __repr__(self) -> str:
        names = [ch.name for ch in self._channels]
        return f"NoiseModel(channels={names})"

def _apply_single_qubit_error(
    state: np.ndarray, qubit: int, num_qubits: int, pauli: int
) -> np.ndarray:
    n = num_qubits
    new_state = state.copy()
    dim = len(state)

    for i in range(dim):
        bit = (i >> (n - 1 - qubit)) & 1
        j = i ^ (1 << (n - 1 - qubit))  # bit-flip index

        if pauli == 0:    # X: bit flip
            new_state[i], new_state[j] = state[j], state[i]
        elif pauli == 1:  # Y: bit flip + phase
            new_state[i] = 1j * state[j] if bit == 0 else -1j * state[j]
        elif pauli == 2 and bit == 1:  # Z: phase flip
            new_state[i] = -state[i]

    return new_state
