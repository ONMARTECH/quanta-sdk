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
    "T2Relaxation", "Crosstalk", "ReadoutError",
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
        """Applies amplitude damping using Kraus operators.

        K0 = [[1, 0], [0, √(1-γ)]]  (no-decay branch)
        K1 = [[0, √γ], [0, 0]]      (decay branch)

        Statevector application: randomly apply K0 or K1 based
        on the probability of decay, then renormalize.
        """
        n = num_qubits
        dim = len(state)
        sqrt_1mg = np.sqrt(1 - self.gamma)
        sqrt_g = np.sqrt(self.gamma)

        # Compute probability of decay (K1 branch)
        p_decay = 0.0
        for i in range(dim):
            if (i >> (n - 1 - qubit)) & 1:  # qubit in |1⟩
                p_decay += self.gamma * abs(state[i]) ** 2

        new_state = state.copy()

        if rng.random() < p_decay and p_decay > 1e-15:
            # Apply K1: |1⟩ → |0⟩ transition
            for i in range(dim):
                if (i >> (n - 1 - qubit)) & 1:  # qubit in |1⟩
                    j = i ^ (1 << (n - 1 - qubit))  # paired |0⟩ index
                    new_state[j] = sqrt_g * state[i]
                    new_state[i] = 0
                # |0⟩ states already zeroed by K1
                else:
                    new_state[i] = 0
        else:
            # Apply K0: dampen |1⟩ amplitude
            for i in range(dim):
                if (i >> (n - 1 - qubit)) & 1:  # qubit in |1⟩
                    new_state[i] = sqrt_1mg * state[i]

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


@dataclass
class T2Relaxation(NoiseChannel):
    """T2 relaxation (pure dephasing) channel.

    Models decoherence where the qubit loses phase information
    without energy loss. T2 is always <= 2*T1.

    In real hardware:
      - IBM Eagle: T2 ~ 100-200 μs
      - Google Sycamore: T2 ~ 10-20 μs

    Attributes:
        gamma: Dephasing rate (probability of phase randomization).
    """

    gamma: float = 0.01

    @property
    def name(self) -> str:
        return f"T2Relaxation(γ={self.gamma})"

    def apply(
        self, state: np.ndarray, qubit: int, num_qubits: int, rng: np.random.Generator
    ) -> np.ndarray:
        if rng.random() > self.gamma:
            return state  # No dephasing

        # Pure dephasing: apply random phase to |1⟩ component
        n = num_qubits
        new_state = state.copy()
        phase = np.exp(1j * rng.uniform(0, 2 * np.pi))

        for i in range(len(state)):
            if (i >> (n - 1 - qubit)) & 1:  # qubit in |1⟩
                new_state[i] *= phase

        return new_state


@dataclass
class Crosstalk(NoiseChannel):
    """Crosstalk noise: correlated errors between neighboring qubits.

    When a gate operates on a qubit, crosstalk causes unintended
    rotations on physically adjacent qubits. This is a major error
    source in superconducting processors.

    In real hardware:
      - ZZ crosstalk: ~0.1-1% per gate
      - Frequency collision: ~0.01-0.1%

    Attributes:
        probability: Probability of crosstalk per gate.
        neighbor_offset: Which neighbor is affected (+1 = next qubit).
    """

    probability: float = 0.005
    neighbor_offset: int = 1

    @property
    def name(self) -> str:
        return f"Crosstalk(p={self.probability}, offset={self.neighbor_offset})"

    def apply(
        self, state: np.ndarray, qubit: int, num_qubits: int, rng: np.random.Generator
    ) -> np.ndarray:
        if rng.random() > self.probability:
            return state  # No crosstalk

        neighbor = qubit + self.neighbor_offset
        if neighbor < 0 or neighbor >= num_qubits:
            return state  # No neighbor to affect

        # ZZ-type crosstalk: small Z rotation on neighbor
        # conditioned on the state of the target qubit
        n = num_qubits
        new_state = state.copy()
        angle = rng.uniform(0.01, 0.1) * np.pi  # Small unwanted rotation

        for i in range(len(state)):
            target_bit = (i >> (n - 1 - qubit)) & 1
            neighbor_bit = (i >> (n - 1 - neighbor)) & 1
            # ZZ interaction: phase depends on both qubits
            if target_bit and neighbor_bit:
                new_state[i] *= np.exp(1j * angle)
            elif target_bit or neighbor_bit:
                new_state[i] *= np.exp(-1j * angle / 2)

        return new_state


@dataclass
class ReadoutError(NoiseChannel):
    """Readout error: measurement bit-flip during readout.

    Models the classical error that occurs when reading out a qubit.
    The qubit state is correct, but the measurement result may flip.

    In real hardware:
      - IBM Eagle: ~0.5-2% readout error
      - Google Sycamore: ~0.5-1%

    Attributes:
        p0_to_1: Probability of reading |0⟩ as |1⟩.
        p1_to_0: Probability of reading |1⟩ as |0⟩.
    """

    p0_to_1: float = 0.01
    p1_to_0: float = 0.02

    @property
    def name(self) -> str:
        return f"ReadoutError(p01={self.p0_to_1}, p10={self.p1_to_0})"

    def apply(
        self, state: np.ndarray, qubit: int, num_qubits: int, rng: np.random.Generator
    ) -> np.ndarray:
        # ReadoutError doesn't modify the quantum state.
        # It modifies measurement results (applied post-measurement).
        return state

    def apply_to_counts(
        self,
        counts: dict[str, int],
        rng: np.random.Generator,
    ) -> dict[str, int]:
        """Applies readout error to measurement counts.

        Flips individual bits in measurement results based on
        per-bit error probabilities.

        Args:
            counts: Measurement result counts.
            rng: Random number generator.

        Returns:
            Modified counts with readout errors applied.
        """
        noisy_counts: dict[str, int] = {}

        for bitstring, count in counts.items():
            for _ in range(count):
                noisy_bits = list(bitstring)
                for idx, bit in enumerate(noisy_bits):
                    if bit == "0" and rng.random() < self.p0_to_1:
                        noisy_bits[idx] = "1"
                    elif bit == "1" and rng.random() < self.p1_to_0:
                        noisy_bits[idx] = "0"
                result = "".join(noisy_bits)
                noisy_counts[result] = noisy_counts.get(result, 0) + 1

        return noisy_counts
