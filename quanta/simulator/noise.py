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

_I2 = np.eye(2, dtype=complex)
_I4 = np.eye(4, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


class NoiseChannel(ABC):
    """Abstract interface for a quantum noise channel."""

    @abstractmethod
    def apply(
        self, state: np.ndarray, qubit: int, num_qubits: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Applies stochastic noise trajectory step to a statevector."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Channel name."""
        ...

    def kraus_operators(self) -> list[np.ndarray]:
        """Returns CPTP Kraus operators {K_k} satisfying sum_k K_k† K_k = I."""
        raise NotImplementedError(
            f"{type(self).__name__} does not define kraus_operators()."
        )


def _apply_kraus_stochastic(
    state: np.ndarray,
    kraus_ops: list[np.ndarray],
    target_qubits: tuple[int, ...],
    num_qubits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Applies an arbitrary k-qubit CPTP Kraus channel stochastically to a statevector."""
    m = len(target_qubits)
    state_tensor = state.reshape([2] * num_qubits)
    in_axes = list(range(m, 2 * m))
    t_axes = list(target_qubits)

    branches: list[np.ndarray] = []
    probs: list[float] = []
    for k_op in kraus_ops:
        k_tensor = np.asarray(k_op, dtype=complex).reshape([2] * (2 * m))
        cand = np.tensordot(k_tensor, state_tensor, axes=(in_axes, t_axes))
        cand = np.moveaxis(cand, list(range(m)), t_axes)
        p_k = float(np.real(np.vdot(cand, cand)))
        branches.append(cand)
        probs.append(max(0.0, p_k))

    total_p = sum(probs)
    if total_p <= 1e-15:
        return state
    p_arr = np.array(probs, dtype=np.float64) / total_p
    idx = int(rng.choice(len(kraus_ops), p=p_arr))
    chosen = branches[idx].reshape(-1)
    norm = float(np.linalg.norm(chosen))
    if norm > 1e-15:
        chosen = chosen / norm
    return chosen


def resolve_channel_qubit_targets(
    channel: NoiseChannel,
    qubits: tuple[int, ...],
    num_qubits: int,
) -> list[tuple[int, ...]]:
    """Resolves the target qubit tuple(s) for a noise channel given gate qubits."""
    if isinstance(channel, ReadoutError) or not qubits:
        return []

    kraus = channel.kraus_operators()
    dim = kraus[0].shape[0]
    if dim == 2:
        return [(q,) for q in qubits]
    if dim == 4:
        if num_qubits < 2:
            return []
        if len(qubits) >= 2:
            return [(qubits[0], qubits[1])]
        q = int(qubits[0])
        if isinstance(channel, Crosstalk):
            nb = q + int(channel.neighbor_offset)
            if 0 <= nb < num_qubits and nb != q:
                return [(q, nb)]
        return [(q, q + 1) if q + 1 < num_qubits else (q - 1, q)]

    arity = int(round(np.log2(dim)))
    if len(qubits) >= arity:
        return [tuple(qubits[:arity])]
    return []


@dataclass
class Depolarizing(NoiseChannel):
    """Depolarizing channel: random Pauli (X, Y, Z) error."""

    probability: float = 0.01

    @property
    def name(self) -> str:
        return f"Depolarizing(p={self.probability})"

    def kraus_operators(self) -> list[np.ndarray]:
        p = float(self.probability)
        return [
            np.sqrt(max(0.0, 1.0 - p)) * _I2,
            np.sqrt(p / 3.0) * _X,
            np.sqrt(p / 3.0) * _Y,
            np.sqrt(p / 3.0) * _Z,
        ]

    def apply(
        self, state: np.ndarray, qubit: int, num_qubits: int, rng: np.random.Generator
    ) -> np.ndarray:
        if self.probability <= 0.0 or rng.random() >= self.probability:
            return state
        pauli = int(rng.integers(0, 3))
        return _apply_single_qubit_error(state, qubit, num_qubits, pauli)


@dataclass
class BitFlip(NoiseChannel):
    """Bit-flip channel: random |0⟩↔|1⟩ flip (X error)."""

    probability: float = 0.01

    @property
    def name(self) -> str:
        return f"BitFlip(p={self.probability})"

    def kraus_operators(self) -> list[np.ndarray]:
        p = float(self.probability)
        return [
            np.sqrt(max(0.0, 1.0 - p)) * _I2,
            np.sqrt(p) * _X,
        ]

    def apply(
        self, state: np.ndarray, qubit: int, num_qubits: int, rng: np.random.Generator
    ) -> np.ndarray:
        if self.probability <= 0.0 or rng.random() >= self.probability:
            return state
        return _apply_single_qubit_error(state, qubit, num_qubits, pauli=0)


@dataclass
class PhaseFlip(NoiseChannel):
    """Phase-flip channel: random phase error (Z error)."""

    probability: float = 0.01

    @property
    def name(self) -> str:
        return f"PhaseFlip(p={self.probability})"

    def kraus_operators(self) -> list[np.ndarray]:
        p = float(self.probability)
        return [
            np.sqrt(max(0.0, 1.0 - p)) * _I2,
            np.sqrt(p) * _Z,
        ]

    def apply(
        self, state: np.ndarray, qubit: int, num_qubits: int, rng: np.random.Generator
    ) -> np.ndarray:
        if self.probability <= 0.0 or rng.random() >= self.probability:
            return state
        return _apply_single_qubit_error(state, qubit, num_qubits, pauli=2)


@dataclass
class AmplitudeDamping(NoiseChannel):
    """Amplitude damping: energy loss (T1 decay)."""

    gamma: float = 0.01

    @property
    def name(self) -> str:
        return f"AmplitudeDamping(γ={self.gamma})"

    def kraus_operators(self) -> list[np.ndarray]:
        g = float(self.gamma)
        k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(max(0.0, 1.0 - g))]], dtype=complex)
        k1 = np.array([[0.0, np.sqrt(g)], [0.0, 0.0]], dtype=complex)
        return [k0, k1]

    def apply(
        self, state: np.ndarray, qubit: int, num_qubits: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Applies amplitude damping using vectorized Kraus trajectory step."""
        if self.gamma <= 0.0:
            return state

        s = state.copy().reshape([2] * num_qubits)
        idx0: list[slice | int] = [slice(None)] * num_qubits
        idx1: list[slice | int] = [slice(None)] * num_qubits
        idx0[qubit] = 0
        idx1[qubit] = 1
        t0, t1 = tuple(idx0), tuple(idx1)

        p_decay = self.gamma * float(np.sum(np.abs(s[t1]) ** 2))
        if rng.random() < p_decay and p_decay > 1e-15:
            s[t0] = np.sqrt(self.gamma) * s[t1]
            s[t1] = 0.0
        else:
            s[t1] = np.sqrt(max(0.0, 1.0 - self.gamma)) * s[t1]

        new_state = s.reshape(-1)
        norm = float(np.linalg.norm(new_state))
        if norm > 1e-15:
            new_state /= norm
        return new_state


class NoiseModel:
    """Noise model: manages multiple channels.

    Supports builder pattern and preset profiles:

        # Builder pattern
        >>> model = (NoiseModel.builder()
        ...     .depolarizing(0.01)
        ...     .with_readout(0.01, 0.02)
        ...     .build())

        # Preset profiles
        >>> ibm = NoiseModel.ibm_heron()
        >>> ionq = NoiseModel.ionq_aria()

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

    @property
    def channels(self) -> list[NoiseChannel]:
        """Returns the list of noise channels."""
        return self._channels

    def apply_noise(
        self,
        state: np.ndarray,
        qubits: tuple[int, ...],
        num_qubits: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Applies all gate-level channels to the specified qubits respecting arity."""
        for channel in self._channels:
            if isinstance(channel, ReadoutError):
                continue
            targets_list = resolve_channel_qubit_targets(channel, qubits, num_qubits)
            for target in targets_list:
                if len(target) == 1:
                    state = channel.apply(state, target[0], num_qubits, rng)
                elif len(target) == 2 and isinstance(channel, Crosstalk):
                    state = channel.apply_2q(state, target[0], target[1], num_qubits, rng)
                else:
                    state = _apply_kraus_stochastic(
                        state, channel.kraus_operators(), target, num_qubits, rng
                    )
        return state

    def describe(self) -> str:
        """Returns a formatted table of noise channels.

        Example:
            >>> NoiseModel.ibm_heron().describe()
            '=== Noise Model (3 channels) ===\\n...'
        """
        lines = [f"=== Noise Model ({len(self._channels)} channels) ==="]
        lines.append(f"  {'#':<4} {'Channel':<35} {'Type':<15}")
        lines.append("  " + "─" * 54)
        for i, ch in enumerate(self._channels):
            ch_type = type(ch).__name__
            lines.append(f"  {i + 1:<4} {ch.name:<35} {ch_type:<15}")
        return "\n".join(lines)

    # ── Builder Pattern ──

    @classmethod
    def builder(cls) -> _NoiseModelBuilder:
        """Start building a noise model with fluent API."""
        return _NoiseModelBuilder()

    # ── Preset Profiles ──

    @classmethod
    def ibm_heron(cls) -> NoiseModel:
        """IBM Heron r3 noise profile."""
        return (
            cls.builder()
            .depolarizing(0.005)
            .with_amplitude_damping(0.003)
            .with_t2(0.005)
            .with_readout(0.008, 0.012)
            .build()
        )

    @classmethod
    def ionq_aria(cls) -> NoiseModel:
        """IonQ Aria noise profile."""
        return (
            cls.builder()
            .depolarizing(0.003)
            .with_amplitude_damping(0.001)
            .build()
        )

    @classmethod
    def google_willow(cls) -> NoiseModel:
        """Google Willow noise profile."""
        return (
            cls.builder()
            .depolarizing(0.003)
            .with_amplitude_damping(0.005)
            .with_t2(0.008)
            .with_crosstalk(0.002)
            .build()
        )

    def __repr__(self) -> str:
        names = [ch.name for ch in self._channels]
        return f"NoiseModel(channels={names})"


class _NoiseModelBuilder:
    """Fluent builder for constructing noise models."""

    def __init__(self) -> None:
        self._model = NoiseModel()

    def depolarizing(self, probability: float = 0.01) -> _NoiseModelBuilder:
        """Add depolarizing channel."""
        self._model.add(Depolarizing(probability))
        return self

    def with_bit_flip(self, probability: float = 0.01) -> _NoiseModelBuilder:
        """Add bit-flip channel."""
        self._model.add(BitFlip(probability))
        return self

    def with_phase_flip(self, probability: float = 0.01) -> _NoiseModelBuilder:
        """Add phase-flip channel."""
        self._model.add(PhaseFlip(probability))
        return self

    def with_amplitude_damping(self, gamma: float = 0.01) -> _NoiseModelBuilder:
        """Add amplitude damping (T1 decay)."""
        self._model.add(AmplitudeDamping(gamma))
        return self

    def with_t2(self, gamma: float = 0.01) -> _NoiseModelBuilder:
        """Add T2 relaxation (dephasing)."""
        self._model.add(T2Relaxation(gamma))
        return self

    def with_crosstalk(
        self, probability: float = 0.005, offset: int = 1,
    ) -> _NoiseModelBuilder:
        """Add crosstalk noise."""
        self._model.add(Crosstalk(probability, offset))
        return self

    def with_readout(
        self, p0_to_1: float = 0.01, p1_to_0: float = 0.02,
    ) -> _NoiseModelBuilder:
        """Add readout error."""
        self._model.add(ReadoutError(p0_to_1, p1_to_0))
        return self

    def build(self) -> NoiseModel:
        """Build and return the noise model."""
        return self._model


def _apply_single_qubit_error(
    state: np.ndarray, qubit: int, num_qubits: int, pauli: int
) -> np.ndarray:
    s = state.copy().reshape([2] * num_qubits)
    idx0: list[slice | int] = [slice(None)] * num_qubits
    idx1: list[slice | int] = [slice(None)] * num_qubits
    idx0[qubit] = 0
    idx1[qubit] = 1
    t0, t1 = tuple(idx0), tuple(idx1)

    if pauli == 0:  # X: bit flip
        tmp = s[t0].copy()
        s[t0] = s[t1]
        s[t1] = tmp
    elif pauli == 1:  # Y: [[0, -1j], [1j, 0]]
        tmp = s[t0].copy()
        s[t0] = -1j * s[t1]
        s[t1] = 1j * tmp
    elif pauli == 2:  # Z: [[1, 0], [0, -1]]
        s[t1] = -s[t1]

    return s.reshape(-1)


@dataclass
class T2Relaxation(NoiseChannel):
    """T2 relaxation (pure dephasing) channel.

    Models decoherence where the qubit loses phase information
    without energy loss. T2 is always <= 2*T1.

    Attributes:
        gamma: Dephasing rate (probability of phase randomization).
    """

    gamma: float = 0.01

    @property
    def name(self) -> str:
        return f"T2Relaxation(γ={self.gamma})"

    def kraus_operators(self) -> list[np.ndarray]:
        g = float(self.gamma)
        return [
            np.sqrt(max(0.0, 1.0 - g / 2.0)) * _I2,
            np.sqrt(g / 2.0) * _Z,
        ]

    def apply(
        self, state: np.ndarray, qubit: int, num_qubits: int, rng: np.random.Generator
    ) -> np.ndarray:
        if self.gamma <= 0.0 or rng.random() >= self.gamma:
            return state

        s = state.copy().reshape([2] * num_qubits)
        idx1: list[slice | int] = [slice(None)] * num_qubits
        idx1[qubit] = 1
        phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
        s[tuple(idx1)] *= phase
        return s.reshape(-1)


@dataclass
class Crosstalk(NoiseChannel):
    """Crosstalk noise: correlated ZZ interaction between qubits.

    Models unintended coherent ZZ rotation exp(-i * angle/2 * Z⊗Z) =
    diag(exp(-i*a/2), exp(i*a/2), exp(i*a/2), exp(-i*a/2)) with probability p.

    Attributes:
        probability: Probability of crosstalk per gate.
        neighbor_offset: Which neighbor is affected (+1 = next qubit) for 1Q gates.
        angle: ZZ rotation angle in radians.
    """

    probability: float = 0.005
    neighbor_offset: int = 1
    angle: float = 0.05 * np.pi

    @property
    def name(self) -> str:
        return f"Crosstalk(p={self.probability}, offset={self.neighbor_offset})"

    def kraus_operators(self) -> list[np.ndarray]:
        p = float(self.probability)
        a = float(self.angle)
        u_zz = np.diag([
            np.exp(-1j * a / 2.0),
            np.exp(1j * a / 2.0),
            np.exp(1j * a / 2.0),
            np.exp(-1j * a / 2.0),
        ]).astype(complex)
        return [
            np.sqrt(max(0.0, 1.0 - p)) * _I4,
            np.sqrt(p) * u_zz,
        ]

    def apply_2q(
        self,
        state: np.ndarray,
        q0: int,
        q1: int,
        num_qubits: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Applies 2-qubit traceless ZZ crosstalk on (q0, q1)."""
        if (
            self.probability <= 0.0
            or q0 == q1
            or q0 < 0
            or q1 < 0
            or q0 >= num_qubits
            or q1 >= num_qubits
            or rng.random() >= self.probability
        ):
            return state

        a = float(self.angle)
        phase_even = np.exp(-1j * a / 2.0)
        phase_odd = np.exp(1j * a / 2.0)

        s = state.copy().reshape([2] * num_qubits)
        for b0 in (0, 1):
            for b1 in (0, 1):
                slc: list[slice | int] = [slice(None)] * num_qubits
                slc[q0] = b0
                slc[q1] = b1
                s[tuple(slc)] *= phase_even if (b0 == b1) else phase_odd
        return s.reshape(-1)

    def apply(
        self, state: np.ndarray, qubit: int, num_qubits: int, rng: np.random.Generator
    ) -> np.ndarray:
        neighbor = int(qubit) + int(self.neighbor_offset)
        if neighbor < 0 or neighbor >= num_qubits:
            return state
        return self.apply_2q(state, int(qubit), neighbor, num_qubits, rng)


@dataclass
class ReadoutError(NoiseChannel):
    """Readout error: measurement bit-flip during readout.

    Attributes:
        p0_to_1: Probability of reading |0⟩ as |1⟩.
        p1_to_0: Probability of reading |1⟩ as |0⟩.
    """

    p0_to_1: float = 0.01
    p1_to_0: float = 0.02

    @property
    def name(self) -> str:
        return f"ReadoutError(p01={self.p0_to_1}, p10={self.p1_to_0})"

    def kraus_operators(self) -> list[np.ndarray]:
        return [_I2.copy()]

    def apply(
        self, state: np.ndarray, qubit: int, num_qubits: int, rng: np.random.Generator
    ) -> np.ndarray:
        return state

    def apply_to_counts(
        self,
        counts: dict[str, int],
        rng: np.random.Generator,
    ) -> dict[str, int]:
        """Applies readout error to measurement counts."""
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
