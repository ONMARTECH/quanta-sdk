"""
quanta.simulator.sparse -- Sparse statevector simulator.

Stores only non-zero amplitudes in a dictionary, enabling simulation
of circuits with up to ~50 qubits when the state remains sparse
(most amplitudes ≈ 0).

Physics:
    A quantum state |ψ⟩ = Σ αᵢ|i⟩ often has many αᵢ = 0.
    Instead of storing all 2^n amplitudes (exponential),
    we store only {i: αᵢ | αᵢ ≠ 0} (proportional to sparsity).

    Memory: O(k) where k = number of non-zero amplitudes
    Dense:  O(2^n) — for 40 qubits = 8 TB!
    Sparse: O(k) — for Grover with 40 qubits, k ≈ 2^20

Limitations:
    - Gates that create superposition increase k rapidly
    - After O(n) Hadamard layers, state becomes dense → falls back
    - Best for: GHZ states, oracle circuits, QFT on sparse inputs

Example:
    >>> from quanta.simulator.sparse import SparseSimulator
    >>> sim = SparseSimulator(40)  # 40 qubits!
    >>> sim.apply("X", (0,))      # |0...01⟩
    >>> sim.apply("H", (0,))      # (|0...00⟩ + |0...01⟩) / √2
    >>> len(sim._amplitudes)       # Only 2 non-zero entries
    2
"""

from __future__ import annotations

import numpy as np

from quanta.core.gates import GATE_REGISTRY, MultiParametricGate, ParametricGate
from quanta.core.types import QuantaError
from quanta.simulator.base import SimulatorBackend

__all__ = ["SparseSimulator"]


class SparseSimulatorError(QuantaError):
    """Sparse simulator error."""


class SparseSimulator(SimulatorBackend):
    """Dict-based sparse statevector simulator.

    Stores quantum state as {basis_index: amplitude} dictionary.
    Only non-zero amplitudes are stored, enabling large qubit counts
    when the state remains sparse.

    Args:
        num_qubits: Number of qubits (up to ~50 practical).
        seed: Random seed for measurement sampling.
        threshold: Amplitudes with |α| < threshold are pruned.
    """

    MAX_QUBITS = 50

    def __init__(
        self,
        num_qubits: int,
        seed: int | None = None,
        threshold: float = 1e-15,
        **_kwargs: object,
    ) -> None:
        if num_qubits > self.MAX_QUBITS:
            raise SparseSimulatorError(
                f"SparseSimulator supports up to {self.MAX_QUBITS} qubits, "
                f"requested: {num_qubits}"
            )
        self.num_qubits = num_qubits
        self._amplitudes: dict[int, complex] = {0: complex(1.0)}
        self._rng = np.random.default_rng(seed)
        self._threshold = threshold

    # ── Core Gate Application ──

    def apply(
        self,
        gate_name: str,
        qubits: tuple[int, ...],
        params: tuple[float, ...] = (),
    ) -> None:
        """Applies a quantum gate via sparse state manipulation."""
        matrix = self._get_gate_matrix(gate_name, params)
        n_gate_qubits = len(qubits)

        if n_gate_qubits == 1:
            self._apply_1q(matrix, qubits[0])
        elif n_gate_qubits == 2:
            self._apply_2q(matrix, qubits[0], qubits[1])
        else:
            self._apply_nq(matrix, qubits)

    def _apply_1q(self, matrix: np.ndarray, qubit: int) -> None:
        """Optimized single-qubit gate application on sparse state.

        Qubit ordering matches dense tensor simulator:
        qubit i → bit position (n-1-i) in basis state integer.
        """
        new_amps: dict[int, complex] = {}
        bit_pos = self.num_qubits - 1 - qubit  # MSB convention
        mask = 1 << bit_pos
        u00, u01, u10, u11 = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]

        # Group basis states by pairs (bit=0 and bit=1)
        processed = set()
        for basis in list(self._amplitudes):
            partner = basis ^ mask
            pair_key = min(basis, partner)
            if pair_key in processed:
                continue
            processed.add(pair_key)

            b0 = basis & ~mask  # bit=0 version
            b1 = basis | mask   # bit=1 version

            a0 = self._amplitudes.get(b0, 0.0)
            a1 = self._amplitudes.get(b1, 0.0)

            new_a0 = u00 * a0 + u01 * a1
            new_a1 = u10 * a0 + u11 * a1

            if abs(new_a0) > self._threshold:
                new_amps[b0] = new_a0
            if abs(new_a1) > self._threshold:
                new_amps[b1] = new_a1

        self._amplitudes = new_amps

    def _apply_2q(
        self, matrix: np.ndarray, q0: int, q1: int
    ) -> None:
        """Optimized two-qubit gate on sparse state (MSB convention)."""
        new_amps: dict[int, complex] = {}
        bp0 = self.num_qubits - 1 - q0  # MSB convention
        bp1 = self.num_qubits - 1 - q1
        mask0 = 1 << bp0
        mask1 = 1 << bp1
        mask_both = mask0 | mask1

        processed: set[int] = set()
        for basis in list(self._amplitudes):
            group = basis & ~mask_both
            if group in processed:
                continue
            processed.add(group)

            # 4 basis states: |00⟩, |01⟩, |10⟩, |11⟩ for (q0, q1)
            bases = [group, group | mask1, group | mask0, group | mask_both]
            amps = [self._amplitudes.get(b, 0.0) for b in bases]

            for i in range(4):
                new_val = sum(matrix[i, j] * amps[j] for j in range(4))
                if abs(new_val) > self._threshold:
                    new_amps[bases[i]] = new_val

        self._amplitudes = new_amps

    def _apply_nq(
        self, matrix: np.ndarray, qubits: tuple[int, ...]
    ) -> None:
        """General N-qubit gate on sparse state (MSB convention)."""
        n_gate = len(qubits)
        dim = 2 ** n_gate
        # Convert qubit indices to bit positions (MSB)
        bit_positions = [self.num_qubits - 1 - q for q in qubits]
        masks = [1 << bp for bp in bit_positions]
        all_mask = sum(masks)

        new_amps: dict[int, complex] = {}
        processed: set[int] = set()

        for basis in list(self._amplitudes):
            group = basis & ~all_mask
            if group in processed:
                continue
            processed.add(group)

            # Enumerate: index bit k maps to qubit k's bit position
            bases = []
            for idx in range(dim):
                b = group
                for bit_k in range(n_gate):
                    if idx & (1 << (n_gate - 1 - bit_k)):
                        b |= masks[bit_k]
                bases.append(b)

            amps = [self._amplitudes.get(b, 0.0) for b in bases]

            for i in range(dim):
                new_val = sum(matrix[i, j] * amps[j] for j in range(dim))
                if abs(new_val) > self._threshold:
                    new_amps[bases[i]] = new_val

        self._amplitudes = new_amps

    # ── Measurement ──

    def probabilities(self) -> np.ndarray:
        """Returns full probability array (dense, for compatibility)."""
        dim = 2 ** self.num_qubits
        probs = np.zeros(dim)
        for basis, amp in self._amplitudes.items():
            probs[basis] = abs(amp) ** 2
        return probs

    def sample(self, shots: int) -> dict[str, int]:
        """Measurement sampling from sparse state."""
        if not self._amplitudes:
            raise SparseSimulatorError("Cannot sample from empty state")

        bases = list(self._amplitudes.keys())
        amps = np.array([self._amplitudes[b] for b in bases])
        probs = np.abs(amps) ** 2

        # Normalize (handle numerical drift)
        total = probs.sum()
        if total < 1e-10:
            raise SparseSimulatorError("State has near-zero norm")
        probs /= total

        indices = self._rng.choice(len(bases), size=shots, p=probs)
        unique, counts = np.unique(indices, return_counts=True)

        fmt = f"0{self.num_qubits}b"
        return {
            format(bases[idx], fmt): int(cnt)
            for idx, cnt in zip(unique, counts, strict=True)
        }

    # ── State Access ──

    @property
    def state(self) -> np.ndarray:
        """Returns dense statevector (for compatibility / testing)."""
        dim = 2 ** self.num_qubits
        sv = np.zeros(dim, dtype=complex)
        for basis, amp in self._amplitudes.items():
            sv[basis] = amp
        return sv

    @state.setter
    def state(self, new_state: np.ndarray) -> None:
        """Sets state from dense array."""
        self._amplitudes = {}
        for i, amp in enumerate(new_state):
            if abs(amp) > self._threshold:
                self._amplitudes[i] = complex(amp)

    def apply_phase(self, index: int, phase: complex) -> None:
        """Applies phase to a specific basis state (Grover oracle)."""
        if index in self._amplitudes:
            self._amplitudes[index] *= phase
            if abs(self._amplitudes[index]) < self._threshold:
                del self._amplitudes[index]

    def apply_noise(
        self,
        noise_model: object,
        qubits: tuple[int, ...],
        rng: np.random.Generator,
    ) -> None:
        """Noise via dense conversion (fallback)."""
        dense = self.state
        dense = noise_model.apply_noise(dense, qubits, self.num_qubits, rng)  # type: ignore[union-attr]
        self.state = dense

    @property
    def max_qubits(self) -> int:
        """Maximum qubit count."""
        return self.MAX_QUBITS

    @property
    def num_nonzero(self) -> int:
        """Current number of non-zero amplitudes."""
        return len(self._amplitudes)

    @property
    def sparsity(self) -> float:
        """Fraction of state that is zero: 1.0 = maximally sparse."""
        dim = 2 ** self.num_qubits
        return 1.0 - len(self._amplitudes) / dim

    @property
    def memory_bytes(self) -> int:
        """Estimated memory usage in bytes."""
        # Each entry: int key (~28 bytes) + complex value (~32 bytes)
        return len(self._amplitudes) * 60

    # ── Internal ──

    def _get_gate_matrix(
        self, name: str, params: tuple[float, ...]
    ) -> np.ndarray:
        """Gets gate matrix from registry."""
        gate = GATE_REGISTRY.get(name)
        if gate is None:
            raise SparseSimulatorError(f"Unknown gate: {name}")

        if isinstance(gate, MultiParametricGate):
            if not params:
                raise SparseSimulatorError(f"{name} requires parameters")
            return gate(*params).matrix
        if isinstance(gate, ParametricGate):
            if not params:
                raise SparseSimulatorError(f"{name} requires parameters")
            return gate(params[0]).matrix

        return gate.matrix

    def __repr__(self) -> str:
        return (
            f"SparseSimulator(qubits={self.num_qubits}, "
            f"nonzero={self.num_nonzero}, "
            f"sparsity={self.sparsity:.4f})"
        )
