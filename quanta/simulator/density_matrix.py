"""
quanta.simulator.density_matrix -- Density matrix simulator.

Simulates quantum circuits using density matrices (rho = |psi><psi|)
instead of statevectors. Enables:
  - Mixed states (noise, decoherence)
  - Kraus operator noise channels natively via exact O(4^n) tensor contraction
  - Trace operations (partial trace)

Tradeoff: O(4^n) memory vs statevector's O(2^n), so max ~13 qubits.
But critical for real-world accuracy when noise matters.

Example:
    >>> sim = DensityMatrixSimulator(num_qubits=2)
    >>> sim.apply("H", (0,))
    >>> sim.apply("CX", (0, 1))
    >>> sim.probabilities()
    array([0.5, 0. , 0. , 0.5])
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quanta.core.gates import GATE_REGISTRY, MultiParametricGate, ParametricGate
from quanta.core.types import QuantaError
from quanta.simulator.base import SimulatorBackend
from quanta.simulator.noise import ReadoutError, resolve_channel_qubit_targets

__all__ = ["DensityMatrixError", "DensityMatrixSimulator"]


class DensityMatrixError(QuantaError):
    """Density matrix simulator error."""


class DensityMatrixSimulator(SimulatorBackend):
    """Vectorized density matrix simulator for mixed-state quantum circuits.

    State is represented as rho (2^n x 2^n complex matrix), operated on via
    [2] * (2 * n) ket/bra tensor contractions in O(2^k * 4^n) per k-qubit op.
    Gate application: rho' = U @ rho @ U.H
    Noise (Kraus): rho' = sum_k (K_k @ rho @ K_k.H)

    Args:
        num_qubits: Number of qubits to simulate.
        seed: Random seed for reproducibility.
    """

    MAX_QUBITS = 13  # Memory: 4^13 * 16 = ~1 GB

    __slots__ = ("num_qubits", "_rho", "_rng", "_seed")

    def __init__(self, num_qubits: int, seed: int | None = None) -> None:
        if num_qubits > self.MAX_QUBITS:
            raise DensityMatrixError(
                f"Max {self.MAX_QUBITS} qubits for density matrix. "
                f"Requested: {num_qubits}"
            )

        self.num_qubits = num_qubits
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        dim = 2 ** num_qubits
        self._rho = np.zeros((dim, dim), dtype=complex)
        self._rho[0, 0] = 1.0  # |0><0|

    @property
    def max_qubits(self) -> int:
        return self.MAX_QUBITS

    def _contract_superoperator(
        self, op_matrix: np.ndarray, qubits: tuple[int, ...]
    ) -> np.ndarray:
        """Computes M @ rho @ M† in O(2^k * 4^n) using [2]*(2n) tensor contraction."""
        n = self.num_qubits
        dim = 1 << n
        m = len(qubits)
        if m == 0:
            return self._rho.copy()

        op_tensor = np.asarray(op_matrix, dtype=complex).reshape([2] * (2 * m))
        rho_tensor = self._rho.reshape([2] * (2 * n))

        in_axes = list(range(m, 2 * m))
        out_axes = list(range(m))
        ket_axes = list(qubits)
        bra_axes = [n + q for q in qubits]

        # Contract M on ket axes: M_{i, i'} rho_{i', j'}
        tmp = np.tensordot(op_tensor, rho_tensor, axes=(in_axes, ket_axes))
        tmp = np.moveaxis(tmp, out_axes, ket_axes)

        # Contract M* on bra axes: tmp_{i, j'} conj(M)_{j, j'}
        tmp = np.tensordot(op_tensor.conj(), tmp, axes=(in_axes, bra_axes))
        tmp = np.moveaxis(tmp, out_axes, bra_axes)

        return np.ascontiguousarray(tmp).reshape(dim, dim)

    def apply(
        self,
        gate_name: str,
        qubits: tuple[int, ...],
        params: tuple[float, ...] = (),
    ) -> None:
        """Applies a unitary gate: rho' = U @ rho @ U.H"""
        matrix = self._get_gate_matrix(gate_name, params)
        self._rho = self._contract_superoperator(matrix, qubits)
        self._enforce_cptp()

    def apply_kraus(
        self,
        kraus_ops: list[np.ndarray],
        qubits: tuple[int, ...],
    ) -> None:
        """Applies a Kraus channel: rho' = sum_k (K_k @ rho @ K_k.H)

        Args:
            kraus_ops: List of Kraus operators. Must satisfy sum(K.H @ K) = I.
            qubits: Target qubits.
        """
        if not kraus_ops:
            raise DensityMatrixError("Kraus operators list cannot be empty.")

        target_dim = 2 ** len(qubits)
        completeness = np.zeros((target_dim, target_dim), dtype=complex)
        checked_ops: list[np.ndarray] = []
        for k in kraus_ops:
            k_mat = np.asarray(k, dtype=complex)
            if k_mat.shape != (target_dim, target_dim):
                raise DensityMatrixError(
                    f"Kraus operator shape {k_mat.shape} does not match "
                    f"target qubits dimension ({target_dim}, {target_dim})"
                )
            completeness += k_mat.conj().T @ k_mat
            checked_ops.append(k_mat)

        eye = np.eye(target_dim, dtype=complex)
        dev = float(np.max(np.abs(completeness - eye)))
        if dev >= 1e-12:
            raise DensityMatrixError(
                f"Kraus operators violate completeness relation: "
                f"||sum K^dagger K - I||_inf = {dev:.2e} >= 1e-12"
            )

        new_rho = np.zeros_like(self._rho)
        for k_mat in checked_ops:
            new_rho += self._contract_superoperator(k_mat, qubits)
        self._rho = new_rho
        self._enforce_cptp()

    def apply_noise(
        self,
        noise_model: Any,
        qubits: tuple[int, ...],
        rng: np.random.Generator | None = None,
    ) -> None:
        """Applies every applicable gate-level channel's Kraus map deterministically."""
        del rng  # Deterministic CPTP evolution requires no stochastic sampling
        channels = getattr(noise_model, "channels", [])
        for channel in channels:
            if isinstance(channel, ReadoutError):
                continue
            targets_list = resolve_channel_qubit_targets(
                channel, qubits, self.num_qubits
            )
            kraus_ops = channel.kraus_operators()
            for target in targets_list:
                self.apply_kraus(kraus_ops, target)

    def _enforce_cptp(self) -> None:
        """Enforces Hermiticity and trace conservation Tr(rho)=1 in O(4^n)."""
        # 1. Hermiticity: rho = (rho + rho^dagger) / 2
        self._rho = 0.5 * (self._rho + self._rho.conj().T)

        # 2. Trace preservation: |Tr(rho) - 1.0| < 1e-12
        tr = np.trace(self._rho)
        tr_val = float(np.real(tr))
        if abs(tr_val - 1.0) >= 1e-12 or abs(float(np.imag(tr))) >= 1e-12:
            raise DensityMatrixError(
                f"Density matrix violates trace conservation: Tr(rho) = {tr_val:.4e} "
                f"(deviation {abs(tr_val - 1.0):.2e} >= 1e-12)"
            )

        # 3. Diagonal positivity guard (Necessary condition rho_ii >= 0 in O(2^n))
        min_diag = float(np.min(np.real(np.diag(self._rho))))
        if min_diag < -1e-12:
            raise DensityMatrixError(
                f"Density matrix violates positivity: min diagonal = {min_diag:.2e} < -1e-12"
            )

        if tr_val > 0.0:
            self._rho /= tr_val

    def apply_depolarizing(self, qubit: int, p: float) -> None:
        """Applies depolarizing noise on a single qubit.

        rho' = (1-p) * rho + (p/3) * (X rho X + Y rho Y + Z rho Z)
        """
        if p <= 0:
            return

        eye = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        kraus = [
            np.sqrt(1 - p) * eye,
            np.sqrt(p / 3) * X,
            np.sqrt(p / 3) * Y,
            np.sqrt(p / 3) * Z,
        ]
        self.apply_kraus(kraus, (qubit,))

    def _get_gate_matrix(
        self, name: str, params: tuple[float, ...]
    ) -> np.ndarray:
        """Gets gate matrix from name."""
        gate = GATE_REGISTRY.get(name)
        if gate is None:
            raise DensityMatrixError(f"Unknown gate: {name}")

        if isinstance(gate, MultiParametricGate):
            if not params:
                raise DensityMatrixError(f"{name} requires parameters")
            return gate(*params).matrix
        if isinstance(gate, ParametricGate):
            if not params:
                raise DensityMatrixError(f"{name} requires parameters")
            return gate(params[0]).matrix

        return gate.matrix

    def probabilities(self) -> np.ndarray:
        """Returns measurement probabilities: P(i) = rho[i,i]."""
        probs = np.real(np.diag(self._rho))
        probs = np.maximum(probs, 0.0)
        s = float(np.sum(probs))
        if s > 0:
            probs /= s
        return probs

    def sample(self, shots: int) -> dict[str, int]:
        """Samples measurement results from density matrix."""
        probs = self.probabilities()
        n = self.num_qubits
        indices = self._rng.choice(len(probs), size=shots, p=probs)
        unique, unique_counts = np.unique(indices, return_counts=True)
        fmt = f"0{n}b"
        return {
            format(idx, fmt): int(cnt)
            for idx, cnt in zip(unique, unique_counts, strict=True)
        }

    @property
    def purity(self) -> float:
        """Trace(rho^2). Pure state = 1.0, maximally mixed = 1/d."""
        # Tr(rho^2) = sum_{i,j} |rho_{ij}|^2 since rho is Hermitian (O(4^n) vs O(8^n)!)
        return float(np.real(np.vdot(self._rho, self._rho)))

    @property
    def state(self) -> np.ndarray:
        """Copy of the density matrix."""
        return self._rho.copy()

    @state.setter
    def state(self, value: Any) -> None:
        """Sets the density matrix with CPTP validation."""
        dim = 1 << self.num_qubits
        arr = np.asarray(value, dtype=complex)
        if arr.shape != (dim, dim):
            raise DensityMatrixError(
                f"Density matrix shape mismatch: expected {(dim, dim)}, got {arr.shape}"
            )
        self._rho = arr.copy()
        self._enforce_cptp()

    def reset(self) -> None:
        """Resets the density matrix to |00...0><00...0|."""
        self.__init__(self.num_qubits, seed=self._seed)  # type: ignore[misc]
