"""
quanta.simulator.density_matrix -- Density matrix simulator.

Simulates quantum circuits using density matrices (rho = |psi><psi|)
instead of statevectors. Enables:
  - Mixed states (noise, decoherence)
  - Kraus operator noise channels natively
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

import numpy as np

from quanta.core.gates import GATE_REGISTRY, MultiParametricGate, ParametricGate
from quanta.core.types import QuantaError

__all__ = ["DensityMatrixSimulator"]


class DensityMatrixError(QuantaError):
    """Density matrix simulator error."""


class DensityMatrixSimulator:
    """Density matrix simulator for mixed-state quantum circuits.

    State is represented as rho (2^n x 2^n complex matrix).
    Gate application: rho' = U @ rho @ U.H
    Noise (Kraus): rho' = sum_k (K_k @ rho @ K_k.H)

    Args:
        num_qubits: Number of qubits to simulate.
        seed: Random seed for reproducibility.
    """

    MAX_QUBITS = 13  # Memory: 4^13 * 16 = ~1 GB

    __slots__ = ("num_qubits", "_rho", "_rng")

    def __init__(self, num_qubits: int, seed: int | None = None) -> None:
        if num_qubits > self.MAX_QUBITS:
            raise DensityMatrixError(
                f"Max {self.MAX_QUBITS} qubits for density matrix. "
                f"Requested: {num_qubits}"
            )

        self.num_qubits = num_qubits
        self._rng = np.random.default_rng(seed)

        dim = 2 ** num_qubits
        self._rho = np.zeros((dim, dim), dtype=complex)
        self._rho[0, 0] = 1.0  # |0><0|

    def apply(
        self,
        gate_name: str,
        qubits: tuple[int, ...],
        params: tuple[float, ...] = (),
    ) -> None:
        """Applies a unitary gate: rho' = U @ rho @ U.H"""
        matrix = self._get_gate_matrix(gate_name, params)
        full = self._expand_gate(matrix, qubits)
        self._rho = full @ self._rho @ full.conj().T
        self._enforce_cptp()

    def apply_kraus(
        self,
        kraus_ops: list[np.ndarray],
        qubits: tuple[int, ...],
    ) -> None:
        """Applies a Kraus channel: rho' = sum_k (K_k @ rho @ K_k.H)

        This is the native way to apply noise in density matrix formalism.
        No stochastic sampling needed (unlike statevector noise).

        Args:
            kraus_ops: List of Kraus operators. Must satisfy sum(K.H @ K) = I.
            qubits: Target qubits.
        """
        if not kraus_ops:
            raise DensityMatrixError("Kraus operators list cannot be empty.")

        target_dim = 2 ** len(qubits)
        completeness = np.zeros((target_dim, target_dim), dtype=complex)
        for k in kraus_ops:
            k_mat = np.asarray(k, dtype=complex)
            if k_mat.shape != (target_dim, target_dim):
                raise DensityMatrixError(
                    f"Kraus operator shape {k_mat.shape} does not match "
                    f"target qubits dimension ({target_dim}, {target_dim})"
                )
            completeness += k_mat.conj().T @ k_mat

        eye = np.eye(target_dim, dtype=complex)
        dev = float(np.max(np.abs(completeness - eye)))
        if dev >= 1e-12:
            raise DensityMatrixError(
                f"Kraus operators violate completeness relation: "
                f"||sum K^dagger K - I||_inf = {dev:.2e} >= 1e-12"
            )

        new_rho = np.zeros_like(self._rho)
        for k in kraus_ops:
            full_k = self._expand_gate(k, qubits)
            new_rho += full_k @ self._rho @ full_k.conj().T
        self._rho = new_rho
        self._enforce_cptp()

    def _enforce_cptp(self) -> None:
        """Enforces Hermiticity, trace preservation Tr(rho)=1, and complete positivity rho >= 0."""
        # 1. Hermiticity: rho = (rho + rho^dagger) / 2
        self._rho = (self._rho + self._rho.conj().T) / 2.0

        # 2. Trace preservation: |Tr(rho) - 1.0| < 1e-12
        tr = np.trace(self._rho)
        tr_val = float(np.real(tr))
        if abs(tr_val - 1.0) >= 1e-12 or abs(float(np.imag(tr))) >= 1e-12:
            raise DensityMatrixError(
                f"Density matrix violates trace conservation: Tr(rho) = {tr_val:.4e} "
                f"(deviation {abs(tr_val - 1.0):.2e} >= 1e-12)"
            )

        # 3. Complete positivity: eigenvalues >= -1e-12
        evals, evecs = np.linalg.eigh(self._rho)
        min_eval = float(np.min(evals))
        if min_eval < -1e-12:
            raise DensityMatrixError(
                f"Density matrix violates complete positivity: "
                f"min eigenvalue = {min_eval:.2e} < -1e-12"
            )

        # Clip numerical negative eigenvalues and re-normalize trace
        clipped_evals = np.maximum(evals, 0.0)
        s = float(np.sum(clipped_evals))
        if s > 0:
            clipped_evals /= s
        self._rho = evecs @ np.diag(clipped_evals) @ evecs.conj().T

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

    def _expand_gate(
        self, gate: np.ndarray, qubits: tuple[int, ...]
    ) -> np.ndarray:
        """Expands gate to full system using tensor contraction."""
        n = self.num_qubits
        num_gate_qubits = len(qubits)
        dim = 2 ** n

        # Tensor contraction approach
        state_shape = [2] * n
        gate_tensor = gate.reshape([2] * (2 * num_gate_qubits))

        # Build identity in tensor form, then contract
        full = np.zeros((dim, dim), dtype=complex)

        for i in range(dim):
            ei = np.zeros(dim, dtype=complex)
            ei[i] = 1.0
            ei_t = ei.reshape(state_shape)

            gate_axes = list(range(num_gate_qubits, 2 * num_gate_qubits))
            state_axes = list(qubits)
            out = np.tensordot(gate_tensor, ei_t, axes=(gate_axes, state_axes))
            out = np.moveaxis(out, list(range(num_gate_qubits)), list(qubits))
            full[:, i] = out.reshape(-1)

        return full

    def _get_gate_matrix(
        self, name: str, params: tuple[float, ...]
    ) -> np.ndarray:
        """Gets gate matrix from name."""
        gate = GATE_REGISTRY.get(name)
        if gate is None:
            raise DensityMatrixError(f"Unknown gate: {name}")

        # Parametric gates need params to build matrix
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
        counts: dict[str, int] = {}
        for idx in indices:
            bs = format(idx, f"0{n}b")
            counts[bs] = counts.get(bs, 0) + 1
        return counts

    @property
    def purity(self) -> float:
        """Trace(rho^2). Pure state = 1.0, maximally mixed = 1/d."""
        return float(np.real(np.trace(self._rho @ self._rho)))

    @property
    def state(self) -> np.ndarray:
        """Copy of the density matrix."""
        return self._rho.copy()
