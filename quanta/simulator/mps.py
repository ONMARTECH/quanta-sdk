"""
quanta.simulator.mps -- Matrix Product State simulator.

Tensor network simulator for circuits with low entanglement.
Represents the quantum state as a chain of tensors:

    |ψ⟩ = Σ A[0]^{i₁} · A[1]^{i₂} · ... · A[n-1]^{iₙ} |i₁i₂...iₙ⟩

Memory: O(n · χ²) where χ = bond dimension
Dense:  O(2^n) — exponential, max ~27 qubits
MPS:    O(n · χ²) — polynomial for fixed χ, 100+ qubits

The bond dimension χ controls accuracy vs memory trade-off:
    χ = 1: product state (no entanglement)
    χ = 2^(n/2): exact (equivalent to dense)
    χ = 32-256: practical compromise

Limitations:
    - Highly entangled states require large χ (exponential)
    - Deep random circuits → MPS becomes dense
    - Best for: QAOA, VQE, quantum chemistry, 1D systems

Example:
    >>> from quanta.simulator.mps import MPSSimulator
    >>> sim = MPSSimulator(100, chi_max=64)  # 100 qubits!
    >>> sim.apply("H", (0,))
    >>> for i in range(99):
    ...     sim.apply("CX", (i, i+1))
    >>> sim.truncation_error  # Check accuracy
    0.0  # GHZ state is exact at χ=2
"""

from __future__ import annotations

import numpy as np

from quanta.core.gates import GATE_REGISTRY, MultiParametricGate, ParametricGate
from quanta.core.types import QuantaError
from quanta.simulator.base import SimulatorBackend

__all__ = ["MPSSimulator"]


class MPSSimulatorError(QuantaError):
    """MPS simulator error."""


class MPSSimulator(SimulatorBackend):
    """Matrix Product State quantum simulator.

    Uses SVD-based tensor decomposition to maintain a compressed
    representation of the quantum state. Accuracy controlled by
    bond dimension parameter chi_max.

    Args:
        num_qubits: Number of qubits (practically unlimited for low χ).
        seed: Random seed for measurement sampling.
        chi_max: Maximum bond dimension (controls accuracy/memory trade-off).
            Higher → more accurate, more memory. Default 64.
    """

    def __init__(
        self,
        num_qubits: int,
        seed: int | None = None,
        chi_max: int = 64,
        **_kwargs: object,
    ) -> None:
        self.num_qubits = num_qubits
        self.chi_max = chi_max
        self._rng = np.random.default_rng(seed)
        self._total_trunc_error: float = 0.0

        # Initialize MPS: |00...0⟩
        # Each tensor A[i] has shape (χ_left, 2, χ_right)
        # For |00...0⟩, all tensors are [[[1, 0]]] with χ=1
        self._tensors: list[np.ndarray] = []
        for _i in range(num_qubits):
            # Shape: (1, 2, 1) — bond dim 1 on each side
            t = np.zeros((1, 2, 1), dtype=complex)
            t[0, 0, 0] = 1.0  # |0⟩ state
            self._tensors.append(t)

    # ── Gate Application ──

    def apply(
        self,
        gate_name: str,
        qubits: tuple[int, ...],
        params: tuple[float, ...] = (),
    ) -> None:
        """Applies a quantum gate to the MPS state."""
        matrix = self._get_gate_matrix(gate_name, params)

        if len(qubits) == 1:
            self._apply_1q(matrix, qubits[0])
        elif len(qubits) == 2:
            self._apply_2q(matrix, qubits[0], qubits[1])
        else:
            # For 3+ qubit gates, decompose into sequence of 2q
            self._apply_nq(matrix, qubits)

    def _apply_1q(self, matrix: np.ndarray, qubit: int) -> None:
        """Single-qubit gate: contract with local tensor.

        A[q] has shape (χ_L, 2, χ_R).
        U has shape (2, 2).
        Result: A'[q] = U @ A[q] along physical index.
        """
        # A[q]: (χ_L, 2, χ_R) → contract U(2,2) on axis 1
        t = self._tensors[qubit]  # (χ_L, 2, χ_R)
        # Reshape to (χ_L * χ_R, 2), apply U, reshape back
        chi_l, _, chi_r = t.shape
        t_flat = t.reshape(chi_l, 2, chi_r).transpose(0, 2, 1).reshape(-1, 2)
        # t_flat: (χ_L * χ_R, 2)
        result = t_flat @ matrix.T  # (χ_L * χ_R, 2)
        self._tensors[qubit] = result.reshape(chi_l, chi_r, 2).transpose(0, 2, 1)

    def _apply_2q(
        self, matrix: np.ndarray, q0: int, q1: int
    ) -> None:
        """Two-qubit gate on adjacent or non-adjacent qubits.

        For adjacent qubits (|q1-q0| == 1):
            1. Contract A[q0] and A[q1] into Θ (χ_L, 4, χ_R)
            2. Apply gate: Θ' = U @ Θ
            3. SVD to split back: Θ' = A'[q0] · S · A'[q1]
            4. Truncate to chi_max

        For non-adjacent: swap qubits to make adjacent, apply, swap back.
        """
        if abs(q1 - q0) == 1:
            self._apply_2q_adjacent(matrix, min(q0, q1), max(q0, q1),
                                    swapped=(q0 > q1))
        else:
            # SWAP chain: bring qubits adjacent
            self._apply_2q_long_range(matrix, q0, q1)

    def _apply_2q_adjacent(
        self,
        matrix: np.ndarray,
        left: int,
        right: int,
        swapped: bool = False,
    ) -> None:
        """Apply 2-qubit gate to adjacent qubits left, left+1."""
        A = self._tensors[left]    # (χ_L, 2, χ_M)
        B = self._tensors[right]   # (χ_M, 2, χ_R)

        chi_l = A.shape[0]
        chi_r = B.shape[2]

        # Contract: Θ = A · B → (χ_L, 2, 2, χ_R)
        # A: (χ_L, 2, χ_M), B: (χ_M, 2, χ_R)
        theta = np.einsum("ijk,klm->ijlm", A, B)  # (χ_L, 2, 2, χ_R)

        # Reshape for gate application: (χ_L, 4, χ_R)
        if swapped:
            # If q0 > q1, we need to swap physical indices
            theta = theta.transpose(0, 2, 1, 3)  # swap physical indices

        theta = theta.reshape(chi_l, 4, chi_r)

        # Apply gate: U(4,4) @ Θ(χ_L, 4, χ_R)
        gate = matrix.reshape(4, 4)
        theta_new = np.einsum("ij,kjl->kil", gate, theta)  # (χ_L, 4, χ_R)

        if swapped:
            theta_new = theta_new.reshape(chi_l, 2, 2, chi_r)
            theta_new = theta_new.transpose(0, 2, 1, 3)
            theta_new = theta_new.reshape(chi_l, 4, chi_r)

        # SVD: split back into two tensors
        theta_mat = theta_new.reshape(chi_l * 2, 2 * chi_r)
        U, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)

        # Truncate to chi_max
        chi_new = min(len(S), self.chi_max)
        if chi_new < len(S):
            trunc = np.sum(S[chi_new:] ** 2)
            self._total_trunc_error += trunc

        U = U[:, :chi_new]
        S = S[:chi_new]
        Vh = Vh[:chi_new, :]

        # Absorb singular values into right tensor (right-canonical)
        # A' = U reshaped, B' = S @ Vh reshaped
        self._tensors[left] = U.reshape(chi_l, 2, chi_new)
        self._tensors[right] = (np.diag(S) @ Vh).reshape(chi_new, 2, chi_r)

    def _apply_2q_long_range(
        self, matrix: np.ndarray, q0: int, q1: int
    ) -> None:
        """Apply 2-qubit gate on non-adjacent qubits via SWAP chain."""
        swap_matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=complex)

        # Move q0 next to q1
        target_pos = q1 - 1 if q0 < q1 else q1 + 1
        current = q0

        # SWAP chain: move q0 toward q1
        if current < target_pos:
            for i in range(current, target_pos):
                self._apply_2q_adjacent(swap_matrix, i, i + 1)
        else:
            for i in range(current, target_pos, -1):
                self._apply_2q_adjacent(swap_matrix, i - 1, i, swapped=True)

        # Apply the actual gate
        left, right = min(target_pos, q1), max(target_pos, q1)
        self._apply_2q_adjacent(matrix, left, right,
                                swapped=(target_pos > q1))

        # SWAP chain back
        if current < target_pos:
            for i in range(target_pos - 1, current - 1, -1):
                self._apply_2q_adjacent(swap_matrix, i, i + 1)
        else:
            for i in range(target_pos + 1, current + 1):
                self._apply_2q_adjacent(swap_matrix, i - 1, i, swapped=True)

    def _apply_nq(
        self, matrix: np.ndarray, qubits: tuple[int, ...]
    ) -> None:
        """N-qubit gate decomposition (fallback to dense contraction)."""
        # For 3+ qubit gates, convert to dense, apply, convert back
        if self.num_qubits <= 20:
            self._fallback_dense(matrix, qubits)
        else:
            raise MPSSimulatorError(
                f"Cannot apply {len(qubits)}-qubit gate on {self.num_qubits}-qubit "
                f"MPS. Decompose gate into 1q and 2q gates first."
            )

    def _fallback_dense(
        self, matrix: np.ndarray, qubits: tuple[int, ...]
    ) -> None:
        """Fallback: convert to dense, apply gate, convert back."""
        sv = self._to_statevector()
        from quanta.simulator.statevector import StateVectorSimulator
        dense = StateVectorSimulator(self.num_qubits)
        dense._state = sv
        dense.apply("I", (0,), ())  # no-op to init
        # Direct matrix application
        n_gate = len(qubits)
        gate_tensor = matrix.reshape([2] * (2 * n_gate))
        sv_tensor = sv.reshape([2] * self.num_qubits)
        gate_axes = list(range(n_gate, 2 * n_gate))
        state_axes = list(qubits)
        result = np.tensordot(gate_tensor, sv_tensor, axes=(gate_axes, state_axes))
        result = np.moveaxis(result, list(range(n_gate)), list(qubits))
        self._from_statevector(result.reshape(-1))

    # ── Measurement ──

    def probabilities(self) -> np.ndarray:
        """Returns measurement probabilities (converts to dense)."""
        sv = self._to_statevector()
        return np.abs(sv) ** 2

    def sample(self, shots: int) -> dict[str, int]:
        """Sequential qubit-by-qubit sampling (memory efficient)."""
        n = self.num_qubits
        fmt = f"0{n}b"
        counts: dict[str, int] = {}

        for _ in range(shots):
            bitstring = self._sample_single()
            key = format(bitstring, fmt)
            counts[key] = counts.get(key, 0) + 1

        return counts

    def _sample_single(self) -> int:
        """Sample a single bitstring by sequential qubit measurement.

        Uses the MPS structure to efficiently sample one qubit at a time,
        updating partial contractions. O(n · χ²) per sample.
        """
        n = self.num_qubits
        result = 0

        # Start from left, accumulate partial contraction
        # env[i]: left environment tensor up to qubit i
        env = np.ones((1, 1), dtype=complex)

        for i in range(n):
            A = self._tensors[i]  # (χ_L, 2, χ_R)

            # Probability of qubit i = 0
            A0 = A[:, 0, :]  # (χ_L, χ_R)
            A1 = A[:, 1, :]  # (χ_L, χ_R)

            env0 = env @ A0  # (1, χ_R)
            env1 = env @ A1  # (1, χ_R)

            # Compute remaining norm for each choice
            p0 = np.real(np.sum(env0 * np.conj(env0)))
            p1 = np.real(np.sum(env1 * np.conj(env1)))

            total = p0 + p1
            if total < 1e-15:
                break

            p0 /= total

            # Sample
            if self._rng.random() < p0:
                env = env0 / np.sqrt(p0 * total) if p0 > 1e-15 else env0
            else:
                result |= (1 << (n - 1 - i))  # MSB convention
                env = env1 / np.sqrt(p1 * total) if p1 > 1e-15 else env1

        return result

    # ── State Access ──

    @property
    def state(self) -> np.ndarray:
        """Returns dense statevector (for cross-validation testing).

        WARNING: Exponential memory for large qubit counts.
        Only use for n ≤ 20.
        """
        return self._to_statevector()

    @state.setter
    def state(self, new_state: np.ndarray) -> None:
        """Sets MPS from dense statevector."""
        self._from_statevector(new_state)

    @property
    def max_qubits(self) -> int:
        """No hard limit — bounded by chi_max and entanglement."""
        return 10000

    @property
    def truncation_error(self) -> float:
        """Accumulated truncation error from all SVD operations."""
        return self._total_trunc_error

    @property
    def bond_dimensions(self) -> list[int]:
        """Current bond dimensions at each cut."""
        dims = []
        for i in range(self.num_qubits - 1):
            dims.append(self._tensors[i].shape[2])
        return dims

    @property
    def max_bond_dim(self) -> int:
        """Largest current bond dimension."""
        return max(self.bond_dimensions) if self.num_qubits > 1 else 1

    @property
    def memory_bytes(self) -> int:
        """Estimated memory usage in bytes."""
        total = 0
        for t in self._tensors:
            total += t.nbytes
        return total

    # ── Internal Conversion ──

    def _to_statevector(self) -> np.ndarray:
        """Contracts MPS to full statevector. O(2^n) memory!"""
        result = self._tensors[0]  # (1, 2, χ)
        for i in range(1, self.num_qubits):
            # Contract: result(χ_L, 2^i, χ_M) · A[i](χ_M, 2, χ_R)
            # → result(χ_L, 2^(i+1), χ_R)
            result = np.einsum("...j,jkl->...kl", result, self._tensors[i])

        # result should be (1, 2^n, 1)
        return result.reshape(-1)

    def _from_statevector(self, sv: np.ndarray) -> None:
        """Decompose dense statevector into MPS via sequential SVD.

        |ψ⟩ = A[0] · A[1] · ... · A[n-1]

        Each step: reshape current block, SVD, truncate.
        """
        n = self.num_qubits
        psi = sv.reshape([2] * n)

        self._tensors = []
        self._total_trunc_error = 0.0

        remaining = psi
        chi_left = 1

        for _i in range(n - 1):
            # Reshape: (χ_L · 2, 2^remaining)
            mat = remaining.reshape(chi_left * 2, -1)

            U, S, Vh = np.linalg.svd(mat, full_matrices=False)

            # Truncate
            chi_new = min(len(S), self.chi_max)
            if chi_new < len(S):
                self._total_trunc_error += np.sum(S[chi_new:] ** 2)

            U = U[:, :chi_new]
            S = S[:chi_new]
            Vh = Vh[:chi_new, :]

            # Store A[i]: (χ_L, 2, χ_new)
            self._tensors.append(U.reshape(chi_left, 2, chi_new))

            # Pass S into remaining
            remaining = np.diag(S) @ Vh
            chi_left = chi_new

        # Last tensor: (χ_left, 2, 1)
        self._tensors.append(remaining.reshape(chi_left, 2, 1))

    # ── Gate Matrix Resolution ──

    def _get_gate_matrix(
        self, name: str, params: tuple[float, ...]
    ) -> np.ndarray:
        """Gets gate matrix from registry."""
        gate = GATE_REGISTRY.get(name)
        if gate is None:
            raise MPSSimulatorError(f"Unknown gate: {name}")

        if isinstance(gate, MultiParametricGate):
            if not params:
                raise MPSSimulatorError(f"{name} requires parameters")
            return gate(*params).matrix
        if isinstance(gate, ParametricGate):
            if not params:
                raise MPSSimulatorError(f"{name} requires parameters")
            return gate(params[0]).matrix

        return gate.matrix

    def __repr__(self) -> str:
        return (
            f"MPSSimulator(qubits={self.num_qubits}, "
            f"chi_max={self.chi_max}, "
            f"max_bond={self.max_bond_dim}, "
            f"trunc_err={self.truncation_error:.2e})"
        )
