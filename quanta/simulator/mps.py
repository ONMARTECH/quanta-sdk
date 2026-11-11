"""
quanta.simulator.mps -- Matrix Product State simulator.

Tensor network simulator for circuits with low entanglement.
Represents the quantum state as a chain of tensors in mixed-canonical form:

    |ψ⟩ = Σ A[0]^{i₁} · A[1]^{i₂} · ... · A[n-1]^{iₙ} |i₁i₂...iₙ⟩

Memory: O(n · χ²) where χ = bond dimension
Dense:  O(2^n) — exponential, max ~27 qubits
MPS:    O(n · χ²) — polynomial for fixed χ, 100+ qubits

The bond dimension χ controls accuracy vs memory trade-off:
    χ = 1: product state (no entanglement)
    χ = 2^(n/2): exact (equivalent to dense)
    χ = 32-256: practical compromise

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

import math
from typing import Any

import numpy as np

from quanta.core.gates import GATE_REGISTRY, MultiParametricGate, ParametricGate
from quanta.core.types import QuantaError
from quanta.simulator.base import SimulatorBackend
from quanta.simulator.noise import ReadoutError, resolve_channel_qubit_targets

__all__ = ["MPSSimulator", "MPSSimulatorError"]

_SWAP_MATRIX = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=complex,
)


class MPSSimulatorError(QuantaError):
    """MPS simulator error."""


class MPSSimulator(SimulatorBackend):
    """Matrix Product State quantum simulator in mixed-canonical form.

    Uses QR canonicalization + SVD-based tensor decomposition to maintain an
    orthogonality center `self._ortho_center` so every two-site SVD truncates
    exact global Schmidt singular values and sequential Born sampling is exact.

    Args:
        num_qubits: Number of qubits (practically unlimited for low χ).
        seed: Random seed for measurement sampling.
        chi_max: Maximum bond dimension (controls accuracy/memory trade-off).
            Higher → more accurate, more memory. Default 64.
        cutoff: Singular value pruning threshold. Default 1e-12.
    """

    def __init__(
        self,
        num_qubits: int,
        seed: int | None = None,
        chi_max: int = 64,
        cutoff: float = 1e-12,
        **_kwargs: object,
    ) -> None:
        self.num_qubits = num_qubits
        self.chi_max = chi_max
        self.cutoff = cutoff
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._total_trunc_error: float = 0.0
        self._ortho_center: int | None = 0 if num_qubits > 0 else None

        # Initialize MPS: |00...0⟩
        # Each tensor A[i] has shape (χ_left, 2, χ_right)
        self._tensors: list[np.ndarray] = []
        for _i in range(num_qubits):
            t = np.zeros((1, 2, 1), dtype=complex)
            t[0, 0, 0] = 1.0  # |0⟩ state
            self._tensors.append(t)

    # ── Canonical Form & Orthogonality Center ──

    def _move_ortho_center(self, target: int) -> None:
        """Moves the mixed-canonical orthogonality center to `target` in O(|Δ|·χ³).

        After `_move_ortho_center(target)`:
          - Sites `0 .. target-1` are strictly left-isometric: sum_s A_s† A_s = I
          - Sites `target+1 .. n-1` are strictly right-isometric: sum_s B_s B_s† = I
        """
        n = self.num_qubits
        if n <= 1:
            self._ortho_center = 0 if n == 1 else None
            return
        if not (0 <= target < n):
            raise MPSSimulatorError(
                f"Orthogonality center {target} out of bounds for {n} qubits."
            )

        if self._ortho_center is None:
            left_start = 0
            right_start = n - 1
        else:
            if self._ortho_center == target:
                return
            left_start = self._ortho_center if self._ortho_center < target else target
            right_start = self._ortho_center if self._ortho_center > target else target

        # Left-to-right QR sweep: left_start -> target - 1
        for i in range(left_start, target):
            t = self._tensors[i]
            chi_l, d, chi_r = t.shape
            mat = t.reshape(chi_l * d, chi_r)
            q, r = np.linalg.qr(mat, mode="reduced")
            self._tensors[i] = q.reshape(chi_l, d, -1)
            self._tensors[i + 1] = np.tensordot(r, self._tensors[i + 1], axes=(1, 0))

        # Right-to-left LQ sweep (via QR of conjugate transpose): right_start -> target + 1
        for i in range(right_start, target, -1):
            t = self._tensors[i]
            chi_l, d, chi_r = t.shape
            mat = t.reshape(chi_l, d * chi_r)
            q, r = np.linalg.qr(mat.conj().T, mode="reduced")
            q_dag = q.conj().T  # orthonormal rows: q_dag @ q_dag† = I
            l_mat = r.conj().T  # mat == l_mat @ q_dag
            self._tensors[i] = q_dag.reshape(-1, d, chi_r)
            self._tensors[i - 1] = np.tensordot(self._tensors[i - 1], l_mat, axes=(2, 0))

        self._ortho_center = target

    def _split_and_truncate_2q(
        self, theta_new: np.ndarray, left: int, right: int, chi_l: int, chi_r: int
    ) -> None:
        """Splits a two-site orthogonality-centered tensor via SVD with cutoff & chi_max."""
        theta_mat = theta_new.reshape(chi_l * 2, 2 * chi_r)
        u_mat, s_vals, vh_mat = np.linalg.svd(theta_mat, full_matrices=False)

        above_cutoff = int(np.sum(s_vals > self.cutoff))
        chi_keep = max(1, above_cutoff)
        chi_new = min(len(s_vals), chi_keep, self.chi_max)

        if chi_new < len(s_vals):
            discarded = s_vals[chi_new:]
            self._total_trunc_error += float(np.sum(discarded ** 2))
            u_mat = u_mat[:, :chi_new]
            s_vals = s_vals[:chi_new]
            s_kept_norm = float(np.linalg.norm(s_vals))
            if s_kept_norm > 1e-15:
                s_vals = s_vals / s_kept_norm
            vh_mat = vh_mat[:chi_new, :]
        else:
            u_mat = u_mat[:, :chi_new]
            s_vals = s_vals[:chi_new]
            vh_mat = vh_mat[:chi_new, :]

        # A[left] is strictly left-isometric (U† U = I); A[right] holds ortho center
        self._tensors[left] = u_mat.reshape(chi_l, 2, chi_new)
        self._tensors[right] = (s_vals[:, None] * vh_mat).reshape(chi_new, 2, chi_r)
        self._ortho_center = right

    # ── Gate Application ──

    def apply(
        self,
        gate_name: str,
        qubits: tuple[int, ...],
        params: tuple[float, ...] = (),
    ) -> None:
        """Applies a quantum gate to the MPS state."""
        if len(qubits) == 1:
            matrix = self._get_gate_matrix(gate_name, params)
            self._apply_1q(matrix, qubits[0])
        elif len(qubits) == 2:
            matrix = self._get_gate_matrix(gate_name, params)
            self._apply_2q(matrix, qubits[0], qubits[1])
        else:
            if self._decompose_known_multi_qubit_gate(gate_name, qubits, params):
                return
            matrix = self._get_gate_matrix(gate_name, params)
            self._apply_nq(matrix, qubits, gate_name=gate_name)

    def _decompose_known_multi_qubit_gate(
        self,
        gate_name: str,
        qubits: tuple[int, ...],
        params: tuple[float, ...],
    ) -> bool:
        """Decomposes standard 3Q/4Q gates into exact 1Q+2Q sequences in O(χ³)."""
        del params
        pi = math.pi
        if gate_name == "CCX" and len(qubits) == 3:
            q0, q1, q2 = qubits
            # Exact 6-CX Sleator-Weinfurter Toffoli decomposition
            self.apply("H", (q2,))
            self.apply("CX", (q1, q2))
            self.apply("TDG", (q2,))
            self.apply("CX", (q0, q2))
            self.apply("T", (q2,))
            self.apply("CX", (q1, q2))
            self.apply("TDG", (q2,))
            self.apply("CX", (q0, q2))
            self.apply("T", (q1,))
            self.apply("T", (q2,))
            self.apply("H", (q2,))
            self.apply("CX", (q0, q1))
            self.apply("T", (q0,))
            self.apply("TDG", (q1,))
            self.apply("CX", (q0, q1))
            return True

        if gate_name == "CSWAP" and len(qubits) == 3:
            q0, q1, q2 = qubits
            self.apply("CX", (q2, q1))
            self.apply("CCX", (q0, q1, q2))
            self.apply("CX", (q2, q1))
            return True

        if gate_name == "RCCX" and len(qubits) == 3:
            q0, q1, q2 = qubits
            # Exact 4-CX decomposition matching Quanta's RCCX unitary
            self.apply("H", (q2,))
            self.apply("CX", (q1, q2))
            self.apply("TDG", (q2,))
            self.apply("CX", (q0, q2))
            self.apply("T", (q2,))
            self.apply("CX", (q1, q2))
            self.apply("TDG", (q2,))
            self.apply("CX", (q0, q2))
            self.apply("T", (q2,))
            self.apply("H", (q2,))
            return True

        if gate_name == "RC3X" and len(qubits) == 4:
            q0, q1, q2, q3 = qubits
            # Exact 8-CX Gray-code decomposition matching Quanta's RC3X unitary
            self.apply("H", (q3,))
            self.apply("P", (q3,), (pi / 8,))
            self.apply("CX", (q2, q3))
            self.apply("P", (q3,), (-pi / 8,))
            self.apply("CX", (q1, q3))
            self.apply("P", (q3,), (pi / 8,))
            self.apply("CX", (q2, q3))
            self.apply("P", (q3,), (-pi / 8,))
            self.apply("CX", (q0, q3))
            self.apply("P", (q3,), (pi / 8,))
            self.apply("CX", (q2, q3))
            self.apply("P", (q3,), (-pi / 8,))
            self.apply("CX", (q1, q3))
            self.apply("P", (q3,), (pi / 8,))
            self.apply("CX", (q2, q3))
            self.apply("P", (q3,), (-pi / 8,))
            self.apply("CX", (q0, q3))
            self.apply("H", (q3,))
            return True

        return False

    def _apply_1q(self, matrix: np.ndarray, qubit: int) -> None:
        """Single-qubit unitary gate: contract with local tensor.

        Note: 1Q unitaries preserve both left-isometry and right-isometry,
        so `self._ortho_center` does not need to move!
        """
        t = self._tensors[qubit]  # (χ_L, 2, χ_R)
        chi_l, _, chi_r = t.shape
        t_flat = t.reshape(chi_l, 2, chi_r).transpose(0, 2, 1).reshape(-1, 2)
        result = t_flat @ matrix.T
        self._tensors[qubit] = result.reshape(chi_l, chi_r, 2).transpose(0, 2, 1)

    def _apply_2q(
        self, matrix: np.ndarray, q0: int, q1: int
    ) -> None:
        """Two-qubit gate on adjacent or non-adjacent qubits."""
        if abs(q1 - q0) == 1:
            self._apply_2q_adjacent(
                matrix, min(q0, q1), max(q0, q1), swapped=(q0 > q1)
            )
        else:
            self._apply_2q_long_range(matrix, q0, q1)

    def _apply_2q_adjacent(
        self,
        matrix: np.ndarray,
        left: int,
        right: int,
        swapped: bool = False,
    ) -> None:
        """Apply 2-qubit gate to adjacent qubits `left` and `right = left + 1`."""
        self._move_ortho_center(left)

        A = self._tensors[left]    # (χ_L, 2, χ_M)
        B = self._tensors[right]   # (χ_M, 2, χ_R)

        chi_l = A.shape[0]
        chi_r = B.shape[2]

        theta = np.einsum("ijk,klm->ijlm", A, B)  # (χ_L, 2, 2, χ_R)
        if swapped:
            theta = theta.transpose(0, 2, 1, 3)

        theta = theta.reshape(chi_l, 4, chi_r)
        gate = matrix.reshape(4, 4)
        theta_new = np.einsum("ij,kjl->kil", gate, theta)

        if swapped:
            theta_new = theta_new.reshape(chi_l, 2, 2, chi_r)
            theta_new = theta_new.transpose(0, 2, 1, 3)
            theta_new = theta_new.reshape(chi_l, 4, chi_r)

        self._split_and_truncate_2q(theta_new, left, right, chi_l, chi_r)

    def _apply_2q_long_range(
        self, matrix: np.ndarray, q0: int, q1: int
    ) -> None:
        """Apply 2-qubit gate on non-adjacent qubits via SWAP chain."""
        target_pos = q1 - 1 if q0 < q1 else q1 + 1
        current = q0

        if current < target_pos:
            for i in range(current, target_pos):
                self._apply_2q_adjacent(_SWAP_MATRIX, i, i + 1)
        else:
            for i in range(current, target_pos, -1):
                self._apply_2q_adjacent(_SWAP_MATRIX, i - 1, i, swapped=True)

        left, right = min(target_pos, q1), max(target_pos, q1)
        self._apply_2q_adjacent(
            matrix, left, right, swapped=(target_pos > q1)
        )

        if current < target_pos:
            for i in range(target_pos - 1, current - 1, -1):
                self._apply_2q_adjacent(_SWAP_MATRIX, i, i + 1)
        else:
            for i in range(target_pos + 1, current + 1):
                self._apply_2q_adjacent(_SWAP_MATRIX, i - 1, i, swapped=True)

    def _apply_nq(
        self,
        matrix: np.ndarray,
        qubits: tuple[int, ...],
        gate_name: str = "",
    ) -> None:
        """N-qubit gate execution (decomposes known 3Q/4Q or falls back to dense for n <= 20)."""
        if gate_name and self._decompose_known_multi_qubit_gate(gate_name, qubits, ()):
            return
        if self.num_qubits <= 20:
            self._fallback_dense(matrix, qubits)
        else:
            raise MPSSimulatorError(
                f"Cannot apply custom {len(qubits)}-qubit gate on {self.num_qubits}-qubit "
                f"MPS. Decompose gate into 1q and 2q gates first."
            )

    def _fallback_dense(
        self, matrix: np.ndarray, qubits: tuple[int, ...]
    ) -> None:
        """Fallback: convert to dense, apply gate, convert back."""
        sv = self._to_statevector()
        n_gate = len(qubits)
        gate_tensor = matrix.reshape([2] * (2 * n_gate))
        sv_tensor = sv.reshape([2] * self.num_qubits)
        gate_axes = list(range(n_gate, 2 * n_gate))
        state_axes = list(qubits)
        result = np.tensordot(gate_tensor, sv_tensor, axes=(gate_axes, state_axes))
        result = np.moveaxis(result, list(range(n_gate)), list(qubits))
        self._from_statevector(result.reshape(-1))

    # ── Open Quantum Systems: Stochastic Kraus Trajectories ──

    def apply_noise(
        self,
        noise_model: Any,
        qubits: tuple[int, ...],
        rng: np.random.Generator,
    ) -> None:
        """Applies stochastic CPTP Kraus trajectory jumps natively on the MPS."""
        channels = getattr(noise_model, "channels", [])
        for channel in channels:
            if isinstance(channel, ReadoutError):
                continue
            targets_list = resolve_channel_qubit_targets(
                channel, qubits, self.num_qubits
            )
            kraus_ops = channel.kraus_operators()
            for target in targets_list:
                if len(target) == 1:
                    self._apply_kraus_1q_trajectory(kraus_ops, target[0], rng)
                elif len(target) == 2:
                    self._apply_kraus_2q_trajectory(
                        kraus_ops, target[0], target[1], rng
                    )

    def _apply_kraus_1q_trajectory(
        self,
        kraus_ops: list[np.ndarray],
        qubit: int,
        rng: np.random.Generator,
    ) -> None:
        """Stochastic 1-qubit Kraus jump at orthogonality center `qubit`."""
        self._move_ortho_center(qubit)
        A = self._tensors[qubit]  # (χ_L, 2, χ_R), ||A||_F == 1

        candidates: list[np.ndarray] = []
        probs: list[float] = []
        for k_mat in kraus_ops:
            cand = np.einsum("ab,ibj->iaj", k_mat, A)
            p_k = float(np.real(np.vdot(cand, cand)))
            candidates.append(cand)
            probs.append(max(0.0, p_k))

        total_p = sum(probs)
        if total_p <= 1e-15:
            return
        p_arr = np.array(probs, dtype=np.float64) / total_p
        idx = int(rng.choice(len(kraus_ops), p=p_arr))
        chosen = candidates[idx]
        norm = math.sqrt(float(np.real(np.vdot(chosen, chosen))))
        if norm > 1e-15:
            chosen = chosen / norm
        self._tensors[qubit] = chosen
        self._ortho_center = qubit

    def _apply_kraus_2q_adjacent_trajectory(
        self,
        kraus_ops: list[np.ndarray],
        left: int,
        right: int,
        swapped: bool,
        rng: np.random.Generator,
    ) -> None:
        """Stochastic 2-qubit Kraus jump on adjacent sites `left, right = left + 1`."""
        self._move_ortho_center(left)
        A = self._tensors[left]
        B = self._tensors[right]
        chi_l = A.shape[0]
        chi_r = B.shape[2]

        theta = np.einsum("ijk,klm->ijlm", A, B)
        if swapped:
            theta = theta.transpose(0, 2, 1, 3)
        theta = theta.reshape(chi_l, 4, chi_r)

        candidates: list[np.ndarray] = []
        probs: list[float] = []
        for k_mat in kraus_ops:
            k4 = np.asarray(k_mat, dtype=complex).reshape(4, 4)
            cand = np.einsum("ij,kjl->kil", k4, theta)
            p_k = float(np.real(np.vdot(cand, cand)))
            candidates.append(cand)
            probs.append(max(0.0, p_k))

        total_p = sum(probs)
        if total_p <= 1e-15:
            return
        p_arr = np.array(probs, dtype=np.float64) / total_p
        idx = int(rng.choice(len(kraus_ops), p=p_arr))
        theta_new = candidates[idx]
        norm = math.sqrt(float(np.real(np.vdot(theta_new, theta_new))))
        if norm > 1e-15:
            theta_new = theta_new / norm

        if swapped:
            theta_new = theta_new.reshape(chi_l, 2, 2, chi_r)
            theta_new = theta_new.transpose(0, 2, 1, 3)
            theta_new = theta_new.reshape(chi_l, 4, chi_r)

        self._split_and_truncate_2q(theta_new, left, right, chi_l, chi_r)

    def _apply_kraus_2q_trajectory(
        self,
        kraus_ops: list[np.ndarray],
        q0: int,
        q1: int,
        rng: np.random.Generator,
    ) -> None:
        """Stochastic 2-qubit Kraus jump on arbitrary `(q0, q1)` via SWAP routing."""
        if abs(q1 - q0) == 1:
            self._apply_kraus_2q_adjacent_trajectory(
                kraus_ops, min(q0, q1), max(q0, q1), swapped=(q0 > q1), rng=rng
            )
            return

        target_pos = q1 - 1 if q0 < q1 else q1 + 1
        current = q0
        if current < target_pos:
            for i in range(current, target_pos):
                self._apply_2q_adjacent(_SWAP_MATRIX, i, i + 1)
        else:
            for i in range(current, target_pos, -1):
                self._apply_2q_adjacent(_SWAP_MATRIX, i - 1, i, swapped=True)

        left, right = min(target_pos, q1), max(target_pos, q1)
        self._apply_kraus_2q_adjacent_trajectory(
            kraus_ops, left, right, swapped=(target_pos > q1), rng=rng
        )

        if current < target_pos:
            for i in range(target_pos - 1, current - 1, -1):
                self._apply_2q_adjacent(_SWAP_MATRIX, i, i + 1)
        else:
            for i in range(target_pos + 1, current + 1):
                self._apply_2q_adjacent(_SWAP_MATRIX, i - 1, i, swapped=True)

    # ── Measurement ──

    def probabilities(self) -> np.ndarray:
        """Returns measurement probabilities (converts to dense)."""
        sv = self._to_statevector()
        return np.abs(sv) ** 2

    def sample(self, shots: int) -> dict[str, int]:
        """Exact sequential qubit-by-qubit Born sampling in O(shots · n · χ²)."""
        if shots <= 0:
            return {}
        if self.num_qubits > 1:
            # Right-canonicalize sites 1..n-1 so conditional marginals P(q_i | q_{<i}) are exact!
            self._move_ortho_center(0)

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

        Requires `self._ortho_center == 0` so sites `1..n-1` satisfy sum_s B_s B_s† = I.
        """
        n = self.num_qubits
        result = 0
        env = np.ones((1, 1), dtype=complex)

        for i in range(n):
            A = self._tensors[i]  # (χ_L, 2, χ_R)
            A0 = A[:, 0, :]
            A1 = A[:, 1, :]

            env0 = env @ A0
            env1 = env @ A1

            p0 = float(np.real(np.sum(env0 * np.conj(env0))))
            p1 = float(np.real(np.sum(env1 * np.conj(env1))))

            total = p0 + p1
            if total < 1e-15:
                break

            prob0 = p0 / total
            if self._rng.random() < prob0:
                env = env0 / math.sqrt(p0) if p0 > 1e-15 else env0
            else:
                result |= (1 << (n - 1 - i))  # MSB convention
                env = env1 / math.sqrt(p1) if p1 > 1e-15 else env1

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

    @classmethod
    def from_statevector(
        cls,
        statevector: np.ndarray,
        chi_max: int = 64,
        cutoff: float = 1e-12,
    ) -> MPSSimulator:
        """Creates an MPSSimulator initialized from a full statevector."""
        n = int(round(math.log2(len(statevector))))
        sim = cls(n, chi_max=chi_max, cutoff=cutoff)
        sim._from_statevector(statevector)
        return sim

    def norm(self) -> float:
        """Returns the L2 norm of the MPS state ||psi||."""
        if self.num_qubits <= 20:
            return float(np.linalg.norm(self.state))
        env = np.ones((1, 1), dtype=complex)
        for t in self._tensors:
            env = np.einsum("ab,asi,bsj->ij", env, t, t.conj(), optimize=True)
        return float(np.sqrt(np.real(env[0, 0])))

    @property
    def memory_bytes(self) -> int:
        """Estimated memory usage in bytes."""
        total = 0
        for t in self._tensors:
            total += t.nbytes
        return total

    def reset(self) -> None:
        """Resets MPS state while preserving chi_max, cutoff, and seed."""
        self.__init__(
            self.num_qubits,
            seed=self._seed,
            chi_max=self.chi_max,
            cutoff=self.cutoff,
        )  # type: ignore[misc]

    # ── Entanglement & Schmidt Analysis ──

    def schmidt_spectrum(
        self,
        cut: int | None = None,
        *,
        bipartition_cut: int | None = None,
    ) -> np.ndarray:
        """Computes the Schmidt spectrum (singular values) across a bipartition cut."""
        if cut is None:
            if bipartition_cut is not None:
                cut = bipartition_cut
            else:
                raise ValueError("Must specify cut or bipartition_cut")

        n = self.num_qubits
        if n < 2:
            raise MPSSimulatorError("Entanglement requires at least 2 qubits")

        if n == 2 and cut == 1:
            c = 0
        elif 0 <= cut < n - 1:
            c = cut
        else:
            raise ValueError(
                f"Invalid cut index {cut} for {n}-qubit system. "
                f"Valid range is 0 <= cut < {n - 1}."
            )

        self._move_ortho_center(c)
        chi_l, d, chi_r = self._tensors[c].shape
        mat = self._tensors[c].reshape(chi_l * d, chi_r)
        _U, S, _Vh = np.linalg.svd(mat, full_matrices=False)

        S = S[S > 1e-12]
        s_norm = float(np.linalg.norm(S))
        S = S / s_norm if s_norm > 1e-15 else np.array([1.0], dtype=float)
        return S.astype(float)

    def entanglement_entropy(
        self,
        cut: int | None = None,
        *,
        bipartition_cut: int | None = None,
    ) -> float:
        """Computes the von Neumann entanglement entropy across a bipartition cut."""
        S = self.schmidt_spectrum(cut=cut, bipartition_cut=bipartition_cut)
        probs = S ** 2
        probs = probs[probs > 1e-15]
        return float(-np.sum(probs * np.log(probs)))

    # ── Internal Conversion ──

    def _to_statevector(self) -> np.ndarray:
        """Contracts MPS to full statevector. O(2^n) memory!"""
        result = self._tensors[0]
        for i in range(1, self.num_qubits):
            result = np.einsum("...j,jkl->...kl", result, self._tensors[i])
        return result.reshape(-1)

    def _from_statevector(self, sv: np.ndarray) -> None:
        """Decomposes dense statevector into left-canonical MPS via sequential SVD."""
        n = self.num_qubits
        psi = np.asarray(sv, dtype=complex).reshape([2] * n)

        self._tensors = []
        self._total_trunc_error = 0.0

        remaining = psi
        chi_left = 1

        for _i in range(n - 1):
            mat = remaining.reshape(chi_left * 2, -1)
            u_mat, s_vals, vh_mat = np.linalg.svd(mat, full_matrices=False)

            above_cutoff = int(np.sum(s_vals > self.cutoff))
            chi_keep = max(1, above_cutoff)
            chi_new = min(len(s_vals), chi_keep, self.chi_max)

            if chi_new < len(s_vals):
                self._total_trunc_error += float(np.sum(s_vals[chi_new:] ** 2))
                u_mat = u_mat[:, :chi_new]
                s_vals = s_vals[:chi_new]
                s_norm = float(np.linalg.norm(s_vals))
                if s_norm > 1e-15:
                    s_vals = s_vals / s_norm
                vh_mat = vh_mat[:chi_new, :]
            else:
                u_mat = u_mat[:, :chi_new]
                s_vals = s_vals[:chi_new]
                vh_mat = vh_mat[:chi_new, :]

            self._tensors.append(u_mat.reshape(chi_left, 2, chi_new))
            remaining = s_vals[:, None] * vh_mat
            chi_left = chi_new

        self._tensors.append(remaining.reshape(chi_left, 2, 1))
        self._ortho_center = n - 1 if n > 0 else None

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
