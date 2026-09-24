"""
quanta.simulator.lindblad -- Lindblad master equation solver for open quantum systems.

Implements the Lindblad (Gorini-Kossakowski-Sudarshan-Lindblad, GKSL) master equation:
    d rho / dt = -i [H, rho] + sum_k ( L_k rho L_k^dagger - 1/2 {L_k^dagger L_k, rho} )

Uses Liouvillian superoperator matrix vectorization:
    | d rho / dt >> = L | rho >>
where L is given by:
    L = -i (I (x) H - H^T (x) I)
        + sum_k ( conj(L_k) (x) L_k - 1/2 I (x) L_k^dagger L_k - 1/2 L_k^T conj(L_k) (x) I )
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.linalg import expm

from quanta.core.types import QuantaError

__all__ = [
    "LindbladMasterEquation",
    "vectorize",
    "unvectorize",
    "build_liouvillian",
    "LindbladError",
]


class LindbladError(QuantaError):
    """Exception raised for errors in Lindblad master equation evolution."""


def vectorize(rho: np.ndarray) -> np.ndarray:
    """Vectorizes a density matrix using column-stacking convention: |rho>> = vec(rho).

    Args:
        rho: Density matrix of shape (d, d).

    Returns:
        1D column-stacked complex vector of length d^2.
    """
    return rho.reshape(-1, order="F")


def unvectorize(v: np.ndarray, dim: int) -> np.ndarray:
    """Reconstructs density matrix from column-stacked vector.

    Args:
        v: Complex vector of length dim^2.
        dim: Hilbert space dimension.

    Returns:
        Matrix of shape (dim, dim).
    """
    return v.reshape((dim, dim), order="F")


def build_liouvillian(
    H: np.ndarray, jump_ops: Sequence[np.ndarray] | None = None
) -> np.ndarray:
    """Constructs the Liouvillian superoperator matrix in column-stacking convention.

    Formula:
        L = -i (I (x) H - H^T (x) I)
            + sum_k ( conj(L_k) (x) L_k - 1/2 I (x) L_k^dagger L_k - 1/2 L_k^T conj(L_k) (x) I )

    Args:
        H: Hamiltonian matrix of shape (d, d).
        jump_ops: Optional sequence of jump operators L_k of shape (d, d).

    Returns:
        Liouvillian superoperator matrix of shape (d^2, d^2).
    """
    d = H.shape[0]
    eye = np.eye(d, dtype=complex)

    # Unitary commutator component: -i [H, rho] -> -i (I (x) H - H^T (x) I)
    L = -1j * (np.kron(eye, H) - np.kron(H.T, eye))

    if jump_ops:
        for L_k in jump_ops:
            L_dag_L = L_k.conj().T @ L_k
            L_T_L_bar = L_k.T @ np.conj(L_k)  # Equals (L_k^dagger @ L_k)^T
            # Dissipator superoperator:
            # conj(L_k) (x) L_k - 1/2 I (x) L_k^dagger L_k - 1/2 L_k^T conj(L_k) (x) I
            term_sandwich = np.kron(np.conj(L_k), L_k)
            term_left = 0.5 * np.kron(eye, L_dag_L)
            term_right = 0.5 * np.kron(L_T_L_bar, eye)
            L += term_sandwich - term_left - term_right

    return L


class LindbladMasterEquation:
    """Lindblad master equation solver for open quantum systems.

    Simulates Markovian open quantum system dynamics with exact Liouvillian
    matrix exponentials or Runge-Kutta 4th order (RK4) integration, strictly
    preserving complete positivity and trace preservation (CPTP).

    Args:
        H: System Hamiltonian matrix of shape (d, d).
        jump_ops: Sequence of Lindblad jump operators L_k of shape (d, d).
    """

    def __init__(
        self,
        H: np.ndarray,
        jump_ops: Sequence[np.ndarray] | None = None,
    ) -> None:
        H_mat = np.asarray(H, dtype=complex)
        if H_mat.ndim != 2 or H_mat.shape[0] != H_mat.shape[1]:
            raise LindbladError(f"Hamiltonian must be a square matrix, got shape {H_mat.shape}")

        self.dim = H_mat.shape[0]
        # Enforce Hermiticity of H
        self.H = (H_mat + H_mat.conj().T) / 2.0

        self.jump_ops: list[np.ndarray] = []
        if jump_ops:
            for i, op in enumerate(jump_ops):
                op_mat = np.asarray(op, dtype=complex)
                if op_mat.shape != (self.dim, self.dim):
                    raise LindbladError(
                        f"Jump operator {i} shape {op_mat.shape} does not match "
                        f"Hamiltonian dimension ({self.dim}, {self.dim})"
                    )
                self.jump_ops.append(op_mat)

        self._liouvillian = build_liouvillian(self.H, self.jump_ops)

    @property
    def liouvillian(self) -> np.ndarray:
        """The (d^2, d^2) Liouvillian superoperator matrix."""
        return self._liouvillian.copy()

    def rhs_matrix(self, rho: np.ndarray) -> np.ndarray:
        """Evaluates d rho / dt directly in matrix form.

        d rho / dt = -i [H, rho] + sum_k ( L_k rho L_k^dagger - 1/2 {L_k^dagger L_k, rho} )
        """
        # Commutator
        drho = -1j * (self.H @ rho - rho @ self.H)

        for L_k in self.jump_ops:
            L_dag_L = L_k.conj().T @ L_k
            sandwich = L_k @ rho @ L_k.conj().T
            anticomm = 0.5 * (L_dag_L @ rho + rho @ L_dag_L)
            drho += sandwich - anticomm

        return drho

    def evolve(
        self,
        rho_0: np.ndarray,
        t_span: tuple[float, float],
        steps: int = 100,
        method: str = "expm",
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Evolves initial density matrix rho_0 through the Lindblad master equation.

        Args:
            rho_0: Initial density matrix of shape (d, d).
            t_span: Tuple of (t_start, t_end).
            steps: Number of evolution steps.
            method: Integration method, "expm" (matrix exponential propagator) or "rk4".

        Returns:
            times: Array of time points of length (steps + 1).
            states: List of density matrices at each time point.
        """
        if steps < 1:
            raise LindbladError(f"Number of steps must be >= 1, got {steps}")

        rho = np.asarray(rho_0, dtype=complex)
        if rho.shape != (self.dim, self.dim):
            raise LindbladError(
                f"Initial state shape {rho.shape} does not match dimension {self.dim}"
            )

        # Validate and clean initial state
        rho = self._clean_and_verify_cptp(rho)

        t0, t1 = t_span
        total_time = t1 - t0
        dt = total_time / steps
        times = np.linspace(t0, t1, steps + 1)
        states: list[np.ndarray] = [rho]

        if method == "expm":
            # Propagator for single time step dt: P = exp(L * dt)
            propagator = expm(self._liouvillian * dt)
            rho_vec = vectorize(rho)

            for _ in range(steps):
                rho_vec = propagator @ rho_vec
                rho = unvectorize(rho_vec, self.dim)
                rho = self._clean_and_verify_cptp(rho)
                rho_vec = vectorize(rho)
                states.append(rho)

        elif method == "rk4":
            for _ in range(steps):
                k1 = self.rhs_matrix(rho)
                k2 = self.rhs_matrix(rho + 0.5 * dt * k1)
                k3 = self.rhs_matrix(rho + 0.5 * dt * k2)
                k4 = self.rhs_matrix(rho + dt * k3)
                rho = rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                rho = self._clean_and_verify_cptp(rho)
                states.append(rho)

        else:
            raise LindbladError(f"Unsupported evolution method '{method}'. Use 'expm' or 'rk4'.")

        return times, states

    def _clean_and_verify_cptp(self, rho: np.ndarray) -> np.ndarray:
        """Enforces Hermiticity, checks trace conservation and complete positivity."""
        # 1. Hermiticity
        rho_herm = (rho + rho.conj().T) / 2.0

        # 2. Trace preservation check
        tr = np.trace(rho_herm)
        tr_val = float(np.real(tr))
        if abs(tr_val - 1.0) >= 1e-10:
            raise LindbladError(
                f"Trace conservation violated during Lindblad evolution: Tr(rho) = {tr_val:.4e} "
                f"(deviation {abs(tr_val - 1.0):.2e} >= 1e-10)"
            )

        # 3. Complete positivity check: eigenvalues >= -1e-10
        evals, evecs = np.linalg.eigh(rho_herm)
        min_eval = float(np.min(evals))
        if min_eval < -1e-10:
            raise LindbladError(
                f"Complete positivity violated during Lindblad evolution: "
                f"min eigenvalue = {min_eval:.2e} < -1e-10"
            )

        # Clip numerical negative eigenvalues and re-normalize trace
        clipped_evals = np.maximum(evals, 0.0)
        s = float(np.sum(clipped_evals))
        if s > 0:
            clipped_evals /= s
        return evecs @ np.diag(clipped_evals) @ evecs.conj().T

    def steady_state(self, tol: float = 1e-10) -> np.ndarray:
        """Computes the stationary state rho_ss satisfying L |rho_ss>> = 0 and Tr(rho_ss) = 1.

        Solves the null space of the Liouvillian with the trace normalization constraint.

        Returns:
            Stationary state density matrix of shape (d, d).
        """
        d = self.dim
        # Augmented system: add trace condition sum_i rho_ii = 1
        # Tr(rho) in column stacking is vec(I)^H @ vec(rho)
        trace_row = np.conj(vectorize(np.eye(d, dtype=complex)))

        A = np.vstack([self._liouvillian, trace_row])
        b = np.zeros(d * d + 1, dtype=complex)
        b[-1] = 1.0

        # Solve least squares
        vec_ss, residuals, rank, s = np.linalg.lstsq(A, b, rcond=tol)
        rho_ss = unvectorize(vec_ss, d)
        return self._clean_and_verify_cptp(rho_ss)

    @staticmethod
    def expectation_value(observable: np.ndarray, rho: np.ndarray) -> float:
        """Computes <O> = Tr(O @ rho)."""
        return float(np.real(np.trace(observable @ rho)))
