"""
quanta.qml.lie_algebra -- Dynamical Lie Algebras (DLA) and Barren Plateau bounds.

Implements the algebraic framework for quantum controllability and trainability
analysis in Quantum Machine Learning (QML) and Variational Quantum Algorithms (VQA).

Note on Real vs Complex Vector Space:
Quantum unitary Lie algebras u(d) and su(d) are real Lie algebras (closed under real
linear combinations and Lie brackets [A, B] = AB - BA, NOT under multiplication by i,
since i * (i H) = -H is Hermitian rather than skew-Hermitian). Orthogonalization is
therefore performed strictly over R using the real inner product:
    <A, B>_R = Re Tr(A^dagger @ B)

Theory References:
- Ragone et al., "Unified Theory of Barren Plateaus in Quantum Neural Networks",
  Nature Communications 15, 7172 / arXiv:2309.09342 (2024).
- Fontana et al., "Characterizing barren plateaus in quantum ansatze via the adjoint
  representation", Nature Communications 15, 7171 / arXiv:2309.07902 (2024).
- Larocca et al., "Diagnosing Barren Plateaus with Tools from Quantum Optimal Control",
  Quantum 6, 824 (2022).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from quanta.core.types import QuantaError

__all__ = [
    "dynamical_lie_algebra",
    "barren_plateau_bound",
    "commutator",
    "hilbert_schmidt_inner_product",
    "lie_algebra_dimension",
    "is_in_algebra",
    "pauli_dla",
    "is_barren_plateau_immune",
    "LieAlgebraError",
]


class LieAlgebraError(QuantaError):
    """Exception raised for Lie algebraic errors."""


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Computes the Lie bracket (commutator): [A, B] = A @ B - B @ A."""
    return A @ B - B @ A


def hilbert_schmidt_inner_product(A: np.ndarray, B: np.ndarray) -> complex:
    """Computes the Hilbert-Schmidt inner product: <A, B>_HS = Tr(A^dagger @ B)."""
    return complex(np.trace(A.conj().T @ B))


def _matrix_to_real_vec(A: np.ndarray) -> np.ndarray:
    """Isometrically embeds A in M_d(C) into R^{2 d^2} under <A, B>_R = Re Tr(A^dagger B)."""
    flat = np.asarray(A, dtype=np.complex128).reshape(-1, order="F")
    return np.concatenate([flat.real, flat.imag])


def _real_vec_to_matrix(v: np.ndarray, d: int) -> np.ndarray:
    """Reconstructs a d x d complex matrix from its R^{2 d^2} real embedding."""
    d2 = d * d
    comp = v[:d2] + 1j * v[d2:]
    return comp.reshape((d, d), order="F")


def dynamical_lie_algebra(
    generators: Sequence[np.ndarray],
    tol: float = 1e-10,
) -> list[np.ndarray]:
    """Computes an R-orthonormal basis of the real Dynamical Lie Algebra g = <G_k>_Lie.

    Iteratively expands the algebra through Lie commutators [A, B] = AB - BA,
    orthonormalizing candidate basis elements over R under the real inner product
    <A, B>_R = Re Tr(A^dagger @ B) (so Hermitian H and skew-Hermitian iH are never
    conflated over C). For skew-Hermitian generators in su(d), Im Tr(E_i^dagger E_j) = 0
    identically, so the returned basis also satisfies Tr(E_i^dagger E_j) = delta_ij.

    Args:
        generators: Initial set of matrix generators (typically skew-Hermitian i H_k in su(2^n)).
        tol: Numerical tolerance for linear independence and singular values over R.

    Returns:
        List of R-orthonormal matrix basis elements E_k with Re Tr(E_i^dagger @ E_j) = delta_ij.
    """
    if not generators:
        return []

    d = generators[0].shape[0]
    for i, g in enumerate(generators):
        if g.ndim != 2 or g.shape[0] != d or g.shape[1] != d:
            raise LieAlgebraError(
                f"Generator {i} shape {g.shape} does not match expected ({d}, {d})"
            )

    max_real_dim = 2 * d * d
    col_vectors = [_matrix_to_real_vec(g) for g in generators]
    M = np.column_stack(col_vectors)

    # Initial real basis orthonormalization via real SVD
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    rank = int(np.sum(s > tol))
    if rank == 0:
        return []

    B_mat = U[:, :rank].copy()
    basis: list[np.ndarray] = [
        _real_vec_to_matrix(B_mat[:, i], d) for i in range(rank)
    ]

    new_indices = list(range(rank))

    # Iterative commutator expansion over R until closure
    while new_indices and B_mat.shape[1] < max_real_dim:
        candidates: list[np.ndarray] = []
        n_current = B_mat.shape[1]

        for i in new_indices:
            E_i = basis[i]
            for j in range(n_current):
                if i == j:
                    continue
                E_j = basis[j]
                C = commutator(E_i, E_j)
                v_C = _matrix_to_real_vec(C)
                if float(np.linalg.norm(v_C)) < tol:
                    continue

                # Orthogonal projection over R against current real-orthonormal basis
                proj = B_mat @ (B_mat.T @ v_C)
                v_orth = v_C - proj
                norm_orth = float(np.linalg.norm(v_orth))
                if norm_orth > tol:
                    candidates.append(v_orth / norm_orth)

        if not candidates:
            break

        C_mat = np.column_stack(candidates)
        # Second-pass Gram-Schmidt re-projection for machine precision
        C_mat = C_mat - B_mat @ (B_mat.T @ C_mat)
        U_new, s_new, _ = np.linalg.svd(C_mat, full_matrices=False)
        rank_new = int(np.sum(s_new > tol))

        if rank_new == 0:
            break

        available_dim = max_real_dim - B_mat.shape[1]
        rank_add = min(rank_new, available_dim)
        new_cols = U_new[:, :rank_add]
        # Re-orthogonalize new_cols against B_mat and QR-orthonormalize
        new_cols = new_cols - B_mat @ (B_mat.T @ new_cols)
        new_cols, _ = np.linalg.qr(new_cols)

        start_idx = len(basis)
        for k in range(rank_add):
            basis.append(_real_vec_to_matrix(new_cols[:, k], d))

        new_indices = list(range(start_idx, start_idx + rank_add))
        B_mat = np.column_stack([B_mat, new_cols])

    return basis


def barren_plateau_bound(dla_dim: int, n_qubits: int) -> float:
    """Computes the analytical upper bound on the gradient variance Var[∂_θ <O>].

    Based on the unified Dynamical Lie Algebra (DLA) theorem (Ragone et al., 2024):
        Var[∂_θ <O>] <= C / dim(g)
    For an n-qubit system with Hilbert space dimension d = 2^n:
    - When dim(g) ~ 4^n (e.g. full su(2^n) with dim = 4^n - 1), Var <= 1 / (4^n - 1),
      demonstrating an exponential barren plateau.
    - When dim(g) ~ poly(n), Var exhibits polynomial decay, avoiding barren plateaus.

    Args:
        dla_dim: Real dimension of the dynamical Lie algebra dim_R(g).
        n_qubits: Number of qubits n.

    Returns:
        float: Analytical upper bound on the gradient variance.
    """
    if dla_dim <= 0:
        raise ValueError(f"DLA dimension must be positive, got {dla_dim}")
    if n_qubits < 1:
        raise ValueError(f"Number of qubits must be >= 1, got {n_qubits}")

    return 1.0 / float(dla_dim)


def lie_algebra_dimension(
    generators: Sequence[np.ndarray], tol: float = 1e-10
) -> int:
    """Returns the real dimension dim_R(g) of the Dynamical Lie Algebra."""
    return len(dynamical_lie_algebra(generators, tol=tol))


def is_in_algebra(
    A: np.ndarray, basis: Sequence[np.ndarray], tol: float = 1e-10
) -> bool:
    """Checks whether matrix A belongs to the real Lie algebra spanned over R by basis."""
    v_A = _matrix_to_real_vec(A)
    if not basis:
        return float(np.linalg.norm(v_A)) < tol

    B_mat = np.column_stack([_matrix_to_real_vec(b) for b in basis])
    # Use least-squares / QR projection in case user supplied a non-orthonormal basis
    Q, _ = np.linalg.qr(B_mat)
    proj = Q @ (Q.T @ v_A)
    residual = v_A - proj
    return float(np.linalg.norm(residual)) < tol


def is_barren_plateau_immune(
    dla_dim: int, n_qubits: int, threshold_degree: float = 2.0
) -> bool:
    """Tests if a quantum circuit architecture is immune to barren plateaus.

    An architecture is immune if its dynamical Lie algebra dimension scales polynomially
    with the number of qubits (e.g. free fermions dim(so(2n)) = n(2n - 1) = 2n^2 - n)
    rather than exponentially dim(g) ~ 4^n.
    """
    if dla_dim <= 0:
        raise ValueError(f"DLA dimension must be positive, got {dla_dim}")
    if n_qubits < 1:
        raise ValueError(f"Number of qubits must be >= 1, got {n_qubits}")
    poly_limit = max(int(2 * (n_qubits ** threshold_degree)), 6)
    return dla_dim <= poly_limit


def pauli_dla(pauli_strings: Sequence[str], tol: float = 1e-10) -> list[np.ndarray]:
    """Constructs the real DLA in su(2^n) from Pauli string operators {i P_k}.

    Args:
        pauli_strings: List of Pauli strings, e.g. ["X", "Y", "Z"] or ["XX", "ZZ"].
        tol: Basis tolerance.

    Returns:
        R-orthonormal basis of the generated Lie algebra.
    """
    pauli_dict = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    generators: list[np.ndarray] = []
    for s in pauli_strings:
        mat = np.array([[1.0]], dtype=complex)
        for ch in s.upper():
            mat = np.kron(mat, pauli_dict[ch])
        # Skew-Hermitian generator i * P in u(2^n) / su(2^n)
        generators.append(1j * mat)

    return dynamical_lie_algebra(generators, tol=tol)
