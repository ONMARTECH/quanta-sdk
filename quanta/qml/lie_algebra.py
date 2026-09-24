"""
quanta.qml.lie_algebra -- Dynamical Lie Algebras (DLA) and Barren Plateau bounds.

Implements the algebraic framework for quantum controllability and trainability
analysis in Quantum Machine Learning (QML) and Variational Quantum Algorithms (VQA).

Theory References:
- Ragone et al., "Unified Theory of Barren Plateaus in Quantum Neural Networks",
  Nature Communications / arXiv:2309.09342 (2023).
- Fontana et al., "The Adjoint is All You Need: Characterizing Barren Plateaus in QML",
  arXiv:2309.07902 (2023).
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


def dynamical_lie_algebra(
    generators: Sequence[np.ndarray],
    tol: float = 1e-10,
) -> list[np.ndarray]:
    """Computes the orthonormal basis of the Dynamical Lie Algebra (DLA) g = <i H_k>_Lie.

    Iteratively expands the algebra through Lie commutators [A, B] = AB - BA,
    orthonormalizing candidate basis elements with SVD / Gram-Schmidt until algebraic closure.

    Args:
        generators: Initial set of matrix generators (Hermitian or skew-Hermitian).
        tol: Numerical tolerance for linear independence and singular values.

    Returns:
        List of orthonormal matrix basis elements E_k with Tr(E_i^dagger @ E_j) = delta_ij.
    """
    if not generators:
        return []

    # Validate shapes
    d = generators[0].shape[0]
    for i, g in enumerate(generators):
        if g.ndim != 2 or g.shape[0] != d or g.shape[1] != d:
            raise LieAlgebraError(
                f"Generator {i} shape {g.shape} does not match expected ({d}, {d})"
            )

    d2 = d * d
    # Flatten generators to column vectors: length d^2
    col_vectors = [g.reshape(-1, order="F").astype(complex) for g in generators]
    M = np.column_stack(col_vectors)

    # Initial basis orthonormalization via SVD
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    rank = int(np.sum(s > tol))
    if rank == 0:
        return []

    # Basis matrix of shape (d^2, rank)
    B_mat = U[:, :rank].copy()
    basis: list[np.ndarray] = [
        B_mat[:, i].reshape((d, d), order="F") for i in range(rank)
    ]

    new_indices = list(range(rank))

    # Iterative commutator expansion until closure or reaching max dimension d^2
    while new_indices and B_mat.shape[1] < d2:
        candidates: list[np.ndarray] = []
        n_current = B_mat.shape[1]

        for i in new_indices:
            E_i = basis[i]
            for j in range(n_current):
                E_j = basis[j]
                # Compute Lie bracket [E_i, E_j]
                C = commutator(E_i, E_j)
                norm_C = float(np.linalg.norm(C))
                if norm_C < tol:
                    continue

                v_C = C.reshape(-1, order="F")
                # Orthogonal projection against current basis
                proj = B_mat @ (B_mat.conj().T @ v_C)
                v_orth = v_C - proj
                if float(np.linalg.norm(v_orth)) > tol:
                    candidates.append(v_orth)

        if not candidates:
            # Algebra closed!
            break

        # SVD on candidate orthogonal directions
        C_mat = np.column_stack(candidates)
        # Re-project for numerical precision
        C_mat = C_mat - B_mat @ (B_mat.conj().T @ C_mat)
        U_new, s_new, _ = np.linalg.svd(C_mat, full_matrices=False)
        rank_new = int(np.sum(s_new > tol))

        if rank_new == 0:
            break

        # Limit to remaining dimension
        available_dim = d2 - B_mat.shape[1]
        rank_add = min(rank_new, available_dim)
        new_cols = U_new[:, :rank_add]

        start_idx = len(basis)
        for k in range(rank_add):
            basis.append(new_cols[:, k].reshape((d, d), order="F"))

        new_indices = list(range(start_idx, start_idx + rank_add))
        B_mat = np.column_stack([B_mat, new_cols])

    return basis


def barren_plateau_bound(dla_dim: int, n_qubits: int) -> float:
    """Computes the analytical upper bound on the gradient variance Var[∂_θ <O>].

    Based on the unified Dynamical Lie Algebra (DLA) theorem (Ragone et al., 2023):
        Var[∂_θ <O>] <= C / dim(g)
    For an n-qubit system with Hilbert space dimension d = 2^n:
    - When dim(g) ~ 4^n (e.g. full su(2^n) with dim = 4^n - 1), Var <= 1 / (4^n - 1),
      demonstrating an exponential barren plateau.
    - When dim(g) ~ poly(n), Var exhibits polynomial decay, avoiding barren plateaus.

    Args:
        dla_dim: Dimension of the dynamical Lie algebra dim(g).
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
    """Returns the dimension of the Dynamical Lie Algebra generated by generators."""
    return len(dynamical_lie_algebra(generators, tol=tol))


def is_in_algebra(
    A: np.ndarray, basis: Sequence[np.ndarray], tol: float = 1e-10
) -> bool:
    """Checks whether matrix A belongs to the Lie algebra spanned by basis."""
    if not basis:
        return float(np.linalg.norm(A)) < tol

    B_mat = np.column_stack([b.reshape(-1, order="F") for b in basis])
    v_A = A.reshape(-1, order="F")

    proj = B_mat @ (B_mat.conj().T @ v_A)
    residual = v_A - proj
    return float(np.linalg.norm(residual)) < tol


def is_barren_plateau_immune(
    dla_dim: int, n_qubits: int, threshold_degree: float = 2.0
) -> bool:
    """Tests if a quantum circuit architecture is immune to barren plateaus.

    An architecture is immune if its dynamical Lie algebra dimension scales polynomially
    with the number of qubits (e.g. free fermions dim(g) = 2n^2 - n) rather than
    exponentially dim(g) ~ 4^n - 1.
    """
    max_dim = (4 ** n_qubits) - 1
    # Polynomial bound accommodating quadratic / cubic free-fermionic algebras (e.g. 2n^2 - n)
    poly_limit = max(int(2 * (n_qubits ** threshold_degree)), 6)
    return dla_dim <= poly_limit or dla_dim < (max_dim // 2)


def pauli_dla(pauli_strings: Sequence[str], tol: float = 1e-10) -> list[np.ndarray]:
    """Constructs the DLA from a list of Pauli string operators {i P_k}.

    Args:
        pauli_strings: List of Pauli strings, e.g. ["X", "Y", "Z"] or ["XX", "ZZ"].
        tol: Basis tolerance.

    Returns:
        Orthonormal basis of the generated Lie algebra.
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
        # Skew-Hermitian generator i * P
        generators.append(1j * mat)

    return dynamical_lie_algebra(generators, tol=tol)
