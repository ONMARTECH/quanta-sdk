"""PyTest Test Suite: Empirical Verification of Milestone M1 Mathematical Theorems.

Target: docs/theory/continuous_quantum_neural_dynamics.md
"""

import numpy as np
import pytest
from scipy.linalg import eigh, expm
from scipy.special import jv


# Helper functions
def pauli_vector(vec):
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return vec[0] * sigma_x + vec[1] * sigma_y + vec[2] * sigma_z


def math_sinc(x):
    """Unnormalized sinc: sin(x)/x with sinc(0)=1."""
    if np.isscalar(x):
        if np.abs(x) < 1e-12:
            return 1.0 - x**2 / 6.0
        return np.sin(x) / x
    res = np.empty_like(x, dtype=np.float64)
    small = np.abs(x) < 1e-12
    res[small] = 1.0 - x[small] ** 2 / 6.0
    res[~small] = np.sin(x[~small]) / x[~small]
    return res


def daleckii_krein_matrix(lambdas, t):
    dim = len(lambdas)
    M = np.zeros((dim, dim), dtype=np.complex128)
    for a in range(dim):
        for b in range(dim):
            la, lb = lambdas[a], lambdas[b]
            l_bar = 0.5 * (la + lb)
            delta = la - lb
            M[a, b] = -1j * t * np.exp(-1j * l_bar * t) * math_sinc(0.5 * delta * t)
    return M


# ------------------------------------------------------------------------------
# 1. Tsirelson Bound & Bell-CHSH
# ------------------------------------------------------------------------------
def test_tsirelson_operator_identity():
    """Verify B^2 = 4 I + [A, A'] (x) [B, B'] and catch doc line 184 minus sign."""
    np.random.seed(42)
    for _ in range(10):
        a = np.random.randn(3)
        a /= np.linalg.norm(a)
        a_p = np.random.randn(3)
        a_p /= np.linalg.norm(a_p)
        b = np.random.randn(3)
        b /= np.linalg.norm(b)
        b_p = np.random.randn(3)
        b_p /= np.linalg.norm(b_p)

        A = pauli_vector(a)
        Ap = pauli_vector(a_p)
        B = pauli_vector(b)
        Bp = pauli_vector(b_p)

        B_op = np.kron(A, B - Bp) + np.kron(Ap, B + Bp)
        B_sq = B_op @ B_op

        comm_A = A @ Ap - Ap @ A
        comm_B = B @ Bp - Bp @ B
        true_B_sq = 4.0 * np.eye(4, dtype=np.complex128) + np.kron(comm_A, comm_B)
        assert np.allclose(B_sq, true_B_sq, atol=1e-12)


def test_tsirelson_bound_upper_limit():
    """Verify spectral norm of B_op never exceeds 2*sqrt(2)."""
    np.random.seed(123)
    two_sqrt_two = 2.0 * np.sqrt(2.0)
    for _ in range(100):
        a = np.random.randn(3)
        a /= np.linalg.norm(a)
        a_p = np.random.randn(3)
        a_p /= np.linalg.norm(a_p)
        b = np.random.randn(3)
        b /= np.linalg.norm(b)
        b_p = np.random.randn(3)
        b_p /= np.linalg.norm(b_p)

        A = pauli_vector(a)
        Ap = pauli_vector(a_p)
        B = pauli_vector(b)
        Bp = pauli_vector(b_p)
        B_op = np.kron(A, B - Bp) + np.kron(Ap, B + Bp)

        max_val = np.max(np.abs(np.linalg.eigvalsh(B_op)))
        assert max_val <= two_sqrt_two + 1e-12


def test_bell_chsh_optimal_violation_vectors():
    """Verify doc line 211 vector flaw and correct vector saturation."""
    psi_singlet = np.array([0, 1, -1, 0], dtype=np.complex128) / np.sqrt(2.0)
    a = np.array([0.0, 0.0, 1.0])
    ap = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, -1.0]) / np.sqrt(2.0)

    # Document's vector bp = 1/sqrt(2)*(1, 0, -1)
    bp_doc = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)

    def corr(u, v):
        op = np.kron(pauli_vector(u), pauli_vector(v))
        return np.real(np.vdot(psi_singlet, op @ psi_singlet))

    S_doc = corr(a, b) - corr(a, bp_doc) + corr(ap, b) + corr(ap, bp_doc)
    # Flaw: S_doc == 0.0
    assert np.isclose(S_doc, 0.0, atol=1e-10)

    # Corrected vector bp = 1/sqrt(2)*(-1, 0, 1)
    bp_correct = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)
    S_correct = corr(a, b) - corr(a, bp_correct) + corr(ap, b) + corr(ap, bp_correct)
    assert np.isclose(S_correct, 2.0 * np.sqrt(2.0), atol=1e-12)


# ------------------------------------------------------------------------------
# 2. CTQW 1D Bessel Wave Function & Ballistic Dispersion
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("J,t", [(1.0, 1.0), (1.5, 2.0), (0.5, 4.0)])
def test_ctqw_bessel_exactness(J, t):
    """Verify Bessel wave function matches matrix exp and dispersion is exactly 2 J^2 t^2."""
    N = 101
    center = 50
    H = np.zeros((N, N), dtype=np.complex128)
    for i in range(N - 1):
        H[i, i + 1] = -J
        H[i + 1, i] = -J

    psi0 = np.zeros(N, dtype=np.complex128)
    psi0[center] = 1.0
    psi_t = expm(-1j * H * t) @ psi0

    xs = np.arange(N) - center
    alpha = (1j) ** xs * jv(xs, 2.0 * J * t)

    assert np.max(np.abs(psi_t - alpha)) < 1e-12
    probs = np.abs(alpha) ** 2
    assert np.isclose(np.sum(probs), 1.0, atol=1e-10)
    assert np.isclose(np.sum(xs**2 * probs), 2.0 * J**2 * t**2, atol=1e-9)


# ------------------------------------------------------------------------------
# 3. Daleckii-Krein Spectral Formula vs Finite Difference
# ------------------------------------------------------------------------------
def test_daleckii_krein_dU_vs_finite_difference():
    """Verify Daleckii-Krein formula matches central finite difference of matrix exponential."""
    np.random.seed(42)
    dim = 4
    H = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    H = (H + H.conj().T) / 2.0
    Omega = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    Omega = (Omega + Omega.conj().T) / 2.0
    t = 1.8
    eps = 1e-7

    dU_fd = (expm(-1j * (H + eps * Omega) * t) - expm(-1j * (H - eps * Omega) * t)) / (2.0 * eps)
    lambdas, V = eigh(H)
    M = daleckii_krein_matrix(lambdas, t)
    dU_dk = V @ ((V.conj().T @ Omega @ V) * M) @ V.conj().T

    assert np.max(np.abs(dU_dk - dU_fd)) < 1e-6


def test_daleckii_krein_expectation_re_vs_im():
    """Catch Re vs Im bug in document lines 832, 834, 840."""
    np.random.seed(99)
    dim = 4
    H = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    H = (H + H.conj().T) / 2.0
    Omega = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    Omega = (Omega + Omega.conj().T) / 2.0
    obs = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    obs = (obs + obs.conj().T) / 2.0

    psi0 = np.random.randn(dim) + 1j * np.random.randn(dim)
    psi0 /= np.linalg.norm(psi0)
    t = 1.5
    eps = 1e-7

    def exp_val(phi):
        psi = expm(-1j * (H + phi * Omega) * t) @ psi0
        return np.real(np.vdot(psi, obs @ psi))

    d_fd = (exp_val(eps) - exp_val(-eps)) / (2.0 * eps)

    lambdas, V = eigh(H)
    M = daleckii_krein_matrix(lambdas, t)
    psi_t = expm(-1j * H * t) @ psi0
    inner_matrix = V @ ((V.conj().T @ Omega @ V) * M) @ V.conj().T
    overlap = np.vdot(psi_t, obs @ inner_matrix @ psi0)

    grad_re = 2.0 * np.real(overlap)
    grad_im = 2.0 * np.imag(overlap)

    # 2*Re matches finite difference
    assert np.isclose(grad_re, d_fd, atol=1e-5)
    # 2*Im fails finite difference
    assert not np.isclose(grad_im, d_fd, atol=1e-2)


# ------------------------------------------------------------------------------
# 4. Ehrenfest Time Derivative vs Finite Difference
# ------------------------------------------------------------------------------
def test_ehrenfest_time_derivative_sign():
    """Verify exact Ehrenfest time derivative and catch document sign error."""
    np.random.seed(77)
    dim = 4
    H = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    H = (H + H.conj().T) / 2.0
    obs = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    obs = (obs + obs.conj().T) / 2.0

    psi0 = np.random.randn(dim) + 1j * np.random.randn(dim)
    psi0 /= np.linalg.norm(psi0)
    t = 1.2
    eps = 1e-7

    def exp_t(tau):
        psi = expm(-1j * H * tau) @ psi0
        return np.real(np.vdot(psi, obs @ psi))

    d_fd = (exp_t(t + eps) - exp_t(t - eps)) / (2.0 * eps)
    psi_t = expm(-1j * H * t) @ psi0

    # True derivative: +2 * Im[ < O H > ] == i * < [H, O] >
    d_correct = +2.0 * np.imag(np.vdot(psi_t, obs @ H @ psi_t))
    # Document formula: -2 * Im[ < O H > ]
    d_doc = -2.0 * np.imag(np.vdot(psi_t, obs @ H @ psi_t))

    assert np.isclose(d_correct, d_fd, atol=1e-5)
    # Doc formula is strictly negative of true derivative
    assert np.isclose(d_doc, -d_fd, atol=1e-5)
