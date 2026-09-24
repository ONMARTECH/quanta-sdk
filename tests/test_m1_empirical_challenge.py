"""tests/test_m1_empirical_challenge.py -- Empirical verification suite for Milestone M1.

Authored by Challenger 1 (EMPIRICAL CHALLENGER):
1. Exact Hamiltonian evolution vs Scipy expm(-1j * H * t) on TFIM and Heisenberg models
   across 2, 3, and 4 qubits. Confirm ||U_spectral - U_scipy||_inf < 1e-12.
2. Suzuki-Trotter 2nd and 4th order convergence slopes on log-log plot (slopes 2 and 4).
3. Machine-precision unitarity on 100 randomly generated Hermitian Hamiltonians:
   confirm ||U^dagger U - I||_inf < 1e-14.
4. Lindblad master equation spontaneous emission analytical comparison:
   rho_11(t) = exp(-gamma * t), confirm error < 1e-10.
5. Adversarial edge cases and stress boundaries.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from quanta.core.custom_gate import CustomGateError, custom_gate
from quanta.core.equivalence import unitaries_equivalent
from quanta.layer3.hamiltonian import (
    _matrix_exp,
    spectral_unitary_evolution,
    suzuki_trotter_step,
)
from quanta.simulator.lindblad import (
    LindbladError,
    LindbladMasterEquation,
)

# ==============================================================================
# Helpers
# ==============================================================================


def build_tfim_hamiltonian(n_qubits: int, J: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Builds Transverse Field Ising Model: H = - sum_i J_i Z_i Z_{i+1} - sum_i h_i X_i."""
    dim = 2**n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    eye2 = np.eye(2, dtype=complex)

    # Coupling terms
    for i in range(n_qubits - 1):
        term = 1.0
        for j in range(n_qubits):
            op = Z if j in (i, i + 1) else eye2
            term = np.kron(term, op) if isinstance(term, np.ndarray) else op
        H -= J[i] * term

    # Transverse field terms
    for i in range(n_qubits):
        term = 1.0
        for j in range(n_qubits):
            op = X if j == i else eye2
            term = np.kron(term, op) if isinstance(term, np.ndarray) else op
        H -= h[i] * term

    return H


def build_heisenberg_hamiltonian(
    n_qubits: int, Jx: float, Jy: float, Jz: float
) -> np.ndarray:
    """Builds 1D Heisenberg XYZ Hamiltonian."""
    dim = 2**n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    eye2 = np.eye(2, dtype=complex)

    for i in range(n_qubits - 1):
        for J_val, pauli in [(Jx, X), (Jy, Y), (Jz, Z)]:
            term = 1.0
            for j in range(n_qubits):
                op = pauli if j in (i, i + 1) else eye2
                term = np.kron(term, op) if isinstance(term, np.ndarray) else op
            H += J_val * term
    return H


# ==============================================================================
# Challenge 1: Exact Spectral Evolution vs Scipy expm
# ==============================================================================


class TestChallenge1ExactEvolutionVsScipy:
    """Challenge 1: Verify ||U_spectral - U_scipy||_inf < 1e-12 on non-commuting Hamiltonians."""

    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_tfim_exact_evolution_vs_scipy(self, n_qubits: int) -> None:
        """TFIM non-commuting Hamiltonian evolution matches scipy.linalg.expm within 1e-12."""
        np.random.seed(42 + n_qubits)
        times = [0.001, 0.05, 0.2, 1.0, 2.5, 5.0]

        for _trial in range(5):
            J = np.random.uniform(0.2, 2.0, size=n_qubits - 1)
            h = np.random.uniform(0.2, 2.0, size=n_qubits)
            H = build_tfim_hamiltonian(n_qubits, J, h)

            for t in times:
                U_spectral = spectral_unitary_evolution(H, t)
                U_scipy = expm(-1j * H * t)

                diff_inf = np.max(np.abs(U_spectral - U_scipy))
                assert (
                    diff_inf < 1e-12
                ), f"TFIM n={n_qubits}, t={t}: diff_inf={diff_inf:.2e} >= 1e-12"

                # Also verify _matrix_exp
                U_mat_exp = _matrix_exp(-1j * H * t)
                diff_mat_exp = np.max(np.abs(U_mat_exp - U_scipy))
                assert (
                    diff_mat_exp < 1e-12
                ), f"_matrix_exp diff={diff_mat_exp:.2e} >= 1e-12"

    def test_heisenberg_xyz_evolution_vs_scipy(self) -> None:
        """Heisenberg XYZ 3-qubit model matches scipy expm within 1e-12."""
        H = build_heisenberg_hamiltonian(3, Jx=1.2, Jy=0.8, Jz=-0.5)
        for t in [0.01, 0.1, 0.5, 1.5, 3.0]:
            U_spec = spectral_unitary_evolution(H, t)
            U_ref = expm(-1j * H * t)
            diff = np.max(np.abs(U_spec - U_ref))
            assert diff < 1e-12, f"Heisenberg XYZ diff {diff:.2e} >= 1e-12 at t={t}"


# ==============================================================================
# Challenge 2: Suzuki-Trotter 2nd and 4th Order Convergence Slopes
# ==============================================================================


class TestChallenge2SuzukiTrotterConvergence:
    """Challenge 2: Verify Trotter 2nd and 4th order empirical convergence slopes (2 and 4)."""

    def test_suzuki_trotter_global_error_slopes_log_log(self) -> None:
        """Measures global error at fixed T=1.0 and verifies slope 2 and slope 4 on log-log plot."""
        # Non-commuting terms: X and Z
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        terms = [X, Z]
        H_total = X + Z

        T = 1.0
        U_exact = spectral_unitary_evolution(H_total, T)

        # Step sizes
        dts = [0.2, 0.1, 0.05, 0.025, 0.0125]
        errs_2nd: list[float] = []
        errs_4th: list[float] = []

        for dt in dts:
            N = int(round(T / dt))
            # 2nd order step repeated N times
            U2_step = suzuki_trotter_step(terms, dt, order=2)
            U2_total = np.linalg.matrix_power(U2_step, N)
            errs_2nd.append(float(np.linalg.norm(U2_total - U_exact, 2)))

            # 4th order step repeated N times
            U4_step = suzuki_trotter_step(terms, dt, order=4)
            U4_total = np.linalg.matrix_power(U4_step, N)
            errs_4th.append(float(np.linalg.norm(U4_total - U_exact, 2)))

        log_dt = np.log(dts)
        slope_2nd, _ = np.polyfit(log_dt, np.log(errs_2nd), 1)
        slope_4th, _ = np.polyfit(log_dt, np.log(errs_4th), 1)

        # Assert empirical slopes match theoretical expectations:
        # Order 2 slope should be within [1.95, 2.05]
        assert (
            1.95 <= slope_2nd <= 2.05
        ), f"Order 2 convergence slope {slope_2nd:.4f} is not close to 2.0"

        # Order 4 slope should be within [3.90, 4.10]
        assert (
            3.90 <= slope_4th <= 4.10
        ), f"Order 4 convergence slope {slope_4th:.4f} is not close to 4.0"

    def test_suzuki_trotter_multi_qubit_slopes(self) -> None:
        """Verifies Trotter order 2 and 4 slopes on a 2-qubit interacting Hamiltonian."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        eye2 = np.eye(2, dtype=complex)

        ZZ = np.kron(Z, Z)
        XI = np.kron(X, eye2)
        IX = np.kron(eye2, X)
        terms = [ZZ, XI + IX]
        H_total = ZZ + XI + IX

        T = 0.5
        U_exact = spectral_unitary_evolution(H_total, T)
        dts = [0.1, 0.05, 0.025, 0.0125]
        errs_2 = []
        errs_4 = []

        for dt in dts:
            N = int(round(T / dt))
            U2 = np.linalg.matrix_power(suzuki_trotter_step(terms, dt, order=2), N)
            U4 = np.linalg.matrix_power(suzuki_trotter_step(terms, dt, order=4), N)
            errs_2.append(np.linalg.norm(U2 - U_exact, 2))
            errs_4.append(np.linalg.norm(U4 - U_exact, 2))

        log_dt = np.log(dts)
        slope_2, _ = np.polyfit(log_dt, np.log(errs_2), 1)
        slope_4, _ = np.polyfit(log_dt, np.log(errs_4), 1)

        assert 1.95 <= slope_2 <= 2.05, f"2-qubit Trotter order 2 slope: {slope_2:.4f}"
        assert 3.90 <= slope_4 <= 4.10, f"2-qubit Trotter order 4 slope: {slope_4:.4f}"


# ==============================================================================
# Challenge 3: Machine-Precision Unitarity on 100 Random Hamiltonians
# ==============================================================================


class TestChallenge3MachinePrecisionUnitarity:
    """Challenge 3: Test machine-precision unitarity on 100 random Hermitian Hamiltonians."""

    def test_unitarity_100_random_hermitian_hamiltonians(self) -> None:
        """Confirm ||U^dagger U - I||_inf < 1e-14 across 100 random Hermitian Hamiltonians."""
        np.random.seed(2026_09_24)
        max_error_observed = 0.0

        for trial in range(100):
            # Vary dimensions between 2 and 16
            dim = np.random.choice([2, 3, 4, 5, 6, 8, 12, 16])
            # Random complex matrix
            A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
            # Hermitian part
            H = (A + A.conj().T) / 2.0
            t = float(np.random.uniform(0.01, 10.0))

            U = spectral_unitary_evolution(H, t)
            eye = np.eye(dim, dtype=complex)

            # Two-sided unitarity verification
            right_diff = float(np.max(np.abs(U @ U.conj().T - eye)))
            left_diff = float(np.max(np.abs(U.conj().T @ U - eye)))
            err = max(right_diff, left_diff)

            if err > max_error_observed:
                max_error_observed = err

            assert (
                err < 1e-14
            ), f"Unitarity violated at trial {trial} (dim={dim}, t={t}): err={err:.2e} >= 1e-14"

        # Assert max error is strictly below 1e-14
        assert max_error_observed < 1e-14, f"Max error {max_error_observed:.2e} >= 1e-14"

    def test_degenerate_spectrum_unitarity(self) -> None:
        """Spectral evolution on highly degenerate Hamiltonians preserves unitarity < 1e-14."""
        # 8-dimensional identity scaled Hamiltonian (completely degenerate)
        H_deg = 3.5 * np.eye(8, dtype=complex)
        U = spectral_unitary_evolution(H_deg, dt=1.7)
        diff = np.max(np.abs(U.conj().T @ U - np.eye(8)))
        assert diff < 1e-14, f"Degenerate Hamiltonian unitarity diff: {diff:.2e}"


# ==============================================================================
# Challenge 4: Lindblad Master Equation vs Analytical Decay
# ==============================================================================


class TestChallenge4LindbladSpontaneousEmission:
    """Challenge 4: Test Lindblad master equation on 2-level atom with spontaneous emission."""

    def test_spontaneous_emission_decay_expm(self) -> None:
        """Compare numerical rho(t) with analytical decay rho_11(t) = exp(-gamma*t) (< 1e-10)."""
        gamma = 0.75
        H = np.zeros((2, 2), dtype=complex)
        # Spontaneous emission jump operator: L = sqrt(gamma) * |0><1|
        sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
        L = np.sqrt(gamma) * sigma_minus

        solver = LindbladMasterEquation(H, [L])
        # Initial excited state |1><1|
        rho_0 = np.array([[0, 0], [0, 1]], dtype=complex)

        times, states = solver.evolve(rho_0, t_span=(0.0, 5.0), steps=100, method="expm")

        max_rho11_err = 0.0
        for t, rho_t in zip(times, states, strict=True):
            analytical_rho11 = np.exp(-gamma * t)
            err = abs(float(rho_t[1, 1].real) - analytical_rho11)
            if err > max_rho11_err:
                max_rho11_err = err

            # Ground state population must be 1 - exp(-gamma * t)
            analytical_rho00 = 1.0 - np.exp(-gamma * t)
            assert abs(float(rho_t[0, 0].real) - analytical_rho00) < 1e-10

            # Coherences must be identically 0
            assert abs(rho_t[0, 1]) < 1e-12
            assert abs(rho_t[1, 0]) < 1e-12

            # Trace preservation
            assert abs(np.trace(rho_t) - 1.0) < 1e-12

        assert max_rho11_err < 1e-10, f"Max rho_11 error {max_rho11_err:.2e} >= 1e-10"

    def test_spontaneous_emission_decay_rk4(self) -> None:
        """Compare RK4 solver against analytical decay rho_11(t) = exp(-gamma*t) (< 1e-10)."""
        gamma = 0.5
        H = np.zeros((2, 2), dtype=complex)
        sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
        L = np.sqrt(gamma) * sigma_minus

        solver = LindbladMasterEquation(H, [L])
        rho_0 = np.array([[0, 0], [0, 1]], dtype=complex)

        # RK4 with 200 steps
        times, states = solver.evolve(rho_0, t_span=(0.0, 4.0), steps=200, method="rk4")

        max_err = 0.0
        for t, rho_t in zip(times, states, strict=True):
            err = abs(float(rho_t[1, 1].real) - np.exp(-gamma * t))
            if err > max_err:
                max_err = err

        assert max_err < 1e-10, f"RK4 max error {max_err:.2e} >= 1e-10"

    def test_spontaneous_emission_steady_state(self) -> None:
        """Steady state under spontaneous emission must be pure ground state |0><0|."""
        gamma = 1.2
        H = np.zeros((2, 2), dtype=complex)
        sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
        solver = LindbladMasterEquation(H, [np.sqrt(gamma) * sigma_minus])

        rho_ss = solver.steady_state()
        expected = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        diff = np.max(np.abs(rho_ss - expected))
        assert diff < 1e-10, f"Steady state mismatch: {diff:.2e} >= 1e-10"


# ==============================================================================
# Challenge 5: Adversarial Boundary & Negative Testing
# ==============================================================================


class TestAdversarialStressAndBoundaries:
    """Stress-tests edge cases, invalid inputs, and physical conservation boundaries."""

    def test_trotter_invalid_order_raises(self) -> None:
        """Trotter step with unsupported order raises ValueError."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        with pytest.raises(ValueError, match="Unsupported Trotter order"):
            suzuki_trotter_step([X, Z], dt=0.1, order=3)

    def test_trotter_empty_terms_raises(self) -> None:
        """Empty term matrices list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            suzuki_trotter_step([], dt=0.1)

    def test_lindblad_invalid_initial_trace_rejected(self) -> None:
        """Non-unit trace initial state is rejected with LindbladError."""
        solver = LindbladMasterEquation(np.eye(2, dtype=complex))
        bad_rho = np.array([[2.0, 0], [0, 0]], dtype=complex)  # Tr = 2
        with pytest.raises(LindbladError, match="Trace conservation violated"):
            solver.evolve(bad_rho, t_span=(0.0, 1.0), steps=10)

    def test_custom_gate_rejects_non_unitary(self) -> None:
        """custom_gate rejects non-unitary matrices with CustomGateError."""
        non_unitary = np.array([[1.0, 0.5], [0.0, 1.0]], dtype=complex)
        with pytest.raises(CustomGateError):
            custom_gate("BAD", non_unitary)

    def test_unitaries_equivalent_rejects_scaled_matrices(self) -> None:
        """unitaries_equivalent rejects scaled identity 2*I vs I."""
        eye2 = np.eye(2, dtype=complex)
        assert not unitaries_equivalent(eye2, 2.0 * eye2)
