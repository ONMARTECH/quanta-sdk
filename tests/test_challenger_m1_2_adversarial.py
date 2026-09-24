"""
tests/test_challenger_m1_2_adversarial.py -- Challenger 2 Stress Test Suite for Milestone M1.

Empirical verification of:
1. 1,000 successive Kraus operations (amplitude damping + phase damping + depolarizing channel).
   Confirms |Tr(rho) - 1.0| < 1e-12 at every single step.
2. Daleckii-Krein matrix exponential autograd vs central finite differences (h=1e-5)
   across 10 random parameterized Hamiltonians: confirms relative error < 1e-5.
3. Dynamical Lie Algebra closure:
   - 1-qubit {iX, iY, iZ} generates 3-dim su(2).
   - 2-qubit Heisenberg exchange generates 3-dim abelian algebra, and full Heisenberg
     chain with local fields generates 15-dim su(4).
4. Barren plateau analytical variance bounds vs empirical gradient variance over
   100 Haar-random states for n=2, 3, 4 qubits.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from quanta.qml.lie_algebra import (
    barren_plateau_bound,
    dynamical_lie_algebra,
    hilbert_schmidt_inner_product,
    pauli_dla,
)
from quanta.simulator.density_matrix import DensityMatrixSimulator
from quanta.torch.ops import daleckii_krein_spectral_derivative


class TestKraus1000OperationsCPTP:
    """Stress Test 1: 1,000 successive Kraus operations maintaining Tr(rho) == 1.0 within 1e-12."""

    def test_1000_successive_kraus_single_qubit(self):
        """Runs 1,000 mixed Kraus operations on 1-qubit state, checking trace deviation."""
        sim = DensityMatrixSimulator(num_qubits=1, seed=42)
        sim.apply("H", (0,))

        def amp_damping(gamma: float) -> list[np.ndarray]:
            K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
            K1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
            return [K0, K1]

        def phase_damping(lam: float) -> list[np.ndarray]:
            K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - lam)]], dtype=complex)
            K1 = np.array([[0.0, 0.0], [0.0, np.sqrt(lam)]], dtype=complex)
            return [K0, K1]

        max_deviation = 0.0
        for step in range(1000):
            mod = step % 3
            if mod == 0:
                sim.apply_kraus(amp_damping(0.005), (0,))
            elif mod == 1:
                sim.apply_kraus(phase_damping(0.005), (0,))
            else:
                sim.apply_depolarizing(0, 0.005)

            tr = np.trace(sim.state)
            dev = abs(float(np.real(tr)) - 1.0)
            imag_dev = abs(float(np.imag(tr)))
            total_dev = max(dev, imag_dev)
            if total_dev > max_deviation:
                max_deviation = total_dev

            assert total_dev < 1e-12, f"Step {step}: Tr(rho) deviation {total_dev} >= 1e-12"

        assert max_deviation < 1e-12
        # Positivity check
        evals = np.linalg.eigvalsh(sim.state)
        assert np.all(evals >= -1e-14)

    def test_1000_successive_kraus_two_qubit(self):
        """Runs 1,000 mixed Kraus operations across 2 qubits of an entangled Bell state."""
        sim = DensityMatrixSimulator(num_qubits=2, seed=1337)
        sim.apply("H", (0,))
        sim.apply("CX", (0, 1))

        def amp_damping(gamma: float) -> list[np.ndarray]:
            K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
            K1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
            return [K0, K1]

        def phase_damping(lam: float) -> list[np.ndarray]:
            K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - lam)]], dtype=complex)
            K1 = np.array([[0.0, 0.0], [0.0, np.sqrt(lam)]], dtype=complex)
            return [K0, K1]

        max_deviation = 0.0
        for step in range(1000):
            q = step % 2
            mod = (step // 2) % 3
            if mod == 0:
                sim.apply_kraus(amp_damping(0.01), (q,))
            elif mod == 1:
                sim.apply_kraus(phase_damping(0.01), (q,))
            else:
                sim.apply_depolarizing(q, 0.01)

            tr = np.trace(sim.state)
            dev = abs(float(np.real(tr)) - 1.0)
            if dev > max_deviation:
                max_deviation = dev

            assert dev < 1e-12, f"Step {step}: Tr(rho) deviation {dev} >= 1e-12"

        assert max_deviation < 1e-12


class TestDaleckiiKreinAutogradVsFiniteDiff:
    """Stress Test 2: Daleckii-Krein matrix exponential autograd vs central finite differences."""

    def test_daleckii_krein_across_10_random_hamiltonians(self):
        """Confirms relative gradient error < 1e-5 across 10 random parameterized Hamiltonians."""
        torch.manual_seed(20260924)
        h = 1e-5

        for i in range(10):
            dim = 4 if i < 6 else 8
            # Generate random Hermitian H0 and Omega
            A = torch.randn(dim, dim, dtype=torch.complex128)
            H0 = (A + A.mH) / 2.0
            B = torch.randn(dim, dim, dtype=torch.complex128)
            Omega = (B + B.mH) / 2.0

            t = float(torch.empty(1).uniform_(0.5, 2.5).item())
            theta = float(torch.randn(1).item())

            H = H0 + theta * Omega

            # 1. Analytical Daleckii-Krein Fréchet derivative dU/dtheta
            evals, evecs = torch.linalg.eigh(H)
            dU_dk = daleckii_krein_spectral_derivative(evals, evecs, t, Omega)

            # 2. Central finite difference (step h = 1e-5)
            U_plus = torch.linalg.matrix_exp(-1j * (H + h * Omega) * t)
            U_minus = torch.linalg.matrix_exp(-1j * (H - h * Omega) * t)
            dU_fd = (U_plus - U_minus) / (2.0 * h)

            # Matrix Frobenius relative error
            norm_diff = torch.linalg.norm(dU_dk - dU_fd).item()
            norm_dk = torch.linalg.norm(dU_dk).item()
            rel_err_matrix = norm_diff / norm_dk

            assert rel_err_matrix < 1e-5, (
                f"Hamiltonian {i} (dim={dim}) matrix relative error {rel_err_matrix} >= 1e-5"
            )

            # 3. Scalar expectation value gradient
            psi0 = torch.randn(dim, dtype=torch.complex128)
            psi0 = psi0 / torch.linalg.norm(psi0)
            O_mat = torch.randn(dim, dim, dtype=torch.complex128)
            O_mat = (O_mat + O_mat.mH) / 2.0

            U = torch.linalg.matrix_exp(-1j * H * t)
            psi = U @ psi0
            dpsi = dU_dk @ psi0
            grad_dk = 2.0 * torch.real(torch.dot(psi.conj(), O_mat @ dpsi)).item()

            psi_plus = U_plus @ psi0
            psi_minus = U_minus @ psi0
            E_plus = torch.real(torch.dot(psi_plus.conj(), O_mat @ psi_plus)).item()
            E_minus = torch.real(torch.dot(psi_minus.conj(), O_mat @ psi_minus)).item()
            grad_fd = (E_plus - E_minus) / (2.0 * h)

            rel_err_scalar = abs(grad_dk - grad_fd) / max(abs(grad_dk), 1e-6)
            assert rel_err_scalar < 1e-5, (
                f"Hamiltonian {i} (dim={dim}) scalar relative error {rel_err_scalar} >= 1e-5"
            )


class TestDynamicalLieAlgebraHeisenbergClosure:
    """Stress Test 3: DLA closure on 1-qubit {iX, iY, iZ} and 2-qubit Heisenberg spin chain."""

    def test_dla_1_qubit_su2_dimension(self):
        """Pauli generators {iX, iY, iZ} generate exactly 3-dimensional su(2)."""
        basis = pauli_dla(["X", "Y", "Z"])
        assert len(basis) == 3, f"Expected dim(su(2)) = 3, got {len(basis)}"

        # Verify Hilbert-Schmidt orthonormality
        for i in range(3):
            for j in range(3):
                prod = hilbert_schmidt_inner_product(basis[i], basis[j])
                expected = 1.0 if i == j else 0.0
                assert abs(prod - expected) < 1e-10

    def test_dla_2_qubit_heisenberg_exchange_dimension(self):
        """2-qubit Heisenberg exchange {XX, YY, ZZ} generates 3-dim abelian algebra."""
        basis = pauli_dla(["XX", "YY", "ZZ"])
        # Since [XX, YY] = [YY, ZZ] = [ZZ, XX] = 0, commutators vanish identically
        assert len(basis) == 3, f"Expected 3-dim abelian exchange algebra, got {len(basis)}"

    def test_dla_2_qubit_heisenberg_chain_with_fields_dimension(self):
        """2-qubit Heisenberg spin chain with local fields generates 15-dimensional su(4)."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        eye2 = np.eye(2, dtype=complex)

        XX = np.kron(X, X)
        YY = np.kron(Y, Y)
        ZZ = np.kron(Z, Z)
        X1 = np.kron(X, eye2)
        Z1 = np.kron(Z, eye2)
        X2 = np.kron(eye2, X)
        Z2 = np.kron(eye2, Z)

        # Standard quantum control configuration: Heisenberg interaction + single controls {X1, Z1}
        H_heis = XX + YY + ZZ
        basis_control = dynamical_lie_algebra([1j * H_heis, 1j * X1, 1j * Z1])
        assert len(basis_control) == 15, f"Expected su(4) dim=15, got {len(basis_control)}"

        # Full Heisenberg chain with local fields
        basis_full = dynamical_lie_algebra(
            [1j * XX, 1j * YY, 1j * ZZ, 1j * X1, 1j * X2, 1j * Z1, 1j * Z2]
        )
        assert len(basis_full) == 15, f"Expected su(4) dim=15, got {len(basis_full)}"


class TestBarrenPlateauAnalyticalVarianceHaar:
    """Stress Test 4: Barren plateau variance bounds vs empirical variance over 100 Haar states."""

    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_barren_plateau_variance_vs_haar_states(self, n_qubits: int):
        """Empirical gradient variance over 100 Haar states matches analytical Haar variance."""
        np.random.seed(42 + n_qubits)
        d = 2 ** n_qubits

        # Pauli generator G = 0.5 * X_0, observable obs = Z_0
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        eye2 = np.eye(2, dtype=complex)

        G = 0.5 * X
        obs = Z
        for _ in range(n_qubits - 1):
            G = np.kron(G, eye2)
            obs = np.kron(obs, eye2)

        # Commutator A = i [G, obs] = Y_0 \otimes eye2
        A = 1j * (G @ obs - obs @ G)

        # Exact analytical Haar state variance: Tr(A^2) / (d (d + 1)) = 1 / (d + 1)
        exact_haar_var = float(np.real(np.trace(A @ A.conj().T))) / (d * (d + 1))
        expected_closed_form = 1.0 / (d + 1)
        assert abs(exact_haar_var - expected_closed_form) < 1e-12

        # Empirical variance over 100 Haar-random states
        grads = []
        for _ in range(100):
            z = np.random.randn(d) + 1j * np.random.randn(d)
            psi = z / np.linalg.norm(z)
            grad = float(np.real(psi.conj().T @ A @ psi))
            grads.append(grad)

        emp_var = float(np.var(grads))
        emp_mean = float(np.mean(grads))

        # Mean gradient must be close to zero (unbiased estimator)
        assert abs(emp_mean) < 0.1, f"Mean gradient {emp_mean} far from 0.0"

        # Empirical variance must agree with analytical Haar variance within standard error (N=100)
        # For N=100, standard error of sample variance is ~ 0.14 * exact_var
        rel_diff = abs(emp_var - exact_haar_var) / exact_haar_var
        err_msg = (
            f"n={n_qubits}: Empirical var {emp_var} differs from "
            f"analytical {exact_haar_var} by {rel_diff * 100:.1f}%"
        )
        assert rel_diff < 0.35, err_msg

        # Confirm analytical DLA bound scales as O(1/dim(g))
        dla_dim = (4 ** n_qubits) - 1
        dla_bound = barren_plateau_bound(dla_dim, n_qubits)
        assert dla_bound == 1.0 / dla_dim
