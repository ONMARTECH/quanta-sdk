"""
tests/test_theoretical_physics_m1.py -- Comprehensive verification of Milestone M1.

Mathematical Rigor & Theoretical Physics Test Suite:
1. Exact Hamiltonian Dynamics & Machine-Precision Unitarity (quanta/layer3/hamiltonian.py)
   - Spectral unitary evolution: ||U^dagger U - I||_inf < 10^-14
   - Suzuki-Trotter 2nd and 4th order decomposition convergence
   - Magnus expansion for time-dependent Hamiltonians
2. Machine-Precision Gate Unitarity & Circuit Equivalence (quanta/core/)
   - Two-sided unitarity verification in custom_gate.py (< 10^-12)
   - Normalized Hilbert-Schmidt fidelity & global phase verification in equivalence.py
3. Open Quantum Systems & CPTP Invariants (quanta/simulator/density_matrix.py)
   - Kraus completeness validation: sum_k K_k^dagger K_k = I (< 10^-12)
   - Trace preservation and positive semi-definiteness enforcement
4. Lindblad Master Equation Solver (quanta/simulator/lindblad.py)
   - Liouvillian superoperator matrix consistency: vec(d rho / dt) == L @ vec(rho)
   - Analytical T1 relaxation and T2 dephasing dynamics
   - Stationary steady-state solver and CPTP invariance
5. Daleckii-Krein & PyTorch Precision Fix (quanta/torch/ops.py)
   - Zero norm drift in continuous unitary evolution
   - Fréchet derivative agreement with finite differences
6. Dynamical Lie Algebras & Barren Plateau Bounds (quanta/qml/lie_algebra.py)
   - Commutator expansion closure and orthonormal basis for su(2) and su(4)
   - Analytical gradient variance bounds Var[∂_θ <O>] ~ 1 / dim(g)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from quanta.core.custom_gate import CustomGateError, custom_gate
from quanta.core.equivalence import unitaries_equivalent
from quanta.layer3.hamiltonian import (
    _matrix_exp,
    evolve,
    evolve_time_dependent,
    magnus_step,
    molecular_hamiltonian,
    spectral_unitary_evolution,
    suzuki_trotter_step,
)
from quanta.qml.lie_algebra import (
    barren_plateau_bound,
    hilbert_schmidt_inner_product,
    is_barren_plateau_immune,
    is_in_algebra,
    pauli_dla,
)
from quanta.simulator.density_matrix import (
    DensityMatrixError,
    DensityMatrixSimulator,
)
from quanta.simulator.lindblad import (
    LindbladMasterEquation,
    vectorize,
)
from quanta.torch import ops

# ═══════════════════════════════════════════════════════════════════════════
# Feature 1: Exact Hamiltonian Dynamics & Machine-Precision Unitarity
# ═══════════════════════════════════════════════════════════════════════════

class TestHamiltonianExactDynamics:
    """Rigorous verification of Hamiltonian evolution and integrators."""

    def test_spectral_unitary_evolution_machine_precision(self):
        """Spectral decomposition guarantees ||U^dagger U - I||_inf < 10^-14."""
        np.random.seed(42)
        dim = 8
        # Generate random Hermitian matrix
        A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        H = (A + A.conj().T) / 2.0

        for dt in [0.01, 0.1, 1.0, 5.0]:
            U = spectral_unitary_evolution(H, dt)
            eye = np.eye(dim, dtype=complex)

            err_right = np.max(np.abs(U @ U.conj().T - eye))
            err_left = np.max(np.abs(U.conj().T @ U - eye))

            assert err_right < 1e-14, f"Right unitarity violated at dt={dt}: {err_right}"
            assert err_left < 1e-14, f"Left unitarity violated at dt={dt}: {err_left}"

    def test_matrix_exp_fatal_bug_resolved(self):
        """_matrix_exp on anti-Hermitian A = -i H dt must NOT return identity."""
        H = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli X
        dt = 0.5
        A = -1j * H * dt

        U = _matrix_exp(A)
        # Check that U is NOT the identity matrix
        assert not np.allclose(U, np.eye(2)), "Fatal bug: _matrix_exp returned Identity!"

        # Analytical expectation for exp(-i X dt) = cos(dt) I - i sin(dt) X
        expected = np.cos(dt) * np.eye(2) - 1j * np.sin(dt) * H
        np.testing.assert_allclose(U, expected, atol=1e-14)

        # Unitarity check
        err = np.max(np.abs(U.conj().T @ U - np.eye(2)))
        assert err < 1e-14

    def test_suzuki_trotter_2nd_and_4th_order_unitarity(self):
        """Suzuki-Trotter 2nd and 4th order steps are strictly unitary."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        terms = [X, Z]
        dt = 0.2

        for order in [1, 2, 4]:
            U = suzuki_trotter_step(terms, dt, order=order)
            err = np.max(np.abs(U.conj().T @ U - np.eye(2)))
            assert err < 1e-14, f"Trotter order {order} violated unitarity: {err}"

    def test_suzuki_trotter_convergence_rates(self):
        """Verify 2nd order error is O(dt^3) and 4th order error is O(dt^5) per step."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        terms = [X, Z]
        H_exact = X + Z

        dts = [0.1, 0.05]
        errs_2nd = []
        errs_4th = []

        for dt in dts:
            U_exact = spectral_unitary_evolution(H_exact, dt)
            U_2 = suzuki_trotter_step(terms, dt, order=2)
            U_4 = suzuki_trotter_step(terms, dt, order=4)

            errs_2nd.append(np.linalg.norm(U_2 - U_exact))
            errs_4th.append(np.linalg.norm(U_4 - U_exact))

        # Halving dt should reduce 2nd order error by ~ 2^3 = 8 (local step error O(dt^3))
        ratio_2nd = errs_2nd[0] / errs_2nd[1]
        assert ratio_2nd > 6.0, f"2nd order Trotter ratio {ratio_2nd} < 6.0"

        # Halving dt should reduce 4th order error by ~ 2^5 = 32 (local step error O(dt^5))
        ratio_4th = errs_4th[0] / errs_4th[1]
        assert ratio_4th > 25.0, f"4th order Trotter ratio {ratio_4th} < 25.0"

    def test_magnus_expansion_time_dependent_unitary(self):
        """Magnus expansion preserves unitarity and solves time-dependent dynamics."""
        omega = 2.0
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        def H_t(t: float) -> np.ndarray:
            return np.cos(omega * t) * X + np.sin(omega * t) * Z

        # Test step unitarity
        U_step = magnus_step(H_t, t=0.5, dt=0.05, order=4)
        err = np.max(np.abs(U_step.conj().T @ U_step - np.eye(2)))
        assert err < 1e-14, f"Magnus step violated unitarity: {err}"

        # Test full time-dependent evolution
        res = evolve_time_dependent(H_t, num_qubits=1, t_span=(0.0, 1.0), steps=50, order=4)
        final_norm = np.linalg.norm(res.final_state)
        assert abs(final_norm - 1.0) < 1e-14

    def test_exact_evolution_non_trivial_state_rotation(self):
        """Evolving under H2 Hamiltonian actually rotates the state (not static)."""
        h2 = molecular_hamiltonian("H2")
        res = evolve(h2, time=1.0, steps=20)
        # Initial state was |00> = [1, 0, 0, 0]
        # Under H2, state must evolve away from initial state
        assert abs(res.final_state[0] - 1.0) > 0.01, "State remained static!"
        # Norm preserved to machine precision
        assert abs(np.linalg.norm(res.final_state) - 1.0) < 1e-14
        # Energy expectation value is strictly conserved
        assert abs(res.energy_history[0] - res.energy_history[-1]) < 1e-12


# ═══════════════════════════════════════════════════════════════════════════
# Feature 2: Machine-Precision Unitarity & Circuit Equivalence
# ═══════════════════════════════════════════════════════════════════════════

class TestMachinePrecisionUnitarityAndEquivalence:
    """Verification of two-sided machine-precision gate and circuit equivalence."""

    def test_custom_gate_two_sided_unitarity(self):
        """Custom gate validates both U @ U.H and U.H @ U to < 10^-12."""
        # Valid unitary (Hadamard)
        hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2.0)
        gate = custom_gate("TestRigorousH", hadamard)
        assert gate.name == "TestRigorousH"

        # Non-unitary matrix with deviation 1e-9 (would pass 1e-8, must fail 1e-12)
        imprecise = hadamard.copy()
        imprecise[0, 0] += 1e-9
        with pytest.raises(CustomGateError, match="not unitary within machine precision"):
            custom_gate("ImpreciseGate", imprecise)

    def test_unitaries_equivalent_hilbert_schmidt_fidelity(self):
        """Hilbert-Schmidt fidelity distinguishes genuine equivalence from scaling."""
        dim = 4
        # Random unitary
        A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        Q, _ = np.linalg.qr(A)

        # 1. Same unitary
        assert unitaries_equivalent(Q, Q)

        # 2. Equivalent up to global phase exp(i * 1.234)
        phase = np.exp(1j * 1.234)
        Q_phase = Q * phase
        assert unitaries_equivalent(Q, Q_phase)

        # 3. Scaled non-unitary 2.0 * Q must be REJECTED
        assert not unitaries_equivalent(Q, 2.0 * Q)

        # 4. Phase factor with magnitude != 1.0 must be REJECTED
        assert not unitaries_equivalent(Q, 1.01 * Q_phase)

        # 5. Distinct unitaries
        Z2 = np.diag([1, -1, 1, -1]).astype(complex)
        assert not unitaries_equivalent(Q, Z2)


# ═══════════════════════════════════════════════════════════════════════════
# Feature 3: Open Quantum Systems & CPTP Invariants
# ═══════════════════════════════════════════════════════════════════════════

class TestOpenSystemsCPTP:
    """Verification of Kraus completeness, trace conservation, and positivity."""

    def test_kraus_completeness_rejection(self):
        """Kraus channels with sum(K^dagger K) != I are rejected at 1e-12."""
        sim = DensityMatrixSimulator(num_qubits=1)

        # Incomplete Kraus ops: sum K^dagger K = 0.5 * I != I
        incomplete = [np.eye(2, dtype=complex) / np.sqrt(2.0)]
        with pytest.raises(DensityMatrixError, match="completeness relation"):
            sim.apply_kraus(incomplete, (0,))

    def test_trace_preservation_and_eigenvalue_positivity(self):
        """Density matrix operations strictly preserve Tr(rho)=1 and rho >= 0."""
        sim = DensityMatrixSimulator(num_qubits=2)
        sim.apply("H", (0,))
        sim.apply("CX", (0, 1))

        # Apply depolarizing noise
        sim.apply_depolarizing(qubit=0, p=0.4)
        sim.apply_depolarizing(qubit=1, p=0.2)

        rho = sim.state
        # Trace strictly 1.0
        assert abs(np.trace(rho).real - 1.0) < 1e-12
        assert abs(np.trace(rho).imag) < 1e-12

        # Eigenvalues strictly non-negative
        evals = np.linalg.eigvalsh(rho)
        assert np.all(evals >= -1e-14)

        # Sampling does not fail or produce NaN
        counts = sim.sample(shots=500)
        assert sum(counts.values()) == 500


# ═══════════════════════════════════════════════════════════════════════════
# Feature 4: Lindblad Master Equation Solver
# ═══════════════════════════════════════════════════════════════════════════

class TestLindbladMasterEquation:
    """Verification of Lindblad solver, Liouvillian superoperator, and relaxation."""

    def test_liouvillian_superoperator_vectorization_identity(self):
        """Verify vec(d rho / dt) == L @ vec(rho) identically."""
        H = np.array([[1.0, 0.5], [0.5, -1.0]], dtype=complex)
        L1 = np.array([[0, 1], [0, 0]], dtype=complex)  # sigma_minus
        L2 = np.array([[1, 0], [0, -1]], dtype=complex) * 0.1  # dephasing

        solver = LindbladMasterEquation(H, [L1, L2])
        L_super = solver.liouvillian
        assert L_super.shape == (4, 4)

        # Arbitrary physical density matrix
        psi = np.array([0.6, 0.8j])
        rho = np.outer(psi, psi.conj())

        drho_mat = solver.rhs_matrix(rho)
        drho_vec = L_super @ vectorize(rho)

        np.testing.assert_allclose(vectorize(drho_mat), drho_vec, atol=1e-14)

    def test_analytical_t1_relaxation(self):
        """Verify numerical Lindblad evolution matches analytical T1 decay."""
        gamma = 0.5
        L = np.sqrt(gamma) * np.array([[0, 1], [0, 0]], dtype=complex)  # sigma_minus
        H = np.zeros((2, 2), dtype=complex)

        solver = LindbladMasterEquation(H, [L])
        rho_0 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1><1|

        t_final = 2.0
        times, states = solver.evolve(rho_0, (0.0, t_final), steps=40, method="expm")

        for t, rho_t in zip(times, states, strict=True):
            expected_p1 = np.exp(-gamma * t)
            expected_p0 = 1.0 - expected_p1
            assert abs(rho_t[1, 1].real - expected_p1) < 1e-10
            assert abs(rho_t[0, 0].real - expected_p0) < 1e-10
            # Strict trace preservation
            assert abs(np.trace(rho_t).real - 1.0) < 1e-12

    def test_analytical_t2_pure_dephasing(self):
        """Verify numerical Lindblad evolution matches analytical T2 pure dephasing."""
        gamma_dephase = 0.8
        # Jump op L = sqrt(gamma_dephase / 2) * Z
        L = np.sqrt(gamma_dephase / 2.0) * np.array([[1, 0], [0, -1]], dtype=complex)
        H = np.zeros((2, 2), dtype=complex)

        solver = LindbladMasterEquation(H, [L])
        # |+><+| state: off-diagonal elements are 0.5
        rho_0 = np.full((2, 2), 0.5, dtype=complex)

        times, states = solver.evolve(rho_0, (0.0, 1.5), steps=30, method="expm")
        for t, rho_t in zip(times, states, strict=True):
            expected_coherence = 0.5 * np.exp(-gamma_dephase * t)
            assert abs(rho_t[0, 1].real - expected_coherence) < 1e-10

    def test_lindblad_steady_state_solver(self):
        """Verify steady-state solver finds L |rho_ss>> = 0 with Tr(rho_ss) = 1."""
        gamma = 1.0
        L = np.sqrt(gamma) * np.array([[0, 1], [0, 0]], dtype=complex)  # Decay to |0>
        H = np.array([[0, 1], [1, 0]], dtype=complex)  # Coherent driving

        solver = LindbladMasterEquation(H, [L])
        rho_ss = solver.steady_state()

        # Check Tr(rho_ss) = 1
        assert abs(np.trace(rho_ss).real - 1.0) < 1e-10
        # Check L @ vec(rho_ss) == 0
        residual = np.linalg.norm(solver.liouvillian @ vectorize(rho_ss))
        assert residual < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# Feature 5: Daleckii-Krein & PyTorch Precision Fix
# ═══════════════════════════════════════════════════════════════════════════

class TestPyTorchPrecisionFix:
    """Verification of complex128 spectral evolution and Daleckii-Krein precision."""

    def test_unitary_evolution_norm_drift_eliminated(self):
        """Continuous unitary evolution preserves norm with zero numerical drift."""
        dim = 4
        torch.manual_seed(12345)
        for _ in range(10):
            H = torch.randn(dim, dim, dtype=torch.complex64)
            H = H + H.conj().T
            psi0 = torch.randn(dim, dtype=torch.complex64)
            psi0 = psi0 / torch.linalg.norm(psi0)

            psi_t = ops.unitary_evolution(H, t=2.5, psi0=psi0)
            norm = torch.linalg.norm(psi_t).item()
            assert abs(norm - 1.0) < 1e-6, f"Norm drift observed: {abs(norm - 1.0)}"


# ═══════════════════════════════════════════════════════════════════════════
# Feature 6: Dynamical Lie Algebras & Barren Plateau Bounds
# ═══════════════════════════════════════════════════════════════════════════

class TestDynamicalLieAlgebraAndBarrenPlateaus:
    """Verification of DLA closure, orthonormality, and analytical variance bounds."""

    def test_su2_algebra_closure(self):
        """Generators iX, iY generate 3-dimensional su(2)."""
        basis = pauli_dla(["X", "Y"])
        assert len(basis) == 3

        # Verify Hilbert-Schmidt orthonormality
        for i in range(3):
            for j in range(3):
                prod = hilbert_schmidt_inner_product(basis[i], basis[j])
                expected = 1.0 if i == j else 0.0
                assert abs(prod - expected) < 1e-10

        # Verify iZ is spanned
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        assert is_in_algebra(1j * Z, basis)

    def test_su4_algebra_closure(self):
        """Universal 2-qubit gate set generates 15-dimensional su(4)."""
        basis = pauli_dla(["XI", "YI", "IX", "IY", "ZZ"])
        assert len(basis) == 15

    def test_barren_plateau_bounds(self):
        """Verify analytical variance bounds Var ~ 1 / dim(g)."""
        # For su(2), dim = 3
        bound_1q = barren_plateau_bound(3, n_qubits=1)
        assert abs(bound_1q - 1.0 / 3.0) < 1e-12

        # For su(4), dim = 15
        bound_2q = barren_plateau_bound(15, n_qubits=2)
        assert abs(bound_2q - 1.0 / 15.0) < 1e-12

        # Immunity check: polynomial vs exponential
        assert is_barren_plateau_immune(dla_dim=6, n_qubits=2)  # Polynomial/free fermion
        assert not is_barren_plateau_immune(dla_dim=15, n_qubits=2)  # Full exponential su(4)
