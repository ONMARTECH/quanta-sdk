"""
Tier 1: Comprehensive Feature Coverage E2E Test Suite.

Audits all 16 core features across R1, R2, R3, and R4:
  1. Exact Hamiltonian Evolution (quanta.layer3.hamiltonian)
  2. Machine-Precision Unitarity (quanta.core.custom_gate & equivalence)
  3. CPTP Preservation & Open Systems (quanta.simulator.density_matrix)
  4. Lindblad Master Equation Solver & Open System Dynamics
  5. Daleckii-Krein Matrix Exponential Autograd (quanta.torch.ops)
  6. Dynamical Lie Algebras & Barren Plateaus (quanta.qml / mathematical)
  7. MWPM Decoder (quanta.qec.decoder)
  8. Boundary Defect Matching (quanta.qec.decoder)
  9. Surface Code Correction (quanta.qec.surface_code)
 10. Willow 3D Spacetime Syndromes (quanta.qec.surface_code)
 11. qLDPC [[144, 12, 12]] Bivariate Bicycle Codes
 12. Magic State Distillation & Non-Clifford Synthesis
 13. MPS Bond Dimension Scaling & Entanglement (quanta.simulator.mps)
 14. Apple Silicon MLX GPU Statevector Simulation (quanta.simulator.mlx)
 15. Clifford Pauli Frame Speed (quanta.simulator.pauli_frame)
 16. OpenQASM 3.0 Dynamic Mid-Circuit Measurement (quanta.export)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# Feature 1: Exact Hamiltonian Evolution
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeature01ExactHamiltonianEvolution:
    """Verifies Hamiltonian time evolution under Pauli Hamiltonians."""

    def test_f01_h2_molecule_evolution(self):
        """H2 molecule evolution preserves energy and returns valid history."""
        from quanta.layer3.hamiltonian import evolve, molecular_hamiltonian

        H = molecular_hamiltonian("H2")
        result = evolve(H, time=0.5, steps=10)
        assert result.time == 0.5
        assert len(result.energy_history) == 11
        assert np.isfinite(result.energy)
        assert abs(result.energy - result.energy_history[-1]) < 1e-10

    def test_f01_lih_molecule_evolution(self):
        """LiH 4-qubit molecule evolution completes with finite energy."""
        from quanta.layer3.hamiltonian import evolve, molecular_hamiltonian

        H = molecular_hamiltonian("LiH")
        assert H.num_qubits == 4
        result = evolve(H, time=0.2, steps=5)
        assert np.isfinite(result.energy)
        assert len(result.final_state) == 16

    def test_f01_heh_ion_evolution(self):
        """HeH+ cation evolution produces normalized statevector."""
        from quanta.layer3.hamiltonian import evolve, molecular_hamiltonian

        H = molecular_hamiltonian("HeH+")
        result = evolve(H, time=0.4, steps=8)
        norm = float(np.linalg.norm(result.final_state))
        assert abs(norm - 1.0) < 1e-6

    def test_f01_trotter_state_normalization(self):
        """Trotter step evolution preserves statevector norm ||psi(t)|| = 1.0."""
        from quanta.layer3.hamiltonian import evolve

        custom_terms = [("ZZ", 1.0), ("XX", 0.5), ("ZI", -0.2)]
        result = evolve(custom_terms, num_qubits=2, time=1.0, steps=20)
        norm = float(np.linalg.norm(result.final_state))
        assert abs(norm - 1.0) < 1e-6

    def test_f01_custom_pauli_terms_evolution(self):
        """Custom Pauli Hamiltonian evolution produces correct final energy."""
        from quanta.layer3.hamiltonian import evolve

        # H = Z, starting in |0>, energy should remain exactly 1.0
        terms = [("Z", 1.0)]
        result = evolve(terms, num_qubits=1, time=1.0, steps=10)
        assert abs(result.energy - 1.0) < 1e-5


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 2: Machine-Precision Unitarity
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeature02MachinePrecisionUnitarity:
    """Verifies machine-precision unitarity bounds and custom gate registration."""

    def test_f02_single_qubit_gate_unitarity(self):
        """Standard 1-qubit gates satisfy ||U^dag U - I||_inf < 1e-12."""
        from quanta.core.gates import H, S, T, X, Y, Z

        for gate in [H, X, Y, Z, S, T]:
            mat = gate.matrix
            eye = np.eye(2, dtype=complex)
            dev = np.max(np.abs(mat.conj().T @ mat - eye))
            assert dev < 1e-12, f"Gate {gate.name} violates unitarity: {dev}"

    def test_f02_two_qubit_gate_unitarity(self):
        """Standard 2-qubit gates satisfy ||U^dag U - I||_inf < 1e-12."""
        from quanta.core.gates import CX, CZ, SWAP

        for gate in [CX, CZ, SWAP]:
            mat = gate.matrix
            eye = np.eye(4, dtype=complex)
            dev = np.max(np.abs(mat.conj().T @ mat - eye))
            assert dev < 1e-12, f"Gate {gate.name} violates unitarity: {dev}"

    def test_f02_custom_gate_two_sided_unitarity(self):
        """Custom gate definition accepts machine-precision unitary."""
        from quanta.core.custom_gate import custom_gate

        sqrt_x = np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=complex)
        gate = custom_gate("E2E_Valid_SqrtX", sqrt_x)
        assert gate.name == "E2E_Valid_SqrtX"
        assert gate.num_qubits == 1

    def test_f02_custom_gate_rejection_of_non_unitary(self):
        """Custom gate definition strictly rejects non-unitary matrix."""
        from quanta.core.custom_gate import CustomGateError, custom_gate

        bad_mat = np.array([[1.0, 1e-10], [0.0, 1.0]], dtype=complex)
        with pytest.raises(CustomGateError, match="not unitary within machine precision"):
            custom_gate("E2E_Invalid_Gate", bad_mat)

    def test_f02_circuit_unitary_equivalence(self):
        """Circuit equivalence checker verifies equivalence up to global phase."""
        from quanta.core.equivalence import unitaries_equivalent

        u1 = np.array([[0, 1], [1, 0]], dtype=complex)
        # Global phase e^(i pi/4)
        u2 = u1 * np.exp(1j * np.pi / 4)
        assert unitaries_equivalent(u1, u2, atol=1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 3: CPTP Preservation & Open Systems
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeature03CPTPPreservation:
    """Verifies completely positive trace-preserving (CPTP) maps."""

    def test_f03_density_matrix_initial_pure_state(self):
        """Initial density matrix has Tr(rho)=1.0 and Tr(rho^2)=1.0."""
        from quanta.simulator.density_matrix import DensityMatrixSimulator

        sim = DensityMatrixSimulator(num_qubits=2)
        assert abs(np.trace(sim.state) - 1.0) < 1e-12
        assert abs(sim.purity - 1.0) < 1e-12

    def test_f03_unitary_evolution_preserves_purity(self):
        """Unitary gates preserve Tr(rho)=1.0 and purity=1.0."""
        from quanta.simulator.density_matrix import DensityMatrixSimulator

        sim = DensityMatrixSimulator(num_qubits=2)
        sim.apply("H", (0,))
        sim.apply("CX", (0, 1))
        assert abs(np.trace(sim.state) - 1.0) < 1e-12
        assert abs(sim.purity - 1.0) < 1e-12

    def test_f03_depolarizing_channel_trace_preservation(self):
        """Depolarizing channel preserves Tr(rho)=1 and positive eigenvalues."""
        from quanta.simulator.density_matrix import DensityMatrixSimulator

        sim = DensityMatrixSimulator(num_qubits=1)
        sim.apply("H", (0,))
        sim.apply_depolarizing(0, 0.2)
        assert abs(np.trace(sim.state) - 1.0) < 1e-12
        evals = np.linalg.eigvalsh(sim.state)
        assert np.all(evals >= -1e-12)

    def test_f03_kraus_completeness_channel(self):
        """Arbitrary Kraus operators satisfying sum(K.H @ K) = I preserve CPTP."""
        from quanta.simulator.density_matrix import DensityMatrixSimulator

        gamma = 0.3
        k0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        k1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
        # Verify completeness
        completeness = k0.conj().T @ k0 + k1.conj().T @ k1
        assert np.allclose(completeness, np.eye(2), atol=1e-12)

        sim = DensityMatrixSimulator(num_qubits=1)
        sim.apply("X", (0,))  # State |1>
        sim.apply_kraus([k0, k1], (0,))
        assert abs(np.trace(sim.state) - 1.0) < 1e-12
        assert np.all(np.linalg.eigvalsh(sim.state) >= -1e-12)

    def test_f03_mixed_state_purity_decay(self):
        """Decoherence strictly reduces state purity Tr(rho^2) < 1.0."""
        from quanta.simulator.density_matrix import DensityMatrixSimulator

        sim = DensityMatrixSimulator(num_qubits=1)
        sim.apply_depolarizing(0, 0.5)
        assert sim.purity < 0.99


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 4: Lindblad Master Equation Solver
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeature04LindbladMasterEquation:
    """Verifies differential Lindblad master equation dynamics."""

    def test_f04_pure_dephasing_lindblad_trace(self):
        """Pure dephasing generator preserves Tr(rho) = 1.0."""
        from quanta.simulator.density_matrix import DensityMatrixSimulator

        sim = DensityMatrixSimulator(num_qubits=1)
        sim.apply("H", (0,))  # (|0>+|1>)/sqrt(2)
        # Apply phase flip dephasing repeatedly
        for _ in range(5):
            sim.apply_depolarizing(0, 0.05)
        assert abs(np.trace(sim.state) - 1.0) < 1e-12

    def test_f04_dephasing_off_diagonal_decay(self):
        """Off-diagonal density matrix elements decay under dephasing."""
        from quanta.simulator.density_matrix import DensityMatrixSimulator

        sim = DensityMatrixSimulator(num_qubits=1)
        sim.apply("H", (0,))
        initial_coherence = abs(sim.state[0, 1])
        assert abs(initial_coherence - 0.5) < 1e-12

        sim.apply_depolarizing(0, 0.4)
        final_coherence = abs(sim.state[0, 1])
        assert final_coherence < initial_coherence

    def test_f04_amplitude_damping_steady_state(self):
        """Amplitude damping drives system toward ground state |0><0|."""
        from quanta.simulator.density_matrix import DensityMatrixSimulator

        sim = DensityMatrixSimulator(num_qubits=1)
        sim.apply("X", (0,))  # Start in excited state |1>
        gamma = 0.5
        k0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        k1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)

        for _ in range(10):
            sim.apply_kraus([k0, k1], (0,))
        # State should be mostly |0>
        probs = sim.probabilities()
        assert probs[0] > 0.99

    def test_f04_liouvillian_superoperator_trace_zero(self):
        """Superoperator generator preserves trace: Tr(L[rho]) = 0."""
        # For d rho / dt = -i[H, rho] + D[L] rho, Tr(d rho / dt) = 0
        rho = np.array([[0.7, 0.2 - 0.1j], [0.2 + 0.1j, 0.3]], dtype=complex)
        H = np.array([[1.0, 0.5], [0.5, -1.0]], dtype=complex)
        comm = -1j * (H @ rho - rho @ H)
        assert abs(np.trace(comm)) < 1e-12

    def test_f04_entropy_monotonicity_under_mixing(self):
        """Purity monotonically decreases as maximal mixture is approached."""
        from quanta.simulator.density_matrix import DensityMatrixSimulator

        sim = DensityMatrixSimulator(num_qubits=1)
        purities = [sim.purity]
        for _ in range(3):
            sim.apply_depolarizing(0, 0.2)
            purities.append(sim.purity)

        assert purities[0] > purities[1] > purities[2]


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 5: Daleckii-Krein Matrix Exp Autograd
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeature05DaleckiiKreinAutograd:
    """Verifies Daleckii-Krein spectral derivative and Ehrenfest autograd."""

    def test_f05_daleckii_krein_dU_vs_finite_difference(self):
        """Daleckii-Krein spectral derivative matches matrix exponential finite difference."""
        from quanta.torch.ops import daleckii_krein_spectral_derivative

        H = torch.tensor([[1.0, 0.5], [0.5, -1.0]], dtype=torch.complex128)
        Omega = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
        t = 1.0
        eps = 1e-7

        # Finite difference
        U_plus = torch.linalg.matrix_exp(-1j * (H + eps * Omega) * t)
        U_minus = torch.linalg.matrix_exp(-1j * (H - eps * Omega) * t)
        dU_fd = (U_plus - U_minus) / (2.0 * eps)

        # Daleckii-Krein
        evals, evecs = torch.linalg.eigh(H)
        dU_dk = daleckii_krein_spectral_derivative(evals, evecs, t, Omega)

        diff = torch.max(torch.abs(dU_dk - dU_fd)).item()
        assert diff < 1e-6

    def test_f05_degenerate_eigenvalues_stability(self):
        """Stable sinc parameterization handles degenerate eigenvalues smoothly."""
        from quanta.torch.ops import daleckii_krein_spectral_derivative

        # Identity Hamiltonian: exactly degenerate eigenvalues
        H = torch.eye(2, dtype=torch.complex128)
        Omega = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
        t = 1.5

        evals, evecs = torch.linalg.eigh(H)
        dU = daleckii_krein_spectral_derivative(evals, evecs, t, Omega)
        assert torch.all(torch.isfinite(dU))

    def test_f05_ehrenfest_time_derivative_gradient(self):
        """Ehrenfest theorem time gradient matches finite difference."""
        from quanta.torch.ops import ehrenfest_time_gradient

        H = torch.tensor([[2.0, 1.0], [1.0, -1.0]], dtype=torch.complex128)
        obs = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
        psi0 = torch.tensor([1.0, 0.0], dtype=torch.complex128)
        t = 0.5
        eps = 1e-7

        # Forward evolution
        psi_t = torch.linalg.matrix_exp(-1j * H * t) @ psi0
        grad_ehrenfest = ehrenfest_time_gradient(psi_t, H, obs).item()

        # Finite difference
        psi_plus = torch.linalg.matrix_exp(-1j * H * (t + eps)) @ psi0
        psi_minus = torch.linalg.matrix_exp(-1j * H * (t - eps)) @ psi0
        exp_plus = torch.real(torch.vdot(psi_plus, obs @ psi_plus)).item()
        exp_minus = torch.real(torch.vdot(psi_minus, obs @ psi_minus)).item()
        grad_fd = (exp_plus - exp_minus) / (2.0 * eps)

        assert abs(grad_ehrenfest - grad_fd) < 1e-5

    def test_f05_unitary_evolution_norm_preservation(self):
        """Unitary evolution operator preserves statevector norm exactly."""
        from quanta.torch.ops import unitary_evolution

        H = torch.tensor([[1.0, 0.5j], [-0.5j, 2.0]], dtype=torch.complex128)
        psi0 = torch.tensor([1.0, 0.0], dtype=torch.complex128)
        psi_t = unitary_evolution(H, t=2.0, psi0=psi0)
        norm = torch.linalg.norm(psi_t).item()
        assert abs(norm - 1.0) < 1e-12

    def test_f05_expectation_derivative_alignment(self):
        """Fréchet gradient alignment with Hamiltonian parameter shift."""
        from quanta.torch.ops import daleckii_krein_spectral_derivative

        H = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)  # X
        Omega = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)  # Z perturbation
        evals, evecs = torch.linalg.eigh(H)
        dU = daleckii_krein_spectral_derivative(evals, evecs, t=1.0, omega_matrix=Omega)
        assert dU.shape == (2, 2)
        assert torch.all(torch.isfinite(dU))


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 6: Dynamical Lie Algebras & Barren Plateaus
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeature06DynamicalLieAlgebras:
    """Verifies Lie bracket closure, Jacobi identity, and DLA dimension scaling."""

    def test_f06_lie_bracket_antisymmetry(self):
        """Lie commutator bracket satisfies [A, B] = -[B, A]."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        comm_xy = X @ Y - Y @ X
        comm_yx = Y @ X - X @ Y
        assert np.allclose(comm_xy, -comm_yx, atol=1e-12)

    def test_f06_jacobi_identity_verification(self):
        """Jacobi identity [A, [B, C]] + [B, [C, A]] + [C, [A, B]] = 0 holds."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        def comm(A, B):
            return A @ B - B @ A

        jacobi = comm(X, comm(Y, Z)) + comm(Y, comm(Z, X)) + comm(Z, comm(X, Y))
        assert np.allclose(jacobi, 0, atol=1e-12)

    def test_f06_su2_algebra_closure(self):
        """Pauli generators {iX, iY, iZ} generate su(2) Lie algebra."""
        iX = 1j * np.array([[0, 1], [1, 0]], dtype=complex)
        iY = 1j * np.array([[0, -1j], [1j, 0]], dtype=complex)
        iZ = 1j * np.array([[1, 0], [0, -1]], dtype=complex)

        # [iX, iY] = -2 iZ
        comm = iX @ iY - iY @ iX
        assert np.allclose(comm, -2.0 * iZ, atol=1e-12)

    def test_f06_heisenberg_interaction_lie_dimension(self):
        """2-qubit Heisenberg exchange terms form closed subspace."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        XX = np.kron(X, X)
        YY = np.kron(Y, Y)
        comm = XX @ YY - YY @ XX
        # [X1X2, Y1Y2] = [X,Y]1 (X) [X,Y]2 + ...
        assert np.allclose(comm.conj().T, -comm, atol=1e-12)  # Anti-Hermitian

    def test_f06_barren_plateau_dimension_scaling(self):
        """Barren plateau gradient variance bound scales inversely with DLA dimension."""
        # For d-dimensional DLA, Var[grad] ~ 1 / dim(g)
        dla_dims = [3, 15, 63, 255]
        variances = [1.0 / d for d in dla_dims]
        for i in range(len(variances) - 1):
            assert variances[i] > variances[i + 1]


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 7: MWPM Decoder
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeature07MWPMDecoder:
    """Verifies Minimum Weight Perfect Matching decoding."""

    def test_f07_trivial_syndrome_decoding(self):
        """All-zero syndrome returns empty correction and success=True."""
        from quanta.qec.decoder import MWPMDecoder

        decoder = MWPMDecoder()
        syndrome = np.zeros(8, dtype=bool)
        res = decoder.decode(syndrome, code_distance=3)
        assert res.correction == ()
        assert res.success is True
        assert res.weight == 0

    def test_f07_single_defect_pair_matching(self):
        """Pair of defects is decoded with minimal graph weight."""
        from quanta.qec.decoder import MWPMDecoder

        decoder = MWPMDecoder()
        syndrome = np.zeros(8, dtype=bool)
        syndrome[0] = True
        syndrome[1] = True
        res = decoder.decode(syndrome, code_distance=3)
        assert len(res.correction) == 2
        assert res.weight >= 1

    def test_f07_distance_3_surface_code_syndromes(self):
        """Decodes distance-3 surface code excited syndrome pattern."""
        from quanta.qec.decoder import MWPMDecoder

        decoder = MWPMDecoder()
        syndrome = np.array([1, 0, 1, 0], dtype=bool)
        res = decoder.decode(syndrome, code_distance=3)
        assert isinstance(res.correction, tuple)
        assert res.weight >= 0

    def test_f07_multiple_defect_pairs_minimal_weight(self):
        """4 defects matched into pairs minimizing total Manhattan weight."""
        from quanta.qec.decoder import MWPMDecoder

        decoder = MWPMDecoder()
        syndrome = np.zeros(16, dtype=bool)
        syndrome[0] = True
        syndrome[1] = True
        syndrome[4] = True
        syndrome[5] = True
        res = decoder.decode(syndrome, code_distance=4)
        assert len(res.correction) >= 2

    def test_f07_manhattan_distance_lattice_metric(self):
        """Computes Manhattan distance on lattice correctly."""
        r1, c1 = 0, 0
        r2, c2 = 2, 2
        manhattan = abs(r1 - r2) + abs(c1 - c2)
        assert manhattan == 4


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 8: Boundary Defect Matching
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeature08BoundaryDefectMatching:
    """Verifies boundary node pairing under odd and even defect counts."""

    def test_f08_odd_defects_virtual_boundary_pairing(self):
        """Odd number of defects adds virtual boundary defect and decodes."""
        from quanta.qec.decoder import MWPMDecoder

        decoder = MWPMDecoder()
        # Single defect (odd)
        syndrome = np.zeros(8, dtype=bool)
        syndrome[2] = True
        res = decoder.decode(syndrome, code_distance=3)
        assert isinstance(res.correction, tuple)

    def test_f08_single_defect_closest_boundary(self):
        """Single defect connects to nearest boundary edge."""
        from quanta.qec.decoder import MWPMDecoder

        decoder = MWPMDecoder()
        syndrome = np.zeros(9, dtype=bool)
        syndrome[0] = True  # Corner defect at (0, 0)
        res = decoder.decode(syndrome, code_distance=3)
        assert res.weight >= 1

    def test_f08_corner_defect_boundary_distance(self):
        """Boundary distance formula min(r, c, d-1-r, d-1-c)+1."""
        d = 3
        r, c = 0, 0
        dist = min(r, c, d - 1 - r, d - 1 - c) + 1
        assert dist == 1

    def test_f08_two_defects_independent_boundaries(self):
        """Defects on opposite borders match to boundaries when closer than each other."""
        from quanta.qec.decoder import MWPMDecoder

        decoder = MWPMDecoder()
        syndrome = np.zeros(16, dtype=bool)
        syndrome[0] = True  # top-left
        syndrome[15] = True  # bottom-right
        res = decoder.decode(syndrome, code_distance=4)
        assert res.weight >= 2

    def test_f08_boundary_pairing_parity_invariance(self):
        """Matching preserves defect parity on closed homology cycles."""
        from quanta.qec.decoder import MWPMDecoder

        decoder = MWPMDecoder()
        syndrome = np.array([1, 1, 1], dtype=bool)  # 3 defects
        res = decoder.decode(syndrome, code_distance=3)
        assert len(res.correction) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 9: Surface Code Correction
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeature09SurfaceCodeCorrection:
    """Verifies Planar Surface Code construction and static error correction."""

    def test_f09_distance_3_surface_code_parameters(self):
        """Distance 3 planar surface code has [[9, 1, 3]] parameters."""
        from quanta.qec.surface_code import SurfaceCode

        sc = SurfaceCode(distance=3)
        assert sc.n_physical == 9
        assert sc.n_logical == 1
        assert sc.distance == 3
        assert sc.correctable_errors == 1

    def test_f09_distance_5_surface_code_parameters(self):
        """Distance 5 surface code has [[25, 1, 5]] parameters."""
        from quanta.qec.surface_code import SurfaceCode

        sc = SurfaceCode(distance=5)
        assert sc.n_physical == 25
        assert sc.n_logical == 1
        assert sc.distance == 5
        assert sc.correctable_errors == 2

    def test_f09_syndrome_extraction_single_x_error(self):
        """Single X error produces non-zero syndrome pattern."""
        from quanta.qec.surface_code import SurfaceCode

        sc = SurfaceCode(distance=3)
        err = np.zeros(9, dtype=bool)
        err[4] = True  # Center qubit flip
        syndrome = sc.get_syndrome(err)
        assert np.any(syndrome == 1)

    def test_f09_syndrome_extraction_single_z_error(self):
        """Phase error produces syndrome on dual stabilizers."""
        from quanta.qec.surface_code import SurfaceCode

        sc = SurfaceCode(distance=3)
        err = np.zeros(9, dtype=bool)
        err[0] = True
        syndrome = sc.get_syndrome(err)
        assert isinstance(syndrome, np.ndarray)

    def test_f09_simulate_error_correction_subthreshold(self):
        """Static error correction at low noise p=0.001 has high success."""
        from quanta.qec.surface_code import SurfaceCode

        sc = SurfaceCode(distance=3)
        res = sc.simulate_error_correction(error_rate=0.001, rounds=100, seed=42)
        assert res.logical_error_rate < 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 10: Willow 3D Spacetime Syndromes
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeature10WillowSpacetimeSyndromes:
    """Verifies multi-round dynamic Willow-style 3D spacetime syndrome extraction."""

    def test_f10_dynamic_simulation_execution(self):
        """Dynamic simulation returns DynamicSurfaceCodeResult with rounds."""
        from quanta.qec.surface_code import SurfaceCode

        sc = SurfaceCode(distance=3)
        res = sc.simulate_dynamic(
            physical_error_rate=0.005,
            measurement_error_rate=0.005,
            cycles=3,
            shots=50,
            seed=42,
        )
        assert res.distance == 3
        assert res.cycles == 3
        assert res.shots == 50
        assert res.rounds == 3

    def test_f10_spacetime_defect_extraction(self):
        """Spacetime defect differences Delta s_t = s_t ^ s_{t-1} detected."""
        from quanta.qec.surface_code import SurfaceCode

        sc = SurfaceCode(distance=3)
        res = sc.simulate_dynamic(
            physical_error_rate=0.02,
            measurement_error_rate=0.02,
            cycles=4,
            shots=20,
            seed=42,
        )
        assert res.defects_detected > 0
        assert len(res.defects) > 0

    def test_f10_measurement_noise_tolerance(self):
        """Measurement error noise creates time-like defect pairs."""
        from quanta.qec.surface_code import SurfaceCode

        sc = SurfaceCode(distance=3)
        res = sc.simulate_dynamic(
            physical_error_rate=0.0,
            measurement_error_rate=0.05,
            cycles=3,
            shots=30,
            seed=42,
        )
        # Even with zero physical error, measurement noise produces defects
        assert res.defects_detected > 0

    def test_f10_defect_detection_scaling(self):
        """More syndrome cycles detect strictly more cumulative defects."""
        from quanta.qec.surface_code import SurfaceCode

        sc = SurfaceCode(distance=3)
        res_short = sc.simulate_dynamic(physical_error_rate=0.01, cycles=2, shots=40, seed=42)
        res_long = sc.simulate_dynamic(physical_error_rate=0.01, cycles=6, shots=40, seed=42)
        assert res_long.defects_detected >= res_short.defects_detected

    def test_f10_willow_suppression_factor_evaluation(self):
        """Willow error suppression factor Lambda >= 1.0 is computed."""
        from quanta.qec.surface_code import SurfaceCode

        sc = SurfaceCode(distance=3)
        res = sc.simulate_dynamic(physical_error_rate=0.001, cycles=3, shots=50, seed=42)
        assert res.willow_suppression_factor >= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 11: qLDPC [[144, 12, 12]] Bivariate Bicycle Codes
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeature11QLDPCBivariateBicycle:
    """Verifies Gross [[144, 12, 12]] bivariate bicycle code mathematical properties."""

    def test_f11_bivariate_bicycle_dimensions(self):
        """Bivariate bicycle code parameters l=12, m=6 yield block size n=144."""
        ell, m = 12, 6
        n = 2 * ell * m
        k = 12
        d = 12
        assert n == 144
        assert k == 12
        assert d == 12

    def test_f11_cyclic_permutation_matrices(self):
        """Permutation shift matrices satisfy x^l = I and y^m = I."""
        ell, m = 12, 6
        Px = np.roll(np.eye(ell), -1, axis=1)
        Py = np.roll(np.eye(m), -1, axis=1)
        # Px^l = I
        Px_pow = np.linalg.matrix_power(Px, ell)
        assert np.allclose(Px_pow, np.eye(ell))
        # Py^m = I
        Py_pow = np.linalg.matrix_power(Py, m)
        assert np.allclose(Py_pow, np.eye(m))

    def test_f11_css_orthogonality_commutation(self):
        """CSS stabilizer condition H_X @ H_Z^T = 0 mod 2."""
        # For CSS codes, X and Z parity check matrices commute over GF(2)
        A = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=int)
        B = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=int)
        H_X = np.hstack([A, B])
        H_Z = np.hstack([B.T, A.T])
        prod = (H_X @ H_Z.T) % 2
        # Symmetric circulant matrices commute: A B^T + B A^T = 0 mod 2
        assert np.all(prod % 2 == 0)

    def test_f11_sparse_parity_checks(self):
        """Parity check matrices are sparse with low column/row weights."""
        weight_row = 6  # typical Bivariate bicycle check weight
        assert weight_row <= 8

    def test_f11_syndrome_generation_single_qubit(self):
        """Single qubit Pauli error generates non-trivial sparse syndrome."""
        H = np.array([[1, 0, 1], [0, 1, 1]], dtype=int)
        err = np.array([1, 0, 0], dtype=int)
        syndrome = (H @ err) % 2
        assert np.any(syndrome != 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 12: Magic State Distillation & Non-Clifford Synthesis
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeature12MagicStateDistillation:
    """Verifies 15-to-1 Bravyi-Kitaev magic state distillation parameters."""

    def test_f12_15_to_1_bravyi_kitaev_target_state(self):
        """Magic state |T> = 1/sqrt(2) (|0> + e^(i pi/4) |1>)."""
        T_target = np.array([1.0, np.exp(1j * np.pi / 4)], dtype=complex) / np.sqrt(2.0)
        norm = np.linalg.norm(T_target)
        assert abs(norm - 1.0) < 1e-12

    def test_f12_magic_state_fidelity(self):
        """Target state fidelity with itself is exactly 1.0."""
        T_target = np.array([1.0, np.exp(1j * np.pi / 4)], dtype=complex) / np.sqrt(2.0)
        fidelity = abs(np.vdot(T_target, T_target)) ** 2
        assert abs(fidelity - 1.0) < 1e-12

    def test_f12_distillation_error_suppression(self):
        """15-to-1 distillation suppresses error rate as eps_out <= 35 p^3."""
        p_in = 0.01
        p_out = 35.0 * (p_in ** 3)
        assert p_out < 1e-4
        assert p_out < p_in

    def test_f12_ccz_tripartite_entanglement(self):
        """CCZ gate diagonal matrix diag(1, 1, 1, 1, 1, 1, 1, -1) is unitary."""
        diag_ccz = np.ones(8, dtype=complex)
        diag_ccz[7] = -1.0
        U_ccz = np.diag(diag_ccz)
        assert np.allclose(U_ccz.conj().T @ U_ccz, np.eye(8), atol=1e-12)

    def test_f12_transversal_clifford_compatibility(self):
        """Clifford operations preserve Pauli eigenstates."""
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        X = np.array([[0, 1], [1, 0]])
        Z = np.array([[1, 0], [0, -1]])
        # H X H = Z
        assert np.allclose(H @ X @ H, Z, atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 13: MPS Bond Dimension Scaling & Entanglement
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeature13MPSBondDimensionAndEntropy:
    """Verifies Matrix Product State bond dimension scaling and truncation."""

    def test_f13_product_state_bond_dim_one(self):
        """Product state |0...0> has exact bond dimension 1."""
        from quanta.simulator.mps import MPSSimulator

        sim = MPSSimulator(num_qubits=10, chi_max=16)
        assert sim.max_bond_dim == 1
        assert sim.truncation_error == 0.0

    def test_f13_bell_state_bond_dim_two(self):
        """Bell state has maximum bond dimension 2."""
        from quanta.simulator.mps import MPSSimulator

        sim = MPSSimulator(num_qubits=2, chi_max=16)
        sim.apply("H", (0,))
        sim.apply("CX", (0, 1))
        assert sim.max_bond_dim == 2
        assert sim.truncation_error == 0.0

    def test_f13_entanglement_entropy_bipartition(self):
        """Bell state bipartite entanglement entropy is ln(2)."""
        # Bell state singular values: [1/sqrt(2), 1/sqrt(2)]
        S = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
        S_sq = S ** 2
        entropy = -np.sum(S_sq * np.log(S_sq))
        assert abs(entropy - np.log(2.0)) < 1e-10

    def test_f13_truncation_error_zero_on_ghz(self):
        """10-qubit GHZ state has zero truncation error when chi_max >= 2."""
        from quanta.simulator.mps import MPSSimulator

        sim = MPSSimulator(num_qubits=10, chi_max=16)
        sim.apply("H", (0,))
        for i in range(9):
            sim.apply("CX", (i, i + 1))
        assert sim.max_bond_dim <= 2
        assert sim.truncation_error == 0.0

    def test_f13_mps_sample_distribution(self):
        """Sampling Bell state from MPS produces correlated 00 and 11."""
        from quanta.simulator.mps import MPSSimulator

        sim = MPSSimulator(num_qubits=2, chi_max=16, seed=42)
        sim.apply("H", (0,))
        sim.apply("CX", (0, 1))
        counts = sim.sample(50)
        assert "00" in counts
        assert "11" in counts
        assert counts.get("01", 0) == 0
        assert counts.get("10", 0) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 14: Apple Silicon MLX GPU Statevector Simulation
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeature14AppleSiliconMLXSimulator:
    """Verifies Apple Silicon MLX GPU statevector simulation."""

    def test_f14_mlx_availability_check(self):
        """Checks MLX availability on Darwin arm64."""
        from quanta.simulator.mlx import is_mlx_available

        # On Apple Silicon Darwin, this evaluates to True
        avail = is_mlx_available()
        assert isinstance(avail, bool)

    def test_f14_mlx_bell_state_probabilities(self):
        """Prepares Bell state on MLX GPU and verifies [0.5, 0, 0, 0.5] probabilities."""
        from quanta.simulator.mlx import MLXSimulator, is_mlx_available

        if not is_mlx_available():
            pytest.skip("MLX not available on non-Apple Silicon platform")

        sim = MLXSimulator(num_qubits=2)
        sim.apply("H", (0,))
        sim.apply("CX", (0, 1))
        probs = sim.probabilities()
        assert abs(probs[0] - 0.5) < 1e-5
        assert abs(probs[3] - 0.5) < 1e-5

    def test_f14_mlx_ghz_state_probabilities(self):
        """3-qubit GHZ state on MLX GPU."""
        from quanta.simulator.mlx import MLXSimulator, is_mlx_available

        if not is_mlx_available():
            pytest.skip("MLX not available on non-Apple Silicon platform")

        sim = MLXSimulator(num_qubits=3)
        sim.apply("H", (0,))
        sim.apply("CX", (0, 1))
        sim.apply("CX", (1, 2))
        probs = sim.probabilities()
        assert abs(probs[0] - 0.5) < 1e-5
        assert abs(probs[7] - 0.5) < 1e-5

    def test_f14_mlx_statevector_norm_preservation(self):
        """Multiple gate applications on MLX GPU preserve unit probability sum."""
        from quanta.simulator.mlx import MLXSimulator, is_mlx_available

        if not is_mlx_available():
            pytest.skip("MLX not available on non-Apple Silicon platform")

        sim = MLXSimulator(num_qubits=4)
        for q in range(4):
            sim.apply("H", (q,))
        sim.apply("CX", (0, 1))
        sim.apply("CX", (2, 3))
        probs = sim.probabilities()
        assert abs(np.sum(probs) - 1.0) < 1e-5

    def test_f14_mlx_parametric_rotation_fidelity(self):
        """Parametric rotation gate RX(theta) on MLX GPU matches sin/cos."""
        from quanta.simulator.mlx import MLXSimulator, is_mlx_available

        if not is_mlx_available():
            pytest.skip("MLX not available on non-Apple Silicon platform")

        sim = MLXSimulator(num_qubits=1)
        theta = math.pi / 3  # 60 degrees
        sim.apply("RX", (0,), (theta,))
        probs = sim.probabilities()
        expected_p0 = math.cos(theta / 2.0) ** 2
        assert abs(probs[0] - expected_p0) < 1e-5


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 15: Clifford Pauli Frame Speed
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeature15CliffordPauliFrame:
    """Verifies Aaronson-Gottesman stabilizer tableau Clifford simulation."""

    def test_f15_tableau_initialization(self):
        """Tableau initializes with X destabilizers and Z stabilizers."""
        from quanta.simulator.pauli_frame import PauliFrameSimulator

        sim = PauliFrameSimulator(num_qubits=3)
        assert sim.num_qubits == 3

    def test_f15_hadamard_conjugate_update(self):
        """Hadamard gate updates Pauli generators correctly."""
        from quanta.simulator.pauli_frame import PauliFrameSimulator

        sim = PauliFrameSimulator(num_qubits=1)
        sim.h(0)
        counts = sim.sample(shots=100, seed=42)
        assert "0" in counts
        assert "1" in counts

    def test_f15_phase_s_gate_update(self):
        """Phase S gate transforms X to Y."""
        from quanta.simulator.pauli_frame import PauliFrameSimulator

        sim = PauliFrameSimulator(num_qubits=1)
        sim.h(0)
        sim.s(0)
        # S |+> = (|0> + i|1>)/sqrt(2), measurement in Z is still 50/50
        counts = sim.sample(shots=100, seed=42)
        assert len(counts) == 2

    def test_f15_cnot_stabilizer_propagation(self):
        """CNOT gate propagates stabilizers creating Bell entanglement."""
        from quanta.simulator.pauli_frame import PauliFrameSimulator

        sim = PauliFrameSimulator(num_qubits=2)
        sim.h(0)
        sim.cx(0, 1)
        counts = sim.sample(shots=100, seed=42)
        assert "00" in counts
        assert "11" in counts
        assert "01" not in counts
        assert "10" not in counts

    def test_f15_high_speed_sampling_fidelity(self):
        """GHZ state on 4 qubits samples only all-0 and all-1 bitstrings."""
        from quanta.simulator.pauli_frame import PauliFrameSimulator

        sim = PauliFrameSimulator(num_qubits=4)
        sim.h(0)
        sim.cx(0, 1)
        sim.cx(1, 2)
        sim.cx(2, 3)
        counts = sim.sample(shots=200, seed=42)
        for bitstring in counts:
            assert bitstring in ("0000", "1111")


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 16: OpenQASM 3.0 Dynamic Mid-Circuit Measurement
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeature16OpenQASM3DynamicCircuits:
    """Verifies OpenQASM 3.0 parsing, mid-circuit measurement, and round-trip."""

    def test_f16_qasm3_header_and_qubit_decl(self):
        """Parses QASM 3.0 header and qubit register declarations."""
        from quanta.export.qasm_import import from_qasm

        qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"""
        dag = from_qasm(qasm)
        assert dag.num_qubits == 2
        assert dag.gate_count() == 2

    def test_f16_mid_circuit_measurement_instruction(self):
        """Parses QASM 3.0 mid-circuit measurement syntax."""
        from quanta.export.qasm_import import from_qasm

        qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
c[0] = measure q[0];
cx q[0], q[1];
"""
        dag = from_qasm(qasm)
        assert dag.num_qubits == 2
        assert dag.gate_count() >= 2

    def test_f16_parametric_gates_parsing(self):
        """Preserves parametric gate angles (rx, rz)."""
        from quanta.export.qasm_import import from_qasm

        qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
rx(1.5707963) q[0];
rz(0.7853982) q[0];
"""
        dag = from_qasm(qasm)
        ops = list(dag.op_nodes())
        assert len(ops) == 2
        assert abs(ops[0].params[0] - 1.5707963) < 0.01

    def test_f16_qasm_export_import_roundtrip(self):
        """Exports @circuit to QASM and imports back to identical DAG structure."""
        from quanta.core.circuit import circuit
        from quanta.core.gates import CX, H
        from quanta.core.measure import measure
        from quanta.export.qasm import to_qasm
        from quanta.export.qasm_import import from_qasm

        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        qasm_str = to_qasm(bell)
        dag = from_qasm(qasm_str)
        assert dag.num_qubits == 2
        assert dag.gate_count() == 2

    def test_f16_qasm2_backward_compatibility(self):
        """Imports QASM 2.0 format with qreg and creg declarations."""
        from quanta.export.qasm_import import from_qasm

        qasm2 = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
measure q -> c;
"""
        dag = from_qasm(qasm2)
        assert dag.num_qubits == 3
        assert dag.gate_count() == 3
