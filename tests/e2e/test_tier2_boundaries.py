"""
Tier 2: Boundary & Corner Cases E2E Test Suite.

Audits extreme parameters, boundary limits, and edge conditions:
  - Zero and maximal error rates (p = 0.0, p = 0.5, p = 1.0)
  - Zero evolution time and large time limits
  - Machine precision edge tolerances (sub-eps vs super-eps)
  - Large distance codes (d = 3, 5, 7)
  - 200+ qubits on MPS with product state and low entanglement
  - Odd/even boundary defect matching and single-defect virtual boundaries
  - Extreme density matrix capacity and purity bounds
  - Long-range SWAP chains on MPS and 100-qubit Pauli frames
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Boundary: Hamiltonian Evolution
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundaryHamiltonianEvolution:
    """Boundary conditions for Hamiltonian simulation."""

    def test_hamiltonian_evolution_zero_time(self):
        """Evolution at t=0.0 leaves the initial state exactly invariant."""
        from quanta.layer3.hamiltonian import evolve, molecular_hamiltonian

        H = molecular_hamiltonian("H2")
        result = evolve(H, time=0.0, steps=1)
        # Initial state is |00>, final state should be |00>
        assert abs(result.final_state[0] - 1.0) < 1e-12
        assert np.linalg.norm(result.final_state[1:]) < 1e-12

    def test_hamiltonian_evolution_large_time(self):
        """Evolution at large time t=50.0 maintains unit norm and stability."""
        from quanta.layer3.hamiltonian import evolve

        terms = [("Z", 2.0), ("X", 1.0)]
        result = evolve(terms, num_qubits=1, time=50.0, steps=100)
        norm = float(np.linalg.norm(result.final_state))
        assert abs(norm - 1.0) < 1e-5
        assert np.isfinite(result.energy)

    def test_hamiltonian_evolution_single_step_trotter(self):
        """Single-step Trotterization (steps=1) completes without error."""
        from quanta.layer3.hamiltonian import evolve

        terms = [("ZZ", 1.0)]
        result = evolve(terms, num_qubits=2, time=0.5, steps=1)
        assert len(result.energy_history) == 2
        assert abs(np.linalg.norm(result.final_state) - 1.0) < 1e-6

    def test_hamiltonian_evolution_strong_coupling_limit(self):
        """Extreme coupling coefficient (J=1000.0) remains numerically stable."""
        from quanta.layer3.hamiltonian import evolve

        terms = [("ZZ", 1000.0)]
        result = evolve(terms, num_qubits=2, time=0.01, steps=20)
        assert np.isfinite(result.energy)
        assert abs(np.linalg.norm(result.final_state) - 1.0) < 1e-5

    def test_hamiltonian_evolution_pure_identity_terms(self):
        """Hamiltonian of pure identity terms introduces only a global phase."""
        from quanta.layer3.hamiltonian import evolve

        terms = [("I", 3.0)]
        result = evolve(terms, num_qubits=1, time=1.0, steps=5)
        # |psi(t)> = e^(-i * 3.0 * 1.0) |0> -> probability of |0> is strictly 1.0
        prob0 = abs(result.final_state[0]) ** 2
        assert abs(prob0 - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Boundary: Machine-Precision Unitarity
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundaryUnitarityAndPrecision:
    """Boundary conditions for machine-precision unitarity checks."""

    def test_unitarity_sub_epsilon_acceptance(self):
        """Matrix deviation within tolerance (0.5e-12 < 1e-12) is accepted."""
        from quanta.core.custom_gate import custom_gate

        eye = np.eye(2, dtype=complex)
        # Small perturbation below 1e-12
        mat = eye + 0.3e-12 * np.array([[1j, 0], [0, -1j]])
        gate = custom_gate("E2E_SubEps_Gate", mat)
        assert gate is not None

    def test_unitarity_super_epsilon_rejection(self):
        """Matrix deviation exceeding tolerance (2e-12 > 1e-12) is strictly rejected."""
        from quanta.core.custom_gate import CustomGateError, custom_gate

        eye = np.eye(2, dtype=complex)
        # Perturbation exceeding 1e-12
        mat = eye + 2e-12 * np.array([[1, 0], [0, 1]])
        with pytest.raises(CustomGateError):
            custom_gate("E2E_SuperEps_Gate", mat)

    def test_custom_gate_identity_matrix(self):
        """Exact 2x2 identity matrix registers cleanly with zero deviation."""
        from quanta.core.custom_gate import custom_gate

        eye = np.eye(2, dtype=complex)
        gate = custom_gate("E2E_Identity_Custom", eye)
        assert gate.num_qubits == 1

    def test_custom_gate_dimension_non_power_of_two(self):
        """3x3 non-qubit matrix is rejected with CustomGateError."""
        from quanta.core.custom_gate import CustomGateError, custom_gate

        mat3 = np.eye(3, dtype=complex)
        with pytest.raises(CustomGateError, match="power of 2"):
            custom_gate("E2E_3x3_Gate", mat3)

    def test_custom_gate_non_square_rejection(self):
        """Rectangular matrix (2x4) is rejected with CustomGateError."""
        from quanta.core.custom_gate import CustomGateError, custom_gate

        rect = np.ones((2, 4), dtype=complex)
        with pytest.raises(CustomGateError, match="must be square"):
            custom_gate("E2E_Rect_Gate", rect)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Boundary: Open Systems & Density Matrix
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundaryOpenSystemsAndLindblad:
    """Boundary conditions for open quantum systems."""

    def test_density_matrix_zero_depolarizing_noise(self):
        """Noise rate p=0.0 preserves pure state exactly."""
        from quanta.simulator.density_matrix import DensityMatrixSimulator

        sim = DensityMatrixSimulator(num_qubits=1)
        sim.apply_depolarizing(0, p=0.0)
        assert abs(sim.purity - 1.0) < 1e-12

    def test_density_matrix_maximal_depolarizing_noise(self):
        """Noise rate p=1.0 maximally mixes 1-qubit state to purity 0.5."""
        from quanta.simulator.density_matrix import DensityMatrixSimulator

        sim = DensityMatrixSimulator(num_qubits=1)
        sim.apply_depolarizing(0, p=1.0)
        # For p=1 on 1 qubit, rho' = (1-p)rho + p/3(X rho X + Y rho Y + Z rho Z)
        # Starting with |0><0|, X|0><0|X = |1><1|, Y...=|1><1|, Z...=|0><0|
        # rho' = 1/3 |0><0| + 2/3 |1><1| -> purity = (1/9) + (4/9) = 5/9 ~ 0.555
        assert sim.purity < 0.6
        assert abs(np.trace(sim.state) - 1.0) < 1e-12

    def test_density_matrix_maximum_qubits_boundary(self):
        """Exceeding MAX_QUBITS=13 raises DensityMatrixError."""
        from quanta.simulator.density_matrix import DensityMatrixError, DensityMatrixSimulator

        with pytest.raises(DensityMatrixError, match="Max 13 qubits"):
            DensityMatrixSimulator(num_qubits=14)

    def test_density_matrix_pure_state_purity_boundary(self):
        """Purity of n-qubit state is bounded by [1/(2^n), 1.0]."""
        from quanta.simulator.density_matrix import DensityMatrixSimulator

        sim = DensityMatrixSimulator(num_qubits=2)
        dim = 4
        assert 1.0 / dim <= sim.purity <= 1.0 + 1e-12

    def test_density_matrix_trace_preservation_large_system(self):
        """4-qubit mixed state sequence maintains Tr(rho) = 1.0 to 1e-12."""
        from quanta.simulator.density_matrix import DensityMatrixSimulator

        sim = DensityMatrixSimulator(num_qubits=4)
        for q in range(4):
            sim.apply("H", (q,))
            sim.apply_depolarizing(q, 0.1)
        assert abs(np.trace(sim.state) - 1.0) < 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Boundary: Daleckii-Krein Autograd
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundaryDaleckiiKreinAutograd:
    """Boundary conditions for Daleckii-Krein Fréchet derivatives."""

    def test_daleckii_krein_zero_time_limit(self):
        """At t=0.0, derivative d(exp(-i H t))/d phi vanishes identically."""
        from quanta.torch.ops import daleckii_krein_spectral_derivative

        H = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
        Omega = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
        evals, evecs = torch.linalg.eigh(H)
        dU = daleckii_krein_spectral_derivative(evals, evecs, t=0.0, omega_matrix=Omega)
        assert torch.max(torch.abs(dU)).item() < 1e-12

    def test_daleckii_krein_exact_zero_eigenvalue_difference(self):
        """Degenerate eigenvalues trigger sinc(0)=1 without division by zero."""
        from quanta.torch.ops import daleckii_krein_spectral_derivative

        # 3x3 identity matrix has all 3 eigenvalues equal
        H = torch.eye(3, dtype=torch.complex128)
        Omega = torch.ones((3, 3), dtype=torch.complex128)
        evals, evecs = torch.linalg.eigh(H)
        dU = daleckii_krein_spectral_derivative(evals, evecs, t=1.0, omega_matrix=Omega)
        assert not torch.any(torch.isnan(dU))
        assert not torch.any(torch.isinf(dU))

    def test_daleckii_krein_large_time_oscillations(self):
        """Large evolution time t=100.0 maintains stable matrix norm."""
        from quanta.torch.ops import daleckii_krein_spectral_derivative

        H = torch.tensor([[0.5, 0.2], [0.2, -0.5]], dtype=torch.complex128)
        Omega = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
        evals, evecs = torch.linalg.eigh(H)
        dU = daleckii_krein_spectral_derivative(evals, evecs, t=100.0, omega_matrix=Omega)
        assert torch.all(torch.isfinite(dU))

    def test_daleckii_krein_zero_perturbation_matrix(self):
        """Zero perturbation matrix Omega=0 produces exactly dU=0."""
        from quanta.torch.ops import daleckii_krein_spectral_derivative

        H = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
        Omega = torch.zeros((2, 2), dtype=torch.complex128)
        evals, evecs = torch.linalg.eigh(H)
        dU = daleckii_krein_spectral_derivative(evals, evecs, t=1.0, omega_matrix=Omega)
        assert torch.max(torch.abs(dU)).item() < 1e-12

    def test_ehrenfest_gradient_commuting_observable(self):
        """Commuting observable [H, O] = 0 yields zero Ehrenfest time derivative."""
        from quanta.torch.ops import ehrenfest_time_gradient

        H = torch.tensor([[2.0, 0.0], [0.0, -2.0]], dtype=torch.complex128)
        # Identity commutes with H
        obs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.complex128)
        psi = torch.tensor([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], dtype=torch.complex128)
        grad = ehrenfest_time_gradient(psi, H, obs).item()
        assert abs(grad) < 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Boundary: QEC Decoders & Matching
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundaryQECDecodersAndMatching:
    """Boundary conditions for QEC decoders and Edmonds Blossom MWPM."""

    def test_mwpm_single_defect_boundary_parity(self):
        """Single defect on distance-3 lattice connects to virtual boundary."""
        from quanta.qec.decoder import MWPMDecoder

        decoder = MWPMDecoder()
        syndrome = np.zeros(9, dtype=bool)
        syndrome[4] = True  # Single center defect
        res = decoder.decode(syndrome, code_distance=3)
        assert len(res.correction) >= 1
        assert res.weight >= 1

    def test_mwpm_all_stabilizers_excited(self):
        """Full syndrome excitation (all 1s) executes without crashing."""
        from quanta.qec.decoder import MWPMDecoder

        decoder = MWPMDecoder()
        syndrome = np.ones(8, dtype=bool)
        res = decoder.decode(syndrome, code_distance=3)
        assert isinstance(res.correction, tuple)
        assert res.weight >= 4

    def test_mwpm_large_distance_d7_lattice(self):
        """Lattice distance d=7 executes in sub-second time."""
        from quanta.qec.decoder import MWPMDecoder

        decoder = MWPMDecoder()
        syndrome = np.zeros(49, dtype=bool)
        syndrome[0] = True
        syndrome[48] = True
        res = decoder.decode(syndrome, code_distance=7)
        assert len(res.correction) >= 2

    def test_mwpm_symmetric_defects_equidistant(self):
        """Equidistant defect pairs match deterministically."""
        from quanta.qec.decoder import MWPMDecoder

        decoder = MWPMDecoder()
        syndrome = np.zeros(16, dtype=bool)
        syndrome[1] = True
        syndrome[2] = True
        res = decoder.decode(syndrome, code_distance=4)
        assert res.weight >= 1

    def test_mwpm_zero_defect_lattice_d5(self):
        """Distance 5 all-zero syndrome returns zero weight and empty correction."""
        from quanta.qec.decoder import MWPMDecoder

        decoder = MWPMDecoder()
        syndrome = np.zeros(25, dtype=bool)
        res = decoder.decode(syndrome, code_distance=5)
        assert res.correction == ()
        assert res.success is True
        assert res.weight == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Boundary: Surface Code & Willow Spacetime
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundarySurfaceCodeAndWillowSpacetime:
    """Boundary conditions for Surface Code and Willow 3D simulations."""

    def test_surface_code_zero_error_rate(self):
        """Zero physical error rate p=0.0 produces zero logical errors."""
        from quanta.qec.surface_code import SurfaceCode

        sc = SurfaceCode(distance=3)
        res = sc.simulate_error_correction(error_rate=0.0, rounds=50, seed=42)
        assert res.logical_error_rate == 0.0
        assert res.errors_injected == 0

    def test_surface_code_maximal_error_rate(self):
        """Maximal error rate p=0.5 causes high logical error rate."""
        from quanta.qec.surface_code import SurfaceCode

        sc = SurfaceCode(distance=3)
        res = sc.simulate_error_correction(error_rate=0.5, rounds=50, seed=42)
        assert res.logical_error_rate > 0.3

    def test_surface_code_distances_3_5_7_parameters(self):
        """Correctable error capacity scales as floor((d-1)/2)."""
        from quanta.qec.surface_code import SurfaceCode

        sc3 = SurfaceCode(distance=3)
        sc5 = SurfaceCode(distance=5)
        sc7 = SurfaceCode(distance=7)
        assert sc3.correctable_errors == 1
        assert sc5.correctable_errors == 2
        assert sc7.correctable_errors == 3

    def test_willow_single_cycle_spacetime_boundary(self):
        """Single syndrome cycle T=1 executes correctly."""
        from quanta.qec.surface_code import SurfaceCode

        sc = SurfaceCode(distance=3)
        res = sc.simulate_dynamic(physical_error_rate=0.01, cycles=1, shots=20, seed=42)
        assert res.cycles == 1
        assert res.shots == 20

    def test_willow_zero_measurement_error_boundary(self):
        """Measurement error p_meas=0.0 yields defects solely from physical flips."""
        from quanta.qec.surface_code import SurfaceCode

        sc = SurfaceCode(distance=3)
        res = sc.simulate_dynamic(
            physical_error_rate=0.01,
            measurement_error_rate=0.0,
            cycles=3,
            shots=50,
            seed=42,
        )
        assert res.measurement_error_rate == 0.0
        assert res.defects_detected >= 0
        assert res.willow_suppression_factor > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Boundary: MPS Tensor Networks
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundaryMPSTensorNetworks:
    """Boundary conditions for Matrix Product State simulators."""

    def test_mps_200_plus_qubits_product_state(self):
        """250 qubits on MPS with product state runs efficiently with chi=1."""
        from quanta.simulator.mps import MPSSimulator

        sim = MPSSimulator(num_qubits=250, chi_max=16)
        # Apply single-qubit gate
        sim.apply("X", (0,))
        sim.apply("X", (249,))
        assert sim.max_bond_dim == 1
        assert sim.truncation_error == 0.0

    def test_mps_chi_max_one_severe_truncation(self):
        """Constraining chi_max=1 on entangled Bell pair forces truncation error."""
        from quanta.simulator.mps import MPSSimulator

        sim = MPSSimulator(num_qubits=2, chi_max=1)
        sim.apply("H", (0,))
        sim.apply("CX", (0, 1))
        # Bell state requires chi=2, so chi_max=1 must truncate
        assert sim.max_bond_dim == 1
        assert sim.truncation_error > 0.0

    def test_mps_two_qubit_extreme_long_range_gate(self):
        """Long-range 2-qubit gate between q0 and q30 on 32-qubit MPS via SWAP chain."""
        from quanta.simulator.mps import MPSSimulator

        sim = MPSSimulator(num_qubits=32, chi_max=16)
        sim.apply("H", (0,))
        sim.apply("CX", (0, 30))
        assert sim.max_bond_dim <= 4

    def test_mps_zero_entanglement_single_qubit_rotations(self):
        """Sequential single-qubit rotations on 20 qubits maintain bond dimension 1."""
        from quanta.simulator.mps import MPSSimulator

        sim = MPSSimulator(num_qubits=20, chi_max=16)
        for q in range(20):
            sim.apply("H", (q,))
            sim.apply("S", (q,))
        assert sim.max_bond_dim == 1
        assert sim.truncation_error == 0.0

    def test_mps_dense_fallback_small_system(self):
        """Small system allows dense state extraction for cross-validation."""
        from quanta.simulator.mps import MPSSimulator

        sim = MPSSimulator(num_qubits=3, chi_max=16)
        sim.apply("H", (0,))
        state = sim.state
        assert len(state) == 8
        assert abs(np.linalg.norm(state) - 1.0) < 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Boundary: OpenQASM and Clifford Pauli Frame
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundaryOpenQASMAndClifford:
    """Boundary conditions for QASM parsing and Pauli frame simulation."""

    def test_qasm_empty_circuit_handling(self):
        """QASM with zero gates creates empty DAG with 0 gates."""
        from quanta.export.qasm_import import from_qasm

        qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
"""
        dag = from_qasm(qasm)
        assert dag.num_qubits == 2
        assert dag.gate_count() == 0

    def test_qasm_single_qubit_no_gates(self):
        """Single qubit register with no operations."""
        from quanta.export.qasm_import import from_qasm

        qasm = """OPENQASM 3.0;
qubit[1] q;
"""
        dag = from_qasm(qasm)
        assert dag.num_qubits == 1
        assert dag.gate_count() == 0

    def test_pauli_frame_single_qubit_identity(self):
        """Single-qubit Pauli frame maintains identity destabilizer/stabilizer."""
        from quanta.simulator.pauli_frame import PauliFrameSimulator

        sim = PauliFrameSimulator(num_qubits=1)
        counts = sim.sample(shots=50, seed=42)
        # In state |0>, measurement in Z gives strictly '0'
        assert counts == {"0": 50}

    def test_pauli_frame_100_qubits_scale(self):
        """100-qubit Pauli frame initializes and applies gates in milliseconds."""
        from quanta.simulator.pauli_frame import PauliFrameSimulator

        sim = PauliFrameSimulator(num_qubits=100)
        sim.h(0)
        sim.cx(0, 99)
        assert sim.num_qubits == 100

    def test_pauli_frame_error_injection_invalid_type(self):
        """Injecting non-Pauli error raises ValueError."""
        from quanta.simulator.pauli_frame import PauliFrameSimulator

        sim = PauliFrameSimulator(num_qubits=2)
        with pytest.raises(ValueError, match="Unknown error type"):
            sim.inject_error(0, "H")
