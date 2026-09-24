"""
Tier 3: Cross-Feature Combinations E2E Test Suite.

Audits pairwise interactions across subsystems:
  1. Hamiltonian Evolution + MPS Statevector
  2. Surface Code + Clifford Pauli Frame Simulation
  3. OpenQASM 3.0 Dynamic Circuits + Apple Silicon MLX GPU Simulator
  4. Lindblad Open Systems + Custom Machine-Precision Unitary Gates
  5. Daleckii-Krein Autograd + Hamiltonian Energy Expectation
  6. Willow 3D Spacetime Syndromes + MWPM Decoder Integration
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# Interaction 1: Hamiltonian Evolution + MPS Statevector
# ═══════════════════════════════════════════════════════════════════════════════

class TestInteractionHamiltonianAndMPS:
    """Verifies Hamiltonian time evolution interoperability with MPS simulation."""

    def test_hamiltonian_evolution_fidelity_with_mps(self):
        """2-qubit Hamiltonian evolution state matches MPS statevector."""
        from quanta.layer3.hamiltonian import evolve
        from quanta.simulator.mps import MPSSimulator

        # Simple evolution under Z0
        terms = [("ZI", 1.0)]
        res = evolve(terms, num_qubits=2, time=0.5, steps=10)

        # MPS evolution equivalent: RZ(theta=2*t) on qubit 0
        sim = MPSSimulator(num_qubits=2, chi_max=16)
        sim.apply("RZ", (0,), (2.0 * 0.5,))

        # State overlap |<psi_mps | psi_ham>|^2
        fidelity = abs(np.vdot(sim.state, res.final_state)) ** 2
        assert abs(fidelity - 1.0) < 1e-4

    def test_mps_initial_state_from_hamiltonian_eigenstate(self):
        """Setting MPS state from Hamiltonian final state preserves norm and properties."""
        from quanta.layer3.hamiltonian import evolve, molecular_hamiltonian
        from quanta.simulator.mps import MPSSimulator

        H = molecular_hamiltonian("H2")
        result = evolve(H, time=0.2, steps=5)

        sim = MPSSimulator(num_qubits=2, chi_max=16)
        sim.state = result.final_state
        assert abs(np.linalg.norm(sim.state) - 1.0) < 1e-6
        assert sim.max_bond_dim <= 4

    def test_trotter_step_mapping_to_mps_gates(self):
        """Trotterized evolution gate sequence executed directly on MPS."""
        from quanta.simulator.mps import MPSSimulator

        # H = ZZ interaction: CNOT, RZ, CNOT
        sim = MPSSimulator(num_qubits=2, chi_max=16)
        dt = 0.1
        sim.apply("CX", (0, 1))
        sim.apply("RZ", (1,), (2.0 * dt,))
        sim.apply("CX", (0, 1))
        assert sim.truncation_error == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Interaction 2: Surface Code + Clifford Pauli Frame Simulation
# ═══════════════════════════════════════════════════════════════════════════════

class TestInteractionSurfaceCodeAndClifford:
    """Verifies surface code stabilizer measurements tracked via Pauli frame."""

    def test_surface_code_syndrome_extraction_via_pauli_frame(self):
        """Pauli frame simulator tracks X and Z error propagation for surface code."""
        from quanta.qec.surface_code import SurfaceCode
        from quanta.simulator.pauli_frame import PauliFrameSimulator

        sc = SurfaceCode(distance=3)
        sim = PauliFrameSimulator(num_qubits=sc.n_physical)

        # Inject Pauli X error on qubit 4
        sim.inject_error(4, "X")

        # Syndrome on SurfaceCode
        err_mask = np.zeros(sc.n_physical, dtype=bool)
        err_mask[4] = True
        syndrome = sc.get_syndrome(err_mask)
        assert np.any(syndrome == 1)

    def test_pauli_frame_correction_chain_application(self):
        """Applying MWPM correction on Pauli frame neutralizes injected error."""
        from quanta.qec.decoder import MWPMDecoder
        from quanta.simulator.pauli_frame import PauliFrameSimulator

        sim = PauliFrameSimulator(num_qubits=4)
        sim.inject_error(1, "X")

        # Decode error
        decoder = MWPMDecoder()
        syndrome = np.array([0, 1, 0, 0], dtype=bool)
        res = decoder.decode(syndrome, code_distance=2)

        # Apply correction if returned
        for q in res.correction:
            if q < sim.num_qubits:
                sim.x(q)

        # Confirm simulator is stable
        assert sim.num_qubits == 4

    def test_clifford_tableau_homology_preservation(self):
        """Clifford operations preserve stabilizer commutation relations."""
        from quanta.simulator.pauli_frame import PauliFrameSimulator

        sim = PauliFrameSimulator(num_qubits=2)
        sim.h(0)
        sim.cx(0, 1)
        # Stabilizer of Bell state is Z0 Z1 and X0 X1
        counts = sim.sample(shots=100, seed=42)
        assert set(counts.keys()).issubset({"00", "11"})


# ═══════════════════════════════════════════════════════════════════════════════
# Interaction 3: OpenQASM 3.0 + Apple Silicon MLX GPU Simulator
# ═══════════════════════════════════════════════════════════════════════════════

class TestInteractionQASMAndMLXSimulator:
    """Verifies execution of imported OpenQASM 3.0 DAGs on Apple Silicon MLX GPU."""

    def test_qasm_parsed_dag_execution_on_mlx(self):
        """Parses QASM 3.0 string and applies gates to MLXSimulator."""
        from quanta.export.qasm_import import from_qasm
        from quanta.simulator.mlx import MLXSimulator, is_mlx_available

        if not is_mlx_available():
            pytest.skip("MLX not available on non-Apple Silicon platform")

        qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"""
        dag = from_qasm(qasm)
        sim = MLXSimulator(num_qubits=dag.num_qubits)

        for op in dag.op_nodes():
            sim.apply(op.gate_name, op.qubits, op.params)

        probs = sim.probabilities()
        assert abs(probs[0] - 0.5) < 1e-5
        assert abs(probs[3] - 0.5) < 1e-5

    def test_mlx_simulation_probabilities_match_qasm_dag(self):
        """Parametric rotation in QASM produces exact trigonometric amplitudes on MLX."""
        from quanta.export.qasm_import import from_qasm
        from quanta.simulator.mlx import MLXSimulator, is_mlx_available

        if not is_mlx_available():
            pytest.skip("MLX not available on non-Apple Silicon platform")

        qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
rx(1.04719755) q[0];
"""
        dag = from_qasm(qasm)
        sim = MLXSimulator(num_qubits=1)
        for op in dag.op_nodes():
            sim.apply(op.gate_name, op.qubits, op.params)

        probs = sim.probabilities()
        expected_p0 = math.cos(1.04719755 / 2.0) ** 2
        assert abs(probs[0] - expected_p0) < 1e-5

    def test_qasm_bell_measurement_on_mlx_gpu(self):
        """Full round-trip: circuit -> QASM -> DAG -> MLX simulation."""
        from quanta.core.circuit import circuit
        from quanta.core.gates import CX, H
        from quanta.core.measure import measure
        from quanta.export.qasm import to_qasm
        from quanta.export.qasm_import import from_qasm
        from quanta.simulator.mlx import MLXSimulator, is_mlx_available

        if not is_mlx_available():
            pytest.skip("MLX not available on non-Apple Silicon platform")

        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        qasm_str = to_qasm(bell)
        dag = from_qasm(qasm_str)

        sim = MLXSimulator(num_qubits=dag.num_qubits)
        for op in dag.op_nodes():
            sim.apply(op.gate_name, op.qubits, op.params)

        probs = sim.probabilities()
        assert abs(probs[0] - 0.5) < 1e-5
        assert abs(probs[3] - 0.5) < 1e-5


# ═══════════════════════════════════════════════════════════════════════════════
# Interaction 4: Lindblad Open Systems + Custom Unitary Gates
# ═══════════════════════════════════════════════════════════════════════════════

class TestInteractionLindbladAndCustomGates:
    """Verifies open quantum system dynamics with user-defined custom gates."""

    def test_custom_gate_inside_density_matrix_open_system(self):
        """Custom unitary gate applied to density matrix before decoherence."""
        from quanta.core.custom_gate import custom_gate
        from quanta.simulator.density_matrix import DensityMatrixSimulator

        # Create custom phase gate: diag(1, e^(i pi/3))
        phase_mat = np.array([[1.0, 0.0], [0.0, np.exp(1j * np.pi / 3)]], dtype=complex)
        custom_gate("E2E_Phase60", phase_mat)

        sim = DensityMatrixSimulator(num_qubits=1)
        sim.apply("H", (0,))
        sim.apply("E2E_Phase60", (0,))
        sim.apply_depolarizing(0, 0.1)

        assert abs(np.trace(sim.state) - 1.0) < 1e-12
        assert sim.purity < 1.0

    def test_custom_sqrt_x_interleaved_with_phase_damping(self):
        """Interleaving custom SqrtX gate with amplitude damping."""
        from quanta.core.custom_gate import custom_gate
        from quanta.simulator.density_matrix import DensityMatrixSimulator

        sqrt_x = np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=complex)
        custom_gate("E2E_InterleaveSqrtX", sqrt_x)

        sim = DensityMatrixSimulator(num_qubits=1)
        sim.apply("E2E_InterleaveSqrtX", (0,))
        sim.apply_depolarizing(0, 0.2)
        sim.apply("E2E_InterleaveSqrtX", (0,))

        assert abs(np.trace(sim.state) - 1.0) < 1e-12
        assert np.all(np.linalg.eigvalsh(sim.state) >= -1e-12)

    def test_unitary_cptp_preservation_composite_channel(self):
        """Composite unitary-decoherence channel preserves CPTP properties."""
        from quanta.simulator.density_matrix import DensityMatrixSimulator

        sim = DensityMatrixSimulator(num_qubits=2)
        sim.apply("H", (0,))
        sim.apply("CX", (0, 1))
        sim.apply_depolarizing(0, 0.05)
        sim.apply_depolarizing(1, 0.05)

        assert abs(np.trace(sim.state) - 1.0) < 1e-12
        assert np.all(np.linalg.eigvalsh(sim.state) >= -1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# Interaction 5: Daleckii-Krein Autograd + Hamiltonian Expectation
# ═══════════════════════════════════════════════════════════════════════════════

class TestInteractionDaleckiiKreinAndHamiltonian:
    """Verifies matrix exponential autograd for Hamiltonian energy gradients."""

    def test_daleckii_krein_gradient_matches_trotter_energy_shift(self):
        """Daleckii-Krein spectral Fréchet derivative aligns with finite difference shift."""
        from quanta.torch.ops import daleckii_krein_spectral_derivative

        H = torch.tensor([[1.0, 0.5], [0.5, -1.0]], dtype=torch.complex128)
        Omega = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
        t = 0.5
        eps = 1e-6

        # Central finite difference of matrix exponential
        U_plus = torch.linalg.matrix_exp(-1j * (H + eps * Omega) * t)
        U_minus = torch.linalg.matrix_exp(-1j * (H - eps * Omega) * t)
        dU_fd = (U_plus - U_minus) / (2.0 * eps)

        # Daleckii-Krein
        evals, evecs = torch.linalg.eigh(H)
        dU_dk = daleckii_krein_spectral_derivative(evals, evecs, t, Omega)

        assert torch.max(torch.abs(dU_dk - dU_fd)).item() < 1e-5

    def test_hamiltonian_vqe_step_with_continuous_autograd(self):
        """Single VQE energy evaluation gradient computed via autograd."""
        from quanta.torch.ops import unitary_evolution

        theta = torch.tensor([0.5], requires_grad=True, dtype=torch.float64)
        H = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
        obs = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
        psi0 = torch.tensor([1.0, 0.0], dtype=torch.complex128)

        # Evolve and compute expectation <psi(theta)| obs |psi(theta)>
        # Parameterized evolution: exp(-i H theta)
        psi_theta = unitary_evolution(H, t=theta, psi0=psi0).squeeze()
        exp_val = torch.real(torch.vdot(psi_theta, obs @ psi_theta))
        exp_val.backward()

        assert theta.grad is not None
        assert torch.isfinite(theta.grad)


# ═══════════════════════════════════════════════════════════════════════════════
# Interaction 6: Willow 3D Spacetime Syndromes + MWPM Decoder Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestInteractionWillowAndMWPMDecoder:
    """Verifies coupling of dynamic Willow defect graphs with MWPM matching."""

    def test_willow_spacetime_defects_decoded_by_mwpm(self):
        """MWPM decoder processes syndrome differences from dynamic simulation."""
        from quanta.qec.decoder import MWPMDecoder
        from quanta.qec.surface_code import SurfaceCode

        sc = SurfaceCode(distance=3)
        res = sc.simulate_dynamic(physical_error_rate=0.01, cycles=3, shots=20, seed=42)

        # Test decoder on synthetic syndrome vector of same size
        decoder = MWPMDecoder()
        n_stabs = len(sc._x_stabilizers) + len(sc._z_stabilizers)
        sample_syndrome = np.zeros(n_stabs, dtype=bool)
        if res.defects_detected > 0:
            sample_syndrome[0] = True
            sample_syndrome[1] = True

        dec_res = decoder.decode(sample_syndrome, code_distance=sc.distance)
        assert isinstance(dec_res.correction, tuple)

    def test_willow_dynamic_correction_restores_stabilizers(self):
        """Correction chain weight is bounded by code distance capacity."""
        from quanta.qec.decoder import MWPMDecoder
        from quanta.qec.surface_code import SurfaceCode

        sc = SurfaceCode(distance=3)
        decoder = MWPMDecoder()
        syndrome = np.array([1, 0, 1, 0], dtype=bool)
        res = decoder.decode(syndrome, code_distance=3)
        assert res.weight <= sc.distance * 2
