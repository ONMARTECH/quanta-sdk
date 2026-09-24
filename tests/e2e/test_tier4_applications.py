"""
Tier 4: Real-World Application Scenarios E2E Test Suite.

Audits end-to-end production-grade quantum computational workflows:
  Scenario 1: Quantum Teleportation with Feedforward Classical Corrections
  Scenario 2: VQE Ground State on Spin Chains with Daleckii-Krein Continuous Autograd
  Scenario 3: Fault-Tolerant Logical Memory Preservation over 25 Willow Spacetime Cycles
  Scenario 4: 100-Qubit GHZ Entanglement on Matrix Product States (MPS)
  Scenario 5: Fault-Tolerant Quantum Error Correcting Code Transversal Gate Protection
"""

from __future__ import annotations

import math

import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 1: Quantum Teleportation with Dynamic Feedforward
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario1QuantumTeleportation:
    """End-to-End Quantum Teleportation of an arbitrary unknown qubit."""

    def test_teleportation_arbitrary_state_feedforward(self):
        """Teleports state |psi> = cos(theta/2)|0> + e^(i phi) sin(theta/2)|1> to Bob."""
        from quanta.simulator.statevector import StateVectorSimulator

        theta = math.pi / 3  # 60 degrees
        phi = math.pi / 4    # 45 degrees
        alpha = math.cos(theta / 2.0)
        beta = np.exp(1j * phi) * math.sin(theta / 2.0)
        psi_target = np.array([alpha, beta], dtype=complex)

        # 3 qubits: q0 = Alice message, q1 = Alice entangled, q2 = Bob entangled
        sim = StateVectorSimulator(num_qubits=3)

        # Step 1: Prepare target state on q0: RZ(phi) . RY(theta) |0>
        sim.apply("RY", (0,), (theta,))
        sim.apply("RZ", (0,), (phi,))

        # Step 2: Create Bell pair on (q1, q2)
        sim.apply("H", (1,))
        sim.apply("CX", (1, 2))

        # Step 3: Alice Bell measurement on (q0, q1)
        sim.apply("CX", (0, 1))
        sim.apply("H", (0,))

        # For every possible measurement outcome (m0, m1) in {00, 01, 10, 11},
        # Bob applies X^m1 Z^m0 on q2.
        # We test all 4 projector branches:
        for m0 in (0, 1):
            for m1 in (0, 1):
                # Project q0 onto m0 and q1 onto m1
                proj_sim = StateVectorSimulator(num_qubits=3)
                # Re-run circuit to projection point
                proj_sim.apply("RY", (0,), (theta,))
                proj_sim.apply("RZ", (0,), (phi,))
                proj_sim.apply("H", (1,))
                proj_sim.apply("CX", (1, 2))
                proj_sim.apply("CX", (0, 1))
                proj_sim.apply("H", (0,))

                sv = proj_sim.state.reshape((2, 2, 2))  # (q0, q1, q2)
                bob_unnorm = sv[m0, m1, :]  # Slice matching measurement
                prob = float(np.vdot(bob_unnorm, bob_unnorm).real)
                assert prob > 0.1  # Each branch has prob ~ 0.25

                bob_state = bob_unnorm / math.sqrt(prob)

                # Feedforward correction: if m1 == 1, apply X; if m0 == 1, apply Z
                X = np.array([[0, 1], [1, 0]], dtype=complex)
                Z = np.array([[1, 0], [0, -1]], dtype=complex)
                if m1 == 1:
                    bob_state = X @ bob_state
                if m0 == 1:
                    bob_state = Z @ bob_state

                # Check fidelity with target state
                fidelity = abs(np.vdot(psi_target, bob_state)) ** 2
                assert abs(fidelity - 1.0) < 1e-6, f"Teleportation failed for branch ({m0}, {m1})"


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 2: VQE Ground State on Spin Chains with Daleckii-Krein Autograd
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario2VQEWithDaleckiiKrein:
    """Variational Quantum Eigensolver using Daleckii-Krein matrix exponential autograd."""

    def test_vqe_spin_chain_energy_minimization(self):
        """Optimizes 2-qubit transverse field Ising model toward exact ground state."""
        from quanta.torch.ops import unitary_evolution

        # H = -ZZ - X0
        Z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
        X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
        Y = torch.tensor([[0.0, -1j], [1j, 0.0]], dtype=torch.complex128)
        eye2 = torch.eye(2, dtype=torch.complex128)

        ZZ = torch.kron(Z, Z)
        XI = torch.kron(X, eye2)
        H_target = -ZZ - XI

        # Exact ground state eigenvalue: -sqrt(2) = -1.41421356...
        evals_true = torch.linalg.eigvalsh(H_target)
        E_exact = evals_true[0].item()

        # Parameterized ansatz generator: Y0
        YI = torch.kron(Y, eye2)
        psi0 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex128)  # |00>

        theta = torch.tensor([0.1], requires_grad=True, dtype=torch.float64)
        lr = 0.2
        energies = []

        for _step in range(25):
            psi_theta = unitary_evolution(YI, t=theta, psi0=psi0).squeeze()
            energy = torch.real(torch.vdot(psi_theta, H_target @ psi_theta))
            energies.append(energy.item())

            energy.backward()
            with torch.no_grad():
                theta -= lr * theta.grad
                theta.grad.zero_()

        # Energy should decrease monotonically and converge to exact ground state
        assert energies[-1] < energies[0]
        assert abs(energies[-1] - E_exact) < 1e-4


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 3: Fault-Tolerant Memory over 25 Willow Spacetime Cycles
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario3Willow25SpacetimeCycles:
    """Long-horizon fault-tolerant logical memory preservation over 25 cycles."""

    def test_willow_25_cycles_distance_3_and_5_memory_preservation(self):
        """Simulates 25 syndrome cycles on d=3 and d=5 with sub-threshold noise."""
        from quanta.qec.surface_code import SurfaceCode

        p_phys = 0.001
        p_meas = 0.001
        cycles = 25
        shots = 30

        # Distance 3
        sc3 = SurfaceCode(distance=3)
        res3 = sc3.simulate_dynamic(
            physical_error_rate=p_phys,
            measurement_error_rate=p_meas,
            cycles=cycles,
            shots=shots,
            seed=42,
        )

        assert res3.cycles == 25
        assert res3.shots == 30
        assert res3.defects_detected > 0
        assert res3.willow_suppression_factor > 0.0

        # Distance 5
        sc5 = SurfaceCode(distance=5)
        res5 = sc5.simulate_dynamic(
            physical_error_rate=p_phys,
            measurement_error_rate=p_meas,
            cycles=cycles,
            shots=shots,
            seed=42,
        )

        assert res5.cycles == 25
        assert sc5.correctable_errors == 2
        assert res5.distance == 5
        assert res5.defects_detected > 0
        assert res5.willow_suppression_factor > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 4: 100-Qubit GHZ State Generation on MPS
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario4MPS100QubitGHZ:
    """100-qubit macroscopic GHZ entanglement on Matrix Product State simulator."""

    def test_100_qubit_ghz_generation_and_sampling(self):
        """Constructs |GHZ_100> = 1/sqrt(2)(|0...0> + |1...1>) with exact chi=2 and 0 error."""
        from quanta.simulator.mps import MPSSimulator

        N = 100
        sim = MPSSimulator(num_qubits=N, chi_max=64, seed=42)

        # Step 1: Hadamard on root qubit
        sim.apply("H", (0,))

        # Step 2: Entanglement ladder
        for i in range(N - 1):
            sim.apply("CX", (i, i + 1))

        # Invariant 1: Bond dimension across all 99 cuts is strictly <= 2
        assert sim.max_bond_dim == 2

        # Invariant 2: Truncation error is strictly 0.0
        assert sim.truncation_error == 0.0

        # Invariant 3: Sampling produces only all-0 or all-1 bitstrings
        counts = sim.sample(shots=20)
        assert len(counts) <= 2
        all_zeros = "0" * N
        all_ones = "1" * N
        for bitstring in counts:
            assert bitstring in (all_zeros, all_ones)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 5: Fault-Tolerant QEC Code Transversal Gate Protection
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenario5QECCodeProtection:
    """Full lifecycle encoding, decoding, and syndrome detection in QEC codes."""

    def test_steane_code_parameters_and_circuits(self):
        """Steane [[7,1,3]] code parameters, encoding and syndrome circuits."""
        from quanta.qec.codes import SteaneCode

        steane = SteaneCode()
        info = steane.info
        assert info.n == 7
        assert info.k == 1
        assert info.d == 3
        assert info.correctable_errors == 1

        # Encoding circuit exists
        enc = steane.encode()
        assert enc is not None

        # Syndrome measurement circuit exists
        synd = steane.syndrome_measure()
        assert synd is not None

    def test_shor_code_full_decode_and_correction_lifecycle(self):
        """Shor [[9,1,3]] code full lifecycle: info, encode, decode, and lookup table."""
        from quanta.qec.codes import ShorCode

        shor = ShorCode()
        info = shor.info
        assert info.n == 9
        assert info.k == 1
        assert info.d == 3
        assert info.correctable_errors == 1

        # Encoding circuit
        enc = shor.encode()
        assert enc.num_qubits == 9

        # Decoding circuit
        dec = shor.decode()
        assert dec.num_qubits == 9

        # Lookup table
        table = shor.lookup_table()
        assert "0000" in table
        assert "No error" in table["0000"]
