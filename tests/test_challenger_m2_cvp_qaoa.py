"""tests/test_challenger_m2_cvp_qaoa.py — Empirical Challenger Test Suite for Milestone M2.

Adversarial stress-testing and empirical verification for:
  - CVP to QUBO and Ising reduction on ill-conditioned / skewed lattices (kappa >= 10^3 to 10^7).
  - Floating-point precision, overflow, and numerical stability bounds (< 10^-12).
  - Dual-mode QAOA parity (vectorized statevector vs native circuit synthesis) on extreme angles.
  - Multi-layer QAOA convergence and ground-state recovery on ill-conditioned lattices.
  - Empirical reproduction of edge-case bugs and failure modes in QAOA input normalization.
"""

from __future__ import annotations

import numpy as np
import pytest

from quanta.dag.dag_circuit import DAGCircuit
from quanta.primitives import _compute_expectation
from quanta.primitives.cvp import CVPHamiltonian
from quanta.primitives.qaoa import QAOA
from quanta.simulator.statevector import StateVectorSimulator


class TestIllConditionedLatticeEquivalence:
    """Stress-tests mathematical equivalence on ill-conditioned and skewed lattices."""

    @pytest.mark.parametrize("kappa", [1e3, 1e4, 1e5, 1e6, 1e7])
    def test_cvp_qubo_ising_equivalence_high_condition_numbers(self, kappa: float) -> None:
        """Verifies ||Bz - t||^2 == x^T Q x + C == Ising energy across kappa in [10^3, 10^7]."""
        rng = np.random.default_rng(int(kappa) % 100000 + 42)
        n = 3
        # Synthesize random matrix with exact condition number kappa via SVD
        u, _ = np.linalg.qr(rng.normal(size=(n, n)))
        v, _ = np.linalg.qr(rng.normal(size=(n, n)))
        s = np.geomspace(1.0, 1.0 / kappa, n)
        b = u @ np.diag(s) @ v.T
        t = rng.normal(size=n)

        cvp = CVPHamiltonian(b, t, bits_per_coord=2)
        pauli_list = cvp.to_pauli_list()
        n_states = 1 << cvp.num_qubits  # 64 states

        max_qubo_err = 0.0
        max_ising_err = 0.0
        max_pauli_err = 0.0

        for idx in range(n_states):
            bitstr = format(idx, f"0{cvp.num_qubits}b")
            x = cvp.bitstring_to_x(bitstr)
            z, _, _ = cvp.decode_sample(bitstr)

            sq_dist = cvp.evaluate_squared_distance(z)
            q_val = cvp.qubo_energy(x)
            is_val = cvp.ising_energy(x)

            psi = np.zeros(n_states, dtype=complex)
            psi[idx] = 1.0
            pauli_val, _ = _compute_expectation(psi, pauli_list)

            max_qubo_err = max(max_qubo_err, abs(sq_dist - q_val))
            max_ising_err = max(max_ising_err, abs(sq_dist - is_val))
            max_pauli_err = max(max_pauli_err, abs(sq_dist - pauli_val))

        # Precision bound: floating-point error must remain below 10^-12
        assert max_qubo_err < 1e-12, (
            f"QUBO error {max_qubo_err:.3e} exceeded 1e-12 at kappa={kappa}"
        )
        assert max_ising_err < 1e-12, (
            f"Ising error {max_ising_err:.3e} exceeded 1e-12 at kappa={kappa}"
        )
        assert max_pauli_err < 1e-12, (
            f"Pauli error {max_pauli_err:.3e} exceeded 1e-12 at kappa={kappa}"
        )

    def test_overdetermined_skewed_lattice_equivalence(self) -> None:
        """Verifies exact equivalence on non-square overdetermined lattice (m=4, n=2)."""
        rng = np.random.default_rng(404)
        m, n = 4, 2
        b = rng.normal(size=(m, n))
        # Skew column 1 to create ill-conditioning
        b[:, 1] = b[:, 0] * 0.999 + rng.normal(scale=1e-4, size=m)
        t = rng.normal(size=m)

        cvp = CVPHamiltonian(b, t, bits_per_coord=3)  # 6 qubits
        assert cvp.dim == 2
        assert cvp.num_qubits == 6

        for idx in range(1 << cvp.num_qubits):
            bitstr = format(idx, f"0{cvp.num_qubits}b")
            x = cvp.bitstring_to_x(bitstr)
            z, _, _ = cvp.decode_sample(bitstr)

            sq_dist = cvp.evaluate_squared_distance(z)
            q_val = cvp.qubo_energy(x)
            is_val = cvp.ising_energy(x)

            assert abs(sq_dist - q_val) < 1e-12
            assert abs(sq_dist - is_val) < 1e-12

    def test_extreme_coordinate_centering_and_offsets(self) -> None:
        """Verifies mathematical stability with large negative z_min and offsets."""
        b = np.array([[5.0, 1.0], [1.0, 4.0]])
        t = np.array([-250.0, 310.0])
        z_min = np.array([-60, 70])
        cvp = CVPHamiltonian(b, t, bits_per_coord=3, z_min=z_min)

        # Scale can be ~10^4, relative error should remain <= 10^-12
        for idx in range(1 << cvp.num_qubits):
            bitstr = format(idx, f"0{cvp.num_qubits}b")
            x = cvp.bitstring_to_x(bitstr)
            z, _, _ = cvp.decode_sample(bitstr)

            sq_dist = cvp.evaluate_squared_distance(z)
            q_val = cvp.qubo_energy(x)
            is_val = cvp.ising_energy(x)

            rel_q_err = abs(sq_dist - q_val) / max(1.0, abs(sq_dist))
            rel_is_err = abs(sq_dist - is_val) / max(1.0, abs(sq_dist))

            assert rel_q_err < 1e-12
            assert rel_is_err < 1e-12


class TestQAOADualModeAndParityStress:
    """Stress-tests QAOA simulation fidelity, circuit synthesis, and convergence."""

    def test_dual_mode_fidelity_parity_extreme_angles(self) -> None:
        """Verifies statevector simulation matches gate-level circuit for arbitrary angles."""
        rng = np.random.default_rng(777)
        b = np.array([[10.0, 1.0], [0.0, 0.01]])  # kappa = 1000
        t = np.array([2.5, 0.005])
        cvp = CVPHamiltonian(b, t, bits_per_coord=2)
        qaoa = QAOA(cvp, p=3)

        # Test across 5 sets of random angles including negative and large magnitudes
        for _ in range(5):
            gamma = rng.uniform(-5.0, 5.0, size=3)
            beta = rng.uniform(-5.0, 5.0, size=3)

            psi_sim = qaoa.simulate_statevector(gamma, beta)
            circ_def = qaoa.to_circuit(gamma, beta)
            dag = DAGCircuit.from_builder(circ_def.build())
            sim = StateVectorSimulator(dag.num_qubits)
            for op in dag.op_nodes():
                sim.apply(op.gate_name, op.qubits, op.params)

            fidelity = abs(np.vdot(psi_sim, sim.state)) ** 2
            assert abs(fidelity - 1.0) < 1e-12, f"Fidelity {fidelity:.16f} deviated from 1.0."

    def test_qaoa_ground_state_recovery_ill_conditioned_lattice(self) -> None:
        """Verifies QAOA finds true ground-state closest vector on a kappa=10^3 lattice."""
        rng = np.random.default_rng(888)
        n = 2
        u, _ = np.linalg.qr(rng.normal(size=(n, n)))
        v, _ = np.linalg.qr(rng.normal(size=(n, n)))
        s = np.array([10.0, 10.0 / 1000.0])  # kappa = 1000
        b = u @ np.diag(s) @ v.T

        z_star = np.array([1, -1])
        t = b @ z_star + np.array([0.005, -0.005])

        cvp = CVPHamiltonian(b, t, bits_per_coord=2, center=z_star)
        qaoa = QAOA(cvp, p=1)
        res = qaoa.run(optimizer="COBYLA", max_iter=100, seed=42, num_starts=5)

        z_opt, _, dist_opt = cvp.decode_sample(res.best_bitstring)
        true_dist = cvp.evaluate_distance(z_star)

        assert np.all(z_opt == z_star)
        assert abs(dist_opt - true_dist) < 1e-12

    def test_qaoa_zeno_quenching_monotonicity(self) -> None:
        """Verifies Zeno quenching keeps multiple quenched qubits pinned to |0>."""
        n_q = 4
        costs = np.arange(1 << n_q, dtype=np.float64)

        # Freeze qubits 0 and 2 (weights 0.0)
        weights = [0.0, 1.0, 0.0, 1.0]
        qaoa = QAOA(costs, num_qubits=n_q, p=2, mixer_weights=weights)

        # Start from |0000>
        init_state = np.zeros(16, dtype=complex)
        init_state[0] = 1.0
        qaoa.initial_state = init_state

        state = qaoa.simulate_statevector(gamma=[0.4, 0.6], beta=[0.3, 0.5])
        probs = np.abs(state) ** 2

        # Qubits 0 and 2 must have zero probability of being 1
        for idx in range(16):
            q0 = (idx >> 3) & 1
            q2 = (idx >> 1) & 1
            if q0 == 1 or q2 == 1:
                assert probs[idx] < 1e-15, f"Quenched qubit flipped at basis index {idx}."


class TestDefectReproductionAndFailureModes:
    """Empirically reproduces and documents edge-case defects in QAOA implementation."""

    def test_defect_1_qubo_matrix_ising_offset_calculation(self) -> None:
        """Demonstrates that QAOA(qubo_matrix) produces an exact _ising_offset."""
        arr = np.array([[2.0, 4.0], [0.0, 6.0]])
        qaoa = QAOA(arr, p=1)

        # For x0, x1 in {0, 1}:
        # QUBO = 2*x0^2 + 6*x1^2 + 4*x0*x1 = 2*x0 + 6*x1 + 4*x0*x1
        # With x = (1 - Z)/2:
        # Expected Ising offset is: 0.5*Tr(Q) + 0.25*Q_01 = 0.5*(8) + 0.25*(4) = 4 + 1 = 5.0
        expected_ising_offset = 5.0
        assert abs(qaoa._ising_offset - expected_ising_offset) < 1e-12, (
            f"Expected Ising offset {expected_ising_offset}, but got {qaoa._ising_offset} "
            f"(deviates by {qaoa._ising_offset - expected_ising_offset})."
        )

    def test_defect_2_initial_params_dimension_validation(self) -> None:
        """Demonstrates that invalid initial_params length is rejected with ValueError."""
        qaoa = QAOA(np.eye(2), p=2)
        # Passing 3 parameters when 2*p = 4 is required should raise ValueError
        with pytest.raises(ValueError, match="initial_params length"):
            qaoa.run(initial_params=np.array([0.1, 0.2, 0.3]), max_iter=5)

    def test_defect_3_non_diagonal_pauli_operator_handling(self) -> None:
        """Demonstrates that non-diagonal and >2-body Pauli terms are rejected."""
        # Non-diagonal cost Hamiltonian is invalid for standard QAOA and should be rejected
        with pytest.raises(ValueError, match="diagonal"):
            QAOA([("XX", 1.0)], p=1)
        # Higher than 2-body terms should also be rejected
        with pytest.raises(ValueError, match="Only up to 2-body"):
            QAOA([("ZZZ", 1.0)], p=1)
