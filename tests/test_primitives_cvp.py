"""tests/test_primitives_cvp.py — Unit tests for CVP Hamiltonian & QAOA primitive.

Verifies:
  1. Exact mathematical equivalence between Euclidean distance squared ||Bz - t||^2,
     QUBO cost x^T Q x + C, and Ising expectation <psi_z| H_C |psi_z> to machine
     precision (< 10^-12).
  2. Dual-mode simulation fidelity between fast statevector evolution and native
     gate-level circuits.
  3. QAOA optimization performance on CVP instances across dimensions n=2, 3, 4.
  4. Warm-starting, biased non-commuting mixers, and Zeno transverse quenching.
"""

from __future__ import annotations

import numpy as np
import pytest

from quanta.dag.dag_circuit import DAGCircuit
from quanta.primitives import _compute_expectation
from quanta.primitives.cvp import CVPHamiltonian
from quanta.primitives.qaoa import QAOA, QAOAResult
from quanta.simulator.statevector import StateVectorSimulator


class TestCVPHamiltonianFormulation:
    """Verifies algebraic and numerical properties of CVPHamiltonian."""

    def test_qubo_dimension_and_structure(self) -> None:
        """Verifies QUBO matrix shape, upper-triangularity, and qubit counts."""
        b = np.array([[2.0, 0.5], [0.0, 1.5]])
        t = np.array([1.2, 0.8])
        bits = 3
        cvp = CVPHamiltonian(b, t, bits_per_coord=bits)

        assert cvp.dim == 2
        assert cvp.num_qubits == 6
        assert cvp.qubo_matrix.shape == (6, 6)
        # Verify strictly upper-triangular (zeros strictly below diagonal)
        assert np.all(np.tril(cvp.qubo_matrix, k=-1) == 0.0)
        # Gram matrix must be symmetric positive-definite
        assert np.allclose(cvp.gram_matrix, cvp.gram_matrix.T)
        eigvals = np.linalg.eigvalsh(cvp.gram_matrix)
        assert np.all(eigvals > 0)

    def test_exact_numerical_equivalence_n2(self) -> None:
        """Verifies ||Bz - t||^2 == x^T Q x + C == Ising energy across all 2^N states for n=2."""
        rng = np.random.default_rng(101)
        b = rng.normal(size=(2, 2)) + 2.0 * np.eye(2)
        t = rng.normal(size=2)
        cvp = CVPHamiltonian(b, t, bits_per_coord=2)

        pauli_list = cvp.to_pauli_list()
        n_states = 1 << cvp.num_qubits

        max_qubo_err = 0.0
        max_ising_err = 0.0
        max_pauli_err = 0.0

        for idx in range(n_states):
            bitstr = format(idx, f"0{cvp.num_qubits}b")
            x = cvp.bitstring_to_x(bitstr)
            z, _, _ = cvp.decode_sample(bitstr)

            direct_sq = cvp.evaluate_squared_distance(z)
            qubo_val = cvp.qubo_energy(x)
            ising_val = cvp.ising_energy(x)

            psi = np.zeros(n_states, dtype=complex)
            psi[idx] = 1.0
            pauli_val, _ = _compute_expectation(psi, pauli_list)

            max_qubo_err = max(max_qubo_err, abs(direct_sq - qubo_val))
            max_ising_err = max(max_ising_err, abs(direct_sq - ising_val))
            max_pauli_err = max(max_pauli_err, abs(direct_sq - pauli_val))

        assert max_qubo_err < 1e-12, f"QUBO error {max_qubo_err:.3e} exceeds 1e-12."
        assert max_ising_err < 1e-12, f"Ising error {max_ising_err:.3e} exceeds 1e-12."
        assert max_pauli_err < 1e-12, f"Pauli error {max_pauli_err:.3e} exceeds 1e-12."

    def test_exact_numerical_equivalence_n3(self) -> None:
        """Verifies machine precision equivalence across all 64 states for n=3, K=2."""
        rng = np.random.default_rng(202)
        b = rng.normal(size=(3, 3)) + 3.0 * np.eye(3)
        t = rng.normal(size=3)
        cvp = CVPHamiltonian(b, t, bits_per_coord=2, center=np.array([1, 0, -1]))

        n_states = 1 << cvp.num_qubits  # 2^6 = 64
        pauli_list = cvp.to_pauli_list()

        for idx in range(n_states):
            bitstr = format(idx, f"0{cvp.num_qubits}b")
            x = cvp.bitstring_to_x(bitstr)
            z, v, dist = cvp.decode_sample(bitstr)

            direct_dist = cvp.evaluate_distance(z)
            assert abs(dist - direct_dist) < 1e-12

            direct_sq = cvp.evaluate_squared_distance(z)
            qubo_val = cvp.qubo_energy(x)
            ising_val = cvp.ising_energy(x)

            psi = np.zeros(n_states, dtype=complex)
            psi[idx] = 1.0
            pauli_val, _ = _compute_expectation(psi, pauli_list)

            assert abs(direct_sq - qubo_val) < 1e-12
            assert abs(direct_sq - ising_val) < 1e-12
            assert abs(direct_sq - pauli_val) < 1e-12

    def test_exact_numerical_equivalence_n4_sampled(self) -> None:
        """Verifies machine precision equivalence for n=4 across sampled basis states."""
        rng = np.random.default_rng(303)
        b = rng.normal(size=(4, 4)) + 4.0 * np.eye(4)
        t = rng.normal(size=4)
        cvp = CVPHamiltonian(b, t, bits_per_coord=2)

        # 4 * 2 = 8 qubits (256 states)
        assert cvp.num_qubits == 8
        pauli_list = cvp.to_pauli_list()

        sample_indices = rng.choice(256, size=50, replace=False)
        for idx in sample_indices:
            bitstr = format(int(idx), f"0{cvp.num_qubits}b")
            x = cvp.bitstring_to_x(bitstr)
            z, _, dist = cvp.decode_sample(bitstr)

            direct_sq = cvp.evaluate_squared_distance(z)
            qubo_val = cvp.qubo_energy(x)
            ising_val = cvp.ising_energy(x)

            psi = np.zeros(256, dtype=complex)
            psi[idx] = 1.0
            pauli_val, _ = _compute_expectation(psi, pauli_list)

            assert abs(direct_sq - qubo_val) < 1e-12
            assert abs(direct_sq - ising_val) < 1e-12
            assert abs(direct_sq - pauli_val) < 1e-12

    def test_coordinate_centering_and_bounding(self) -> None:
        """Verifies coordinate bounding and center shifts."""
        b = np.eye(3)
        t = np.array([2.5, -1.2, 0.3])
        center = np.array([2, -1, 0])
        cvp = CVPHamiltonian(b, t, bits_per_coord=3, center=center)

        # bits_per_coord = 3 -> span = 8, half_span = 4
        # z_min = center - 4
        assert np.all(cvp.z_min == center - 4)
        assert np.all(cvp.z_max == center + 3)

        # Test encoding and decoding bounds
        assert cvp.z_min is not None
        z_lower = cvp.z_min
        x_lower = cvp.z_to_x(z_lower)
        assert x_lower is not None
        assert np.all(x_lower == 0)

        z_upper = cvp.z_max
        x_upper = cvp.z_to_x(z_upper)
        assert x_upper is not None
        assert np.all(x_upper == 1)

        # Out-of-bounds returns None
        assert cvp.z_to_x(z_lower - 1) is None
        assert cvp.z_to_x(z_upper + 1) is None

    def test_coordinate_roundtrip_exhaustive(self) -> None:
        """Verifies x -> z -> x bijection for all representable integer vectors."""
        b = np.eye(2)
        t = np.zeros(2)
        cvp = CVPHamiltonian(b, t, bits_per_coord=2, center=np.array([0, 0]))

        for idx in range(16):
            bitstr = format(idx, "04b")
            x = cvp.bitstring_to_x(bitstr)
            z = cvp.x_to_z(x)
            x_rec = cvp.z_to_x(z)
            assert x_rec is not None
            assert np.all(x == x_rec)
            assert cvp.x_to_bitstring(x) == bitstr

    def test_input_validation_errors(self) -> None:
        """Verifies robust exception handling for invalid inputs."""
        b = np.eye(3)
        t = np.zeros(2)  # Dimension mismatch (2 != 3)
        with pytest.raises(ValueError, match="Target dimension"):
            CVPHamiltonian(b, t)

        t_valid = np.zeros(3)
        with pytest.raises(ValueError, match="bits_per_coord"):
            CVPHamiltonian(b, t_valid, bits_per_coord=0)

        with pytest.raises(ValueError, match="center length"):
            CVPHamiltonian(b, t_valid, center=np.array([1, 2]))

        with pytest.raises(ValueError, match="center length"):
            CVPHamiltonian(b, t_valid, z_min=np.array([0, 0, 0]), center=np.array([1, 2]))

        cvp = CVPHamiltonian(b, t_valid, bits_per_coord=2)
        with pytest.raises(ValueError, match="Bitstring length"):
            cvp.bitstring_to_x("01")


class TestQAOAPrimitive:
    """Verifies QAOA variational algorithm and dual-mode simulation engine."""

    def test_dual_mode_fidelity_parity(self) -> None:
        """Verifies statevector fidelity parity between fast simulator and native circuit."""
        b = np.array([[2.0, 1.0], [1.0, 3.0]])
        t = np.array([2.5, 3.5])
        cvp = CVPHamiltonian(b, t, bits_per_coord=2)

        qaoa = QAOA(cvp, p=2)
        gamma = np.array([0.42, 0.85])
        beta = np.array([0.31, 0.67])

        # Fast statevector simulation
        psi_sim = qaoa.simulate_statevector(gamma, beta)

        # Gate-level circuit synthesis
        circ_def = qaoa.to_circuit(gamma, beta)
        dag = DAGCircuit.from_builder(circ_def.build())
        sim = StateVectorSimulator(dag.num_qubits)
        for op in dag.op_nodes():
            sim.apply(op.gate_name, op.qubits, op.params)

        psi_circuit = sim.state

        # Compute statevector overlap fidelity: |<psi_sim | psi_circ>|^2
        fidelity = abs(np.vdot(psi_sim, psi_circuit)) ** 2
        assert abs(fidelity - 1.0) < 1e-12, f"Fidelity {fidelity:.16f} deviates from 1.0."

    def test_qaoa_solves_cvp_n2(self) -> None:
        """Verifies QAOA finds the ground-truth closest vector for a 2D lattice."""
        b = np.array([[2.0, 0.0], [0.0, 2.0]])
        # Target closest to [2, 2] = B @ [1, 1]
        t = np.array([2.1, 1.9])
        cvp = CVPHamiltonian(b, t, bits_per_coord=2, center=np.array([1, 1]))

        qaoa = QAOA(cvp, p=1)
        result = qaoa.run(optimizer="COBYLA", max_iter=100, seed=42, num_starts=5)

        z_opt, _, dist_opt = cvp.decode_sample(result.best_bitstring)
        true_dist = cvp.evaluate_distance([1, 1])

        assert np.all(z_opt == np.array([1, 1]))
        assert abs(dist_opt - true_dist) < 1e-12
        assert isinstance(result, QAOAResult)
        assert len(result.counts) > 0
        assert "QAOA Optimization Result" in result.summary()

    def test_qaoa_solves_cvp_n3(self) -> None:
        """Verifies QAOA finds ground truth for a 3D non-orthogonal lattice."""
        b = np.array([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, 2.0]])
        # Ground truth z* = [1, -1, 1]
        z_star = np.array([1, -1, 1])
        v_star = b @ z_star
        # Perturb slightly to form target
        t = v_star + np.array([0.05, -0.08, 0.04])

        cvp = CVPHamiltonian(b, t, bits_per_coord=2, center=z_star)
        qaoa = QAOA(cvp, p=2)
        result = qaoa.run(optimizer="COBYLA", max_iter=80, seed=123, num_starts=5)

        z_opt, _, dist_opt = cvp.decode_sample(result.best_bitstring)
        assert np.all(z_opt == z_star)
        assert dist_opt <= cvp.evaluate_distance(z_star) + 1e-6

    def test_qaoa_solves_cvp_n4(self) -> None:
        """Verifies QAOA optimization on a 4D lattice instance."""
        b = np.diag([2.0, 2.0, 2.0, 2.0])
        z_star = np.array([1, 0, -1, 1])
        t = b @ z_star + np.array([0.1, -0.1, 0.05, -0.05])

        cvp = CVPHamiltonian(b, t, bits_per_coord=2, center=z_star)
        qaoa = QAOA(cvp, p=2)
        result = qaoa.run(optimizer="COBYLA", max_iter=100, seed=777, num_starts=5)

        z_opt, _, dist_opt = cvp.decode_sample(result.best_bitstring)
        assert np.all(z_opt == z_star)
        assert dist_opt < 0.2

    def test_qaoa_multi_layer_ansatz(self) -> None:
        """Verifies multi-layer ansatz execution across p=1, 2, 3."""
        b = np.array([[1.5, 0.5], [0.0, 2.0]])
        t = np.array([1.6, 2.1])
        cvp = CVPHamiltonian(b, t, bits_per_coord=2)

        for p_layers in [1, 2, 3]:
            qaoa = QAOA(cvp, p=p_layers)
            result = qaoa.run(max_iter=30, seed=42, num_starts=2)
            assert len(result.optimal_gamma) == p_layers
            assert len(result.optimal_beta) == p_layers
            assert result.num_layers == p_layers
            assert result.num_qubits == 4

    def test_qaoa_warm_start_initial_state(self) -> None:
        """Verifies warm-start state preparation with biased initial angles."""
        b = np.eye(2) * 2.0
        t = np.array([2.1, 2.1])  # Target near z = [1, 1]
        cvp = CVPHamiltonian(b, t, bits_per_coord=2, center=np.array([1, 1]))

        # Ground truth bitstring for z=[1, 1]:
        # z_min is [-1, -1]. delta = [2, 2] = binary 10 for each coord.
        # bits: coord0=(bit0=0, bit1=1), coord1=(bit0=0, bit1=1) -> '0101'
        x_target = cvp.z_to_x([1, 1])
        assert x_target is not None

        # Warm start angles: theta = 0 for |0>, theta = pi for |1>
        # Regularized to avoid singularities: delta = 0.05
        eps = 0.05
        warm_angles = []
        for bit in x_target:
            prob = 1.0 - eps if bit == 1 else eps
            warm_angles.append(2.0 * np.arcsin(np.sqrt(prob)))

        qaoa = QAOA(cvp, p=1, warm_start_angles=warm_angles)
        result = qaoa.run(max_iter=40, seed=42, num_starts=3)

        # Warm start should find optimal bitstring with high probability
        assert result.best_bitstring == cvp.x_to_bitstring(x_target)
        z_rec, _, _ = cvp.decode_sample(result.best_bitstring)
        assert np.all(z_rec == np.array([1, 1]))

    def test_qaoa_biased_mixer_decomposition(self) -> None:
        """Verifies biased non-commuting mixer RY(-theta) RX(2beta) RY(theta) execution."""
        b = np.eye(2) * 2.0
        t = np.array([2.0, 2.0])
        cvp = CVPHamiltonian(b, t, bits_per_coord=2)

        warm_angles = [np.pi * 0.8, np.pi * 0.2, np.pi * 0.8, np.pi * 0.2]
        qaoa = QAOA(cvp, p=1, mixer_type="biased", warm_start_angles=warm_angles)

        # Parity test between statevector and circuit for biased mixer
        gamma = [0.3]
        beta = [0.5]
        psi_sim = qaoa.simulate_statevector(gamma, beta)
        circ_def = qaoa.to_circuit(gamma, beta)
        dag = DAGCircuit.from_builder(circ_def.build())
        sim = StateVectorSimulator(dag.num_qubits)
        for op in dag.op_nodes():
            sim.apply(op.gate_name, op.qubits, op.params)

        fidelity = abs(np.vdot(psi_sim, sim.state)) ** 2
        assert abs(fidelity - 1.0) < 1e-12

    def test_qaoa_zeno_mixer_quenching(self) -> None:
        """Verifies Quantum Zeno mixer quenching w_i = 0 freezes targeted qubits."""
        n_q = 4
        costs = np.arange(1 << n_q, dtype=np.float64)

        # Quench qubit 0: weight = 0.0 (frozen)
        weights = [0.0, 1.0, 1.0, 1.0]
        qaoa = QAOA(costs, num_qubits=n_q, p=1, mixer_weights=weights)

        # Initial state: |0000>
        init_state = np.zeros(16, dtype=complex)
        init_state[0] = 1.0
        qaoa.initial_state = init_state

        state = qaoa.simulate_statevector(gamma=[0.5], beta=[0.8])
        probs = np.abs(state) ** 2

        # Qubit 0 was initially 0. Since its mixer was quenched (w=0),
        # it must remain strictly in state |0> (i.e. all indices with bit 0 = 1 have probability 0)
        # In our qubit ordering, qubit 0 is MSB of the integer index: (idx >> 3) & 1
        for idx in range(16):
            q0_val = (idx >> 3) & 1
            if q0_val == 1:
                assert probs[idx] < 1e-15, f"Quenched qubit 0 transitioned to 1 at index {idx}."

    def test_qaoa_input_varieties(self) -> None:
        """Verifies QAOA initialization from Pauli list, QUBO matrix, and callable."""
        # 1. From Pauli list
        pauli_obs = [("ZZ", 1.0), ("ZI", -0.5), ("IZ", 0.5), ("II", 2.0)]
        qaoa1 = QAOA(pauli_obs, p=1)
        res1 = qaoa1.run(max_iter=20, seed=42)
        assert res1.num_qubits == 2

        # 2. From QUBO matrix
        q_mat = np.array([[2.0, 1.0], [0.0, 3.0]])
        qaoa2 = QAOA(q_mat, p=1)
        res2 = qaoa2.run(max_iter=20, seed=42)
        assert res2.num_qubits == 2

        # 3. From Callable cost function
        def cost_fn(idx: int) -> float:
            return float(bin(idx).count("1"))

        qaoa3 = QAOA(cost_fn, num_qubits=3, p=1)
        res3 = qaoa3.run(max_iter=20, seed=42)
        assert res3.num_qubits == 3

    def test_qaoa_resampling_and_result_properties(self) -> None:
        """Verifies QAOAResult resampling and metadata consistency."""
        costs = np.array([4.0, 1.0, 3.0, 0.0])
        qaoa = QAOA(costs, num_qubits=2, p=1)
        result = qaoa.run(max_iter=30, seed=42)

        # Resample with 500 shots
        counts_resample = result.sample(shots=500, seed=99)
        assert sum(counts_resample.values()) == 500

        # Best cost should match the best bitstring cost
        best_idx = int(result.best_bitstring, 2)
        assert abs(result.best_cost - costs[best_idx]) < 1e-12

        # Solutions list should be sorted by cost
        for i in range(len(result.all_solutions) - 1):
            assert result.all_solutions[i][1] <= result.all_solutions[i + 1][1]
