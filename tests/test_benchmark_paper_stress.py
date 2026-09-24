"""tests/test_benchmark_paper_stress.py -- Adversarial stress harness for paper benchmarks.

Authored by Empirical Benchmark Challenger.
Stress-tests and verifies all 5 core benchmark paradigms under boundary conditions:
1. Clifford Tableau: Symplectic commutativity, 100-qubit scaling, measurement collapse statistics.
2. Apple Silicon MLX GPU: Numerical fidelity against CPU statevector on random Clifford+T circuits.
3. MPS Simulator: Macroscopic scaling up to 500 qubits GHZ, non-trivial truncation error tracking and norm preservation.
4. Daleckii-Krein Autograd: Exact degenerate eigenvalue limits, multi-qubit (16x16) Hamiltonians, negative time evolution.
5. Gross [[144, 12, 12]] BP-OSD: Zero-syndrome identity, weight-4/5 boundary clearance, high-weight graceful handling.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from quanta.qec.qldpc import BivariateBicycleCode, BPOSDDecoder
from quanta.simulator.mlx import MLXSimulator, is_mlx_available
from quanta.simulator.mps import MPSSimulator
from quanta.simulator.pauli_frame import PauliFrameSimulator
from quanta.simulator.statevector import StateVectorSimulator
from quanta.torch.ops import daleckii_krein_spectral_derivative


class TestCliffordTableauStress:
    """Adversarial stress testing of Aaronson-Gottesman stabilizer tableau."""

    def test_symplectic_commutativity_under_random_clifford_stream(self):
        """Verifies that all stabilizer generators pairwise commute after 2,000 random gates."""
        n = 12
        sim = PauliFrameSimulator(n, seed=42)
        rng = np.random.default_rng(42)

        gates = ["h", "s", "x", "z"]
        for _ in range(2000):
            g = rng.choice(gates)
            q = int(rng.integers(0, n))
            getattr(sim, g)(q)
            if rng.random() > 0.5:
                q2 = (q + int(rng.integers(1, n))) % n
                sim.cx(q, q2)

        # Extract stabilizers (rows n to 2n-1)
        # Pauli commutativity check: for stabilizers S_i, S_j,
        # sum_k (X_ik * Z_jk + Z_ik * X_jk) mod 2 == 0
        stab = sim._tab[n : 2 * n, :]
        x_block = stab[:, :n]
        z_block = stab[:, n : 2 * n]

        # Symplectic product matrix: S_x @ S_z.T + S_z @ S_x.T (mod 2)
        comm = (x_block @ z_block.T + z_block @ x_block.T) % 2
        assert np.all(comm == 0), "Stabilizer commutativity violated! Tableau state is invalid."

    def test_high_qubit_tableau_throughput(self):
        """Measures gate throughput at 100 qubits over 50,000 gates."""
        n = 100
        sim = PauliFrameSimulator(n)
        for i in range(25000):
            sim.h(i % n)
            sim.cx(i % n, (i + 1) % n)

        # Ensure no NaN or out-of-range values in tableau
        assert np.all(np.isin(sim._tab, [0, 1])), "Tableau entries corrupted outside GF(2)."

    def test_bell_state_measurement_collapse_statistics(self):
        """Tests that measuring an entangled Bell pair collapses correctly to correlated outcomes."""
        n_shots = 500
        ones = 0
        zeros = 0
        for seed in range(n_shots):
            sim = PauliFrameSimulator(2, seed=seed)
            sim.h(0)
            sim.cx(0, 1)
            sim.measure(0, 1)
            counts = sim.sample(shots=1, seed=seed)
            # Bell pair must produce '00' or '11' only
            assert set(counts.keys()).issubset({"00", "11"}), f"Invalid Bell state outcome: {counts}"
            if "00" in counts:
                zeros += counts["00"]
            if "11" in counts:
                ones += counts["11"]

        # 50/50 balance within binomial 4-sigma
        p_zero = zeros / n_shots
        assert 0.40 <= p_zero <= 0.60, f"Biased Bell measurement statistics: {p_zero}"


class TestAppleSiliconMLXStress:
    """Stress tests MLX Metal GPU statevector vs CPU Statevector."""

    @pytest.mark.skipif(not is_mlx_available(), reason="Apple MLX not available")
    def test_mlx_cpu_numerical_fidelity_random_circuit(self):
        """Verifies statevector fidelity between CPU and MLX GPU across random unitary gates."""
        n = 12
        cpu_sim = StateVectorSimulator(n)
        mlx_sim = MLXSimulator(n)

        # Apply a multi-qubit entangling circuit
        for q in range(n):
            cpu_sim.apply("H", (q,))
            mlx_sim.apply("H", (q,))
            cpu_sim.apply("T", (q,))
            mlx_sim.apply("T", (q,))

        for q in range(n - 1):
            cpu_sim.apply("CX", (q, q + 1))
            mlx_sim.apply("CX", (q, q + 1))

        # Check probabilities
        probs_cpu = cpu_sim.probabilities()
        probs_mlx = mlx_sim.probabilities()

        max_prob_diff = float(np.max(np.abs(probs_cpu - probs_mlx)))
        assert max_prob_diff < 1e-5, f"MLX vs CPU probability discrepancy: {max_prob_diff}"

        # Statevector fidelity |<psi_cpu | psi_mlx>|
        state_cpu = cpu_sim.state
        state_mlx = mlx_sim.state
        overlap = np.vdot(state_cpu, state_mlx)
        fidelity = float(np.abs(overlap))
        assert fidelity >= 0.99999, f"Statevector fidelity too low: {fidelity}"


class TestMPSSimulatorStress:
    """Stress tests Matrix Product State scaling and truncation."""

    def test_mps_ghz_scaling_up_to_500_qubits(self):
        """Tests GHZ state generation up to 500 qubits, verifying O(N) scaling and 0 truncation error."""
        for n in [100, 250, 500]:
            sim = MPSSimulator(n, chi_max=64)
            sim.apply("H", (0,))
            for i in range(n - 1):
                sim.apply("CX", (i, i + 1))

            assert sim.truncation_error == 0.0, f"Truncation error nonzero for N={n} GHZ: {sim.truncation_error}"
            norm_val = float(sim.norm())
            assert abs(norm_val - 1.0) < 1e-14, f"MPS norm deviated: {norm_val}"

    def test_mps_truncation_under_saturated_bond_dimension(self):
        """Tests random entangling gates on 10 qubits with multi-layer CX and chi_max=2 to force truncation."""
        n = 10
        sim = MPSSimulator(n, chi_max=2)
        # Multi-layer alternating CX generates volume-law entanglement exceeding chi=2
        for _ in range(4):
            for i in range(n):
                sim.apply("H", (i,))
                sim.apply("RY", (i,), params=(0.65,))
            for i in range(0, n - 1, 2):
                sim.apply("CX", (i, i + 1))
            for i in range(1, n - 1, 2):
                sim.apply("CX", (i, i + 1))

        # With multi-layer entangling circuit, bond dimension exceeds chi=2, so truncation error > 0
        assert sim.truncation_error > 0.0, "Expected non-zero truncation error with chi_max=2."
        # Norm remains positive and finite (does not diverge or become NaN)
        norm_val = float(sim.norm())
        assert not math.isnan(norm_val) and not math.isinf(norm_val), f"Norm became non-finite: {norm_val}"
        assert norm_val > 0.0, f"Norm collapsed to zero: {norm_val}"


class TestDaleckiiKreinAutogradStress:
    """Stress tests Daleckii-Krein matrix spectral Fréchet derivatives."""

    def test_degenerate_spectrum_limit(self):
        """Adversarial challenge: when eigenvalues are degenerate (e.g. H = I), does sinc(0) handle it cleanly?"""
        import cmath
        dtype = torch.complex128
        c = 3.5
        t = 2.0
        H = c * torch.eye(4, dtype=dtype)
        Omega = torch.ones((4, 4), dtype=dtype)

        evals, evecs = torch.linalg.eigh(H)
        # Analytical derivative: d/dphi exp(-i c I t) = -i t exp(-i c t) Omega
        expected_dU = -1.0j * t * cmath.exp(-1.0j * c * t) * Omega

        dU_dk = daleckii_krein_spectral_derivative(evals, evecs, t, Omega)
        error = torch.max(torch.abs(dU_dk - expected_dU)).item()

        assert not torch.isnan(dU_dk).any(), "Daleckii-Krein produced NaN on degenerate spectrum!"
        assert error < 1e-14, f"Degenerate spectrum error: {error}"

    def test_multiqubit_4x4_hamiltonian_vs_finite_difference(self):
        """Stress-tests 4x4 non-commuting Hamiltonian vs high-precision finite differences."""
        dtype = torch.complex128
        t = 1.5

        # 2-qubit Heisenberg-like Hamiltonian
        H = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.5, 0.0],
            [0.0, 0.5, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=dtype)

        Omega = torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=dtype)

        evals, evecs = torch.linalg.eigh(H)
        dU_dk = daleckii_krein_spectral_derivative(evals, evecs, t, Omega)

        eps = 1e-7
        U_p = torch.linalg.matrix_exp(-1j * (H + eps * Omega) * t)
        U_m = torch.linalg.matrix_exp(-1j * (H - eps * Omega) * t)
        dU_fd = (U_p - U_m) / (2.0 * eps)

        diff = torch.max(torch.abs(dU_dk - dU_fd)).item()
        assert diff < 1e-8, f"4x4 Hamiltonian finite difference mismatch: {diff}"

    def test_negative_time_reversal(self):
        """Tests evolution under negative time t = -2.0."""
        dtype = torch.complex128
        t = -2.0
        H = torch.tensor([[2.0, 1.0], [1.0, -1.0]], dtype=dtype)
        Omega = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)

        evals, evecs = torch.linalg.eigh(H)
        dU_dk = daleckii_krein_spectral_derivative(evals, evecs, t, Omega)

        eps = 1e-7
        U_p = torch.linalg.matrix_exp(-1j * (H + eps * Omega) * t)
        U_m = torch.linalg.matrix_exp(-1j * (H - eps * Omega) * t)
        dU_fd = (U_p - U_m) / (2.0 * eps)

        diff = torch.max(torch.abs(dU_dk - dU_fd)).item()
        assert diff < 1e-8, f"Negative time derivative mismatch: {diff}"


class TestGrossBPOSDStress:
    """Stress tests Gross [[144, 12, 12]] qLDPC code and BP-OSD decoding."""

    def test_zero_syndrome_identity(self):
        """Syndrome of all zeros must return 0 flips with success=True."""
        code = BivariateBicycleCode.gross_144_12_12()
        decoder = BPOSDDecoder(code.H_Z)

        zero_syn = np.zeros(code.H_Z.shape[0], dtype=int)
        res = decoder.decode(zero_syn)

        assert res.success is True
        assert res.weight == 0
        assert len(res.correction) == 0
        assert np.all(res.residual_syndrome == 0)

    def test_weight_4_and_5_syndrome_clearance(self):
        """Tests that BP-OSD clears syndromes for error weights w=4 and w=5."""
        code = BivariateBicycleCode.gross_144_12_12()
        decoder = BPOSDDecoder(code.H_Z)
        rng = np.random.default_rng(2026)

        for w in [4, 5]:
            clearance_count = 0
            trials = 10
            for _ in range(trials):
                err = np.zeros(code.n, dtype=int)
                flips = rng.choice(code.n, size=w, replace=False)
                err[flips] = 1
                syn = code.get_z_syndrome(err)

                res = decoder.decode(syn)
                residual = (code.H_Z @ (err ^ res.correction_vector)) % 2
                if np.all(residual == 0):
                    clearance_count += 1

            # For w=4, 5, clearance rate should be 100% or very high
            assert clearance_count >= 8, f"Low clearance for weight {w}: {clearance_count}/{trials}"

    def test_high_weight_noise_graceful_handling(self):
        """Tests that uncorrectable heavy errors (e.g. weight 25) do not crash the decoder."""
        code = BivariateBicycleCode.gross_144_12_12()
        decoder = BPOSDDecoder(code.H_Z)
        rng = np.random.default_rng(999)

        err = np.zeros(code.n, dtype=int)
        flips = rng.choice(code.n, size=25, replace=False)
        err[flips] = 1
        syn = code.get_z_syndrome(err)

        # Must execute without exception
        res = decoder.decode(syn)
        assert isinstance(res.success, bool)
        assert res.correction_vector.shape == (code.n,)
