"""tests/test_dialectical_adversarial.py — Empirical Challenger Adversarial Test Suite.

Adversarially stress-tests dialectical frontiers, biophysical parameters,
effective qubit capacity (Thm 6), non-classical contextuality & Bell-CHSH (Thm 7),
Quantum Question Order (QQO) equality, and thermal decoherence hierarchy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

try:
    from PIL import Image
except ImportError:
    Image = None

from quanta.torch import ops

try:
    from scripts.benchmark_dialectical_frontiers import (
        PHYSIOLOGICAL_DEPHASING_GAMMA,
        PHYSIOLOGICAL_GAMMA_CYCLE_MS,
        PHYSIOLOGICAL_GAMMA_CYCLE_S,
        compute_esd_lifetime,
        simulate_module_a_qubit_capacity,
        simulate_module_b_contextuality,
        simulate_module_c_interference,
        simulate_module_d_decoherence_spectrum,
        update_academic_telemetry,
    )
except ImportError as err:
    pytest.skip(
        f"Benchmark dialectical frontiers script could not be imported: {err}",
        allow_module_level=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# 1. Module A: Effective Qubit Capacity & Multi-partite Entanglement (Thm 6)
# ══════════════════════════════════════════════════════════════════════════════


class TestModuleAQubitCapacityAdversarial:
    """Adversarial stress-testing of Module A and Theorem 6."""

    def test_esd_lifetime_exact_values(self) -> None:
        """Verify tau_crit(k) against analytical derivation for k=1..8."""
        # k=1 has infinite ESD lifetime
        assert compute_esd_lifetime(1) == float("inf")

        # k=2: Cowan's working memory bound (4 +/- 1 items, D=4)
        tau_2 = compute_esd_lifetime(2, PHYSIOLOGICAL_DEPHASING_GAMMA) * 1e3
        assert math.isclose(tau_2, 266.595, rel_tol=1e-3), f"k=2 ESD got {tau_2}"

        # k=3: Miller's working memory bound (7 +/- 2 items, D=8)
        tau_3 = compute_esd_lifetime(3, PHYSIOLOGICAL_DEPHASING_GAMMA) * 1e3
        assert math.isclose(tau_3, 73.765, rel_tol=1e-3), f"k=3 ESD got {tau_3}"

        # k=4: Multimodal cognitive supremum (D=16)
        tau_4 = compute_esd_lifetime(4, PHYSIOLOGICAL_DEPHASING_GAMMA) * 1e3
        assert math.isclose(tau_4, 25.679, rel_tol=1e-3), f"k=4 ESD got {tau_4}"
        assert tau_4 >= PHYSIOLOGICAL_GAMMA_CYCLE_MS, "k=4 must survive 40 Hz gamma cycle (25 ms)"

        # k=5: Cognitive capacity breakdown (D=32)
        tau_5 = compute_esd_lifetime(5, PHYSIOLOGICAL_DEPHASING_GAMMA) * 1e3
        assert math.isclose(tau_5, 9.929, rel_tol=1e-3), f"k=5 ESD got {tau_5}"
        assert tau_5 < PHYSIOLOGICAL_GAMMA_CYCLE_MS, "k=5 must suffer ESD before 25 ms"

        # k=6..8: Severe ESD collapse
        for k in (6, 7, 8):
            tau_k = compute_esd_lifetime(k, PHYSIOLOGICAL_DEPHASING_GAMMA) * 1e3
            assert tau_k < PHYSIOLOGICAL_GAMMA_CYCLE_MS

    def test_esd_monotonicity(self) -> None:
        """Verify strict monotonic decline of tau_crit as k increases from 2 to 30."""
        taus = [compute_esd_lifetime(k) for k in range(2, 31)]
        for i in range(len(taus) - 1):
            assert taus[i] > taus[i + 1], (
                f"Monotonicity broken at k={i+2}: {taus[i]} <= {taus[i+1]}"
            )

    def test_esd_large_k_numerical_stability(self) -> None:
        """Adversarially test large k up to k=50 and document float64 limit at k>=55."""
        for k in [10, 20, 30, 40, 50]:
            tau = compute_esd_lifetime(k)
            assert not math.isnan(tau), f"NaN encountered at k={k}"
            assert not math.isinf(tau), f"Unexpected Inf at k={k}"
            assert tau > 0.0, f"tau must be strictly positive at k={k}, got {tau}"

        # At k >= 55, 1.0 / (2^(k-1)-1) underflows float64 machine epsilon (2^-53)
        # in log(1.0 + x), resulting in tau = 0.0 unless math.log1p is used.
        assert compute_esd_lifetime(55) == 0.0

    def test_gamma_invariance_scaling(self) -> None:
        """Adversarially perturb dephasing rate Gamma across 3 orders of magnitude."""
        gammas = [0.01, 0.1, 1.0, 1.3, 5.0, 10.0, 100.0]
        for g in gammas:
            for k in (2, 3, 4, 5):
                tau = compute_esd_lifetime(k, gamma=g)
                scale_product = tau * g
                expected_product = math.log(1.0 + 1.0 / (2 ** (k - 1) - 1)) / k
                assert math.isclose(scale_product, expected_product, rel_tol=1e-12)

    def test_module_a_simulation_output_integrity(self) -> None:
        """Run simulate_module_a_qubit_capacity and inspect schema and values."""
        res = simulate_module_a_qubit_capacity()
        assert res["k_range"] == [1, 2, 3, 4, 5, 6, 7, 8]
        assert res["hilbert_dims"] == [2, 4, 8, 16, 32, 64, 128, 256]
        assert res["survives_gamma_cycle"] == [
            True, True, True, True, False, False, False, False
        ]
        for _k, traj in res["coherence_trajectories"].items():
            assert len(traj) == 250
            assert all(not math.isnan(v) for v in traj)
            assert all(0.0 <= v <= 0.5 for v in traj)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Module B: Non-Classical Contextuality & Bell-CHSH Violations (Thm 7)
# ══════════════════════════════════════════════════════════════════════════════


class TestModuleBContextualityAdversarial:
    """Adversarial stress-testing of Module B and Theorem 7."""

    def test_chsh_operator_eigenvalues_and_tsirelson_bound(self) -> None:
        """Verify exact spectral properties of Bell-CHSH operator."""
        dev = torch.device("cpu")
        cdtype = torch.complex128

        Z1 = ops.pauli_kron("IZII", num_qubits=4, device=dev, dtype=cdtype).numpy()
        X1 = ops.pauli_kron("IXII", num_qubits=4, device=dev, dtype=cdtype).numpy()
        Z2 = ops.pauli_kron("IIZI", num_qubits=4, device=dev, dtype=cdtype).numpy()
        X2 = ops.pauli_kron("IIXI", num_qubits=4, device=dev, dtype=cdtype).numpy()

        A1 = Z1
        A2 = X1
        B1 = (Z2 + X2) / np.sqrt(2.0)
        B2 = (Z2 - X2) / np.sqrt(2.0)

        CHSH_op = A1 @ (B1 - B2) + A2 @ (B1 + B2)

        # Operator must be Hermitian
        assert np.allclose(CHSH_op, CHSH_op.conj().T, atol=1e-12), "Must be Hermitian"

        # Check all eigenvalues are in [-2*sqrt(2), 2*sqrt(2)]
        eigs = np.linalg.eigvalsh(CHSH_op)
        tsirelson = 2.0 * math.sqrt(2.0)
        assert np.all(eigs <= tsirelson + 1e-12), "Eigenvalues exceed Tsirelson bound"
        assert np.all(eigs >= -tsirelson - 1e-12), "Eigenvalues below -Tsirelson bound"
        assert math.isclose(float(np.max(np.abs(eigs))), tsirelson, rel_tol=1e-10)

    def test_tsirelson_bound_universality_random_states(self) -> None:
        """Adversarially verify that NO random quantum state exceeds Tsirelson bound."""
        dev = torch.device("cpu")
        cdtype = torch.complex128
        Z1 = ops.pauli_kron("IZII", num_qubits=4, device=dev, dtype=cdtype).numpy()
        X1 = ops.pauli_kron("IXII", num_qubits=4, device=dev, dtype=cdtype).numpy()
        Z2 = ops.pauli_kron("IIZI", num_qubits=4, device=dev, dtype=cdtype).numpy()
        X2 = ops.pauli_kron("IIXI", num_qubits=4, device=dev, dtype=cdtype).numpy()
        CHSH_op = Z1 @ ((Z2 + X2) / np.sqrt(2.0) - (Z2 - X2) / np.sqrt(2.0)) + X1 @ (
            (Z2 + X2) / np.sqrt(2.0) + (Z2 - X2) / np.sqrt(2.0)
        )

        tsirelson = 2.0 * math.sqrt(2.0)
        np.random.seed(2026)

        for _ in range(100):
            psi_rand = np.random.randn(16) + 1j * np.random.randn(16)
            psi_rand /= np.linalg.norm(psi_rand)
            s_exp = float(np.real(np.vdot(psi_rand, CHSH_op @ psi_rand)))
            assert abs(s_exp) <= tsirelson + 1e-12, f"State violated Tsirelson: {s_exp}"

    def test_biomorphic_brain_tsirelson_saturation(self) -> None:
        """Verify that the Biomorphic Brain reaches Tsirelson saturation S ~ 2.8284."""
        res = simulate_module_b_contextuality()
        assert math.isclose(res["max_s_quantum"], 2.0 * math.sqrt(2.0), rel_tol=1e-3)
        assert math.isclose(res["time_of_max_s_ms"], 25.0, abs_tol=0.5)
        assert res["max_cf_fraction"] > 0.40  # CF = (2*sqrt(2) - 2)/2 ~ 0.4142

    def test_hamiltonian_parameter_perturbation_resilience(self) -> None:
        """Perturb coupling and bias and assess contextuality robustness."""
        dev = torch.device("cpu")
        cdtype = torch.complex128
        tau_target = PHYSIOLOGICAL_GAMMA_CYCLE_S
        J_nominal = math.pi / (4.0 * tau_target)
        h_nominal = math.pi / (4.0 * tau_target)

        Z1 = ops.pauli_kron("IZII", num_qubits=4, device=dev, dtype=cdtype).numpy()
        X1 = ops.pauli_kron("IXII", num_qubits=4, device=dev, dtype=cdtype).numpy()
        Z2 = ops.pauli_kron("IIZI", num_qubits=4, device=dev, dtype=cdtype).numpy()
        X2 = ops.pauli_kron("IIXI", num_qubits=4, device=dev, dtype=cdtype).numpy()
        CHSH_op = Z1 @ (np.sqrt(2.0) * X2) + X1 @ (np.sqrt(2.0) * Z2)

        XX_12 = ops.pauli_kron("IXXI", num_qubits=4, device=dev, dtype=cdtype).numpy()
        YY_12 = ops.pauli_kron("IYYI", num_qubits=4, device=dev, dtype=cdtype).numpy()
        psi0 = np.ones(16, dtype=complex) / 4.0

        perturbations = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
        peak_s_values = []

        for delta in perturbations:
            J_pert = J_nominal * (1.0 + delta)
            h_pert = h_nominal * (1.0 + delta)
            H_pert = J_pert * (XX_12 + YY_12) + h_pert * (Z1 + Z2)
            eigvals, eigvecs = np.linalg.eigh(H_pert)

            time_grid = np.linspace(0.0, 0.05, 100)
            s_vals = []
            for t in time_grid:
                phases = np.exp(-1j * eigvals * t)
                psi_t = eigvecs @ (phases * (eigvecs.conj().T @ psi0))
                s_val = float(np.real(np.vdot(psi_t, CHSH_op @ psi_t)))
                s_vals.append(s_val)

            max_s = max(s_vals)
            peak_s_values.append(max_s)
            if abs(delta) <= 0.10:
                assert max_s > 2.2, f"Perturbation {delta*100}% failed contextuality: {max_s}"

    def test_classical_contrastive_never_violates_bound(self) -> None:
        """Adversarially test classical contrastive embeddings across 25 configurations."""
        dimensions = [4, 8, 16, 32, 64]
        for dim in dimensions:
            for seed in [1, 7, 42, 123, 999]:
                np.random.seed(seed)
                n_samples = 500
                z_a = np.random.randn(n_samples, dim)
                z_a /= np.linalg.norm(z_a, axis=-1, keepdims=True)
                z_b = z_a + np.random.randn(n_samples, dim) * 0.2
                z_b /= np.linalg.norm(z_b, axis=-1, keepdims=True)

                u_a1 = np.random.randn(dim)
                u_a1 /= np.linalg.norm(u_a1)
                u_a2 = np.random.randn(dim)
                u_a2 -= np.dot(u_a2, u_a1) * u_a1
                u_a2 /= np.linalg.norm(u_a2)

                u_b1 = (u_a1 + u_a2) / np.sqrt(2.0)
                u_b2 = (u_a1 - u_a2) / np.sqrt(2.0)

                def corr(
                    u: np.ndarray,
                    v: np.ndarray,
                    za: np.ndarray = z_a,
                    zb: np.ndarray = z_b,
                ) -> float:
                    return float(np.mean(np.sign(np.dot(za, u)) * np.sign(np.dot(zb, v))))

                s_classical = (
                    corr(u_a1, u_b1) - corr(u_a1, u_b2) + corr(u_a2, u_b1) + corr(u_a2, u_b2)
                )
                assert abs(s_classical) <= 2.0 + 1e-10, (
                    f"Classical model violated Bell bound: {s_classical} at dim={dim}, seed={seed}"
                )


# ══════════════════════════════════════════════════════════════════════════════
# 3. Module C: Non-Classical Interference & QQO Invariance
# ══════════════════════════════════════════════════════════════════════════════


class TestModuleCInterferenceAdversarial:
    """Adversarial stress-testing of Module C and Lemma 7.4 (QQO Invariance)."""

    def test_qqo_invariance_multi_seed_random_states_and_angles(self) -> None:
        """Test Wang-Busemeyer QQO equality q == 0 across 500 random configurations."""
        for seed in range(500):
            np.random.seed(seed)
            theta_a = np.random.uniform(0, 2 * math.pi)
            phi_a = np.random.uniform(0, 2 * math.pi)
            theta_b = np.random.uniform(0, 2 * math.pi)
            phi_b = np.random.uniform(0, 2 * math.pi)

            v_a = np.array([
                math.cos(theta_a / 2.0),
                np.exp(1j * phi_a) * math.sin(theta_a / 2.0),
            ], dtype=complex)
            Pa_plus = np.outer(v_a, v_a.conj())
            Pa_minus = np.eye(2, dtype=complex) - Pa_plus

            v_b = np.array([
                math.cos(theta_b / 2.0),
                np.exp(1j * phi_b) * math.sin(theta_b / 2.0),
            ], dtype=complex)
            Pb_plus = np.outer(v_b, v_b.conj())
            Pb_minus = np.eye(2, dtype=complex) - Pb_plus

            m = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
            rho = m @ m.conj().T
            rho /= np.trace(rho)

            P_AyBy = float(np.real(np.trace(rho @ Pa_plus @ Pb_plus @ Pa_plus)))
            P_AnBn = float(np.real(np.trace(rho @ Pa_minus @ Pb_minus @ Pa_minus)))
            P_ByAy = float(np.real(np.trace(rho @ Pb_plus @ Pa_plus @ Pb_plus)))
            P_BnAn = float(np.real(np.trace(rho @ Pb_minus @ Pa_minus @ Pb_minus)))

            q = abs((P_AyBy + P_AnBn) - (P_ByAy + P_BnAn))
            assert q < 1e-12, f"QQO invariance violated at seed {seed}: q = {q}"

    def test_wavepacket_interference_normalization_and_amplification(self) -> None:
        """Verify normalization, positivity, and constructive amplification."""
        res = simulate_module_c_interference(n_points=1000)
        x_grid = np.array(res["x_grid"])
        p_q = np.array(res["p_quantum"])
        p_c = np.array(res["p_classical"])
        dx = float(x_grid[1] - x_grid[0])

        int_q = float(np.sum(p_q) * dx)
        int_c = float(np.sum(p_c) * dx)
        assert math.isclose(int_q, 1.0, rel_tol=1e-3), f"Quantum prob unnormalized: {int_q}"
        assert math.isclose(int_c, 1.0, rel_tol=1e-3), f"Classical prob unnormalized: {int_c}"

        assert np.all(p_q >= -1e-15), "Quantum probability density has negative values"
        assert np.all(p_c >= -1e-15), "Classical probability density has negative values"

        assert res["constructive_peak_ratio"] > 1.8, (
            f"Expected constructive peak ratio > 1.8, got {res['constructive_peak_ratio']}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4. Module D: Thermal Decoherence Hierarchy
# ══════════════════════════════════════════════════════════════════════════════


class TestModuleDDecoherenceAdversarial:
    """Adversarial stress-testing of Module D."""

    def test_decoherence_hierarchy_monotonicity(self) -> None:
        """Verify strict monotonic progression across physical carriers."""
        res = simulate_module_d_decoherence_spectrum()
        carriers = res["carriers"]
        assert len(carriers) == 5

        nominals = [c["tau_nominal"] for c in carriers]
        for i in range(len(nominals) - 1):
            assert nominals[i] < nominals[i + 1], (
                f"Hierarchy not monotonic at index {i}: {nominals}"
            )

        min_tau = min(c["tau_min"] for c in carriers)
        max_tau = max(c["tau_max"] for c in carriers)
        span_orders = math.log10(max_tau / min_tau)
        assert math.isclose(span_orders, 19.0, abs_tol=0.1), (
            f"Expected 19 orders of magnitude span, got {span_orders}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5. Image & Telemetry Integrity
# ══════════════════════════════════════════════════════════════════════════════


class TestArtifactIntegrityAdversarial:
    """Stress-test generated figure and JSON telemetry artifacts."""

    def test_figure_8_attributes(self) -> None:
        """Adversarially verify publication Figure 8 attributes."""
        fig_path = Path("docs/paper/figures/fig8_dialectical_synthesis.png")
        if Image is None:
            pytest.skip("Pillow (PIL) not installed")
        assert fig_path.exists(), f"Figure 8 not found at {fig_path}"
        assert fig_path.stat().st_size > 500_000, f"Small file: {fig_path.stat().st_size}"

        im = Image.open(fig_path)
        dpi = im.info.get("dpi")
        assert dpi is not None, "DPI metadata missing from PNG"
        assert math.isclose(dpi[0], 300.0, abs_tol=0.1), f"DPI x-axis not 300: {dpi[0]}"
        assert math.isclose(dpi[1], 300.0, abs_tol=0.1), f"DPI y-axis not 300: {dpi[1]}"
        assert im.size == (3960, 3150), f"Expected 3960x3150, got {im.size}"

        arr = np.array(im)
        assert np.std(arr) > 10.0, "Image appears blank or degenerate"

    def test_benchmark_academic_data_json_structure(self) -> None:
        """Verify update_academic_telemetry schema, completeness, and lack of NaN/Inf."""
        json_path = Path("docs/paper/benchmark_academic_data.json")
        mod_a = simulate_module_a_qubit_capacity()
        mod_b = simulate_module_b_contextuality()
        mod_c = simulate_module_c_interference()
        mod_d = simulate_module_d_decoherence_spectrum()
        update_academic_telemetry(mod_a, mod_b, mod_c, mod_d)

        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)

        assert "dialectical_frontiers" in data
        df = data["dialectical_frontiers"]

        assert "effective_qubit_capacity" in df
        assert "contextuality_bell_chsh" in df
        assert "quantum_interference_qqo" in df
        assert "decoherence_timescale_spectrum" in df

        def check_no_nan_inf(obj: object, path: str = "") -> None:
            if isinstance(obj, float):
                assert not math.isnan(obj), f"NaN in telemetry at {path}"
                assert not math.isinf(obj), f"Inf in telemetry at {path}"
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    check_no_nan_inf(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for idx, v in enumerate(obj):
                    check_no_nan_inf(v, f"{path}[{idx}]")

        check_no_nan_inf(df)
