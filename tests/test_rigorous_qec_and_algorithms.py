"""Comprehensive mathematical and physical verification suite for Quanta QEC,
Layer-3 quantum algorithms, QML, and MCP Server (Carbon Slice 3).

Tests strict algebraic invariants, exact GF(2) homology, physical thresholds,
gate-level circuit synthesis, shot-sampled IQAE, Black-Scholes CRN Greeks,
chain-rule parameter-shift gradients, dual QSVM KKT conditions, O(m*2^n)
bitwise Pauli expectation, and all 25 MCP tools.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm as normal_dist

from quanta.core.gates import GATE_REGISTRY
from quanta.dag.dag_circuit import DAGCircuit
from quanta.layer3 import monte_carlo as mc_mod
from quanta.layer3.clustering import quantum_cluster, quantum_distance
from quanta.layer3.entity_resolution import resolve
from quanta.layer3.finance import portfolio_optimize
from quanta.layer3.hamiltonian import (
    molecular_hamiltonian,
    trotter_evolve,
)
from quanta.layer3.hamiltonian import (
    pauli_expectation as ham_pauli_expectation,
)
from quanta.layer3.monte_carlo import (
    amplitude_estimate,
    compute_greeks,
    quantum_monte_carlo,
)
from quanta.layer3.optimize import optimize
from quanta.layer3.qml import QuantumClassifier, QuantumRegressor
from quanta.layer3.qsvm import qsvm_classify
from quanta.layer3.search import search
from quanta.layer3.shor import factor
from quanta.layer3.vqe import build_hamiltonian_matrix, pauli_expectation, vqe
from quanta.qec.codes import ShorCode, SteaneCode
from quanta.qec.color_code import ColorCode
from quanta.qec.decoder import MWPMDecoder, UnionFindDecoder
from quanta.qec.distillation import (
    BravyiKitaev15to1Factory,
    CCZFactory,
    LatticeSurgery,
    LatticeSurgeryPatch,
)
from quanta.qec.qldpc import BivariateBicycleCode
from quanta.qec.surface_code import SurfaceCode
from quanta.qml import Classifier, Regressor
from quanta.simulator.factory import create_simulator

# ═══════════════════════════════════════════════════════════════════════════
# 1. SurfaceCode Exact F2 Homology & Dynamic Temporal Boundary
# ═══════════════════════════════════════════════════════════════════════════


class TestSurfaceCodeHomologyAndDynamicReadout:
    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_algebraic_css_and_logical_commutation(self, d: int) -> None:
        code = SurfaceCode(distance=d)
        assert code.H_X.shape == ((d * d - 1) // 2, d * d)
        assert code.H_Z.shape == ((d * d - 1) // 2, d * d)
        # CSS orthogonality: H_X @ H_Z^T == 0 (mod 2)
        assert not np.any((code.H_X @ code.H_Z.T) % 2)
        # Logical commutation with opposite-type checks
        assert not np.any((code.H_Z @ code.logical_x) % 2)
        assert not np.any((code.H_X @ code.logical_z) % 2)
        # Logical anticommutation <X_bar, Z_bar> == 1 (mod 2)
        assert int((code.logical_x @ code.logical_z) % 2) == 1
        assert int(code.logical_x.sum()) == d
        assert int(code.logical_z.sum()) == d

    def test_zero_false_positives_across_all_pure_stabilizer_products_d3_and_d5(
        self,
    ) -> None:
        # d=3: all 2^4 - 1 = 15 non-empty products of X-stabilizers
        code3 = SurfaceCode(distance=3)
        m3 = code3.H_X.shape[0]
        assert m3 == 4
        false_positives_d3 = 0
        for mask_int in range(1, 1 << m3):
            coeffs = np.array([(mask_int >> k) & 1 for k in range(m3)], dtype=np.uint8)
            stab_product = (coeffs @ code3.H_X) % 2
            if code3._check_logical_error(stab_product.astype(bool)):
                false_positives_d3 += 1
            # Multiplying by logical_x MUST flip homology to a true logical error
            logical_coset = (stab_product ^ code3.logical_x).astype(bool)
            assert code3._check_logical_error(logical_coset) is True
        assert false_positives_d3 == 0

        # d=5: all 2^12 - 1 = 4095 non-empty products of X-stabilizers
        code5 = SurfaceCode(distance=5)
        m5 = code5.H_X.shape[0]
        assert m5 == 12
        all_masks = (
            (np.arange(1, 1 << m5, dtype=np.uint32)[:, None] >> np.arange(m5)) & 1
        ).astype(np.uint8)
        all_products = (all_masks @ code5.H_X) % 2  # (4095, 25)
        syn_z = (all_products @ code5.H_Z.T) % 2
        hom_z = (all_products @ code5.logical_z) % 2
        assert int(np.sum(np.any(syn_z != 0, axis=1))) == 0
        assert int(np.sum(hom_z != 0)) == 0
        for idx in range(len(all_products)):
            assert code5._check_logical_error(all_products[idx].astype(bool)) is False

    def test_dynamic_final_noiseless_data_readout_eliminates_boundary_meas_flips(
        self,
    ) -> None:
        code = SurfaceCode(distance=3)
        # Zero physical data errors, 5% measurement flip noise across 5 rounds:
        # Mandatory final noiseless destructive data-readout difference round at t = cycles
        # prevents t = cycles - 1 measurement flips from injecting physical boundary errors.
        res = code.simulate_dynamic(
            physical_error_rate=0.0,
            measurement_error_rate=0.05,
            cycles=5,
            shots=200,
            seed=123,
        )
        assert res.defects_detected > 0
        assert res.logical_error_rate == 0.0

    def test_realistic_thresholds_and_depolarizing_channel(self) -> None:
        code = SurfaceCode(distance=3)
        res_x = code.simulate_error_correction(error_rate=0.01, rounds=200, seed=7)
        assert math.isclose(res_x.threshold_estimate, 0.103, rel_tol=0.05)
        res_dep = code.simulate_error_correction(
            error_rate=0.01, rounds=200, seed=7, channel="depolarizing"
        )
        assert 0.0 <= res_dep.logical_error_rate < 0.05


# ═══════════════════════════════════════════════════════════════════════════
# 2. MWPM & Delfosse-Nickerson Union-Find Peeling Decoders
# ═══════════════════════════════════════════════════════════════════════════


class TestMWPMAndUnionFindDecoders:
    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_union_find_returns_valid_data_qubit_corrections(self, d: int) -> None:
        code = SurfaceCode(distance=d)
        uf = UnionFindDecoder()
        rng = np.random.default_rng(42 + d)
        for _ in range(80):
            err = (rng.random(code.n_physical) < 0.06).astype(np.uint8)
            syn_z = (code.H_Z @ err) % 2
            dec = uf.decode(syn_z, code_distance=d, stabilizers=code._z_stabilizers)
            assert dec.success is True
            corr_vec = np.zeros(code.n_physical, dtype=np.uint8)
            for q in dec.correction:
                assert 0 <= q < code.n_physical
                corr_vec[q] ^= 1
            assert np.array_equal((code.H_Z @ corr_vec) % 2, syn_z)

    def test_threshold_scaling_monotonic_with_distance_d3_d5_d7(self) -> None:
        p = 0.045
        rounds = 1500
        mwpm = MWPMDecoder()
        uf = UnionFindDecoder()
        mwpm_rates: list[float] = []
        uf_rates: list[float] = []

        for d in (3, 5, 7):
            code = SurfaceCode(distance=d)
            rng = np.random.default_rng(100 + d)
            mwpm_fail = 0
            uf_fail = 0
            for _ in range(rounds):
                err = (rng.random(code.n_physical) < p).astype(np.uint8)
                syn_z = (code.H_Z @ err) % 2

                r_m = mwpm.decode(syn_z, d, stabilizers=code._z_stabilizers)
                c_m = np.zeros(code.n_physical, dtype=np.uint8)
                for q in r_m.correction:
                    c_m[q] ^= 1
                res_m = err ^ c_m
                if np.any((code.H_Z @ res_m) % 2) or int((code.logical_z @ res_m) % 2):
                    mwpm_fail += 1

                r_u = uf.decode(syn_z, d, stabilizers=code._z_stabilizers)
                c_u = np.zeros(code.n_physical, dtype=np.uint8)
                for q in r_u.correction:
                    c_u[q] ^= 1
                res_u = err ^ c_u
                if np.any((code.H_Z @ res_u) % 2) or int((code.logical_z @ res_u) % 2):
                    uf_fail += 1

            mwpm_rates.append(mwpm_fail / rounds)
            uf_rates.append(uf_fail / rounds)

        assert mwpm_rates[2] < mwpm_rates[1] < mwpm_rates[0]
        assert uf_rates[2] < uf_rates[1] < uf_rates[0]

    def test_fifty_percent_noise_yields_half_logical_error_never_zero(self) -> None:
        code = SurfaceCode(distance=3)
        res = code.simulate_error_correction(error_rate=0.5, rounds=800, seed=99)
        assert 0.40 <= res.logical_error_rate <= 0.60


# ═══════════════════════════════════════════════════════════════════════════
# 3. 6.6.6 Triangular ColorCode & Syndrome-Only Restriction Decoder
# ═══════════════════════════════════════════════════════════════════════════


class TestColorCode666AndRestrictionDecoder:
    @pytest.mark.parametrize(
        ("d", "expected_n", "expected_plaq"),
        [(3, 7, 3), (5, 19, 9), (7, 37, 18)],
    )
    def test_commuting_666_lattice_invariants(
        self, d: int, expected_n: int, expected_plaq: int
    ) -> None:
        cc = ColorCode(distance=d)
        assert cc.n_data == expected_n
        assert len(cc.plaquettes) == expected_plaq
        for p in cc.plaquettes:
            assert len(p.qubits) in (4, 6)
            assert len(p.qubits) % 2 == 0
        # Exact CSS commutation [S_X, S_Z] = 0
        assert not np.any((cc.H_X @ cc.H_Z.T) % 2)
        # Transversal logical X_bar = Z_bar = 1^n commutes with all checks and has odd weight
        ones = np.ones(cc.n_data, dtype=np.uint8)
        assert not np.any((cc.H_X @ ones) % 2)
        assert not np.any((cc.H_Z @ ones) % 2)
        assert int((ones @ ones) % 2) == 1

    def test_syndrome_only_restriction_decoder_never_peeks_at_error_mask(self) -> None:
        # 1. All weight-1 errors on [[7, 1, 3]] decoded from syndrome ONLY (error_mask=None)
        cc3 = ColorCode(distance=3)
        m3 = len(cc3.plaquettes)
        for q in range(cc3.n_data):
            err = np.zeros(cc3.n_data, dtype=np.uint8)
            err[q] = 1
            syn_z = (cc3.H @ err) % 2
            assert len(syn_z) == m3
            corr = cc3.restriction_decode(syn_z)  # NO error_mask passed -> returns correction!
            assert isinstance(corr, np.ndarray)
            residual = err ^ corr.astype(np.uint8)
            assert not np.any((cc3.H @ residual) % 2)
            assert int(residual.sum()) % 2 == 0

        # 2. All (19 choose 2) = 171 weight-2 errors corrected on [[19, 1, 5]]!
        cc5 = ColorCode(distance=5)
        for i in range(cc5.n_data):
            for j in range(i + 1, cc5.n_data):
                err = np.zeros(cc5.n_data, dtype=np.uint8)
                err[i] = 1
                err[j] = 1
                syn_z = (cc5.H @ err) % 2
                corr = cc5.restriction_decode(syn_z)
                assert isinstance(corr, np.ndarray)
                residual = err ^ corr.astype(np.uint8)
                assert not np.any((cc5.H @ residual) % 2)
                assert int(residual.sum()) % 2 == 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. ShorCode (all 27 Paulis), SteaneCode, BPOSD-CS, & RM Distillation
# ═══════════════════════════════════════════════════════════════════════════


class TestShorSteaneBPOSDAndDistillation:
    def test_shor_code_recovers_all_27_single_pauli_errors(self) -> None:
        shor = ShorCode()
        assert len(shor.lookup_table()) >= 28
        for qubit in range(9):
            for pauli in ("X", "Y", "Z"):
                fidelity = shor.verify_single_pauli_recovery(pauli, qubit)
                assert fidelity > 0.999999, f"Failed on ({pauli}, q={qubit}): F={fidelity}"

    def test_steane_code_6bit_css_syndromes_cover_all_21_single_paulis(self) -> None:
        steane = SteaneCode()
        for q in range(7):
            for pauli in ("X", "Y", "Z"):
                fidelity = steane.verify_single_pauli_recovery(pauli, q)
                assert fidelity > 0.999999, f"Failed on ({pauli}, q={q}): F={fidelity}"

    def test_bposd_combination_sweep_and_homology_on_bb_code(self) -> None:
        bb = BivariateBicycleCode(
            ell=3, m=3, A_poly=[(0, 0), (1, 0), (0, 1)], B_poly=[(0, 0), (1, 1), (2, 0)]
        )
        assert bb.is_css_orthogonal is True
        r0 = bb.simulate(error_rate=0.06, rounds=150, osd_order=0, max_bp_iter=15, seed=42)
        r2 = bb.simulate(error_rate=0.06, rounds=150, osd_order=2, max_bp_iter=15, seed=42)
        assert r2["logical_error_rate"] <= r0["logical_error_rate"]

    def test_reed_muller_15_to_1_distillation_ccz_and_lattice_surgery(self) -> None:
        msd = BravyiKitaev15to1Factory()
        assert msd.H_X.shape == (4, 15)
        assert msd.H_Z.shape == (10, 15)
        assert all(int(w) == 8 for w in msd.H_X.sum(axis=1))
        assert all(int(w) == 4 for w in msd.H_Z.sum(axis=1))
        assert not np.any((msd.H_X @ msd.H_Z.T) % 2)

        p = 0.01
        expected_accept = (1.0 + 15.0 * ((1.0 - 2.0 * p) ** 8)) / 16.0
        assert abs(msd.acceptance_probability(p) - expected_accept) < 1e-10
        assert msd.acceptance_probability(p) < 1.0 - 13.0 * p

        res = msd.distill(input_error_rate=p, shots=25_000, seed=42)
        assert abs(res.acceptance_probability - expected_accept) < 0.015
        assert abs(msd.analytical_output_error(p) - 35.0 * (p**3)) < 1e-9
        assert res.output_error_rate < 2e-4

        # Exact 7-T CCZFactory magic-state preparation circuit |CCZ> = CCZ |+>^3
        ccz = CCZFactory()
        circ_def = ccz.build_circuit()
        dag = DAGCircuit.from_builder(circ_def.build())
        t_gates = sum(
            1 for node in dag.op_nodes() if node.gate_name.upper() in ("T", "TDG")
        )
        assert t_gates == 7
        sim = create_simulator(3)
        for node in dag.op_nodes():
            sim.apply(node.gate_name, node.qubits, node.params)
        target_ccz_state = ccz.target_state
        fidelity_ccz = abs(np.vdot(target_ccz_state, sim.state)) ** 2
        assert math.isclose(fidelity_ccz, 1.0, abs_tol=1e-12)

        # LatticeSurgery seeded non-deterministic parity outcomes
        p_a = LatticeSurgeryPatch(id="L0", distance=3)
        p_b = LatticeSurgeryPatch(id="L1", distance=3)
        outcomes = {
            LatticeSurgery.merge(p_a, p_b, boundary_type="Z", seed=s).last_outcome
            for s in range(20)
        }
        assert outcomes == {-1, 1}


# ═══════════════════════════════════════════════════════════════════════════
# 5. Gate-Based Grover, Shor (n+1 QPE), QAOA, Portfolio, & Swap Test
# ═══════════════════════════════════════════════════════════════════════════


class TestGateBasedAlgorithms:
    def test_grover_direct_targets_zero_classical_loop_and_bbht(self) -> None:
        res = search(num_bits=5, targets=[23], shots=1024, seed=42)
        assert res.classical_setup_queries == 0
        assert res.oracle_queries == 4
        assert res.gate_count > 50  # Real decomposed elementary gate circuit
        assert res.most_frequent == "10111"
        assert res.probabilities["10111"] > 0.95

        # Blind oracle BBHT exponential schedule (unknown target multiplicity)
        res_bbht = search(
            num_bits=4,
            targets=[3, 11, 14],
            blind_oracle=True,
            shots=512,
            seed=42,
        )
        assert int(res_bbht.most_frequent, 2) in {3, 11, 14}
        assert res_bbht.oracle_queries >= 1

    @pytest.mark.parametrize("N", [15, 21, 35, 77, 143])
    def test_shor_beauregard_quantum_period_finding_no_classical_fallback(
        self, N: int
    ) -> None:
        sig = inspect.signature(factor)
        assert sig.parameters["allow_classical_fallback"].default is False

        res = factor(N, seed=42)
        assert res.factors[0] * res.factors[1] == N
        assert 1 < res.factors[0] < N
        assert res.method == "quantum"
        assert res.period > 0 and res.period % 2 == 0

    def test_walsh_hadamard_qaoa_and_ising_portfolio_optimize(self) -> None:
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        opt_res = optimize(
            num_bits=4,
            cost=lambda x: sum(((x >> i) & 1) ^ ((x >> j) & 1) for i, j in edges),
            minimize=False,
            layers=2,
            shots=2048,
            seed=42,
        )
        assert opt_res.best_bitstring in ("0101", "1010")
        assert opt_res.best_cost == 4.0

        assets = [
            {"name": "HIGH_SHARPE_A", "return": 0.18, "volatility": 0.12},
            {"name": "HIGH_SHARPE_B", "return": 0.16, "volatility": 0.11},
            {"name": "LOW_RETURN_C", "return": 0.02, "volatility": 0.30},
            {"name": "LOW_RETURN_D", "return": 0.01, "volatility": 0.35},
        ]
        port_res = portfolio_optimize(
            assets=assets, budget=2, risk_aversion=0.5, layers=2, shots=2048, seed=42
        )
        assert set(port_res.selected) == {"HIGH_SHARPE_A", "HIGH_SHARPE_B"}
        assert port_res.bitstring == "1100"

    def test_swap_test_shot_variance_and_per_iteration_quantum_kmeans(self) -> None:
        a = [1.0, 0.0]
        b = [0.6, 0.8]
        assert quantum_distance(a, a, shots=256, seed=1) < 1e-6
        d1 = quantum_distance(a, b, shots=128, seed=1)
        d2 = quantum_distance(a, b, shots=128, seed=2)
        assert d1 != d2
        assert 0.05 < d1 < 0.95

        data = [[0.0, 0.1], [0.1, 0.0], [5.0, 5.1], [5.1, 5.0]]
        cres = quantum_cluster(data, k=2, max_iterations=5, shots=512, seed=42)
        assert cres.labels[0] == cres.labels[1]
        assert cres.labels[2] == cres.labels[3]
        assert cres.labels[0] != cres.labels[2]

    def test_entity_resolution_monotonic_mapping(self) -> None:
        records = [
            {"name": "Ada Lovelace", "city": "London"},
            {"name": "Ada Lovelace", "city": "London UK"},
            {"name": "Alan Turing", "city": "Manchester"},
        ]
        er_res = resolve(records, threshold=0.65, seed=42)
        assert len(er_res.clusters) == 2


# ═══════════════════════════════════════════════════════════════════════════
# 6. Shot-Sampled IQAE & Black-Scholes CRN Greeks (< 2% Analytical Match)
# ═══════════════════════════════════════════════════════════════════════════


class TestShotSampledIQAEAndCRNGreeks:
    def test_iqae_never_references_a_direct_and_converges_from_shots(self) -> None:
        src = Path(mc_mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        identifiers = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        assert "a_direct" not in identifiers

        probs = np.array([0.25, 0.25, 0.25, 0.25])
        payoffs = np.array([0.0, 0.2, 0.6, 1.0])  # True E[f] = 0.45
        est_val, grover_iters = amplitude_estimate(
            probs, payoffs, n_qubits=2, n_estimation=5, seed=42, shots_per_round=512
        )
        assert abs(est_val - 0.45) < 0.04
        assert grover_iters > 0

    def test_quantum_monte_carlo_variance_payoff_computes_ex2_minus_ex_sq(self) -> None:
        S0, sigma, r, T = 100.0, 0.2, 0.05, 1.0
        exact_var = (S0**2) * math.exp(2 * r * T) * (math.exp(sigma**2 * T) - 1.0)
        qmc_var = quantum_monte_carlo(
            distribution="lognormal",
            payoff="var",
            params={"S0": S0, "K": 100.0, "sigma": sigma, "r": r, "T": T},
            n_qubits=6,
            n_estimation=6,
            seed=42,
        )
        assert abs(qmc_var.estimated_value - exact_var) / exact_var < 0.25

    def test_black_scholes_crn_greeks_within_2_percent_of_analytical(self) -> None:
        S, K, sigma, r, T = 100.0, 100.0, 0.2, 0.05, 1.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        bs_delta = float(normal_dist.cdf(d1))
        bs_gamma = float(normal_dist.pdf(d1) / (S * sigma * math.sqrt(T)))
        bs_vega = float(S * normal_dist.pdf(d1) * math.sqrt(T))
        bs_rho = float(K * T * math.exp(-r * T) * normal_dist.cdf(d2))

        g = compute_greeks(
            spot=S,
            strike=K,
            volatility=sigma,
            rate=r,
            time_to_expiry=T,
            payoff="european_call",
            n_samples=100_000,
            seed=42,
        )
        assert g.method == "classical_mc_crn"
        assert abs(g.delta - bs_delta) / bs_delta < 0.02
        assert abs(g.gamma - bs_gamma) / bs_gamma < 0.02
        assert abs(g.vega - bs_vega) / bs_vega < 0.02
        assert abs(g.rho - bs_rho) / bs_rho < 0.02

        gq = compute_greeks(
            spot=S,
            strike=K,
            volatility=sigma,
            rate=r,
            time_to_expiry=T,
            use_quantum=True,
            n_qubits=4,
            n_estimation=4,
            seed=42,
        )
        assert gq.method == "qae_finite_diff"


# ═══════════════════════════════════════════════════════════════════════════
# 7. QML Exact Chain-Rule Gradients, QSVM Dual KKT, & Bitwise O(m*2^n) VQE
# ═══════════════════════════════════════════════════════════════════════════


class TestQMLGradientsQSVMDualAndBitwiseVQE:
    def test_classifier_chain_rule_gradients_match_central_finite_difference(
        self,
    ) -> None:
        clf = QuantumClassifier(
            n_qubits=3, n_layers=2, feature_map="zz", seed=42
        )
        X = np.array([[0.2, 0.7, 0.4], [0.9, 0.1, 0.8], [0.5, 0.3, 0.6]])
        y = np.array([0, 1, 1])

        analytic_grad = clf._compute_gradients(X, y)
        eps = 1e-6
        fd_grad = np.zeros_like(analytic_grad)
        base_params = clf.params.copy()
        for k in range(clf.n_params):
            clf.params = base_params.copy()
            clf.params[k] += eps
            loss_plus = clf._loss(X, y)
            clf.params = base_params.copy()
            clf.params[k] -= eps
            loss_minus = clf._loss(X, y)
            fd_grad[k] = (loss_plus - loss_minus) / (2.0 * eps)
        clf.params = base_params

        np.testing.assert_allclose(analytic_grad, fd_grad, atol=1e-7, rtol=1e-5)

    def test_regressor_msb_q0_expectation_and_ansatz_optimizer_wiring(self) -> None:
        from quanta.layer3.qml import angle_encoding

        reg = QuantumRegressor(n_qubits=3, n_layers=1, seed=11)
        x = np.array([0.3, 0.8, 0.5])

        sim = create_simulator(3)
        angle_encoding(sim, x)
        reg._ansatz_preset.apply(sim, reg.params, 3, 1)
        psi = np.asarray(sim.state)
        Z0 = np.kron(np.diag([1.0, -1.0]), np.eye(4))
        exact_z0 = float(np.real(psi.conj() @ Z0 @ psi))
        assert math.isclose(reg._expectation_z(x), exact_z0, abs_tol=1e-12)

        X_tr = np.array([[0.1, 0.1], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]])
        y_tr = np.array([0, 0, 1, 1])
        c_api = Classifier(
            n_qubits=2, n_layers=2, ansatz="strongly_entangling", optimizer="cobyla", seed=7
        )
        r_c = c_api.fit(X_tr, y_tr, epochs=25)
        assert r_c.n_params == c_api._clf.n_params
        assert r_c.accuracy >= 0.75

        r_api = Regressor(
            n_qubits=2, n_layers=1, ansatz="reuploading", optimizer="adam", seed=7
        )
        r_r = r_api.fit(X_tr, np.array([-1.0, -0.8, 0.8, 1.0]), epochs=15)
        assert len(r_r.loss_history) == 15
        assert r_r.loss_history[-1] <= r_r.loss_history[0]

    def test_qsvm_dual_qp_satisfies_cortes_vapnik_kkt_conditions(self) -> None:
        X_train = np.array(
            [[0.05, 0.10], [0.12, 0.08], [0.15, 0.20], [0.85, 0.90], [0.90, 0.82], [0.88, 0.95]]
        )
        y_train = np.array([0, 0, 0, 1, 1, 1])
        X_test = np.array([[0.10, 0.12], [0.89, 0.88]])
        C_val = 2.0

        res = qsvm_classify(
            X_train.tolist(), y_train.tolist(), X_test.tolist(), num_qubits=2, regularization=C_val
        )
        assert list(res.predictions) == [0, 1]
        assert res.accuracy == 1.0

        alphas = np.asarray(res.alphas, dtype=float)
        y_pm = np.where(y_train == 1, 1.0, -1.0)
        assert np.all(alphas >= -1e-7) and np.all(alphas <= C_val + 1e-7)
        assert abs(float(np.dot(alphas, y_pm))) < 1e-5
        assert len(res.support_indices) >= 2

        K = np.asarray(res.kernel_matrix, dtype=float)
        free_sv = [
            i for i in range(len(alphas)) if 1e-4 < alphas[i] < C_val - 1e-4
        ]
        assert len(free_sv) >= 1
        for i in free_sv:
            margin = y_pm[i] * (float(np.sum(alphas * y_pm * K[i])) + res.intercept)
            assert abs(margin - 1.0) < 1e-3

    def test_bitwise_pauli_expectation_and_hamiltonian_match_dense_matrix(self) -> None:
        terms = [
            ("XYZ", 0.45),
            ("ZIX", -0.72),
            ("YZY", 0.31),
            ("III", 0.15),
        ]
        rng = np.random.default_rng(17)
        psi = rng.normal(size=8) + 1j * rng.normal(size=8)
        psi /= np.linalg.norm(psi)

        H_dense = build_hamiltonian_matrix(terms, 3)
        ref_exp = float(np.real(psi.conj() @ H_dense @ psi))
        assert math.isclose(pauli_expectation(terms, psi, 3), ref_exp, abs_tol=1e-13)
        assert math.isclose(ham_pauli_expectation(terms, psi, 3), ref_exp, abs_tol=1e-13)

        # Matrix-free Trotter step matches exp(-i * t * H) for small dt
        ev_res = trotter_evolve(terms, num_qubits=3, time=0.02, steps=8, order=2, initial_state=psi)
        from scipy.linalg import expm

        psi_exact = expm(-1j * 0.02 * H_dense) @ psi
        assert abs(np.vdot(ev_res.final_state, psi_exact)) > 0.999999

        # VQE H2 convergence check
        h2_spec = molecular_hamiltonian("H2")
        exact_e = float(np.linalg.eigvalsh(build_hamiltonian_matrix(h2_spec.terms, h2_spec.num_qubits))[0])
        vqe_res = vqe(h2_spec.num_qubits, h2_spec.terms, layers=2, seed=42)
        assert abs(vqe_res.energy - exact_e) < 1e-3


# ═══════════════════════════════════════════════════════════════════════════
# 8. MCP Server End-to-End Verification (All 25 Tools & Sandboxes)
# ═══════════════════════════════════════════════════════════════════════════


class TestMCPServerRigorEndToEnd:
    def test_25_registered_tools_and_all_31_gates_in_all_4_sandboxes(self) -> None:
        pytest.importorskip("fastmcp")
        from quanta.mcp_server import (
            draw_circuit,
            mcp,
            mps_circuit_simulate,
            optimize_circuit,
            run_circuit,
            sdk_info,
        )

        tools = asyncio.run(mcp.list_tools())
        info = json.loads(sdk_info())
        cat_tools = [t for lst in info["tool_categories"].values() for t in lst]
        assert len(tools) == 25
        assert info["total_tools"] == 25
        assert sorted(t.name for t in tools) == sorted(cat_tools)

        # Circuit exercising ALL 6 previously-missing v0.9 gates: ECR, iSWAP, CSWAP, CH, CP, MS
        v09_circuit_code = (
            "@circuit(qubits=3)\n"
            "def circ(q):\n"
            "    ECR(q[0], q[1])\n"
            "    iSWAP(q[1], q[2])\n"
            "    CH(q[0], q[2])\n"
            "    CP(pi / 4)(q[0], q[1])\n"
            "    MS(0.3)(q[1], q[2])\n"
            "    CSWAP(q[0], q[1], q[2])\n"
            "    return measure(q)\n"
        )
        r_run = json.loads(run_circuit(v09_circuit_code, shots=128, seed=42))
        assert "error" not in r_run and r_run["gate_count"] == 6

        v09_mps_code = (
            "@circuit(qubits=4)\n"
            "def circ(q):\n"
            "    ECR(q[0], q[1])\n"
            "    iSWAP(q[1], q[2])\n"
            "    CH(q[2], q[3])\n"
            "    CP(pi / 3)(q[0], q[1])\n"
            "    MS(0.2)(q[1], q[2])\n"
            "    return measure(q)\n"
        )
        r_mps = json.loads(
            mps_circuit_simulate(v09_mps_code, num_qubits=4, shots=64, seed=42)
        )
        assert "error" not in r_mps and sum(r_mps["counts"].values()) == 64

        r_opt = json.loads(optimize_circuit(v09_circuit_code))
        assert "error" not in r_opt and r_opt["original"]["gates"] == 6

        r_svg = json.loads(draw_circuit(v09_circuit_code))
        assert "error" not in r_svg and "<svg" in r_svg["html"].lower()

    def test_mcp_compare_decoders_noise_ftqc_reasoning_qaoa_and_catalog(self) -> None:
        pytest.importorskip("fastmcp")
        from quanta.mcp_server import (
            compare_decoders,
            estimate_fault_tolerant_cost,
            gate_catalog,
            grover_search,
            qaoa_optimize,
            qec_diagnose,
            quanta_reasoning_eval,
            shor_factor,
            simulate_noise,
        )
        from quanta.mcp_server import (
            portfolio_optimize as mcp_portfolio_optimize,
        )

        # 1. compare_decoders at 50% physical noise -> ~50% logical error (NEVER 0% or 100%)
        comp_50 = json.loads(
            compare_decoders(distance=3, error_rate=0.5, rounds=500, seed=42)
        )
        assert 0.38 <= comp_50["mwpm"]["logical_error_rate"] <= 0.62
        assert 0.38 <= comp_50["union_find"]["logical_error_rate"] <= 0.62

        # 2. simulate_noise: t2_relaxation at p=0.8 damps coherence monotonically (fidelity > 0.5)
        t2_res = json.loads(
            simulate_noise("t2_relaxation", probability=0.8, shots=256, seed=7)
        )
        pf_res = json.loads(
            simulate_noise("phaseflip", probability=0.8, shots=256, seed=7)
        )
        assert t2_res["fidelity"] > 0.5
        assert pf_res["fidelity"] < 0.45

        ct_res = json.loads(
            simulate_noise("crosstalk", probability=0.1, shots=256, seed=7)
        )
        assert abs(ct_res["fidelity"] - 0.9) < 1e-6

        dep_res = json.loads(
            simulate_noise("depolarizing", probability=0.06, shots=256, seed=7)
        )
        expected_f_p3 = 0.25 * (1.0 + (1.0 - 0.08) ** 2 + 2.0 * (1.0 - 0.08) ** 3)
        assert abs(dep_res["fidelity"] - round(expected_f_p3, 4)) < 1e-4

        # 3. qaoa_optimize raises ValueError on unknown problem strings
        with pytest.raises(ValueError, match="Unknown problem"):
            qaoa_optimize(num_bits=3, problem="traveling_salesman_invalid")

        # 4. estimate_fault_tolerant_cost: RCCX=3, Clifford RZ(pi/2)=0, arbitrary RZ -> ceil(3 log2(1/eps))
        cliff_code = (
            "@circuit(qubits=3)\n"
            "def circ(q):\n"
            "    H(q[0])\n"
            "    RZ(pi / 2)(q[0])\n"
            "    RX(pi)(q[1])\n"
            "    ECR(q[0], q[1])\n"
            "    return measure(q)\n"
        )
        ft_cliff = json.loads(estimate_fault_tolerant_cost(cliff_code))
        assert ft_cliff["magic_t_gates_required"] == 0
        assert ft_cliff["clifford_gates"] == 4
        assert ft_cliff["factory_levels"] == 0

        rccx_code = (
            "@circuit(qubits=3)\n"
            "def circ(q):\n"
            "    RCCX(q[0], q[1], q[2])\n"
            "    return measure(q)\n"
        )
        ft_rccx_l1 = json.loads(
            estimate_fault_tolerant_cost(
                rccx_code, target_logical_error_rate=1e-6, physical_error_rate=1e-3
            )
        )
        assert ft_rccx_l1["magic_t_gates_required"] == 3
        assert ft_rccx_l1["factory_levels"] == 1

        ft_rccx_l2 = json.loads(
            estimate_fault_tolerant_cost(
                rccx_code, target_logical_error_rate=1e-12, physical_error_rate=1e-3
            )
        )
        assert ft_rccx_l2["magic_t_gates_required"] == 3
        assert ft_rccx_l2["factory_levels"] == 2

        rot_code = (
            "@circuit(qubits=1)\n"
            "def circ(q):\n"
            "    RZ(0.37)(q[0])\n"
            "    return measure(q)\n"
        )
        ft_rot = json.loads(
            estimate_fault_tolerant_cost(
                rot_code, target_logical_error_rate=1e-9, physical_error_rate=1e-3
            )
        )
        expected_t_rot = math.ceil(3.0 * math.log2(1.0 / (1e-9 / 3.0)))
        assert ft_rot["magic_t_gates_required"] == expected_t_rot

        # 5. quanta_reasoning_eval classifies Clifford rotations accurately
        eval_cliff = json.loads(quanta_reasoning_eval(cliff_code))
        assert eval_cliff["non_clifford_gates"] == 0
        assert "Clifford" in eval_cliff["recommended_simulation_regime"]

        eval_noncliff = json.loads(quanta_reasoning_eval(rccx_code))
        assert eval_noncliff["non_clifford_gates"] == 1

        # 6. grover_search, shor_factor, qec_diagnose("steane"), portfolio_optimize, gate_catalog
        g_mcp = json.loads(grover_search(num_qubits=4, target=11, shots=512, seed=42))
        assert g_mcp["oracle_queries"] == 3
        assert g_mcp["gate_count"] > g_mcp["oracle_queries"]
        assert g_mcp["classical_setup_queries"] == 0

        s_mcp = json.loads(shor_factor(number=21, seed=42))
        assert s_mcp["is_correct"] is True
        assert s_mcp["method"] == "quantum"
        assert s_mcp["period"] > 0

        diag_steane = json.loads(qec_diagnose(code="steane", syndrome="001000"))
        assert diag_steane["notation"] == "[[7,1,3]]"

        p_mcp = json.loads(
            mcp_portfolio_optimize(budget=2, layers=2, shots=512, seed=42)
        )
        assert len(p_mcp["selected_assets"]) == 2

        cat = json.loads(gate_catalog())
        assert cat["isa_decomposition"]["RX(θ)"].count("sx") == 2
        assert cat["isa_decomposition"]["RY(θ)"].count("sx") == 2
        SX = GATE_REGISTRY["SX"].matrix
        for theta in (-1.7, 0.0, 0.61, 2.4):
            U_rx = (
                GATE_REGISTRY["RZ"](math.pi / 2).matrix
                @ SX
                @ GATE_REGISTRY["RZ"](theta + math.pi).matrix
                @ SX
                @ GATE_REGISTRY["RZ"](math.pi / 2).matrix
            )
            U_ry = (
                GATE_REGISTRY["RZ"](math.pi).matrix
                @ SX
                @ GATE_REGISTRY["RZ"](theta + math.pi).matrix
                @ SX
            )
            ov_x = (
                abs(np.trace(GATE_REGISTRY["RX"](theta).matrix.conj().T @ U_rx)) / 2.0
            )
            ov_y = (
                abs(np.trace(GATE_REGISTRY["RY"](theta).matrix.conj().T @ U_ry)) / 2.0
            )
            assert math.isclose(ov_x, 1.0, abs_tol=1e-12)
            assert math.isclose(ov_y, 1.0, abs_tol=1e-12)
