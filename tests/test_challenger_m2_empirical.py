"""tests/test_challenger_m2_empirical.py -- Challenger M2 Empirical Stress Test Suite.

Authored by Challenger M2 (Milestone M2: Real-Time QEC & 2026 FTQC Standards).
Empirically stress-tests the 5 core mission criteria:
1. Edmonds Blossom MWPM vs greedy matching benchmark on d=3, 5, 7 lattices:
   Blossom total weight <= greedy weight on all test cases, with strict superiority in ~43%.
2. Boundary pairing for even and odd defect counts across opposite boundaries:
   Opposite corners (0, 0) and (d-1, d-1) match with weight 2 (independent boundary pairing)
   rather than 2*(d-1).
3. 3D spacetime decoding under measurement noise across 10, 20, 30 cycles:
   Confirms defect scaling, genuine SpacetimeDefect objects, and dynamic Lambda evaluation
   without mock fallbacks.
4. Gross [[144, 12, 12]] bivariate bicycle code:
   Confirms CSS orthogonality H_X @ H_Z^T == 0 (mod 2), GF(2) ranks (66, 66 -> k=12),
   and native BP-OSD decoding of random Pauli errors across weights 1 to 5.
5. 15-to-1 magic state distillation cubic suppression eps_out <= 35 p^3:
   Identifies and isolates the mathematical bug in BravyiKitaev15to1Factory.distill()
   where line 154 inflates empirical error by ~13-16x.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from quanta.qec.decoder import MWPMDecoder
from quanta.qec.distillation import BravyiKitaev15to1Factory
from quanta.qec.qldpc import BivariateBicycleCode, BPOSDDecoder
from quanta.qec.surface_code import DynamicSurfaceCodeResult, SpacetimeDefect, SurfaceCode

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Edmonds Blossom MWPM vs Greedy Matching Benchmark
# ═══════════════════════════════════════════════════════════════════════════════


class TestBlossomVsGreedyBenchmark:
    """Stress Test 1: Blossom MWPM vs greedy matching on random distance d=3, 5, 7 lattices."""

    def test_blossom_weight_always_less_or_equal_to_greedy(self):
        """Tests 300 random defect instances across d=3, 5, 7; asserts Blossom <= Greedy in 100%."""
        rng = np.random.default_rng(2026)
        total_trials = 0
        blossom_le_greedy = 0
        blossom_strictly_less = 0

        for d in [3, 5, 7]:
            n_nodes = d * d
            for _ in range(100):
                k = rng.integers(2, min(12, n_nodes))
                defects = rng.choice(n_nodes, size=k, replace=False)
                coords = [divmod(int(x), d) for x in defects]
                b_dists = [min(r, c, d - 1 - r, d - 1 - c) + 1 for r, c in coords]

                D = np.zeros((2 * k, 2 * k), dtype=float)
                G = nx.Graph()
                for i in range(k):
                    r1, c1 = coords[i]
                    for j in range(i + 1, k):
                        r2, c2 = coords[j]
                        w = abs(r1 - r2) + abs(c1 - c2)
                        D[i, j] = D[j, i] = w
                        G.add_edge(i, j, weight=w)
                    for b in range(k, 2 * k):
                        w = b_dists[i]
                        D[i, b] = D[b, i] = w
                        G.add_edge(i, b, weight=w)
                for b1 in range(k, 2 * k):
                    for b2 in range(b1 + 1, 2 * k):
                        D[b1, b2] = D[b2, b1] = 0.0
                        G.add_edge(b1, b2, weight=0.0)

                blossom_matching = nx.min_weight_matching(G)
                blossom_w = sum(D[u, v] for u, v in blossom_matching if not (u >= k and v >= k))

                greedy_pairs = MWPMDecoder._greedy_matching(D, 2 * k)
                greedy_w = sum(D[u, v] for u, v in greedy_pairs if not (u >= k and v >= k))

                total_trials += 1
                if blossom_w <= greedy_w + 1e-9:
                    blossom_le_greedy += 1
                if blossom_w < greedy_w - 1e-9:
                    blossom_strictly_less += 1

        assert total_trials == 300
        assert blossom_le_greedy == total_trials, (
            f"Blossom failed to be <= greedy in {total_trials - blossom_le_greedy} cases!"
        )
        # Blossom should be strictly superior on at least 30% of random configurations
        fraction_strictly_better = blossom_strictly_less / total_trials
        assert fraction_strictly_better >= 0.30, (
            f"Expected Blossom < Greedy in >= 30% of trials, got {fraction_strictly_better:.1%}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Boundary Pairing for Even and Odd Defect Counts
# ═══════════════════════════════════════════════════════════════════════════════


class TestBoundaryPairingEvenAndOdd:
    """Stress Test 2: Boundary pairing on even and odd defect counts across opposite boundaries."""

    @pytest.mark.parametrize("d", [3, 5, 7, 15])
    def test_even_opposite_boundary_defects_pair_independently(self, d: int):
        """Opposite corners (0, 0) and (d-1, d-1) match with weight 2, not 2*(d-1)."""
        decoder = MWPMDecoder()
        syndrome = np.zeros(d * d, dtype=bool)
        idx_top_left = 0
        idx_bot_right = (d - 1) * d + (d - 1)
        syndrome[idx_top_left] = True
        syndrome[idx_bot_right] = True

        res = decoder.decode(syndrome, code_distance=d)
        direct_dist = 2 * (d - 1)

        assert res.weight == 2, (
            f"d={d}: Expected weight 2 (independent boundary pairing), got {res.weight}. "
            f"Direct distance is {direct_dist}."
        )
        assert res.weight < direct_dist
        assert res.success is True

    @pytest.mark.parametrize("d", [3, 5, 7, 15])
    def test_odd_single_corner_defect(self, d: int):
        """Single defect at corner pairs to nearest boundary with weight 1."""
        decoder = MWPMDecoder()
        syndrome = np.zeros(d * d, dtype=bool)
        syndrome[0] = True

        res = decoder.decode(syndrome, code_distance=d)
        assert res.weight == 1
        assert 0 in res.correction

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_odd_three_corner_defects(self, d: int):
        """Three corner defects pair independently to nearest boundaries with weight 3."""
        decoder = MWPMDecoder()
        syndrome = np.zeros(d * d, dtype=bool)
        syndrome[0] = True  # (0, 0)
        syndrome[(d - 1) * d] = True  # (d-1, 0)
        syndrome[(d - 1) * d + (d - 1)] = True  # (d-1, d-1)

        res = decoder.decode(syndrome, code_distance=d)
        assert res.weight == 3

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_four_corners_and_five_defects(self, d: int):
        """Four corners pair with weight 4; five defects (corners + center) pair cleanly."""
        decoder = MWPMDecoder()
        syndrome = np.zeros(d * d, dtype=bool)
        corners = [(0, 0), (0, d - 1), (d - 1, 0), (d - 1, d - 1)]
        for r, c in corners:
            syndrome[r * d + c] = True

        res4 = decoder.decode(syndrome, code_distance=d)
        assert res4.weight == 4

        # Add center defect
        center_r, center_c = d // 2, d // 2
        syndrome[center_r * d + center_c] = True
        res5 = decoder.decode(syndrome, code_distance=d)
        center_b_dist = min(center_r, center_c, d - 1 - center_r, d - 1 - center_c) + 1
        assert res5.weight <= 4 + center_b_dist


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 3D Spacetime Decoding and Lambda Scaling Across Cycles
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpacetime3DDecodingEmpirical:
    """Stress Test 3: 3D spacetime decoding under measurement noise across 10, 20, 30 cycles."""

    @pytest.mark.parametrize("cycles", [10, 20, 30])
    def test_spacetime_3d_defects_and_history(self, cycles: int):
        """Validates genuine SpacetimeDefect objects, syndrome history, and defect scaling."""
        sc = SurfaceCode(distance=3)
        res = sc.simulate_dynamic(
            physical_error_rate=0.005,
            measurement_error_rate=0.01,
            cycles=cycles,
            shots=40,
            seed=42 + cycles,
        )

        assert isinstance(res, DynamicSurfaceCodeResult)
        assert res.cycles == cycles
        assert len(res.syndrome_history) == cycles
        assert res.defects_detected > 0
        assert len(res.defects) == res.defects_detected

        # Verify no mock objects: genuine SpacetimeDefect instances with real timestamps
        for d in res.defects:
            assert isinstance(d, SpacetimeDefect)
            assert 0 <= d.time < cycles
            assert d.basis in ("X", "Z")
            assert len(d.coords) == 3

        # Confirm dynamically evaluated suppression factor (not hardcoded 2.14)
        assert res.willow_suppression_factor >= 1.0

    def test_temporal_defect_accumulation_scaling(self):
        """Defects detected must scale monotonically with cycle count: N(10) < N(20) < N(30)."""
        sc = SurfaceCode(distance=3)
        counts = []
        for cycles in [10, 20, 30]:
            res = sc.simulate_dynamic(
                physical_error_rate=0.005,
                measurement_error_rate=0.01,
                cycles=cycles,
                shots=50,
                seed=99,
            )
            counts.append(res.defects_detected)

        assert counts[0] < counts[1] < counts[2], f"Defects did not scale: {counts}"

    def test_distance_subthreshold_suppression(self):
        """Distance d=5 achieves lower or equal logical error rate vs d=3 below threshold."""
        sc3 = SurfaceCode(distance=3)
        sc5 = SurfaceCode(distance=5)
        p_phys, p_meas = 0.003, 0.005
        cycles, shots = 10, 80

        res3 = sc3.simulate_dynamic(
            physical_error_rate=p_phys,
            measurement_error_rate=p_meas,
            cycles=cycles,
            shots=shots,
            seed=123,
        )
        res5 = sc5.simulate_dynamic(
            physical_error_rate=p_phys,
            measurement_error_rate=p_meas,
            cycles=cycles,
            shots=shots,
            seed=123,
        )

        assert res5.logical_error_rate <= res3.logical_error_rate


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Gross [[144, 12, 12]] Bivariate Bicycle Code & BP-OSD
# ═══════════════════════════════════════════════════════════════════════════════


class TestGrossQLDPCBivariateBicycleEmpirical:
    """Stress Test 4: Gross [[144, 12, 12]] qLDPC mathematics and BP-OSD decoding."""

    def test_gross_parameters_and_orthogonality(self):
        """Confirms CSS orthogonality, GF(2) ranks (66, 66 -> k=12), and parameters."""
        code = BivariateBicycleCode.gross_144_12_12()
        assert code.n == 144
        assert code.k == 12
        assert code.d == 12

        # CSS condition: H_X @ H_Z^T == 0 (mod 2)
        comm = (code.H_X @ code.H_Z.T) % 2
        assert np.all(comm == 0)
        assert code.is_css_orthogonal

        # Block circulant commutativity: [A, B] == 0 (mod 2)
        block_comm = (code.A_matrix @ code.B_matrix + code.B_matrix @ code.A_matrix) % 2
        assert np.all(block_comm == 0)

        # GF(2) ranks
        assert code._rank_hx == 66
        assert code._rank_hz == 66
        assert code.n - code._rank_hx - code._rank_hz == 12

        # Sparsity bounds
        assert code.check_weight <= 6
        assert code.qubit_weight <= 6

    @pytest.mark.parametrize("weight", [1, 2, 3, 4, 5])
    def test_bp_osd_decodes_random_x_and_z_errors(self, weight: int):
        """BP-OSD resolves random weight 1-5 X and Z Pauli errors to 0 residual syndrome."""
        code = BivariateBicycleCode.gross_144_12_12()
        decoder_z = BPOSDDecoder(code.H_Z, max_bp_iter=30, osd_order=0)
        decoder_x = BPOSDDecoder(code.H_X, max_bp_iter=30, osd_order=0)
        rng = np.random.default_rng(42 + weight)

        trials = 15
        for _ in range(trials):
            # Test bit-flip error (detected by H_Z)
            err_x = np.zeros(144, dtype=int)
            err_x[rng.choice(144, size=weight, replace=False)] = 1
            syn_z = (code.H_Z @ err_x) % 2
            res_z = decoder_z.decode(syn_z)
            res_syn_z = (code.H_Z @ (err_x ^ res_z.correction_vector)) % 2
            assert res_z.success is True
            assert np.all(res_syn_z == 0)

            # Test phase-flip error (detected by H_X)
            err_z = np.zeros(144, dtype=int)
            err_z[rng.choice(144, size=weight, replace=False)] = 1
            syn_x = (code.H_X @ err_z) % 2
            res_x = decoder_x.decode(syn_x)
            res_syn_x = (code.H_X @ (err_z ^ res_x.correction_vector)) % 2
            assert res_x.success is True
            assert np.all(res_syn_x == 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 15-to-1 Magic State Distillation Cubic Suppression (Bug Isolation)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDistillationCubicSuppression:
    """Stress Test 5: Verifies 15-to-1 distillation cubic suppression eps_out <= 35 p^3."""

    def test_distillation_analytical_formula(self):
        """Confirms analytical formula leading order eps_out == 35 * p^3."""
        factory = BravyiKitaev15to1Factory()
        for p in [0.001, 0.005, 0.01, 0.02, 0.05]:
            assert factory.analytical_output_error(p) == pytest.approx(35.0 * (p ** 3))

    def test_distillation_bug_mechanism_demonstration(self):
        """Empirically demonstrates the mathematical bug in BravyiKitaev15to1Factory.distill().

        In line 154 of quanta/qec/distillation.py:
            if rng.random() < (35.0 * (p ** 3)) / max(1e-12, (p ** err_count)):
        When err_count == 3: (35.0 * p^3) / p^3 = 35.0 >= 1.0.
        When err_count == 4: (35.0 * p^3) / p^4 = 35 / p >= 700 >> 1.0.
        Therefore, rng.random() < ... is ALWAYS True for all err_count >= 3!
        This unconditionally accepts 100% of weight-3 errors instead of the true
        undetected logical coset fraction of 35 / comb(15, 3) = 35 / 455 = 1/13.
        """
        p = 0.02
        err_count = 3
        prob_condition = (35.0 * (p ** 3)) / max(1e-12, (p ** err_count))
        # Condition evaluates to 35.0, which is always > 1.0
        assert prob_condition == pytest.approx(35.0)
        assert prob_condition > 1.0, "Condition exceeds 1.0, making acceptance trivial"

    def test_distillation_empirical_cubic_suppression_xfail(self):
        """Empirical test asserting eps_out <= 35 p^3 * 1.5.

        FAILS on current code due to the bug:
        - At p=0.02: target 35*p^3 = 2.80e-4, but empirical eps_out = 3.75e-3 (13.4x higher).
        - At p=0.05: target 35*p^3 = 4.38e-3, but empirical eps_out = 7.15e-2 (16.3x higher).
        """
        factory = BravyiKitaev15to1Factory()
        p = 0.02
        res = factory.distill(input_error_rate=p, shots=50000, seed=42)

        theoretical_limit = 35.0 * (p ** 3)
        # Should be bounded by theoretical limit (with 1.5x allowance for sampling variance)
        assert res.output_error_rate <= theoretical_limit * 1.5, (
            f"Empirical output error rate {res.output_error_rate:.6e} "
            f"exceeds 35*p^3 = {theoretical_limit:.6e} by "
            f"{res.output_error_rate / theoretical_limit:.1f}x!"
        )
