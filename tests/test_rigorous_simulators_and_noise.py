"""
tests/test_rigorous_simulators_and_noise.py

Comprehensive mathematical and physical verification suite for:
  1. MPSSimulator mixed-canonical orthogonality, SVD cutoff, exact Born-rule
     marginal sampling, native 3Q/4Q gate decompositions (CCX, CSWAP, RCCX, RC3X)
     at 30-50+ qubits, and stochastic Kraus noise trajectories.
  2. PauliFrameSimulator Aaronson-Gottesman commutation phase signs, full Clifford
     gate set (H, S, SDG, SX, SXdg, X, Y, Z, CX, CZ, SWAP, ECR, iSWAP, I),
     exact analytical GF(2) affine-subspace Born probabilities, and seed preservation.
  3. Vectorized DensityMatrixSimulator superoperator contraction, CPTP completeness,
     traceless ZZ Crosstalk Kraus operators, channel arity routing, and multi-trajectory
     static noise execution in quanta.runner.run().
  4. Hardware backend ISA & endianness invariants across IBMBackend, IBMRestBackend
     (all 25 Heron ISA gates at unit process fidelity + SamplerV2 shots & bit order),
     GoogleBackend (n-qubit zero padding), IonQBackend (symmetric swap targets +
     multinomial sampling), CuStateVecSimulator (little-endian wire & target ordering),
     and accelerated.tensor_contract (.get() host transfer).
"""

from __future__ import annotations

import math
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quanta.backends.google import GoogleBackend
from quanta.backends.ibm import IBMBackend
from quanta.backends.ibm_rest import IBMRestBackend
from quanta.backends.ionq import IonQBackend
from quanta.core.circuit import circuit
from quanta.core.gates import (
    CX,
    GATE_REGISTRY,
    SWAP,
    H,
    MultiParametricGate,
    ParametricGate,
    X,
)
from quanta.core.measure import measure
from quanta.core.types import QuantaError
from quanta.dag.dag_circuit import DAGCircuit
from quanta.runner import CircuitRunner, run
from quanta.simulator.accelerated import tensor_contract
from quanta.simulator.density_matrix import DensityMatrixSimulator
from quanta.simulator.factory import create_simulator
from quanta.simulator.mlx import MLXSimulator, is_mlx_available
from quanta.simulator.mps import MPSSimulator
from quanta.simulator.noise import (
    AmplitudeDamping,
    BitFlip,
    Crosstalk,
    Depolarizing,
    NoiseModel,
    PhaseFlip,
    ReadoutError,
    T2Relaxation,
)
from quanta.simulator.pauli_frame import PauliFrameSimulator
from quanta.simulator.router import analyze_circuit, select_simulator
from quanta.simulator.statevector import StateVectorSimulator

# ═══════════════════════════════════════════════════════════════════════════
# 1. MPSSimulator Mixed-Canonical Form, Born Sampling, 3Q/4Q & Kraus Noise
# ═══════════════════════════════════════════════════════════════════════════


class TestMPSSimulatorRigorous:
    """Exact tensor-network invariants for MPSSimulator."""

    def test_unbalanced_entanglement_born_sampling(self) -> None:
        """RY(0.4)(0); CX(0, 1) must sample ~96.05% '00' and ~3.95% '11' (never 50/50)."""
        sim = MPSSimulator(2, seed=2026)
        sim.apply("RY", (0,), (0.4,))
        sim.apply("CX", (0, 1))

        # Before sampling, ortho_center is at site 1 from _apply_2q_adjacent(0, 1)
        assert sim._ortho_center == 1

        shots = 25_000
        counts = sim.sample(shots)
        assert sim._ortho_center == 0  # Moved once before shot loop

        assert set(counts.keys()) <= {"00", "11"}
        p00_exact = math.cos(0.2) ** 2  # ~0.9605305
        p11_exact = math.sin(0.2) ** 2  # ~0.0394695
        p00_emp = counts.get("00", 0) / shots
        p11_emp = counts.get("11", 0) / shots

        assert abs(p00_emp - p00_exact) < 0.008
        assert abs(p11_emp - p11_exact) < 0.008

    def test_multiqubit_unbalanced_mps_sampling_matches_statevector(self) -> None:
        """Sequential conditional marginals P(q_i | q_{<i}) match exact |psi|^2 on 5 qubits."""
        mps = MPSSimulator(5, seed=77, chi_max=32)
        sv = StateVectorSimulator(5, seed=77)

        ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = [
            ("RY", (0,), (0.5,)),
            ("CX", (0, 3), ()),
            ("RX", (3,), (1.1,)),
            ("CX", (3, 1), ()),
            ("RY", (2,), (0.3,)),
            ("CZ", (1, 4), ()),
            ("CX", (2, 4), ()),
        ]
        for g, qs, ps in ops:
            mps.apply(g, qs, ps)
            sv.apply(g, qs, ps)

        np.testing.assert_allclose(mps.probabilities(), sv.probabilities(), atol=1e-13)

        shots = 30_000
        counts = mps.sample(shots)
        emp = np.zeros(32, dtype=float)
        for bitstr, cnt in counts.items():
            emp[int(bitstr, 2)] = cnt / shots

        assert np.max(np.abs(emp - sv.probabilities())) < 0.012

    def test_swap_and_product_gates_keep_unit_bond_dimension(self) -> None:
        """SWAP and separable 2Q evolutions prune zero singular values via cutoff."""
        sim = MPSSimulator(10, chi_max=64, cutoff=1e-12)
        assert sim.cutoff == 1e-12

        for q in range(10):
            sim.apply("RX", (q,), (0.17 * (q + 1),))
            sim.apply("RZ", (q,), (0.31 * (q + 1),))

        # Long-range SWAPs on an unentangled product state must keep bond dimension 1
        sim.apply("SWAP", (0, 9))
        sim.apply("SWAP", (2, 7))
        sim.apply("SWAP", (8, 1))

        assert sim.max_bond_dim == 1
        assert sim.bond_dimensions == [1] * 9
        assert abs(sim.norm() - 1.0) < 1e-13

    def test_mixed_canonical_isometry_and_cutoff_truncation_tracking(self) -> None:
        """_move_ortho_center maintains strict left/right isometries and tracks truncation error."""
        sim = MPSSimulator(6, chi_max=2, cutoff=1e-12)
        for q in range(6):
            sim.apply("H", (q,))
        for q in range(5):
            sim.apply("RZZ", (q, q + 1), (0.65,))
            sim.apply("RX", (q,), (0.45,))
        # Entangle across cut 2 with long-range gates so rank > 2
        sim.apply("CX", (0, 4))
        sim.apply("CX", (1, 5))

        assert sim.max_bond_dim <= 2
        assert sim.truncation_error > 0.0
        assert abs(sim.norm() - 1.0) < 1e-12

        sim._move_ortho_center(3)
        assert sim._ortho_center == 3
        for left_idx in range(3):
            t = sim._tensors[left_idx]
            chi_l, d, chi_r = t.shape
            mat = t.reshape(chi_l * d, chi_r)
            np.testing.assert_allclose(mat.conj().T @ mat, np.eye(chi_r), atol=1e-12)
        for right_idx in range(4, 6):
            t = sim._tensors[right_idx]
            chi_l, d, chi_r = t.shape
            mat = t.reshape(chi_l, d * chi_r)
            np.testing.assert_allclose(mat @ mat.conj().T, np.eye(chi_l), atol=1e-12)

    @pytest.mark.parametrize(
        ("gate_name", "qubits"),
        [
            ("CCX", (0, 1, 2)),
            ("CCX", (4, 1, 3)),
            ("CSWAP", (0, 1, 2)),
            ("CSWAP", (3, 4, 0)),
            ("RCCX", (2, 0, 4)),
            ("RC3X", (3, 0, 4, 1)),
        ],
    )
    def test_mps_3q_and_4q_decompositions_match_statevector_exact(
        self, gate_name: str, qubits: tuple[int, ...]
    ) -> None:
        """Decomposed CCX, CSWAP, RCCX, RC3X match exact statevector up to global phase."""
        mps = MPSSimulator(5, chi_max=64)
        sv = StateVectorSimulator(5)

        # Prepare rich non-trivial entangled input state across all 5 qubits
        prep = [
            ("RY", (0,), (0.73,)),
            ("RX", (1,), (1.19,)),
            ("H", (2,), ()),
            ("RY", (3,), (2.05,)),
            ("RX", (4,), (0.61,)),
            ("CX", (0, 2), ()),
            ("CZ", (1, 3), ()),
            ("RZ", (2,), (0.44,)),
            ("RZ", (4,), (0.91,)),
        ]
        for g, qs, ps in prep:
            mps.apply(g, qs, ps)
            sv.apply(g, qs, ps)

        mps.apply(gate_name, qubits)
        sv.apply(gate_name, qubits)

        fidelity = abs(np.vdot(sv.state, mps.state))
        assert abs(fidelity - 1.0) < 1e-13

    def test_mps_3q_toffoli_and_fredkin_on_40_qubits(self) -> None:
        """30-40+ qubit MPS executes CCX, CSWAP, and RCCX natively in O(chi^3) without dense fallback."""
        sim = MPSSimulator(40, chi_max=16, seed=99)
        sim.apply("X", (0,))
        sim.apply("X", (19,))
        sim.apply("CCX", (0, 19, 39))  # Flips q39 -> 1
        sim.apply("CSWAP", (0, 39, 5))  # Swaps q39=1 into q5=1, q39=0
        sim.apply("RCCX", (0, 5, 10))  # Flips q10 -> 1 (with -i phase)

        counts = sim.sample(64)
        expected = ["0"] * 40
        expected[0] = "1"
        expected[5] = "1"
        expected[10] = "1"
        expected[19] = "1"
        expected_str = "".join(expected)
        assert counts == {expected_str: 64}
        assert sim.max_bond_dim <= 4

    def test_mps_stochastic_kraus_trajectories_converge_to_density_matrix(self) -> None:
        """Ensemble average of MPS Kraus jumps matches exact CPTP density matrix."""
        nm = NoiseModel().add(AmplitudeDamping(0.3)).add(Crosstalk(0.25, angle=0.8))
        dm = DensityMatrixSimulator(2)
        dm.apply("H", (0,))
        dm.apply_noise(nm, (0,))
        dm.apply("CX", (0, 1))
        dm.apply_noise(nm, (0, 1))

        rng = np.random.default_rng(12345)
        n_traj = 2500
        rho_emp = np.zeros((4, 4), dtype=complex)
        for _ in range(n_traj):
            mps = MPSSimulator(2, chi_max=8)
            mps.apply("H", (0,))
            mps.apply_noise(nm, (0,), rng)
            mps.apply("CX", (0, 1))
            mps.apply_noise(nm, (0, 1), rng)
            psi = mps.state
            rho_emp += np.outer(psi, psi.conj())
        rho_emp /= n_traj

        assert np.max(np.abs(rho_emp - dm.state)) < 0.035


# ═══════════════════════════════════════════════════════════════════════════
# 2. PauliFrameSimulator Stabilizer Algebra & Exact Analytical Marginals
# ═══════════════════════════════════════════════════════════════════════════


class TestPauliFrameRigorous:
    """Exact Gottesman-Knill stabilizer phase & affine subspace tests."""

    def test_commutation_phase_sign_bell_minus_state(self) -> None:
        """H(0); H(1); CZ(0,1); SDG(0); H(0); SDG(1); H(1) yields (|00> - |11>)/sqrt(2)."""
        pf = PauliFrameSimulator(2, seed=42)
        sv = StateVectorSimulator(2, seed=42)

        seq = [
            ("H", (0,)),
            ("H", (1,)),
            ("CZ", (0, 1)),
            ("SDG", (0,)),
            ("H", (0,)),
            ("SDG", (1,)),
            ("H", (1,)),
        ]
        for g, qs in seq:
            pf.apply(g, qs)
            sv.apply(g, qs)

        # Computational basis probabilities: 50% '00', 50% '11'
        # (Old buggy _rowmult yielded 50% '01', 50% '10' due to inverted XZ/ZX signs!)
        np.testing.assert_allclose(pf.probabilities(), [0.5, 0.0, 0.0, 0.5], atol=1e-15)
        np.testing.assert_allclose(pf.probabilities(), sv.probabilities(), atol=1e-15)

        # Probe relative phase (+i on |11>): CX(0,1) -> SDG(0) -> H(0) maps deterministically to |00>
        pf.apply("CX", (0, 1))
        pf.apply("SDG", (0,))
        pf.apply("H", (0,))
        sv.apply("CX", (0, 1))
        sv.apply("SDG", (0,))
        sv.apply("H", (0,))
        np.testing.assert_allclose(pf.probabilities(), [1.0, 0.0, 0.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(pf.probabilities(), sv.probabilities(), atol=1e-15)

    def test_random_clifford_circuits_all_gates_match_statevector_exact(self) -> None:
        """Random Clifford circuits including SX, SXdg, ECR, iSWAP match StateVector to < 1e-14."""
        rng = np.random.default_rng(314159)
        one_q_gates = ("H", "S", "SDG", "SX", "SXdg", "X", "Y", "Z", "I")
        two_q_gates = ("CX", "CZ", "SWAP", "ECR", "iSWAP")

        for trial in range(30):
            n = int(rng.integers(2, 7))
            pf = PauliFrameSimulator(n, seed=100 + trial)
            sv = StateVectorSimulator(n, seed=100 + trial)

            for _ in range(25):
                if rng.random() < 0.55:
                    g = one_q_gates[int(rng.integers(0, len(one_q_gates)))]
                    q = int(rng.integers(0, n))
                    pf.apply(g, (q,))
                    sv.apply(g, (q,))
                else:
                    g = two_q_gates[int(rng.integers(0, len(two_q_gates)))]
                    q0, q1 = rng.choice(n, size=2, replace=False)
                    pf.apply(g, (int(q0), int(q1)))
                    sv.apply(g, (int(q0), int(q1)))

            tab_before = pf.state
            probs_pf = pf.probabilities()
            probs_sv = sv.probabilities()

            # Exact analytical floats (zero sampling noise) and non-mutating tableau
            np.testing.assert_array_equal(pf.state, tab_before)
            np.testing.assert_allclose(probs_pf, probs_sv, atol=1e-14)
            assert abs(float(np.sum(probs_pf)) - 1.0) < 1e-15

    def test_probabilities_qubit_guard_and_seed_preservation_across_reset(self) -> None:
        """probabilities() raises QuantaError for n > 26 and reset() preserves seed."""
        big_pf = PauliFrameSimulator(27, seed=11)
        big_pf.apply("H", (0,))
        with pytest.raises(QuantaError, match="26"):
            big_pf.probabilities()

        # 20-qubit GHZ exact analytical probabilities (zero sampling noise!)
        pf20 = PauliFrameSimulator(20, seed=19)
        pf20.apply("H", (0,))
        for q in range(19):
            pf20.apply("CX", (q, q + 1))
        probs20 = pf20.probabilities()
        assert probs20[0] == 0.5
        assert probs20[-1] == 0.5
        assert float(np.sum(probs20)) == 1.0

        # Seed preservation across reset() in PauliFrameSimulator
        pf_seed = PauliFrameSimulator(3, seed=777)
        pf_seed.apply("H", (0,))
        pf_seed.apply("H", (1,))
        c1 = pf_seed.sample(shots=64)
        pf_seed.reset()
        pf_seed.apply("H", (0,))
        pf_seed.apply("H", (1,))
        c2 = pf_seed.sample(shots=64)
        assert c1 == c2

        # Router & Factory Clifford gate detection and seed forwarding
        analysis = analyze_circuit(["H", "SDG", "SX", "SXdg", "ECR", "iSWAP", "I"], num_qubits=40)
        assert analysis["is_clifford"] is True
        assert analysis["recommended_method"] == "clifford"

        routed = select_simulator(40, ["H", "SDG", "ECR", "iSWAP"], seed=555)
        assert isinstance(routed, PauliFrameSimulator)
        assert routed._seed == 555

        fac_pf = create_simulator(4, method="pauli_frame", seed=555)
        assert isinstance(fac_pf, PauliFrameSimulator)
        assert fac_pf._seed == 555

        if is_mlx_available():
            mlx_sim = MLXSimulator(2, seed=888)
            mlx_sim.apply("H", (0,))
            s1 = mlx_sim.sample(64)
            mlx_sim.reset()
            mlx_sim.apply("H", (0,))
            s2 = mlx_sim.sample(64)
            assert s1 == s2


# ═══════════════════════════════════════════════════════════════════════════
# 3. Vectorized DensityMatrixSimulator, CPTP Crosstalk & Multi-Trajectory Noise
# ═══════════════════════════════════════════════════════════════════════════


class TestDensityMatrixAndNoisyRunner:
    """CPTP completeness, vectorized superoperator evolution, and runner trajectories."""

    def test_vectorized_density_matrix_matches_statevector_and_rejects_non_cptp(self) -> None:
        """Vectorized [2]*(2n) contraction matches |psi><psi| on non-adjacent 1Q/2Q/3Q ops."""
        dm = DensityMatrixSimulator(4, seed=42)
        sv = StateVectorSimulator(4, seed=42)

        ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = [
            ("H", (0,), ()),
            ("RY", (2,), (0.85,)),
            ("CX", (2, 0), ()),
            ("ECR", (1, 3), ()),
            ("CCX", (0, 3, 1), ()),
            ("iSWAP", (3, 2), ()),
        ]
        for g, qs, ps in ops:
            dm.apply(g, qs, ps)
            sv.apply(g, qs, ps)

        expected_rho = np.outer(sv.state, sv.state.conj())
        np.testing.assert_allclose(dm.state, expected_rho, atol=1e-14)
        np.testing.assert_allclose(dm.probabilities(), sv.probabilities(), atol=1e-14)

        # Non-CPTP Kraus operators must raise QuantaError
        bad_kraus = [0.5 * np.eye(2, dtype=complex)]
        with pytest.raises(QuantaError, match="completeness"):
            dm.apply_kraus(bad_kraus, (0,))

    def test_all_noise_channels_exact_cptp_completeness_and_crosstalk_zz(self) -> None:
        """Every gate-level NoiseChannel satisfies sum K_k† K_k == I to machine precision."""
        channels = [
            Depolarizing(0.15),
            BitFlip(0.2),
            PhaseFlip(0.25),
            AmplitudeDamping(0.35),
            T2Relaxation(0.4),
            Crosstalk(0.18, angle=0.62),
        ]
        for ch in channels:
            kraus = ch.kraus_operators()
            dim = kraus[0].shape[0]
            comp = sum(k.conj().T @ k for k in kraus)
            assert np.max(np.abs(comp - np.eye(dim))) < 1e-14

        # Verify Crosstalk coherent unitary is traceless ZZ: diag(e^{-ia/2}, e^{ia/2}, e^{ia/2}, e^{-ia/2})
        angle = 0.74
        ct = Crosstalk(1.0, angle=angle)
        _, k1 = ct.kraus_operators()
        expected_zz = np.diag([
            np.exp(-0.5j * angle),
            np.exp(0.5j * angle),
            np.exp(0.5j * angle),
            np.exp(-0.5j * angle),
        ])
        np.testing.assert_allclose(k1, expected_zz, atol=1e-15)

        # Verify arity routing: 1Q gate on 3-qubit register routes 2Q Crosstalk to neighbor (q, q+1) or (q-1, q)
        nm = NoiseModel().add(Crosstalk(1.0, angle=math.pi / 2)).add(ReadoutError(0.1, 0.1))
        dm = DensityMatrixSimulator(3)
        dm.apply("H", (2,))
        # q=2 is the last qubit, so Crosstalk acts on (1, 2) while ReadoutError is skipped!
        dm.apply_noise(nm, (2,))
        assert abs(np.trace(dm.state) - 1.0) < 1e-14
        np.testing.assert_allclose(dm.state, dm.state.conj().T, atol=1e-14)

    @pytest.mark.parametrize(
        ("num_qubits", "method", "use_backend_str"),
        [
            (1, "dense", False),           # Exact CPTP DensityMatrix path (n <= 11)
            (13, "dense", False),          # Independent Monte Carlo trajectories path (n > 11)
            (2, "density_matrix", False),  # Explicit density_matrix method
            (4, "mps", False),             # MPS Kraus trajectory path
            (4, "pauli_frame", False),     # Stabilizer Pauli trajectory path
            (2, "mps", True),              # String backend="mps"
        ],
    )
    def test_static_noise_x_x_bitflip_02_gives_68_32_distribution(
        self, num_qubits: int, method: str, use_backend_str: bool
    ) -> None:
        """X(0); X(0) with BitFlip(0.2) must yield ~68% q0='0' and ~32% q0='1' (never 100%/0%)."""
        @circuit(qubits=num_qubits)
        def x_twice(q):
            X(q[0])
            X(q[0])
            return measure(q[0])

        nm = NoiseModel().add(BitFlip(0.2))
        shots = 2500
        if use_backend_str:
            res = run(x_twice, shots=shots, seed=2026, backend=method, noise=nm)
        else:
            res = run(x_twice, shots=shots, seed=2026, method=method, noise=nm)

        assert res.statevector is None
        assert sum(res.counts.values()) == shots
        p0 = res.counts.get("0", 0) / shots
        p1 = res.counts.get("1", 0) / shots

        # Analytical: P(0) = 0.8^2 + 0.2^2 = 0.68, P(1) = 2 * 0.8 * 0.2 = 0.32
        assert 0.63 <= p0 <= 0.73, f"Expected ~0.68 for ({num_qubits}, {method}), got {p0:.4f}"
        assert 0.27 <= p1 <= 0.37, f"Expected ~0.32 for ({num_qubits}, {method}), got {p1:.4f}"

    @pytest.mark.skipif(not is_mlx_available(), reason="Apple MLX not available")
    def test_string_backend_mlx_noiseless_and_noisy_trajectories(self) -> None:
        """run(circ, backend='mlx') works both noiselessly and with per-shot noise trajectories."""
        @circuit(qubits=2)
        def x_twice_mlx(q):
            X(q[0])
            X(q[0])
            return measure(q[0])

        res_clean = run(x_twice_mlx, shots=256, seed=42, backend="mlx")
        assert res_clean.counts == {"0": 256}
        assert res_clean.statevector is not None

        nm = NoiseModel().add(BitFlip(0.2))
        res_noisy = run(x_twice_mlx, shots=2000, seed=42, backend="mlx", noise=nm)
        assert res_noisy.statevector is None
        p0 = res_noisy.counts.get("0", 0) / 2000
        p1 = res_noisy.counts.get("1", 0) / 2000
        assert 0.63 <= p0 <= 0.73
        assert 0.27 <= p1 <= 0.37

        runner = CircuitRunner(backend="density_matrix", shots=256, seed=7)
        res_dm = runner.run(x_twice_mlx)
        assert res_dm.counts == {"0": 256}


# ═══════════════════════════════════════════════════════════════════════════
# 4. Hardware Backend ISA Fidelity & Endianness Verification
# ═══════════════════════════════════════════════════════════════════════════


def _eval_heron_isa_unitary(isa_ops: list[str], num_qubits: int) -> np.ndarray:
    """Evaluates a sequence of Heron QASM 3.0 ISA instructions into an exact 2^n x 2^n unitary."""
    dim = 1 << num_qubits
    u_total = np.eye(dim, dtype=complex)

    sx_mat = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)
    x_mat = np.array([[0, 1], [1, 0]], dtype=complex)
    cz_mat = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)

    for inst in isa_ops:
        s = inst.strip().rstrip(";")
        if not s:
            continue
        m_rz = re.fullmatch(r"rz\(([^)]+)\)\s+\$(\d+)", s)
        if m_rz:
            angle = float(m_rz.group(1))
            q = int(m_rz.group(2))
            rz_mat = np.array(
                [[np.exp(-0.5j * angle), 0.0], [0.0, np.exp(0.5j * angle)]],
                dtype=complex,
            )
            u_step = _embed_gate_unitary(rz_mat, (q,), num_qubits)
            u_total = u_step @ u_total
            continue

        m_1q = re.fullmatch(r"(sx|x)\s+\$(\d+)", s)
        if m_1q:
            gname, q_str = m_1q.group(1), m_1q.group(2)
            mat = sx_mat if gname == "sx" else x_mat
            u_step = _embed_gate_unitary(mat, (int(q_str),), num_qubits)
            u_total = u_step @ u_total
            continue

        m_cz = re.fullmatch(r"cz\s+\$(\d+),\s*\$(\d+)", s)
        if m_cz:
            q0, q1 = int(m_cz.group(1)), int(m_cz.group(2))
            u_step = _embed_gate_unitary(cz_mat, (q0, q1), num_qubits)
            u_total = u_step @ u_total
            continue

        raise AssertionError(f"Non-Heron-ISA instruction emitted: {inst!r}")

    return u_total


def _embed_gate_unitary(
    gate_matrix: np.ndarray, qubits: tuple[int, ...], num_qubits: int
) -> np.ndarray:
    """Embeds a k-qubit gate matrix into an n-qubit operator using Quanta's big-endian convention."""
    dim = 1 << num_qubits
    cols = []
    sv = StateVectorSimulator(num_qubits)
    k = len(qubits)
    gate_tensor = gate_matrix.reshape([2] * (2 * k))
    gate_axes = list(range(k, 2 * k))
    state_axes = list(qubits)
    for basis_idx in range(dim):
        psi = np.zeros(dim, dtype=complex)
        psi[basis_idx] = 1.0
        res = np.tensordot(gate_tensor, psi.reshape([2] * num_qubits), axes=(gate_axes, state_axes))
        res = np.moveaxis(res, list(range(k)), state_axes)
        cols.append(res.reshape(-1))
    del sv
    return np.column_stack(cols)


class TestHardwareBackendsISAAndEndianness:
    """Exact unitary process fidelity and bitstring endianness across all hardware adapters."""

    def test_ibm_rest_all_25_gates_have_unit_process_fidelity(self) -> None:
        """Every gate in IBMRestBackend._gate_to_isa achieves |Tr(U_isa† U_ref)| / 2^k == 1.0."""
        supported_gates = [
            "H", "X", "Y", "Z", "S", "T", "CX", "CZ", "CY", "CCX",
            "SWAP", "RX", "RY", "RZ", "I", "SDG", "TDG", "P", "SX", "SXdg",
            "U", "RXX", "RZZ", "RCCX", "RC3X",
        ]
        rng = np.random.default_rng(20261111)

        for gate_name in supported_gates:
            gate_obj = GATE_REGISTRY[gate_name]
            if isinstance(gate_obj, MultiParametricGate):
                n_qubits = 1
                trials = [tuple(rng.uniform(-math.pi, math.pi, size=3)) for _ in range(10)]
                ref_mats = [gate_obj(*p).matrix for p in trials]
            elif isinstance(gate_obj, ParametricGate):
                n_qubits = gate_obj.num_qubits
                trials = [(float(rng.uniform(-math.pi, math.pi)),) for _ in range(10)]
                ref_mats = [gate_obj(p[0]).matrix for p in trials]
            else:
                n_qubits = gate_obj.num_qubits
                trials = [()]
                ref_mats = [gate_obj.matrix]

            qubits = tuple(range(n_qubits))
            dim = 1 << n_qubits
            for params, u_ref in zip(trials, ref_mats, strict=True):
                isa_ops = IBMRestBackend._gate_to_isa(gate_name, qubits, params)
                u_isa = _eval_heron_isa_unitary(isa_ops, n_qubits)
                process_fidelity = abs(np.trace(u_isa.conj().T @ u_ref)) / dim
                assert abs(process_fidelity - 1.0) < 1e-13, (
                    f"Gate {gate_name}{params} failed ISA fidelity: {process_fidelity}"
                )

    def test_ibm_rest_submit_sampler_shots_payload_and_little_endian_counts(self) -> None:
        """submit_sampler sends shots in pubs & options, and _parse_counts reverses c[0]=q[0]."""
        @circuit(qubits=3)
        def x_on_q0(q):
            X(q[0])
            return measure(q)

        dag = DAGCircuit.from_builder(x_on_q0.build())
        backend = IBMRestBackend(api_key="dummy-key", backend_name="ibm_torino")

        captured_payloads: list[dict[str, Any]] = []

        def fake_api_call(method: str, endpoint: str, data: dict | None = None) -> dict:
            if method == "POST" and endpoint == "/jobs":
                assert data is not None
                captured_payloads.append(data)
                return {"id": "job-123", "status": "Queued"}
            raise AssertionError(f"Unexpected call {method} {endpoint}")

        with patch.object(backend, "_api_call", side_effect=fake_api_call):
            job = backend.submit_sampler(dag, shots=1536)

        assert job.job_id == "job-123"
        assert len(captured_payloads) == 1
        params = captured_payloads[0]["params"]
        assert params["pubs"][0][1] is None
        assert params["pubs"][0][2] == 1536
        assert params["options"]["default_shots"] == 1536

        # In QASM 3, c[0] = measure $0 => q[0]=1, q[1]=0, q[2]=0 => integer 0x1 ("001" little-endian)
        # Quanta big-endian requires "100"!
        raw_samples = {
            "results": [{"data": {"c": {"num_bits": 3, "samples": ["0x1", "0x1", "0x4"]}}}]
        }
        assert backend._parse_counts(raw_samples) == {"100": 2, "001": 1}

        raw_precounted = {
            "results": [{"data": {"c": {"num_bits": 3, "counts": {"001": 7, "100": 3}}}}]
        }
        assert backend._parse_counts(raw_precounted) == {"100": 7, "001": 3}

    def test_ibm_backend_reverses_qiskit_little_endian_counts(self) -> None:
        """IBMBackend reverses Qiskit little-endian '001' (c[0]=1) to Quanta big-endian '100'."""
        @circuit(qubits=3)
        def x_q0(q):
            X(q[0])
            return measure(q)

        dag = DAGCircuit.from_builder(x_q0.build())

        # 1. Local Aer path
        mock_qc = MagicMock()
        mock_qiskit = SimpleNamespace(QuantumCircuit=MagicMock(from_qasm_str=MagicMock(return_value=mock_qc)))
        mock_aer_res = MagicMock()
        mock_aer_res.get_counts.return_value = {"001": 100}
        mock_aer_sim = MagicMock()
        mock_aer_sim.run.return_value.result.return_value = mock_aer_res
        mock_qiskit_aer = SimpleNamespace(AerSimulator=MagicMock(return_value=mock_aer_sim))

        with patch.dict("sys.modules", {"qiskit": mock_qiskit, "qiskit_aer": mock_qiskit_aer}):
            local_backend = IBMBackend(simulate_locally=True)
            res_local = local_backend.execute(dag, shots=100, seed=42)
            assert res_local.counts == {"100": 100}

        # 2. Hardware SamplerV2 path
        pub_res = SimpleNamespace(data=SimpleNamespace(c=SimpleNamespace(get_counts=lambda: {"001": 80, "010": 20})))
        mock_sampler = MagicMock()
        mock_sampler.run.return_value.result.return_value = [pub_res]
        mock_runtime = SimpleNamespace(
            QiskitRuntimeService=MagicMock(),
            SamplerV2=MagicMock(return_value=mock_sampler),
        )
        with patch.dict("sys.modules", {"qiskit": mock_qiskit, "qiskit_ibm_runtime": mock_runtime}):
            hw_backend = IBMBackend(token="tok-xyz", simulate_locally=False)
            res_hw = hw_backend.execute(dag, shots=100)
            assert res_hw.counts == {"100": 80, "010": 20}

    def test_google_backend_preserves_full_qubit_width_on_all_zero_state(self) -> None:
        """|000> on 3 qubits must format as '000', NEVER '0'."""
        @circuit(qubits=3)
        def all_zero(q):
            return measure(q)

        dag = DAGCircuit.from_builder(all_zero.build())

        mock_cirq_res = MagicMock()
        mock_cirq_res.histogram.return_value = {0: 1024}
        mock_sim = MagicMock()
        mock_sim.run.return_value = mock_cirq_res
        mock_cirq = MagicMock()
        mock_cirq.Simulator.return_value = mock_sim

        with patch.dict("sys.modules", {"cirq": mock_cirq}):
            backend = GoogleBackend(simulate_locally=True)
            res = backend.execute(dag, shots=1024, seed=1)
            assert res.counts == {"000": 1024}
            assert "0" not in res.counts

    def test_ionq_backend_swap_targets_schema_and_stochastic_sampling(self) -> None:
        """IonQ emits {'gate': 'swap', 'targets': [q0, q1]} and samples via rng.choice."""
        @circuit(qubits=3)
        def ionq_circ(q):
            H(q[0])
            CX(q[0], q[1])
            SWAP(q[1], q[2])
            return measure(q)

        dag = DAGCircuit.from_builder(ionq_circ.build())
        backend = IonQBackend(target="simulator", api_key="dummy")
        ops = backend._dag_to_ionq_circuit(dag)

        assert ops[0] == {"gate": "h", "target": 0}
        assert ops[1] == {"gate": "cnot", "control": 0, "target": 1}
        assert ops[2] == {"gate": "swap", "targets": [1, 2]}
        assert "control" not in ops[2]

        # Stochastic sampling check: prob=0.02 at shots=10 used to deterministically round(0.2)=0 ALWAYS!
        saw_tail = False
        for s in range(40):
            c = IonQBackend._parse_results({"0": 0.98, "1": 0.02}, num_qubits=2, shots=10, seed=s)
            assert sum(c.values()) == 10
            if c.get("10", 0) > 0:
                saw_tail = True
        assert saw_tail, "Multinomial sampling must occasionally draw rare state '10' (state_idx=1 -> q0=1,q1=0)"

        # Large-shot convergence & endianness check (IonQ state "1" = q0=1, q1=0 -> "10")
        c_large = IonQBackend._parse_results({"0": 0.75, "1": 0.25}, num_qubits=2, shots=20_000, seed=42)
        assert sum(c_large.values()) == 20_000
        assert abs(c_large["00"] / 20_000 - 0.75) < 0.015
        assert abs(c_large["10"] / 20_000 - 0.25) < 0.015

    def test_custatevec_little_endian_wire_and_target_ordering_matches_statevector(self) -> None:
        """Software emulator of cuStateVec's little-endian apply_matrix matches StateVectorSimulator."""
        # Build a true mathematical emulator of NVIDIA cuStateVec's little-endian wire convention:
        #   - State index bit w (2^w) is cuStateVec wire w.
        #   - In targets = [t_0, ..., t_{k-1}], t_0 is LSB (bit 0) of gate matrix, t_{k-1} is MSB (bit k-1).
        state_store: dict[int, np.ndarray] = {}

        class FakeCuPyArray:
            def __init__(self, arr: np.ndarray) -> None:
                self._arr = np.array(arr, dtype=np.complex128, copy=True)
                state_store[id(self._arr)] = self._arr

            @property
            def data(self) -> SimpleNamespace:
                return SimpleNamespace(ptr=id(self._arr))

            @property
            def size(self) -> int:
                return self._arr.size

            def fill(self, val: complex) -> None:
                self._arr.fill(val)

            def __setitem__(self, idx: Any, val: Any) -> None:
                self._arr[idx] = val

            def __getitem__(self, idx: Any) -> Any:
                return self._arr[idx]

            def __abs__(self) -> FakeCuPyArray:
                return FakeCuPyArray(np.abs(self._arr))

            def __pow__(self, power: float) -> FakeCuPyArray:
                return FakeCuPyArray(self._arr ** power)

            def get(self) -> np.ndarray:
                return self._arr.copy()

        class FakeCuPyModule:
            complex128 = np.complex128
            cuda = SimpleNamespace(alloc=lambda sz: SimpleNamespace(ptr=1, size=sz))

            @staticmethod
            def zeros(dim: int, dtype: Any = np.complex128) -> FakeCuPyArray:
                return FakeCuPyArray(np.zeros(dim, dtype=dtype))

            @staticmethod
            def asarray(arr: Any, dtype: Any = np.complex128, order: str = "C") -> FakeCuPyArray:
                del order
                if isinstance(arr, FakeCuPyArray):
                    return arr
                return FakeCuPyArray(np.asarray(arr, dtype=dtype))

            @staticmethod
            def asnumpy(arr: Any) -> np.ndarray:
                return arr.get() if isinstance(arr, FakeCuPyArray) else np.asarray(arr)

            @staticmethod
            def abs(arr: FakeCuPyArray) -> FakeCuPyArray:
                return abs(arr)

        def fake_apply_matrix(
            handle: Any,
            sv_ptr: int,
            sv_dtype: Any,
            n_index_bits: int,
            mat_ptr: int,
            mat_dtype: Any,
            layout: Any,
            adjoint: int,
            n_targets: int,
            targets: list[int],
            n_controls: int,
            controls: list[int],
            control_bits: list[int],
            compute_type: Any,
            ws_ptr: int,
            ws_size: int,
        ) -> None:
            del handle, sv_dtype, mat_dtype, layout, adjoint, n_controls, controls, control_bits, compute_type, ws_ptr, ws_size
            sv_buf = state_store[sv_ptr]
            mat_buf = state_store[mat_ptr]
            dim = 1 << n_index_bits
            out = np.zeros_like(sv_buf)
            sub_dim = 1 << n_targets

            for idx in range(dim):
                amp = sv_buf[idx]
                if amp == 0.0:
                    continue
                # Extract input column col_bits where targets[b] is bit b of matrix column
                col = 0
                base_idx = idx
                for b, wire in enumerate(targets):
                    if (idx >> wire) & 1:
                        col |= 1 << b
                        base_idx &= ~(1 << wire)
                for row in range(sub_dim):
                    m_val = mat_buf[row, col]
                    if m_val == 0.0:
                        continue
                    dst_idx = base_idx
                    for b, wire in enumerate(targets):
                        if (row >> b) & 1:
                            dst_idx |= 1 << wire
                    out[dst_idx] += m_val * amp

            sv_buf[:] = out

        fake_cusv = SimpleNamespace(
            create=lambda: 99,
            destroy=lambda h: None,
            cudaDataType=SimpleNamespace(CUDA_C_64F=1),
            MatrixLayout=SimpleNamespace(ROW=0),
            ComputeType=SimpleNamespace(COMPUTE_64F=1),
            apply_matrix_get_workspace_size=lambda *args: 64,
            apply_matrix=fake_apply_matrix,
        )

        import quanta.simulator.custatevec as cusv_mod

        with (
            patch.object(cusv_mod, "HAS_CUQUANTUM", True),
            patch.object(cusv_mod, "cp", FakeCuPyModule, create=True),
            patch.object(cusv_mod, "cusv", fake_cusv, create=True),
        ):
            gpu_sim = cusv_mod.CuStateVecSimulator(3, seed=42)
            ref_sim = StateVectorSimulator(3, seed=42)

            asymmetric_ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = [
                ("RY", (0,), (0.71,)),
                ("RX", (2,), (1.23,)),
                ("CX", (0, 2), ()),
                ("CX", (2, 1), ()),
                ("CCX", (2, 0, 1), ()),
                ("CZ", (0, 1), ()),
            ]
            for g, qs, ps in asymmetric_ops:
                gpu_sim.apply(g, qs, ps)
                ref_sim.apply(g, qs, ps)

            np.testing.assert_allclose(gpu_sim.state, ref_sim.state, atol=1e-13)
            np.testing.assert_allclose(gpu_sim.probabilities(), ref_sim.probabilities(), atol=1e-13)

            gpu_sim.reset()
            np.testing.assert_allclose(gpu_sim.state, [1, 0, 0, 0, 0, 0, 0, 0], atol=1e-15)

    def test_accelerated_tensor_contract_handles_device_array_with_get(self) -> None:
        """tensor_contract uses .get() when device array forbids implicit __array__."""
        class StrictDeviceArray:
            def __init__(self, arr: np.ndarray) -> None:
                self._arr = np.asarray(arr)

            def reshape(self, *shape: Any) -> StrictDeviceArray:
                return StrictDeviceArray(self._arr.reshape(*shape))

            def get(self) -> np.ndarray:
                return self._arr.copy()

            def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
                raise TypeError("Implicit conversion to a NumPy array is not allowed")

        class FakeCuPyBackend:
            @staticmethod
            def asarray(x: Any) -> StrictDeviceArray:
                return x if isinstance(x, StrictDeviceArray) else StrictDeviceArray(np.asarray(x))

            @staticmethod
            def tensordot(a: StrictDeviceArray, b: StrictDeviceArray, axes: Any) -> StrictDeviceArray:
                return StrictDeviceArray(np.tensordot(a._arr, b._arr, axes=axes))

            @staticmethod
            def moveaxis(a: StrictDeviceArray, src: Any, dst: Any) -> StrictDeviceArray:
                return StrictDeviceArray(np.moveaxis(a._arr, src, dst))

        import quanta.simulator.accelerated as acc_mod

        with (
            patch.object(acc_mod, "_initialized", True),
            patch.object(acc_mod, "_backend_name", "cupy"),
            patch.object(acc_mod, "_xp", FakeCuPyBackend),
        ):
            psi0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
            psi1 = tensor_contract(H.matrix, psi0, (0,), 2)
            psi2 = tensor_contract(CX.matrix, psi1, (0, 1), 2)
            assert isinstance(psi2, np.ndarray)
            np.testing.assert_allclose(psi2, [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)], atol=1e-14)
