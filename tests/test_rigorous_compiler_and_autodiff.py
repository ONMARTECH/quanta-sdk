"""Rigorous mathematical & physical verification suite for Compiler, Synthesis, Routing, QASM, Autodiff, QAOA, DLA & Torch."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.stats import unitary_group
from torch.autograd import gradcheck

from quanta import circuit, measure, run
from quanta.compiler.passes.optimize import CancelInverses, IdentityRemoval, MergeRotations
from quanta.compiler.passes.routing import RouteToTopology, Topology
from quanta.compiler.passes.translate import GATE_SETS, TranslateToTarget
from quanta.compiler.synthesis import heron_1q_decompose, kak_decompose, zyz_decompose
from quanta.core.custom_gate import custom_gate
from quanta.core.gates import (
    CCX,
    CH,
    CP,
    CRX,
    CRY,
    CRZ,
    CSWAP,
    CX,
    CY,
    CZ,
    ECR,
    GATE_REGISTRY,
    MS,
    RC3X,
    RCCX,
    RX,
    RXX,
    RY,
    RZ,
    RZZ,
    SDG,
    SWAP,
    SX,
    TDG,
    H,
    I,
    P,
    S,
    SXdg,
    T,
    U,
    X,
    Y,
    Z,
    iSWAP,
)
from quanta.core.types import QuantaError
from quanta.dag.dag_circuit import DAGCircuit
from quanta.export.qasm import to_qasm
from quanta.export.qasm_import import QASMImportError, from_qasm
from quanta.gradients import (
    adjoint_gradient,
    expectation,
    finite_diff,
    multi_expectation,
    natural_gradient,
    parameter_shift,
    quantum_fisher_information,
)
from quanta.primitives import Estimator
from quanta.primitives.qaoa import QAOA
from quanta.qml.lie_algebra import (
    hilbert_schmidt_inner_product,
    is_barren_plateau_immune,
    is_in_algebra,
    lie_algebra_dimension,
    pauli_dla,
)
from quanta.simulator.noise import Depolarizing, NoiseModel
from quanta.simulator.statevector import StateVectorSimulator
from quanta.torch import QuantumLayer, ops

# ═══════════════════════════════════════════════════════════════════════════
#  Helpers: Exact Unitary Extraction & Hilbert-Schmidt Process Fidelity
# ═══════════════════════════════════════════════════════════════════════════


def _dag_to_unitary(dag: DAGCircuit) -> np.ndarray:
    """Extracts the exact 2^n x 2^n unitary matrix of a static DAGCircuit."""
    n = dag.num_qubits
    dim = 1 << n
    cols: list[np.ndarray] = []
    op_list = dag.op_nodes()
    for basis_idx in range(dim):
        sim = StateVectorSimulator(n)
        init = np.zeros(dim, dtype=np.complex128)
        init[basis_idx] = 1.0
        sim.state = init
        for op in op_list:
            sim.apply(op.gate_name, op.qubits, op.params)
        cols.append(sim.state.copy())
    return np.column_stack(cols)


def _insts_to_unitary(insts: list, num_qubits: int) -> np.ndarray:
    """Extracts the exact 2^n x 2^n unitary matrix of an Instruction sequence."""
    dim = 1 << num_qubits
    cols: list[np.ndarray] = []
    for basis_idx in range(dim):
        sim = StateVectorSimulator(num_qubits)
        init = np.zeros(dim, dtype=np.complex128)
        init[basis_idx] = 1.0
        sim.state = init
        for inst in insts:
            sim.apply(inst.gate_name, inst.qubits, inst.params)
        cols.append(sim.state.copy())
    return np.column_stack(cols)


def _process_fidelity(u_target: np.ndarray, u_synth: np.ndarray) -> float:
    """Hilbert-Schmidt process fidelity |Tr(U_target^† U_synth)|^2 / d^2 in [0, 1]."""
    d = u_target.shape[0]
    overlap = np.trace(u_target.conj().T @ u_synth)
    return float((abs(overlap) ** 2) / (d * d))


# ═══════════════════════════════════════════════════════════════════════════
#  1. Gates, Controlled Unitaries & Physical Relative Phases
# ═══════════════════════════════════════════════════════════════════════════


class TestGatesAndControlledRelativePhases:
    """Verifies CRX/CRY/CRZ, controlled(), inverse, and RCCX/RC3X unitarity."""

    @pytest.mark.parametrize("angle", [-2.3, -0.7, 0.0, 0.85, math.pi])
    def test_crx_cry_crz_exact_matrices_and_unitarity(self, angle: float) -> None:
        for gate_cls, base_cls, name in [
            (CRX, RX, "CRX"),
            (CRY, RY, "CRY"),
            (CRZ, RZ, "CRZ"),
        ]:
            assert name in GATE_REGISTRY
            bound = gate_cls(angle)
            mat = bound.matrix
            expected = np.eye(4, dtype=np.complex128)
            expected[2:, 2:] = base_cls(angle).matrix
            np.testing.assert_allclose(mat, expected, atol=1e-15)
            np.testing.assert_allclose(mat.conj().T @ mat, np.eye(4), atol=1e-15)

    def test_fixed_gate_controlled_executes_in_circuit_run(self) -> None:
        @circuit(qubits=3)
        def ctrl_circ(q):
            X(q[0])
            X(q[1])
            X.controlled(1)(q[0], q[2])
            Z.controlled(2)(q[0], q[1], q[2])
            return measure(q)

        res = run(ctrl_circ, shots=256, seed=7)
        assert res.counts == {"111": 256}

    def test_bound_parametric_controlled_preserves_global_to_relative_phase(self) -> None:
        """RZ(π)|0⟩ = -i|0⟩ vs P(π)|0⟩ = +|0⟩: controlling on |+⟩|0⟩ must distinguish relative phase."""

        @circuit(qubits=2)
        def crz_interference(q):
            H(q[0])
            RZ(math.pi).controlled(1)(q[0], q[1])
            H(q[0])
            return measure(q)

        @circuit(qubits=2)
        def cp_interference(q):
            H(q[0])
            P(math.pi).controlled(1)(q[0], q[1])
            H(q[0])
            return measure(q)

        # For P(π).controlled(1) on |+0⟩ = (|00⟩ + |10⟩)/√2: |10⟩ gets +1 => H(q[0]) gives 100% |00⟩
        res_cp = run(cp_interference, shots=200, seed=11)
        assert res_cp.counts == {"00": 200}

        # For RZ(π).controlled(1) on |+0⟩: |10⟩ gets e^{-iπ/2} = -i => after H(q[0]),
        # P(00) = |(1 - i)/2|^2 = 0.5 and P(10) = |(1 + i)/2|^2 = 0.5!
        dag_crz = DAGCircuit.from_builder(crz_interference.build())
        u_crz = _dag_to_unitary(dag_crz)
        psi_crz = u_crz[:, 0]
        probs = np.abs(psi_crz) ** 2
        assert probs[0] == pytest.approx(0.5, abs=1e-14)
        assert probs[2] == pytest.approx(0.5, abs=1e-14)

    def test_bound_parametric_and_multiparametric_inverse_and_controlled(self) -> None:
        @circuit(qubits=2)
        def inv_ctrl(q):
            H(q[0])
            RY(0.7).controlled(1)(q[0], q[1])
            RY(0.7).inverse.controlled(1)(q[0], q[1])
            U(0.6, 0.4, 1.1)(q[1])
            U(0.6, 0.4, 1.1).inverse(q[1])
            RZ(0.5)(q[0])
            RZ(0.5).inverse(q[0])
            H(q[0])
            return measure(q)

        res = run(inv_ctrl, shots=256, seed=42)
        assert res.counts == {"00": 256}

    def test_rccx_and_rc3x_strict_unitarity_and_truth_table(self) -> None:
        for gate_obj, n_q in [(RCCX, 3), (RC3X, 4)]:
            mat = gate_obj.matrix
            dim = 1 << n_q
            np.testing.assert_allclose(mat.conj().T @ mat, np.eye(dim), atol=1e-15)
            np.testing.assert_allclose(mat @ mat.conj().T, np.eye(dim), atol=1e-15)

            for idx in range(dim):
                out = mat[:, idx]
                probs = np.abs(out) ** 2
                expected_idx = idx ^ 1 if (idx >> 1) == ((1 << (n_q - 1)) - 1) else idx
                assert probs[expected_idx] == pytest.approx(1.0, abs=1e-15)


# ═══════════════════════════════════════════════════════════════════════════
#  2. Universal 1Q ZYZ / Heron & 2Q Cartan/KAK SU(4) Gate Synthesis
# ═══════════════════════════════════════════════════════════════════════════


class TestUniversalSynthesisAndTranslate:
    """Verifies ZYZ, Heron 1Q, Cartan/KAK 2Q Haar-random synthesis & target translation."""

    def test_zyz_decompose_haar_and_gimbal_locks(self) -> None:
        rng = np.random.default_rng(2026)
        specials = [
            I.matrix,
            X.matrix,
            Y.matrix,
            Z.matrix,
            H.matrix,
            S.matrix,
            T.matrix,
            SX.matrix,
            RZ(0.9).matrix,
            RY(math.pi).matrix,
            RY(1e-14).matrix,
            RY(math.pi - 1e-14).matrix,
        ]
        matrices = specials + [unitary_group.rvs(2, random_state=rng) for _ in range(50)]

        for u_mat in matrices:
            alpha, beta, gamma, delta = zyz_decompose(u_mat)
            recon = (
                np.exp(1j * alpha)
                * (RZ(beta).matrix @ RY(gamma).matrix @ RZ(delta).matrix)
            )
            err = float(np.linalg.norm(u_mat - recon, ord="fro"))
            assert err < 1e-13

    def test_heron_1q_decompose_haar_and_gate_budget(self) -> None:
        rng = np.random.default_rng(2027)
        allowed = {"RZ", "SX", "X"}
        matrices = [I.matrix, X.matrix, Y.matrix, Z.matrix, H.matrix, SX.matrix] + [
            unitary_group.rvs(2, random_state=rng) for _ in range(50)
        ]

        for u_mat in matrices:
            insts = heron_1q_decompose(u_mat, qubit=0)
            names = [inst.gate_name for inst in insts]
            assert set(names) <= allowed
            assert names.count("SX") <= 2
            assert names.count("RZ") <= 3
            u_synth = _insts_to_unitary(insts, num_qubits=1)
            assert _process_fidelity(u_mat, u_synth) == pytest.approx(1.0, abs=1e-14)

    def test_kak_decompose_60_haar_random_u4_and_named_gates(self) -> None:
        rng = np.random.default_rng(2028)
        named_2q = [
            np.kron(H.matrix, S.matrix),  # 0 CX
            CX.matrix,  # 1 CX
            CZ.matrix,  # 1 CX
            CY.matrix,  # 1 CX
            CH.matrix,  # 1 CX
            ECR.matrix,  # 1 CX
            iSWAP.matrix,  # 2 CX
            RXX(0.8).matrix @ RZZ(-1.1).matrix,  # 2 CX
            SWAP.matrix,  # 3 CX
        ]
        haar_2q = [unitary_group.rvs(4, random_state=rng) for _ in range(60)]

        # Check optimal CX counts on canonical classes
        assert sum(1 for g in kak_decompose(np.kron(H.matrix, T.matrix)) if g.gate_name == "CX") == 0
        assert sum(1 for g in kak_decompose(ECR.matrix) if g.gate_name == "CX") == 1
        assert sum(1 for g in kak_decompose(iSWAP.matrix) if g.gate_name == "CX") == 2

        for u_mat in named_2q + haar_2q:
            insts = kak_decompose(u_mat, qubits=(0, 1))
            cx_count = sum(1 for inst in insts if inst.gate_name == "CX")
            assert cx_count <= 3
            u_synth = _insts_to_unitary(insts, num_qubits=2)
            fid = _process_fidelity(u_mat, u_synth)
            assert fid == pytest.approx(1.0, abs=1e-13)

    def test_translate_iswap_and_ecr_process_fidelity_exact_one(self) -> None:
        for gate_obj in (iSWAP, ECR):
            @circuit(qubits=2)
            def two_q_circ(q, g=gate_obj):
                g(q[0], q[1])

            dag = DAGCircuit.from_builder(two_q_circ.build())
            for preset in ("ibm", "ibm_heron", "google", "ionq", "universal"):
                translated = TranslateToTarget(preset).run(dag)
                illegal = {op.gate_name for op in translated.op_nodes()} - GATE_SETS[preset]
                assert not illegal
                u_trans = _dag_to_unitary(translated)
                assert _process_fidelity(gate_obj.matrix, u_trans) == pytest.approx(1.0, abs=1e-13)

    @pytest.mark.parametrize(
        "preset",
        ["ibm", "ibm_heron", "google", "google_willow", "ionq", "ionq_aria", "quantinuum", "universal"],
    )
    def test_translate_all_hardware_targets_universal_fallback(self, preset: str) -> None:
        rng = np.random.default_rng(99)
        haar_u4 = unitary_group.rvs(4, random_state=rng)
        custom_2q = custom_gate(f"Haar2Q_{preset}", haar_u4)

        @circuit(qubits=3)
        def rich_circ(q):
            H(q[0])
            RY(0.73)(q[1])
            T(q[2])
            iSWAP(q[0], q[1])
            ECR(q[1], q[2])
            CRY(0.61)(q[0], q[2])
            RY(0.45).controlled(1)(q[1], q[0])
            custom_2q(q[0], q[2])
            CCX(q[0], q[1], q[2])
            RCCX(q[2], q[0], q[1])

        dag = DAGCircuit.from_builder(rich_circ.build())
        u_orig = _dag_to_unitary(dag)

        translated = TranslateToTarget(preset).run(dag)
        illegal = {op.gate_name for op in translated.op_nodes()} - GATE_SETS[preset]
        assert not illegal, f"Illegal gates {illegal} remained for preset {preset}"

        u_trans = _dag_to_unitary(translated)
        assert _process_fidelity(u_orig, u_trans) == pytest.approx(1.0, abs=1e-12)


# ═══════════════════════════════════════════════════════════════════════════
#  3. Routing Permutations, Heavy-Hex Topology, Optimization & QASM 3.0
# ═══════════════════════════════════════════════════════════════════════════


class TestRoutingOptimizationAndQASM:
    """Verifies SWAP bitstring preservation, 3Q routing, Heavy-Hex, wire optimization & QASM 3.0."""

    def test_routing_preserves_exact_logical_bitstring_order_and_3q_fidelity(self) -> None:
        @circuit(qubits=5)
        def non_local_circ(q):
            X(q[0])
            RY(0.8)(q[3])
            CX(q[0], q[4])  # non-adjacent on line(5)
            CCX(q[0], q[4], q[2])  # 3-qubit Toffoli across non-adjacent qubits
            CSWAP(q[2], q[4], q[1])  # swaps |1⟩ on q[4] into q[1] -> |111xx⟩
            return measure(q)

        dag = DAGCircuit.from_builder(non_local_circ.build())
        u_orig = _dag_to_unitary(dag)
        psi_orig = u_orig[:, 0]
        probs_orig = np.abs(psi_orig) ** 2

        topo = Topology.line(5)

        # Mode 1: Default restore_layout=True -> full unitary & statevector match logical order
        router_restore = RouteToTopology(topo, restore_layout=True)
        routed_restore = router_restore.run(dag)
        for op in routed_restore.op_nodes():
            assert len(op.qubits) <= 2
            if len(op.qubits) == 2:
                assert topo.is_connected(op.qubits[0], op.qubits[1])

        u_routed = _dag_to_unitary(routed_restore)
        assert _process_fidelity(u_orig, u_routed) == pytest.approx(1.0, abs=1e-12)

        res_orig = run(dag, shots=1024, seed=123)
        res_routed = run(routed_restore, shots=1024, seed=123)
        for bitstr, p in enumerate(probs_orig):
            b = format(bitstr, "05b")
            if p < 1e-12:
                assert res_routed.counts.get(b, 0) == 0
                assert res_orig.counts.get(b, 0) == 0

        # Mode 2: restore_layout=False -> MeasureSpec remapping still returns logical bitstring order!
        router_remap = RouteToTopology(topo, restore_layout=False)
        routed_remap = router_remap.run(dag)
        res_remap = run(routed_remap, shots=2048, seed=123)
        for idx, p in enumerate(probs_orig):
            b = format(idx, "05b")
            emp = res_remap.counts.get(b, 0) / 2048.0
            assert abs(emp - p) < 0.04

    def test_heavy_hex_topology_degree_bounds_and_backend_presets(self) -> None:
        hh = Topology.heavy_hex(2, 3)
        degrees = [len(hh.neighbors(q)) for q in range(hh.num_qubits)]
        assert max(degrees) <= 3
        assert 2 in degrees and 3 in degrees

        for backend_name, expected_n in [("ibm_fez", 156), ("ibm_brisbane", 127)]:
            topo = Topology.from_backend(backend_name)
            assert topo.num_qubits == expected_n
            degs = [len(topo.neighbors(q)) for q in range(topo.num_qubits)]
            assert max(degs) <= 3, f"{backend_name} had non-heavy-hex degree {max(degs)}"
            assert degs.count(2) > 0 and degs.count(3) > 0
            # Connectedness check
            assert len(topo.shortest_path(0, expected_n - 1)) >= 2

    def test_cancel_inverses_and_merge_rotations_per_wire_and_angle_wrapping(self) -> None:
        @circuit(qubits=3)
        def interleaved_cancel(q):
            H(q[0])
            H(q[1])
            Z(q[2])
            H(q[0])  # cancels H(q[0]) across H(q[1]) and Z(q[2])!
            S(q[1])
            Z(q[2])  # cancels Z(q[2])
            SDG(q[1])  # cancels S(q[1]), exposing H(q[1])!
            H(q[1])  # cascading cancellation on wire 1!

        dag = DAGCircuit.from_builder(interleaved_cancel.build())
        opt = CancelInverses().run(dag)
        assert opt.gate_count() == 0

        @circuit(qubits=2)
        def rotation_wrap(q):
            RZ(0.2)(q[0])
            RY(0.4)(q[1])
            RZ(0.3)(q[0])
            RY(0.6)(q[1])
            RZ(-0.5 - 1e-12)(q[0])  # sums to -1e-12 -> cancels completely!
            # 2π wrap-around cancellation on wire 0:
            RX(1.5 * math.pi)(q[0])
            RX(0.5 * math.pi + 1e-12)(q[0])

        dag_rot = DAGCircuit.from_builder(rotation_wrap.build())
        merged = IdentityRemoval().run(MergeRotations().run(dag_rot))
        ops_out = merged.op_nodes()
        assert len(ops_out) == 1
        assert ops_out[0].gate_name == "RY"
        assert ops_out[0].qubits == (1,)
        assert ops_out[0].params[0] == pytest.approx(1.0, abs=1e-12)

    def test_qasm3_roundtrip_all_34_sdk_gates_and_strict_error_on_unknown(self) -> None:
        @circuit(qubits=4)
        def all_gates_circ(q):
            H(q[0])
            X(q[1])
            Y(q[2])
            Z(q[3])
            S(q[0])
            SDG(q[1])
            T(q[2])
            TDG(q[3])
            SX(q[0])
            SXdg(q[1])
            I(q[2])
            RX(0.31)(q[0])
            RY(0.52)(q[1])
            RZ(0.73)(q[2])
            P(0.94)(q[3])
            U(0.41, 0.62, 0.83)(q[0])
            CX(q[0], q[1])
            CZ(q[1], q[2])
            CY(q[2], q[3])
            CH(q[3], q[0])
            SWAP(q[0], q[2])
            iSWAP(q[1], q[3])
            ECR(q[0], q[1])
            RXX(0.37)(q[1], q[2])
            RZZ(0.59)(q[2], q[3])
            CRX(0.44)(q[0], q[2])
            CRY(0.66)(q[1], q[3])
            CRZ(0.88)(q[2], q[0])
            CP(1.05)(q[3], q[1])
            MS(0.48)(q[0], q[3])
            CCX(q[0], q[1], q[2])
            CSWAP(q[1], q[2], q[3])
            RCCX(q[0], q[2], q[3])
            RC3X(q[0], q[1], q[2], q[3])
            return measure(q)

        dag_orig = DAGCircuit.from_builder(all_gates_circ.build())
        qasm_str = to_qasm(dag_orig)
        dag_roundtrip = from_qasm(qasm_str)

        assert dag_roundtrip.gate_count() == 34
        u_orig = _dag_to_unitary(dag_orig)
        u_rt = _dag_to_unitary(dag_roundtrip)
        assert _process_fidelity(u_orig, u_rt) == pytest.approx(1.0, abs=1e-13)

        # Unrecognized statements MUST raise QASMImportError (subclass of QuantaError)
        with pytest.raises(QASMImportError):
            from_qasm('OPENQASM 3.0;\nqubit[2] q;\nbogus_gate q[0];')
        assert issubclass(QASMImportError, QuantaError)

        with pytest.raises(QASMImportError):
            from_qasm('OPENQASM 3.0;\nqubit[2] q;\ncx q[0];')


# ═══════════════════════════════════════════════════════════════════════════
#  4. QFIM, 4-Term Shift, Adjoint Diff, Estimator, QAOA, DLA & Torch Layer
# ═══════════════════════════════════════════════════════════════════════════


class TestAutodiffVariationalPhysicsAndTorch:
    """Verifies exact QFIM, 4-term shift, O(G) adjoint diff, noisy Estimator, Egger QAOA, DLA, and QuantumLayer."""

    def test_qfim_exact_on_ry_and_phase_gate_and_entangled_off_diagonal(self) -> None:
        # 1. RY(θ) on cos(φ/2)|0⟩ + e^{iχ} sin(φ/2)|1⟩: exact F_θθ = 1 - <ψ0|Y|ψ0>^2
        phi, chi = 0.9, 0.4
        psi_init = np.array(
            [math.cos(phi / 2), np.exp(1j * chi) * math.sin(phi / 2)], dtype=complex
        )
        y_exp = float(np.real(psi_init.conj() @ Y.matrix @ psi_init))
        expected_f_ry = 1.0 - y_exp ** 2

        def ry_state(p: np.ndarray) -> np.ndarray:
            return RY(p[0]).matrix @ psi_init

        f_ry = quantum_fisher_information(ry_state, [0.73])
        assert f_ry[0, 0] == pytest.approx(expected_f_ry, abs=1e-13)

        # Specifically on |0⟩: F_θθ == 1.0 (NOT 2.0!)
        f_ry_zero = quantum_fisher_information(
            lambda p: RY(p[0]).matrix @ np.array([1.0, 0.0], dtype=complex), [0.73]
        )
        assert f_ry_zero[0, 0] == pytest.approx(1.0, abs=1e-14)
        ng = natural_gradient(
            lambda p: math.cos(p[0]),
            lambda p: RY(p[0]).matrix @ np.array([1.0, 0.0], dtype=complex),
            [0.73],
            regularization=0.0,
        )
        assert ng.gradients[0] == pytest.approx(-math.sin(0.73), abs=1e-12)

        # 2. P(θ) on α|0⟩ + β|1⟩: exact F_θθ = 4 |α|^2 |β|^2 (gauge-invariant Berry subtraction!)
        alpha = math.sqrt(0.3) * np.exp(0.2j)
        beta = math.sqrt(0.7) * np.exp(-0.5j)
        psi_ab = np.array([alpha, beta], dtype=complex)
        expected_f_p = 4.0 * (abs(alpha) ** 2) * (abs(beta) ** 2)

        for shift_val in (math.pi / 2.0, 0.7):
            f_p = quantum_fisher_information(
                lambda p: P(p[0]).matrix @ psi_ab, [1.1], shift=shift_val
            )
            assert f_p[0, 0] == pytest.approx(expected_f_p, abs=1e-13)

        # 3. Entangled 3-parameter circuit off-diagonal & evaluation count check
        call_counter = 0

        def ent_state(p: np.ndarray) -> np.ndarray:
            nonlocal call_counter
            call_counter += 1
            sim = StateVectorSimulator(2)
            sim.apply("H", (0,))
            sim.apply("P", (0,), (float(p[0]),))
            sim.apply("RY", (1,), (float(p[1]),))
            sim.apply("CX", (0, 1))
            sim.apply("RZ", (1,), (float(p[2]),))
            return sim.state

        p0 = np.array([0.55, 1.15, -0.65])
        call_counter = 0
        f_mat = quantum_fisher_information(ent_state, p0)
        # 1 + 2*3 + 4*(3*2/2) = 19 calls (NOT O(p^2) redundant single-shifts!)
        assert call_counter == 19

        # Compare against 8th-order finite-difference Fubini-Study metric
        eps = 1e-6
        psi0 = ent_state(p0)
        dpsi = []
        for idx in range(3):
            e = np.zeros(3)
            e[idx] = eps
            dpsi.append((ent_state(p0 + e) - ent_state(p0 - e)) / (2.0 * eps))
        f_fd = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                f_fd[i, j] = 4.0 * float(
                    np.real(
                        np.vdot(dpsi[i], dpsi[j])
                        - np.vdot(dpsi[i], psi0) * np.vdot(psi0, dpsi[j])
                    )
                )
        np.testing.assert_allclose(f_mat, f_fd, atol=1e-8)

    def test_four_term_parameter_shift_exact_on_crx_cry_crz(self) -> None:
        def controlled_cost(p: np.ndarray) -> float:
            sim = StateVectorSimulator(2)
            sim.apply("RY", (0,), (0.9,))  # partial superposition on control!
            sim.apply("RY", (1,), (0.4,))
            sim.apply("CX", (1, 0))  # entangle control & target before CRY!
            sim.apply("CRY", (0, 1), (float(p[0]),))
            sim.apply("CRX", (1, 0), (float(p[1]),))
            sim.apply("CRZ", (0, 1), (float(p[2]),))
            sim.apply("RY", (0,), (float(p[3]),))
            return multi_expectation(sim.state, [("ZX", 0.7), ("YY", -0.4), ("IZ", 0.5)], 2)

        p0 = np.array([0.63, -0.87, 1.12, 0.41])
        res_auto = parameter_shift(
            controlled_cost, p0, rule="auto", gate_types=["CRY", "CRX", "CRZ", "RY"]
        )
        res_4term = parameter_shift(controlled_cost, p0, rule="four-term")
        assert res_auto.num_evaluations == 1 + 4 * 3 + 2 * 1

        # Verify against 4-point O(h^4) Richardson finite difference (< 1e-12 accuracy)
        h = 1e-4
        exact_grad = np.zeros(4)
        for i in range(4):
            e = np.zeros(4)
            e[i] = h
            exact_grad[i] = (
                8.0 * (controlled_cost(p0 + e) - controlled_cost(p0 - e))
                - (controlled_cost(p0 + 2 * e) - controlled_cost(p0 - 2 * e))
            ) / (12.0 * h)

        np.testing.assert_allclose(res_auto.gradients, exact_grad, atol=1e-11)
        np.testing.assert_allclose(res_4term.gradients, exact_grad, atol=1e-11)

    def test_adjoint_gradient_o_g_agreement_to_1e_9(self) -> None:
        @circuit(qubits=4)
        def deep_var(q, p):
            RY(p[0])(q[0])
            RX(p[1])(q[1])
            RZ(p[2])(q[2])
            P(p[3])(q[3])
            U(p[4], p[5], p[6])(q[0])
            CX(q[0], q[1])
            RXX(p[7])(q[1], q[2])
            RZZ(p[8])(q[2], q[3])
            CRX(p[9])(q[0], q[2])
            CRY(p[10])(q[1], q[3])
            CRZ(p[11])(q[2], q[0])
            CP(p[12])(q[3], q[1])
            MS(p[13])(q[0], q[3])
            # Shared & affine-scaled parameter test:
            RY(2.0 * p[0] - 0.5 * p[14])(q[2])
            RZ(p[15])(q[3])

        rng = np.random.default_rng(314)
        p0 = rng.uniform(-math.pi, math.pi, size=16)
        ham = [("ZXYI", 0.8), ("IZZX", -0.55), ("YYII", 0.35), ("XXXX", 0.25)]

        adj = adjoint_gradient(deep_var, p0, ham)
        assert adj.num_evaluations == 1
        assert adj.method == "adjoint"

        def scalar_cost(vec: np.ndarray) -> float:
            dag = DAGCircuit.from_builder(deep_var.build(p=vec))
            sim = StateVectorSimulator(4)
            for op in dag.op_nodes():
                sim.apply(op.gate_name, op.qubits, op.params)
            return multi_expectation(sim.state, ham, 4)

        fd = finite_diff(scalar_cost, p0, epsilon=1e-7)
        assert adj.function_value == pytest.approx(fd.function_value, abs=1e-12)
        assert float(np.max(np.abs(adj.gradients - fd.gradients))) < 1e-9

    def test_noisy_estimator_density_matrix_and_high_qubit_trajectories(self) -> None:
        p_depol = 0.15
        noise = NoiseModel().add(Depolarizing(p_depol))

        @circuit(qubits=2)
        def x_prep(q):
            X(q[0])

        # Exact analytical expectation after X(q[0]) + Depolarizing(p) on q[0]:
        # ρ_0 = (1 - 2p/3)|1⟩⟨1| + (2p/3)|0⟩⟨0| => <Z_0> = -(1 - 4p/3)
        expected_z0 = -(1.0 - 4.0 * p_depol / 3.0)
        est1 = Estimator(seed=1, noise=noise).run(x_prep, [[("ZI", 1.0)]])
        est2 = Estimator(seed=999, noise=noise).run(x_prep, [[("ZI", 1.0)]])
        assert est1.value == pytest.approx(expected_z0, abs=1e-13)
        assert est2.value == pytest.approx(expected_z0, abs=1e-13)
        assert est1.metadata[0]["simulation_mode"] == "density_matrix"

        # n = 12 > 11 triggers multi-trajectory Monte Carlo (>= 256 trajectories)
        @circuit(qubits=12)
        def x_prep_12(q):
            X(q[0])

        est_mc = Estimator(seed=42, noise=noise).run(
            x_prep_12, [[("Z" + "I" * 11, 1.0)]], shots=300
        )
        assert "monte_carlo_trajectories(300)" in est_mc.metadata[0]["simulation_mode"]
        # Must NOT collapse to ±1.0; SE <= 1/sqrt(300) ~ 0.058
        assert abs(est_mc.value - expected_z0) < 0.12

    def test_qaoa_egger_warm_start_eigenstate_invariance_and_shot_conservation(self) -> None:
        warm_angles = [0.37, 1.24, 2.15]
        weights = [1.0, 0.65, 0.9]
        qaoa = QAOA(
            cost_hamiltonian=[("ZZI", 1.0), ("IZZ", -0.7), ("ZII", 0.3)],
            p=2,
            mixer_type="biased",
            warm_start_angles=warm_angles,
            mixer_weights=weights,
        )

        # Construct |ψ_warm⟩ = ⊗_i RY(θ_i)|0⟩
        psi_warm = qaoa.simulate_statevector(gamma=[0.0, 0.0], beta=[0.0, 0.0])

        # For ANY beta when gamma == 0, |<ψ_warm | U_M(β) | ψ_warm>|^2 == 1.0 identically!
        for betas in ([0.45, -1.13], [1.9, 0.77], [-2.4, 2.8]):
            psi_sim = qaoa.simulate_statevector(gamma=[0.0, 0.0], beta=betas)
            fid_sim = abs(np.vdot(psi_warm, psi_sim)) ** 2
            assert fid_sim == pytest.approx(1.0, abs=1e-14)

            circ_ws = qaoa.to_circuit(gamma=[0.0, 0.0], beta=betas)
            dag_ws = DAGCircuit.from_builder(circ_ws.build())
            u_ws = _dag_to_unitary(dag_ws)
            psi_circ = u_ws[:, 0]
            fid_circ = abs(np.vdot(psi_warm, psi_circ)) ** 2
            assert fid_circ == pytest.approx(1.0, abs=1e-14)

        # Strict shot conservation: sum(counts.values()) == shots (even for shots=3 on 3 qubits = 8 states!)
        res = qaoa.run(max_iter=15, shots=3, seed=5, num_starts=1)
        assert sum(res.counts.values()) == 3

    def test_lie_algebra_real_inner_product_dimensions_and_barren_plateau_immunity(self) -> None:
        # 1. H and iH are R-orthogonal (Re Tr(H† (iH)) = 0), NEVER conflated over C!
        assert lie_algebra_dimension([X.matrix, 1j * X.matrix]) == 2
        su2_basis = pauli_dla(["X", "Y"])
        assert len(su2_basis) == 3  # dim_R(su(2)) == 3
        assert is_in_algebra(1j * Z.matrix, su2_basis)
        assert not is_in_algebra(Z.matrix, su2_basis)  # Hermitian Z is NOT in skew-Hermitian su(2)!

        # Verify orthonormal Hilbert-Schmidt inner product on skew-Hermitian su(2) basis
        for i in range(3):
            for j in range(3):
                ip = hilbert_schmidt_inner_product(su2_basis[i], su2_basis[j])
                assert abs(ip - (1.0 if i == j else 0.0)) < 1e-12

        # 2. Free-fermion / matchgate 2-qubit generators -> dim_R(so(4)) == 6
        so4_basis = pauli_dla(["XX", "YY", "ZI", "IZ"])
        assert len(so4_basis) == 6

        # 3. Universal 2-qubit generators -> dim_R(su(4)) == 15
        su4_basis = pauli_dla(["XI", "YI", "IX", "IY", "ZZ"])
        assert len(su4_basis) == 15

        # 4. Barren plateau immunity: exponential subalgebra < max_dim // 2 MUST NOT be flagged immune!
        # At n = 5, poly_limit = 50, max_dim = 1023; dla_dim = 300 < 511 is still exponential!
        assert is_barren_plateau_immune(dla_dim=15, n_qubits=3)  # 2*3^2 - 3 = 15 <= 18
        assert not is_barren_plateau_immune(dla_dim=300, n_qubits=5)

    def test_quantum_layer_custom_circuit_fn_amplitude_iqp_gradcheck_and_lru_cache(self) -> None:
        # 1. Custom circuit_fn with non-linear x^2 and coupled w[0]*w[1] defaults to backprop
        def custom_vqc(
            x: torch.Tensor,
            weights: torch.Tensor,
            num_qubits: int,
            _num_layers: int,
            _parsed_obs: list,
        ) -> torch.Tensor:
            B = x.shape[0]
            c_dtype = torch.complex128
            state = torch.zeros((B, 4), dtype=c_dtype, device=x.device)
            state[:, 0] = 1.0
            state = state.reshape(B, 2, 2)

            # Non-RY(pi*x) input encoding: RX(0.7 * x[:, 0]^2) on q0
            ang_x = 0.7 * (x[:, 0] ** 2)
            cx0 = torch.cos(ang_x / 2.0).to(c_dtype)
            sx0 = (-1j * torch.sin(ang_x / 2.0).to(c_dtype))
            g0 = torch.stack(
                [torch.stack([cx0, sx0], dim=-1), torch.stack([sx0, cx0], dim=-1)], dim=-2
            )
            state = torch.bmm(g0, state.reshape(B, 2, 2)).reshape(B, 2, 2)

            # Non-standard coupled weight gate: RY(1.7 * w[0] + w[1]^2) on q1
            ang_w = 1.7 * weights[0] + (weights[1] ** 2)
            cw = torch.cos(ang_w / 2.0).to(c_dtype)
            sw = torch.sin(ang_w / 2.0).to(c_dtype)
            ry1 = torch.stack([torch.stack([cw, -sw]), torch.stack([sw, cw])])
            state = torch.moveaxis(torch.tensordot(ry1, state, dims=([1], [2])), 0, 2)
            return state

        custom_layer = QuantumLayer(
            num_qubits=2,
            circuit_fn=custom_vqc,
            num_params=2,
            observables=["Z0", "X1"],
            dtype=torch.float64,
            device="cpu",
        )
        assert custom_layer.diff_method == "backprop"
        x_cust = torch.tensor([[0.6, -0.4], [1.1, 0.3]], dtype=torch.float64, requires_grad=True)
        assert gradcheck(lambda inp: custom_layer(inp), (x_cust,), eps=1e-6, atol=1e-8)

        # Check joint input + weights gradient via torch.func.functional_call + gradcheck
        w_cust = custom_layer.weights.detach().clone().requires_grad_(True)
        assert gradcheck(
            lambda inp, w: torch.func.functional_call(custom_layer, {"weights": w}, (inp,)),
            (x_cust, w_cust),
            eps=1e-6,
            atol=1e-8,
        )

        loss_c = custom_layer(x_cust).sum()
        loss_c.backward()
        assert custom_layer.weights.grad is not None
        assert torch.isfinite(custom_layer.weights.grad).all()

        # Verify custom_layer weight gradient against central finite difference to < 1e-9
        eps = 1e-6
        with torch.no_grad():
            w_orig = custom_layer.weights.clone()
            for j in range(2):
                custom_layer.weights.copy_(w_orig)
                custom_layer.weights[j] += eps
                fp = float(custom_layer(x_cust.detach()).sum().item())
                custom_layer.weights.copy_(w_orig)
                custom_layer.weights[j] -= eps
                fm = float(custom_layer(x_cust.detach()).sum().item())
                fd_wj = (fp - fm) / (2.0 * eps)
                assert abs(float(custom_layer.weights.grad[j].item()) - fd_wj) < 1e-9
            custom_layer.weights.copy_(w_orig)

        # 2. Amplitude encoding (padded 3 -> 4 and full 4 -> 4) + gradcheck
        amp_layer = QuantumLayer(
            num_qubits=2,
            num_layers=1,
            encoding="amplitude",
            observables=["ZZ", "XI"],
            dtype=torch.float64,
            device="cpu",
        )
        assert amp_layer.in_features == 4
        # With zero weights on real_amplitudes, check exact L2-normalized state expectations
        amp_id = QuantumLayer(
            num_qubits=2,
            circuit_fn="real_amplitudes",
            num_layers=1,
            init_method="zeros",
            encoding="amplitude",
            observables=["Z0", "Z1"],
            dtype=torch.float64,
            device="cpu",
        )
        # Input [3.0, 4.0] padded to [3, 4, 0, 0] / 5 -> P(00)=9/25, P(01)=16/25 -> <Z0>=+1, <Z1>=(9-16)/25=-0.28
        y_amp = amp_id(torch.tensor([[3.0, 4.0]], dtype=torch.float64))
        assert float(y_amp[0, 0].item()) == pytest.approx(1.0, abs=1e-12)
        assert float(y_amp[0, 1].item()) == pytest.approx(-7.0 / 25.0, abs=1e-12)

        x_amp = torch.tensor([[0.8, -0.5, 0.3], [0.4, 0.9, -0.6]], dtype=torch.float64, requires_grad=True)
        assert gradcheck(lambda inp: amp_layer(inp), (x_amp,), eps=1e-6, atol=1e-8)

        # 3. IQP encoding: exact statevector verification vs StateVectorSimulator + gradcheck
        iqp_zero = QuantumLayer(
            num_qubits=3,
            circuit_fn="real_amplitudes",
            num_layers=1,
            init_method="zeros",
            encoding="iqp",
            observables=["X0", "Y1", "X0 X1 Z2"],
            dtype=torch.float64,
            device="cpu",
        )
        x_iqp_val = [0.45, -0.82, 1.15]
        sim_iqp = StateVectorSimulator(3)
        for q in range(3):
            sim_iqp.apply("H", (q,))
        for q in range(3):
            sim_iqp.apply("RZ", (q,), (x_iqp_val[q],))
        for q1 in range(3):
            for q2 in range(q1 + 1, 3):
                sim_iqp.apply("RZZ", (q1, q2), (x_iqp_val[q1] * x_iqp_val[q2],))
        # real_amplitudes(num_layers=1, init_method='zeros') applies linear CX entanglers after RY(0)
        sim_iqp.apply("CX", (0, 1))
        sim_iqp.apply("CX", (1, 2))

        y_iqp = iqp_zero(torch.tensor([x_iqp_val], dtype=torch.float64))
        assert float(y_iqp[0, 0].item()) == pytest.approx(
            expectation(sim_iqp.state, "XII", 3), abs=1e-12
        )
        assert float(y_iqp[0, 1].item()) == pytest.approx(
            expectation(sim_iqp.state, "IYI", 3), abs=1e-12
        )
        assert float(y_iqp[0, 2].item()) == pytest.approx(
            expectation(sim_iqp.state, "XXZ", 3), abs=1e-12
        )

        iqp_trainable = QuantumLayer(
            num_qubits=2,
            circuit_fn="strongly_entangling",
            num_layers=1,
            encoding="iqp",
            observables=["Z0", "X1"],
            dtype=torch.float64,
            device="cpu",
        )
        x_iqp = torch.tensor([[0.5, -0.9], [-0.4, 0.7]], dtype=torch.float64, requires_grad=True)
        assert gradcheck(lambda inp: iqp_trainable(inp), (x_iqp,), eps=1e-6, atol=1e-8)

        # 4. Unknown encoding MUST raise ValueError (never silently skip x!)
        with pytest.raises(ValueError, match="Unsupported encoding"):
            QuantumLayer(num_qubits=2, encoding="basis_bogus")

        # 5. Bounded LRU Pauli cache <= 256 entries
        ops._PAULI_KRON_CACHE.clear()
        pauli_chars = "IXYZ"
        for idx in range(320):
            # Generate 320 distinct 5-qubit Pauli strings (4^5 = 1024 available)
            s = "".join(pauli_chars[(idx >> (2 * k)) & 3] for k in range(5))
            ops.pauli_kron(s, num_qubits=5, device="cpu", dtype=torch.complex64)
        assert len(ops._PAULI_KRON_CACHE) <= 256
