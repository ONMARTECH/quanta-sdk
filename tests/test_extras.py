"""
tests/test_extras.py — Ek modüllerin testleri.

QASM export, gürültü, eşdeğerlik, QEC, transpiler testleri.
"""

import numpy as np
import pytest

from quanta import circuit, H, X, Z, S, T, CX, CZ, SWAP, RZ, measure, run


# ═══════════════════════════════════════════
#  OpenQASM Export Testleri
# ═══════════════════════════════════════════

class TestQASMExport:
    """OpenQASM 3.0 çıktı testleri."""

    def test_bell_state_qasm_has_header(self):
        from quanta.export.qasm import to_qasm

        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        qasm = to_qasm(bell)
        assert "OPENQASM 3.0;" in qasm
        assert 'include "stdgates.inc";' in qasm

    def test_bell_state_qasm_has_gates(self):
        from quanta.export.qasm import to_qasm

        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        qasm = to_qasm(bell)
        assert "h q[0];" in qasm
        assert "cx q[0], q[1];" in qasm

    def test_parametric_gate_qasm(self):
        from quanta.export.qasm import to_qasm

        @circuit(qubits=1)
        def rz_circ(q):
            RZ(np.pi / 4)(q[0])
            return measure(q)

        qasm = to_qasm(rz_circ)
        assert "rz(" in qasm

    def test_qasm_roundtrip_gate_names(self):
        """QASM export → parse → kapı isimleri korunmalı."""
        from quanta.export.qasm import to_qasm, from_qasm_gates

        @circuit(qubits=2)
        def test_circ(q):
            H(q[0])
            CX(q[0], q[1])

        qasm = to_qasm(test_circ)
        gates = from_qasm_gates(qasm)
        names = [g[0] for g in gates]
        assert "H" in names
        assert "CX" in names


# ═══════════════════════════════════════════
#  Gürültü Modeli Testleri
# ═══════════════════════════════════════════

class TestNoise:
    """Gürültü kanalları testleri."""

    def test_no_noise_preserves_state(self):
        """Sıfır olasılıklı gürültü durumu değiştirmemeli."""
        from quanta.simulator.noise import Depolarizing
        rng = np.random.default_rng(42)
        state = np.array([1, 0], dtype=complex)
        result = Depolarizing(0.0).apply(state, 0, 1, rng)
        np.testing.assert_allclose(result, state)

    def test_full_noise_changes_state(self):
        """P=1 gürültü durumu değiştirmeli."""
        from quanta.simulator.noise import BitFlip
        rng = np.random.default_rng(42)
        state = np.array([1, 0], dtype=complex)
        result = BitFlip(1.0).apply(state, 0, 1, rng)
        # X uygulanmış olmalı: [1,0] → [0,1]
        np.testing.assert_allclose(result, [0, 1])

    def test_noise_model_chains_channels(self):
        from quanta.simulator.noise import NoiseModel, Depolarizing
        model = NoiseModel()
        model.add(Depolarizing(0.0))
        model.add(Depolarizing(0.0))
        assert len(model._channels) == 2

    def test_amplitude_damping_decays_excited(self):
        """Amplitude damping |1⟩ → |0⟩ yönünde çözmeli."""
        from quanta.simulator.noise import AmplitudeDamping
        rng = np.random.default_rng(42)
        state = np.array([0, 1], dtype=complex)
        result = AmplitudeDamping(1.0).apply(state, 0, 1, rng)
        # Tam bozunma: |1⟩ → |0⟩
        assert abs(result[0]) > abs(result[1])


# ═══════════════════════════════════════════
#  Devre Eşdeğerlik Testleri
# ═══════════════════════════════════════════

class TestEquivalence:
    """Devre eşdeğerlik ve fidelity testleri."""

    def test_same_circuit_is_equivalent(self):
        from quanta.core.equivalence import circuits_equivalent

        @circuit(qubits=1)
        def c1(q):
            H(q[0])

        @circuit(qubits=1)
        def c2(q):
            H(q[0])

        assert circuits_equivalent(c1, c2)

    def test_different_circuits_not_equivalent(self):
        from quanta.core.equivalence import circuits_equivalent

        @circuit(qubits=1)
        def c1(q):
            H(q[0])

        @circuit(qubits=1)
        def c2(q):
            X(q[0])

        assert not circuits_equivalent(c1, c2)

    def test_hh_equivalent_to_identity(self):
        """H·H = I."""
        from quanta.core.equivalence import get_unitary

        @circuit(qubits=1)
        def hh(q):
            H(q[0])
            H(q[0])

        u = get_unitary(hh)
        np.testing.assert_allclose(u, np.eye(2), atol=1e-8)

    def test_fidelity_same_circuit_is_one(self):
        from quanta.core.equivalence import fidelity

        @circuit(qubits=1)
        def c(q):
            H(q[0])

        assert fidelity(c, c) > 0.99

    def test_fidelity_orthogonal_is_low(self):
        from quanta.core.equivalence import fidelity

        @circuit(qubits=1)
        def c1(q):
            H(q[0])

        @circuit(qubits=1)
        def c2(q):
            X(q[0])
            H(q[0])

        f = fidelity(c1, c2)
        assert f < 0.99


# ═══════════════════════════════════════════
#  QEC Testleri
# ═══════════════════════════════════════════

class TestQEC:
    """Kuantum hata düzeltme kodu testleri."""

    def test_bitflip_code_info(self):
        from quanta.qec.codes import BitFlipCode
        code = BitFlipCode()
        assert code.info.n == 3
        assert code.info.k == 1

    def test_bitflip_encode_creates_ghz(self):
        """BitFlip encode: |0⟩ → |000⟩, |1⟩ → |111⟩."""
        from quanta.qec.codes import BitFlipCode
        code = BitFlipCode()
        encode = code.encode()
        result = run(encode, shots=100, seed=42)
        # |000⟩ = initial state encoded
        assert "000" in result.counts

    def test_steane_code_info(self):
        from quanta.qec.codes import SteaneCode
        code = SteaneCode()
        assert code.info.n == 7
        assert code.info.d == 3
        assert code.info.correctable_errors == 1

    def test_phaseflip_encode_runs(self):
        from quanta.qec.codes import PhaseFlipCode
        code = PhaseFlipCode()
        encode = code.encode()
        result = run(encode, shots=100, seed=42)
        assert result.shots == 100


# ═══════════════════════════════════════════
#  Transpiler Testleri
# ═══════════════════════════════════════════

class TestTranspiler:
    """Gate set transpilasyonu testleri."""

    def test_native_gates_unchanged(self):
        """Hedef sette olan kapılar değişmemeli."""
        from quanta.compiler.passes.translate import TranslateToTarget
        from quanta.dag.dag_circuit import DAGCircuit

        @circuit(qubits=2)
        def native(q):
            CX(q[0], q[1])
            RZ(np.pi)(q[0])

        dag = DAGCircuit.from_builder(native.build())
        transpiled = TranslateToTarget("ibm").run(dag)
        assert transpiled.gate_count() == dag.gate_count()

    def test_h_decomposes_to_rz_ry(self):
        """H → RZ(π)·RY(π/2) (IBM gate set'inde)."""
        from quanta.compiler.passes.translate import TranslateToTarget
        from quanta.dag.dag_circuit import DAGCircuit

        @circuit(qubits=1)
        def h_only(q):
            H(q[0])

        dag = DAGCircuit.from_builder(h_only.build())
        transpiled = TranslateToTarget("ibm").run(dag)
        gate_names = [op.gate_name for op in transpiled.op_nodes()]
        assert "H" not in gate_names
        assert "RZ" in gate_names or "RY" in gate_names

    def test_swap_decomposes_to_3_cx(self):
        """SWAP → 3 CX."""
        from quanta.compiler.passes.translate import TranslateToTarget
        from quanta.dag.dag_circuit import DAGCircuit

        @circuit(qubits=2)
        def swap_circ(q):
            SWAP(q[0], q[1])

        dag = DAGCircuit.from_builder(swap_circ.build())
        transpiled = TranslateToTarget("ibm").run(dag)
        cx_count = sum(1 for op in transpiled.op_nodes() if op.gate_name == "CX")
        assert cx_count == 3

    def test_invalid_gate_set_raises(self):
        from quanta.compiler.passes.translate import TranslateToTarget
        with pytest.raises(ValueError):
            TranslateToTarget("nonexistent")


# ═══════════════════════════════════════════
#  Durum Görselleştirme Testleri
# ═══════════════════════════════════════════

class TestVisualizeState:
    """Durum vektörü görselleştirme testleri."""

    def test_show_probabilities_returns_string(self):
        from quanta.visualize_state import show_probabilities
        from quanta.result import Result
        r = Result(counts={"00": 500, "11": 500}, shots=1000, num_qubits=2)
        s = show_probabilities(r)
        assert isinstance(s, str)
        assert "|00⟩" in s

    def test_show_statevector_shows_nonzero(self):
        from quanta.visualize_state import show_statevector
        sv = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=complex)
        s = show_statevector(sv, num_qubits=2)
        assert "|00⟩" in s
        assert "|11⟩" in s
        assert "|01⟩" not in s  # Sıfır amplitüd gösterilmemeli

    def test_show_phases_returns_arrows(self):
        from quanta.visualize_state import show_phases
        sv = np.array([1, 0], dtype=complex)
        s = show_phases(sv, num_qubits=1)
        assert "→" in s or "↑" in s  # Faz oku olmalı
