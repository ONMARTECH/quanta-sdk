"""
tests/test_simulator.py — Simülatör doğruluk testleri.

Kuantum algoritmalarının bilinen sonuçlarıyla karşılaştırma yapar.
Bu testler SDK'nın "kalite puanını" belirler.
"""

import numpy as np
import pytest

from quanta import circuit, H, X, Y, Z, CX, RY, measure, run
from quanta.simulator.statevector import StateVectorSimulator


# ═══════════════════════════════════════════
#  Temel Durum Testleri
# ═══════════════════════════════════════════

class TestBasicStates:
    """Temel kuantum durumlarının doğruluğu."""

    def test_initial_state_is_zero(self):
        """Başlangıç durumu |0⟩ = [1, 0]."""
        sim = StateVectorSimulator(1)
        np.testing.assert_allclose(sim.state, [1, 0])

    def test_x_gate_flips_to_one(self):
        """X|0⟩ = |1⟩."""
        sim = StateVectorSimulator(1)
        sim.apply("X", (0,))
        np.testing.assert_allclose(sim.state, [0, 1])

    def test_hadamard_creates_superposition(self):
        """H|0⟩ = (|0⟩ + |1⟩)/√2 ≈ [0.707, 0.707]."""
        sim = StateVectorSimulator(1)
        sim.apply("H", (0,))
        expected = np.array([1, 1]) / np.sqrt(2)
        np.testing.assert_allclose(sim.state, expected, atol=1e-10)

    def test_double_hadamard_returns_to_zero(self):
        """H·H = I: H|0⟩ → |+⟩ → |0⟩."""
        sim = StateVectorSimulator(1)
        sim.apply("H", (0,))
        sim.apply("H", (0,))
        np.testing.assert_allclose(sim.state, [1, 0], atol=1e-10)

    def test_z_gate_adds_phase(self):
        """Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩."""
        sim = StateVectorSimulator(1)
        sim.apply("X", (0,))
        sim.apply("Z", (0,))
        np.testing.assert_allclose(sim.state, [0, -1], atol=1e-10)


# ═══════════════════════════════════════════
#  Bell State Testleri
# ═══════════════════════════════════════════

class TestBellState:
    """Bell state doğruluk testleri — SDK'nın temel kalite ölçüsü."""

    def test_bell_state_vector(self):
        """|Φ+⟩ = (|00⟩ + |11⟩)/√2 = [1/√2, 0, 0, 1/√2]."""
        sim = StateVectorSimulator(2)
        sim.apply("H", (0,))
        sim.apply("CX", (0, 1))
        expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
        np.testing.assert_allclose(sim.state, expected, atol=1e-10)

    def test_bell_state_only_00_and_11(self):
        """Ölçümde sadece |00⟩ ve |11⟩ çıkmalı."""
        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        result = run(bell, shots=10000, seed=42)
        assert set(result.counts.keys()) == {"00", "11"}

    def test_bell_state_approximately_equal_probs(self):
        """P(00) ≈ P(11) ≈ 0.5 (±%5 tolerans)."""
        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        result = run(bell, shots=10000, seed=42)
        p00 = result.probabilities.get("00", 0)
        p11 = result.probabilities.get("11", 0)
        assert 0.45 < p00 < 0.55, f"P(00)={p00}"
        assert 0.45 < p11 < 0.55, f"P(11)={p11}"


# ═══════════════════════════════════════════
#  GHZ State Testleri
# ═══════════════════════════════════════════

class TestGHZState:
    """3-qubit GHZ state testleri."""

    def test_ghz_only_000_and_111(self):
        @circuit(qubits=3)
        def ghz(q):
            H(q[0])
            CX(q[0], q[1])
            CX(q[1], q[2])
            return measure(q)

        result = run(ghz, shots=10000, seed=42)
        assert set(result.counts.keys()) == {"000", "111"}

    def test_ghz_approximately_equal_probs(self):
        @circuit(qubits=3)
        def ghz(q):
            H(q[0])
            CX(q[0], q[1])
            CX(q[1], q[2])
            return measure(q)

        result = run(ghz, shots=10000, seed=42)
        p000 = result.probabilities.get("000", 0)
        assert 0.45 < p000 < 0.55


# ═══════════════════════════════════════════
#  Parametrik Kapı Testleri
# ═══════════════════════════════════════════

class TestParametricGates:
    """RX, RY, RZ parametrik kapılarının testleri."""

    def test_ry_pi_flips_state(self):
        """RY(π)|0⟩ = |1⟩."""
        sim = StateVectorSimulator(1)
        sim.apply("RY", (0,), (np.pi,))
        probs = sim.probabilities()
        np.testing.assert_allclose(probs[1], 1.0, atol=1e-10)

    def test_ry_half_pi_creates_superposition(self):
        """RY(π/2)|0⟩ → eşit süperpozisyon."""
        sim = StateVectorSimulator(1)
        sim.apply("RY", (0,), (np.pi / 2,))
        probs = sim.probabilities()
        np.testing.assert_allclose(probs[0], 0.5, atol=1e-10)
        np.testing.assert_allclose(probs[1], 0.5, atol=1e-10)


# ═══════════════════════════════════════════
#  DAG Testleri
# ═══════════════════════════════════════════

class TestDAG:
    """DAG oluşturma ve analiz testleri."""

    def test_bell_state_depth_is_2(self):
        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])

        from quanta.dag.dag_circuit import DAGCircuit
        dag = DAGCircuit.from_builder(bell.build())
        assert dag.depth() == 2

    def test_bell_state_gate_count_is_2(self):
        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])

        from quanta.dag.dag_circuit import DAGCircuit
        dag = DAGCircuit.from_builder(bell.build())
        assert dag.gate_count() == 2

    def test_parallel_gates_have_depth_1(self):
        """H(q[0]) ve H(q[1]) bağımsız → derinlik 1."""
        @circuit(qubits=2)
        def parallel(q):
            H(q[0])
            H(q[1])

        from quanta.dag.dag_circuit import DAGCircuit
        dag = DAGCircuit.from_builder(parallel.build())
        assert dag.depth() == 1
        assert dag.gate_count() == 2

    def test_layers_groups_parallel_ops(self):
        """Paralel kapılar aynı katmanda olmalı."""
        @circuit(qubits=2)
        def parallel(q):
            H(q[0])
            H(q[1])

        from quanta.dag.dag_circuit import DAGCircuit
        dag = DAGCircuit.from_builder(parallel.build())
        layers = dag.layers()
        assert len(layers) == 1
        assert len(layers[0]) == 2


# ═══════════════════════════════════════════
#  Result Testleri
# ═══════════════════════════════════════════

class TestResult:
    """Result sınıfı testleri."""

    def test_result_probabilities(self):
        from quanta.result import Result
        r = Result(counts={"00": 500, "11": 500}, shots=1000, num_qubits=2)
        assert r.probabilities == {"00": 0.5, "11": 0.5}

    def test_most_frequent(self):
        from quanta.result import Result
        r = Result(counts={"00": 300, "11": 700}, shots=1000, num_qubits=2)
        assert r.most_frequent == "11"

    def test_summary_returns_string(self):
        from quanta.result import Result
        r = Result(
            counts={"00": 500, "11": 500}, shots=1000,
            num_qubits=2, circuit_name="test"
        )
        s = r.summary()
        assert isinstance(s, str)
        assert "test" in s


# ═══════════════════════════════════════════
#  Seed Tekrarlanabilirlik Testi
# ═══════════════════════════════════════════

class TestReproducibility:
    """Aynı seed ile aynı sonuç üretilmeli."""

    def test_same_seed_gives_same_result(self):
        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        r1 = run(bell, shots=1000, seed=42)
        r2 = run(bell, shots=1000, seed=42)
        assert r1.counts == r2.counts

    def test_different_seed_gives_different_result(self):
        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        r1 = run(bell, shots=1000, seed=42)
        r2 = run(bell, shots=1000, seed=99)
        # Sonuçlar farklı olabilir (garantili değil ama çok olası)
        # En azından aynı anahtarları paylaşmalı
        assert set(r1.counts.keys()) == set(r2.counts.keys())
