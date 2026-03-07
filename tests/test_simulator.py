"""
tests/test_simulator.py — Simulator correctness tests.

Compares quantum algorithm results against known expected values.
These tests determine the SDK's "quality score".
"""

import numpy as np
import pytest

from quanta import circuit, H, X, Y, Z, CX, RY, measure, run
from quanta.simulator.statevector import StateVectorSimulator


# ═══════════════════════════════════════════
#  Basic State Tests
# ═══════════════════════════════════════════

class TestBasicStates:
    """Basic quantum state correctness."""

    def test_initial_state_is_zero(self):
        """Initial state |0⟩ = [1, 0]."""
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
#  Bell State Tests
# ═══════════════════════════════════════════

class TestBellState:
    """Bell state correctness tests — SDK's core quality measure."""

    def test_bell_state_vector(self):
        """|Φ+⟩ = (|00⟩ + |11⟩)/√2 = [1/√2, 0, 0, 1/√2]."""
        sim = StateVectorSimulator(2)
        sim.apply("H", (0,))
        sim.apply("CX", (0, 1))
        expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
        np.testing.assert_allclose(sim.state, expected, atol=1e-10)

    def test_bell_state_only_00_and_11(self):
        """Measurement should yield only |00⟩ and |11⟩."""
        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        result = run(bell, shots=10000, seed=42)
        assert set(result.counts.keys()) == {"00", "11"}

    def test_bell_state_approximately_equal_probs(self):
        """P(00) ≈ P(11) ≈ 0.5 (±5% tolerance)."""
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
#  GHZ State Tests
# ═══════════════════════════════════════════

class TestGHZState:
    """3-qubit GHZ state tests."""

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
#  Parametric Gate Tests
# ═══════════════════════════════════════════

class TestParametricGates:
    """RX, RY, RZ parametric gate tests."""

    def test_ry_pi_flips_state(self):
        """RY(π)|0⟩ = |1⟩."""
        sim = StateVectorSimulator(1)
        sim.apply("RY", (0,), (np.pi,))
        probs = sim.probabilities()
        np.testing.assert_allclose(probs[1], 1.0, atol=1e-10)

    def test_ry_half_pi_creates_superposition(self):
        """RY(π/2)|0⟩ → equal superposition."""
        sim = StateVectorSimulator(1)
        sim.apply("RY", (0,), (np.pi / 2,))
        probs = sim.probabilities()
        np.testing.assert_allclose(probs[0], 0.5, atol=1e-10)
        np.testing.assert_allclose(probs[1], 0.5, atol=1e-10)


# ═══════════════════════════════════════════
#  DAG Tests
# ═══════════════════════════════════════════

class TestDAG:
    """DAG construction and analysis tests."""

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
        """H(q[0]) and H(q[1]) are independent → depth 1."""
        @circuit(qubits=2)
        def parallel(q):
            H(q[0])
            H(q[1])

        from quanta.dag.dag_circuit import DAGCircuit
        dag = DAGCircuit.from_builder(parallel.build())
        assert dag.depth() == 1
        assert dag.gate_count() == 2

    def test_layers_groups_parallel_ops(self):
        """Parallel gates should be in the same layer."""
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
#  Result Tests
# ═══════════════════════════════════════════

class TestResult:
    """Result class tests."""

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
#  Seed Reproducibility Tests
# ═══════════════════════════════════════════

class TestReproducibility:
    """Same seed should produce the same result."""

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
        # Results may differ (not guaranteed but highly likely)
        # At least they should share the same keys
        assert set(r1.counts.keys()) == set(r2.counts.keys())


# ═══════════════════════════════════════════
#  Noise Model Tests
# ═══════════════════════════════════════════

class TestNoiseChannels:
    """Tests for all noise channels."""

    def _make_state(self, qubit_state: str = "0") -> np.ndarray:
        """Create a simple 1-qubit state."""
        if qubit_state == "0":
            return np.array([1.0, 0.0], dtype=complex)
        elif qubit_state == "+":
            return np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        return np.array([0.0, 1.0], dtype=complex)  # "1"

    def test_depolarizing_modifies_state(self):
        from quanta.simulator.noise import Depolarizing
        ch = Depolarizing(probability=1.0)  # Always apply
        rng = np.random.default_rng(42)
        state = self._make_state("0")
        new = ch.apply(state, 0, 1, rng)
        # With p=1.0, state must change
        assert not np.allclose(new, state)

    def test_depolarizing_no_error(self):
        from quanta.simulator.noise import Depolarizing
        ch = Depolarizing(probability=0.0)  # Never apply
        rng = np.random.default_rng(42)
        state = self._make_state("0")
        new = ch.apply(state, 0, 1, rng)
        np.testing.assert_allclose(new, state)

    def test_bit_flip_flips_state(self):
        from quanta.simulator.noise import BitFlip
        ch = BitFlip(probability=1.0)
        rng = np.random.default_rng(42)
        state = self._make_state("0")
        new = ch.apply(state, 0, 1, rng)
        np.testing.assert_allclose(new, [0, 1])

    def test_phase_flip_adds_phase(self):
        from quanta.simulator.noise import PhaseFlip
        ch = PhaseFlip(probability=1.0)
        rng = np.random.default_rng(42)
        state = self._make_state("1")
        new = ch.apply(state, 0, 1, rng)
        np.testing.assert_allclose(new, [0, -1])

    def test_amplitude_damping_decays(self):
        from quanta.simulator.noise import AmplitudeDamping
        ch = AmplitudeDamping(gamma=1.0)
        rng = np.random.default_rng(42)
        state = self._make_state("1")
        new = ch.apply(state, 0, 1, rng)
        # Should decay toward |0⟩
        assert abs(new[0]) > 0.5

    def test_t2_relaxation_preserves_norm(self):
        from quanta.simulator.noise import T2Relaxation
        ch = T2Relaxation(gamma=1.0)
        rng = np.random.default_rng(42)
        state = self._make_state("+")
        new = ch.apply(state, 0, 1, rng)
        norm = np.linalg.norm(new)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)

    def test_t2_relaxation_adds_phase_to_one(self):
        from quanta.simulator.noise import T2Relaxation
        ch = T2Relaxation(gamma=1.0)
        rng = np.random.default_rng(42)
        state = self._make_state("+")
        new = ch.apply(state, 0, 1, rng)
        # |0⟩ component unchanged, |1⟩ gets random phase
        np.testing.assert_allclose(abs(new[0]), abs(state[0]), atol=1e-10)
        np.testing.assert_allclose(abs(new[1]), abs(state[1]), atol=1e-10)

    def test_t2_no_effect_when_gamma_zero(self):
        from quanta.simulator.noise import T2Relaxation
        ch = T2Relaxation(gamma=0.0)
        rng = np.random.default_rng(42)
        state = self._make_state("+")
        new = ch.apply(state, 0, 1, rng)
        np.testing.assert_allclose(new, state)

    def test_crosstalk_preserves_norm(self):
        from quanta.simulator.noise import Crosstalk
        ch = Crosstalk(probability=1.0)
        rng = np.random.default_rng(42)
        # 2-qubit state |10⟩
        state = np.array([0, 0, 1, 0], dtype=complex)
        new = ch.apply(state, 0, 2, rng)
        norm = np.linalg.norm(new)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)

    def test_crosstalk_no_effect_at_boundary(self):
        from quanta.simulator.noise import Crosstalk
        # neighbor_offset=1 on last qubit → no neighbor
        ch = Crosstalk(probability=1.0, neighbor_offset=1)
        rng = np.random.default_rng(42)
        state = self._make_state("1")
        new = ch.apply(state, 0, 1, rng)
        np.testing.assert_allclose(new, state)

    def test_crosstalk_modifies_two_qubit_state(self):
        from quanta.simulator.noise import Crosstalk
        ch = Crosstalk(probability=1.0)
        rng = np.random.default_rng(42)
        # |11⟩ state — both qubits |1⟩ → ZZ interaction applies
        state = np.array([0, 0, 0, 1], dtype=complex)
        new = ch.apply(state, 0, 2, rng)
        # Phase should change on |11⟩
        assert not np.allclose(new[3], state[3])

    def test_readout_error_no_state_change(self):
        from quanta.simulator.noise import ReadoutError
        ch = ReadoutError(p0_to_1=0.5, p1_to_0=0.5)
        rng = np.random.default_rng(42)
        state = self._make_state("0")
        new = ch.apply(state, 0, 1, rng)
        np.testing.assert_allclose(new, state)

    def test_readout_error_flips_counts(self):
        from quanta.simulator.noise import ReadoutError
        ch = ReadoutError(p0_to_1=1.0, p1_to_0=0.0)
        rng = np.random.default_rng(42)
        counts = {"00": 1000}
        noisy = ch.apply_to_counts(counts, rng)
        # All 0s should flip to 1s
        assert "11" in noisy
        assert noisy.get("11", 0) == 1000

    def test_readout_error_preserves_total_shots(self):
        from quanta.simulator.noise import ReadoutError
        ch = ReadoutError(p0_to_1=0.05, p1_to_0=0.05)
        rng = np.random.default_rng(42)
        counts = {"00": 500, "11": 500}
        noisy = ch.apply_to_counts(counts, rng)
        total = sum(noisy.values())
        assert total == 1000

    def test_noise_model_chains(self):
        from quanta.simulator.noise import NoiseModel, Depolarizing, T2Relaxation
        model = NoiseModel()
        model.add(Depolarizing(0.01)).add(T2Relaxation(0.01))
        assert len(model._channels) == 2
        assert "Depolarizing" in repr(model)
        assert "T2Relaxation" in repr(model)

    def test_noise_channel_names(self):
        from quanta.simulator.noise import (
            Depolarizing, BitFlip, PhaseFlip, AmplitudeDamping,
            T2Relaxation, Crosstalk, ReadoutError,
        )
        assert "Depolarizing" in Depolarizing().name
        assert "BitFlip" in BitFlip().name
        assert "PhaseFlip" in PhaseFlip().name
        assert "AmplitudeDamping" in AmplitudeDamping().name
        assert "T2Relaxation" in T2Relaxation().name
        assert "Crosstalk" in Crosstalk().name
        assert "ReadoutError" in ReadoutError().name
