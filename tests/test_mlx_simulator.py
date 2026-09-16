"""
tests/test_mlx_simulator.py -- Unit tests for Apple Silicon Metal/MLX Simulator.

Tests correctness, gate application, entanglement, sampling, and factory integration
for MLXSimulator on Apple Silicon.
"""

import numpy as np
import pytest

from quanta.simulator.factory import create_simulator
from quanta.simulator.mlx import MLXSimulator, is_mlx_available

pytestmark = pytest.mark.skipif(
    not is_mlx_available(),
    reason="Apple MLX is only available on Darwin arm64 with mlx installed",
)


class TestMLXBasics:
    """Basic initialization and statevector properties."""

    def test_initial_state_zero(self):
        sim = MLXSimulator(num_qubits=2)
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1.0
        np.testing.assert_allclose(sim.state, expected, atol=1e-6)

    def test_repr(self):
        sim = MLXSimulator(num_qubits=3)
        assert "MLXSimulator(qubits=3, device=metal_gpu)" in repr(sim)

    def test_reset(self):
        sim = MLXSimulator(num_qubits=2)
        sim.apply("X", (0,))
        assert sim.probabilities()[2] > 0.99
        sim.reset()
        assert sim.probabilities()[0] > 0.99


class TestMLXSingleQubitGates:
    """Test standard single-qubit gate operations."""

    def test_x_gate(self):
        sim = MLXSimulator(num_qubits=1)
        sim.apply("X", (0,))
        np.testing.assert_allclose(sim.state, [0, 1], atol=1e-6)

    def test_hadamard_gate(self):
        sim = MLXSimulator(num_qubits=1)
        sim.apply("H", (0,))
        expected = np.array([1, 1]) / np.sqrt(2)
        np.testing.assert_allclose(sim.state, expected, atol=1e-6)

    def test_double_hadamard_identity(self):
        sim = MLXSimulator(num_qubits=1)
        sim.apply("H", (0,))
        sim.apply("H", (0,))
        np.testing.assert_allclose(sim.state, [1, 0], atol=1e-6)

    def test_y_and_z_gates(self):
        sim = MLXSimulator(num_qubits=1)
        sim.apply("X", (0,))
        sim.apply("Z", (0,))
        np.testing.assert_allclose(sim.state, [0, -1], atol=1e-6)

        sim.reset()
        sim.apply("Y", (0,))
        np.testing.assert_allclose(sim.state, [0, 1j], atol=1e-6)

    def test_parametric_rotation_gates(self):
        # RX(pi) |0> = -i |1>
        sim = MLXSimulator(num_qubits=1)
        sim.apply("RX", (0,), params=(np.pi,))
        np.testing.assert_allclose(np.abs(sim.state) ** 2, [0, 1], atol=1e-5)

        # RZ rotation on superposition
        sim = MLXSimulator(num_qubits=1)
        sim.apply("H", (0,))
        sim.apply("RZ", (0,), params=(np.pi / 2,))
        probs = sim.probabilities()
        np.testing.assert_allclose(probs, [0.5, 0.5], atol=1e-5)


class TestMLXMultiQubitGates:
    """Test two-qubit and multi-qubit gates (CX, CZ, SWAP, CCX)."""

    def test_bell_state_phi_plus(self):
        sim = MLXSimulator(num_qubits=2)
        sim.apply("H", (0,))
        sim.apply("CX", (0, 1))
        expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
        np.testing.assert_allclose(sim.state, expected, atol=1e-6)

    def test_cz_gate(self):
        sim = MLXSimulator(num_qubits=2)
        sim.apply("X", (0,))
        sim.apply("X", (1,))
        sim.apply("CZ", (0, 1))
        expected = np.array([0, 0, 0, -1])
        np.testing.assert_allclose(sim.state, expected, atol=1e-6)

    def test_swap_gate(self):
        sim = MLXSimulator(num_qubits=2)
        sim.apply("X", (0,))  # |10> (qubit 0 is MSB)
        sim.apply("SWAP", (0, 1))
        # Now qubit 1 is 1 -> |01>
        probs = sim.probabilities()
        assert probs[1] > 0.99

    def test_toffoli_ccx(self):
        sim = MLXSimulator(num_qubits=3)
        sim.apply("X", (0,))
        sim.apply("X", (1,))
        sim.apply("CCX", (0, 1, 2))
        expected = np.zeros(8)
        expected[7] = 1.0  # |111>
        np.testing.assert_allclose(sim.probabilities(), expected, atol=1e-5)


class TestMLXEntanglementAndSampling:
    """Test GHZ state, probability sums, and random sampling."""

    def test_ghz_3_qubit(self):
        sim = MLXSimulator(num_qubits=3)
        sim.apply("H", (0,))
        sim.apply("CX", (0, 1))
        sim.apply("CX", (1, 2))
        probs = sim.probabilities()
        assert np.isclose(probs[0], 0.5, atol=1e-5)
        assert np.isclose(probs[7], 0.5, atol=1e-5)
        assert np.sum(probs) == pytest.approx(1.0, rel=1e-5)

    def test_sampling_bell_state(self):
        sim = MLXSimulator(num_qubits=2, seed=42)
        sim.apply("H", (0,))
        sim.apply("CX", (0, 1))
        counts = sim.sample(shots=1000)
        assert set(counts.keys()) == {"00", "11"}
        assert 400 < counts["00"] < 600
        assert 400 < counts["11"] < 600

    def test_apply_phase(self):
        sim = MLXSimulator(num_qubits=2)
        sim.apply("H", (0,))
        sim.apply("H", (1,))
        sim.apply_phase(3, -1.0)
        st = sim.state
        assert np.isclose(st[3].real, -0.5, atol=1e-5)
        assert np.isclose(st[0].real, 0.5, atol=1e-5)


class TestMLXFactoryIntegration:
    """Test factory creation with 'mlx' and 'metal' method tags."""

    def test_create_via_mlx(self):
        sim = create_simulator(num_qubits=3, method="mlx", seed=123)
        assert isinstance(sim, MLXSimulator)
        assert sim.num_qubits == 3

    def test_create_via_metal(self):
        sim = create_simulator(num_qubits=4, method="metal", seed=123)
        assert isinstance(sim, MLXSimulator)
        assert sim.num_qubits == 4


class TestMLXEdgeCasesAndErrors:
    """Test error branches, edge cases, and noise application in MLXSimulator."""

    def test_max_qubits_property_and_limit(self):
        sim = MLXSimulator(num_qubits=2)
        assert sim.max_qubits >= 30

        with pytest.raises(Exception, match="supports up to"):
            MLXSimulator(num_qubits=100)

    def test_state_setter_dimension_mismatch(self):
        sim = MLXSimulator(num_qubits=2)
        with pytest.raises(Exception, match="does not match"):
            sim.state = [1.0, 0.0]  # Expected 4, got 2

    def test_unknown_gate_error(self):
        sim = MLXSimulator(num_qubits=2)
        with pytest.raises(Exception, match="Unknown gate"):
            sim.apply("FOO_INVALID_GATE", (0,))

    def test_parametric_gate_missing_params(self):
        sim = MLXSimulator(num_qubits=2)
        with pytest.raises(Exception, match="requires parameters"):
            sim.apply("RX", (0,), params=())

        with pytest.raises(Exception, match="requires parameters"):
            sim.apply("U", (0,), params=())

    def test_apply_noise(self):
        from quanta.simulator.noise import BitFlip, NoiseModel
        sim = MLXSimulator(num_qubits=2, seed=42)
        noise = NoiseModel().add(BitFlip(1.0))
        rng = np.random.default_rng(42)
        sim.apply_noise(noise, (0,), rng)
        # Bit flip on qubit 0 transforms |00> into |10> (index 2)
        assert sim.probabilities()[2] > 0.99

    def test_is_mlx_available_fallbacks(self, monkeypatch):
        import platform

        from quanta.simulator.mlx import is_mlx_available

        monkeypatch.setattr(platform, "system", lambda: "Linux")
        assert is_mlx_available() is False

        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        assert is_mlx_available() is False
