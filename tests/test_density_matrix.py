"""Tests for quanta.simulator.density_matrix."""

import numpy as np
import pytest

from quanta.simulator.density_matrix import DensityMatrixError, DensityMatrixSimulator


def test_init_and_state():
    sim = DensityMatrixSimulator(num_qubits=2)
    assert sim.num_qubits == 2
    assert sim.state.shape == (4, 4)
    # Initially in |00><00|
    expected = np.zeros((4, 4), dtype=complex)
    expected[0, 0] = 1.0
    np.testing.assert_allclose(sim.state, expected)
    assert sim.purity == pytest.approx(1.0)


def test_max_qubits_exceeded():
    with pytest.raises(DensityMatrixError, match="Max 13 qubits"):
        DensityMatrixSimulator(num_qubits=14)


def test_hadamard_gate():
    sim = DensityMatrixSimulator(num_qubits=1)
    sim.apply("H", (0,))
    probs = sim.probabilities()
    np.testing.assert_allclose(probs, [0.5, 0.5])
    assert sim.purity == pytest.approx(1.0)


def test_bell_state():
    sim = DensityMatrixSimulator(num_qubits=2)
    sim.apply("H", (0,))
    sim.apply("CX", (0, 1))

    probs = sim.probabilities()
    np.testing.assert_allclose(probs, [0.5, 0.0, 0.0, 0.5])
    assert sim.purity == pytest.approx(1.0)

    # Sample
    counts = sim.sample(shots=100)
    assert set(counts.keys()).issubset({"00", "11"})
    assert sum(counts.values()) == 100


def test_parametric_gate():
    sim = DensityMatrixSimulator(num_qubits=1)
    sim.apply("RX", (0,), params=(np.pi,))
    probs = sim.probabilities()
    np.testing.assert_allclose(probs, [0.0, 1.0], atol=1e-7)


def test_depolarizing_noise():
    sim = DensityMatrixSimulator(num_qubits=1)
    sim.apply("H", (0,))
    initial_purity = sim.purity

    # Apply depolarizing noise
    sim.apply_depolarizing(qubit=0, p=0.3)
    assert sim.purity < initial_purity

    # Test p <= 0 does nothing
    p_before = sim.purity
    sim.apply_depolarizing(qubit=0, p=0.0)
    assert sim.purity == p_before


def test_unknown_gate_and_missing_params():
    sim = DensityMatrixSimulator(num_qubits=1)
    with pytest.raises(DensityMatrixError, match="Unknown gate"):
        sim.apply("NONEXISTENT_GATE", (0,))

    with pytest.raises(DensityMatrixError, match="requires parameters"):
        sim.apply("RX", (0,))
