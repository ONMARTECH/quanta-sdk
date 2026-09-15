import numpy as np
import pytest

# Skip this entire file immediately if cuquantum or cupy is missing
pytest.importorskip("cupy")
pytest.importorskip("cuquantum")

from quanta.simulator.custatevec import CuStateVecSimulator, SimulatorError
from quanta.simulator.factory import create_simulator


def test_custatevec_initialization():
    """Test that the GPU simulator initializes correctly in |0...0> state."""
    sim = create_simulator(num_qubits=2, method="custatevec")

    # State should be [1, 0, 0, 0]
    expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_allclose(sim.state, expected, atol=1e-7)

def test_custatevec_hadamard():
    """Test a simple Hadamard gate on GPU."""
    sim = create_simulator(num_qubits=1, method="custatevec")
    sim.apply("H", (0,))

    expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    np.testing.assert_allclose(sim.state, expected, atol=1e-7)

def test_custatevec_bell_state():
    """Test creating a Bell state on GPU."""
    sim = create_simulator(num_qubits=2, method="custatevec")
    sim.apply("H", (0,))
    sim.apply("CX", (0, 1))

    expected = np.array([1/np.sqrt(2), 0.0, 0.0, 1/np.sqrt(2)])
    np.testing.assert_allclose(sim.state, expected, atol=1e-7)

    # Check probabilities
    probs = sim.probabilities()
    np.testing.assert_allclose(probs, [0.5, 0.0, 0.0, 0.5], atol=1e-7)

def test_max_qubits_exceeded():
    """Test error handling when requesting too many qubits for GPU."""
    # Temporarily override MAX_QUBITS to test the exception throw gracefully
    old_max = CuStateVecSimulator.MAX_QUBITS
    CuStateVecSimulator.MAX_QUBITS = 5

    try:
        with pytest.raises(SimulatorError, match="Max 5 qubits supported"):
            create_simulator(6, method="custatevec")
    finally:
        CuStateVecSimulator.MAX_QUBITS = old_max

def test_custatevec_parametric():
    """Test parametric gates on GPU."""
    sim = create_simulator(num_qubits=1, method="custatevec")
    sim.apply("RX", (0,), params=(np.pi,))

    expected = np.array([0.0, -1j])
    np.testing.assert_allclose(sim.state, expected, atol=1e-7)
