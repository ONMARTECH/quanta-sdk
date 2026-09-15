"""Tests for quanta.simulator.accelerated — GPU/JIT acceleration module."""

from unittest.mock import MagicMock, patch

import numpy as np

from quanta.simulator.accelerated import (
    _detect_backend,
    get_array_module,
    get_backend_info,
    tensor_contract,
    xp,
)


def test_accelerated_basic():
    arr_mod = xp()
    assert arr_mod is np or hasattr(arr_mod, "ndarray")
    assert get_array_module() is arr_mod

    info = get_backend_info()
    assert "backend" in info
    assert "device" in info


def test_tensor_contract_single_qubit():
    # H gate on |0>
    h_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    state = np.array([1, 0], dtype=complex)
    new_state = tensor_contract(h_gate, state, qubits=(0,), num_qubits=1)
    expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
    np.testing.assert_allclose(new_state, expected)


def test_tensor_contract_two_qubits():
    # CX gate on |10> -> |11>
    cx_gate = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex)
    state = np.array([0, 0, 1, 0], dtype=complex)  # |10>
    new_state = tensor_contract(cx_gate, state, qubits=(0, 1), num_qubits=2)
    expected = np.array([0, 0, 0, 1], dtype=complex)  # |11>
    np.testing.assert_allclose(new_state, expected)


def test_backend_detection_mocking():
    # Test JAX detection mock
    mock_jax = MagicMock()
    mock_device = MagicMock()
    mock_device.platform = "gpu"
    mock_jax.devices.return_value = [mock_device]
    mock_jax.jit = lambda f: f

    with patch.dict("sys.modules", {"jax": mock_jax, "jax.numpy": np}):
        _detect_backend()
        info = get_backend_info()
        assert "backend" in info

    # Reset detection to default
    _detect_backend()
