"""Tests for quanta.core.custom_gate — User-defined quantum gates."""

import numpy as np
import pytest

from quanta.core.circuit import circuit
from quanta.core.custom_gate import CustomGateError, custom_gate
from quanta.core.gates import GATE_REGISTRY
from quanta.core.measure import measure
from quanta.runner import run


@pytest.fixture(autouse=True)
def clean_registry():
    initial_keys = set(GATE_REGISTRY.keys())
    yield
    current_keys = set(GATE_REGISTRY.keys())
    for k in current_keys - initial_keys:
        del GATE_REGISTRY[k]


def test_custom_gate_single_qubit():
    sqrt_z = custom_gate("TestSqrtZ", [[1, 0], [0, 1j]])
    assert sqrt_z.name == "TestSqrtZ"
    assert sqrt_z.num_qubits == 1
    assert "CustomGate" in repr(sqrt_z)

    @circuit(qubits=1)
    def circ(q):
        sqrt_z(q[0])
        return measure(q)

    res = run(circ, shots=50)
    assert res is not None
    assert "0" in res.counts


def test_custom_gate_two_qubit():
    diag_cz = custom_gate("TestCustomCZ", np.diag([1.0, 1.0, 1.0, -1.0]))
    assert diag_cz.num_qubits == 2

    @circuit(qubits=2)
    def circ(q):
        diag_cz(q[0], q[1])
        return measure(q)

    res = run(circ, shots=50)
    assert "00" in res.counts


def test_custom_gate_not_square():
    with pytest.raises(CustomGateError, match="must be square"):
        custom_gate("BadRect", [[1, 0, 0], [0, 1, 0]])


def test_custom_gate_not_power_of_two():
    with pytest.raises(CustomGateError, match="power of 2"):
        custom_gate("BadDim3", np.eye(3))


def test_custom_gate_not_unitary():
    with pytest.raises(CustomGateError, match="not unitary"):
        custom_gate("BadNonUnitary", [[1, 1], [0, 1]])


def test_custom_gate_duplicate_name():
    with pytest.raises(CustomGateError, match="already registered"):
        custom_gate("H", [[0, 1], [1, 0]])
