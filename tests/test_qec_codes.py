"""Tests for quanta.qec.codes — Standard Quantum Error Correction codes."""

import pytest

from quanta.core.circuit import CircuitDefinition
from quanta.qec.codes import (
    BitFlipCode,
    CodeInfo,
    PhaseFlipCode,
    QECCode,
    ShorCode,
    SteaneCode,
    correct_error,
)


def test_code_info():
    info = CodeInfo("TestCode", n=5, k=1, d=3)
    assert info.n == 5
    assert info.k == 1
    assert info.d == 3
    assert info.correctable_errors == 1
    assert "TestCode" in repr(info)
    assert "[[5,1,3]]" in repr(info)


def test_qec_base_class():
    base = QECCode()
    with pytest.raises(NotImplementedError):
        _ = base.info
    with pytest.raises(NotImplementedError):
        base.encode()
    with pytest.raises(NotImplementedError):
        base.decode()
    with pytest.raises(NotImplementedError):
        base.syndrome_measure()
    with pytest.raises(NotImplementedError):
        base.lookup_table()


def test_bit_flip_code():
    code = BitFlipCode()
    assert code.info.name == "BitFlip"
    assert code.info.n == 3

    enc = code.encode()
    assert isinstance(enc, CircuitDefinition)
    assert enc.num_qubits == 3

    dec = code.decode()
    assert isinstance(dec, CircuitDefinition)
    assert dec.num_qubits == 3

    syn = code.syndrome_measure()
    assert isinstance(syn, CircuitDefinition)
    assert syn.num_qubits == 5

    assert correct_error(code, "00") == "No error detected"
    assert "X on qubit 2" in correct_error(code, "01")
    assert "X on qubit 1" in correct_error(code, "10")
    assert "X on qubit 0" in correct_error(code, "11")
    assert correct_error(code, "unknown") == "No error detected"


def test_phase_flip_code():
    code = PhaseFlipCode()
    assert code.info.name == "PhaseFlip"
    assert code.info.n == 3

    enc = code.encode()
    assert isinstance(enc, CircuitDefinition)
    assert enc.num_qubits == 3

    dec = code.decode()
    assert isinstance(dec, CircuitDefinition)
    assert dec.num_qubits == 3

    assert correct_error(code, "00") == "No error detected"
    assert "Z on qubit 2" in correct_error(code, "01")
    assert "Z on qubit 1" in correct_error(code, "10")
    assert "Z on qubit 0" in correct_error(code, "11")


def test_steane_code():
    code = SteaneCode()
    assert code.info.name == "Steane"
    assert code.info.n == 7
    assert code.info.d == 3

    enc = code.encode()
    assert isinstance(enc, CircuitDefinition)
    assert enc.num_qubits == 7

    syn = code.syndrome_measure()
    assert isinstance(syn, CircuitDefinition)
    assert syn.num_qubits == 13


def test_shor_code():
    code = ShorCode()
    assert code.info.name == "Shor"
    assert code.info.n == 9
    assert code.info.d == 3

    enc = code.encode()
    assert isinstance(enc, CircuitDefinition)
    assert enc.num_qubits == 9

    dec = code.decode()
    assert isinstance(dec, CircuitDefinition)
    assert dec.num_qubits == 9

    table = code.lookup_table()
    assert "0000" in table
    assert "X on qubit 8" in correct_error(code, "0001")
    assert "X on qubit 5" in correct_error(code, "0100")
