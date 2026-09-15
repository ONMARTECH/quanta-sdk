"""Tests for quanta.result — Circuit execution result with rich display."""

import numpy as np

from quanta.result import Result


def test_result_properties():
    res = Result(
        counts={"00": 700, "11": 300},
        shots=1000,
        num_qubits=2,
        circuit_name="bell",
        gate_count=2,
        depth=2,
    )
    probs = res.probabilities
    assert probs["00"] == 0.7
    assert probs["11"] == 0.3
    assert res.most_frequent == "00"


def test_result_dirac_notation_no_statevector():
    # Fallback to measurement probabilities
    res = Result(
        counts={"00": 500, "11": 500, "01": 0},
        shots=1000,
        num_qubits=2,
    )
    dirac = res.dirac_notation()
    assert "|00>" in dirac and "|11>" in dirac

    # Empty counts fallback to |0>
    empty_res = Result(counts={}, shots=100, num_qubits=2)
    assert empty_res.dirac_notation() == "|0>"


def test_result_dirac_notation_with_statevector():
    # Real, purely imaginary, and complex amplitudes
    sv = np.array([1 / np.sqrt(3), 1j / np.sqrt(3), (1 + 1j) / np.sqrt(6), 0.0], dtype=complex)
    res = Result(
        counts={"00": 333, "01": 333, "10": 334},
        shots=1000,
        num_qubits=2,
        statevector=sv,
    )
    dirac = res.dirac_notation()
    assert "|00>" in dirac
    assert "j|01>" in dirac
    assert "|10>" in dirac
    assert "|11>" not in dirac

    # All zeros statevector fallback
    res_zero = Result(
        counts={},
        shots=100,
        num_qubits=2,
        statevector=np.zeros(4, dtype=complex),
    )
    assert res_zero.dirac_notation() == "|00>"


def test_result_histogram():
    empty_res = Result(counts={}, shots=100, num_qubits=2)
    assert empty_res.histogram() == "No measurement results."

    res = Result(
        counts={"00": 500, "11": 500},
        shots=1000,
        num_qubits=2,
    )
    hist = res.histogram(width=20)
    assert "█" in hist
    assert "|00>" in hist
    assert "|11>" in hist


def test_result_summary():
    # Small result with statevector
    sv = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=complex)
    res = Result(
        counts={"00": 500, "11": 500},
        shots=1000,
        num_qubits=2,
        circuit_name="bell_circuit",
        gate_count=2,
        depth=2,
        statevector=sv,
    )
    summary = res.summary()
    assert "Quanta Result: bell_circuit" in summary
    assert "Qubits: 2" in summary
    assert "Shots:  1000" in summary
    assert str(res) == summary
    assert "bell_circuit" in repr(res)

    # Result with > 16 states to trigger overflow message
    many_counts = {format(i, "05b"): 10 for i in range(25)}
    many_res = Result(
        counts=many_counts,
        shots=250,
        num_qubits=5,
    )
    many_summary = many_res.summary()
    assert "... +" in many_summary


def test_result_repr_html():
    res = Result(
        counts={"00": 512, "11": 512},
        shots=1024,
        num_qubits=2,
        circuit_name="my_bell",
        gate_count=2,
        depth=2,
    )
    html = res._repr_html_()
    assert "my_bell" in html
    assert "2 qubits" in html
    assert "1,024 shots" in html
    assert "|00⟩" in html

    # With overflow and dirac
    many_counts = {format(i, "05b"): 10 for i in range(20)}
    many_res = Result(
        counts=many_counts,
        shots=200,
        num_qubits=5,
        circuit_name="",
    )
    many_html = many_res._repr_html_()
    assert "more states" in many_html
    assert "circuit" in many_html
