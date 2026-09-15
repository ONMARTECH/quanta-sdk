"""Tests for quanta.visualize — ASCII circuit drawing."""


from quanta import CCX, CX, RX, H, circuit, measure
from quanta.visualize import draw


def test_draw_no_op_circuit():
    @circuit(qubits=1)
    def no_op(q):
        pass

    out = draw(no_op)
    assert "q[0]:" in out


def test_draw_single_qubit():
    @circuit(qubits=1)
    def single(q):
        H(q[0])
        return measure(q)

    out = draw(single)
    assert "q[0]:" in out
    assert "H" in out
    assert "M" in out


def test_draw_bell_state():
    @circuit(qubits=2)
    def bell(q):
        H(q[0])
        CX(q[0], q[1])
        return measure(q)

    out = draw(bell)
    assert "q[0]:" in out
    assert "q[1]:" in out
    assert "H" in out


def test_draw_toffoli_and_parametric():
    @circuit(qubits=3)
    def complex_c(q):
        H(q[0])
        RX(1.57)(q[1])
        CCX(q[0], q[1], q[2])
        return measure(q)

    out = draw(complex_c)
    assert "q[2]:" in out
    assert "Rx" in out or "RX" in out
