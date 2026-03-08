"""
tests/conftest.py — Ortak test fixture'ları.

Tüm test dosyalarında kullanılan ortak devre tanımları
ve yardımcı fonksiyonlar burada tanımlanır.
"""

import pytest

from quanta import CX, H, X, circuit, measure


@pytest.fixture
def bell_circuit():
    """Bell state devresi fixture'ı."""
    @circuit(qubits=2)
    def bell(q):
        H(q[0])
        CX(q[0], q[1])
        return measure(q)
    return bell


@pytest.fixture
def ghz_circuit():
    """GHZ state devresi fixture'ı."""
    @circuit(qubits=3)
    def ghz(q):
        H(q[0])
        CX(q[0], q[1])
        CX(q[1], q[2])
        return measure(q)
    return ghz


@pytest.fixture
def x_circuit():
    """Basit X kapısı devresi (|0⟩ → |1⟩)."""
    @circuit(qubits=1)
    def flip(q):
        X(q[0])
        return measure(q)
    return flip
