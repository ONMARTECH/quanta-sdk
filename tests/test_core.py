"""
tests/test_core.py — Core modül testleri.

QubitRef, QubitRegister, Gate sınıfları ve @circuit dekoratörünü test eder.
"""

import numpy as np
import pytest

from quanta.core.circuit import CircuitDefinition, circuit
from quanta.core.gates import CCX, CX, CZ, RY, RZ, SWAP, H, S, T, X, Y, Z
from quanta.core.types import (
    CircuitError,
    QubitIndexError,
    QubitRef,
    QubitRegister,
)

# ═══════════════════════════════════════════
#  QubitRef Testleri
# ═══════════════════════════════════════════

class TestQubitRef:
    """QubitRef immutable referans testleri."""

    def test_creation_and_index(self):
        q = QubitRef(index=3)
        assert q.index == 3

    def test_repr_shows_bracket_notation(self):
        assert repr(QubitRef(0)) == "q[0]"
        assert repr(QubitRef(5)) == "q[5]"

    def test_frozen_cannot_modify(self):
        q = QubitRef(0)
        with pytest.raises(AttributeError):
            q.index = 5  # type: ignore

    def test_equality_and_hashing(self):
        """Aynı indeksli QubitRef'ler eşit ve hashlenebilir."""
        q1 = QubitRef(0)
        q2 = QubitRef(0)
        assert q1 == q2
        assert hash(q1) == hash(q2)
        assert len({q1, q2}) == 1  # Set'te tek eleman


# ═══════════════════════════════════════════
#  QubitRegister Testleri
# ═══════════════════════════════════════════

class TestQubitRegister:
    """QubitRegister indeksleme ve iterasyon testleri."""

    def test_creation_and_length(self):
        reg = QubitRegister(5)
        assert len(reg) == 5

    def test_indexing_returns_qubitref(self):
        reg = QubitRegister(3)
        assert reg[0] == QubitRef(0)
        assert reg[2] == QubitRef(2)

    def test_negative_indexing(self):
        reg = QubitRegister(3)
        assert reg[-1] == QubitRef(2)

    def test_out_of_range_raises_error(self):
        reg = QubitRegister(3)
        with pytest.raises(QubitIndexError):
            reg[3]

    def test_iteration(self):
        reg = QubitRegister(3)
        qubits = list(reg)
        assert len(qubits) == 3
        assert qubits[0] == QubitRef(0)

    def test_zero_qubits_raises_error(self):
        with pytest.raises(CircuitError):
            QubitRegister(0)


# ═══════════════════════════════════════════
#  Gate Testleri
# ═══════════════════════════════════════════

class TestGates:
    """Kapı matrisleri ve özelliklerinin testleri."""

    def test_hadamard_matrix_is_correct(self):
        """H kapısı: [[1,1],[1,-1]] / √2."""
        expected = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        np.testing.assert_allclose(H.matrix, expected, atol=1e-10)

    def test_pauli_x_is_bit_flip(self):
        """X kapısı: [[0,1],[1,0]]."""
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_allclose(X.matrix, expected, atol=1e-10)

    def test_pauli_y_matrix(self):
        expected = np.array([[0, -1j], [1j, 0]])
        np.testing.assert_allclose(Y.matrix, expected, atol=1e-10)

    def test_pauli_z_is_phase_flip(self):
        expected = np.array([[1, 0], [0, -1]])
        np.testing.assert_allclose(Z.matrix, expected, atol=1e-10)

    def test_all_single_qubit_gates_are_unitary(self):
        """Tüm tek-qubit kapılar unitär olmalı: U†U = I."""
        for gate in [H, X, Y, Z, S, T]:
            m = gate.matrix
            product = m.conj().T @ m
            np.testing.assert_allclose(
                product, np.eye(2), atol=1e-10,
                err_msg=f"{gate.name} kapısı unitär değil!"
            )

    def test_cnot_matrix_is_correct(self):
        """CX kapısı 4×4 matris."""
        assert CX.matrix.shape == (4, 4)
        assert CX.num_qubits == 2

    def test_all_two_qubit_gates_are_unitary(self):
        for gate in [CX, CZ, SWAP]:
            m = gate.matrix
            product = m.conj().T @ m
            np.testing.assert_allclose(
                product, np.eye(4), atol=1e-10,
                err_msg=f"{gate.name} kapısı unitär değil!"
            )

    def test_toffoli_is_8x8_unitary(self):
        m = CCX.matrix
        assert m.shape == (8, 8)
        product = m.conj().T @ m
        np.testing.assert_allclose(product, np.eye(8), atol=1e-10)

    def test_parametric_ry_creates_correct_matrix(self):
        """RY(π) = iY = [[0,-1],[1,0]]."""
        m = RY(np.pi).matrix
        expected = np.array([[0, -1], [1, 0]], dtype=complex)
        np.testing.assert_allclose(m, expected, atol=1e-10)

    def test_parametric_rz_at_zero_is_identity(self):
        """RZ(0) = I."""
        m = RZ(0).matrix
        np.testing.assert_allclose(m, np.eye(2), atol=1e-10)


# ═══════════════════════════════════════════
#  Circuit Dekoratörü Testleri
# ═══════════════════════════════════════════

class TestCircuit:
    """@circuit dekoratörü testleri."""

    def test_circuit_creates_definition(self):
        @circuit(qubits=2)
        def my_circuit(q):
            H(q[0])
        assert isinstance(my_circuit, CircuitDefinition)
        assert my_circuit.num_qubits == 2

    def test_circuit_preserves_name(self):
        @circuit(qubits=1)
        def fancy_name(q):
            pass
        assert fancy_name.name == "fancy_name"

    def test_build_records_instructions(self):
        @circuit(qubits=2)
        def test_circ(q):
            H(q[0])
            CX(q[0], q[1])

        builder = test_circ.build()
        assert len(builder.instructions) == 2
        assert builder.instructions[0].gate_name == "H"
        assert builder.instructions[1].gate_name == "CX"

    def test_invalid_qubit_count_raises_error(self):
        with pytest.raises(CircuitError):
            @circuit(qubits=0)
            def bad(q):
                pass

    def test_broadcast_records_multiple_instructions(self):
        """H(q) = H(q[0]), H(q[1]), H(q[2])."""
        @circuit(qubits=3)
        def broadcast_test(q):
            H(q)  # Broadcast: tüm qubit'lere

        builder = broadcast_test.build()
        assert len(builder.instructions) == 3
        assert all(i.gate_name == "H" for i in builder.instructions)
