"""Tests for v0.9 new gates: ECR, iSWAP, CSWAP, CH, CP, MS."""

from __future__ import annotations

import math

import numpy as np

from quanta.core.gates import (
    CH,
    CP,
    CSWAP,
    ECR,
    GATE_REGISTRY,
    MS,
    iSWAP,
)


class TestGateRegistry:
    """Registry now has 31 gates."""

    def test_registry_count(self):
        assert len(GATE_REGISTRY) == 31

    def test_new_gates_in_registry(self):
        for name in ("ECR", "iSWAP", "CSWAP", "CH", "CP", "MS"):
            assert name in GATE_REGISTRY, f"{name} missing from registry"


class TestECR:
    """Echoed Cross-Resonance gate (IBM Heron native)."""

    def test_matrix_shape(self):
        assert ECR.matrix.shape == (4, 4)

    def test_unitary(self):
        m = ECR.matrix
        identity = m @ m.conj().T
        np.testing.assert_allclose(identity, np.eye(4), atol=1e-10)

    def test_involutory(self):
        """ECR² = I (up to global phase)."""
        m2 = ECR.matrix @ ECR.matrix
        # ECR is involutory: ECR² = I
        assert np.allclose(np.abs(m2), np.eye(4), atol=1e-10)

    def test_num_qubits(self):
        assert ECR.num_qubits == 2


class TestiSWAP:
    """Imaginary SWAP gate (Google Sycamore native)."""

    def test_matrix_shape(self):
        assert iSWAP.matrix.shape == (4, 4)

    def test_unitary(self):
        m = iSWAP.matrix
        np.testing.assert_allclose(m @ m.conj().T, np.eye(4), atol=1e-10)

    def test_swap_with_phase(self):
        """iSWAP swaps |01⟩ ↔ |10⟩ with phase i."""
        m = iSWAP.matrix
        # |01⟩ → i|10⟩
        assert m[2, 1] == 1j
        # |10⟩ → i|01⟩
        assert m[1, 2] == 1j
        # |00⟩ and |11⟩ unchanged
        assert m[0, 0] == 1
        assert m[3, 3] == 1

    def test_num_qubits(self):
        assert iSWAP.num_qubits == 2


class TestCSWAP:
    """Controlled-SWAP (Fredkin) gate."""

    def test_matrix_shape(self):
        assert CSWAP.matrix.shape == (8, 8)

    def test_unitary(self):
        m = CSWAP.matrix
        np.testing.assert_allclose(m @ m.conj().T, np.eye(8), atol=1e-10)

    def test_fredkin_action(self):
        """When control=1, swaps targets."""
        m = CSWAP.matrix
        # |101⟩ (idx 5) → |110⟩ (idx 6)
        assert m[6, 5] == 1
        # |110⟩ (idx 6) → |101⟩ (idx 5)
        assert m[5, 6] == 1
        # When control=0, no swap
        assert m[1, 1] == 1  # |001⟩ stays
        assert m[2, 2] == 1  # |010⟩ stays

    def test_num_qubits(self):
        assert CSWAP.num_qubits == 3


class TestCH:
    """Controlled-Hadamard gate."""

    def test_matrix_shape(self):
        assert CH.matrix.shape == (4, 4)

    def test_unitary(self):
        m = CH.matrix
        np.testing.assert_allclose(m @ m.conj().T, np.eye(4), atol=1e-10)

    def test_controlled_action(self):
        """When control=0, identity on target. When control=1, H on target."""
        m = CH.matrix
        # |00⟩ → |00⟩
        assert m[0, 0] == 1
        # |01⟩ → |01⟩
        assert m[1, 1] == 1
        # |10⟩ → (|10⟩ + |11⟩)/√2
        s = 1 / np.sqrt(2)
        assert abs(m[2, 2] - s) < 1e-10
        assert abs(m[3, 2] - s) < 1e-10

    def test_num_qubits(self):
        assert CH.num_qubits == 2


class TestCP:
    """Controlled-Phase (parametric) gate."""

    def test_matrix_at_pi(self):
        """CP(π) = CZ."""
        m = CP(math.pi).matrix
        expected = np.diag([1, 1, 1, -1]).astype(complex)
        np.testing.assert_allclose(m, expected, atol=1e-10)

    def test_matrix_at_zero(self):
        """CP(0) = I."""
        m = CP(0).matrix
        np.testing.assert_allclose(m, np.eye(4), atol=1e-10)

    def test_unitary(self):
        m = CP(math.pi / 3).matrix
        np.testing.assert_allclose(m @ m.conj().T, np.eye(4), atol=1e-10)

    def test_num_qubits(self):
        assert CP(1.0).num_qubits == 2


class TestMS:
    """Mølmer-Sørensen gate (trapped-ion native)."""

    def test_matrix_at_pi(self):
        """MS(π) should produce Bell-like entanglement."""
        m = MS(math.pi).matrix
        assert m.shape == (4, 4)

    def test_matrix_at_zero(self):
        """MS(0) = I."""
        m = MS(0).matrix
        np.testing.assert_allclose(m, np.eye(4), atol=1e-10)

    def test_unitary(self):
        m = MS(math.pi / 4).matrix
        np.testing.assert_allclose(m @ m.conj().T, np.eye(4), atol=1e-10)

    def test_ms_equals_rxx(self):
        """MS gate has same matrix structure as RXX."""
        from quanta.core.gates import RXX
        theta = 0.7
        np.testing.assert_allclose(MS(theta).matrix, RXX(theta).matrix, atol=1e-10)

    def test_num_qubits(self):
        assert MS(1.0).num_qubits == 2


class TestDecompositions:
    """Test that new gates can be transpiled to CX-based gate sets."""

    def test_iswap_decomposition(self):
        from quanta.compiler.passes.translate import _DECOMPOSITION_RULES
        assert "iSWAP" in _DECOMPOSITION_RULES

    def test_ch_decomposition(self):
        from quanta.compiler.passes.translate import _DECOMPOSITION_RULES
        assert "CH" in _DECOMPOSITION_RULES

    def test_ecr_decomposition(self):
        from quanta.compiler.passes.translate import _DECOMPOSITION_RULES
        assert "ECR" in _DECOMPOSITION_RULES

    def test_cswap_decomposition(self):
        from quanta.compiler.passes.translate import _DECOMPOSITION_RULES
        assert "CSWAP" in _DECOMPOSITION_RULES


class TestNewGatesInCircuit:
    """Test that new gates work inside @circuit decorator."""

    def test_ch_circuit(self):
        from quanta import H, circuit, measure, run
        from quanta.core.gates import CH as ch_gate

        @circuit(qubits=2)
        def ch_test(q):
            H(q[0])
            ch_gate(q[0], q[1])
            return measure(q)

        result = run(ch_test, shots=1024)
        assert sum(result.counts.values()) == 1024

    def test_ecr_circuit(self):
        from quanta import circuit, measure, run
        from quanta.core.gates import ECR as ecr_gate

        @circuit(qubits=2)
        def ecr_test(q):
            ecr_gate(q[0], q[1])
            return measure(q)

        result = run(ecr_test, shots=1024)
        assert sum(result.counts.values()) == 1024

    def test_cp_circuit(self):
        from quanta import H, circuit, measure, run
        from quanta.core.gates import CP as cp_gate

        @circuit(qubits=2)
        def cp_test(q):
            H(q[0])
            cp_gate(math.pi)(q[0], q[1])
            return measure(q)

        result = run(cp_test, shots=1024)
        assert sum(result.counts.values()) == 1024
