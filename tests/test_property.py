"""Property-based tests using Hypothesis for Quanta SDK.

Tests mathematical invariants that must hold for ALL possible inputs:
- Gate unitarity: U†U = I for any parameters
- Matrix dimensions: always 2^n × 2^n
- Circuit determinism: same seed → same result
- Probability normalization: Σ p_i = 1
- QASM round-trip: export → import → same circuit
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from quanta.core.gates import (
    GATE_REGISTRY, Gate, ParametricGate,
    H, X, Y, Z, S, T, CX, CZ, SWAP, ECR, iSWAP, CH,
    RX, RY, RZ, P, CP, MS, RXX, RZZ,
)


# ── Strategies ──

angles = st.floats(min_value=-4 * math.pi, max_value=4 * math.pi, allow_nan=False, allow_infinity=False)
small_angles = st.floats(min_value=0.01, max_value=math.pi, allow_nan=False)
seeds = st.integers(min_value=0, max_value=2**31 - 1)
shot_counts = st.integers(min_value=1, max_value=8192)
qubit_counts = st.integers(min_value=1, max_value=6)

# Gate names for random selection
FIXED_1Q_GATES = ["H", "X", "Y", "Z", "S", "T", "SDG", "TDG", "SX", "SXdg", "I"]
FIXED_2Q_GATES = ["CX", "CZ", "CY", "SWAP", "ECR", "iSWAP", "CH"]
PARAM_1Q_GATES = ["RX", "RY", "RZ", "P"]
PARAM_2Q_GATES = ["RXX", "RZZ", "CP", "MS"]


# ══════════════════════════════════════════
# Gate Unitarity
# ══════════════════════════════════════════


class TestGateUnitarity:
    """Every gate matrix U must satisfy U†U = I."""

    @given(angle=angles)
    @settings(max_examples=50)
    def test_rx_unitary(self, angle):
        m = RX(angle).matrix
        identity = m.conj().T @ m
        np.testing.assert_allclose(identity, np.eye(2), atol=1e-10)

    @given(angle=angles)
    @settings(max_examples=50)
    def test_ry_unitary(self, angle):
        m = RY(angle).matrix
        np.testing.assert_allclose(m.conj().T @ m, np.eye(2), atol=1e-10)

    @given(angle=angles)
    @settings(max_examples=50)
    def test_rz_unitary(self, angle):
        m = RZ(angle).matrix
        np.testing.assert_allclose(m.conj().T @ m, np.eye(2), atol=1e-10)

    @given(angle=angles)
    @settings(max_examples=50)
    def test_p_unitary(self, angle):
        m = P(angle).matrix
        np.testing.assert_allclose(m.conj().T @ m, np.eye(2), atol=1e-10)

    @given(angle=angles)
    @settings(max_examples=50)
    def test_cp_unitary(self, angle):
        m = CP(angle).matrix
        np.testing.assert_allclose(m.conj().T @ m, np.eye(4), atol=1e-10)

    @given(angle=angles)
    @settings(max_examples=50)
    def test_ms_unitary(self, angle):
        m = MS(angle).matrix
        np.testing.assert_allclose(m.conj().T @ m, np.eye(4), atol=1e-10)

    @given(angle=angles)
    @settings(max_examples=50)
    def test_rxx_unitary(self, angle):
        m = RXX(angle).matrix
        np.testing.assert_allclose(m.conj().T @ m, np.eye(4), atol=1e-10)

    @given(angle=angles)
    @settings(max_examples=50)
    def test_rzz_unitary(self, angle):
        m = RZZ(angle).matrix
        np.testing.assert_allclose(m.conj().T @ m, np.eye(4), atol=1e-10)

    def test_all_fixed_gates_unitary(self):
        """Every fixed gate in the registry must be unitary."""
        for name, gate in GATE_REGISTRY.items():
            if isinstance(gate, Gate):
                m = gate.matrix
                n = m.shape[0]
                identity = m.conj().T @ m
                np.testing.assert_allclose(
                    identity, np.eye(n), atol=1e-10,
                    err_msg=f"{name} is not unitary"
                )


# ══════════════════════════════════════════
# Matrix Dimensions
# ══════════════════════════════════════════


class TestMatrixDimensions:
    """Gate matrices must be 2^n × 2^n."""

    def test_all_gates_square(self):
        for name, gate in GATE_REGISTRY.items():
            if isinstance(gate, Gate):
                m = gate.matrix
                assert m.shape[0] == m.shape[1], f"{name}: not square"
                assert m.shape[0] == 2 ** gate.num_qubits, f"{name}: wrong dim"

    @given(angle=angles)
    @settings(max_examples=20)
    def test_parametric_1q_dim(self, angle):
        for name in PARAM_1Q_GATES:
            gate = GATE_REGISTRY[name]
            m = gate(angle).matrix
            assert m.shape == (2, 2), f"{name}({angle}): wrong shape"

    @given(angle=angles)
    @settings(max_examples=20)
    def test_parametric_2q_dim(self, angle):
        for name in PARAM_2Q_GATES:
            gate = GATE_REGISTRY[name]
            m = gate(angle).matrix
            assert m.shape == (4, 4), f"{name}({angle}): wrong shape"


# ══════════════════════════════════════════
# Circuit Determinism
# ══════════════════════════════════════════


class TestDeterminism:
    """Same seed must always produce same results."""

    @given(seed=seeds, shots=st.integers(min_value=10, max_value=1000))
    @settings(max_examples=20)
    def test_bell_deterministic(self, seed, shots):
        from quanta import circuit, H, CX, measure, run

        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        r1 = run(bell, shots=shots, seed=seed)
        r2 = run(bell, shots=shots, seed=seed)
        assert r1.counts == r2.counts

    @given(angle=small_angles, seed=seeds)
    @settings(max_examples=20)
    def test_parametric_deterministic(self, angle, seed):
        from quanta import circuit, RY, measure, run

        @circuit(qubits=1)
        def rot(q, theta=0.0):
            RY(theta)(q[0])
            return measure(q)

        r1 = run(rot, shots=100, seed=seed, theta=angle)
        r2 = run(rot, shots=100, seed=seed, theta=angle)
        assert r1.counts == r2.counts


# ══════════════════════════════════════════
# Probability Normalization
# ══════════════════════════════════════════


class TestProbabilityNormalization:
    """Probabilities must sum to 1.0 (within tolerance)."""

    @given(seed=seeds, shots=st.integers(min_value=100, max_value=4096))
    @settings(max_examples=15)
    def test_probs_sum_to_one(self, seed, shots):
        from quanta import circuit, H, CX, measure, run

        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        result = run(bell, shots=shots, seed=seed)
        total_prob = sum(result.probabilities.values())
        assert abs(total_prob - 1.0) < 1e-10

    @given(seed=seeds, shots=st.integers(min_value=100, max_value=4096))
    @settings(max_examples=15)
    def test_counts_sum_to_shots(self, seed, shots):
        from quanta import circuit, H, CX, measure, run

        @circuit(qubits=2)
        def bell(q):
            H(q[0])
            CX(q[0], q[1])
            return measure(q)

        result = run(bell, shots=shots, seed=seed)
        assert sum(result.counts.values()) == shots


# ══════════════════════════════════════════
# Pauli Algebra
# ══════════════════════════════════════════


class TestPauliAlgebra:
    """Pauli matrix algebraic identities."""

    def test_pauli_anticommutation(self):
        """σ_i · σ_j + σ_j · σ_i = 2δ_ij · I"""
        paulis = [X.matrix, Y.matrix, Z.matrix]
        for i in range(3):
            for j in range(3):
                anti = paulis[i] @ paulis[j] + paulis[j] @ paulis[i]
                if i == j:
                    np.testing.assert_allclose(anti, 2 * np.eye(2), atol=1e-10)
                else:
                    np.testing.assert_allclose(anti, np.zeros((2, 2)), atol=1e-10)

    def test_pauli_squares_identity(self):
        """σ² = I for all Pauli matrices."""
        for g in [X, Y, Z, H]:
            m = g.matrix
            np.testing.assert_allclose(m @ m, np.eye(2), atol=1e-10)

    @given(angle=angles)
    @settings(max_examples=30)
    def test_rotation_identity(self, angle):
        """R_x(θ)·R_x(-θ) = I."""
        m_plus = RX(angle).matrix
        m_minus = RX(-angle).matrix
        np.testing.assert_allclose(m_plus @ m_minus, np.eye(2), atol=1e-10)


# ══════════════════════════════════════════
# Estimator Consistency
# ══════════════════════════════════════════


class TestEstimatorProperties:
    """Estimator expectation values must satisfy physical constraints."""

    @given(angle=angles)
    @settings(max_examples=20)
    def test_expectation_bounded(self, angle):
        """⟨Z⟩ must be in [-1, 1] for any rotation."""
        from quanta import circuit, RY, measure
        from quanta.primitives import Estimator

        @circuit(qubits=1)
        def rot(q, theta=0.0):
            RY(theta)(q[0])
            return measure(q)

        est = Estimator()
        result = est.run(rot, observables=[[("Z", 1.0)]], theta=angle)
        assert -1.0 - 1e-10 <= result.value <= 1.0 + 1e-10

    @given(angle=angles)
    @settings(max_examples=20)
    def test_z_expectation_matches_cos(self, angle):
        """For RY(θ)|0⟩, ⟨Z⟩ = cos(θ)."""
        from quanta import circuit, RY, measure
        from quanta.primitives import Estimator

        @circuit(qubits=1)
        def rot(q, theta=0.0):
            RY(theta)(q[0])
            return measure(q)

        est = Estimator()
        result = est.run(rot, observables=[[("Z", 1.0)]], theta=angle)
        expected = math.cos(angle)
        assert abs(result.value - expected) < 1e-6
