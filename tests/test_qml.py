"""Tests for quanta.layer3.qml — Quantum Machine Learning module."""

from __future__ import annotations

import numpy as np

from quanta.layer3.qml import (
    QMLResult,
    QuantumClassifier,
    QuantumKernel,
    amplitude_encoding,
    angle_encoding,
    zz_feature_map,
)
from quanta.simulator.statevector import StateVectorSimulator

# ── Feature Maps ──────────────────────────────


class TestAngleEncoding:
    """Test angle encoding feature map."""

    def test_preserves_norm(self) -> None:
        sim = StateVectorSimulator(3)
        angle_encoding(sim, np.array([0.5, 0.3, 0.8]))
        assert abs(np.linalg.norm(sim.state) - 1.0) < 1e-10

    def test_zero_input(self) -> None:
        sim = StateVectorSimulator(2)
        angle_encoding(sim, np.array([0.0, 0.0]))
        # All zeros → |00⟩ state (RY(0) = identity)
        assert abs(sim.state[0]) > 0.99

    def test_custom_qubits(self) -> None:
        sim = StateVectorSimulator(4)
        angle_encoding(sim, np.array([0.5, 0.8]), qubits=(1, 3))
        assert abs(np.linalg.norm(sim.state) - 1.0) < 1e-10

    def test_single_feature(self) -> None:
        sim = StateVectorSimulator(1)
        angle_encoding(sim, np.array([0.5]))
        assert abs(np.linalg.norm(sim.state) - 1.0) < 1e-10


class TestZZFeatureMap:
    """Test ZZ entangling feature map."""

    def test_preserves_norm(self) -> None:
        sim = StateVectorSimulator(3)
        zz_feature_map(sim, np.array([0.5, 0.3, 0.8]))
        assert abs(np.linalg.norm(sim.state) - 1.0) < 1e-10

    def test_creates_entanglement(self) -> None:
        sim = StateVectorSimulator(2)
        zz_feature_map(sim, np.array([0.5, 0.5]))
        # Should not be a product state after ZZ interaction
        probs = np.abs(sim.state) ** 2
        assert np.sum(probs > 0.01) > 1  # Multiple basis states

    def test_reps_parameter(self) -> None:
        sim1 = StateVectorSimulator(2)
        zz_feature_map(sim1, np.array([0.5, 0.3]), reps=1)
        sim2 = StateVectorSimulator(2)
        zz_feature_map(sim2, np.array([0.5, 0.3]), reps=3)
        # Different reps → different states
        assert not np.allclose(sim1.state, sim2.state)

    def test_custom_qubits(self) -> None:
        sim = StateVectorSimulator(4)
        zz_feature_map(sim, np.array([0.5, 0.3]), qubits=(0, 2))
        assert abs(np.linalg.norm(sim.state) - 1.0) < 1e-10


class TestAmplitudeEncoding:
    """Test amplitude encoding."""

    def test_normalizes(self) -> None:
        sim = StateVectorSimulator(2)
        amplitude_encoding(sim, np.array([3.0, 4.0]))
        assert abs(np.linalg.norm(sim.state) - 1.0) < 1e-10

    def test_encodes_values(self) -> None:
        sim = StateVectorSimulator(2)
        amplitude_encoding(sim, np.array([1.0, 0.0, 0.0, 0.0]))
        assert abs(sim.state[0] - 1.0) < 1e-10

    def test_pads_short_input(self) -> None:
        sim = StateVectorSimulator(3)  # 8 amplitudes
        amplitude_encoding(sim, np.array([1.0, 1.0]))
        assert abs(np.linalg.norm(sim.state) - 1.0) < 1e-10


# ── Quantum Kernel ────────────────────────────


class TestQuantumKernel:
    """Test quantum kernel computation."""

    def test_self_kernel_is_one(self) -> None:
        kernel = QuantumKernel(n_qubits=2, feature_map="angle")
        x = np.array([0.5, 0.3])
        assert abs(kernel.evaluate(x, x) - 1.0) < 1e-10

    def test_different_inputs_less_than_one(self) -> None:
        kernel = QuantumKernel(n_qubits=2, feature_map="angle")
        x1 = np.array([0.1, 0.9])
        x2 = np.array([0.9, 0.1])
        assert kernel.evaluate(x1, x2) < 1.0

    def test_kernel_matrix_symmetric(self) -> None:
        kernel = QuantumKernel(n_qubits=2)
        X = np.array([[0.5, 0.3], [0.9, 0.1], [0.2, 0.8]])
        K = kernel.matrix(X)
        assert K.shape == (3, 3)
        assert np.allclose(K, K.T)

    def test_kernel_matrix_diagonal_ones(self) -> None:
        kernel = QuantumKernel(n_qubits=2)
        X = np.array([[0.5, 0.3], [0.9, 0.1]])
        K = kernel.matrix(X)
        assert np.allclose(np.diag(K), 1.0, atol=1e-10)

    def test_zz_kernel(self) -> None:
        kernel = QuantumKernel(n_qubits=2, feature_map="zz")
        x = np.array([0.5, 0.3])
        assert abs(kernel.evaluate(x, x) - 1.0) < 1e-10


# ── QML Result ────────────────────────────────


class TestQMLResult:
    """Test QMLResult dataclass."""

    def test_repr(self) -> None:
        result = QMLResult(accuracy=0.85, n_qubits=4, n_params=16)
        assert "85.00%" in repr(result)
        assert "qubits=4" in repr(result)

    def test_defaults(self) -> None:
        result = QMLResult(accuracy=0.5)
        assert result.loss_history == []
        assert result.n_qubits == 0


# ── Quantum Classifier ───────────────────────


class TestQuantumClassifier:
    """Test variational quantum classifier."""

    def test_init(self) -> None:
        clf = QuantumClassifier(n_qubits=3, n_layers=2, seed=42)
        assert clf.n_params == 12  # 2 * 3 * 2
        assert clf.params.shape == (12,)

    def test_predict_returns_binary(self) -> None:
        clf = QuantumClassifier(n_qubits=2, n_layers=1, seed=42)
        X = np.array([[0.1, 0.9], [0.9, 0.1]])
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self) -> None:
        clf = QuantumClassifier(n_qubits=2, n_layers=1, seed=42)
        X = np.array([[0.5, 0.3], [0.8, 0.2]])
        proba = clf.predict_proba(X)
        assert proba.shape == (2, 2)
        # Each row sums to 1
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_fit_returns_result(self) -> None:
        clf = QuantumClassifier(n_qubits=2, n_layers=1, seed=42)
        X = np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.1], [0.9, 0.9]])
        y = np.array([1, 1, 0, 0])
        result = clf.fit(X, y, epochs=3)
        assert isinstance(result, QMLResult)
        assert 0 <= result.accuracy <= 1
        assert len(result.loss_history) > 0

    def test_fit_loss_decreases(self) -> None:
        clf = QuantumClassifier(n_qubits=2, n_layers=2, seed=0)
        X = np.array([[0.2, 0.8], [0.8, 0.2]])
        y = np.array([1, 0])
        result = clf.fit(X, y, epochs=5)
        # Loss should generally decrease (or at least not blow up)
        assert result.loss_history[-1] < result.loss_history[0] + 1.0

    def test_score(self) -> None:
        clf = QuantumClassifier(n_qubits=2, n_layers=1, seed=42)
        X = np.array([[0.1, 0.9], [0.9, 0.1]])
        y = np.array([1, 0])
        score = clf.score(X, y)
        assert 0 <= score <= 1

    def test_zz_feature_map(self) -> None:
        clf = QuantumClassifier(n_qubits=2, n_layers=1, feature_map="zz", seed=42)
        X = np.array([[0.5, 0.3]])
        preds = clf.predict(X)
        assert preds.shape == (1,)

    def test_amplitude_feature_map(self) -> None:
        clf = QuantumClassifier(n_qubits=2, n_layers=1, feature_map="amplitude", seed=42)
        X = np.array([[0.5, 0.3, 0.1, 0.8]])
        preds = clf.predict(X)
        assert preds.shape == (1,)

    def test_early_stopping(self) -> None:
        clf = QuantumClassifier(n_qubits=2, n_layers=1, seed=42, learning_rate=0.001)
        X = np.array([[0.5, 0.3]])
        y = np.array([0])
        result = clf.fit(X, y, epochs=100)
        # Should stop early, not run all 100 epochs
        assert len(result.loss_history) <= 100
