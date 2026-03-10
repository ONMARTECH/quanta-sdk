"""
quanta.layer3.qml -- Quantum Machine Learning (QML) module.

Provides quantum-enhanced classification and regression using
parameterized quantum circuits (PQC) with classical optimizers.

Architecture:
  - QuantumFeatureMap: Encodes classical data into quantum states
  - QuantumKernel: Quantum kernel for SVM-style classification
  - QuantumClassifier: End-to-end variational quantum classifier

Supported feature maps:
  - ZZFeatureMap: Entangling feature map using ZZ interactions
  - AngleEncoding: Simple RY rotation encoding
  - AmplitudeEncoding: Log-depth amplitude encoding

Example:
    >>> from quanta.layer3.qml import QuantumClassifier
    >>> clf = QuantumClassifier(n_qubits=4, n_layers=2, feature_map="angle")
    >>> clf.fit(X_train, y_train, epochs=50)
    >>> predictions = clf.predict(X_test)
    >>> print(f"Accuracy: {clf.score(X_test, y_test):.2%}")

Theory:
    Variational quantum classifiers use:
    1. Feature map U_φ(x): Encodes input x into quantum state
    2. Variational ansatz V(θ): Parameterized unitary
    3. Measurement: Extracts class probabilities

    Total unitary: |ψ⟩ = V(θ) · U_φ(x) · |0⟩
    Loss: cross-entropy between measured probabilities and labels
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from quanta.simulator.statevector import StateVectorSimulator

__all__ = [
    "QuantumClassifier",
    "QuantumKernel",
    "QMLResult",
    "angle_encoding",
    "zz_feature_map",
    "amplitude_encoding",
]


# ── Feature Maps ──────────────────────────────────


def angle_encoding(
    sim: StateVectorSimulator,
    x: np.ndarray,
    qubits: tuple[int, ...] | None = None,
) -> None:
    """Encode features via RY rotations.

    Each feature x_i → RY(π · x_i) on qubit i.
    Simple and effective for small feature spaces.

    Args:
        sim: Simulator to apply encoding.
        x: Feature vector (values in [0, 1]).
        qubits: Which qubits to use (default: first len(x)).
    """
    if qubits is None:
        qubits = tuple(range(len(x)))
    for i, q in enumerate(qubits):
        if i < len(x):
            sim.apply("RY", (q,), (math.pi * float(x[i]),))


def zz_feature_map(
    sim: StateVectorSimulator,
    x: np.ndarray,
    qubits: tuple[int, ...] | None = None,
    reps: int = 2,
) -> None:
    """ZZ entangling feature map.

    Encodes features with entanglement:
      1. H on all qubits
      2. RZ(x_i) on each qubit
      3. RZZ(x_i * x_j) on adjacent pairs
    Repeated `reps` times for expressiveness.

    Args:
        sim: Simulator to apply encoding.
        x: Feature vector.
        qubits: Which qubits to use.
        reps: Number of repetitions.
    """
    n = len(x)
    if qubits is None:
        qubits = tuple(range(n))

    for _ in range(reps):
        # Hadamard layer
        for q in qubits[:n]:
            sim.apply("H", (q,))

        # Phase encoding
        for i, q in enumerate(qubits[:n]):
            sim.apply("RZ", (q,), (2.0 * float(x[i]),))

        # Entanglement encoding (ZZ interaction)
        for i in range(n - 1):
            q1, q2 = qubits[i], qubits[i + 1]
            phi = 2.0 * float(x[i]) * float(x[i + 1])
            sim.apply("CX", (q1, q2))
            sim.apply("RZ", (q2,), (phi,))
            sim.apply("CX", (q1, q2))


def amplitude_encoding(
    sim: StateVectorSimulator,
    x: np.ndarray,
) -> None:
    """Amplitude encoding: embed data directly into state amplitudes.

    Encodes 2^n features into n qubits.
    Input is normalized to unit vector.

    Args:
        sim: Simulator to apply encoding.
        x: Feature vector (length must be ≤ 2^n_qubits).
    """
    n_qubits = sim._num_qubits
    dim = 2 ** n_qubits

    # Pad and normalize
    padded = np.zeros(dim)
    padded[:len(x)] = x
    norm = np.linalg.norm(padded)
    if norm > 1e-12:
        padded = padded / norm

    # Direct state preparation
    sim._state = padded.astype(complex)


# ── Variational Ansatz ────────────────────────────


def _variational_layer(
    sim: StateVectorSimulator,
    params: np.ndarray,
    n_qubits: int,
    offset: int = 0,
) -> int:
    """Apply one variational layer (RY + RZ + CX chain).

    Returns number of parameters consumed.
    """
    idx = offset
    for q in range(n_qubits):
        sim.apply("RY", (q,), (float(params[idx]),))
        idx += 1
        sim.apply("RZ", (q,), (float(params[idx]),))
        idx += 1

    for q in range(n_qubits - 1):
        sim.apply("CX", (q, q + 1))

    return idx - offset


# ── Quantum Kernel ────────────────────────────────


class QuantumKernel:
    """Quantum kernel for kernel-based classification.

    Computes kernel matrix K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²
    using the ZZ feature map for quantum advantage.

    Example:
        >>> kernel = QuantumKernel(n_qubits=4, feature_map="zz")
        >>> K = kernel.matrix(X_train)
        >>> # Use with any kernel SVM
    """

    def __init__(
        self,
        n_qubits: int = 4,
        feature_map: str = "zz",
        reps: int = 2,
    ) -> None:
        """Initialize quantum kernel.

        Args:
            n_qubits: Number of qubits.
            feature_map: "zz" or "angle".
            reps: Feature map repetitions.
        """
        self.n_qubits = n_qubits
        self.feature_map = feature_map
        self.reps = reps

    def _encode(self, x: np.ndarray) -> np.ndarray:
        """Encode data into quantum state and return statevector."""
        sim = StateVectorSimulator(self.n_qubits)
        if self.feature_map == "zz":
            zz_feature_map(sim, x, reps=self.reps)
        else:
            angle_encoding(sim, x)
        return sim.state

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel value K(x1, x2) = |⟨φ(x1)|φ(x2)⟩|²."""
        s1 = self._encode(x1)
        s2 = self._encode(x2)
        return float(abs(np.vdot(s1, s2)) ** 2)

    def matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute kernel matrix for dataset X.

        Args:
            X: Data matrix (n_samples, n_features).

        Returns:
            (n_samples, n_samples) kernel matrix.
        """
        n = len(X)
        K = np.zeros((n, n))
        states = [self._encode(X[i]) for i in range(n)]
        for i in range(n):
            for j in range(i, n):
                K[i, j] = float(abs(np.vdot(states[i], states[j])) ** 2)
                K[j, i] = K[i, j]
        return K


# ── QML Result ────────────────────────────────────


@dataclass
class QMLResult:
    """Result of quantum machine learning training.

    Attributes:
        accuracy: Training accuracy.
        loss_history: Loss values during training.
        predictions: Last predictions made.
        n_qubits: Number of qubits used.
        n_params: Number of trainable parameters.
    """

    accuracy: float
    loss_history: list[float] = field(default_factory=list)
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    n_qubits: int = 0
    n_params: int = 0

    def __repr__(self) -> str:
        return (
            f"QMLResult(accuracy={self.accuracy:.2%}, "
            f"qubits={self.n_qubits}, params={self.n_params})"
        )


# ── Quantum Classifier ───────────────────────────


class QuantumClassifier:
    """Variational Quantum Classifier (VQC).

    End-to-end quantum classification pipeline:
      1. Feature map encodes classical data → quantum state
      2. Variational ansatz parameterizes the classifier
      3. Measurement gives class probabilities
      4. Classical optimizer updates parameters

    Example:
        >>> clf = QuantumClassifier(n_qubits=4, n_layers=2)
        >>> clf.fit(X_train, y_train, epochs=30)
        >>> acc = clf.score(X_test, y_test)
        >>> print(f"Accuracy: {acc:.2%}")
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        feature_map: str = "angle",
        learning_rate: float = 0.1,
        seed: int | None = None,
    ) -> None:
        """Initialize quantum classifier.

        Args:
            n_qubits: Number of qubits.
            n_layers: Variational ansatz depth.
            feature_map: "angle", "zz", or "amplitude".
            learning_rate: Gradient descent step size.
            seed: Random seed.
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map = feature_map
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        # 2 params per qubit per layer (RY + RZ)
        self.n_params = 2 * n_qubits * n_layers
        self.params = self.rng.uniform(
            -math.pi, math.pi, size=self.n_params,
        )
        self.loss_history: list[float] = []

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: encode data + apply ansatz + measure.

        Returns probabilities for each computational basis state.
        """
        sim = StateVectorSimulator(self.n_qubits)

        # Feature encoding
        if self.feature_map == "zz":
            zz_feature_map(sim, x)
        elif self.feature_map == "amplitude":
            amplitude_encoding(sim, x)
        else:
            angle_encoding(sim, x)

        # Variational layers
        idx = 0
        for _ in range(self.n_layers):
            consumed = _variational_layer(sim, self.params, self.n_qubits, idx)
            idx += consumed

        # Measurement probabilities
        probs = np.abs(sim.state) ** 2
        return probs

    def _predict_proba(self, x: np.ndarray) -> float:
        """Get probability of class 1 for a single sample.

        Uses parity of measurement: P(class=1) = Σ P(|i⟩) where popcount(i) is odd.
        """
        probs = self._forward(x)
        # Parity-based class probability
        p1 = sum(
            probs[i] for i in range(len(probs))
            if bin(i).count("1") % 2 == 1
        )
        return float(p1)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        eps = 1e-12
        total = 0.0
        for xi, yi in zip(X, y, strict=False):
            p = np.clip(self._predict_proba(xi), eps, 1 - eps)
            total += -(yi * math.log(p) + (1 - yi) * math.log(1 - p))
        return total / len(X)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
    ) -> QMLResult:
        """Train the quantum classifier.

        Uses parameter-shift rule for quantum-native gradients.

        Args:
            X: Training data (n_samples, n_features).
            y: Binary labels (0 or 1).
            epochs: Number of training epochs.

        Returns:
            QMLResult with accuracy and loss history.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.loss_history = []

        for epoch in range(epochs):
            # Compute loss
            loss = self._loss(X, y)
            self.loss_history.append(loss)

            # Parameter-shift gradients
            gradients = np.zeros(self.n_params)
            shift = math.pi / 2

            for i in range(self.n_params):
                # +shift
                self.params[i] += shift
                loss_plus = self._loss(X, y)
                self.params[i] -= shift

                # -shift
                self.params[i] -= shift
                loss_minus = self._loss(X, y)
                self.params[i] += shift

                gradients[i] = (loss_plus - loss_minus) / 2

            # Gradient descent
            self.params -= self.learning_rate * gradients

            # Early stopping
            if epoch > 5 and abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-6:
                break

        # Final accuracy
        predictions = self.predict(X)
        accuracy = float(np.mean(predictions == y))

        return QMLResult(
            accuracy=accuracy,
            loss_history=self.loss_history,
            predictions=predictions,
            n_qubits=self.n_qubits,
            n_params=self.n_params,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Args:
            X: Data matrix (n_samples, n_features).

        Returns:
            Array of predicted labels (0 or 1).
        """
        X = np.asarray(X, dtype=float)
        return np.array([
            1 if self._predict_proba(xi) > 0.5 else 0
            for xi in X
        ])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Data matrix (n_samples, n_features).

        Returns:
            Array of (P(class=0), P(class=1)) per sample.
        """
        X = np.asarray(X, dtype=float)
        probas = np.array([self._predict_proba(xi) for xi in X])
        return np.column_stack([1 - probas, probas])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy on test data.

        Args:
            X: Test data.
            y: True labels.

        Returns:
            Accuracy as a float.
        """
        predictions = self.predict(X)
        return float(np.mean(predictions == np.asarray(y)))
