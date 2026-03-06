"""
quanta.layer3.qsvm -- Quantum Support Vector Machine.

Quantum-enhanced machine learning for binary classification.
Uses quantum kernel estimation to compute feature maps in
exponentially high-dimensional Hilbert space.

The quantum advantage: kernel evaluation in O(log n) vs O(n^2)
for classical SVMs on large datasets.

Example:
    >>> from quanta.layer3.qsvm import qsvm_classify
    >>> X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    >>> y_train = [0, 1, 1, 0]  # XOR problem
    >>> result = qsvm_classify(X_train, y_train, [[0.5, 0.5]])
    >>> print(result.predictions)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quanta.simulator.statevector import StateVectorSimulator

__all__ = ["qsvm_classify", "QSVMResult"]


@dataclass
class QSVMResult:
    """Result of QSVM classification.

    Attributes:
        predictions: Predicted labels for test data.
        kernel_matrix: Quantum kernel matrix (training).
        accuracy: Training accuracy.
        num_qubits: Qubits used in feature map.
    """
    predictions: list[int]
    kernel_matrix: np.ndarray
    accuracy: float
    num_qubits: int

    def summary(self) -> str:
        lines = [
            "=== QSVM Classification ===",
            f"  Qubits:   {self.num_qubits}",
            f"  Accuracy: {self.accuracy:.1%}",
            f"  Predictions: {self.predictions}",
            f"  Kernel size: {self.kernel_matrix.shape[0]}x{self.kernel_matrix.shape[1]}",
        ]
        return "\n".join(lines)


def _encode_data_point(
    sim: StateVectorSimulator,
    x: list[float],
    num_qubits: int,
) -> None:
    """Encodes a data point into quantum state using ZZFeatureMap.

    Applies:
      1. Hadamard on all qubits
      2. RZ(x[i]) on qubit i
      3. Entangling CX + RZ(x[i]*x[j]) for pairs
    """
    n_features = len(x)

    # Layer: H + data encoding
    for q in range(num_qubits):
        sim.apply("H", (q,))

    for q in range(min(n_features, num_qubits)):
        sim.apply("RZ", (q,), (x[q] * np.pi,))

    # Entangling layer
    for q in range(min(n_features - 1, num_qubits - 1)):
        sim.apply("CX", (q, q + 1))
        interaction = x[q] * x[q + 1] * np.pi
        sim.apply("RZ", (q + 1,), (interaction,))
        sim.apply("CX", (q, q + 1))

    # Second encoding layer (depth-2 for better expressivity)
    for q in range(num_qubits):
        sim.apply("H", (q,))

    for q in range(min(n_features, num_qubits)):
        sim.apply("RZ", (q,), (x[q] * np.pi,))


def _quantum_kernel(
    x1: list[float],
    x2: list[float],
    num_qubits: int,
) -> float:
    """Computes quantum kernel: K(x1, x2) = |<phi(x1)|phi(x2)>|^2.

    This is the fidelity between two quantum-encoded states.
    """
    # Encode x1
    sim1 = StateVectorSimulator(num_qubits)
    _encode_data_point(sim1, x1, num_qubits)
    state1 = sim1.state

    # Encode x2
    sim2 = StateVectorSimulator(num_qubits)
    _encode_data_point(sim2, x2, num_qubits)
    state2 = sim2.state

    # Kernel = fidelity = |<psi1|psi2>|^2
    overlap = np.abs(np.dot(state1.conj(), state2)) ** 2
    return float(overlap)


def _build_kernel_matrix(
    X: list[list[float]],
    num_qubits: int,
) -> np.ndarray:
    """Builds the full quantum kernel (Gram) matrix."""
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            k_ij = _quantum_kernel(X[i], X[j], num_qubits)
            K[i, j] = k_ij
            K[j, i] = k_ij
    return K


def qsvm_classify(
    X_train: list[list[float]],
    y_train: list[int],
    X_test: list[list[float]],
    num_qubits: int | None = None,
    regularization: float = 1.0,
) -> QSVMResult:
    """Quantum SVM classification.

    Uses quantum kernel estimation for binary classification.
    The kernel is computed as fidelity between quantum-encoded states.

    Args:
        X_train: Training feature vectors.
        y_train: Training labels (0 or 1).
        X_test: Test feature vectors to classify.
        num_qubits: Qubits for feature map (auto = max(2, n_features)).
        regularization: SVM regularization parameter C.

    Returns:
        QSVMResult with predictions and kernel matrix.

    Example:
        >>> X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        >>> y = [0, 1, 1, 0]  # XOR
        >>> result = qsvm_classify(X, y, [[0.3, 0.7]])
    """
    n_features = len(X_train[0])
    n_train = len(X_train)

    if num_qubits is None:
        num_qubits = max(2, n_features)
    num_qubits = min(num_qubits, 10)  # Limit for simulator

    # Convert labels to +1/-1
    labels = np.array([1 if y == 1 else -1 for y in y_train], dtype=float)

    # Build quantum kernel matrix
    K_train = _build_kernel_matrix(X_train, num_qubits)

    # Solve SVM dual problem (simplified)
    # alpha = (K + (1/C)*I)^-1 @ y
    K_reg = K_train + (1.0 / regularization) * np.eye(n_train)
    try:
        alphas = np.linalg.solve(K_reg, labels)
    except np.linalg.LinAlgError:
        alphas = np.linalg.lstsq(K_reg, labels, rcond=None)[0]

    # Training accuracy
    train_pred = np.sign(K_train @ alphas)
    accuracy = float(np.mean(train_pred == labels))

    # Classify test points
    predictions = []
    for x_test in X_test:
        k_test = np.array([
            _quantum_kernel(x_test, X_train[i], num_qubits)
            for i in range(n_train)
        ])
        score = float(k_test @ alphas)
        predictions.append(1 if score > 0 else 0)

    return QSVMResult(
        predictions=predictions,
        kernel_matrix=K_train,
        accuracy=accuracy,
        num_qubits=num_qubits,
    )
