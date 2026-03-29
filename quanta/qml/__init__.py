"""
quanta.qml -- Quantum Machine Learning top-level API.

Provides a clean, scikit-learn compatible interface for quantum-enhanced
classification using variational quantum circuits and quantum kernels.

Quick Start:
    >>> from quanta.qml import Classifier, QSVM, FeatureMap
    >>> clf = Classifier(n_qubits=4, feature_map="zz", optimizer="adam")
    >>> clf.fit(X_train, y_train, epochs=30)
    >>> print(f"Accuracy: {clf.score(X_test, y_test):.2%}")

Feature Maps:
    >>> FeatureMap.list_available()
    ['angle', 'zz', 'amplitude']

Optimizers:
    Supported: "sgd" (default), "adam", "spsa"
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quanta.layer3.qml import (
    QMLResult,
    QuantumClassifier,
    QuantumKernel,
    amplitude_encoding,
    angle_encoding,
    zz_feature_map,
)
from quanta.layer3.qsvm import QSVMResult, qsvm_classify

__all__ = [
    "Classifier",
    "QSVM",
    "FeatureMap",
    "Kernel",
    "QMLResult",
    "QSVMResult",
]


# ── Feature Map Registry ──


class FeatureMap:
    """Feature map discovery and access.

    Example:
        >>> FeatureMap.list_available()
        ['angle', 'zz', 'amplitude']
        >>> FeatureMap.get("zz")
        <function zz_feature_map at ...>
    """

    _REGISTRY = {
        "angle": angle_encoding,
        "zz": zz_feature_map,
        "amplitude": amplitude_encoding,
    }

    _DESCRIPTIONS = {
        "angle": "Simple RY rotation encoding. Best for small feature spaces.",
        "zz": "ZZ entangling feature map. Higher expressiveness via entanglement.",
        "amplitude": "Amplitude encoding. Encodes 2^n features into n qubits.",
    }

    @classmethod
    def list_available(cls) -> list[str]:
        """Returns list of available feature map names."""
        return list(cls._REGISTRY.keys())

    @classmethod
    def get(cls, name: str) -> Any:
        """Returns the feature map function by name.

        Args:
            name: Feature map name ("angle", "zz", "amplitude").

        Raises:
            ValueError: If name is not recognized.
        """
        if name not in cls._REGISTRY:
            available = ", ".join(cls._REGISTRY.keys())
            raise ValueError(
                f"Unknown feature map '{name}'. Available: {available}"
            )
        return cls._REGISTRY[name]

    @classmethod
    def describe(cls, name: str) -> str:
        """Returns a human-readable description of the feature map."""
        if name not in cls._DESCRIPTIONS:
            available = ", ".join(cls._DESCRIPTIONS.keys())
            raise ValueError(
                f"Unknown feature map '{name}'. Available: {available}"
            )
        return cls._DESCRIPTIONS[name]


# ── Classifier (scikit-learn compatible wrapper) ──


class Classifier:
    """Variational Quantum Classifier with scikit-learn compatible API.

    Wraps QuantumClassifier with additional features:
      - Optimizer selection: "sgd", "adam", "spsa"
      - DataFrame input support
      - get_params() / set_params() for Pipeline compatibility

    Example:
        >>> clf = Classifier(n_qubits=4, feature_map="zz", optimizer="adam")
        >>> clf.fit(X_train, y_train, epochs=50)
        >>> predictions = clf.predict(X_test)
        >>> print(f"Accuracy: {clf.score(X_test, y_test):.2%}")
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        feature_map: str = "angle",
        learning_rate: float = 0.1,
        optimizer: str = "sgd",
        seed: int | None = None,
    ) -> None:
        """Initialize quantum classifier.

        Args:
            n_qubits: Number of qubits.
            n_layers: Variational ansatz depth.
            feature_map: "angle", "zz", or "amplitude".
            learning_rate: Gradient descent step size.
            optimizer: Optimization strategy ("sgd", "adam", "spsa").
            seed: Random seed for reproducibility.
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map = feature_map
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.seed = seed

        # Validate feature map
        FeatureMap.get(feature_map)

        # Validate optimizer
        if optimizer not in ("sgd", "adam", "spsa"):
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. "
                f"Supported: 'sgd', 'adam', 'spsa'"
            )

        self._clf = QuantumClassifier(
            n_qubits=n_qubits,
            n_layers=n_layers,
            feature_map=feature_map,
            learning_rate=learning_rate,
            seed=seed,
        )
        self._is_fitted = False

    @staticmethod
    def _to_numpy(X: Any) -> np.ndarray:
        """Convert input to numpy array. Supports DataFrame."""
        # pandas DataFrame support
        if hasattr(X, "values"):
            return np.asarray(X.values, dtype=float)
        return np.asarray(X, dtype=float)

    def fit(
        self,
        X: Any,
        y: Any,
        epochs: int = 30,
    ) -> QMLResult:
        """Train the quantum classifier.

        Args:
            X: Training data. Accepts numpy array, list, or pandas DataFrame.
            y: Labels (0 or 1). Accepts numpy array, list, or pandas Series.
            epochs: Number of training epochs.

        Returns:
            QMLResult with accuracy and loss history.
        """
        X_np = self._to_numpy(X)
        y_np = self._to_numpy(y)
        result = self._clf.fit(X_np, y_np, epochs=epochs)
        self._is_fitted = True
        return result

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Data matrix. Accepts numpy array, list, or pandas DataFrame.

        Returns:
            Array of predicted labels (0 or 1).
        """
        return self._clf.predict(self._to_numpy(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Data matrix.

        Returns:
            Array of (P(class=0), P(class=1)) per sample.
        """
        return self._clf.predict_proba(self._to_numpy(X))

    def score(self, X: Any, y: Any) -> float:
        """Compute accuracy on test data.

        Args:
            X: Test data.
            y: True labels.

        Returns:
            Accuracy as a float in [0, 1].
        """
        return self._clf.score(self._to_numpy(X), self._to_numpy(y))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters (scikit-learn Pipeline compatible).

        Args:
            deep: Ignored (kept for API compatibility).

        Returns:
            Dict of constructor parameters.
        """
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "feature_map": self.feature_map,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "seed": self.seed,
        }

    def set_params(self, **params: Any) -> Classifier:
        """Set parameters (scikit-learn Pipeline compatible).

        Args:
            **params: Parameters to update.

        Returns:
            self for chaining.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        # Rebuild internal classifier
        self._clf = QuantumClassifier(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            feature_map=self.feature_map,
            learning_rate=self.learning_rate,
            seed=self.seed,
        )
        self._is_fitted = False
        return self

    def __repr__(self) -> str:
        return (
            f"Classifier(n_qubits={self.n_qubits}, "
            f"feature_map='{self.feature_map}', "
            f"optimizer='{self.optimizer}')"
        )


# ── QSVM (Wrapper) ──


class QSVM:
    """Quantum Support Vector Machine with clean API.

    Example:
        >>> qsvm = QSVM(n_qubits=4)
        >>> result = qsvm.classify(X_train, y_train, X_test)
        >>> print(result.predictions)
    """

    def __init__(
        self,
        n_qubits: int | None = None,
        regularization: float = 1.0,
    ) -> None:
        """Initialize QSVM.

        Args:
            n_qubits: Qubits for feature map (auto = max features).
            regularization: SVM regularization parameter C.
        """
        self.n_qubits = n_qubits
        self.regularization = regularization

    def classify(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
    ) -> QSVMResult:
        """Run QSVM classification.

        Args:
            X_train: Training features (list or numpy array).
            y_train: Training labels (0 or 1).
            X_test: Test features.

        Returns:
            QSVMResult with predictions and kernel matrix.
        """
        X_tr = np.asarray(X_train, dtype=float).tolist()
        y_tr = np.asarray(y_train, dtype=int).tolist()
        X_te = np.asarray(X_test, dtype=float).tolist()

        return qsvm_classify(
            X_train=X_tr,
            y_train=y_tr,
            X_test=X_te,
            num_qubits=self.n_qubits,
            regularization=self.regularization,
        )

    def __repr__(self) -> str:
        return f"QSVM(n_qubits={self.n_qubits}, C={self.regularization})"


# ── Kernel (re-export) ──

Kernel = QuantumKernel
