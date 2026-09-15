"""
quanta.qml -- Quantum Machine Learning top-level API.

Provides a clean, scikit-learn compatible interface for quantum-enhanced
classification, regression, and kernel methods using variational circuits.

Quick Start (Classification):
    >>> from quanta.qml import Classifier
    >>> clf = Classifier(n_qubits=4, feature_map="zz", optimizer="adam")
    >>> clf.fit(X_train, y_train, epochs=30)
    >>> print(f"Accuracy: {clf.score(X_test, y_test):.2%}")

Quick Start (Continuous Regression):
    >>> from quanta.qml import Regressor
    >>> reg = Regressor(n_qubits=4, feature_map="angle", optimizer="adam")
    >>> reg.fit(X_train, y_train, epochs=30)
    >>> print(f"R2 Score: {reg.score(X_test, y_test):.4f}")

Feature Maps:
    >>> FeatureMap.list_available()
    ['angle', 'zz', 'amplitude']

Optimizers:
    Supported: "sgd", "adam" (default for Regressor), "spsa"
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quanta.layer3.qml import (
    QMLResult,
    QuantumClassifier,
    QuantumKernel,
    QuantumRegressor,
    RegressionResult,
    amplitude_encoding,
    angle_encoding,
    zz_feature_map,
)
from quanta.layer3.qsvm import QSVMResult, qsvm_classify
from quanta.qml.ansatz import Ansatz, AnsatzPreset

# ── Scikit-learn compatibility layer ──
try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
except ImportError:  # pragma: no cover
    class BaseEstimator:  # type: ignore[no-redef]
        """Fallback BaseEstimator when scikit-learn is not installed."""

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            return {}

        def set_params(self, **params: Any) -> Any:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)
            return self

    class ClassifierMixin:  # type: ignore[no-redef]
        """Fallback ClassifierMixin."""
        pass

    class RegressorMixin:  # type: ignore[no-redef]
        """Fallback RegressorMixin."""
        pass


__all__ = [
    "Classifier",
    "Regressor",
    "QSVM",
    "FeatureMap",
    "Ansatz",
    "AnsatzPreset",
    "Kernel",
    "QMLResult",
    "RegressionResult",
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


# ── Helper for Array Conversions ──


def _to_numpy(X: Any) -> np.ndarray:
    """Convert input to numpy array. Supports DataFrame and lists."""
    if hasattr(X, "values"):
        return np.asarray(X.values, dtype=float)
    return np.asarray(X, dtype=float)


# ── Classifier (scikit-learn compatible wrapper) ──


class Classifier(BaseEstimator, ClassifierMixin):
    """Variational Quantum Classifier with scikit-learn compatible API.

    Wraps QuantumClassifier with additional features:
      - Optimizer selection: "sgd", "adam", "spsa"
      - DataFrame and pipeline input support
      - Full get_params() / set_params() for Pipeline & GridSearchCV

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
        ansatz: str = "hardware_efficient",
        learning_rate: float = 0.1,
        optimizer: str = "sgd",
        seed: int | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.seed = seed

        # Validate ansatz
        Ansatz.get(ansatz)

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
        return _to_numpy(X)

    def fit(
        self,
        X: Any,
        y: Any,
        epochs: int = 30,
    ) -> QMLResult:
        X_np = _to_numpy(X)
        y_np = _to_numpy(y)
        self.classes_ = np.unique(y_np)
        self.n_features_in_ = X_np.shape[1] if X_np.ndim > 1 else 1
        result = self._clf.fit(X_np, y_np, epochs=epochs)
        self.result_ = result
        self.is_fitted_ = True
        self._is_fitted = True
        return result

    def predict(self, X: Any) -> np.ndarray:
        return self._clf.predict(_to_numpy(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        return self._clf.predict_proba(_to_numpy(X))

    def score(self, X: Any, y: Any) -> float:
        return self._clf.score(_to_numpy(X), _to_numpy(y))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "feature_map": self.feature_map,
            "ansatz": self.ansatz,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "seed": self.seed,
        }

    def set_params(self, **params: Any) -> Classifier:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
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


# ── Regressor (scikit-learn compatible wrapper) ──


class Regressor(BaseEstimator, RegressorMixin):
    """Variational Quantum Regressor with scikit-learn compatible API.

    Continuous regression model using parameterized quantum circuits:
      - Measures Pauli-Z expectation value on readout qubit(s)
      - Learns affine scaling (weight, bias) to match target continuous domain
      - Optimizes with parameter-shift rule minimizing Mean Squared Error (MSE)
      - Full scikit-learn Pipeline and GridSearchCV integration

    Example:
        >>> reg = Regressor(n_qubits=4, feature_map="angle", optimizer="adam")
        >>> reg.fit(X_train, y_train, epochs=30)
        >>> predictions = reg.predict(X_test)
        >>> print(f"R2 Score: {reg.score(X_test, y_test):.4f}")
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        feature_map: str = "angle",
        ansatz: str = "hardware_efficient",
        learning_rate: float = 0.1,
        optimizer: str = "adam",
        seed: int | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.seed = seed

        Ansatz.get(ansatz)
        FeatureMap.get(feature_map)
        if optimizer not in ("sgd", "adam", "spsa"):
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. "
                f"Supported: 'sgd', 'adam', 'spsa'"
            )

        self._reg = QuantumRegressor(
            n_qubits=n_qubits,
            n_layers=n_layers,
            feature_map=feature_map,
            learning_rate=learning_rate,
            optimizer=optimizer,
            seed=seed,
        )
        self._is_fitted = False

    @staticmethod
    def _to_numpy(X: Any) -> np.ndarray:
        return _to_numpy(X)

    def fit(
        self,
        X: Any,
        y: Any,
        epochs: int = 30,
    ) -> RegressionResult:
        X_np = _to_numpy(X)
        y_np = _to_numpy(y)
        self.n_features_in_ = X_np.shape[1] if X_np.ndim > 1 else 1
        result = self._reg.fit(X_np, y_np, epochs=epochs)
        self.result_ = result
        self.is_fitted_ = True
        self._is_fitted = True
        return result

    def predict(self, X: Any) -> np.ndarray:
        return self._reg.predict(_to_numpy(X))

    def score(self, X: Any, y: Any) -> float:
        return self._reg.score(_to_numpy(X), _to_numpy(y))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "feature_map": self.feature_map,
            "ansatz": self.ansatz,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "seed": self.seed,
        }

    def set_params(self, **params: Any) -> Regressor:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._reg = QuantumRegressor(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            feature_map=self.feature_map,
            learning_rate=self.learning_rate,
            optimizer=self.optimizer,
            seed=self.seed,
        )
        self._is_fitted = False
        return self

    def __repr__(self) -> str:
        return (
            f"Regressor(n_qubits={self.n_qubits}, "
            f"feature_map='{self.feature_map}', "
            f"optimizer='{self.optimizer}')"
        )


# ── QSVM (Scikit-Learn Compatible Wrapper) ──


class QSVM(BaseEstimator, ClassifierMixin):
    """Quantum Support Vector Machine with scikit-learn compatible API.

    Supports both:
      1. Classic classify() API:
         >>> qsvm = QSVM(n_qubits=4)
         >>> result = qsvm.classify(X_train, y_train, X_test)
      2. Scikit-Learn Estimator API (fit / predict / score):
         >>> qsvm = QSVM(n_qubits=4)
         >>> qsvm.fit(X_train, y_train)
         >>> preds = qsvm.predict(X_test)
    """

    def __init__(
        self,
        n_qubits: int | None = None,
        regularization: float = 1.0,
    ) -> None:
        self.n_qubits = n_qubits
        self.regularization = regularization
        self._is_fitted = False

    def fit(self, X: Any, y: Any) -> QSVM:
        X_tr = _to_numpy(X)
        y_tr = np.asarray(y, dtype=int)
        self.X_train_ = X_tr
        self.y_train_ = y_tr
        self.classes_ = np.unique(y_tr)
        self.n_features_in_ = X_tr.shape[1] if X_tr.ndim > 1 else 1
        self.is_fitted_ = True
        self._is_fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not getattr(self, "_is_fitted", False) and not hasattr(self, "X_train_"):
            raise ValueError(
                "This QSVM instance is not fitted yet. Call 'fit' before using this estimator."
            )
        res = self.classify(self.X_train_, self.y_train_, X)
        return np.asarray(res.predictions, dtype=int)

    def score(self, X: Any, y: Any) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == np.asarray(y, dtype=int)))

    def classify(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
    ) -> QSVMResult:
        X_tr = _to_numpy(X_train).tolist()
        y_tr = np.asarray(y_train, dtype=int).tolist()
        X_te = _to_numpy(X_test).tolist()

        return qsvm_classify(
            X_train=X_tr,
            y_train=y_tr,
            X_test=X_te,
            num_qubits=self.n_qubits,
            regularization=self.regularization,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "regularization": self.regularization,
        }

    def set_params(self, **params: Any) -> QSVM:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._is_fitted = False
        return self

    def __repr__(self) -> str:
        return f"QSVM(n_qubits={self.n_qubits}, C={self.regularization})"


# ── Kernel (re-export) ──

Kernel = QuantumKernel
