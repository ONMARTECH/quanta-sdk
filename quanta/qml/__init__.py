"""
quanta.qml -- Quantum Machine Learning top-level scikit-learn compatible API.

Wires ``FeatureMap``, ``Ansatz`` presets (`hardware_efficient`, `strongly_entangling`,
`reuploading`), and optimizers (`sgd`, `adam`, `spsa`, `cobyla`) directly into
``Classifier`` and ``Regressor``.
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

    class RegressorMixin:  # type: ignore[no-redef]
        """Fallback RegressorMixin."""


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

_SUPPORTED_OPTIMIZERS = ("sgd", "adam", "spsa", "cobyla")


class FeatureMap:
    """Feature map discovery and access."""

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
        return list(cls._REGISTRY.keys())

    @classmethod
    def get(cls, name: str) -> Any:
        if name not in cls._REGISTRY:
            available = ", ".join(cls._REGISTRY.keys())
            raise ValueError(
                f"Unknown feature map '{name}'. Available: {available}"
            )
        return cls._REGISTRY[name]

    @classmethod
    def describe(cls, name: str) -> str:
        if name not in cls._DESCRIPTIONS:
            available = ", ".join(cls._DESCRIPTIONS.keys())
            raise ValueError(
                f"Unknown feature map '{name}'. Available: {available}"
            )
        return cls._DESCRIPTIONS[name]


def _to_numpy(X: Any) -> np.ndarray:
    """Convert input (list, ndarray, or DataFrame) to float64 numpy array."""
    if hasattr(X, "values"):
        return np.asarray(X.values, dtype=float)
    return np.asarray(X, dtype=float)


class Classifier(BaseEstimator, ClassifierMixin):
    """Variational Quantum Classifier with scikit-learn compatible API."""

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

        Ansatz.get(ansatz)
        FeatureMap.get(feature_map)
        if optimizer not in _SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. "
                f"Supported: {', '.join(repr(o) for o in _SUPPORTED_OPTIMIZERS)}"
            )

        self._clf = QuantumClassifier(
            n_qubits=n_qubits,
            n_layers=n_layers,
            feature_map=feature_map,
            learning_rate=learning_rate,
            seed=seed,
            ansatz=ansatz,
            optimizer=optimizer,
        )
        self._is_fitted = False

    @staticmethod
    def _to_numpy(X: Any) -> np.ndarray:
        return _to_numpy(X)

    def _rebuild_backend(self) -> None:
        Ansatz.get(self.ansatz)
        FeatureMap.get(self.feature_map)
        if self.optimizer not in _SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer '{self.optimizer}'. "
                f"Supported: {', '.join(repr(o) for o in _SUPPORTED_OPTIMIZERS)}"
            )
        self._clf = QuantumClassifier(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            feature_map=self.feature_map,
            learning_rate=self.learning_rate,
            seed=self.seed,
            ansatz=self.ansatz,
            optimizer=self.optimizer,
        )

    def fit(self, X: Any, y: Any, epochs: int = 30) -> QMLResult:
        # Ensure current self.ansatz and self.optimizer are synchronized (e.g. after sklearn clone)
        if (
            self._clf.ansatz != self.ansatz
            or self._clf.optimizer != self.optimizer
            or self._clf.n_qubits != self.n_qubits
            or self._clf.n_layers != self.n_layers
            or self._clf.feature_map != self.feature_map
            or self._clf.learning_rate != self.learning_rate
        ):
            self._rebuild_backend()

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
        self._rebuild_backend()
        self._is_fitted = False
        return self

    def __repr__(self) -> str:
        return (
            f"Classifier(n_qubits={self.n_qubits}, "
            f"feature_map='{self.feature_map}', "
            f"ansatz='{self.ansatz}', "
            f"optimizer='{self.optimizer}')"
        )


class Regressor(BaseEstimator, RegressorMixin):
    """Variational Quantum Regressor with scikit-learn compatible API."""

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
        if optimizer not in _SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. "
                f"Supported: {', '.join(repr(o) for o in _SUPPORTED_OPTIMIZERS)}"
            )

        self._reg = QuantumRegressor(
            n_qubits=n_qubits,
            n_layers=n_layers,
            feature_map=feature_map,
            learning_rate=learning_rate,
            optimizer=optimizer,
            seed=seed,
            ansatz=ansatz,
        )
        self._is_fitted = False

    @staticmethod
    def _to_numpy(X: Any) -> np.ndarray:
        return _to_numpy(X)

    def _rebuild_backend(self) -> None:
        Ansatz.get(self.ansatz)
        FeatureMap.get(self.feature_map)
        if self.optimizer not in _SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer '{self.optimizer}'. "
                f"Supported: {', '.join(repr(o) for o in _SUPPORTED_OPTIMIZERS)}"
            )
        self._reg = QuantumRegressor(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            feature_map=self.feature_map,
            learning_rate=self.learning_rate,
            optimizer=self.optimizer,
            seed=self.seed,
            ansatz=self.ansatz,
        )

    def fit(self, X: Any, y: Any, epochs: int = 30) -> RegressionResult:
        if (
            self._reg.ansatz != self.ansatz
            or self._reg.optimizer != self.optimizer
            or self._reg.n_qubits != self.n_qubits
            or self._reg.n_layers != self.n_layers
            or self._reg.feature_map != self.feature_map
            or self._reg.learning_rate != self.learning_rate
        ):
            self._rebuild_backend()

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
        self._rebuild_backend()
        self._is_fitted = False
        return self

    def __repr__(self) -> str:
        return (
            f"Regressor(n_qubits={self.n_qubits}, "
            f"feature_map='{self.feature_map}', "
            f"ansatz='{self.ansatz}', "
            f"optimizer='{self.optimizer}')"
        )


class QSVM(BaseEstimator, ClassifierMixin):
    """Quantum Support Vector Machine with scikit-learn compatible API."""

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
        res = qsvm_classify(
            X_train=X_tr.tolist(),
            y_train=y_tr.tolist(),
            X_test=X_tr[:1].tolist(),
            num_qubits=self.n_qubits,
            regularization=self.regularization,
        )
        self.support_ = np.asarray(res.support_indices, dtype=int)
        self.alphas_ = np.asarray(res.alphas, dtype=float)
        self.intercept_ = float(res.intercept)
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


Kernel = QuantumKernel
