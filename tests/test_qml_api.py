"""
test_qml_api — Tests for quanta.qml top-level API (Task 7).

Tests Classifier, QSVM, FeatureMap, Kernel wrappers
and scikit-learn compatibility.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

from quanta.qml import (
    QSVM,
    Classifier,
    FeatureMap,
    Kernel,
    QMLResult,
    QSVMResult,
)


class TestFeatureMap:
    """Feature map registry tests."""

    def test_list_available(self):
        available = FeatureMap.list_available()
        assert "angle" in available
        assert "zz" in available
        assert "amplitude" in available

    def test_get_valid(self):
        fn = FeatureMap.get("angle")
        assert callable(fn)

    def test_get_invalid(self):
        with pytest.raises(ValueError, match="nonexistent"):
            FeatureMap.get("nonexistent")

    def test_describe_valid(self):
        desc = FeatureMap.describe("zz")
        assert "entangling" in desc.lower()

    def test_describe_invalid(self):
        with pytest.raises(ValueError, match="nonexistent"):
            FeatureMap.describe("nonexistent")


class TestClassifier:
    """Classifier wrapper tests."""

    def test_init_defaults(self):
        clf = Classifier()
        assert clf.n_qubits == 4
        assert clf.optimizer == "sgd"
        assert clf.feature_map == "angle"

    def test_init_custom(self):
        clf = Classifier(
            n_qubits=2, n_layers=3, feature_map="zz",
            optimizer="adam", seed=42,
        )
        assert clf.n_qubits == 2
        assert clf.optimizer == "adam"

    def test_invalid_optimizer(self):
        with pytest.raises(ValueError, match="nonexistent"):
            Classifier(optimizer="nonexistent")

    def test_invalid_feature_map(self):
        with pytest.raises(ValueError, match="bad_map"):
            Classifier(feature_map="bad_map")

    def test_get_params(self):
        clf = Classifier(n_qubits=3, optimizer="spsa")
        params = clf.get_params()
        assert params["n_qubits"] == 3
        assert params["optimizer"] == "spsa"
        assert "learning_rate" in params

    def test_set_params(self):
        clf = Classifier()
        result = clf.set_params(n_qubits=6, learning_rate=0.05)
        assert result is clf
        assert clf.n_qubits == 6
        assert clf.learning_rate == 0.05
        assert not clf._is_fitted

    def test_fit_predict_score(self):
        X = np.array([[0.1, 0.2], [0.8, 0.9], [0.2, 0.1], [0.9, 0.8]])
        y = np.array([0, 1, 0, 1])

        clf = Classifier(n_qubits=2, n_layers=1, seed=42)
        result = clf.fit(X, y, epochs=3)

        assert isinstance(result, QMLResult)
        assert clf._is_fitted

        preds = clf.predict(X)
        assert len(preds) == 4
        assert all(p in (0, 1) for p in preds)

        proba = clf.predict_proba(X)
        assert proba.shape == (4, 2)

        score = clf.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_dataframe_input(self):
        """Simulate pandas DataFrame input."""
        class FakeDF:
            def __init__(self, data):
                self.values = np.array(data)

        df = FakeDF([[0.1, 0.2], [0.3, 0.4]])
        result = Classifier._to_numpy(df)
        assert result.shape == (2, 2)

    def test_repr(self):
        clf = Classifier(n_qubits=3, feature_map="zz", optimizer="adam")
        r = repr(clf)
        assert "n_qubits=3" in r
        assert "zz" in r
        assert "adam" in r


class TestQSVM:
    """QSVM wrapper tests."""

    def test_init(self):
        qsvm = QSVM(n_qubits=3, regularization=2.0)
        assert qsvm.n_qubits == 3
        assert qsvm.regularization == 2.0

    def test_classify(self):
        X_train = [[0.0, 0.0], [1.0, 1.0]]
        y_train = [0, 1]
        X_test = [[0.5, 0.5]]

        qsvm = QSVM(n_qubits=2)
        result = qsvm.classify(X_train, y_train, X_test)

        assert isinstance(result, QSVMResult)
        assert len(result.predictions) == 1
        assert result.predictions[0] in (0, 1)

    def test_repr(self):
        r = repr(QSVM(n_qubits=4))
        assert "n_qubits=4" in r


class TestKernel:
    """Kernel alias test."""

    def test_kernel_is_quantum_kernel(self):
        from quanta.layer3.qml import QuantumKernel
        assert Kernel is QuantumKernel

    def test_kernel_evaluate(self):
        k = Kernel(n_qubits=2)
        val = k.evaluate(np.array([0.1, 0.2]), np.array([0.1, 0.2]))
        assert 0.0 <= val <= 1.0
        # Same input → high kernel value
        assert val > 0.9
