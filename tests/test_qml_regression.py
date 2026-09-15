"""tests/test_qml_regression.py -- Tests for quanta.qml regression & sklearn."""

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quanta.layer3.qml import QuantumRegressor, RegressionResult
from quanta.qml import QSVM, Classifier, Regressor


class TestQuantumRegressor:
    """Unit tests for layer3 QuantumRegressor."""

    def test_init_defaults(self):
        reg = QuantumRegressor()
        assert reg.n_qubits == 4
        assert reg.n_layers == 2
        assert reg.feature_map == "angle"
        assert reg.optimizer == "adam"
        assert reg.learning_rate == 0.1
        assert reg.n_params == 16

    def test_predict_shape(self):
        reg = QuantumRegressor(n_qubits=2, n_layers=1, seed=42)
        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        preds = reg.predict(X)
        assert preds.shape == (3,)
        assert isinstance(preds, np.ndarray)

    def test_fit_linear_trend(self):
        np.random.seed(42)
        X = np.linspace(-0.8, 0.8, 10).reshape(-1, 1)
        y = 1.5 * X.squeeze() + 0.2
        reg = QuantumRegressor(n_qubits=2, n_layers=1, optimizer="adam", learning_rate=0.2, seed=42)
        res = reg.fit(X, y, epochs=25)
        assert isinstance(res, RegressionResult)
        assert len(res.loss_history) > 0
        assert res.loss_history[-1] < res.loss_history[0]

    def test_spsa_optimizer(self):
        X = np.linspace(-0.5, 0.5, 6).reshape(-1, 1)
        y = 0.8 * X.squeeze() + 0.1
        reg = QuantumRegressor(n_qubits=2, n_layers=1, optimizer="spsa", learning_rate=0.1, seed=42)
        res = reg.fit(X, y, epochs=10)
        assert isinstance(res, RegressionResult)
        assert res.mse >= 0.0

    def test_sgd_optimizer(self):
        X = np.linspace(-0.5, 0.5, 6).reshape(-1, 1)
        y = 0.5 * X.squeeze()
        reg = QuantumRegressor(n_qubits=2, n_layers=1, optimizer="sgd", learning_rate=0.1, seed=42)
        res = reg.fit(X, y, epochs=10)
        assert isinstance(res, RegressionResult)

    def test_empty_dataset_raises(self):
        reg = QuantumRegressor()
        with pytest.raises(ValueError, match="empty"):
            reg.fit(np.array([]), np.array([]))

    def test_score_r2(self):
        reg = QuantumRegressor(n_qubits=2, n_layers=1, seed=42)
        X = np.array([[0.1], [0.5]])
        y = np.array([1.0, 1.0])
        # Constant target edge case
        score = reg.score(X, y)
        assert isinstance(score, float)


class TestRegressorWrapper:
    """Unit tests for top-level quanta.qml.Regressor wrapper."""

    def test_init_defaults(self):
        reg = Regressor()
        assert reg.n_qubits == 4
        assert reg.optimizer == "adam"
        assert reg.ansatz == "hardware_efficient"

    def test_init_custom(self):
        reg = Regressor(n_qubits=3, n_layers=1, feature_map="zz", optimizer="sgd", seed=10)
        assert reg.n_qubits == 3
        assert reg.feature_map == "zz"
        assert reg.optimizer == "sgd"

    def test_invalid_optimizer(self):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            Regressor(optimizer="invalid_opt")

    def test_invalid_feature_map(self):
        with pytest.raises(ValueError, match="Unknown feature map"):
            Regressor(feature_map="invalid_fm")

    def test_get_and_set_params(self):
        reg = Regressor(n_qubits=2, learning_rate=0.05)
        params = reg.get_params()
        assert params["n_qubits"] == 2
        assert params["learning_rate"] == 0.05

        reg.set_params(n_qubits=4, learning_rate=0.01)
        assert reg.n_qubits == 4
        assert reg.learning_rate == 0.01

    def test_fit_predict_score(self):
        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        y = np.array([0.5, 1.0, 1.5, 2.0])

        reg = Regressor(n_qubits=2, n_layers=1, optimizer="adam", learning_rate=0.1, seed=42)
        res = reg.fit(X, y, epochs=10)

        assert isinstance(res, RegressionResult)
        assert reg.is_fitted_
        preds = reg.predict(X)
        assert len(preds) == 4
        score = reg.score(X, y)
        assert isinstance(score, float)

    def test_dataframe_input(self):
        class FakeDF:
            def __init__(self, data):
                self.values = np.array(data)

        df = FakeDF([[0.1, 0.2], [0.3, 0.4]])
        arr = Regressor._to_numpy(df)
        assert arr.shape == (2, 2)

    def test_repr(self):
        reg = Regressor(n_qubits=3, feature_map="angle", optimizer="adam")
        r = repr(reg)
        assert "n_qubits=3" in r
        assert "angle" in r
        assert "adam" in r


class TestScikitLearnIntegration:
    """End-to-end scikit-learn Pipeline, GridSearchCV, and QSVM tests."""

    def test_pipeline_regressor(self):
        X = np.linspace(-1, 1, 10).reshape(-1, 1)
        y = 1.2 * X.squeeze() + 0.3
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Regressor(n_qubits=2, n_layers=1, optimizer="adam", seed=42)),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (10,)
        score = pipe.score(X, y)
        assert isinstance(score, float)

    def test_gridsearch_regressor(self):
        X = np.linspace(-0.5, 0.5, 8).reshape(-1, 1)
        y = 0.5 * X.squeeze()
        param_grid = {"learning_rate": [0.05, 0.1]}
        grid = GridSearchCV(Regressor(n_qubits=2, n_layers=1, seed=42), param_grid, cv=2)
        grid.fit(X, y)
        assert "learning_rate" in grid.best_params_
        assert grid.best_score_ is not None

    def test_cross_val_score_regressor(self):
        X = np.linspace(-0.5, 0.5, 8).reshape(-1, 1)
        y = 0.5 * X.squeeze()
        scores = cross_val_score(Regressor(n_qubits=2, n_layers=1, seed=42), X, y, cv=2)
        assert len(scores) == 2

    def test_pipeline_classifier(self):
        X = np.array([[0.1, 0.2], [0.8, 0.9], [0.2, 0.1], [0.9, 0.8]])
        y = np.array([0, 1, 0, 1])
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", Classifier(n_qubits=2, n_layers=1, seed=42)),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == 4
        score = pipe.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_qsvm_estimator_interface(self):
        X_train = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y_train = np.array([0, 1, 1, 0])
        X_test = np.array([[0.1, 0.1], [0.9, 0.9]])

        qsvm = QSVM(n_qubits=2)
        # Check unfitted error
        with pytest.raises(ValueError, match="not fitted yet"):
            qsvm.predict(X_test)

        # Fit
        fitted = qsvm.fit(X_train, y_train)
        assert fitted is qsvm
        assert qsvm.is_fitted_

        preds = qsvm.predict(X_test)
        assert len(preds) == 2

        score = qsvm.score(X_train, y_train)
        assert score == 1.0

        # Pipeline test
        pipe_svm = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", QSVM(n_qubits=2)),
        ])
        pipe_svm.fit(X_train, y_train)
        pipe_preds = pipe_svm.predict(X_test)
        assert len(pipe_preds) == 2
