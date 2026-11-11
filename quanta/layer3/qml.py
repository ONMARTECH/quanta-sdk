"""
quanta.layer3.qml -- Quantum Machine Learning (QML) module.

Provides variational quantum classification, continuous quantum regression,
and fidelity quantum kernels with:
  - Exact chain-rule parameter-shift gradients ``(dL / d<Z>) * (d<Z> / dtheta)``
    where parameter-shift is applied strictly to the linear Pauli expectation
    ``<Z>`` (never directly to non-linear cross-entropy ``-log p``!).
  - Consistent MSB ``q[0]`` Pauli-Z expectation readout ``(i >> (n - 1)) & 1 == 0``.
  - Full integration with ``quanta.qml.ansatz.Ansatz`` presets and optimizers
    (``"sgd"``, ``"adam"``, ``"spsa"``, ``"cobyla"``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from quanta.simulator.base import SimulatorBackend
from quanta.simulator.factory import create_simulator


def _get_ansatz_preset(name: str):
    """Lazily resolve Ansatz preset to avoid circular import with quanta.qml."""
    from quanta.qml.ansatz import Ansatz

    return Ansatz.get(name)

__all__ = [
    "QuantumClassifier",
    "QuantumRegressor",
    "QuantumKernel",
    "QMLResult",
    "RegressionResult",
    "angle_encoding",
    "zz_feature_map",
    "amplitude_encoding",
]


# ── Feature Maps ──────────────────────────────────


def angle_encoding(
    sim: SimulatorBackend,
    x: np.ndarray,
    qubits: tuple[int, ...] | None = None,
) -> None:
    """Encode features via RY(pi * x_i) rotations."""
    if qubits is None:
        qubits = tuple(range(min(len(x), sim.num_qubits)))
    for i, q in enumerate(qubits):
        if i < len(x) and q < sim.num_qubits:
            sim.apply("RY", (q,), (math.pi * float(x[i]),))


def zz_feature_map(
    sim: SimulatorBackend,
    x: np.ndarray,
    qubits: tuple[int, ...] | None = None,
    reps: int = 2,
) -> None:
    """ZZ entangling feature map."""
    n = min(len(x), sim.num_qubits)
    if qubits is None:
        qubits = tuple(range(n))

    for _ in range(reps):
        for q in qubits[:n]:
            sim.apply("H", (q,))
        for i, q in enumerate(qubits[:n]):
            sim.apply("RZ", (q,), (2.0 * float(x[i]),))
        for i in range(n - 1):
            q1, q2 = qubits[i], qubits[i + 1]
            phi = 2.0 * float(x[i]) * float(x[i + 1])
            sim.apply("CX", (q1, q2))
            sim.apply("RZ", (q2,), (phi,))
            sim.apply("CX", (q1, q2))


def amplitude_encoding(
    sim: SimulatorBackend,
    x: np.ndarray,
) -> None:
    """Amplitude encoding: embed normalized feature vector into quantum state."""
    n_qubits = sim.num_qubits
    dim = 1 << n_qubits

    padded = np.zeros(dim, dtype=float)
    limit = min(len(x), dim)
    padded[:limit] = np.asarray(x[:limit], dtype=float)
    norm = float(np.linalg.norm(padded))
    if norm > 1e-12:
        padded = padded / norm
    else:
        padded[0] = 1.0

    sim.state = padded.astype(complex)


def _variational_layer(
    sim: SimulatorBackend,
    params: np.ndarray,
    n_qubits: int,
    offset: int = 0,
) -> int:
    """Apply one hardware-efficient variational layer (RY + RZ + CX chain)."""
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
    """Quantum kernel computing K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2."""

    def __init__(
        self,
        n_qubits: int = 4,
        feature_map: str = "zz",
        reps: int = 2,
    ) -> None:
        self.n_qubits = n_qubits
        self.feature_map = feature_map
        self.reps = reps

    def _encode(self, x: np.ndarray) -> np.ndarray:
        sim = create_simulator(self.n_qubits)
        if self.feature_map == "zz":
            zz_feature_map(sim, x, reps=self.reps)
        elif self.feature_map == "amplitude":
            amplitude_encoding(sim, x)
        else:
            angle_encoding(sim, x)
        return np.asarray(sim.state)

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        s1 = self._encode(x1)
        s2 = self._encode(x2)
        return float(abs(np.vdot(s1, s2)) ** 2)

    def matrix(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        K = np.zeros((n, n), dtype=float)
        states = [self._encode(X[i]) for i in range(n)]
        for i in range(n):
            for j in range(i, n):
                K[i, j] = float(abs(np.vdot(states[i], states[j])) ** 2)
                K[j, i] = K[i, j]
        return K


# ── QML Result Dataclasses ────────────────────────


@dataclass
class QMLResult:
    """Result of quantum machine learning classification training."""

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


@dataclass
class RegressionResult:
    """Result of quantum machine learning regression training."""

    r2: float
    mse: float
    loss_history: list[float] = field(default_factory=list)
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    n_qubits: int = 0
    n_params: int = 0

    def __repr__(self) -> str:
        return (
            f"RegressionResult(r2={self.r2:.4f}, mse={self.mse:.4f}, "
            f"qubits={self.n_qubits}, params={self.n_params})"
        )


def _parity_signs(n_qubits: int) -> np.ndarray:
    """Returns (+1/-1) eigenvalues of Z^{otimes n} for basis indices 0 .. 2^n - 1."""
    idx = np.arange(1 << n_qubits, dtype=np.uint32)
    # Bitwise popcount parity
    x = idx ^ (idx >> 16)
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    odd = (x & 1).astype(float)
    return 1.0 - 2.0 * odd


# ── Quantum Classifier ───────────────────────────


class QuantumClassifier:
    """Variational Quantum Classifier (VQC) with exact chain-rule parameter-shift gradients."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        feature_map: str = "angle",
        learning_rate: float = 0.1,
        seed: int | None = None,
        ansatz: str = "hardware_efficient",
        optimizer: str = "sgd",
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map = feature_map
        self.learning_rate = learning_rate
        self.seed = seed
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.rng = np.random.default_rng(seed)

        self._ansatz_preset = _get_ansatz_preset(ansatz)
        if optimizer not in ("sgd", "adam", "spsa", "cobyla"):
            raise ValueError(
                f"Unknown optimizer {optimizer!r}. Supported: 'sgd', 'adam', 'spsa', 'cobyla'"
            )

        self.n_params = self._ansatz_preset.param_count(n_qubits, n_layers)
        self.params = self.rng.uniform(-math.pi, math.pi, size=self.n_params)
        self.loss_history: list[float] = []
        self._parity_eig = _parity_signs(n_qubits)

    def _forward(self, x: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """Forward pass: feature map U_phi(x) + variational ansatz V(theta)."""
        theta = self.params if params is None else params
        sim = create_simulator(self.n_qubits)

        if self.feature_map == "zz":
            zz_feature_map(sim, x)
        elif self.feature_map == "amplitude":
            amplitude_encoding(sim, x)
        else:
            angle_encoding(sim, x)

        self._ansatz_preset.apply(sim, theta, self.n_qubits, self.n_layers)  # type: ignore[arg-type]
        return np.asarray(np.abs(sim.state) ** 2, dtype=float)

    def _expectation_parity_z(
        self, x: np.ndarray, params: np.ndarray | None = None
    ) -> float:
        """Linear Pauli parity expectation <Z^{otimes n}> = 1 - 2 * P(class=1)."""
        probs = self._forward(x, params=params)
        return float(np.dot(self._parity_eig, probs))

    def _predict_proba(self, x: np.ndarray, params: np.ndarray | None = None) -> float:
        """Probability of class 1: p_1 = (1 - <Z^{otimes n}>) / 2."""
        exp_z = self._expectation_parity_z(x, params=params)
        return float(np.clip(0.5 * (1.0 - exp_z), 0.0, 1.0))

    def _loss(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray | None = None
    ) -> float:
        """Binary cross-entropy loss."""
        eps = 1e-12
        total = 0.0
        for xi, yi in zip(X, y, strict=False):
            p = float(np.clip(self._predict_proba(xi, params=params), eps, 1.0 - eps))
            total += -(yi * math.log(p) + (1.0 - yi) * math.log(1.0 - p))
        return total / len(X)

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Exact chain-rule gradient: (dL / d<Z>) * (d<Z> / dtheta_j).

        Applies the parameter-shift rule strictly to the linear Pauli observable
        ``<Z^{otimes n}>(theta +/- pi/2 e_j)`` and multiplies by the exact derivative
        ``dL_m / d<Z>_m = (y_m - p_m) / (2 * p_m * (1 - p_m))``.
        """
        eps = 1e-12
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n_samples = len(X_arr)

        # dL_m / d<Z>_m evaluated at current theta
        dl_dz = np.empty(n_samples, dtype=float)
        for m in range(n_samples):
            p_m = float(np.clip(self._predict_proba(X_arr[m]), eps, 1.0 - eps))
            dl_dz[m] = (y_arr[m] - p_m) / (2.0 * p_m * (1.0 - p_m))

        gradients = np.zeros(self.n_params, dtype=float)
        shift = math.pi / 2.0

        for j in range(self.n_params):
            orig = self.params[j]

            self.params[j] = orig + shift
            z_plus = np.array(
                [self._expectation_parity_z(xi) for xi in X_arr], dtype=float
            )

            self.params[j] = orig - shift
            z_minus = np.array(
                [self._expectation_parity_z(xi) for xi in X_arr], dtype=float
            )

            self.params[j] = orig

            dz_dtheta = 0.5 * (z_plus - z_minus)
            gradients[j] = float(np.mean(dl_dz * dz_dtheta))

        return gradients

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
    ) -> QMLResult:
        """Train the quantum classifier with the configured optimizer."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if len(X_arr) == 0:
            raise ValueError("Training dataset cannot be empty.")

        self.loss_history = []

        if self.optimizer == "cobyla":
            def obj_fn(theta: np.ndarray) -> float:
                val = self._loss(X_arr, y_arr, params=theta)
                self.loss_history.append(val)
                return val

            opt_res = scipy_minimize(
                obj_fn,
                self.params.copy(),
                method="COBYLA",
                options={"maxiter": max(10, epochs * max(2, self.n_params // 2)), "rhobeg": 0.5},
            )
            self.params = np.asarray(opt_res.x, dtype=float)
        else:
            m_state = np.zeros(self.n_params, dtype=float)
            v_state = np.zeros(self.n_params, dtype=float)
            beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

            for t, epoch in enumerate(range(epochs), start=1):
                loss = self._loss(X_arr, y_arr)
                self.loss_history.append(loss)

                if self.optimizer == "spsa":
                    delta = self.rng.choice([-1.0, 1.0], size=self.n_params)
                    ck = 0.1 / ((epoch + 1) ** 0.101)
                    ak = self.learning_rate / ((epoch + 1 + 5) ** 0.602)
                    loss_plus = self._loss(X_arr, y_arr, params=self.params + ck * delta)
                    loss_minus = self._loss(X_arr, y_arr, params=self.params - ck * delta)
                    grad = (loss_plus - loss_minus) / (2.0 * ck * delta)
                    self.params -= ak * grad
                else:
                    gradients = self._compute_gradients(X_arr, y_arr)
                    if self.optimizer == "adam":
                        m_state = beta1 * m_state + (1.0 - beta1) * gradients
                        v_state = beta2 * v_state + (1.0 - beta2) * (gradients**2)
                        m_hat = m_state / (1.0 - beta1**t)
                        v_hat = v_state / (1.0 - beta2**t)
                        self.params -= (
                            self.learning_rate * m_hat / (np.sqrt(v_hat) + adam_eps)
                        )
                    else:
                        self.params -= self.learning_rate * gradients

                if epoch > 5 and abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-6:
                    break

        predictions = self.predict(X_arr)
        accuracy = float(np.mean(predictions == y_arr))

        return QMLResult(
            accuracy=accuracy,
            loss_history=self.loss_history,
            predictions=predictions,
            n_qubits=self.n_qubits,
            n_params=self.n_params,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return np.array([1 if self._predict_proba(xi) > 0.5 else 0 for xi in X_arr])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        probas = np.array([self._predict_proba(xi) for xi in X_arr], dtype=float)
        return np.column_stack([1.0 - probas, probas])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        return float(np.mean(predictions == np.asarray(y)))


# ── Quantum Regressor ─────────────────────────────


class QuantumRegressor:
    """Variational Quantum Regressor (VQR) reading MSB qubit q[0] <Z_0>."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        feature_map: str = "angle",
        learning_rate: float = 0.1,
        optimizer: str = "adam",
        seed: int | None = None,
        ansatz: str = "hardware_efficient",
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map = feature_map
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.seed = seed
        self.ansatz = ansatz
        self.rng = np.random.default_rng(seed)

        self._ansatz_preset = _get_ansatz_preset(ansatz)
        if optimizer not in ("sgd", "adam", "spsa", "cobyla"):
            raise ValueError(
                f"Unknown optimizer {optimizer!r}. Supported: 'sgd', 'adam', 'spsa', 'cobyla'"
            )

        self.n_params = self._ansatz_preset.param_count(n_qubits, n_layers)
        self.params = self.rng.uniform(-math.pi, math.pi, size=self.n_params)
        self.weight = 1.0
        self.bias = 0.0
        self.loss_history: list[float] = []

    def _expectation_z(
        self, x: np.ndarray, params: np.ndarray | None = None
    ) -> float:
        """Evaluate <Z_0> on MSB qubit q[0] (``(i >> (n - 1)) & 1 == 0``)."""
        theta = self.params if params is None else params
        sim = create_simulator(self.n_qubits)

        if self.feature_map == "zz":
            zz_feature_map(sim, x)
        elif self.feature_map == "amplitude":
            amplitude_encoding(sim, x)
        else:
            angle_encoding(sim, x)

        self._ansatz_preset.apply(sim, theta, self.n_qubits, self.n_layers)  # type: ignore[arg-type]

        probs = np.abs(sim.state) ** 2
        # MSB q[0] == 0 corresponds to basis indices [0, 2^(n-1)), q[0] == 1 to [2^(n-1), 2^n)
        half = 1 << (self.n_qubits - 1)
        return float(np.sum(probs[:half]) - np.sum(probs[half:]))

    def _predict_single(self, x: np.ndarray) -> float:
        return float(self.weight * self._expectation_z(x) + self.bias)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return np.array([self._predict_single(xi) for xi in X_arr], dtype=float)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean((preds - y) ** 2))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        preds = self.predict(X_arr)
        ss_res = float(np.sum((y_arr - preds) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        if ss_tot < 1e-12:
            return 1.0 if ss_res < 1e-12 else 0.0
        return float(1.0 - ss_res / ss_tot)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
    ) -> RegressionResult:
        """Train the quantum regressor using parameter-shift / SPSA / COBYLA."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if len(X_arr) == 0:
            raise ValueError("Training dataset cannot be empty.")

        y_mean = float(np.mean(y_arr))
        y_std = float(np.std(y_arr))
        self.bias = y_mean
        self.weight = y_std if y_std > 1e-4 else 1.0
        self.loss_history = []

        if self.optimizer == "cobyla":
            init_vec = np.concatenate([self.params, [self.weight, self.bias]])

            def cobyla_obj(vec: np.ndarray) -> float:
                th = vec[:-2]
                w = float(vec[-2])
                b = float(vec[-1])
                exp_v = np.array(
                    [self._expectation_z(xi, params=th) for xi in X_arr], dtype=float
                )
                val = float(np.mean((w * exp_v + b - y_arr) ** 2))
                self.loss_history.append(val)
                return val

            opt_res = scipy_minimize(
                cobyla_obj,
                init_vec,
                method="COBYLA",
                options={"maxiter": max(15, epochs * max(2, self.n_params // 2)), "rhobeg": 0.4},
            )
            self.params = np.asarray(opt_res.x[:-2], dtype=float)
            self.weight = float(opt_res.x[-2])
            self.bias = float(opt_res.x[-1])
        else:
            m_params = np.zeros(self.n_params, dtype=float)
            v_params = np.zeros(self.n_params, dtype=float)
            m_w, v_w = 0.0, 0.0
            m_b, v_b = 0.0, 0.0
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            shift = math.pi / 2.0

            for t, epoch in enumerate(range(epochs), start=1):
                exp_vals = np.array(
                    [self._expectation_z(xi) for xi in X_arr], dtype=float
                )
                preds = self.weight * exp_vals + self.bias
                residuals = preds - y_arr
                loss = float(np.mean(residuals**2))
                self.loss_history.append(loss)

                grad_b = float(2.0 * np.mean(residuals))
                grad_w = float(2.0 * np.mean(residuals * exp_vals))

                if self.optimizer == "spsa":
                    delta = self.rng.choice([-1.0, 1.0], size=self.n_params)
                    ck = 0.1 / ((epoch + 1) ** 0.101)
                    ak = self.learning_rate / ((epoch + 1 + 10) ** 0.602)

                    preds_plus = np.array(
                        [
                            self.weight
                            * self._expectation_z(xi, params=self.params + ck * delta)
                            + self.bias
                            for xi in X_arr
                        ],
                        dtype=float,
                    )
                    loss_plus = float(np.mean((preds_plus - y_arr) ** 2))

                    preds_minus = np.array(
                        [
                            self.weight
                            * self._expectation_z(xi, params=self.params - ck * delta)
                            + self.bias
                            for xi in X_arr
                        ],
                        dtype=float,
                    )
                    loss_minus = float(np.mean((preds_minus - y_arr) ** 2))

                    grad_spsa = (loss_plus - loss_minus) / (2.0 * ck * delta)
                    self.params -= ak * grad_spsa
                    self.weight -= ak * grad_w
                    self.bias -= ak * grad_b
                else:
                    grad_params = np.zeros(self.n_params, dtype=float)
                    for j in range(self.n_params):
                        orig = self.params[j]
                        self.params[j] = orig + shift
                        exp_plus = np.array(
                            [self._expectation_z(xi) for xi in X_arr], dtype=float
                        )
                        self.params[j] = orig - shift
                        exp_minus = np.array(
                            [self._expectation_z(xi) for xi in X_arr], dtype=float
                        )
                        self.params[j] = orig

                        d_exp = 0.5 * (exp_plus - exp_minus)
                        grad_params[j] = float(
                            np.mean(2.0 * residuals * self.weight * d_exp)
                        )

                    if self.optimizer == "adam":
                        m_params = beta1 * m_params + (1.0 - beta1) * grad_params
                        v_params = beta2 * v_params + (1.0 - beta2) * (grad_params**2)
                        m_hat = m_params / (1.0 - beta1**t)
                        v_hat = v_params / (1.0 - beta2**t)
                        self.params -= (
                            self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
                        )

                        m_w = beta1 * m_w + (1.0 - beta1) * grad_w
                        v_w = beta2 * v_w + (1.0 - beta2) * (grad_w**2)
                        mw_hat = m_w / (1.0 - beta1**t)
                        vw_hat = v_w / (1.0 - beta2**t)
                        self.weight -= (
                            self.learning_rate * mw_hat / (math.sqrt(vw_hat) + eps)
                        )

                        m_b = beta1 * m_b + (1.0 - beta1) * grad_b
                        v_b = beta2 * v_b + (1.0 - beta2) * (grad_b**2)
                        mb_hat = m_b / (1.0 - beta1**t)
                        vb_hat = v_b / (1.0 - beta2**t)
                        self.bias -= (
                            self.learning_rate * mb_hat / (math.sqrt(vb_hat) + eps)
                        )
                    else:
                        self.params -= self.learning_rate * grad_params
                        self.weight -= self.learning_rate * grad_w
                        self.bias -= self.learning_rate * grad_b

                if epoch > 5 and abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-7:
                    break

        final_preds = self.predict(X_arr)
        final_mse = float(np.mean((final_preds - y_arr) ** 2))
        final_r2 = self.score(X_arr, y_arr)

        return RegressionResult(
            r2=final_r2,
            mse=final_mse,
            loss_history=self.loss_history,
            predictions=final_preds,
            n_qubits=self.n_qubits,
            n_params=self.n_params,
        )
