"""
quanta.layer3.qsvm -- Genuine Dual Soft-Margin Quantum Support Vector Machine.

Computes the quantum fidelity Gram matrix K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2
and solves the exact Cortes-Vapnik Dual Soft-Margin Quadratic Program:
    min_alpha  (1/2) alpha^T (y y^T odot K) alpha - 1^T alpha
    s.t.       0 <= alpha_i <= C,   y^T alpha = 0
via ``scipy.optimize.minimize(..., method="SLSQP")``, extracting genuine support
vectors (``alpha_i > 1e-5``) and the exact KKT complementary-slackness intercept ``b``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from quanta.simulator.base import SimulatorBackend
from quanta.simulator.factory import create_simulator

__all__ = ["qsvm_classify", "QSVMResult"]


@dataclass
class QSVMResult:
    """Result of Dual Soft-Margin QSVM classification."""

    predictions: list[int]
    kernel_matrix: np.ndarray
    accuracy: float
    num_qubits: int
    alphas: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    support_indices: list[int] = field(default_factory=list)
    intercept: float = 0.0

    def summary(self) -> str:
        lines = [
            "=== QSVM Classification ===",
            f"  Qubits:          {self.num_qubits}",
            f"  Accuracy:        {self.accuracy:.1%}",
            f"  Support vectors: {len(self.support_indices)}",
            f"  Intercept (b):   {self.intercept:+.4f}",
            f"  Predictions:     {self.predictions}",
            f"  Kernel size:     {self.kernel_matrix.shape[0]}x{self.kernel_matrix.shape[1]}",
        ]
        return "\n".join(lines)


def _encode_data_point(
    sim: SimulatorBackend,
    x: list[float] | np.ndarray,
    num_qubits: int,
) -> None:
    """Encodes a feature vector into a quantum state via depth-2 ZZFeatureMap."""
    n_features = len(x)

    for q in range(num_qubits):
        sim.apply("H", (q,))

    for q in range(min(n_features, num_qubits)):
        sim.apply("RZ", (q,), (float(x[q]) * np.pi,))

    for q in range(min(n_features - 1, num_qubits - 1)):
        sim.apply("CX", (q, q + 1))
        interaction = float(x[q]) * float(x[q + 1]) * np.pi
        sim.apply("RZ", (q + 1,), (interaction,))
        sim.apply("CX", (q, q + 1))

    for q in range(num_qubits):
        sim.apply("H", (q,))

    for q in range(min(n_features, num_qubits)):
        sim.apply("RZ", (q,), (float(x[q]) * np.pi,))


def _encode_state(x: list[float] | np.ndarray, num_qubits: int) -> np.ndarray:
    sim = create_simulator(num_qubits)
    _encode_data_point(sim, x, num_qubits)
    return sim.state


def _quantum_kernel(
    x1: list[float],
    x2: list[float],
    num_qubits: int,
) -> float:
    """Computes quantum kernel fidelity K(x1, x2) = |<phi(x1)|phi(x2)>|^2."""
    state1 = _encode_state(x1, num_qubits)
    state2 = _encode_state(x2, num_qubits)
    return float(np.abs(np.vdot(state1, state2)) ** 2)


def _build_kernel_matrix(
    X: list[list[float]],
    num_qubits: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Builds the symmetric PSD quantum kernel Gram matrix and encoded state cache."""
    n = len(X)
    states = [_encode_state(X[i], num_qubits) for i in range(n)]
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        K[i, i] = 1.0
        for j in range(i + 1, n):
            k_ij = float(np.abs(np.vdot(states[i], states[j])) ** 2)
            K[i, j] = k_ij
            K[j, i] = k_ij
    return K, states


def _solve_dual_svm_qp(
    K: np.ndarray, labels: np.ndarray, C: float, sv_tol: float = 1e-5
) -> tuple[np.ndarray, list[int], float]:
    """Solves the exact Dual Soft-Margin SVM QP via SLSQP and extracts KKT bias ``b``."""
    n = len(labels)
    if C <= 0:
        raise ValueError(f"regularization C must be > 0, got {C}")

    # Single-class edge case
    unique_y = np.unique(labels)
    if len(unique_y) == 1:
        return np.zeros(n, dtype=float), [], float(unique_y[0])

    Q = np.outer(labels, labels) * K
    # Symmetrize + tiny Tikhonov floor for numerical PSD stability
    Q = 0.5 * (Q + Q.T) + 1e-10 * np.eye(n)
    ones = np.ones(n, dtype=float)

    def dual_obj(alpha: np.ndarray) -> float:
        return float(0.5 * (alpha @ (Q @ alpha)) - np.dot(ones, alpha))

    def dual_grad(alpha: np.ndarray) -> np.ndarray:
        return Q @ alpha - ones

    bounds = [(0.0, float(C)) for _ in range(n)]
    constraints = [
        {
            "type": "eq",
            "fun": lambda alpha: float(np.dot(labels, alpha)),
            "jac": lambda alpha: labels,
        }
    ]

    # Feasible balanced initial point satisfying y^T alpha_0 == 0 and 0 <= alpha_0 <= C
    pos_mask = labels > 0
    neg_mask = labels < 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    alpha0 = np.zeros(n, dtype=float)
    init_mass = 0.25 * float(C) * min(n_pos, n_neg)
    alpha0[pos_mask] = init_mass / n_pos
    alpha0[neg_mask] = init_mass / n_neg

    res = scipy_minimize(
        dual_obj,
        alpha0,
        method="SLSQP",
        jac=dual_grad,
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-11, "maxiter": 600},
    )

    alphas = np.clip(np.asarray(res.x, dtype=float), 0.0, float(C))
    # Enforce exact equality y^T alphas == 0 up to machine precision
    residual_eq = float(np.dot(labels, alphas))
    if abs(residual_eq) > 1e-9:
        free_mask = (alphas > sv_tol) & (alphas < C - sv_tol)
        adjust_idx = np.where(free_mask)[0]
        if len(adjust_idx) == 0:
            adjust_idx = np.where(alphas > 0)[0]
        if len(adjust_idx) > 0:
            step = residual_eq / len(adjust_idx)
            for idx in adjust_idx:
                alphas[idx] = float(
                    np.clip(alphas[idx] - labels[idx] * step, 0.0, float(C))
                )

    support_indices = [i for i in range(n) if alphas[i] > sv_tol]
    dual_weights = alphas * labels
    g_train = K @ dual_weights  # Margin scores without bias

    # KKT complementary slackness: for 0 < alpha_i < C, y_i (g_i + b) == 1 => b = y_i - g_i
    free_sv = [i for i in range(n) if sv_tol < alphas[i] < C - sv_tol]
    if free_sv:
        b = float(np.mean([labels[i] - g_train[i] for i in free_sv]))
    elif support_indices:
        # Bounded SVs: compute KKT interval [b_low, b_high]
        b_low_cands = [
            labels[i] - g_train[i]
            for i in range(n)
            if (labels[i] > 0 and alphas[i] < C - sv_tol)
            or (labels[i] < 0 and alphas[i] > sv_tol)
        ]
        b_high_cands = [
            labels[i] - g_train[i]
            for i in range(n)
            if (labels[i] > 0 and alphas[i] > sv_tol)
            or (labels[i] < 0 and alphas[i] < C - sv_tol)
        ]
        if b_low_cands and b_high_cands:
            b = 0.5 * (max(b_low_cands) + min(b_high_cands))
        else:
            b = float(np.mean([labels[i] - g_train[i] for i in support_indices]))
    else:
        b = 0.0

    return alphas, support_indices, b


def qsvm_classify(
    X_train: list[list[float]],
    y_train: list[int],
    X_test: list[list[float]],
    num_qubits: int | None = None,
    regularization: float = 1.0,
) -> QSVMResult:
    """Quantum SVM binary classification via Dual Soft-Margin Quadratic Programming."""
    if not X_train:
        raise ValueError("X_train cannot be empty.")

    n_features = len(X_train[0])

    if num_qubits is None:
        num_qubits = max(2, n_features)
    num_qubits = min(num_qubits, 10)

    labels = np.array([1.0 if y == 1 else -1.0 for y in y_train], dtype=float)
    K_train, train_states = _build_kernel_matrix(X_train, num_qubits)

    alphas, support_indices, b = _solve_dual_svm_qp(
        K_train, labels, C=float(regularization)
    )
    dual_weights = alphas * labels

    train_scores = K_train @ dual_weights + b
    train_pred = np.where(train_scores >= 0.0, 1.0, -1.0)
    accuracy = float(np.mean(train_pred == labels))

    predictions: list[int] = []
    for x_test in X_test:
        test_state = _encode_state(x_test, num_qubits)
        score = b
        for sv_idx in support_indices:
            k_val = float(np.abs(np.vdot(train_states[sv_idx], test_state)) ** 2)
            score += dual_weights[sv_idx] * k_val
        predictions.append(1 if score >= 0.0 else 0)

    return QSVMResult(
        predictions=predictions,
        kernel_matrix=K_train,
        accuracy=accuracy,
        num_qubits=num_qubits,
        alphas=alphas,
        support_indices=support_indices,
        intercept=b,
    )
