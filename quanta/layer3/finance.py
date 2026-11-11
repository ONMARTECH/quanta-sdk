"""
quanta.layer3.finance -- Variational Ising QAOA Portfolio Optimization.

Maps the Markowitz mean-variance portfolio problem with cardinality budget B:
    min_x  -sum_i r_i x_i + risk_aversion * x^T Sigma x + penalty * (sum_i x_i - B)^2
via x_i = (1 - Z_i) / 2 into an exact 2-body Ising Hamiltonian:
    H_C = E_0 I + sum_i h_i Z_i + sum_{i < j} J_{ij} Z_i Z_j,
compiles a multi-layer gate-based QAOA circuit (`H`, `RZ`, `CX-RZ-CX`, `RX`),
optimizes variational parameters `(gamma, beta)` with `scipy.optimize.minimize(method="COBYLA")`,
and selects the portfolio from real `sim.sample(shots)` measurement outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from quanta.core.circuit import CircuitBuilder
from quanta.core.types import Instruction
from quanta.dag.dag_circuit import DAGCircuit
from quanta.simulator.factory import create_simulator

__all__ = ["portfolio_optimize", "PortfolioResult"]


@dataclass
class Asset:
    """Financial asset definition."""

    name: str
    expected_return: float
    risk: float


@dataclass
class PortfolioResult:
    """Result of quantum portfolio optimization."""

    selected: list[str]
    selection_bits: str
    expected_return: float
    portfolio_risk: float
    sharpe_ratio: float
    all_portfolios: list[dict]

    @property
    def bitstring(self) -> str:
        """Alias for selection_bits."""
        return self.selection_bits

    def summary(self) -> str:
        """Pretty portfolio summary."""
        lines = [
            "╔══════════════════════════════════════╗",
            "║  Quantum Portfolio Optimization      ║",
            "╠══════════════════════════════════════╣",
            f"║  Selected: {', '.join(self.selected):<25}║",
            f"║  Expected Return: {self.expected_return:>8.1%}          ║",
            f"║  Portfolio Risk:  {self.portfolio_risk:>8.1%}          ║",
            f"║  Sharpe Ratio:    {self.sharpe_ratio:>8.2f}          ║",
            "╠──────────────────────────────────────╣",
        ]
        for p in self.all_portfolios[:5]:
            bits = p["bits"]
            ret = p["return"]
            risk = p["risk"]
            obj = p["objective"]
            marker = " ◀ BEST" if bits == self.selection_bits else ""
            lines.append(
                f"║  {bits}  ret={ret:+.1%}  risk={risk:.1%}  "
                f"obj={obj:+.3f}{marker}"
            )
        lines.append("╚══════════════════════════════════════╝")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PortfolioResult(selected={self.selected}, "
            f"return={self.expected_return:.1%}, "
            f"risk={self.portfolio_risk:.1%})"
        )


def _build_correlation_matrix(
    assets: list[Asset],
    correlations: np.ndarray | None = None,
) -> np.ndarray:
    """Builds covariance matrix Sigma from asset risks and optional correlation matrix."""
    n = len(assets)
    if correlations is not None:
        corr_arr = np.asarray(correlations, dtype=float)
        cov = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                cov[i, j] = corr_arr[i, j] * assets[i].risk * assets[j].risk
        return cov

    return np.diag([a.risk**2 for a in assets])


def _qubo_to_ising(
    returns: np.ndarray,
    cov: np.ndarray,
    budget: int,
    risk_aversion: float,
    penalty: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Maps Markowitz QUBO E(x) = x^T Q x + c^T x + const via x_i = (1 - Z_i)/2 to Ising (E0, h, J).

    Minimizing E(x) maximizes ``return - risk_aversion * variance - penalty * (sum(x) - budget)^2``.
    """
    n = len(returns)
    # E(x) = sum_i c_i x_i + sum_{i,j} Q_{ij} x_i x_j + penalty * budget^2
    # where (sum_i x_i - B)^2 = sum_i x_i^2 + 2 sum_{i<j} x_i x_j - 2 B sum_i x_i + B^2
    # Since x_i in {0, 1}, x_i^2 = x_i:
    # Linear coeff L_i = -r_i + risk_aversion * cov[i, i] + penalty * (1 - 2 * B)
    # Pairwise coeff M_{ij} (for i < j) = 2 * risk_aversion * cov[i, j] + 2 * penalty
    lin = -returns + risk_aversion * np.diag(cov) + penalty * (1.0 - 2.0 * budget)
    pair = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            pair[i, j] = 2.0 * risk_aversion * cov[i, j] + 2.0 * penalty

    # Substitute x_i = (1 - Z_i)/2:
    # L_i x_i = L_i/2 - (L_i/2) Z_i
    # M_{ij} x_i x_j = M_{ij}/4 - (M_{ij}/4) Z_i - (M_{ij}/4) Z_j + (M_{ij}/4) Z_i Z_j
    e0 = float(penalty * (budget**2) + 0.5 * np.sum(lin) + 0.25 * np.sum(pair))
    h = -0.5 * lin.copy()
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            w = 0.25 * pair[i, j]
            h[i] -= w
            h[j] -= w
            J[i, j] = w

    return e0, h, J


def _build_ising_qaoa_dag(
    n: int, h: np.ndarray, J: np.ndarray, params: np.ndarray
) -> DAGCircuit:
    """Constructs a p-layer gate-based Ising QAOA DAGCircuit (`H`, `RZ`, `CX-RZ-CX`, `RX`)."""
    builder = CircuitBuilder(n)
    layers = len(params) // 2

    for q in range(n):
        builder.record(Instruction("H", (q,)))

    for p in range(layers):
        gamma = float(params[2 * p])
        beta = float(params[2 * p + 1])

        for i in range(n):
            angle_z = 2.0 * gamma * float(h[i])
            if abs(angle_z) > 1e-15:
                builder.record(Instruction("RZ", (i,), (angle_z,)))

        for i in range(n):
            for j in range(i + 1, n):
                angle_zz = 2.0 * gamma * float(J[i, j])
                if abs(angle_zz) > 1e-15:
                    builder.record(Instruction("CX", (i, j)))
                    builder.record(Instruction("RZ", (j,), (angle_zz,)))
                    builder.record(Instruction("CX", (i, j)))

        for i in range(n):
            builder.record(Instruction("RX", (i,), (2.0 * beta,)))

    return DAGCircuit.from_builder(builder)


def portfolio_optimize(
    assets: list[dict],
    budget: int = 2,
    risk_aversion: float = 0.5,
    penalty: float = 2.0,
    shots: int = 2048,
    seed: int | None = None,
    correlations: np.ndarray | None = None,
    layers: int = 3,
) -> PortfolioResult:
    """Quantum portfolio optimization via variationally trained gate-based Ising QAOA."""
    n = len(assets)
    if n == 0:
        raise ValueError("Assets list cannot be empty.")
    if n > 15:
        raise ValueError(f"Max 15 assets supported, given: {n}")

    asset_objs = [
        Asset(
            a["name"],
            float(a.get("return", a.get("expected_return", 0.0))),
            float(a["risk"] if "risk" in a else a["volatility"]),
        )
        for a in assets
    ]

    returns = np.array([a.expected_return for a in asset_objs], dtype=float)
    cov = _build_correlation_matrix(asset_objs, correlations)

    _, h_ising, j_ising = _qubo_to_ising(returns, cov, budget, risk_aversion, penalty)

    dim = 1 << n
    energies = np.empty(dim, dtype=float)
    portfolio_meta: list[dict] = []

    for idx in range(dim):
        bits = format(idx, f"0{n}b")
        selection = np.array([int(b) for b in bits], dtype=float)
        port_return = float(selection @ returns)
        port_variance = float(selection @ cov @ selection)
        port_risk = float(np.sqrt(max(port_variance, 0.0)))
        budget_violation = (float(np.sum(selection)) - budget) ** 2
        objective = (
            port_return - risk_aversion * port_variance - penalty * budget_violation
        )
        energies[idx] = -objective
        portfolio_meta.append(
            {
                "bits": bits,
                "selection": selection,
                "return": port_return,
                "risk": port_risk,
                "variance": port_variance,
                "objective": objective,
                "budget_ok": int(round(float(np.sum(selection)))) == budget,
                "probability": 0.0,
            }
        )

    rng = np.random.default_rng(seed)
    scale = max(
        float(np.max(np.abs(h_ising))) if len(h_ising) else 1.0,
        float(np.max(np.abs(j_ising))) if j_ising.size else 1.0,
        1e-9,
    )
    h_scaled = h_ising / scale
    j_scaled = j_ising / scale

    def qaoa_expectation(params: np.ndarray) -> float:
        dag_eval = _build_ising_qaoa_dag(n, h_scaled, j_scaled, params)
        sim_eval = create_simulator(n)
        for op in dag_eval.op_nodes():
            sim_eval.apply(op.gate_name, op.qubits, op.params)
        return float(np.dot(sim_eval.probabilities(), energies))

    best_params = np.zeros(2 * layers, dtype=float)
    best_exp = float("inf")
    for _ in range(4):
        x0 = rng.uniform(0.05, 0.9 * np.pi, size=2 * layers)
        opt_res = scipy_minimize(
            qaoa_expectation,
            x0,
            method="COBYLA",
            options={"maxiter": 120, "rhobeg": 0.35},
        )
        if opt_res.fun < best_exp:
            best_exp = float(opt_res.fun)
            best_params = np.asarray(opt_res.x, dtype=float)

    unscaled_params = best_params.copy()
    for p in range(layers):
        unscaled_params[2 * p] /= scale

    dag_opt = _build_ising_qaoa_dag(n, h_ising, j_ising, unscaled_params)
    sim_opt = create_simulator(n, seed=seed)
    for op in dag_opt.op_nodes():
        sim_opt.apply(op.gate_name, op.qubits, op.params)

    counts = sim_opt.sample(shots)
    for entry in portfolio_meta:
        entry["probability"] = counts.get(entry["bits"], 0) / float(shots)

    sampled = [p for p in portfolio_meta if counts.get(p["bits"], 0) > 0]
    sampled.sort(key=lambda p: (-p["objective"], -p["probability"]))
    best = sampled[0] if sampled else max(portfolio_meta, key=lambda p: p["objective"])

    all_portfolios = sorted(portfolio_meta, key=lambda p: -p["objective"])
    selected_names = [
        asset_objs[j].name for j, b in enumerate(best["bits"]) if b == "1"
    ]
    sharpe = best["return"] / best["risk"] if best["risk"] > 0 else 0.0

    res = PortfolioResult(
        selected=selected_names,
        selection_bits=best["bits"],
        expected_return=best["return"],
        portfolio_risk=best["risk"],
        sharpe_ratio=sharpe,
        all_portfolios=all_portfolios,
    )
    res.counts = counts  # type: ignore[attr-defined]
    res.gate_count = dag_opt.gate_count()  # type: ignore[attr-defined]
    res.optimal_params = unscaled_params  # type: ignore[attr-defined]
    return res
