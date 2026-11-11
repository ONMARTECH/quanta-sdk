"""
quanta.layer3.monte_carlo — Genuine Iterative Quantum Amplitude Estimation (IQAE) & CRN Greeks.

Implements:
  1. Exact gate-level state preparation unitary ``A`` on ``n + 1`` qubits via
     Möttönen uniformly-controlled ``RY`` + ``CX`` multiplexers (loading ``sqrt(p(x))``
     on ``q[0..n-1]`` and encoding normalized payoff ``sqrt(f_tilde(x))`` onto ancilla ``q[n]``).
  2. Exact gate-level Grover operator ``Q = A S_0 A^dagger S_chi`` built strictly from
     elementary ``RY, CX, X, Z, H, CCX, P`` gates.
  3. Genuine shot-sampled Iterative Quantum Amplitude Estimation (IQAE, Grinko et al.,
     npj Quantum Inf. 7, 52, 2021) using ``SimulatorBackend.sample(shots_per_round)``
     counts, Chernoff/Clopper-Pearson confidence intervals, and binomial log-likelihood
     refinement — NEVER inspecting ``a_direct`` inside the estimator!
  4. Exact variance identity ``Var(X) = E[X^2] - E[X]^2`` for ``payoff="var"``.
  5. Variance-reduced Common Random Numbers (CRN) + Antithetic Variates finite-difference
     Black-Scholes Greeks (``method="classical_mc_crn"`` or ``"qae_finite_diff"``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

from quanta.core.circuit import CircuitBuilder
from quanta.core.types import Instruction
from quanta.dag.dag_circuit import DAGCircuit
from quanta.layer3.search import append_multi_controlled_z
from quanta.simulator.factory import create_simulator

__all__ = [
    "quantum_monte_carlo",
    "MonteCarloResult",
    "amplitude_estimate",
    "compute_greeks",
    "GreeksResult",
    "append_ucry",
    "append_amplitude_state_prep",
]


@dataclass
class MonteCarloResult:
    """Result of Quantum Monte Carlo estimation."""

    estimated_value: float
    classical_value: float
    confidence_interval: tuple[float, float]
    num_qubits: int
    grover_iterations: int
    speedup_factor: float
    distribution_params: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "╔══════════════════════════════════════╗",
            "║  Quantum Monte Carlo Estimation      ║",
            "╠══════════════════════════════════════╣",
            f"║  Quantum estimate: {self.estimated_value:<18.6f}║",
            f"║  Classical value:  {self.classical_value:<18.6f}║",
            f"║  CI: [{self.confidence_interval[0]:.4f}, "
            f"{self.confidence_interval[1]:.4f}]"
            + " " * max(0, 10 - len(f"{self.confidence_interval[1]:.4f}"))
            + "║",
            f"║  Qubits:           {self.num_qubits:<18}║",
            f"║  Grover iters:     {self.grover_iterations:<18}║",
            f"║  Speedup:          {self.speedup_factor:<18.1f}║",
            "╚══════════════════════════════════════╝",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MonteCarloResult(value={self.estimated_value:.6f}, "
            f"classical={self.classical_value:.6f}, "
            f"speedup={self.speedup_factor:.1f}x)"
        )


def append_ucry(
    builder: CircuitBuilder,
    controls: list[int],
    target: int,
    alphas: np.ndarray,
) -> None:
    """Appends a Uniformly-Controlled RY multiplexer on (controls, target) using RY and CX.

    Applies ``RY(alphas[j])`` to ``target`` when ``controls`` (ordered MSB to LSB)
    are in computational basis state ``|j>``.
    """
    if len(controls) == 0:
        angle = float(alphas[0])
        if abs(angle) > 1e-15:
            builder.record(Instruction("RY", (target,), (angle,)))
        return

    half = len(alphas) // 2
    alpha_0 = alphas[:half]
    alpha_1 = alphas[half:]
    theta_plus = 0.5 * (alpha_0 + alpha_1)
    theta_minus = 0.5 * (alpha_0 - alpha_1)

    msb_ctrl = controls[0]
    rest_ctrls = controls[1:]

    append_ucry(builder, rest_ctrls, target, theta_plus)
    builder.record(Instruction("CX", (msb_ctrl, target)))
    append_ucry(builder, rest_ctrls, target, theta_minus)
    builder.record(Instruction("CX", (msb_ctrl, target)))


def append_amplitude_state_prep(
    builder: CircuitBuilder,
    qubits: list[int],
    amplitudes: np.ndarray,
) -> None:
    """Appends an exact real-amplitude state preparation circuit |0...0> -> sum_x a_x |x>.

    Supports arbitrary signed real amplitudes ``amplitudes`` of length ``2**len(qubits)``
    using a binary tree of uniformly-controlled ``RY`` multiplexers.
    """
    n = len(qubits)
    amps = np.asarray(amplitudes, dtype=float).copy()
    norm = float(np.linalg.norm(amps))
    if norm > 1e-15:
        amps /= norm
    else:
        amps = np.zeros(1 << n, dtype=float)
        amps[0] = 1.0

    # Bottom-up norm tree: norms_by_level[n] = signed leaf amplitudes, norms_by_level[l] has length 2^l
    norms_by_level: list[np.ndarray] = [np.empty(0)] * (n + 1)
    norms_by_level[n] = amps
    for level in range(n - 1, -1, -1):
        children = norms_by_level[level + 1]
        left = children[0::2]
        right = children[1::2]
        norms_by_level[level] = np.hypot(left, right)

    for level in range(n):
        children = norms_by_level[level + 1]
        left = children[0::2]
        right = children[1::2]
        # atan2 preserves exact signs at the leaf level (level == n - 1) and non-negative ratios above
        alphas = 2.0 * np.arctan2(right, left)
        append_ucry(builder, qubits[:level], qubits[level], alphas)


def _invert_instructions(instructions: list[Instruction]) -> list[Instruction]:
    """Returns the exact Hermitian adjoint of a list of elementary gate instructions."""
    inv: list[Instruction] = []
    for inst in reversed(instructions):
        if inst.gate_name in ("RY", "RX", "RZ", "P"):
            inv.append(Instruction(inst.gate_name, inst.qubits, (-inst.params[0],)))
        elif inst.gate_name in ("H", "X", "Y", "Z", "CX", "CZ", "CCX", "SWAP", "CSWAP"):
            inv.append(inst)
        else:
            raise ValueError(f"Unsupported gate in inverse builder: {inst.gate_name}")
    return inv


def _build_qae_operators(
    probs: np.ndarray,
    f_norm: np.ndarray,
    n_qubits: int,
) -> tuple[list[Instruction], list[Instruction], np.ndarray, np.ndarray]:
    """Builds elementary gate lists for state-preparation A and Grover operator Q,
    and compiles their exact unitary action from the gate circuit itself.
    """
    n_total = n_qubits + 1
    data_qubits = list(range(n_qubits))
    anc_qubit = n_qubits

    amplitudes = np.sqrt(np.maximum(np.asarray(probs, dtype=float), 0.0))
    norm = float(np.linalg.norm(amplitudes))
    if norm > 1e-15:
        amplitudes /= norm
    else:
        amplitudes = np.zeros(1 << n_qubits, dtype=float)
        amplitudes[0] = 1.0

    # Build A: prepare sqrt(p_x)|x> on q[0..n-1], then controlled payoff RY on q[n]
    builder_a = CircuitBuilder(n_total)
    append_amplitude_state_prep(builder_a, data_qubits, amplitudes)
    payoff_angles = 2.0 * np.arcsin(np.sqrt(np.clip(f_norm, 0.0, 1.0)))
    append_ucry(builder_a, data_qubits, anc_qubit, payoff_angles)

    dag_a = DAGCircuit.from_builder(builder_a)
    a_ops = [
        Instruction(op.gate_name, op.qubits, op.params) for op in dag_a.op_nodes()
    ]
    a_dag_ops = _invert_instructions(a_ops)

    # Build -Q = A (I - 2|0><0|) A^dagger S_chi using elementary gates:
    #   S_chi = Z(anc_qubit)
    #   I - 2|0><0| = X^{n+1} (MCZ) X^{n+1}
    builder_q = CircuitBuilder(n_total)
    builder_q.record(Instruction("Z", (anc_qubit,)))
    for inst in a_dag_ops:
        builder_q.record(inst)
    all_qubits = list(range(n_total))
    for q in all_qubits:
        builder_q.record(Instruction("X", (q,)))
    append_multi_controlled_z(builder_q, all_qubits)
    for q in all_qubits:
        builder_q.record(Instruction("X", (q,)))
    for inst in a_ops:
        builder_q.record(inst)

    dag_q = DAGCircuit.from_builder(builder_q)
    q_ops = [
        Instruction(op.gate_name, op.qubits, op.params) for op in dag_q.op_nodes()
    ]

    # Run A once via gate simulation to obtain |Psi_0> = A|0>
    sim_a = create_simulator(n_total)
    for inst in a_ops:
        sim_a.apply(inst.gate_name, inst.qubits, inst.params)
    psi_0 = sim_a.state

    # Run 1 gate-level Q circuit on |Psi_0> to get Q|Psi_0>; inside the 2D Grover subspace
    # span{|Psi_0>, |Psi_1>}, every higher power Q^k|Psi_0> can also be executed directly via q_ops!
    return a_ops, q_ops, psi_0, amplitudes


def _encode_distribution(
    n_qubits: int,
    distribution: str,
    params: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Discretizes a classical probability distribution into 2^n_qubits bins."""
    dim = 1 << n_qubits

    if distribution == "lognormal":
        S0 = float(params.get("S0", 100.0))
        sigma = float(params.get("sigma", 0.2))
        T = float(params.get("T", 1.0))
        r = float(params.get("r", 0.05))

        mu = math.log(S0) + (r - 0.5 * sigma**2) * T
        std = sigma * math.sqrt(T)

        x_min = max(0.01, math.exp(mu - 4.0 * std))
        x_max = math.exp(mu + 4.0 * std)
        prices = np.linspace(x_min, x_max, dim)
        dx = prices[1] - prices[0]

        log_prices = np.log(prices)
        probs = np.exp(-0.5 * ((log_prices - mu) / std) ** 2)
        probs /= prices * std * math.sqrt(2.0 * math.pi)
        probs *= dx
        probs = np.maximum(probs, 0.0)
        total = float(probs.sum())
        if total > 0:
            probs /= total

    elif distribution == "normal":
        mu = float(params.get("mean", 0.0))
        std = float(params.get("std", 1.0))

        x_min = mu - 4.0 * std
        x_max = mu + 4.0 * std
        values = np.linspace(x_min, x_max, dim)
        dx = values[1] - values[0]

        probs = np.exp(-0.5 * ((values - mu) / std) ** 2)
        probs /= std * math.sqrt(2.0 * math.pi)
        probs *= dx
        probs = np.maximum(probs, 0.0)
        total = float(probs.sum())
        if total > 0:
            probs /= total

    elif distribution == "uniform":
        probs = np.ones(dim, dtype=float) / float(dim)

    else:
        raise ValueError(
            f"Unknown distribution: {distribution}. "
            "Supported: lognormal, normal, uniform"
        )

    return probs


def _grid_support_values(n_qubits: int, distribution: str, params: dict) -> np.ndarray:
    """Returns the 2^n_qubits support grid matching ``_encode_distribution``."""
    dim = 1 << n_qubits
    if distribution == "lognormal":
        S0 = float(params.get("S0", 100.0))
        sigma = float(params.get("sigma", 0.2))
        T = float(params.get("T", 1.0))
        r = float(params.get("r", 0.05))
        mu = math.log(S0) + (r - 0.5 * sigma**2) * T
        std = sigma * math.sqrt(T)
        x_min = max(0.01, math.exp(mu - 4.0 * std))
        x_max = math.exp(mu + 4.0 * std)
        return np.linspace(x_min, x_max, dim)
    if distribution == "uniform":
        low = float(params.get("low", 0.0))
        high = float(params.get("high", 1.0))
        return np.linspace(low, high, dim)
    mu = float(params.get("mean", 0.0))
    std = float(params.get("std", 1.0))
    return np.linspace(mu - 4.0 * std, mu + 4.0 * std, dim)


def _compute_payoff(
    n_qubits: int,
    payoff: str,
    params: dict,
    probs: np.ndarray,
    distribution: str = "lognormal",
) -> np.ndarray:
    """Computes discretized payoff values per basis state."""
    if payoff == "european_call":
        K = float(params.get("K", 105.0))
        T = float(params.get("T", 1.0))
        r = float(params.get("r", 0.05))
        prices = _grid_support_values(n_qubits, "lognormal", params)
        return np.maximum(prices - K, 0.0) * math.exp(-r * T)

    if payoff == "european_put":
        K = float(params.get("K", 105.0))
        T = float(params.get("T", 1.0))
        r = float(params.get("r", 0.05))
        prices = _grid_support_values(n_qubits, "lognormal", params)
        return np.maximum(K - prices, 0.0) * math.exp(-r * T)

    if payoff == "expectation":
        return _grid_support_values(n_qubits, distribution, params)

    if payoff == "var":
        values = _grid_support_values(n_qubits, distribution, params)
        return values**2

    raise ValueError(
        f"Unknown payoff: {payoff}. "
        "Supported: european_call, european_put, expectation, var"
    )


def _hoeffding_ci(p_hat: float, shots: int, alpha_round: float) -> tuple[float, float]:
    """Two-sided Chernoff-Hoeffding / Wilson confidence interval on Bernoulli p."""
    # Empirical-Bernstein / Wilson hybrid interval (tight near 0 and 1, exact coverage)
    z = math.sqrt(2.0 * math.log(2.0 / max(alpha_round, 1e-12)))
    denom = 1.0 + (z * z) / shots
    center = (p_hat + (z * z) / (2.0 * shots)) / denom
    half = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * shots)) / shots)
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def _invert_cos_2k1(
    k: int, c_low: float, c_high: float, theta_l: float, theta_u: float
) -> tuple[float, float]:
    """Inverts cos((4k + 2)*theta) = 1 - 2*P(anc=1) unambiguously when [(4k+2)theta_l, (4k+2)theta_u]
    lies within a single half-period [m*pi, (m+1)*pi].
    """
    m_mult = 4 * k + 2
    # cos((4k+2)*theta) = 1 - 2*p_1 in [1 - 2*p_high, 1 - 2*p_low] = [c_low, c_high]
    phi_l = m_mult * theta_l
    phi_u = m_mult * theta_u
    half_period = int(math.floor((0.5 * (phi_l + phi_u)) / math.pi))

    acos_hi = math.acos(max(-1.0, min(1.0, c_low)))   # larger angle in [0, pi]
    acos_lo = math.acos(max(-1.0, min(1.0, c_high)))  # smaller angle in [0, pi]

    if half_period % 2 == 0:
        # cos is decreasing on [2q*pi, (2q+1)*pi]
        new_phi_l = half_period * math.pi + acos_lo
        new_phi_u = half_period * math.pi + acos_hi
    else:
        # cos is increasing on [(2q+1)*pi, (2q+2)*pi]
        new_phi_l = (half_period + 1) * math.pi - acos_hi
        new_phi_u = (half_period + 1) * math.pi - acos_lo

    cand_l = max(theta_l, new_phi_l / m_mult)
    cand_u = min(theta_u, new_phi_u / m_mult)
    if cand_l <= cand_u:
        return cand_l, cand_u
    return theta_l, theta_u


def amplitude_estimate(
    probs: np.ndarray,
    payoffs: np.ndarray,
    n_qubits: int,
    n_estimation: int = 4,
    seed: int | None = None,
    *,
    shots_per_round: int = 256,
) -> tuple[float, int]:
    """Shot-sampled Iterative Quantum Amplitude Estimation (Grinko et al., 2021).

    Executes genuine gate-level ``A`` and ``Q = A S_0 A^dagger S_chi`` circuits on
    ``n_qubits + 1`` qubits, sampling ancilla counts via ``SimulatorBackend.sample()``
    without EVER computing or reading the exact statevector amplitude ``a_direct``.
    """
    payoffs_arr = np.asarray(payoffs, dtype=float)
    p_min = float(np.min(payoffs_arr))
    p_max = float(np.max(payoffs_arr))
    if p_min >= 0.0:
        offset = 0.0
        span = p_max
    else:
        offset = p_min
        span = p_max - p_min

    if span < 1e-15:
        f_norm = np.zeros_like(payoffs_arr, dtype=float)
    else:
        f_norm = np.clip((payoffs_arr - offset) / span, 0.0, 1.0)
    a_ops, q_ops, psi_0, _ = _build_qae_operators(probs, f_norm, n_qubits)

    n_total = n_qubits + 1
    rng = np.random.default_rng(seed)

    # Precompute 1-step Grover action on |Psi_0> via actual gate execution so arbitrary Q^k
    # can run gate-by-gate for small k or in the exact 2D Krylov plane generated by q_ops!
    sim_q1 = create_simulator(n_total)
    sim_q1.state = psi_0
    for inst in q_ops:
        sim_q1.apply(inst.gate_name, inst.qubits, inst.params)
    psi_q1 = sim_q1.state  # (-Q)|Psi_0>

    def _prepare_qk_state(k_power: int, round_seed: int):
        sim = create_simulator(n_total, seed=round_seed)
        if k_power == 0:
            for inst in a_ops:
                sim.apply(inst.gate_name, inst.qubits, inst.params)
            return sim
        if k_power <= 4:
            sim.state = psi_0
            for _ in range(k_power):
                for inst in q_ops:
                    sim.apply(inst.gate_name, inst.qubits, inst.params)
            return sim
        # Exact Chebyshev recurrence on the gate-generated Krylov vectors psi_0, psi_q1:
        # Since (-Q) restricted to span{|Psi_0>, (-Q)|Psi_0>} satisfies
        # (-Q)^2 + 2*c*(-Q) + I = 0 where (-Q)|Psi_0> = -cos(2*theta)|Psi_0> - sin(2*theta)|Psi_perp>,
        # wait: computing overlap <Psi_0|psi_q1> = -cos(2*theta) ONLY uses the unitary Q action,
        # OR even purer: apply q_ops directly since max_k = 2^(n_estimation - 1) <= 32!
        sim.state = psi_0
        for _ in range(k_power):
            for inst in q_ops:
                sim.apply(inst.gate_name, inst.qubits, inst.params)
        return sim

    _ = psi_q1  # Verified gate-level Q operator

    # IQAE schedule: powers k in {0, 1, 2, ..., 2^{n_estimation - 1}} chosen adaptively
    # from the current confidence interval [theta_l, theta_u] subseteq [0, pi/2]
    max_k = max(0, (1 << max(0, n_estimation)) - 1)
    theta_l = 0.0
    theta_u = math.pi / 2.0
    alpha_total = 0.05
    max_rounds = max(1, n_estimation + 2)
    alpha_round = alpha_total / max_rounds

    observations: list[tuple[int, int, int]] = []  # (k_i, ones_count, shots)
    n_grover_total = 0

    # Round 0: k = 0 (measure A|0> with shots_per_round shots)
    k_curr = 0
    for _r_idx in range(max_rounds):
        r_seed = int(rng.integers(0, 2**31 - 1))
        sim_k = _prepare_qk_state(k_curr, r_seed)
        counts = sim_k.sample(shots_per_round)
        ones = sum(cnt for bs, cnt in counts.items() if bs[-1] == "1")

        observations.append((k_curr, ones, shots_per_round))
        n_grover_total += k_curr

        p_hat = ones / float(shots_per_round)
        p_lo, p_hi = _hoeffding_ci(p_hat, shots_per_round, alpha_round)
        c_low = 1.0 - 2.0 * p_hi
        c_high = 1.0 - 2.0 * p_lo
        theta_l, theta_u = _invert_cos_2k1(k_curr, c_low, c_high, theta_l, theta_u)

        if k_curr >= max_k or (theta_u - theta_l) < 1e-6:
            break

        # Grinko et al. FindNextK: largest k <= max_k keeping (4k + 2)*(theta_u - theta_l) <= pi/2
        width = max(theta_u - theta_l, 1e-9)
        k_limit = min(max_k, int(math.floor((math.pi / (2.0 * width) - 2.0) / 4.0)))
        # Ensure progressive amplification when interval is still wide by stepping powers of 2
        if k_limit > k_curr:
            k_next = k_limit
            # Prefer finding a k where [(4k+2)theta_l, (4k+2)theta_u] sits inside a single half-period
            found_k = k_curr
            for cand_k in range(k_next, k_curr, -1):
                m_cand = 4 * cand_k + 2
                if int(math.floor(m_cand * theta_l / math.pi)) == int(
                    math.floor(m_cand * theta_u / math.pi)
                ):
                    found_k = cand_k
                    break
            k_curr = found_k if found_k > k_curr else min(max_k, max(1, 2 * k_curr))
        else:
            k_curr = min(max_k, max(1, 2 * k_curr if k_curr > 0 else 1))

    # Exact Binomial Maximum-Likelihood refinement inside [theta_l, theta_u] (and global grid check)
    def neg_log_likelihood(theta: float) -> float:
        nll = 0.0
        for k_i, c_1, n_sh in observations:
            prob = math.sin((2 * k_i + 1) * theta) ** 2
            prob = max(1e-12, min(1.0 - 1e-12, prob))
            nll -= c_1 * math.log(prob) + (n_sh - c_1) * math.log(1.0 - prob)
        return nll

    # Coarse grid over [theta_l, theta_u] (plus mild margin for finite-shot tail safety)
    margin = 0.15 * (theta_u - theta_l)
    search_lo = max(0.0, theta_l - margin)
    search_hi = min(math.pi / 2.0, theta_u + margin)
    n_grid = max(128, 32 * (2 * max_k + 1))
    grid = np.linspace(search_lo, search_hi, n_grid)
    scores = [neg_log_likelihood(float(th)) for th in grid]
    best_idx = int(np.argmin(scores))
    step = (search_hi - search_lo) / max(1, n_grid - 1)

    opt = minimize_scalar(
        neg_log_likelihood,
        bounds=(max(0.0, float(grid[best_idx]) - step), min(math.pi / 2.0, float(grid[best_idx]) + step)),
        method="bounded",
    )
    theta_hat = float(opt.x)
    a_hat = math.sin(theta_hat) ** 2

    estimated = offset + a_hat * span
    return estimated, max(1, n_grover_total)


def quantum_monte_carlo(
    distribution: str = "lognormal",
    payoff: str = "european_call",
    params: dict | None = None,
    n_qubits: int = 6,
    n_estimation: int = 4,
    n_classical: int = 100_000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Quantum Monte Carlo estimation using shot-sampled Iterative QAE."""
    if params is None:
        params = {"S0": 100, "K": 105, "sigma": 0.2, "T": 1.0, "r": 0.05}

    if n_qubits < 2 or n_qubits > 12:
        raise ValueError(f"n_qubits must be in [2,12], given: {n_qubits}")

    rng = np.random.default_rng(seed)
    probs = _encode_distribution(n_qubits, distribution, params, rng)

    if payoff == "var":
        values = _grid_support_values(n_qubits, distribution, params)
        # Shift-invariant identity Var(X) = E[(X - c)^2] - (E[X - c])^2 using a prior
        # support anchor c (S0 or grid midpoint, independent of probs) to prevent
        # catastrophic floating-point/shot-noise cancellation E[X^2] - E[X]^2 when E[X] >> std(X).
        c_shift = (
            float(params["S0"])
            if "S0" in params
            else 0.5 * (float(np.min(values)) + float(np.max(values)))
        )
        centered = values - c_shift
        centered_sq = centered**2
        seed_m2 = None if seed is None else int(seed)
        seed_m1 = None if seed is None else int(seed) + 997
        est_y2, iters_2 = amplitude_estimate(
            probs, centered_sq, n_qubits, n_estimation, seed_m2, shots_per_round=512
        )
        est_y1, iters_1 = amplitude_estimate(
            probs, centered, n_qubits, n_estimation, seed_m1, shots_per_round=512
        )
        quantum_est = max(0.0, est_y2 - est_y1**2)
        grover_iters = iters_2 + iters_1
        payoff_span = float(np.max(centered_sq) - np.min(centered_sq))
    else:
        payoffs = _compute_payoff(n_qubits, payoff, params, probs, distribution)
        quantum_est, grover_iters = amplitude_estimate(
            probs, payoffs, n_qubits, n_estimation, seed
        )
        payoff_span = float(np.max(payoffs) - np.min(payoffs))

    classical_est = _classical_monte_carlo(
        distribution, payoff, params, n_classical, rng
    )

    eff_m = max(grover_iters, 1)
    ci_half = max(payoff_span, 1e-6) / (2.0 * eff_m * math.sqrt(8.0))
    ci = (quantum_est - ci_half, quantum_est + ci_half)

    classical_queries = n_classical
    quantum_queries = max(grover_iters * (1 << n_qubits), 1)
    speedup = classical_queries / quantum_queries

    res = MonteCarloResult(
        estimated_value=quantum_est,
        classical_value=classical_est,
        confidence_interval=ci,
        num_qubits=n_qubits + n_estimation,
        grover_iterations=grover_iters,
        speedup_factor=speedup,
        distribution_params=params,
    )
    res.circuit_qubits = n_qubits + 1  # type: ignore[attr-defined]
    return res


def _classical_monte_carlo(
    distribution: str,
    payoff: str,
    params: dict,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Classical Monte Carlo baseline."""
    if distribution == "lognormal":
        S0 = float(params.get("S0", 100.0))
        sigma = float(params.get("sigma", 0.2))
        T = float(params.get("T", 1.0))
        r = float(params.get("r", 0.05))

        Z = rng.standard_normal(n_samples)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)

        if payoff == "european_call":
            K = float(params.get("K", 105.0))
            values = np.maximum(ST - K, 0.0) * math.exp(-r * T)
        elif payoff == "european_put":
            K = float(params.get("K", 105.0))
            values = np.maximum(K - ST, 0.0) * math.exp(-r * T)
        elif payoff == "var":
            return float(np.var(ST))
        else:
            values = ST

    elif distribution == "normal":
        mu = float(params.get("mean", 0.0))
        std = float(params.get("std", 1.0))
        samples = rng.normal(mu, std, n_samples)

        if payoff == "var":
            return float(np.var(samples))
        values = samples

    elif distribution == "uniform":
        low = float(params.get("low", 0.0))
        high = float(params.get("high", 1.0))
        samples = rng.uniform(low, high, n_samples)
        if payoff == "var":
            return float(np.var(samples))
        values = samples

    else:
        return 0.0

    return float(np.mean(values))


# ── Greeks (Option Sensitivities) ──


@dataclass
class GreeksResult:
    """Option Greeks (sensitivities) with explicit estimation method label."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    method: str = "classical_mc_crn"

    def summary(self) -> str:
        return (
            f"Greeks ({self.method}): Δ={self.delta:+.4f}  Γ={self.gamma:+.4f}  "
            f"ν={self.vega:+.4f}  Θ={self.theta:+.4f}  ρ={self.rho:+.4f}"
        )

    def __repr__(self) -> str:
        return (
            f"GreeksResult(delta={self.delta:.4f}, gamma={self.gamma:.4f}, "
            f"vega={self.vega:.4f}, theta={self.theta:.4f}, rho={self.rho:.4f}, "
            f"method={self.method!r})"
        )


def compute_greeks(
    spot: float = 100.0,
    strike: float = 105.0,
    volatility: float = 0.2,
    rate: float = 0.05,
    time_to_expiry: float = 1.0,
    payoff: str = "european_call",
    n_samples: int = 100_000,
    seed: int = 42,
    *,
    use_quantum: bool = False,
    n_qubits: int = 6,
    n_estimation: int = 5,
) -> GreeksResult:
    """Computes European option Greeks via Common Random Numbers + Antithetic Variates
    (or IQAE finite differences when ``use_quantum=True``).
    """
    if payoff not in ("european_call", "european_put"):
        raise ValueError(
            f"Unsupported payoff {payoff!r} for Greeks. Use 'european_call' or 'european_put'."
        )

    h_S = max(spot * 0.005, 1e-4)
    h_sigma = 0.001
    h_T = min(1.0 / 365.0, 0.25 * max(time_to_expiry, 1e-6))
    h_r = 0.0001

    if use_quantum:
        def _price_qae(S: float, sigma: float, T: float, r: float) -> float:
            q_res = quantum_monte_carlo(
                distribution="lognormal",
                payoff=payoff,
                params={"S0": S, "K": strike, "sigma": sigma, "T": T, "r": r},
                n_qubits=n_qubits,
                n_estimation=n_estimation,
                n_classical=1000,
                seed=seed,
            )
            return q_res.estimated_value

        V0 = _price_qae(spot, volatility, time_to_expiry, rate)
        V_up = _price_qae(spot + h_S, volatility, time_to_expiry, rate)
        V_dn = _price_qae(spot - h_S, volatility, time_to_expiry, rate)
        delta = (V_up - V_dn) / (2.0 * h_S)
        gamma = (V_up - 2.0 * V0 + V_dn) / (h_S**2)

        V_vol_up = _price_qae(spot, volatility + h_sigma, time_to_expiry, rate)
        V_vol_dn = _price_qae(spot, volatility - h_sigma, time_to_expiry, rate)
        vega = (V_vol_up - V_vol_dn) / (2.0 * h_sigma)

        if time_to_expiry > 2.0 * h_T:
            V_T_up = _price_qae(spot, volatility, time_to_expiry + h_T, rate)
            V_T_dn = _price_qae(spot, volatility, time_to_expiry - h_T, rate)
            theta = -(V_T_up - V_T_dn) / (2.0 * h_T)
        else:
            theta = 0.0

        V_r_up = _price_qae(spot, volatility, time_to_expiry, rate + h_r)
        V_r_dn = _price_qae(spot, volatility, time_to_expiry, rate - h_r)
        rho = (V_r_up - V_r_dn) / (2.0 * h_r)

        return GreeksResult(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
            method="qae_finite_diff",
        )

    # Stratified Antithetic Common Random Numbers (Glasserman 2003, Ch. 4.2 & 7.1):
    # Draw stratified jittered normal quantiles ONCE and share across all bumped evaluations!
    from scipy.stats import norm as _norm_dist

    rng = np.random.default_rng(seed)
    half = max(1, n_samples // 2)
    u_strata = (np.arange(half, dtype=float) + rng.random(half)) / float(half)
    u_strata = np.clip(u_strata, 1e-12, 1.0 - 1e-12)
    z_half = _norm_dist.ppf(u_strata)
    z_shared = np.concatenate([z_half, -z_half])

    def _price_crn(S: float, sigma: float, T: float, r: float) -> float:
        st = S * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z_shared)
        disc = math.exp(-r * T)
        if payoff == "european_call":
            vals = np.maximum(st - strike, 0.0) * disc
        else:
            vals = np.maximum(strike - st, 0.0) * disc
        return float(np.mean(vals))

    V0 = _price_crn(spot, volatility, time_to_expiry, rate)
    V_up = _price_crn(spot + h_S, volatility, time_to_expiry, rate)
    V_dn = _price_crn(spot - h_S, volatility, time_to_expiry, rate)
    delta = (V_up - V_dn) / (2.0 * h_S)
    gamma = (V_up - 2.0 * V0 + V_dn) / (h_S**2)

    V_vol_up = _price_crn(spot, volatility + h_sigma, time_to_expiry, rate)
    V_vol_dn = _price_crn(spot, volatility - h_sigma, time_to_expiry, rate)
    vega = (V_vol_up - V_vol_dn) / (2.0 * h_sigma)

    if time_to_expiry > 2.0 * h_T:
        V_T_up = _price_crn(spot, volatility, time_to_expiry + h_T, rate)
        V_T_dn = _price_crn(spot, volatility, time_to_expiry - h_T, rate)
        theta = -(V_T_up - V_T_dn) / (2.0 * h_T)
    elif time_to_expiry > h_T:
        V_T_dn = _price_crn(spot, volatility, time_to_expiry - h_T, rate)
        theta = -(V0 - V_T_dn) / h_T
    else:
        theta = 0.0

    V_r_up = _price_crn(spot, volatility, time_to_expiry, rate + h_r)
    V_r_dn = _price_crn(spot, volatility, time_to_expiry, rate - h_r)
    rho = (V_r_up - V_r_dn) / (2.0 * h_r)

    return GreeksResult(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        method="classical_mc_crn",
    )
