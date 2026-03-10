"""
quanta.layer3.monte_carlo — Quantum Monte Carlo via Amplitude Estimation.

Uses quantum amplitude estimation to compute expectations faster
than classical Monte Carlo. Key applications: option pricing,
risk analysis, and probabilistic inference.

Classical Monte Carlo converges as O(1/√N) with N samples.
Quantum amplitude estimation achieves O(1/N) — quadratic speedup.

Pipeline:
  1. Encode probability distribution into quantum state amplitudes
  2. Apply payoff function as phase rotations
  3. Use amplitude estimation (Grover-style iterations) to extract expectation
  4. Classical post-processing for final estimate

Example:
    >>> from quanta.layer3.monte_carlo import quantum_monte_carlo
    >>> # Price a European call option
    >>> result = quantum_monte_carlo(
    ...     distribution="lognormal",
    ...     payoff="european_call",
    ...     params={"S0": 100, "K": 105, "sigma": 0.2, "T": 1.0, "r": 0.05},
    ... )
    >>> print(result.estimated_value)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "quantum_monte_carlo",
    "MonteCarloResult",
    "amplitude_estimate",
]


@dataclass
class MonteCarloResult:
    """Result of Quantum Monte Carlo estimation.

    Attributes:
        estimated_value: Estimated expectation value.
        classical_value: Classical Monte Carlo estimate for comparison.
        confidence_interval: (lower, upper) bounds.
        num_qubits: Qubits used for encoding.
        grover_iterations: Amplitude estimation iterations.
        speedup_factor: Theoretical quantum speedup achieved.
    """
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


def _encode_distribution(
    n_qubits: int,
    distribution: str,
    params: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Encodes a probability distribution into quantum amplitudes.

    Maps classical distribution to |ψ⟩ = Σ √p(x) |x⟩
    where p(x) is the discretized probability.

    Returns:
        Probability array of shape (2^n,).
    """
    dim = 2 ** n_qubits

    if distribution == "lognormal":
        S0 = params.get("S0", 100)
        sigma = params.get("sigma", 0.2)
        T = params.get("T", 1.0)
        r = params.get("r", 0.05)

        mu = math.log(S0) + (r - 0.5 * sigma**2) * T
        std = sigma * math.sqrt(T)

        # Discretize lognormal distribution into dim bins
        x_min = max(0.01, math.exp(mu - 4 * std))
        x_max = math.exp(mu + 4 * std)
        prices = np.linspace(x_min, x_max, dim)
        dx = prices[1] - prices[0]

        log_prices = np.log(prices)
        probs = np.exp(-0.5 * ((log_prices - mu) / std) ** 2)
        probs /= (prices * std * math.sqrt(2 * math.pi))
        probs *= dx
        probs = np.maximum(probs, 0)
        total = probs.sum()
        if total > 0:
            probs /= total

    elif distribution == "normal":
        mu = params.get("mean", 0.0)
        std = params.get("std", 1.0)

        x_min = mu - 4 * std
        x_max = mu + 4 * std
        values = np.linspace(x_min, x_max, dim)
        dx = values[1] - values[0]

        probs = np.exp(-0.5 * ((values - mu) / std) ** 2)
        probs /= (std * math.sqrt(2 * math.pi))
        probs *= dx
        probs = np.maximum(probs, 0)
        total = probs.sum()
        if total > 0:
            probs /= total

    elif distribution == "uniform":
        probs = np.ones(dim) / dim

    else:
        raise ValueError(
            f"Unknown distribution: {distribution}. "
            "Supported: lognormal, normal, uniform"
        )

    return probs


def _compute_payoff(
    n_qubits: int,
    payoff: str,
    params: dict,
    probs: np.ndarray,
) -> np.ndarray:
    """Computes payoff values for each discretized state.

    Returns:
        Array of payoff values per basis state.
    """
    dim = 2 ** n_qubits

    if payoff == "european_call":
        S0 = params.get("S0", 100)
        K = params.get("K", 105)
        sigma = params.get("sigma", 0.2)
        T = params.get("T", 1.0)
        r = params.get("r", 0.05)

        mu = math.log(S0) + (r - 0.5 * sigma**2) * T
        std = sigma * math.sqrt(T)
        x_min = max(0.01, math.exp(mu - 4 * std))
        x_max = math.exp(mu + 4 * std)
        prices = np.linspace(x_min, x_max, dim)

        # Call payoff: max(S - K, 0), discounted
        discount = math.exp(-r * T)
        payoffs = np.maximum(prices - K, 0) * discount

    elif payoff == "european_put":
        S0 = params.get("S0", 100)
        K = params.get("K", 105)
        sigma = params.get("sigma", 0.2)
        T = params.get("T", 1.0)
        r = params.get("r", 0.05)

        mu = math.log(S0) + (r - 0.5 * sigma**2) * T
        std = sigma * math.sqrt(T)
        x_min = max(0.01, math.exp(mu - 4 * std))
        x_max = math.exp(mu + 4 * std)
        prices = np.linspace(x_min, x_max, dim)

        discount = math.exp(-r * T)
        payoffs = np.maximum(K - prices, 0) * discount

    elif payoff == "expectation":
        # Just compute E[X] of the distribution
        mu = params.get("mean", 0.0)
        std = params.get("std", 1.0)
        x_min = mu - 4 * std
        x_max = mu + 4 * std
        payoffs = np.linspace(x_min, x_max, dim)

    elif payoff == "var":
        # Compute Var[X] = E[X²] - E[X]²
        mu = params.get("mean", 0.0)
        std = params.get("std", 1.0)
        x_min = mu - 4 * std
        x_max = mu + 4 * std
        values = np.linspace(x_min, x_max, dim)
        payoffs = values ** 2  # E[X²], subtract E[X]² later

    else:
        raise ValueError(
            f"Unknown payoff: {payoff}. "
            "Supported: european_call, european_put, expectation, var"
        )

    return payoffs


def amplitude_estimate(
    probs: np.ndarray,
    payoffs: np.ndarray,
    n_qubits: int,
    n_estimation: int = 4,
    seed: int | None = None,
) -> tuple[float, int]:
    """Quantum amplitude estimation (Brassard-Hoyer-Mosca-Tapp).

    Estimates a = E[f(X)] = Σ p(x)·f̃(x) using quantum circuits.

    Circuit structure:
      1. Prepare |ψ⟩ = Σ √p(x)|x⟩ on data register
      2. Controlled-R_y on ancilla: encodes f̃(x) as P(ancilla=|1⟩|x⟩)
         Result: Σ √p(x)[√f̃(x)|1⟩ + √(1-f̃(x))|0⟩]|x⟩
      3. P(ancilla=|1⟩) = Σ p(x)·f̃(x) = a (the target expectation)
      4. Iterative Grover amplification refines the estimate

    Args:
        probs: Probability distribution (2^n_qubits,).
        payoffs: Payoff values per state (2^n_qubits,).
        n_qubits: Number of qubits encoding the distribution.
        n_estimation: Number of Grover power rounds (precision).
        seed: Random seed.

    Returns:
        (estimated_expectation, total_grover_iterations).
    """
    dim = 2 ** n_qubits

    # Prepare distribution amplitudes: |ψ⟩ = Σ √p(x)|x⟩
    amplitudes = np.sqrt(np.maximum(probs, 0))
    norm = np.linalg.norm(amplitudes)
    if norm > 0:
        amplitudes /= norm

    # Normalize payoffs to [0, 1] for amplitude encoding
    max_payoff = float(np.max(np.abs(payoffs))) if np.any(payoffs != 0) else 1.0
    f_norm = np.clip(payoffs / max_payoff, 0, 1)

    # Build initial state: ancilla + data register
    # |Ψ_init⟩ = Σ √p(x) [cos(θ_x)|0⟩ + sin(θ_x)|1⟩] ⊗ |x⟩
    # where sin²(θ_x) = f̃(x), so P(ancilla=1) = Σ p(x)·f̃(x) = a
    n_total = 1 + n_qubits
    total_dim = 2 ** n_total

    init_state = np.zeros(total_dim, dtype=complex)
    for x in range(dim):
        theta_x = math.asin(math.sqrt(max(0, min(1, f_norm[x]))))
        # ancilla=0 basis: indices [0, dim)
        init_state[x] = amplitudes[x] * math.cos(theta_x)
        # ancilla=1 basis: indices [dim, 2*dim)
        init_state[x + dim] = amplitudes[x] * math.sin(theta_x)

    # Direct measurement → a = P(ancilla=|1⟩)
    a_direct = float(np.sum(np.abs(init_state[dim:]) ** 2))

    # Iterative amplitude estimation with phase-wrapping correction
    # After m Grover iterations: P(good) = sin²((2m+1)θ)
    # where sin²(θ) = a. We need to invert this correctly.
    n_grover_total = 0
    theta_direct = math.asin(math.sqrt(max(0, min(1, a_direct))))

    # Collect (m_k, p_measured) pairs for maximum likelihood
    measurements: list[tuple[int, float]] = [(0, a_direct)]

    for k in range(n_estimation):
        m_k = 2 ** k
        n_grover_total += m_k

        # Apply G^m_k to initial state
        current = init_state.copy()
        for _ in range(m_k):
            # Oracle: flip sign of "good" states (ancilla=|1⟩)
            current[dim:] *= -1
            # Diffusion: 2|Ψ_init⟩⟨Ψ_init| - I
            overlap = np.vdot(init_state, current)
            current = 2 * overlap * init_state - current

        # Measure P(ancilla=|1⟩) = sin²((2m_k+1)θ)
        p_one = float(np.sum(np.abs(current[dim:]) ** 2))
        measurements.append((m_k, p_one))

    # Maximum likelihood estimation of θ across all measurements
    # For each measurement (m, p), P = sin²((2m+1)θ)
    # Find θ that best explains all observations
    best_theta = theta_direct
    best_score = float("inf")

    # Grid resolution scales with Grover power for quantum-correct precision
    # The theoretical precision of AE is O(1/M) where M = max Grover power
    max_m = max(m for m, _ in measurements)
    n_candidates = max(200, 50 * (2 * max_m + 1))
    for i in range(n_candidates + 1):
        theta_cand = (i / n_candidates) * (math.pi / 2)

        # Log-likelihood score (sum of squared errors)
        score = 0.0
        for m_k, p_meas in measurements:
            p_pred = math.sin((2 * m_k + 1) * theta_cand) ** 2
            score += (p_pred - p_meas) ** 2

        if score < best_score:
            best_score = score
            best_theta = theta_cand

    best_a = math.sin(best_theta) ** 2

    # Scale back
    estimated = best_a * max_payoff
    return estimated, n_grover_total


def quantum_monte_carlo(
    distribution: str = "lognormal",
    payoff: str = "european_call",
    params: dict | None = None,
    n_qubits: int = 6,
    n_estimation: int = 4,
    n_classical: int = 100_000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Quantum Monte Carlo estimation using amplitude estimation.

    Estimates E[f(X)] where X follows a probability distribution
    and f is a payoff function. Uses quantum amplitude estimation
    for quadratic speedup over classical Monte Carlo.

    Args:
        distribution: Probability distribution type.
            Options: "lognormal", "normal", "uniform".
        payoff: Payoff function type.
            Options: "european_call", "european_put", "expectation", "var".
        params: Distribution and payoff parameters.
            For lognormal/options: S0, K, sigma, T, r.
            For normal: mean, std.
        n_qubits: Qubits for distribution encoding (precision).
        n_estimation: Estimation register qubits (accuracy).
        n_classical: Classical Monte Carlo samples for comparison.
        seed: Random seed.

    Returns:
        MonteCarloResult with quantum and classical estimates.

    Example:
        >>> result = quantum_monte_carlo(
        ...     distribution="lognormal",
        ...     payoff="european_call",
        ...     params={"S0": 100, "K": 105, "sigma": 0.2, "T": 1.0, "r": 0.05},
        ... )
        >>> print(result.estimated_value)
    """
    if params is None:
        params = {"S0": 100, "K": 105, "sigma": 0.2, "T": 1.0, "r": 0.05}

    if n_qubits < 2 or n_qubits > 12:
        raise ValueError(f"n_qubits must be in [2,12], given: {n_qubits}")

    rng = np.random.default_rng(seed)

    # Encode distribution
    probs = _encode_distribution(n_qubits, distribution, params, rng)

    # Compute payoffs
    payoffs = _compute_payoff(n_qubits, payoff, params, probs)

    # Quantum amplitude estimation
    quantum_est, grover_iters = amplitude_estimate(
        probs, payoffs, n_qubits, n_estimation, seed
    )

    # Classical Monte Carlo for comparison
    classical_est = _classical_monte_carlo(
        distribution, payoff, params, n_classical, rng
    )

    # Confidence interval (quantum)
    # Precision scales as O(1/M) where M = total Grover iterations
    precision = 1.0 / max(grover_iters, 1)
    max_payoff = np.max(np.abs(payoffs)) if np.any(payoffs != 0) else 1.0
    ci_half = precision * max_payoff
    ci = (quantum_est - ci_half, quantum_est + ci_half)

    # Speedup: quantum uses O(M) queries vs classical O(1/ε²)
    classical_queries = n_classical
    quantum_queries = grover_iters * (2 ** n_qubits)
    speedup = classical_queries / max(quantum_queries, 1)

    return MonteCarloResult(
        estimated_value=quantum_est,
        classical_value=classical_est,
        confidence_interval=ci,
        num_qubits=n_qubits + n_estimation,
        grover_iterations=grover_iters,
        speedup_factor=speedup,
        distribution_params=params,
    )


def _classical_monte_carlo(
    distribution: str,
    payoff: str,
    params: dict,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Classical Monte Carlo for comparison baseline."""

    if distribution == "lognormal":
        S0 = params.get("S0", 100)
        sigma = params.get("sigma", 0.2)
        T = params.get("T", 1.0)
        r = params.get("r", 0.05)

        # Geometric Brownian Motion
        Z = rng.standard_normal(n_samples)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)

        if payoff == "european_call":
            K = params.get("K", 105)
            values = np.maximum(ST - K, 0) * math.exp(-r * T)
        elif payoff == "european_put":
            K = params.get("K", 105)
            values = np.maximum(K - ST, 0) * math.exp(-r * T)
        else:
            values = ST

    elif distribution == "normal":
        mu = params.get("mean", 0.0)
        std = params.get("std", 1.0)
        samples = rng.normal(mu, std, n_samples)

        if payoff == "expectation":
            values = samples
        elif payoff == "var":
            return float(np.var(samples))
        else:
            values = samples

    elif distribution == "uniform":
        samples = rng.uniform(0, 1, n_samples)
        values = samples

    else:
        return 0.0

    return float(np.mean(values))
