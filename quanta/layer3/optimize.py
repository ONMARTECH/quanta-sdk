"""
quanta.layer3.optimize — Declarative quantum optimization.

Solves combinatorial optimization problems using QAOA (Quantum Approximate
Optimization Algorithm).

Example:
    >>> from quanta.layer3.optimize import optimize
    >>> result = optimize(
    ...     num_bits=3,
    ...     cost=lambda x: bin(x ^ (x >> 1)).count('1'),  # cut cost
    ...     minimize=False,  # maximize
    ... )
    >>> print(f"Best: {result.best_bitstring} (cost: {result.best_cost})")

Behind the scenes:
  1. Analyze cost function
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from quanta.result import Result
from quanta.simulator.statevector import StateVectorSimulator

# ── Public API ──
__all__ = ["optimize", "OptimizationResult"]

@dataclass
class OptimizationResult:
    """Optimization result.

    Attributes:
        quantum_result: Raw quantum measurement results.
    """

    best_bitstring: str
    best_cost: float
    all_solutions: list[tuple[str, float, float]]  # (bitstring, cost, probability)
    quantum_result: Result

    def summary(self) -> str:
        lines = [
            "╔═══ Optimization Result ═══",
            f"║ Cost: {self.best_cost}",
        ]
        for bits, cost, prob in self.all_solutions[:5]:
            bar = "█" * int(prob * 30)
            lines.append(f"║ |{bits}⟩  cost={cost:.2f}  P={prob:.3f}  {bar}")
        lines.append("╚" + "═" * 40)
        return "\n".join(lines)

def optimize(
    num_bits: int,
    cost: Callable[[int], float],
    minimize: bool = True,
    layers: int = 2,
    shots: int = 2048,
    seed: int | None = None,
) -> OptimizationResult:
    """Quantum optimization — QAOA based.

    Finds the bitstring that minimizes or maximizes the cost function.

    Args:
        cost: Cost function. int → float.
        minimize: True to minimize, False to maximize.
        seed: Random seed.

    Returns:
    """
    if num_bits < 1 or num_bits > 15:
        raise ValueError(f"num_bits must be in [1,15], given: {num_bits}")

    rng = np.random.default_rng(seed)
    n_states = 2 ** num_bits

    costs = np.array([cost(i) for i in range(n_states)])
    if not minimize:
        costs = -costs

    best_params = _optimize_qaoa_params(num_bits, costs, layers, rng)
    sim = _run_qaoa(num_bits, costs, best_params, seed)

    # Sample results
    counts = sim.sample(shots)

    solutions: list[tuple[str, float, float]] = []
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        real_cost = cost(idx)
        prob = count / shots
        solutions.append((bitstring, real_cost, prob))

    solutions.sort(key=lambda x: x[1] if minimize else -x[1])

    quantum_result = Result(
        counts=counts, shots=shots, num_qubits=num_bits,
        circuit_name=f"optimize(bits={num_bits}, layers={layers})",
    )

    return OptimizationResult(
        best_bitstring=solutions[0][0],
        best_cost=solutions[0][1],
        all_solutions=solutions,
        quantum_result=quantum_result,
    )

def _optimize_qaoa_params(
    n: int, costs: np.ndarray, layers: int, rng: np.random.Generator
) -> np.ndarray:
    """QAOA parametrelerini basit grid search ile optimize eder.

    Args:

    Returns:
        Optimal (gamma, beta) parametreleri.
    """
    best_params = None
    best_expectation = float("inf")

    for _ in range(50):  # 50 rastgele deneme
        params = rng.uniform(0, 2 * np.pi, size=2 * layers)
        sim = _run_qaoa(n, costs, params, seed=None)
        probs = sim.probabilities()
        expectation = np.sum(probs * costs)

        if expectation < best_expectation:
            best_expectation = expectation
            best_params = params

    return best_params if best_params is not None else np.zeros(2 * layers)

def _run_qaoa(
    n: int, costs: np.ndarray, params: np.ndarray, seed: int | None
) -> StateVectorSimulator:
    """Runs the QAOA circuit.

    Each layer: exp(-i·γ·C)·exp(-i·β·B)
    """
    sim = StateVectorSimulator(n, seed=seed)
    layers = len(params) // 2

    for q in range(n):
        sim.apply("H", (q,))

    for p in range(layers):
        gamma = params[2 * p]
        beta = params[2 * p + 1]

        # Cost layer: apply diagonal phase exp(-i*gamma*C)
        state = sim.state  # public copy
        for i in range(len(state)):
            state[i] *= np.exp(-1j * gamma * costs[i])
        sim.state = state  # public setter

        for q in range(n):
            sim.apply("RX", (q,), (2 * beta,))

    return sim
