"""
quanta.layer3.optimize — Gate-compiled Walsh-Hadamard QAOA optimization.

Decomposes arbitrary combinatorial cost functions C(x) into exact Pauli-Z
interactions via the Fast Walsh-Hadamard Transform (FWHT):
    C = sum_{S} c_hat_S prod_{q in S} Z_q
and executes real gate-level QAOA circuits (`H`, `CX`-`RZ`-`CX` parity gadgets,
and `RX` mixer layers) compiled through ``DAGCircuit`` and trained via COBYLA.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from quanta.core.circuit import CircuitBuilder
from quanta.core.types import Instruction
from quanta.dag.dag_circuit import DAGCircuit
from quanta.result import Result
from quanta.simulator.base import SimulatorBackend
from quanta.simulator.factory import create_simulator

# ── Public API ──
__all__ = ["optimize", "OptimizationResult", "walsh_hadamard_terms"]


@dataclass
class OptimizationResult:
    """Optimization result from gate-compiled QAOA."""

    best_bitstring: str
    best_cost: float
    all_solutions: list[tuple[str, float, float]]  # (bitstring, cost, probability)
    quantum_result: Result

    def summary(self) -> str:
        lines = [
            "╔═══ Optimization Result ═══",
            f"║ Cost: {self.best_cost}",
        ]
        for bits, cost_val, prob in self.all_solutions[:5]:
            bar = "█" * int(prob * 30)
            lines.append(f"║ |{bits}⟩  cost={cost_val:.2f}  P={prob:.3f}  {bar}")
        lines.append("╚" + "═" * 40)
        return "\n".join(lines)


def walsh_hadamard_terms(
    num_bits: int, costs: np.ndarray, tol: float = 1e-12
) -> list[tuple[tuple[int, ...], float]]:
    """Computes non-trivial Pauli-Z Walsh-Hadamard terms (qubits_tuple, c_hat_S).

    In MSB-first convention, bit position ``q`` (``0 <= q < num_bits``) corresponds to
    ``(x >> (num_bits - 1 - q)) & 1``, so ``Z_q |x> = (-1)^{bit_q(x)} |x>``.
    """
    c_hat = np.asarray(costs, dtype=float).copy()
    dim = len(c_hat)
    h = 1
    while h < dim:
        for i in range(0, dim, h * 2):
            u = c_hat[i : i + h].copy()
            v = c_hat[i + h : i + 2 * h].copy()
            c_hat[i : i + h] = u + v
            c_hat[i + h : i + 2 * h] = u - v
        h *= 2
    c_hat /= float(dim)

    terms: list[tuple[tuple[int, ...], float]] = []
    for mask in range(1, dim):
        coeff = float(c_hat[mask])
        if abs(coeff) <= tol:
            continue
        qubits = tuple(
            q for q in range(num_bits) if ((mask >> (num_bits - 1 - q)) & 1) == 1
        )
        terms.append((qubits, coeff))
    return terms


def _build_qaoa_dag(
    num_bits: int,
    pauli_terms: list[tuple[tuple[int, ...], float]],
    params: np.ndarray,
) -> DAGCircuit:
    """Builds a real gate-level QAOA DAGCircuit from Pauli-Z terms and (gamma, beta) angles."""
    builder = CircuitBuilder(num_bits)
    layers = len(params) // 2

    for q in range(num_bits):
        builder.record(Instruction("H", (q,)))

    for p in range(layers):
        gamma = float(params[2 * p])
        beta = float(params[2 * p + 1])

        if abs(gamma) > 1e-15:
            for qubits, coeff in pauli_terms:
                angle = 2.0 * gamma * coeff
                if abs(angle) <= 1e-15:
                    continue
                if len(qubits) == 1:
                    builder.record(Instruction("RZ", (qubits[0],), (angle,)))
                else:
                    for idx in range(len(qubits) - 1):
                        builder.record(Instruction("CX", (qubits[idx], qubits[idx + 1])))
                    builder.record(Instruction("RZ", (qubits[-1],), (angle,)))
                    for idx in range(len(qubits) - 2, -1, -1):
                        builder.record(Instruction("CX", (qubits[idx], qubits[idx + 1])))

        for q in range(num_bits):
            builder.record(Instruction("RX", (q,), (2.0 * beta,)))

    return DAGCircuit.from_builder(builder)


def _run_qaoa_dag(
    num_bits: int,
    pauli_terms: list[tuple[tuple[int, ...], float]],
    params: np.ndarray,
    seed: int | None,
) -> tuple[SimulatorBackend, DAGCircuit]:
    """Executes the gate-compiled QAOA DAGCircuit on the simulator."""
    dag = _build_qaoa_dag(num_bits, pauli_terms, params)
    sim = create_simulator(num_bits, seed=seed)
    for op in dag.op_nodes():
        sim.apply(op.gate_name, op.qubits, op.params)
    return sim, dag


def _optimize_qaoa_params(
    n: int,
    costs: np.ndarray,
    pauli_terms: list[tuple[tuple[int, ...], float]],
    layers: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Variationally optimizes QAOA (gamma, beta) parameters via multi-start COBYLA."""
    cost_scale = float(np.max(np.abs(costs - np.mean(costs))))
    if cost_scale < 1e-12:
        return np.zeros(2 * layers, dtype=float)
    scaled_terms = [(qs, coeff / cost_scale) for qs, coeff in pauli_terms]

    def objective(params: np.ndarray) -> float:
        sim, _ = _run_qaoa_dag(n, scaled_terms, params, seed=None)
        probs = sim.probabilities()
        return float(np.dot(probs, costs))

    best_params = np.zeros(2 * layers, dtype=float)
    best_value = float("inf")

    for _ in range(5):
        x0 = rng.uniform(0.0, np.pi, size=2 * layers)
        res = scipy_minimize(
            objective,
            x0,
            method="COBYLA",
            options={"maxiter": 120, "rhobeg": 0.4},
        )
        if res.fun < best_value:
            best_value = float(res.fun)
            best_params = np.asarray(res.x, dtype=float)

    # Rescale gamma parameters back to unscaled cost Hamiltonian
    unscaled_params = best_params.copy()
    for p in range(layers):
        unscaled_params[2 * p] /= cost_scale
    return unscaled_params


def optimize(
    num_bits: int,
    cost: Callable[[int], float],
    minimize: bool = True,
    layers: int = 2,
    shots: int = 2048,
    seed: int | None = None,
) -> OptimizationResult:
    """Gate-based QAOA combinatorial optimization.

    Args:
        num_bits: Number of binary decision variables (1 to 15).
        cost: Objective function ``int -> float``.
        minimize: ``True`` to minimize ``cost``, ``False`` to maximize ``cost``.
        layers: QAOA circuit depth ``p``.
        shots: Measurement shots.
        seed: Random seed.

    Returns:
        OptimizationResult with best sampled solution, all sampled solutions,
        and gate-level ``Result``.
    """
    if num_bits < 1 or num_bits > 15:
        raise ValueError(f"num_bits must be in [1,15], given: {num_bits}")
    if layers < 1:
        raise ValueError(f"layers must be >= 1, given: {layers}")

    rng = np.random.default_rng(seed)
    n_states = 1 << num_bits

    raw_costs = np.array([float(cost(i)) for i in range(n_states)], dtype=float)
    eff_costs = raw_costs if minimize else -raw_costs

    pauli_terms = walsh_hadamard_terms(num_bits, eff_costs)
    best_params = _optimize_qaoa_params(num_bits, eff_costs, pauli_terms, layers, rng)
    sim, dag = _run_qaoa_dag(num_bits, pauli_terms, best_params, seed)

    counts = sim.sample(shots)

    solutions: list[tuple[str, float, float]] = []
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        real_cost = float(raw_costs[idx])
        prob = count / shots
        solutions.append((bitstring, real_cost, prob))

    solutions.sort(key=lambda item: (item[1] if minimize else -item[1], -item[2]))

    quantum_result = Result(
        counts=counts,
        shots=shots,
        num_qubits=num_bits,
        circuit_name=f"optimize(bits={num_bits}, layers={layers})",
        gate_count=dag.gate_count(),
        depth=dag.depth(),
        statevector=sim.state,
    )

    return OptimizationResult(
        best_bitstring=solutions[0][0],
        best_cost=solutions[0][1],
        all_solutions=solutions,
        quantum_result=quantum_result,
    )
