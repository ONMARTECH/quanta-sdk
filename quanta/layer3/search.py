"""
quanta.layer3.search — Declarative quantum search.

No gate knowledge needed. Full quantum abstraction.

Example:
    >>> from quanta.layer3.search import search
    >>> result = search(
    ...     num_bits=3,
    ...     target=lambda x: x == 5,   # find |101⟩
    ...     shots=1024,
    ... )
    >>> print(result.most_frequent)
    '101'

"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from quanta.result import Result
from quanta.simulator.statevector import StateVectorSimulator

# ── Public API ──
__all__ = ["search"]

def search(
    num_bits: int,
    target: Callable[[int], bool] | int,
    shots: int = 1024,
    seed: int | None = None,
) -> Result:
    """Quantum search — finds target using Grover's algorithm.


    Args:
        seed: Random seed.

    Returns:

    Raises:
        ValueError: If num_bits < 1 or no target found.

    Example:
        >>> result.most_frequent
        '1101'
    """
    if num_bits < 1 or num_bits > 15:
        raise ValueError(f"num_bits must be in [1,15], given: {num_bits}")

    check_fn = _normalize_target(target)

    targets = _find_targets(num_bits, check_fn)
    if not targets:
        raise ValueError("No state found satisfying the target condition.")

    n_states = 2 ** num_bits
    n_targets = len(targets)
    iterations = max(1, round(math.pi / 4 * math.sqrt(n_states / n_targets)))

    sim = StateVectorSimulator(num_bits, seed=seed)

    for q in range(num_bits):
        sim.apply("H", (q,))

    for _ in range(iterations):
        _apply_oracle(sim, num_bits, targets)
        _apply_diffusion(sim, num_bits)

    counts = sim.sample(shots)

    return Result(
        counts=counts,
        shots=shots,
        num_qubits=num_bits,
        circuit_name=f"search(bits={num_bits}, targets={len(targets)})",
        gate_count=num_bits + iterations * (2 * num_bits + 4),
        depth=1 + iterations * 4,
        statevector=sim.state,
    )

def _normalize_target(target: Callable[[int], bool] | int) -> Callable[[int], bool]:
    """Normalizes target argument to callable."""
    if isinstance(target, int):
        value = target
        return lambda x: x == value
    return target

def _find_targets(num_bits: int, check: Callable[[int], bool]) -> list[int]:
    """Finds all states satisfying the target condition."""
    return [i for i in range(2 ** num_bits) if check(i)]

def _apply_oracle(sim: StateVectorSimulator, n: int, targets: list[int]) -> None:
    """Oracle: Applies -1 phase to target states.

    """
    for t in targets:
        sim._state[t] *= -1

def _apply_diffusion(sim: StateVectorSimulator, n: int) -> None:
    """Diffusion operator: 2|ψ⟩⟨ψ| - I.

    """
    state = sim._state
    dim = len(state)

    mean = np.sum(state) / dim

    sim._state = 2 * mean - state
