"""
quanta.layer3.search — Genuine gate-compiled Grover quantum search.

Compiles phase oracles and Grover diffusion operators into elementary
quantum gates in ``GATE_REGISTRY`` (`H`, `X`, `Z`, `CX`, `CCX`, `P`)
through ``CircuitBuilder`` -> ``DAGCircuit``:
  - Direct integer / bitstring target specification (`target=5` or `targets=[3, 5]`)
    with ZERO O(2^n) classical enumeration (`classical_setup_queries == 0`).
  - Callable predicate support (`target=lambda x: ...`) with honest tracking of
    `classical_setup_queries` while compiling an exact multi-controlled phase oracle.
  - Boyer-Brassard-Høyer-Tapp (BBHT, 1998) exponential schedule when the number
    of marked solutions is unknown (`blind_oracle=True`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from quanta.core.circuit import CircuitBuilder
from quanta.core.types import Instruction
from quanta.dag.dag_circuit import DAGCircuit
from quanta.result import Result
from quanta.simulator.factory import create_simulator

# ── Public API ──
__all__ = ["search", "append_multi_controlled_z"]


def _append_cp(builder: CircuitBuilder, control: int, target: int, theta: float) -> None:
    """Decompose 2-qubit controlled-phase CP(theta) into elementary P and CX gates."""
    if abs(theta) < 1e-15:
        return
    half = theta / 2.0
    builder.record(Instruction("P", (control,), (half,)))
    builder.record(Instruction("P", (target,), (half,)))
    builder.record(Instruction("CX", (control, target)))
    builder.record(Instruction("P", (target,), (-half,)))
    builder.record(Instruction("CX", (control, target)))


def _append_mcx_dirty(
    builder: CircuitBuilder,
    controls: Sequence[int],
    target: int,
    dirty_ancilla: int,
) -> None:
    """Exact dirty-ancilla multi-controlled X (Barenco et al., Phys. Rev. A 52, 3457, 1995).

    Uses ONLY elementary X, CX, CCX gates and restores ``dirty_ancilla`` cleanly
    without introducing any relative phase.
    """
    m = len(controls)
    if m == 0:
        builder.record(Instruction("X", (target,)))
        return
    if m == 1:
        builder.record(Instruction("CX", (controls[0], target)))
        return
    if m == 2:
        builder.record(Instruction("CCX", (controls[0], controls[1], target)))
        return
    if m == 3:
        c0, c1, c2 = controls[0], controls[1], controls[2]
        builder.record(Instruction("CCX", (c0, c1, dirty_ancilla)))
        builder.record(Instruction("CCX", (c2, dirty_ancilla, target)))
        builder.record(Instruction("CCX", (c0, c1, dirty_ancilla)))
        builder.record(Instruction("CCX", (c2, dirty_ancilla, target)))
        return

    # m >= 4: split controls into A (size >= 2) and B (size >= 2).
    # B[0] serves as dirty ancilla for A; A[0] serves as dirty ancilla for B + [dirty_ancilla].
    split = (m + 1) // 2
    part_a = list(controls[:split])
    part_b = list(controls[split:])
    _append_mcx_dirty(builder, part_a, dirty_ancilla, part_b[0])
    _append_mcx_dirty(builder, part_b + [dirty_ancilla], target, part_a[0])
    _append_mcx_dirty(builder, part_a, dirty_ancilla, part_b[0])
    _append_mcx_dirty(builder, part_b + [dirty_ancilla], target, part_a[0])


def _append_mcp(
    builder: CircuitBuilder,
    controls: Sequence[int],
    target: int,
    theta: float,
) -> None:
    """Zero-ancilla k-controlled phase gate diag(1, ..., 1, exp(i*theta)).

    Uses Barenco Lemma 7.5 recursive phase kickback where ``target`` acts as the
    restored dirty ancilla for the prefix ``MCX`` toggles.
    """
    if abs(theta) < 1e-15:
        return
    k = len(controls)
    if k == 0:
        builder.record(Instruction("P", (target,), (theta,)))
        return
    if k == 1:
        _append_cp(builder, controls[0], target, theta)
        return
    if k == 2 and abs(abs(theta) - math.pi) < 1e-12:
        builder.record(Instruction("H", (target,)))
        builder.record(Instruction("CCX", (controls[0], controls[1], target)))
        builder.record(Instruction("H", (target,)))
        return

    prefix = list(controls[:-1])
    last_ctrl = controls[-1]
    half = theta / 2.0
    _append_cp(builder, last_ctrl, target, half)
    _append_mcx_dirty(builder, prefix, last_ctrl, dirty_ancilla=target)
    _append_cp(builder, last_ctrl, target, -half)
    _append_mcx_dirty(builder, prefix, last_ctrl, dirty_ancilla=target)
    _append_mcp(builder, prefix, target, half)


def append_multi_controlled_z(builder: CircuitBuilder, qubits: Sequence[int]) -> None:
    """Appends an exact all-qubit C^{n-1}Z gate on ``qubits`` using standard gates."""
    n = len(qubits)
    if n == 0:
        return
    if n == 1:
        builder.record(Instruction("Z", (qubits[0],)))
        return
    if n == 2:
        builder.record(Instruction("H", (qubits[1],)))
        builder.record(Instruction("CX", (qubits[0], qubits[1])))
        builder.record(Instruction("H", (qubits[1],)))
        return
    if n == 3:
        builder.record(Instruction("H", (qubits[2],)))
        builder.record(Instruction("CCX", (qubits[0], qubits[1], qubits[2])))
        builder.record(Instruction("H", (qubits[2],)))
        return
    _append_mcp(builder, qubits[:-1], qubits[-1], math.pi)


def _append_phase_oracle(
    builder: CircuitBuilder, num_bits: int, target_states: Sequence[int]
) -> None:
    """Appends gate-level Grover phase oracle flipping |t> -> -|t> for each target."""
    all_qubits = list(range(num_bits))
    for t in target_states:
        zero_qubits = [
            q for q in range(num_bits) if ((t >> (num_bits - 1 - q)) & 1) == 0
        ]
        for q in zero_qubits:
            builder.record(Instruction("X", (q,)))
        append_multi_controlled_z(builder, all_qubits)
        for q in reversed(zero_qubits):
            builder.record(Instruction("X", (q,)))


def _append_grover_diffuser(builder: CircuitBuilder, num_bits: int) -> None:
    """Appends gate-level Grover diffusion operator H^n X^n (MCZ) X^n H^n."""
    all_qubits = list(range(num_bits))
    for q in all_qubits:
        builder.record(Instruction("H", (q,)))
    for q in all_qubits:
        builder.record(Instruction("X", (q,)))
    append_multi_controlled_z(builder, all_qubits)
    for q in all_qubits:
        builder.record(Instruction("X", (q,)))
    for q in all_qubits:
        builder.record(Instruction("H", (q,)))


def _build_grover_dag(
    num_bits: int, target_states: Sequence[int], iterations: int
) -> DAGCircuit:
    """Constructs the full gate-level Grover DAGCircuit for ``iterations`` rounds."""
    builder = CircuitBuilder(num_bits)
    for q in range(num_bits):
        builder.record(Instruction("H", (q,)))
    for _ in range(iterations):
        _append_phase_oracle(builder, num_bits, target_states)
        _append_grover_diffuser(builder, num_bits)
    return DAGCircuit.from_builder(builder)


def _parse_single_target(item: int | str, num_bits: int) -> int:
    """Validates and converts an integer or binary bitstring target into an int."""
    n_states = 1 << num_bits
    if isinstance(item, str):
        s = item.strip()
        if len(s) != num_bits or any(ch not in ("0", "1") for ch in s):
            raise ValueError(
                f"Target bitstring {item!r} must be a {num_bits}-bit binary string."
            )
        val = int(s, 2)
    elif isinstance(item, (int, np.integer)) and not isinstance(item, bool):
        val = int(item)
    else:
        raise ValueError(f"Unsupported target specification: {item!r}")

    if val < 0 or val >= n_states:
        raise ValueError(
            f"Target state {val} out of range [0, {n_states - 1}] for {num_bits} qubits."
        )
    return val


def _resolve_targets(
    num_bits: int,
    target: Callable[[int], bool] | int | str | Sequence[int | str] | None,
    targets: Sequence[int | str] | None,
) -> tuple[list[int], int]:
    """Resolves target list and returns (unique_targets, classical_setup_queries)."""
    if targets is not None:
        raw_list = list(targets)
        parsed = list(dict.fromkeys(_parse_single_target(x, num_bits) for x in raw_list))
        return parsed, 0

    if target is None:
        raise ValueError("Either 'target' or 'targets' must be provided.")

    if isinstance(target, (int, np.integer, str)) and not isinstance(target, bool):
        return [_parse_single_target(target, num_bits)], 0

    if isinstance(target, Sequence) and not isinstance(target, (str, bytes)):
        parsed = list(dict.fromkeys(_parse_single_target(x, num_bits) for x in target))
        return parsed, 0

    if callable(target):
        n_states = 1 << num_bits
        found = [x for x in range(n_states) if target(x)]
        return found, n_states

    raise ValueError(f"Invalid target argument: {target!r}")


def _optimal_grover_iterations(n_states: int, n_targets: int) -> int:
    """Exact Boyer-Brassard-Høyer-Tapp optimal Grover iteration count."""
    if n_targets >= n_states:
        return 0
    theta = math.asin(math.sqrt(n_targets / n_states))
    if theta >= math.pi / 4.0:
        # Already >= 50% success probability in uniform superposition; 1 step can overshoot
        k = round(math.pi / (4.0 * theta) - 0.5)
        return max(0, k)
    return max(1, round(math.pi / (4.0 * theta) - 0.5))


def search(
    num_bits: int,
    target: Callable[[int], bool] | int | str | Sequence[int | str] | None = None,
    shots: int = 1024,
    seed: int | None = None,
    *,
    targets: Sequence[int | str] | None = None,
    blind_oracle: bool = False,
) -> Result:
    """Quantum search — finds marked target state(s) via gate-compiled Grover's algorithm.

    Args:
        num_bits: Number of search register qubits (1 to 15).
        target: Target integer, bitstring, sequence of integers/bitstrings, or
            boolean predicate ``Callable[[int], bool]``.
        shots: Measurement shots for the final circuit.
        seed: Random seed for reproducibility.
        targets: Explicit sequence of target integers/bitstrings (zero O(2^n) classical loop).
        blind_oracle: When True, assumes the number of marked states M is unknown and
            executes the BBHT (1998) exponential randomized Grover schedule.

    Returns:
        Result populated with shot counts, real DAG gate_count/depth, statevector,
        plus ``oracle_queries`` and ``classical_setup_queries`` attributes.

    Raises:
        ValueError: If ``num_bits`` is outside [1, 15] or no valid target exists.
    """
    if num_bits < 1 or num_bits > 15:
        raise ValueError(f"num_bits must be in [1,15], given: {num_bits}")

    target_states, classical_setup_queries = _resolve_targets(num_bits, target, targets)
    if not target_states:
        raise ValueError("No state found satisfying the target condition.")

    n_states = 1 << num_bits
    target_set = set(target_states)

    if blind_oracle:
        rng = np.random.default_rng(seed)
        m_bound = 1.0
        lam = 6.0 / 5.0
        max_m = max(1.0, math.sqrt(n_states))
        total_oracle_queries = 0
        winning_j = _optimal_grover_iterations(n_states, len(target_states))
        max_trials = max(16, 6 * num_bits)

        for trial in range(max_trials):
            j = int(rng.integers(0, int(m_bound) + 1))
            dag_try = _build_grover_dag(num_bits, target_states, j)
            sim_try = create_simulator(
                num_bits, seed=None if seed is None else int(seed) + trial + 1
            )
            for op in dag_try.op_nodes():
                sim_try.apply(op.gate_name, op.qubits, op.params)
            total_oracle_queries += j
            sample_one = sim_try.sample(1)
            candidate_bits = next(iter(sample_one.keys()))
            if int(candidate_bits, 2) in target_set:
                winning_j = j
                break
            m_bound = min(lam * m_bound, max_m)

        iterations = winning_j
        oracle_queries = total_oracle_queries
    else:
        iterations = _optimal_grover_iterations(n_states, len(target_states))
        oracle_queries = iterations

    dag = _build_grover_dag(num_bits, target_states, iterations)
    sim = create_simulator(num_bits, seed=seed)
    for op in dag.op_nodes():
        sim.apply(op.gate_name, op.qubits, op.params)

    counts = sim.sample(shots)

    res = Result(
        counts=counts,
        shots=shots,
        num_qubits=num_bits,
        circuit_name=f"search(bits={num_bits}, targets={len(target_states)})",
        gate_count=dag.gate_count(),
        depth=dag.depth(),
        statevector=sim.state,
    )
    res.oracle_queries = oracle_queries  # type: ignore[attr-defined]
    res.classical_setup_queries = classical_setup_queries  # type: ignore[attr-defined]
    res.dag = dag  # type: ignore[attr-defined]
    return res
