"""
quanta.layer3.shor -- Genuine Semiclassical Shor's Factoring Algorithm.

Implements Griffiths-Niu (Phys. Rev. Lett. 76, 3228, 1996) and Beauregard
(Quant. Inf. Comp. 3, 175, 2003) Semiclassical Iterative Quantum Phase Estimation
on ONLY ``n + 1`` qubits (1 recycled control qubit + ``n = ceil(log2(N))`` work
register qubits initialized to ``|1>``), evolving under genuine unitary controlled
modular multiplication operators ``C-U_{a^{2^k} mod N}`` without any classical
array-masking shortcut or silent trial-division fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from quanta.core.circuit import CircuitBuilder
from quanta.core.gates import GATE_REGISTRY, _MatrixGate
from quanta.core.types import Instruction
from quanta.dag.dag_circuit import DAGCircuit
from quanta.layer3.search import _append_mcp
from quanta.simulator.base import SimulatorBackend
from quanta.simulator.factory import create_simulator

__all__ = ["factor", "factor_recursive", "ShorResult"]


@dataclass
class ShorResult:
    """Result of Shor's factoring algorithm.

    Attributes:
        N: Composite integer that was factored.
        factors: Tuple of two non-trivial factors ``(p, q)`` with ``p * q == N``.
        period: Quantum-discovered multiplicative order ``r`` of ``a mod N``.
        attempts: Number of quantum order-finding attempts executed.
        method: ``"quantum"``, ``"classical_shortcut"`` (even / prime power),
            ``"gcd_lucky"``, or ``"classical_fallback"`` (only if explicitly enabled).
    """

    N: int
    factors: tuple[int, int]
    period: int
    attempts: int
    method: str

    def __repr__(self) -> str:
        return (
            f"ShorResult({self.N} = {self.factors[0]} × {self.factors[1]}, "
            f"period={self.period}, method={self.method})"
        )

    def summary(self) -> str:
        lines = [
            "╔══════════════════════════════════════╗",
            f"║  Shor: {self.N} = {self.factors[0]} × {self.factors[1]}"
            + " "
            * max(
                0,
                22
                - len(str(self.N))
                - len(str(self.factors[0]))
                - len(str(self.factors[1])),
            )
            + "║",
            "╠══════════════════════════════════════╣",
            f"║  Period found: {self.period:<22}║",
            f"║  Attempts:     {self.attempts:<22}║",
            f"║  Method:       {self.method:<22}║",
            "╚══════════════════════════════════════╝",
        ]
        return "\n".join(lines)


def _append_controlled_phase(
    builder: CircuitBuilder, control: int, target: int, angle: float
) -> None:
    """Decomposes a 2-qubit Controlled-Phase gate CP(angle) into RZ and CX gates."""
    half = angle / 2.0
    builder.record(Instruction("RZ", (control,), (half,)))
    builder.record(Instruction("RZ", (target,), (half,)))
    builder.record(Instruction("CX", (control, target)))
    builder.record(Instruction("RZ", (target,), (-half,)))
    builder.record(Instruction("CX", (control, target)))


def _build_qft_dag(n_qubits: int) -> DAGCircuit:
    """Builds a Quantum Fourier Transform circuit via the DAG pipeline."""
    builder = CircuitBuilder(n_qubits)
    for j in range(n_qubits):
        builder.record(Instruction("H", (j,)))
        for k in range(j + 1, n_qubits):
            angle = math.pi / (2 ** (k - j))
            _append_controlled_phase(builder, k, j, angle)

    for i in range(n_qubits // 2):
        builder.record(Instruction("SWAP", (i, n_qubits - 1 - i)))

    return DAGCircuit.from_builder(builder)


def _build_inverse_qft_dag(n_qubits: int) -> DAGCircuit:
    """Builds the exact inverse QFT circuit (QFT†) via the DAG pipeline."""
    builder = CircuitBuilder(n_qubits)
    for i in range(n_qubits // 2):
        builder.record(Instruction("SWAP", (i, n_qubits - 1 - i)))

    for j in range(n_qubits - 1, -1, -1):
        for k in range(n_qubits - 1, j, -1):
            angle = -math.pi / (2 ** (k - j))
            _append_controlled_phase(builder, k, j, angle)
        builder.record(Instruction("H", (j,)))

    return DAGCircuit.from_builder(builder)


def _append_controlled_transposition_gates(
    builder: CircuitBuilder, n_work: int, u: int, v: int
) -> None:
    """Decomposes controlled 2-cycle |1, u> <-> |1, v> on (n_work + 1) qubits into elementary gates.

    Uses Gray-like CNOT alignment + zero-ancilla multi-controlled X (`H` + `_append_mcp` + `H`)
    built strictly out of `H, X, CX, CCX, P` in `GATE_REGISTRY`.
    """
    if u == v:
        return
    diff = u ^ v
    # Lowest bit position where u and v differ
    piv_bit = (diff & -diff).bit_length() - 1
    if ((u >> piv_bit) & 1) == 1:
        u, v = v, u
    q_piv = 1 + (n_work - 1 - piv_bit)

    align_qubits = [
        1 + (n_work - 1 - b)
        for b in range(n_work)
        if b != piv_bit and ((diff >> b) & 1) == 1
    ]
    zero_ctrl_qubits = [
        1 + (n_work - 1 - b)
        for b in range(n_work)
        if b != piv_bit and ((u >> b) & 1) == 0
    ]
    controls = [0] + [1 + q for q in range(n_work) if (1 + q) != q_piv]

    for qb in align_qubits:
        builder.record(Instruction("CX", (q_piv, qb)))
    for qb in zero_ctrl_qubits:
        builder.record(Instruction("X", (qb,)))

    builder.record(Instruction("H", (q_piv,)))
    _append_mcp(builder, controls, q_piv, math.pi)
    builder.record(Instruction("H", (q_piv,)))

    for qb in reversed(zero_ctrl_qubits):
        builder.record(Instruction("X", (qb,)))
    for qb in reversed(align_qubits):
        builder.record(Instruction("CX", (q_piv, qb)))


def _apply_controlled_mod_mult(
    sim: SimulatorBackend, mult: int, N: int, n_work: int
) -> None:
    """Applies the exact (n_work + 1)-qubit unitary C-U_{mult mod N} on q[0..n_work].

    - For ``mult == 1``: identity (no-op).
    - For ``n_work <= 5`` (``N <= 32``): decomposes disjoint permutation cycles into
      elementary reversible `H, X, CX, CCX, P` gates via ``DAGCircuit``.
    - For ``6 <= n_work <= 8`` (``32 < N <= 256``): constructs the exact unitary
      permutation matrix ``Gate`` and executes via ``sim.apply`` (restoring ``GATE_REGISTRY``).
    - For ``n_work >= 9`` (``256 < N < 65536``): applies the exact norm-preserving
      bijective unitary permutation operator on the ``q[0] == 1`` subspace in O(2^n) memory.
    """
    mult_mod = mult % N
    if mult_mod == 1 or math.gcd(mult_mod, N) != 1:
        return

    dim_work = 1 << n_work
    if n_work <= 5:
        visited = [False] * N
        builder = CircuitBuilder(n_work + 1)
        for start in range(1, N):
            if visited[start]:
                continue
            cycle: list[int] = []
            curr = start
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                curr = (curr * mult_mod) % N
            if len(cycle) > 1:
                # (c_0 c_1 ... c_{L-1}) = (c_0 c_{L-1}) ... (c_0 c_2)(c_0 c_1)
                # Applied right-to-left: first swap (c_0, c_1), ..., finally (c_0, c_{L-1})
                for idx in range(1, len(cycle)):
                    _append_controlled_transposition_gates(
                        builder, n_work, cycle[0], cycle[idx]
                    )
        dag = DAGCircuit.from_builder(builder)
        for op in dag.op_nodes():
            sim.apply(op.gate_name, op.qubits, op.params)
        return

    if n_work <= 8:
        total_dim = 1 << (n_work + 1)
        u_mat = np.zeros((total_dim, total_dim), dtype=complex)
        for y in range(dim_work):
            u_mat[y, y] = 1.0
            dest = (mult_mod * y) % N if y < N else y
            u_mat[dim_work + dest, dim_work + y] = 1.0
        gate_name = f"_CU_{mult_mod}_{N}_{n_work}"
        GATE_REGISTRY[gate_name] = _MatrixGate(gate_name, u_mat, n_work + 1)
        try:
            sim.apply(gate_name, tuple(range(n_work + 1)))
        finally:
            GATE_REGISTRY.pop(gate_name, None)
        return

    # Sparse unitary permutation representation for 9 <= n_work <= 16 (N < 65536)
    state = sim.state
    new_state = state.copy()
    ys = np.arange(N, dtype=np.int64)
    dests = (mult_mod * ys) % N
    new_state[dim_work + dests] = state[dim_work + ys]
    sim.state = new_state


def _measure_and_reset_control(
    sim: SimulatorBackend, n_work: int, rng: np.random.Generator
) -> int:
    """Projectively measures control qubit q[0] and resets it to |0>."""
    half = 1 << n_work
    state = sim.state
    p1 = float(np.sum(np.abs(state[half:]) ** 2))
    p1 = max(0.0, min(1.0, p1))
    bit = int(rng.random() < p1)

    collapsed = np.zeros_like(state)
    if bit == 1:
        collapsed[half:] = state[half:]
    else:
        collapsed[:half] = state[:half]

    norm = float(np.linalg.norm(collapsed))
    if norm > 1e-15:
        collapsed /= norm
    sim.state = collapsed

    if bit == 1:
        sim.apply("X", (0,))
    return bit


def _single_iqpe_phase(a: int, N: int, rng: np.random.Generator) -> float:
    """Executes one Griffiths-Niu Semiclassical Iterative QPE shot on n + 1 qubits."""
    n_work = max(2, int(math.ceil(math.log2(N))))
    m_bits = 2 * n_work

    sim_seed = int(rng.integers(0, 2**31 - 1))
    sim = create_simulator(n_work + 1, seed=sim_seed)
    # Initialize work register q[1..n_work] to |0...01> = |1>
    sim.apply("X", (n_work,))

    measured_bits: list[int] = []
    for k in range(m_bits):
        exponent = 1 << (m_bits - 1 - k)
        mult = pow(a, exponent, N)

        sim.apply("H", (0,))
        _apply_controlled_mod_mult(sim, mult, N, n_work)

        # Semiclassical inverse-QFT phase feedback from previously measured LSBs
        theta = -math.pi * sum(
            measured_bits[j] / (1 << (k - j)) for j in range(k) if measured_bits[j]
        )
        if abs(theta) > 1e-15:
            sim.apply("P", (0,), (theta,))

        sim.apply("H", (0,))
        bit_k = _measure_and_reset_control(sim, n_work, rng)
        measured_bits.append(bit_k)

    # b_0 is LSB (2^-m), b_{m-1} is MSB (2^-1)
    int_val = sum(bit << idx for idx, bit in enumerate(measured_bits))
    return int_val / float(1 << m_bits)


def _quantum_order_finding(
    a: int, N: int, seed: int | None = None
) -> int:
    """Quantum period-finding via Semiclassical Iterative QPE on n + 1 qubits."""
    a_int = int(a)
    N_int = int(N)
    if N_int >= 65536:
        raise ValueError(
            f"Statevector Semiclassical Shor supports N < 65536, got {N_int}."
        )

    rng = np.random.default_rng(seed)
    denominators: list[int] = []

    for _shot in range(6):
        phase = _single_iqpe_phase(a_int, N_int, rng)
        r_cand = _continued_fraction_period(phase, N_int)
        if r_cand > 1 and pow(a_int, r_cand, N_int) == 1:
            return r_cand

        for mult in (2, 3, 4, 6, 8, 12):
            cand = r_cand * mult
            if 1 < cand < N_int and pow(a_int, cand, N_int) == 1:
                return cand

        if r_cand > 1:
            denominators.append(r_cand)
            if len(denominators) >= 2:
                lcm_r = math.lcm(*denominators)
                if 1 < lcm_r < N_int and pow(a_int, lcm_r, N_int) == 1:
                    return lcm_r

    return denominators[-1] if denominators else 1


def _continued_fraction_period(phase: float, N: int) -> int:
    """Extracts period denominator from phase using exact continued fractions."""
    if phase < 1e-10:
        return 1

    frac = Fraction(phase).limit_denominator(N)
    if frac.denominator > 1:
        return int(frac.denominator)

    for r in range(1, N + 1):
        if abs(phase * r - round(phase * r)) < 0.5 / N:
            return r
    return 1


def _check_prime_power(N: int) -> int | None:
    """Checks if N is a prime power p^k (k >= 2). Returns base factor p or None."""
    max_k = int(math.log2(N)) + 1
    for k in range(2, max_k + 1):
        p = int(round(N ** (1.0 / k)))
        for cand in (p - 1, p, p + 1):
            if cand >= 2 and cand**k == N:
                return cand
    return None


def factor(
    N: int,
    max_attempts: int = 20,
    seed: int | None = None,
    *,
    allow_classical_fallback: bool = False,
) -> ShorResult:
    """Factors a composite integer using Shor's quantum order-finding algorithm.

    Args:
        N: Composite integer to factor (``1 < N < 65536`` supported for statevector simulation).
        max_attempts: Maximum quantum order-finding attempts.
        seed: Random seed for reproducibility.
        allow_classical_fallback: Default ``False``. When ``False``, never falls back to
            classical trial division if quantum attempts fail.

    Returns:
        ShorResult with factor pair, quantum period, attempt count, and method.
    """
    if N < 2:
        raise ValueError(f"N must be > 1, got {N}")

    if _is_prime(N):
        raise ValueError(
            f"{N} is prime — Shor's algorithm factors composite numbers only"
        )

    if N % 2 == 0:
        return ShorResult(N, (2, N // 2), 0, 0, "classical_shortcut")

    base_p = _check_prime_power(N)
    if base_p is not None:
        p_min, p_max = min(base_p, N // base_p), max(base_p, N // base_p)
        return ShorResult(N, (p_min, p_max), 0, 0, "classical_shortcut")

    rng = np.random.default_rng(seed)
    N_int = int(N)

    coprimes = [
        c
        for c in range(2, min(N_int, 512))
        if math.gcd(c, N_int) == 1 and pow(c, 2, N_int) != 1
    ]

    for attempt in range(max_attempts):
        a = int(rng.integers(2, N_int))
        g = math.gcd(a, N_int)
        if g > 1:
            if coprimes:
                a = int(rng.choice(coprimes))
            else:
                f_pair = (min(g, N_int // g), max(g, N_int // g))
                return ShorResult(N_int, f_pair, 0, attempt + 1, "gcd_lucky")

        trial_seed = None if seed is None else int(seed) + attempt * 37
        r = _quantum_order_finding(a, N_int, seed=trial_seed)

        if r > 1 and r % 2 == 0 and pow(a, r, N_int) == 1:
            x = int(pow(a, r // 2, N_int))
            if x != N_int - 1 and x != 1:
                f1 = math.gcd(x + 1, N_int)
                f2 = math.gcd(x - 1, N_int)

                if 1 < f1 < N_int:
                    f_pair = (min(f1, N_int // f1), max(f1, N_int // f1))
                    return ShorResult(N_int, f_pair, r, attempt + 1, "quantum")
                if 1 < f2 < N_int:
                    f_pair = (min(f2, N_int // f2), max(f2, N_int // f2))
                    return ShorResult(N_int, f_pair, r, attempt + 1, "quantum")

    if allow_classical_fallback:
        for d in range(3, math.isqrt(N_int) + 1, 2):
            if N_int % d == 0:
                return ShorResult(
                    N_int, (d, N_int // d), 0, max_attempts, "classical_fallback"
                )

    raise RuntimeError(
        f"Quantum order-finding did not factor {N_int} within {max_attempts} attempts "
        "(allow_classical_fallback=False)."
    )


def _is_prime(n: int) -> bool:
    """Deterministic primality check."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def factor_recursive(
    N: int,
    max_attempts: int = 20,
    seed: int | None = None,
    *,
    allow_classical_fallback: bool = False,
) -> list[int]:
    """Recursively factors N into sorted prime factors (with multiplicity)."""
    if N < 2:
        return []
    if _is_prime(N):
        return [N]

    result = factor(
        N,
        max_attempts=max_attempts,
        seed=seed,
        allow_classical_fallback=allow_classical_fallback,
    )
    p, q = result.factors

    factors: list[int] = []
    for idx, part in enumerate((p, q)):
        if _is_prime(part):
            factors.append(part)
        else:
            next_seed = None if seed is None else int(seed) + (idx + 1) * 101
            factors.extend(
                factor_recursive(
                    part,
                    max_attempts=max_attempts,
                    seed=next_seed,
                    allow_classical_fallback=allow_classical_fallback,
                )
            )

    factors.sort()
    return factors
