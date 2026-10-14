"""
quanta.layer3.shor -- Shor's Factoring Algorithm.

Factors integers using quantum period-finding.
The algorithm that started the quantum revolution —
proves quantum computers can break RSA encryption.

Pipeline:
  1. Classical reduction to order-finding
  2. Quantum phase estimation with QFT circuit (real gates via DAG pipeline)
  3. Classical continued fractions to extract period
  4. Classical GCD to extract factors

Limitation: simulator-based, so practical for small numbers (< 2^12).
On real quantum hardware, this would factor much larger numbers.

Example:
    >>> from quanta.layer3.shor import factor
    >>> result = factor(15)
    >>> print(result)  # (3, 5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quanta.core.circuit import CircuitBuilder
from quanta.core.types import Instruction
from quanta.dag.dag_circuit import DAGCircuit
from quanta.simulator.factory import create_simulator

__all__ = ["factor", "factor_recursive", "ShorResult"]


@dataclass
class ShorResult:
    """Result of Shor's factoring algorithm.

    Attributes:
        N: Number that was factored.
        factors: Tuple of two factors.
        period: Quantum-discovered period.
        attempts: Number of attempts needed.
        method: "quantum" or "classical_shortcut".
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
            + " " * max(0, 22 - len(str(self.N))
                       - len(str(self.factors[0]))
                       - len(str(self.factors[1]))) + "║",
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
    """Decomposes a 2-qubit Controlled-Phase gate CP(angle) into standard RZ and CX gates.

    Applies phase exp(i * angle) to |11⟩ (up to global phase) via:
      RZ(angle/2) on control, RZ(angle/2) on target,
      CX(control, target), RZ(-angle/2) on target, CX(control, target).
    """
    half = angle / 2.0
    builder.record(Instruction("RZ", (control,), (half,)))
    builder.record(Instruction("RZ", (target,), (half,)))
    builder.record(Instruction("CX", (control, target)))
    builder.record(Instruction("RZ", (target,), (-half,)))
    builder.record(Instruction("CX", (control, target)))


def _build_qft_dag(n_qubits: int) -> DAGCircuit:
    """Builds a true Quantum Fourier Transform circuit via DAG pipeline.

    QFT applies:
      |j⟩ → (1/√N) Σ_k exp(2πi·jk/N) |k⟩

    Circuit structure for each qubit j (MSB=0 to LSB=n-1):
      H(j), then controlled-phase CP(π/2^(k-j)) between control k and target j
      for k > j, followed by SWAP for bit-reversal.

    This goes through the full CircuitBuilder → DAG pipeline.
    """
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
    """Builds the true inverse QFT circuit (QFT†) via DAG pipeline.

    QFT† is the exact hermitian conjugate of QFT: apply SWAP first, then
    reverse order of controlled-phase rotations (with negated angles) and H.
    """
    builder = CircuitBuilder(n_qubits)

    for i in range(n_qubits // 2):
        builder.record(Instruction("SWAP", (i, n_qubits - 1 - i)))

    for j in range(n_qubits - 1, -1, -1):
        for k in range(n_qubits - 1, j, -1):
            angle = -math.pi / (2 ** (k - j))
            _append_controlled_phase(builder, k, j, angle)
        builder.record(Instruction("H", (j,)))

    return DAGCircuit.from_builder(builder)


def _quantum_order_finding(
    a: int, N: int, seed: int | None = None,
) -> int:
    """Quantum period-finding using QFT† circuit via DAG pipeline.

    Finds the multiplicative order r such that a^r ≡ 1 (mod N).
    Uses quantum superposition, modular exponentiation register measurement
    collapse, and gate-level inverse QFT.

    Steps:
      1. Build counting register in uniform superposition via H gates
      2. Apply modular exponentiation |x⟩|a^x mod N⟩ and measure target register,
         collapsing counting register onto periodic state |ψ⟩ = (1/√M) Σ |x_0 + m·r⟩
      3. Apply inverse QFT circuit (real 2-qubit entangling gates through DAG)
      4. Measure and extract period via continued fractions

    Args:
        a: Base for modular exponentiation.
        N: Modulus.
        seed: Random seed.

    Returns:
        Estimated period r.
    """
    n_count = max(6, 2 * int(math.ceil(math.log2(N))))
    n_count = min(n_count, 12)  # Limit for dense QFT simulation

    dim = 2 ** n_count
    rng = np.random.default_rng(seed)

    # Step 1: Build uniform superposition state |+⟩^⊗n
    sim = create_simulator(n_count, seed=seed)
    for q in range(n_count):
        sim.apply("H", (q,))

    # Step 2: Modular exponentiation f(x) = a^x mod N and target register measurement collapse
    a_int = int(a)
    N_int = int(N)
    remainders = np.empty(dim, dtype=np.int64)
    val = 1
    for x in range(dim):
        remainders[x] = val
        val = (val * a_int) % N_int

    # Measure the target register: pick an observed remainder y_0 = a^{x_0} mod N
    observed_x0 = int(rng.integers(0, dim))
    y0 = remainders[observed_x0]

    # Collapse counting register to superposition of all |x⟩ where a^x ≡ y0 (mod N)
    state = sim.state
    mask = (remainders == y0)
    state[~mask] = 0.0
    norm = np.linalg.norm(state)
    if norm > 1e-12:
        state = state / norm
    sim.state = state

    # Step 3: Apply inverse QFT through DAG pipeline (real 2-qubit entangling gates)
    iqft_dag = _build_inverse_qft_dag(n_count)
    for op in iqft_dag.op_nodes():
        sim.apply(op.gate_name, op.qubits, op.params)

    # Step 4: Sample measurement from QFT† interference peaks
    probs = sim.probabilities()
    probs[0] = 0.0  # Exclude trivial zero-frequency peak
    prob_sum = probs.sum()
    if prob_sum <= 1e-12:
        return 1

    probs = probs / prob_sum
    # Sample top interference peaks to extract period via continued fractions
    for _ in range(4):
        measured = int(rng.choice(dim, p=probs))
        phase_estimate = measured / dim
        period = _continued_fraction_period(phase_estimate, N_int)
        if period > 1 and pow(a_int, period, N_int) == 1:
            return period
        # Check small multiples in case measured peak shared a factor with r
        for mult in (2, 3, 4):
            cand = period * mult
            if 1 < cand < N_int and pow(a_int, cand, N_int) == 1:
                return cand

    return period


def _continued_fraction_period(phase: float, N: int) -> int:
    """Extracts period denominator from phase using exact continued fractions."""
    if phase < 1e-10:
        return 1

    from fractions import Fraction

    frac = Fraction(phase).limit_denominator(N)
    if frac.denominator > 1:
        return int(frac.denominator)

    best_r = 1
    for r in range(1, N + 1):
        if abs(phase * r - round(phase * r)) < 0.5 / N:
            best_r = r
            break

    return best_r if best_r > 0 else 1


def _check_prime_power(N: int) -> int | None:
    """Checks if N is a prime power p^k (k >= 2). Returns base factor p or None."""
    max_k = int(math.log2(N)) + 1
    for k in range(2, max_k + 1):
        p = int(round(N ** (1.0 / k)))
        for cand in (p - 1, p, p + 1):
            if cand >= 2 and cand ** k == N:
                return cand
    return None


def factor(
    N: int,
    max_attempts: int = 20,
    seed: int | None = None,
) -> ShorResult:
    """Factors an integer using Shor's algorithm.

    Args:
        N: Integer to factor (must be composite, > 1).
        max_attempts: Maximum quantum attempts.
        seed: Random seed.

    Returns:
        ShorResult with factors and metadata.

    Example:
        >>> result = factor(15)
        >>> print(result.factors)
        (3, 5)
    """
    if N < 2:
        raise ValueError(f"N must be > 1, got {N}")

    # Reject primes early (Shor is for composite numbers)
    if _is_prime(N):
        raise ValueError(
            f"{N} is prime — Shor's algorithm factors "
            f"composite numbers only"
        )

    # Classical shortcut 1: Even numbers
    if N % 2 == 0:
        return ShorResult(N, (2, N // 2), 0, 0, "classical_shortcut")

    # Classical shortcut 2: True prime powers N = p^k (k >= 2)
    base_p = _check_prime_power(N)
    if base_p is not None:
        return ShorResult(N, (base_p, N // base_p), 0, 0, "classical_shortcut")

    # Quantum order-finding approach
    rng = np.random.default_rng(seed)
    N_int = int(N)

    for attempt in range(max_attempts):
        a = int(rng.integers(2, N_int))

        # Check GCD
        g = math.gcd(a, N_int)
        if g > 1:
            # Prefer demonstrating quantum period finding on early attempts if possible,
            # unless no coprime candidates remain
            coprimes = [c for c in range(2, N_int) if math.gcd(c, N_int) == 1 and pow(c, 2, N_int) != 1]
            if coprimes:
                a = int(rng.choice(coprimes))
            else:
                return ShorResult(N, (g, N // g), 0, attempt + 1, "gcd_lucky")

        # Quantum period finding (uses QFT† circuit via DAG pipeline)
        trial_seed = None if seed is None else int(seed) + attempt * 17
        r = _quantum_order_finding(a, N_int, seed=trial_seed)

        if r % 2 == 0:
            x = int(pow(a, r // 2, N_int))
            if x != N_int - 1 and x != 1:
                f1 = math.gcd(x + 1, N_int)
                f2 = math.gcd(x - 1, N_int)

                if 1 < f1 < N:
                    f_pair = (min(f1, N // f1), max(f1, N // f1))
                    return ShorResult(N, f_pair, r, attempt + 1, "quantum")
                if 1 < f2 < N:
                    f_pair = (min(f2, N // f2), max(f2, N // f2))
                    return ShorResult(N, f_pair, r, attempt + 1, "quantum")

    # Fallback: brute force if quantum attempts exhausted
    for i in range(2, int(math.sqrt(N)) + 1):
        if N % i == 0:
            return ShorResult(N, (i, N // i), 0, max_attempts, "classical_fallback")

    raise ValueError(f"Could not factor {N} — it may be prime")


def _is_prime(n: int) -> bool:
    """Simple primality test."""
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
) -> list[int]:
    """Recursively factor N into all prime factors.

    Unlike `factor()` which returns a single factor pair,
    this decomposes composite cofactors until all factors are prime.

    Args:
        N: Integer to factor.
        max_attempts: Max quantum attempts per factoring step.
        seed: Random seed.

    Returns:
        Sorted list of prime factors (with multiplicity).

    Example:
        >>> factor_recursive(9999)  # 3 × 3 × 11 × 101
        [3, 3, 11, 101]
    """
    if N < 2:
        return []
    if _is_prime(N):
        return [N]

    result = factor(N, max_attempts=max_attempts, seed=seed)
    p, q = result.factors

    # Recursively factor each part
    factors: list[int] = []
    for part in (p, q):
        if _is_prime(part):
            factors.append(part)
        else:
            factors.extend(factor_recursive(part, max_attempts, seed))

    factors.sort()
    return factors
