"""
quanta.primitives -- IBM Qiskit V2-compatible Estimator and Sampler.

Zero-dependency implementation of the Qiskit Primitives V2 interface.
Drop-in replacement: same API, but runs on Quanta's simulator.

Example:
    >>> from quanta.primitives import Sampler, Estimator
    >>> result = Sampler().run(bell, shots=4096)
    >>> energy = Estimator().run(bell, observable=[("ZZ", 1.0)])
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from quanta.core.circuit import CircuitDefinition
from quanta.core.types import QuantaError
from quanta.dag.dag_circuit import DAGCircuit
from quanta.result import Result
from quanta.simulator.statevector import StateVectorSimulator

if TYPE_CHECKING:
    from quanta.simulator.noise import NoiseModel

__all__ = [
    "Sampler", "Estimator",
    "SamplerResult", "EstimatorResult",
]


# ── Data classes ──


@dataclass
class SamplerResult:
    """Result from Sampler.run().

    Attributes:
        quasi_dists: Quasi-probability distributions per circuit.
        counts: Raw measurement counts per circuit.
        metadata: Execution metadata (shots, seed, etc.).
    """

    quasi_dists: list[dict[str, float]]
    counts: list[dict[str, int]]
    metadata: list[dict] = field(default_factory=list)

    @property
    def result(self) -> dict[str, int]:
        """Shortcut for single-circuit result."""
        return self.counts[0]

    def __repr__(self) -> str:
        n = len(self.counts)
        return f"SamplerResult(circuits={n}, total_shots={sum(sum(c.values()) for c in self.counts)})"


@dataclass
class EstimatorResult:
    """Result from Estimator.run().

    Attributes:
        values: Expectation values per circuit-observable pair.
        variances: Variance of each expectation value estimate.
        metadata: Execution metadata.
    """

    values: np.ndarray
    variances: np.ndarray
    metadata: list[dict] = field(default_factory=list)

    @property
    def value(self) -> float:
        """Shortcut for single-observable result."""
        return float(self.values[0])

    def __repr__(self) -> str:
        return f"EstimatorResult(values={self.values.round(6).tolist()})"


# ── Pauli → matrix mapping ──

_I2 = np.eye(2, dtype=complex)
_PAULI = {
    "I": _I2,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_string_to_matrix(pauli_str: str) -> np.ndarray:
    """Converts 'XYZ' → X⊗Y⊗Z (tensor product)."""
    result = _PAULI[pauli_str[0]]
    for ch in pauli_str[1:]:
        result = np.kron(result, _PAULI[ch])
    return result


def _compute_expectation(
    statevector: np.ndarray,
    observable: list[tuple[str, float]],
) -> tuple[float, float]:
    """Computes ⟨ψ|O|ψ⟩ and Var(O) for a Pauli observable.

    Args:
        statevector: Quantum state vector.
        observable: List of (pauli_string, coefficient) terms.
            Example: [("ZZ", -1.0), ("XX", 0.5)]

    Returns:
        (expectation_value, variance)
    """
    psi = statevector.astype(complex)
    n_qubits = int(np.log2(len(psi)))

    # Build full observable matrix: O = Σ c_i * P_i
    dim = 2 ** n_qubits
    obs_matrix = np.zeros((dim, dim), dtype=complex)

    for pauli_str, coeff in observable:
        # Pad with I if shorter than n_qubits
        padded = pauli_str.ljust(n_qubits, "I")
        obs_matrix += coeff * _pauli_string_to_matrix(padded)

    # ⟨ψ|O|ψ⟩
    exp_val = float(np.real(psi.conj() @ obs_matrix @ psi))

    # Var(O) = ⟨ψ|O²|ψ⟩ - ⟨ψ|O|ψ⟩²
    exp_sq = float(np.real(psi.conj() @ obs_matrix @ obs_matrix @ psi))
    variance = max(0.0, exp_sq - exp_val ** 2)

    return exp_val, variance


# ── Sampler ──


class Sampler:
    """Qiskit V2-compatible Sampler primitive.

    Samples measurement outcomes from quantum circuits.

    Example:
        >>> from quanta import circuit, H, CX, measure
        >>> from quanta.primitives import Sampler
        >>>
        >>> @circuit(qubits=2)
        ... def bell(q):
        ...     H(q[0])
        ...     CX(q[0], q[1])
        ...     return measure(q)
        >>>
        >>> sampler = Sampler(seed=42)
        >>> result = sampler.run(bell, shots=4096)
        >>> print(result.counts[0])  # {'00': ~2048, '11': ~2048}
    """

    def __init__(
        self,
        seed: int | None = None,
        noise: NoiseModel | None = None,
    ) -> None:
        self._seed = seed
        self._noise = noise

    def run(
        self,
        circuits: CircuitDefinition | list[CircuitDefinition],
        shots: int = 1024,
    ) -> SamplerResult:
        """Sample from one or more circuits.

        Args:
            circuits: Single circuit or list of circuits.
            shots: Number of shots per circuit.

        Returns:
            SamplerResult with counts and quasi-distributions.
        """
        if isinstance(circuits, CircuitDefinition):
            circuits = [circuits]

        all_counts: list[dict[str, int]] = []
        all_dists: list[dict[str, float]] = []
        all_meta: list[dict] = []

        for circ in circuits:
            from quanta.runner import run
            result = run(circ, shots=shots, seed=self._seed, noise=self._noise)
            all_counts.append(result.counts)
            all_dists.append(result.probabilities)
            all_meta.append({
                "shots": shots,
                "circuit_name": circ.name,
                "num_qubits": result.num_qubits,
            })

        return SamplerResult(
            quasi_dists=all_dists,
            counts=all_counts,
            metadata=all_meta,
        )

    async def run_async(
        self,
        circuits: CircuitDefinition | list[CircuitDefinition],
        shots: int = 1024,
    ) -> SamplerResult:
        """Async version of run() for batch parallelism.

        Example:
            >>> results = await sampler.run_async([bell, ghz], shots=4096)
        """
        if isinstance(circuits, CircuitDefinition):
            circuits = [circuits]

        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                None, lambda c=c: self.run(c, shots=shots)
            )
            for c in circuits
        ]
        results = await asyncio.gather(*tasks)

        # Merge results
        all_counts = [r.counts[0] for r in results]
        all_dists = [r.quasi_dists[0] for r in results]
        all_meta = [r.metadata[0] for r in results]

        return SamplerResult(
            quasi_dists=all_dists,
            counts=all_counts,
            metadata=all_meta,
        )


# ── Estimator ──


class Estimator:
    """Qiskit V2-compatible Estimator primitive.

    Computes expectation values ⟨ψ|O|ψ⟩ for Pauli observables.

    Example:
        >>> from quanta import circuit, H, CX, measure
        >>> from quanta.primitives import Estimator
        >>>
        >>> @circuit(qubits=2)
        ... def bell(q):
        ...     H(q[0])
        ...     CX(q[0], q[1])
        ...     return measure(q)
        >>>
        >>> estimator = Estimator()
        >>> result = estimator.run(
        ...     bell,
        ...     observables=[[("ZZ", 1.0), ("XX", 0.5)]],
        ... )
        >>> print(result.value)  # ⟨ψ|ZZ+0.5XX|ψ⟩
    """

    def __init__(
        self,
        seed: int | None = None,
        noise: NoiseModel | None = None,
    ) -> None:
        self._seed = seed
        self._noise = noise

    def run(
        self,
        circuits: CircuitDefinition | list[CircuitDefinition],
        observables: list[list[tuple[str, float]]],
        **kwargs: float,
    ) -> EstimatorResult:
        """Compute expectation values for circuits with observables.

        Args:
            circuits: Single circuit or list of circuits.
            observables: List of observables, each being a list of
                (pauli_string, coefficient) tuples.
                Example: [[("ZZ", 1.0), ("XX", 0.5)]]
            **kwargs: Circuit parameters forwarded to build().

        Returns:
            EstimatorResult with expectation values and variances.
        """
        if isinstance(circuits, CircuitDefinition):
            circuits = [circuits]

        if len(circuits) == 1 and len(observables) > 1:
            circuits = circuits * len(observables)

        if len(circuits) != len(observables):
            raise QuantaError(
                f"Circuits ({len(circuits)}) and observables ({len(observables)}) "
                f"must have same length, or circuits must be length 1."
            )

        values = []
        variances = []
        metadata = []

        for circ, obs in zip(circuits, observables):
            # Simulate to get statevector
            builder = circ.build(**kwargs)
            dag = DAGCircuit.from_builder(builder)
            simulator = StateVectorSimulator(dag.num_qubits, seed=self._seed)
            rng = np.random.default_rng(self._seed)

            for op in dag.op_nodes():
                simulator.apply(op.gate_name, op.qubits, op.params)
                if self._noise is not None:
                    simulator.apply_noise(self._noise, op.qubits, rng)

            exp_val, var = _compute_expectation(simulator.state, obs)
            values.append(exp_val)
            variances.append(var)
            metadata.append({
                "circuit_name": circ.name,
                "num_qubits": dag.num_qubits,
                "observable_terms": len(obs),
            })

        return EstimatorResult(
            values=np.array(values),
            variances=np.array(variances),
            metadata=metadata,
        )

    async def run_async(
        self,
        circuits: CircuitDefinition | list[CircuitDefinition],
        observables: list[list[tuple[str, float]]],
    ) -> EstimatorResult:
        """Async version of run() for batch parallelism."""
        if isinstance(circuits, CircuitDefinition):
            circuits = [circuits]

        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                None,
                lambda c=c, o=o: self.run(c, [o]),
            )
            for c, o in zip(circuits, observables)
        ]
        results = await asyncio.gather(*tasks)

        all_values = np.array([r.values[0] for r in results])
        all_vars = np.array([r.variances[0] for r in results])
        all_meta = [r.metadata[0] for r in results]

        return EstimatorResult(
            values=all_values,
            variances=all_vars,
            metadata=all_meta,
        )
