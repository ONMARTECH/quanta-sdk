"""
quanta.primitives -- IBM Qiskit V2-compatible Estimator and Sampler.

Zero-dependency implementation of the Qiskit Primitives V2 interface.
Drop-in replacement: same API, but runs on Quanta's simulator.

When a NoiseModel is attached to Estimator:
  - For n <= 11 qubits: uses exact CPTP DensityMatrixSimulator evolution to evaluate
    exact mixed-state expectations Tr(ρ O) and variances Tr(ρ O²) - Tr(ρ O)² without
    single-trajectory collapse!
  - For n > 11 qubits (or custom non-Kraus NoiseChannel subclasses): averages
    max(256, shots) independent Monte Carlo quantum trajectories.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from quanta.core.circuit import CircuitDefinition
from quanta.core.types import QuantaError
from quanta.dag.dag_circuit import DAGCircuit
from quanta.simulator.density_matrix import DensityMatrixSimulator
from quanta.simulator.noise import (
    AmplitudeDamping,
    BitFlip,
    Crosstalk,
    Depolarizing,
    PhaseFlip,
    ReadoutError,
    T2Relaxation,
)
from quanta.simulator.statevector import StateVectorSimulator

if TYPE_CHECKING:
    from quanta.simulator.noise import NoiseModel

__all__ = [
    "Sampler",
    "Estimator",
    "SamplerResult",
    "EstimatorResult",
]


# ── Data classes ──


@dataclass
class SamplerResult:
    """Result from Sampler.run()."""

    quasi_dists: list[dict[str, float]]
    counts: list[dict[str, int]]
    metadata: list[dict] = field(default_factory=list)

    @property
    def result(self) -> dict[str, int]:
        """Shortcut for single-circuit result."""
        return self.counts[0]

    def __repr__(self) -> str:
        n = len(self.counts)
        total = sum(sum(c.values()) for c in self.counts)
        return f"SamplerResult(circuits={n}, total_shots={total})"


@dataclass
class EstimatorResult:
    """Result from Estimator.run()."""

    values: np.ndarray
    variances: np.ndarray
    metadata: list[dict] = field(default_factory=list)

    @property
    def value(self) -> float:
        """Shortcut for single-observable result."""
        return float(self.values[0])

    def __repr__(self) -> str:
        return f"EstimatorResult(values={self.values.round(6).tolist()})"


# ── Pauli & Observable Evaluation Helpers ──

_I2 = np.eye(2, dtype=complex)
_X2 = np.array([[0, 1], [1, 0]], dtype=complex)
_Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = {"I": _I2, "X": _X2, "Y": _Y2, "Z": _Z2}

# 4-point Gauss-Legendre quadrature on [-1, 1] for continuous ZZ Crosstalk channel integration
_GL4_NODES = np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526])
_GL4_WEIGHTS = np.array([0.1739274225687269, 0.3260725774312731, 0.3260725774312731, 0.1739274225687269])


def _pauli_string_to_matrix(pauli_str: str) -> np.ndarray:
    """Converts 'XYZ' → X⊗Y⊗Z (tensor product)."""
    result = _PAULI[pauli_str[0]]
    for ch in pauli_str[1:]:
        result = np.kron(result, _PAULI[ch])
    return result


def _apply_pauli_left(
    pauli_str: str,
    state_or_matrix: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    """Applies n-qubit Pauli string P to the left of (2^n,) vector or (2^n, K) matrix in O(n * 2^n * K)."""
    padded = pauli_str.upper().ljust(n_qubits, "I")
    is_vec = state_or_matrix.ndim == 1
    trailing = () if is_vec else (state_or_matrix.shape[1],)
    tensor = state_or_matrix.reshape((2,) * n_qubits + trailing).copy()

    for q, ch in enumerate(padded):
        if ch == "I":
            continue
        mat = _PAULI[ch]
        contracted = np.tensordot(mat, tensor, axes=([1], [q]))
        tensor = np.moveaxis(contracted, 0, q)

    return tensor.reshape(state_or_matrix.shape)


def _apply_observable_left(
    observable: list[tuple[str, float]],
    state_or_matrix: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    """Computes O @ state_or_matrix without allocating dense 2^n x 2^n Pauli Kronecker matrices."""
    acc = np.zeros_like(state_or_matrix, dtype=np.complex128)
    for pauli_str, coeff in observable:
        acc += float(coeff) * _apply_pauli_left(pauli_str, state_or_matrix, n_qubits)
    return acc


def _compute_expectation(
    statevector: np.ndarray,
    observable: list[tuple[str, float]],
) -> tuple[float, float]:
    """Computes ⟨ψ|O|ψ⟩ and Var(O) = ⟨ψ|O²|ψ⟩ - ⟨ψ|O|ψ⟩² in O(|obs| * n * 2^n)."""
    psi = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    n_qubits = int(round(math.log2(len(psi))))
    o_psi = _apply_observable_left(observable, psi, n_qubits)
    exp_val = float(np.real(np.vdot(psi, o_psi)))
    exp_sq = float(np.real(np.vdot(o_psi, o_psi)))
    variance = max(0.0, exp_sq - exp_val ** 2)
    return exp_val, variance


def _compute_density_expectation(
    rho: np.ndarray,
    observable: list[tuple[str, float]],
    n_qubits: int,
) -> tuple[float, float]:
    """Computes exact mixed-state Tr(ρ O) and Tr(ρ O²) - Tr(ρ O)² from density matrix ρ."""
    o_rho = _apply_observable_left(observable, rho, n_qubits)
    exp_val = float(np.real(np.trace(o_rho)))
    o2_rho = _apply_observable_left(observable, o_rho, n_qubits)
    exp_sq = float(np.real(np.trace(o2_rho)))
    variance = max(0.0, exp_sq - exp_val ** 2)
    return exp_val, variance


def _supports_exact_kraus(noise: NoiseModel) -> bool:
    """Checks whether every channel in ``noise`` has an exact CPTP Kraus representation."""
    supported = (
        Depolarizing,
        BitFlip,
        PhaseFlip,
        AmplitudeDamping,
        T2Relaxation,
        Crosstalk,
        ReadoutError,
    )
    return all(isinstance(ch, supported) for ch in noise.channels)


def _apply_noise_kraus_to_density_sim(
    sim: DensityMatrixSimulator,
    noise: NoiseModel,
    qubits: tuple[int, ...],
) -> None:
    """Applies exact CPTP Kraus operators matching ``NoiseModel.apply_noise`` to ``sim``."""
    n = sim.num_qubits
    for ch in noise.channels:
        if isinstance(ch, ReadoutError):
            # ReadoutError leaves the quantum state unchanged (post-measurement classical channel)
            continue
        for q in qubits:
            if isinstance(ch, Depolarizing):
                p = float(ch.probability)
                if p <= 0.0:
                    continue
                s0 = math.sqrt(max(0.0, 1.0 - p))
                sp = math.sqrt(p / 3.0)
                sim.apply_kraus([s0 * _I2, sp * _X2, sp * _Y2, sp * _Z2], (q,))
            elif isinstance(ch, BitFlip):
                p = float(ch.probability)
                if p <= 0.0:
                    continue
                sim.apply_kraus([math.sqrt(max(0.0, 1.0 - p)) * _I2, math.sqrt(p) * _X2], (q,))
            elif isinstance(ch, PhaseFlip):
                p = float(ch.probability)
                if p <= 0.0:
                    continue
                sim.apply_kraus([math.sqrt(max(0.0, 1.0 - p)) * _I2, math.sqrt(p) * _Z2], (q,))
            elif isinstance(ch, AmplitudeDamping):
                g = float(ch.gamma)
                if g <= 0.0:
                    continue
                k0 = np.array([[1.0, 0.0], [0.0, math.sqrt(max(0.0, 1.0 - g))]], dtype=complex)
                k1 = np.array([[0.0, math.sqrt(g)], [0.0, 0.0]], dtype=complex)
                sim.apply_kraus([k0, k1], (q,))
            elif isinstance(ch, T2Relaxation):
                g = float(ch.gamma)
                if g <= 0.0:
                    continue
                # Uniform phase randomization on |1⟩ damps off-diagonals by (1 - γ)
                # Exact 2-operator CPTP Kraus: sqrt(1 - γ/2) I, sqrt(γ/2) Z
                sim.apply_kraus(
                    [math.sqrt(max(0.0, 1.0 - 0.5 * g)) * _I2, math.sqrt(0.5 * g) * _Z2],
                    (q,),
                )
            elif isinstance(ch, Crosstalk):
                p = float(ch.probability)
                neighbor = q + int(ch.neighbor_offset)
                if p <= 0.0 or neighbor < 0 or neighbor >= n or neighbor == q:
                    continue
                a_mid = 0.055 * math.pi
                a_half = 0.045 * math.pi
                kraus_ops: list[np.ndarray] = [math.sqrt(max(0.0, 1.0 - p)) * np.eye(4, dtype=complex)]
                for node, weight in zip(_GL4_NODES, _GL4_WEIGHTS, strict=True):
                    ang = a_mid + a_half * float(node)
                    u_diag = np.diag([
                        1.0,
                        np.exp(-0.5j * ang),
                        np.exp(-0.5j * ang),
                        np.exp(1.0j * ang),
                    ]).astype(complex)
                    kraus_ops.append(math.sqrt(p * float(weight)) * u_diag)
                sim.apply_kraus(kraus_ops, (q, neighbor))


# ── Sampler ──


class Sampler:
    """Qiskit V2-compatible Sampler primitive."""

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
        """Sample from one or more circuits."""
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
        """Async version of run() for batch parallelism."""
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
    """Qiskit V2-compatible Estimator primitive."""

    _DENSITY_MATRIX_MAX_QUBITS = 11
    _MIN_NOISY_TRAJECTORIES = 256

    def __init__(
        self,
        seed: int | None = None,
        noise: NoiseModel | None = None,
        shots: int | None = None,
    ) -> None:
        self._seed = seed
        self._noise = noise
        self._shots = shots

    def run(
        self,
        circuits: CircuitDefinition | list[CircuitDefinition],
        observables: list[list[tuple[str, float]]],
        *,
        shots: int | None = None,
        **kwargs: float,
    ) -> EstimatorResult:
        """Compute expectation values for circuits with observables.

        Args:
            circuits: Single circuit or list of circuits.
            observables: List of observables, each being a list of
                (pauli_string, coefficient) tuples.
            shots: Optional trajectory count override when noise is enabled for n > 11.
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

        eff_shots = shots if shots is not None else self._shots
        values: list[float] = []
        variances: list[float] = []
        metadata: list[dict] = []

        for circ, obs in zip(circuits, observables, strict=True):
            builder = circ.build(**kwargs)
            dag = DAGCircuit.from_builder(builder)
            ops = dag.op_nodes()

            if self._noise is None:
                simulator = StateVectorSimulator(dag.num_qubits, seed=self._seed)
                for op in ops:
                    simulator.apply(op.gate_name, op.qubits, op.params)
                exp_val, var = _compute_expectation(simulator.state, obs)
                sim_mode = "statevector"
            elif (
                dag.num_qubits <= self._DENSITY_MATRIX_MAX_QUBITS
                and _supports_exact_kraus(self._noise)
            ):
                # Exact CPTP mixed-state density matrix evolution: Tr(ρ O)
                dm_sim = DensityMatrixSimulator(dag.num_qubits, seed=self._seed)
                for op in ops:
                    dm_sim.apply(op.gate_name, op.qubits, op.params)
                    _apply_noise_kraus_to_density_sim(dm_sim, self._noise, op.qubits)
                exp_val, var = _compute_density_expectation(dm_sim.state, obs, dag.num_qubits)
                sim_mode = "density_matrix"
            else:
                # Independent Monte Carlo quantum trajectories for n > 11 (or custom channels)
                n_traj = (
                    max(self._MIN_NOISY_TRAJECTORIES, int(eff_shots))
                    if eff_shots is not None
                    else self._MIN_NOISY_TRAJECTORIES
                )
                rng = np.random.default_rng(self._seed)
                traj_exp_sum = 0.0
                traj_exp_sq_sum = 0.0
                for _ in range(n_traj):
                    sv_sim = StateVectorSimulator(dag.num_qubits)
                    for op in ops:
                        sv_sim.apply(op.gate_name, op.qubits, op.params)
                        sv_sim.apply_noise(self._noise, op.qubits, rng)
                    e_t, v_t = _compute_expectation(sv_sim.state, obs)
                    traj_exp_sum += e_t
                    traj_exp_sq_sum += v_t + (e_t ** 2)

                exp_val = traj_exp_sum / n_traj
                exp_sq = traj_exp_sq_sum / n_traj
                var = max(0.0, exp_sq - exp_val ** 2)
                sim_mode = f"monte_carlo_trajectories({n_traj})"

            values.append(exp_val)
            variances.append(var)
            metadata.append({
                "circuit_name": circ.name,
                "num_qubits": dag.num_qubits,
                "observable_terms": len(obs),
                "simulation_mode": sim_mode,
            })

        return EstimatorResult(
            values=np.array(values, dtype=float),
            variances=np.array(variances, dtype=float),
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
            for c, o in zip(circuits, observables, strict=True)
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
