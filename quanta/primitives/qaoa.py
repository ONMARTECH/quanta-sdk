"""quanta.primitives.qaoa — First-class QAOA optimization primitive.

Provides high-performance dual-mode execution (fast vectorized statevector simulation
and native gate-level circuit synthesis) with multi-layer ansatz, warm-starting,
biased mixers, and classical parameter optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quanta.core.circuit import CircuitDefinition, circuit
from quanta.core.gates import RX, RY, RZ, RZZ, H
from quanta.core.measure import measure
from quanta.result import Result
from quanta.simulator.statevector import StateVectorSimulator


@dataclass
class QAOAResult:
    """Result of QAOA optimization execution.

    Attributes:
        best_bitstring: Bitstring achieving minimum objective cost.
        best_cost: Objective cost value for best_bitstring.
        optimal_gamma: Optimized cost layer rotation angles of length p.
        optimal_beta: Optimized mixer layer rotation angles of length p.
        cost_history: Optimization trajectory tracking expectation values.
        counts: Measurement count distribution from optimal statevector.
        probabilities: Quasi-probability distribution of basis states.
        num_qubits: Total number of qubits N.
        num_layers: Number of alternating QAOA layers p.
        optimal_statevector: Final statevector |psi(gamma*, beta*)>.
        all_solutions: List of (bitstring, cost, probability) sorted by cost.
        metadata: Execution and optimization metadata dictionary.
        quantum_result: Standard Quanta Result container.
    """

    best_bitstring: str
    best_cost: float
    optimal_gamma: np.ndarray
    optimal_beta: np.ndarray
    cost_history: list[float]
    counts: dict[str, int]
    probabilities: dict[str, float]
    num_qubits: int
    num_layers: int
    optimal_statevector: np.ndarray | None = None
    all_solutions: list[tuple[str, float, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    quantum_result: Result | None = None

    def summary(self) -> str:
        """Returns a formatted human-readable summary table."""
        lines = [
            "╔═══════════════════ QAOA Optimization Result ═══════════════════",
            f"║ Qubits: {self.num_qubits} | Layers: {self.num_layers}",
            f"║ Best Bitstring: |{self.best_bitstring}⟩",
            f"║ Best Cost:      {self.best_cost:.6f}",
            f"║ Optimal Gamma:  {np.round(self.optimal_gamma, 4).tolist()}",
            f"║ Optimal Beta:   {np.round(self.optimal_beta, 4).tolist()}",
            "╠────────────────────── Top Candidate States ─────────────────────",
        ]
        for bits, cost, prob in self.all_solutions[:5]:
            bar = "█" * int(min(30, prob * 60))
            lines.append(f"║ |{bits}⟩  cost={cost:.4f}  P={prob:.4f}  {bar}")
        lines.append("╚═════════════════════════════════════════════════════════════════")
        return "\n".join(lines)

    def sample(self, shots: int = 1024, seed: int | None = None) -> dict[str, int]:
        """Resamples measurement bitstrings from the optimal statevector."""
        if self.optimal_statevector is None:
            return dict(self.counts)

        probs = np.abs(self.optimal_statevector) ** 2
        p_sum = np.sum(probs)
        if p_sum > 0:
            probs = probs / p_sum

        rng = np.random.default_rng(seed)
        indices = rng.choice(len(probs), size=shots, p=probs)
        unique, unique_counts = np.unique(indices, return_counts=True)
        fmt = f"0{self.num_qubits}b"
        return {
            format(idx, fmt): int(cnt)
            for idx, cnt in zip(unique, unique_counts, strict=True)
        }


class QAOA:
    """Qiskit V2-compatible QAOA primitive with dual simulation and circuit modes.

    Supports:
      - Hamiltonian inputs: CVPHamiltonian, Pauli string list, QUBO matrix, or diagonal array
      - Fast vectorized statevector simulation for variational loops (< 5 ms per step)
      - Gate-level CircuitDefinition generation with native RZZ/RZ/RX/RY gates
      - Multi-layer ansatz (p >= 1)
      - SWR warm-starting initial states |psi_warm>
      - Custom and biased non-commuting mixers: exp(-i beta (X cos(theta) + Z sin(theta)))
      - Per-qubit transverse mixer quenching (Quantum Zeno subspace freezing)
    """

    def __init__(
        self,
        cost_hamiltonian: Any | None = None,
        num_qubits: int | None = None,
        p: int = 1,
        mixer_type: str = "x",
        warm_start_angles: np.ndarray | list[float] | None = None,
        mixer_weights: np.ndarray | list[float] | None = None,
        initial_state: np.ndarray | None = None,
    ) -> None:
        """Initializes the QAOA primitive.

        Args:
            cost_hamiltonian: Problem cost representation (CVPHamiltonian, Pauli list,
                QUBO matrix, or diagonal energy vector).
            num_qubits: Total number of qubits. Inferred if cost_hamiltonian is provided.
            p: Number of alternating QAOA layers (p >= 1).
            mixer_type: Type of mixer Hamiltonian: 'x' (standard transverse),
                or 'biased' (non-commuting biased mixer for warm-starting).
            warm_start_angles: Optional array of single-qubit RY rotation angles theta_i
                for each qubit i in [0, N-1].
            mixer_weights: Optional per-qubit weights w_i in [0, 1] scaling mixer strength,
                e.g. (1 - P_zeno) for Zeno subspace projection.
            initial_state: Optional pre-constructed initial statevector of shape (2^N,).
        """
        if p < 1:
            raise ValueError(f"Number of layers p must be >= 1, got {p}.")
        self.p: int = p
        self.mixer_type: str = mixer_type.lower()
        self.initial_state: np.ndarray | None = (
            np.asarray(initial_state, dtype=complex) if initial_state is not None else None
        )

        self._cvp_instance: Any | None = None
        self._pauli_list: list[tuple[str, float]] | None = None
        self._ising_j: dict[tuple[int, int], float] = {}
        self._ising_h: dict[int, float] = {}
        self._ising_offset: float = 0.0
        self._diagonal_costs: np.ndarray = np.array([], dtype=np.float64)

        if cost_hamiltonian is not None:
            self._parse_cost_hamiltonian(cost_hamiltonian, num_qubits)
        else:
            if num_qubits is None:
                raise ValueError("num_qubits must be specified if cost_hamiltonian is None.")
            self.num_qubits: int = num_qubits

        # Initialize warm-start angles and mixer weights
        if warm_start_angles is not None:
            self.warm_start_angles: np.ndarray | None = np.asarray(
                warm_start_angles, dtype=np.float64
            ).reshape(-1)
            if len(self.warm_start_angles) != self.num_qubits:
                raise ValueError(
                    f"warm_start_angles length {len(self.warm_start_angles)} "
                    f"must match num_qubits {self.num_qubits}."
                )
        else:
            self.warm_start_angles = None

        if mixer_weights is not None:
            self.mixer_weights: np.ndarray | None = np.asarray(
                mixer_weights, dtype=np.float64
            ).reshape(-1)
            if len(self.mixer_weights) != self.num_qubits:
                raise ValueError(
                    f"mixer_weights length {len(self.mixer_weights)} "
                    f"must match num_qubits {self.num_qubits}."
                )
        else:
            self.mixer_weights = None

    def _parse_cost_hamiltonian(self, obj: Any, num_qubits: int | None) -> None:
        """Parses and normalizes heterogeneous cost Hamiltonian representations."""
        # Check if callable first (before np.asarray)
        if callable(obj) and not hasattr(obj, "qubo_matrix"):
            if num_qubits is None:
                raise ValueError("num_qubits must be specified when cost is a callable.")
            self.num_qubits = num_qubits
            dim = 1 << num_qubits
            self._diagonal_costs = np.array([float(obj(i)) for i in range(dim)], dtype=np.float64)
            return

        # Check if CVPHamiltonian instance
        if hasattr(obj, "qubo_matrix") and hasattr(obj, "to_ising"):
            self._cvp_instance = obj
            self.num_qubits = int(obj.num_qubits)
            self._diagonal_costs = obj.diagonal_energy_vector().copy()
            j_dict, h_dict, offset = obj.to_ising()
            self._ising_j = dict(j_dict)
            self._ising_h = dict(h_dict)
            self._ising_offset = float(offset)
            self._pauli_list = obj.to_pauli_list()
            return

        # Check if list of Pauli strings: [("ZZ", 1.0), ("ZI", 0.5), ...]
        if isinstance(obj, list) and obj and isinstance(obj[0], tuple):
            self._pauli_list = list(obj)
            n_q = len(obj[0][0])
            self.num_qubits = n_q

            # Parse Ising couplings from Pauli strings
            self._ising_j = {}
            self._ising_h = {}
            self._ising_offset = 0.0

            for pauli_str, coeff in obj:
                coeff = float(coeff)
                invalid_chars = [ch for ch in pauli_str if ch not in ("Z", "I")]
                if invalid_chars:
                    raise ValueError(
                        f"Cost Hamiltonian must be diagonal (only 'Z' and 'I' allowed, got '{pauli_str}')."
                    )
                z_indices = [idx for idx, ch in enumerate(pauli_str) if ch == "Z"]
                if len(z_indices) > 2:
                    raise ValueError(
                        f"Only up to 2-body Z interactions supported in QAOA ansatz, got {len(z_indices)} in '{pauli_str}'."
                    )
                if len(z_indices) == 2:
                    p, q = min(z_indices), max(z_indices)
                    self._ising_j[(p, q)] = self._ising_j.get((p, q), 0.0) + coeff
                elif len(z_indices) == 1:
                    p = z_indices[0]
                    self._ising_h[p] = self._ising_h.get(p, 0.0) + coeff
                elif len(z_indices) == 0:
                    self._ising_offset += coeff

            # Construct diagonal energy vector
            dim = 1 << n_q
            basis_indices = np.arange(dim, dtype=np.int64)[:, None]
            bit_shifts = np.arange(n_q - 1, -1, -1, dtype=np.int64)[None, :]
            # Spins Z_p = 1 - 2 * bit_p
            bits = (basis_indices >> bit_shifts) & 1
            spins = 1.0 - 2.0 * bits.astype(np.float64)

            energies = np.full(dim, self._ising_offset, dtype=np.float64)
            for (i, j), j_val in self._ising_j.items():
                energies += j_val * (spins[:, i] * spins[:, j])
            for i, h_val in self._ising_h.items():
                energies += h_val * spins[:, i]

            self._diagonal_costs = energies
            return

        # Check if 2D numpy array: QUBO matrix Q
        arr = np.asarray(obj, dtype=np.float64)
        if arr.ndim == 2:
            n_q = arr.shape[0]
            if arr.shape[1] != n_q:
                raise ValueError("QUBO matrix must be square.")
            self.num_qubits = n_q

            # Convert QUBO to Ising
            self._ising_j = {}
            self._ising_h = {}
            for p in range(n_q):
                for q in range(p + 1, n_q):
                    val = 0.25 * (arr[p, q] + arr[q, p])
                    if abs(val) > 1e-15:
                        self._ising_j[(p, q)] = float(val)

            for p in range(n_q):
                sum_cross = sum(
                    arr[min(p, other), max(p, other)] + arr[max(p, other), min(p, other)]
                    for other in range(n_q)
                    if other != p
                )
                self._ising_h[p] = float(-0.5 * arr[p, p] - 0.25 * sum_cross)

            self._ising_offset = float(
                0.5 * np.trace(arr) + 0.25 * np.sum(np.triu(arr + arr.T, k=1))
            )

            # Precompute diagonal costs
            dim = 1 << n_q
            basis_indices = np.arange(dim, dtype=np.int64)[:, None]
            bit_shifts = np.arange(n_q - 1, -1, -1, dtype=np.int64)[None, :]
            bits = ((basis_indices >> bit_shifts) & 1).astype(np.float64)
            self._diagonal_costs = np.sum(bits * (bits @ arr.T), axis=1)
            return

        # Check if 1D numpy array: diagonal cost vector
        if arr.ndim == 1:
            dim = len(arr)
            n_q = int(np.round(np.log2(dim)))
            if (1 << n_q) != dim:
                raise ValueError(f"Diagonal cost vector length {dim} must be power of 2.")
            self.num_qubits = n_q
            self._diagonal_costs = arr.copy()
            return

        raise TypeError(f"Unsupported cost Hamiltonian type: {type(obj)}")

    def simulate_statevector(
        self,
        gamma: np.ndarray | list[float],
        beta: np.ndarray | list[float],
        seed: int | None = None,
    ) -> np.ndarray:
        """Executes fast vectorized statevector evolution for angles (gamma, beta).

        Args:
            gamma: Cost layer angles of length p.
            beta: Mixer layer angles of length p.
            seed: Optional simulator random seed.

        Returns:
            Normalized complex statevector of length 2^num_qubits.
        """
        n = self.num_qubits
        sim = StateVectorSimulator(n, seed=seed)

        # 1. State preparation
        if self.warm_start_angles is not None:
            for q in range(n):
                sim.apply("RY", (q,), (float(self.warm_start_angles[q]),))
        elif self.initial_state is not None:
            sim.state = self.initial_state.copy()
        else:
            for q in range(n):
                sim.apply("H", (q,))

        # 2. Alternating layers
        p_layers = min(len(gamma), len(beta))
        weights = self.mixer_weights if self.mixer_weights is not None else np.ones(n)

        for l_idx in range(p_layers):
            g = float(gamma[l_idx])
            b = float(beta[l_idx])

            # Cost layer: diagonal phase application
            state = sim.state
            state *= np.exp(-1j * g * self._diagonal_costs)
            sim.state = state

            # Mixer layer
            if self.mixer_type == "biased" and self.warm_start_angles is not None:
                # Egger-style biased mixer: RY(-theta) -> RX(2 * beta * w) -> RY(theta)
                for q in range(n):
                    w = float(weights[q])
                    if abs(w) > 1e-12:
                        theta = float(self.warm_start_angles[q])
                        sim.apply("RY", (q,), (-theta,))
                        sim.apply("RX", (q,), (2.0 * b * w,))
                        sim.apply("RY", (q,), (theta,))
            else:
                # Standard transverse mixer with per-qubit weights
                for q in range(n):
                    w = float(weights[q])
                    if abs(w) > 1e-12:
                        sim.apply("RX", (q,), (2.0 * b * w,))

        return sim.state

    def compute_expectation(
        self,
        gamma: np.ndarray | list[float],
        beta: np.ndarray | list[float],
    ) -> float:
        """Computes the energy expectation value ⟨ψ(γ, β)| H_C |ψ(γ, β)⟩."""
        state = self.simulate_statevector(gamma, beta)
        probs = np.abs(state) ** 2
        return float(np.sum(probs * self._diagonal_costs))

    def to_circuit(
        self,
        gamma: np.ndarray | list[float],
        beta: np.ndarray | list[float],
    ) -> CircuitDefinition:
        """Constructs a native Quanta CircuitDefinition parameterized with (gamma, beta).

        Synthesizes standard two-qubit RZZ couplings, single-qubit RZ fields, and
        transverse/biased mixer rotations using Quanta native gates.

        Args:
            gamma: Cost layer angles of length p.
            beta: Mixer layer angles of length p.

        Returns:
            CircuitDefinition instance ready for DAGCircuit compilation or QPU execution.
        """
        n = self.num_qubits
        p_layers = min(len(gamma), len(beta))
        gamma_vals = [float(g) for g in gamma[:p_layers]]
        beta_vals = [float(b) for b in beta[:p_layers]]
        weights = self.mixer_weights if self.mixer_weights is not None else np.ones(n)
        warm_angles = (
            [float(t) for t in self.warm_start_angles]
            if self.warm_start_angles is not None
            else None
        )
        is_biased = self.mixer_type == "biased" and warm_angles is not None
        j_couplings = list(self._ising_j.items())
        h_fields = list(self._ising_h.items())

        @circuit(qubits=n)
        def qaoa_circuit(q: Any) -> Any:
            # 1. Initial State Preparation
            if warm_angles is not None:
                for idx in range(n):
                    RY(warm_angles[idx])(q[idx])
            else:
                for idx in range(n):
                    H(q[idx])

            # 2. Alternating QAOA Layers
            for layer in range(p_layers):
                g = gamma_vals[layer]
                b = beta_vals[layer]

                # Cost layer: RZZ for two-body couplings J_ij
                for (i, j), j_val in j_couplings:
                    if abs(j_val) > 1e-15:
                        RZZ(2.0 * g * j_val)(q[i], q[j])

                # Cost layer: RZ for single-qubit fields h_i
                for i, h_val in h_fields:
                    if abs(h_val) > 1e-15:
                        RZ(2.0 * g * h_val)(q[i])

                # Mixer layer
                if is_biased and warm_angles is not None:
                    for i in range(n):
                        w = float(weights[i])
                        if abs(w) > 1e-12:
                            theta = warm_angles[i]
                            RY(-theta)(q[i])
                            RX(2.0 * b * w)(q[i])
                            RY(theta)(q[i])
                else:
                    for i in range(n):
                        w = float(weights[i])
                        if abs(w) > 1e-12:
                            RX(2.0 * b * w)(q[i])

            return measure(q)

        return qaoa_circuit

    def run(
        self,
        cost_hamiltonian: Any | None = None,
        initial_params: np.ndarray | None = None,
        optimizer: str = "COBYLA",
        max_iter: int = 100,
        shots: int = 1024,
        seed: int | None = None,
        num_starts: int = 5,
    ) -> QAOAResult:
        """Executes the classical-quantum QAOA optimization loop.

        Args:
            cost_hamiltonian: Optional dynamic override of the problem cost.
            initial_params: Optional starting parameters [gamma_1..gamma_p, beta_1..beta_p].
            optimizer: Classical optimizer name: 'COBYLA', 'Nelder-Mead', or 'random'.
            max_iter: Maximum iterations per classical optimization restart.
            shots: Number of measurement samples to draw from the optimal statevector.
            seed: Random seed for reproducibility.
            num_starts: Number of random restarts for multi-start parameter training.

        Returns:
            QAOAResult dataclass containing optimal bitstrings, angles, and telemetry.
        """
        if cost_hamiltonian is not None:
            self._parse_cost_hamiltonian(cost_hamiltonian, None)

        if initial_params is not None:
            arr_init = np.asarray(initial_params, dtype=np.float64).reshape(-1)
            if len(arr_init) != 2 * self.p:
                raise ValueError(
                    f"initial_params length {len(arr_init)} must match 2*p = {2 * self.p}."
                )

        if len(self._diagonal_costs) == 0:
            raise ValueError("No cost Hamiltonian configured for QAOA execution.")

        rng = np.random.default_rng(seed)
        p = self.p
        cost_history: list[float] = []

        def objective(params: np.ndarray) -> float:
            gamma = params[:p]
            beta = params[p:]
            val = self.compute_expectation(gamma, beta)
            cost_history.append(val)
            return val

        # Natural spectral scaling for cost layer rotation angles
        delta_e = float(np.max(self._diagonal_costs) - np.min(self._diagonal_costs))
        gamma_scale = (
            min(2.0 * np.pi, 2.0 * np.pi / delta_e)
            if delta_e > 1e-6
            else 2.0 * np.pi
        )
        rhobeg = float(min(0.5, max(0.01, gamma_scale / 2.0)))

        best_params: np.ndarray | None = None
        best_expectation: float = float("inf")

        # Multi-start parameter training
        try:
            from scipy.optimize import minimize

            starts = 1 if initial_params is not None else max(1, num_starts)
            for start_idx in range(starts):
                if initial_params is not None and start_idx == 0:
                    x0 = np.asarray(initial_params, dtype=np.float64).reshape(-1)
                else:
                    gammas_init = rng.uniform(0.0, gamma_scale, size=p)
                    betas_init = rng.uniform(0.0, np.pi, size=p)
                    x0 = np.concatenate([gammas_init, betas_init])

                res = minimize(
                    objective,
                    x0,
                    method=optimizer,
                    options={"maxiter": max_iter, "rhobeg": rhobeg}
                    if optimizer == "COBYLA"
                    else {"maxiter": max_iter},
                )

                if res.fun < best_expectation:
                    best_expectation = float(res.fun)
                    best_params = res.x.copy()

        except ImportError:
            # Fallback random grid search
            trials = 50
            for _ in range(trials):
                gammas_init = rng.uniform(0.0, gamma_scale, size=p)
                betas_init = rng.uniform(0.0, np.pi, size=p)
                x0 = np.concatenate([gammas_init, betas_init])
                val = objective(x0)
                if val < best_expectation:
                    best_expectation = val
                    best_params = x0.copy()

        if best_params is None:
            best_params = np.zeros(2 * p, dtype=np.float64)

        opt_gamma = best_params[:p]
        opt_beta = best_params[p:]

        # Obtain optimal statevector and measurement sampling
        opt_statevector = self.simulate_statevector(opt_gamma, opt_beta, seed=seed)
        probs = np.abs(opt_statevector) ** 2
        p_sum = np.sum(probs)
        if p_sum > 0:
            probs = probs / p_sum

        indices = rng.choice(len(probs), size=shots, p=probs)
        unique, unique_counts = np.unique(indices, return_counts=True)
        fmt = f"0{self.num_qubits}b"
        counts = {
            format(idx, fmt): int(cnt)
            for idx, cnt in zip(unique, unique_counts, strict=True)
        }
        probabilities = {
            format(idx, fmt): float(probs[idx])
            for idx in unique
        }

        # Include the most probable statevector basis state
        top_idx = int(np.argmax(probs))
        top_bitstr = format(top_idx, fmt)
        if top_bitstr not in counts:
            counts[top_bitstr] = 1
        probabilities[top_bitstr] = float(probs[top_idx])

        # Build candidate solution list sorted by cost
        all_solutions: list[tuple[str, float, float]] = []
        for bitstr, cnt in counts.items():
            idx = int(bitstr, 2)
            c = float(self._diagonal_costs[idx])
            pr = cnt / shots
            all_solutions.append((bitstr, c, pr))

        all_solutions.sort(key=lambda item: item[1])
        best_bitstring = all_solutions[0][0]
        best_cost = all_solutions[0][1]

        quantum_result = Result(
            counts=counts,
            shots=shots,
            num_qubits=self.num_qubits,
            circuit_name=f"QAOA(qubits={self.num_qubits}, p={p})",
        )

        return QAOAResult(
            best_bitstring=best_bitstring,
            best_cost=best_cost,
            optimal_gamma=opt_gamma,
            optimal_beta=opt_beta,
            cost_history=cost_history,
            counts=counts,
            probabilities=probabilities,
            num_qubits=self.num_qubits,
            num_layers=p,
            optimal_statevector=opt_statevector,
            all_solutions=all_solutions,
            quantum_result=quantum_result,
            metadata={
                "optimizer": optimizer,
                "max_iter": max_iter,
                "best_expectation": best_expectation,
                "num_starts": num_starts,
                "mixer_type": self.mixer_type,
            },
        )
