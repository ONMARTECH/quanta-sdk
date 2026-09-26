"""quanta.cognitive.cvp_arbiter — Quantum Zeno Subspace Arbiter & SWR Lattice Memory.

Bridges Quanta SDK's Biomorphic Cognitive Architecture (Quantum Zeno Attention and
Lindblad-attenuated Hippocampal SWR Memory) with lattice Closest Vector Problem (CVP)
Ising/QUBO Hamiltonians.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quanta.cognitive.arbiter import QuantumDecisionArbiter
from quanta.cognitive.memory import CognitiveMemoryManager
from quanta.primitives.cvp import CVPHamiltonian


def gram_schmidt_orthogonalize(basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes Gram-Schmidt orthogonal vectors b_i* and projection coefficients mu.

    Given basis B = [b_0, ..., b_{n-1}] in R^(m x n), computes:
        b_i* = b_i - sum_{j=0}^{i-1} mu_{i, j} b_j*
        mu_{i, j} = <b_i, b_j*> / ||b_j*||^2

    Args:
        basis: Matrix of shape (m, n) with basis vectors as columns.

    Returns:
        Tuple of:
            - b_star: Orthogonal vectors matrix of shape (m, n).
            - mu: Lower-triangular projection coefficients matrix of shape (n, n)
              with unit diagonal.
    """
    m, n = basis.shape
    b_star = np.zeros((m, n), dtype=np.float64)
    mu = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        col = basis[:, i].astype(np.float64)
        v = col.copy()
        for j in range(i):
            norm_sq = float(np.dot(b_star[:, j], b_star[:, j]))
            mu_ij = float(np.dot(col, b_star[:, j]) / norm_sq) if norm_sq > 1e-15 else 0.0
            mu[i, j] = mu_ij
            v -= mu_ij * b_star[:, j]
        b_star[:, i] = v
        mu[i, i] = 1.0

    return b_star, mu


def babai_closest_vector(
    basis: np.ndarray,
    target: np.ndarray,
    b_star: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float], list[float]]:
    """Executes the Babai Nearest Plane algorithm to compute initial integer coordinates.

    Args:
        basis: Matrix of shape (m, n) with basis vectors as columns.
        target: Target vector in R^m.
        b_star: Optional precomputed Gram-Schmidt orthogonal vectors.

    Returns:
        Tuple of:
            - z: Integer coordinate vector in Z^n.
            - continuous_projections: Continuous projection values c_i.
            - ambiguities: Absolute distances |c_i - z_i| in [0.0, 0.5].
    """
    m, n = basis.shape
    if b_star is None:
        b_star, _ = gram_schmidt_orthogonalize(basis)

    b_prime = target.copy().astype(np.float64)
    z = np.zeros(n, dtype=np.int64)
    continuous_projections = [0.0] * n
    ambiguities = [0.0] * n

    for i in range(n - 1, -1, -1):
        norm_sq = float(np.dot(b_star[:, i], b_star[:, i]))
        c_i = float(np.dot(b_prime, b_star[:, i]) / norm_sq) if norm_sq > 1e-15 else 0.0
        z_i = int(np.round(c_i))
        z[i] = z_i
        continuous_projections[i] = c_i
        ambiguities[i] = abs(c_i - z_i)
        b_prime -= z_i * basis[:, i].astype(np.float64)

    return z, continuous_projections, ambiguities


@dataclass
class SubspaceSensitivity:
    """Gram-Schmidt coordinate sensitivity analysis and freeze recommendation.

    Attributes:
        coordinate_index: Lattice coordinate index in [0, n-1].
        gram_schmidt_norm: Euclidean norm ||b_i*|| of orthogonal vector.
        hyperplane_distance: Distance between parallel integer hyperplanes.
        integer_ambiguity: Distance to nearest integer |c_i - round(c_i)| in [0, 0.5].
        sensitivity_score: Normalized confidence score in [0.0, 1.0].
        recommended_freeze: Whether this coordinate should be projectively pinned.
        candidate_integer: Candidate integer value from Babai projection.
        p_zeno: Dynamic Quantum Zeno pinning factor in [0.15, 0.985].
    """

    coordinate_index: int
    gram_schmidt_norm: float
    hyperplane_distance: float
    integer_ambiguity: float
    sensitivity_score: float
    recommended_freeze: bool
    candidate_integer: int
    p_zeno: float = 0.50


@dataclass
class ZenoProjectionResult:
    """Outcome of Quantum Zeno subspace arbitration.

    Attributes:
        pinned_indices: Sorted list of coordinate indices pinned to fixed integers.
        active_indices: Sorted list of coordinate indices remaining quantum active.
        p_zeno_values: Dictionary mapping coordinate index to P_zeno pinning factor.
        frozen_coordinates: Dictionary mapping pinned coordinate index to integer value.
        subspace_reduction_ratio: Fraction of coordinates pinned (pinned / total).
        num_pinned_qubits: Total number of qubits eliminated from quantum register.
        num_active_qubits: Number of qubits remaining in active QAOA register.
        total_qubits: Total number of qubits in full unpruned problem.
        mixer_weights: Per-qubit transverse mixer weights w_a = (1 - P_zeno).
        arbitration_telemetry: Diagnostic dictionary with sensitivity metrics.
    """

    pinned_indices: list[int]
    active_indices: list[int]
    p_zeno_values: dict[int, float]
    frozen_coordinates: dict[int, int]
    subspace_reduction_ratio: float
    num_pinned_qubits: int
    num_active_qubits: int
    total_qubits: int
    mixer_weights: np.ndarray
    arbitration_telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass
class WarmStartConfig:
    """Warm-start quantum state and mixer parameters derived from SWR replay.

    Attributes:
        initial_rotation_angles: List of single-qubit RY rotation angles theta_a.
        recalled_engram_count: Number of high-fidelity engrams retrieved.
        mean_fidelity: Mean retention fidelity F(t) of recalled engrams.
        seeded_qaoa_angles: Optional variational angles [gamma_0, beta_0] to seed optimizer.
        mixer_type: Mixer type: 'biased' (non-commuting) or 'x' (standard).
        bit_priors: Marginal bit probabilities c_a in [delta, 1 - delta].
    """

    initial_rotation_angles: list[float]
    recalled_engram_count: int
    mean_fidelity: float
    seeded_qaoa_angles: list[float] | None
    mixer_type: str = "biased"
    bit_priors: list[float] = field(default_factory=list)

    def __iter__(self) -> Iterator[Any]:
        """Supports tuple unpacking: warm_angles, seed_angles = recall_warm_start(...)."""
        angles_arr = (
            np.array(self.initial_rotation_angles, dtype=np.float64)
            if self.initial_rotation_angles
            else None
        )
        seeds_arr = (
            np.array(self.seeded_qaoa_angles, dtype=np.float64)
            if self.seeded_qaoa_angles is not None
            else None
        )
        return iter((angles_arr, seeds_arr))


@dataclass
class LatticeEngram:
    """Consolidated cognitive engram representing a high-fidelity lattice vector solution.

    Attributes:
        key: Unique engram identifier.
        z_vector: Integer coordinates in Z^n.
        lattice_vector: Real lattice vector v = B * z in R^m.
        residual_norm: Euclidean distance ||v - t||_2.
        dimension: Lattice dimension n.
        condition_number: Condition number kappa(B).
        salience: Dopaminergic priority tag S in [0.2, 3.5].
        retention_fidelity: Biological retention fidelity F(t) in [0.0, 1.0].
        qaoa_optimal_angles: Optimized variational angles [gamma*, beta*].
        is_core_anchor: Whether this engram is an immutable core anchor.
        created_at: Creation timestamp in seconds.
    """

    key: str
    z_vector: list[int]
    lattice_vector: list[float]
    residual_norm: float
    dimension: int
    condition_number: float
    salience: float
    retention_fidelity: float
    qaoa_optimal_angles: list[float] | None = None
    is_core_anchor: bool = True
    created_at: float = field(default_factory=time.time)


class ZenoSubspaceArbiter:
    """Quantum Zeno Subspace Arbiter for Closest Vector Problem (CVP) lattices.

    Eliminates the exponential barren plateau in lattice QAOA by evaluating Gram-Schmidt
    orthogonal sensitivities and dynamically pinning settled coordinate dimensions
    via Quantum Zeno projection (P_zeno >= tau_freeze).
    """

    def __init__(
        self,
        cvp_hamiltonian: CVPHamiltonian | None = None,
        freeze_threshold: float = 0.70,
        exploration_drive: float = 0.20,
        min_active_coords: int = 1,
        arbiter: QuantumDecisionArbiter | None = None,
    ) -> None:
        """Initializes the Zeno Subspace Arbiter.

        Args:
            cvp_hamiltonian: Optional initial CVPHamiltonian instance.
            freeze_threshold: Threshold tau in [0.0, 1.0] above which coordinates are pinned.
            exploration_drive: Drive parameter in [0.0, 1.0] scaling Anti-Zeno tunneling.
            min_active_coords: Minimum number of coordinates to leave active for QAOA.
            arbiter: Optional neural QuantumDecisionArbiter instance.
        """
        self.cvp_hamiltonian = cvp_hamiltonian
        self.freeze_threshold = float(freeze_threshold)
        self.exploration_drive = float(exploration_drive)
        self.min_active_coords = max(0, min_active_coords)
        self._neural_arbiter = arbiter

    def analyze_sensitivities(
        self,
        basis: np.ndarray,
        target: np.ndarray,
    ) -> list[SubspaceSensitivity]:
        """Computes Gram-Schmidt vector norms and coordinate sensitivity scores.

        Args:
            basis: Lattice basis matrix B of shape (m, n).
            target: Target vector t of shape (m,).

        Returns:
            List of SubspaceSensitivity metrics for each coordinate i in [0, n-1].
        """
        basis_arr = np.asarray(basis, dtype=np.float64)
        target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
        m, n = basis_arr.shape

        b_star, _ = gram_schmidt_orthogonalize(basis_arr)
        z_babai, c_vals, ambiguities = babai_closest_vector(basis_arr, target_arr, b_star=b_star)

        gs_norms = [float(np.linalg.norm(b_star[:, i])) for i in range(n)]
        max_gs = max(gs_norms) if gs_norms and max(gs_norms) > 1e-15 else 1.0

        sensitivities: list[SubspaceSensitivity] = []
        for i in range(n):
            amb = float(ambiguities[i])
            # Integer confidence: 1.0 when amb == 0.0 (exact integer), 0.0 when amb == 0.5
            eta_i = max(0.0, min(1.0, 1.0 - 2.0 * amb))
            xi_i = gs_norms[i] / max_gs
            sensitivity_score = float(0.65 * eta_i + 0.35 * xi_i)

            # Dynamic Zeno Pinning Factor P_zeno using Quanta's exact arbiter formula
            margin_factor = math.pow(eta_i, 0.7) if eta_i > 0.0 else 0.0
            raw_zeno = (
                0.25
                + 0.52 * eta_i
                + 0.62 * margin_factor
                - 0.20 * self.exploration_drive
            )
            p_zeno = float(max(0.15, min(0.985, raw_zeno)))

            rec_freeze = bool(p_zeno >= self.freeze_threshold)

            sensitivities.append(
                SubspaceSensitivity(
                    coordinate_index=i,
                    gram_schmidt_norm=gs_norms[i],
                    hyperplane_distance=gs_norms[i],
                    integer_ambiguity=amb,
                    sensitivity_score=sensitivity_score,
                    recommended_freeze=rec_freeze,
                    candidate_integer=int(z_babai[i]),
                    p_zeno=p_zeno,
                )
            )

        return sensitivities

    def compute_zeno_factors(
        self,
        cvp_hamiltonian: CVPHamiltonian | None = None,
    ) -> tuple[np.ndarray, dict[int, int]]:
        """Computes Zeno pinning factors P_zeno and pinned coordinate values.

        Args:
            cvp_hamiltonian: Optional CVPHamiltonian instance (uses self.cvp_hamiltonian if None).

        Returns:
            Tuple of:
                - p_zeno_array: 1D array of P_zeno values for coordinates [0, n-1].
                - pinned_dict: Dictionary mapping pinned coordinate index to integer value.
        """
        h = cvp_hamiltonian or self.cvp_hamiltonian
        if h is None:
            raise ValueError("No CVPHamiltonian provided to compute_zeno_factors.")

        sensitivities = self.analyze_sensitivities(h.basis, h.target)
        n = len(sensitivities)
        p_zeno_arr = np.array([s.p_zeno for s in sensitivities], dtype=np.float64)

        # Candidate pinned coordinates
        pinned_candidates = {
            s.coordinate_index: s.candidate_integer
            for s in sensitivities
            if s.p_zeno >= self.freeze_threshold
        }

        # Ensure minimum active coordinates
        num_pinned = len(pinned_candidates)
        max_allowed_pinned = max(0, n - self.min_active_coords)
        if num_pinned > max_allowed_pinned:
            # Sort by p_zeno ascending to unpin the least confident ones
            sorted_by_zeno = sorted(
                pinned_candidates.keys(),
                key=lambda idx: sensitivities[idx].p_zeno,
            )
            to_unpin = sorted_by_zeno[: (num_pinned - max_allowed_pinned)]
            for idx in to_unpin:
                pinned_candidates.pop(idx, None)

        return p_zeno_arr, pinned_candidates

    def arbitrate_subspace_freezing(
        self,
        basis: np.ndarray,
        target: np.ndarray,
        center_z: np.ndarray | None = None,
        bits_per_coord: int = 3,
    ) -> ZenoProjectionResult:
        """Arbitrates which coordinate dimensions to freeze using Quantum Zeno dynamics.

        Args:
            basis: Lattice basis matrix B of shape (m, n).
            target: Target vector t of shape (m,).
            center_z: Optional reference center vector in Z^n.
            bits_per_coord: Number of binary qubits per coordinate.

        Returns:
            ZenoProjectionResult detailing pinned/active subspaces and mixer weights.
        """
        sensitivities = self.analyze_sensitivities(basis, target)
        n = len(sensitivities)
        total_qubits = n * bits_per_coord

        p_zeno_dict = {s.coordinate_index: s.p_zeno for s in sensitivities}
        pinned_candidates = {
            s.coordinate_index: s.candidate_integer
            for s in sensitivities
            if s.p_zeno >= self.freeze_threshold
        }

        # Guard minimum active coordinates
        num_pinned = len(pinned_candidates)
        max_allowed_pinned = max(0, n - self.min_active_coords)
        if num_pinned > max_allowed_pinned:
            sorted_by_zeno = sorted(
                pinned_candidates.keys(),
                key=lambda idx: sensitivities[idx].p_zeno,
            )
            to_unpin = sorted_by_zeno[: (num_pinned - max_allowed_pinned)]
            for idx in to_unpin:
                pinned_candidates.pop(idx, None)

        pinned_indices = sorted(pinned_candidates.keys())
        active_indices = [i for i in range(n) if i not in pinned_candidates]

        num_pinned_qubits = len(pinned_indices) * bits_per_coord
        num_active_qubits = len(active_indices) * bits_per_coord
        subspace_reduction_ratio = len(pinned_indices) / float(n) if n > 0 else 0.0

        # Construct mixer weights w_a = 1.0 - P_zeno,coord for all N qubits
        mixer_weights = self.build_zeno_mixer_weights(
            total_qubits=total_qubits,
            p_zeno_values=p_zeno_dict,
            bits_per_coord=bits_per_coord,
        )

        telemetry = {
            "sensitivities": [s.__dict__ for s in sensitivities],
            "freeze_threshold": self.freeze_threshold,
            "exploration_drive": self.exploration_drive,
            "min_active_coords": self.min_active_coords,
        }

        return ZenoProjectionResult(
            pinned_indices=pinned_indices,
            active_indices=active_indices,
            p_zeno_values=p_zeno_dict,
            frozen_coordinates=pinned_candidates,
            subspace_reduction_ratio=subspace_reduction_ratio,
            num_pinned_qubits=num_pinned_qubits,
            num_active_qubits=num_active_qubits,
            total_qubits=total_qubits,
            mixer_weights=mixer_weights,
            arbitration_telemetry=telemetry,
        )

    def prune_hamiltonian(
        self,
        cvp_hamiltonian: CVPHamiltonian | None = None,
    ) -> tuple[CVPHamiltonian, dict[int, int]]:
        """Projects CVPHamiltonian onto the active subspace defined by Zeno pinning.

        Folds pinned coordinates into an effective target:
            t_eff = t - sum_{i in I} z_i* b_i
        and constructs a reduced CVPHamiltonian on the remaining active columns.

        Guarantees exact algebraic energy conservation:
            ||B_red * z_active - t_eff||^2 == ||B * z_full - t||^2

        Args:
            cvp_hamiltonian: CVPHamiltonian to prune. Uses self.cvp_hamiltonian if None.

        Returns:
            Tuple of:
                - reduced_hamiltonian: CVPHamiltonian over active coordinates.
                - pinned_coordinates: Dictionary of pinned coordinate values {i: z_i*}.
        """
        h = cvp_hamiltonian or self.cvp_hamiltonian
        if h is None:
            raise ValueError("No CVPHamiltonian provided to prune_hamiltonian.")

        p_zeno_arr, pinned_dict = self.compute_zeno_factors(h)
        n = h.dim

        if not pinned_dict:
            # Nothing to pin; return original
            return h, {}

        active_indices = [i for i in range(n) if i not in pinned_dict]

        if not active_indices:
            # If everything was pinned, unpin coordinate with minimum P_zeno
            min_zeno_coord = int(np.argmin(p_zeno_arr))
            pinned_dict.pop(min_zeno_coord, None)
            active_indices = [min_zeno_coord]

        # Calculate pinned contribution to lattice vector
        m = h.basis.shape[0]
        v_pinned = np.zeros(m, dtype=np.float64)
        for i, z_val in pinned_dict.items():
            v_pinned += float(z_val) * h.basis[:, i].astype(np.float64)

        t_eff = h.target - v_pinned
        reduced_basis = h.basis[:, active_indices]

        # Reduced center vector
        reduced_center: np.ndarray | None = None
        if h.center is not None:
            reduced_center = h.center[active_indices]

        reduced_cvp = CVPHamiltonian(
            basis=reduced_basis,
            target=t_eff,
            bits_per_coord=h.bits_per_coord,
            center=reduced_center,
        )

        return reduced_cvp, pinned_dict

    def build_zeno_mixer_weights(
        self,
        total_qubits: int,
        p_zeno_values: dict[int, float] | np.ndarray,
        bits_per_coord: int = 3,
    ) -> np.ndarray:
        """Constructs per-qubit mixer field weights w_a = (1 - P_zeno).

        Args:
            total_qubits: Total number of qubits in the problem.
            p_zeno_values: P_zeno values mapped by coordinate index or 1D array.
            bits_per_coord: Number of bits per coordinate.

        Returns:
            1D array of length total_qubits with mixer weights in [0.0, 1.0].
        """
        weights = np.ones(total_qubits, dtype=np.float64)
        for a in range(total_qubits):
            coord = a // bits_per_coord
            if isinstance(p_zeno_values, dict):
                p_z = float(p_zeno_values.get(coord, 0.50))
            else:
                p_z = (
                    float(p_zeno_values[coord])
                    if coord < len(p_zeno_values)
                    else 0.50
                )
            weights[a] = max(0.0, min(1.0, 1.0 - p_z))
        return weights


class LatticeEngramMemory:
    """SWR Episodic and Working Memory Manager for lattice vector solutions.

    Leverages Quanta SDK's CognitiveMemoryManager with CSF biophysical shielding
    to consolidate high-fidelity lattice solutions (S >= 2.0). Provides Sharp-Wave
    Ripple (SWR) replay to construct continuous warm-start quantum states |psi_warm>
    and seed classical variational angles (gamma_0, beta_0).
    """

    def __init__(
        self,
        memory_manager: CognitiveMemoryManager | None = None,
        capacity: int = 64,
        dim: int = 64,
    ) -> None:
        """Initializes the Lattice Engram Memory.

        Args:
            memory_manager: Optional existing CognitiveMemoryManager.
            capacity: Hippocampal buffer capacity.
            dim: Embedding dimension for statevectors (default: 64).
        """
        self.dim = dim
        self.memory_manager = memory_manager or CognitiveMemoryManager(
            capacity=capacity,
            dim=dim,
            enable_csf_shielding=True,
        )
        self._engrams: dict[str, list[LatticeEngram]] = {}

    @staticmethod
    def compute_basis_id(basis: np.ndarray | str) -> str:
        """Generates a stable, canonical basis identifier."""
        if isinstance(basis, str):
            return basis
        arr = np.asarray(basis, dtype=np.float64)
        m, n = arr.shape
        digest = hashlib.sha256(arr.tobytes()).hexdigest()[:12]
        return f"lattice_{m}x{n}_{digest}"

    def store_solution(
        self,
        basis_id: str | np.ndarray,
        z: np.ndarray | list[int],
        distance: float,
        angles: np.ndarray | list[float] | None = None,
        basis: np.ndarray | None = None,
        target: np.ndarray | None = None,
        is_ground_truth: bool = False,
        salience: float | None = None,
    ) -> int:
        """Crystallizes a lattice vector solution into a CSF-shielded engram.

        Args:
            basis_id: Basis identifier string or basis matrix.
            z: Integer coordinate vector in Z^n.
            distance: Euclidean distance ||B z - t||_2.
            angles: Optional optimized variational angles [gamma*, beta*].
            basis: Optional basis matrix B for condition number / lattice vector.
            target: Optional target vector t.
            is_ground_truth: True if this is a verified global minimum / benchmark.
            salience: Optional explicit dopaminergic tag (defaults to S >= 2.0 for protection).

        Returns:
            Index of the stored engram in the hippocampal buffer.
        """
        key_id = self.compute_basis_id(basis_id)
        z_arr = np.asarray(z, dtype=np.int64).reshape(-1)
        z_list = [int(v) for v in z_arr]
        dist = float(distance)

        # Dopaminergic salience: S >= 2.0 guarantees CSF immunity to Lindblad decay
        if salience is not None:
            eff_salience = float(salience)
        elif is_ground_truth:
            eff_salience = 3.5
        else:
            eff_salience = float(min(3.5, max(2.0, 2.0 + 1.0 / (1.0 + dist))))

        angles_list = (
            [float(a) for a in np.asarray(angles, dtype=np.float64).reshape(-1)]
            if angles is not None
            else None
        )

        cond_num = 1.0
        lattice_vec: list[float] = []
        if basis is not None:
            b_arr = np.asarray(basis, dtype=np.float64)
            try:
                cond_num = float(np.linalg.cond(b_arr))
            except np.linalg.LinAlgError:
                cond_num = 1.0
            v = b_arr @ z_arr.astype(np.float64)
            lattice_vec = [float(x) for x in v]

        engram = LatticeEngram(
            key=f"{key_id}_sol_{len(self._engrams.get(key_id, []))}",
            z_vector=z_list,
            lattice_vector=lattice_vec,
            residual_norm=dist,
            dimension=len(z_list),
            condition_number=cond_num,
            salience=eff_salience,
            retention_fidelity=1.0,
            qaoa_optimal_angles=angles_list,
            is_core_anchor=bool(is_ground_truth or eff_salience >= 2.0),
        )

        # Register in CognitiveMemoryManager
        idx = self.memory_manager.record_decision(
            key=engram.key,
            content=f"CVP Solution for {key_id}: dim={len(z_list)}, norm={dist:.6f}, z={z_list}",
            salience=eff_salience,
            category="lattice_engram",
            is_core_anchor=engram.is_core_anchor,
        )

        if key_id not in self._engrams:
            self._engrams[key_id] = []
        self._engrams[key_id].append(engram)

        return idx

    def recall_warm_start(
        self,
        basis_id: str | np.ndarray,
        num_qubits: int,
        bits_per_coord: int = 3,
        center: np.ndarray | None = None,
        min_fidelity: float = 0.40,
        delta_regularization: float = 0.05,
    ) -> WarmStartConfig:
        """Replays high-vitality lattice engrams to compute warm-start angles and seeded parameters.

        Args:
            basis_id: Basis identifier string or matrix.
            num_qubits: Number of active qubits in target register.
            bits_per_coord: Number of binary qubits allocated per coordinate.
            center: Optional reference center vector in Z^n.
            min_fidelity: Minimum biological retention fidelity threshold.
            delta_regularization: Regularization delta in (0.0, 0.5) to prevent deterministic trapping.

        Returns:
            WarmStartConfig containing single-qubit RY rotation angles and seed variational angles.
        """
        key_id = self.compute_basis_id(basis_id)
        engrams = self._engrams.get(key_id, [])

        # Refresh retention fidelity via SWR recall
        self.memory_manager.recall_vital_context(top_k=5, min_fidelity=min_fidelity)

        if not engrams:
            # Fallback to uniform superposition |+>^{\otimes N}
            uniform_angles = [float(math.pi / 2.0)] * num_qubits
            return WarmStartConfig(
                initial_rotation_angles=uniform_angles,
                recalled_engram_count=0,
                mean_fidelity=1.0,
                seeded_qaoa_angles=None,
                mixer_type="biased",
                bit_priors=[0.5] * num_qubits,
            )

        n_dim = num_qubits // bits_per_coord
        span = 1 << bits_per_coord
        half_span = 1 << (bits_per_coord - 1)
        if center is not None:
            c_arr = np.asarray(center, dtype=np.int64).reshape(-1)
            z_min = c_arr - half_span
        else:
            z_min = -half_span * np.ones(n_dim, dtype=np.int64)

        # Weight each engram by S * F(t)
        weights = [float(e.salience * e.retention_fidelity) for e in engrams]
        total_weight = sum(weights)
        if total_weight <= 0:
            total_weight = 1.0
        normalized_weights = [w / total_weight for w in weights]

        # Calculate bitwise marginal probabilities p_a
        p_bits = np.zeros(num_qubits, dtype=np.float64)
        for e, w in zip(engrams, normalized_weights, strict=True):
            z_sub = np.array(e.z_vector[:n_dim], dtype=np.int64)
            delta_z = z_sub - z_min[:len(z_sub)]
            x_engram = np.zeros(num_qubits, dtype=np.float64)

            for i in range(min(n_dim, len(delta_z))):
                val = int(max(0, min(span - 1, delta_z[i])))
                for bit in range(bits_per_coord):
                    bit_val = (val >> bit) & 1
                    x_engram[i * bits_per_coord + bit] = float(bit_val)

            p_bits += w * x_engram

        # Regularize: c_a = (1 - 2*delta)*p_a + delta in [delta, 1 - delta]
        delta = float(delta_regularization)
        c_priors = (1.0 - 2.0 * delta) * p_bits + delta
        c_priors = np.clip(c_priors, delta, 1.0 - delta)

        # Warm-start rotation angles theta_a = 2 * arcsin(sqrt(c_a))
        theta_angles = [float(2.0 * math.asin(math.sqrt(float(c)))) for c in c_priors]

        # Retrieve seeded QAOA angles from highest vitality engram
        best_engram = max(engrams, key=lambda e: e.salience * e.retention_fidelity)
        seed_angles = best_engram.qaoa_optimal_angles

        mean_fid = float(np.mean([e.retention_fidelity for e in engrams]))

        return WarmStartConfig(
            initial_rotation_angles=theta_angles,
            recalled_engram_count=len(engrams),
            mean_fidelity=mean_fid,
            seeded_qaoa_angles=seed_angles,
            mixer_type="biased",
            bit_priors=c_priors.tolist(),
        )

    def has_engram(self, basis_id: str | np.ndarray) -> bool:
        """Checks if any engrams are stored for the given basis."""
        key_id = self.compute_basis_id(basis_id)
        return bool(self._engrams.get(key_id))

    def get_engram_count(self, basis_id: str | np.ndarray | None = None) -> int:
        """Returns total engram count for a specific basis or across all bases."""
        if basis_id is not None:
            key_id = self.compute_basis_id(basis_id)
            return len(self._engrams.get(key_id, []))
        return sum(len(engs) for engs in self._engrams.values())
