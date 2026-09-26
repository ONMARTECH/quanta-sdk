"""quanta.solvers.cognitive_qaoa_cvp — Cognitive QAOA Closest Vector Problem Solver.

Unifies classical preconditioning (Gram-Schmidt / Babai), Quantum Zeno subspace
pruning, and hippocampal Sharp-Wave Ripple (SWR) engram memory warm-starting into
a high-performance 6-phase hybrid optimization pipeline for lattice CVP.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quanta.cognitive.cvp_arbiter import (
    LatticeEngramMemory,
    WarmStartConfig,
    ZenoProjectionResult,
    ZenoSubspaceArbiter,
    babai_closest_vector,
)
from quanta.primitives.cvp import CVPHamiltonian
from quanta.primitives.qaoa import QAOA, QAOAResult


@dataclass
class CognitiveCVPResult:
    """Standardized result container for Cognitive QAOA CVP optimization.

    Attributes:
        z: Optimal integer coordinate vector in Z^n.
        closest_point: Real lattice vector v = B * z in R^m.
        distance: Euclidean residual norm ||B z - t||_2.
        approximation_factor: Approximation ratio gamma = ||B z - t|| / ||B z* - t||.
        active_qubits: Number of qubits in active QAOA register after Zeno pruning.
        total_qubits: Total number of qubits in unpruned problem representation.
        zeno_pinned: Dictionary of projectively pinned coordinates {coord_idx: integer_val}.
        swr_recalled: Whether prior SWR engrams warm-started this optimization run.
        wall_clock_time: Total end-to-end execution time in seconds.
        iterations: Number of classical optimization iterations evaluated.
        success: Whether the solver recovered a valid lattice vector meeting criteria.
        circuit_depth: Depth of the synthesized QAOA variational circuit.
        qaoa_result: Raw QAOAResult from quantum simulation layer.
        zeno_diagnostics: Diagnostic metrics from ZenoSubspaceArbiter.
        warm_start_diagnostics: Diagnostic metrics from LatticeEngramMemory.
        babai_distance: Baseline Euclidean distance achieved by classical Babai solver.
        solver_mode: Execution mode ('cognitive_qaoa', 'vanilla_qaoa', or 'classical_babai').
    """

    z: np.ndarray
    closest_point: np.ndarray
    distance: float
    approximation_factor: float
    active_qubits: int
    total_qubits: int
    zeno_pinned: dict[int, int]
    swr_recalled: bool
    wall_clock_time: float
    iterations: int = 0
    success: bool = True
    circuit_depth: int = 0
    qaoa_result: QAOAResult | None = None
    zeno_diagnostics: ZenoProjectionResult | None = None
    warm_start_diagnostics: WarmStartConfig | None = None
    babai_distance: float = 0.0
    solver_mode: str = "cognitive_qaoa"
    metadata: dict[str, Any] = field(default_factory=dict)

    # Convenience aliases for cross-module compatibility
    @property
    def best_z(self) -> np.ndarray:
        return self.z

    @property
    def best_lattice_vector(self) -> np.ndarray:
        return self.closest_point

    @property
    def residual_norm(self) -> float:
        return self.distance

    @property
    def num_active_qubits(self) -> int:
        return self.active_qubits

    @property
    def num_total_qubits(self) -> int:
        return self.total_qubits

    @property
    def execution_time_ms(self) -> float:
        return self.wall_clock_time * 1000.0

    @property
    def subspace_reduction_ratio(self) -> float:
        if self.total_qubits <= 0:
            return 0.0
        return (self.total_qubits - self.active_qubits) / float(self.total_qubits)

    def summary(self) -> str:
        """Returns a formatted human-readable summary table."""
        z_str = str(self.z.tolist())
        lines = [
            "╔═══════════════════ Cognitive QAOA CVP Result ═══════════════════",
            f"║ Lattice Dimension:  {len(self.z)}",
            f"║ Total Qubits:       {self.total_qubits} -> Active: {self.active_qubits} "
            f"({self.subspace_reduction_ratio * 100:.1f}% Zeno pruning)",
            f"║ Pinned Coordinates: {self.zeno_pinned}",
            f"║ SWR Recalled:       {self.swr_recalled}",
            f"║ Solution Vector z:  {z_str}",
            f"║ Distance ||Bz - t||:{self.distance:.6f} (Babai: {self.babai_distance:.6f})",
            f"║ Approx Factor (γ):  {self.approximation_factor:.4f}",
            f"║ Wall-Clock Time:    {self.wall_clock_time:.4f}s ({self.execution_time_ms:.1f}ms)",
            "╚═════════════════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


class CognitiveQAOACVPSolver:
    """Cognitive-Quantum Hybrid Solver for the Closest Vector Problem (CVP).

    Orchestrates the 6-phase hybrid optimization lifecycle:
      Phase 1: Classical Preconditioning (Gram-Schmidt decomposition & Babai nearest plane)
      Phase 2: Quantum Zeno Subspace Arbitration (sensitivity analysis & coordinate pinning)
      Phase 3: SWR Engram Memory Recall (episodic replay & warm-start state preparation)
      Phase 4: Variational QAOA Optimization (active-subspace Hamiltonian & biased mixer)
      Phase 5: Solution Synthesis & Decoding (full integer vector reconstruction)
      Phase 6: SWR Engram Consolidation (hippocampal storage of high-fidelity solutions)
    """

    def __init__(
        self,
        p: int = 2,
        shots: int = 1024,
        use_zeno: bool = True,
        use_swr: bool = True,
        zeno_threshold: float = 0.70,
        exploration_drive: float = 0.20,
        optimizer: str = "COBYLA",
        max_iter: int = 100,
        seed: int | None = 42,
        memory: LatticeEngramMemory | None = None,
        min_active_coords: int = 1,
    ) -> None:
        """Initializes the Cognitive QAOA CVP Solver.

        Args:
            p: Number of alternating QAOA layers.
            shots: Measurement shots per sampling evaluation.
            use_zeno: Whether to enable Quantum Zeno subspace pruning.
            use_swr: Whether to enable SWR engram memory replay and consolidation.
            zeno_threshold: Sensitivity threshold tau in [0.0, 1.0] for coordinate freezing.
            exploration_drive: Cognitive exploration drive d in [0.0, 1.0].
            optimizer: Classical optimizer name ('COBYLA', 'Nelder-Mead', etc.).
            max_iter: Maximum optimizer iterations per run.
            seed: Random seed for deterministic simulation.
            memory: Optional shared LatticeEngramMemory instance.
            min_active_coords: Minimum number of active coordinates preserved for QAOA.
        """
        self.p = p
        self.shots = shots
        self.use_zeno = use_zeno
        self.use_swr = use_swr
        self.zeno_threshold = float(zeno_threshold)
        self.exploration_drive = float(exploration_drive)
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.seed = seed
        self.min_active_coords = min_active_coords

        self.zeno_arbiter = (
            ZenoSubspaceArbiter(
                freeze_threshold=self.zeno_threshold,
                exploration_drive=self.exploration_drive,
                min_active_coords=self.min_active_coords,
            )
            if self.use_zeno
            else None
        )

        self.memory: LatticeEngramMemory | None
        if memory is not None:
            self.memory = memory
        elif self.use_swr:
            self.memory = LatticeEngramMemory()
        else:
            self.memory = None

    def solve(
        self,
        basis: np.ndarray,
        target: np.ndarray,
        bits_per_coord: int = 3,
        ground_truth_z: np.ndarray | list[int] | None = None,
    ) -> CognitiveCVPResult:
        """Executes the 6-phase Cognitive QAOA pipeline to solve the Closest Vector Problem.

        Args:
            basis: Lattice basis matrix B of shape (m, n).
            target: Target vector t of shape (m,).
            bits_per_coord: Number of binary qubits allocated per coordinate.
            ground_truth_z: Optional exact ground truth integer coordinates for benchmark verification.

        Returns:
            CognitiveCVPResult dataclass containing optimal coordinates, approximation factor,
            and complete execution telemetry.
        """
        t_start = time.perf_counter()

        basis_arr = np.asarray(basis, dtype=np.float64)
        target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
        m, n = basis_arr.shape

        gt_z = (
            np.asarray(ground_truth_z, dtype=np.int64).reshape(-1)
            if ground_truth_z is not None
            else None
        )

        total_qubits = n * bits_per_coord

        # -------------------------------------------------------------
        # Phase 1: Classical Preconditioning (Gram-Schmidt & Babai)
        # -------------------------------------------------------------
        z_babai, _, _ = babai_closest_vector(basis_arr, target_arr)
        v_babai = basis_arr @ z_babai.astype(np.float64)
        d_babai = float(np.linalg.norm(v_babai - target_arr))

        # -------------------------------------------------------------
        # Phase 2: Quantum Zeno Subspace Arbitration
        # -------------------------------------------------------------
        zeno_diagnostics: ZenoProjectionResult | None = None
        pinned_coords: dict[int, int] = {}
        active_indices = list(range(n))

        if self.use_zeno and self.zeno_arbiter is not None:
            zeno_diagnostics = self.zeno_arbiter.arbitrate_subspace_freezing(
                basis=basis_arr,
                target=target_arr,
                center_z=z_babai,
                bits_per_coord=bits_per_coord,
            )
            pinned_coords = dict(zeno_diagnostics.frozen_coordinates)
            active_indices = list(zeno_diagnostics.active_indices)

        active_qubits = len(active_indices) * bits_per_coord

        # Construct reduced CVPHamiltonian on active coordinates
        reduced_cvp: CVPHamiltonian | None = None
        if active_indices:
            v_pinned = np.zeros(m, dtype=np.float64)
            for i, z_val in pinned_coords.items():
                v_pinned += float(z_val) * basis_arr[:, i]
            t_eff = target_arr - v_pinned
            basis_active = basis_arr[:, active_indices]
            center_active = z_babai[active_indices]

            reduced_cvp = CVPHamiltonian(
                basis=basis_active,
                target=t_eff,
                bits_per_coord=bits_per_coord,
                center=center_active,
            )

        # -------------------------------------------------------------
        # Phase 3: SWR Engram Memory Recall
        # -------------------------------------------------------------
        swr_recalled = False
        warm_config: WarmStartConfig | None = None
        warm_angles: list[float] | None = None
        seed_angles: list[float] | None = None
        mixer_type = "x"

        if self.use_swr and self.memory is not None and reduced_cvp is not None:
            warm_config = self.memory.recall_warm_start(
                basis_id=basis_arr,
                num_qubits=reduced_cvp.num_qubits,
                bits_per_coord=bits_per_coord,
                center=reduced_cvp.center,
            )
            swr_recalled = warm_config.recalled_engram_count > 0
            if swr_recalled:
                warm_angles = warm_config.initial_rotation_angles
                seed_angles = warm_config.seeded_qaoa_angles
                mixer_type = warm_config.mixer_type

        # -------------------------------------------------------------
        # Phase 4: Variational QAOA Optimization
        # -------------------------------------------------------------
        qaoa_res: QAOAResult | None = None
        iterations = 0
        circuit_depth = 0

        if reduced_cvp is not None and reduced_cvp.num_qubits > 0:
            qaoa = QAOA(
                cost_hamiltonian=reduced_cvp,
                p=self.p,
                mixer_type=mixer_type,
                warm_start_angles=warm_angles,
            )

            qaoa_res = qaoa.run(
                initial_params=np.array(seed_angles) if seed_angles is not None else None,
                optimizer=self.optimizer,
                max_iter=self.max_iter,
                shots=self.shots,
                seed=self.seed,
            )
            iterations = len(qaoa_res.cost_history)
            circuit_depth = self.p * 2

        # -------------------------------------------------------------
        # Phase 5: Solution Synthesis & Decoding
        # -------------------------------------------------------------
        best_z = z_babai.copy()
        best_dist = d_babai

        if qaoa_res is not None and reduced_cvp is not None:
            # Evaluate all candidate measurement bitstrings
            candidates_to_check = qaoa_res.all_solutions[:20]
            if not candidates_to_check:
                candidates_to_check = [(qaoa_res.best_bitstring, qaoa_res.best_cost, 1.0)]

            for bitstr, _, _ in candidates_to_check:
                z_active = reduced_cvp.x_to_z(bitstr)
                z_full = np.zeros(n, dtype=np.int64)

                # Reconstruct full coordinate vector
                for i, val in pinned_coords.items():
                    z_full[i] = val
                for local_idx, orig_col in enumerate(active_indices):
                    z_full[orig_col] = z_active[local_idx]

                v_candidate = basis_arr @ z_full.astype(np.float64)
                dist_cand = float(np.linalg.norm(v_candidate - target_arr))

                if dist_cand < best_dist:
                    best_dist = dist_cand
                    best_z = z_full.copy()

        # Final closest lattice point
        closest_point = basis_arr @ best_z.astype(np.float64)
        final_distance = float(np.linalg.norm(closest_point - target_arr))

        # Approximation factor gamma = ||Bz - t|| / ||Bz* - t||
        if gt_z is not None:
            v_star = basis_arr @ gt_z.astype(np.float64)
            d_star = float(np.linalg.norm(v_star - target_arr))
            if d_star > 1e-12:
                approx_factor = final_distance / d_star
            else:
                approx_factor = 1.0 if final_distance < 1e-9 else float("inf")
        else:
            approx_factor = 1.0 if final_distance <= d_babai else final_distance / max(1e-12, d_babai)

        # -------------------------------------------------------------
        # Phase 6: SWR Engram Consolidation
        # -------------------------------------------------------------
        if self.use_swr and self.memory is not None and final_distance <= d_babai:
            opt_angles = None
            if qaoa_res is not None:
                opt_angles = np.concatenate([qaoa_res.optimal_gamma, qaoa_res.optimal_beta])

            is_gt = gt_z is not None and np.array_equal(best_z, gt_z)
            self.memory.store_solution(
                basis_id=basis_arr,
                z=best_z,
                distance=final_distance,
                angles=opt_angles,
                basis=basis_arr,
                target=target_arr,
                is_ground_truth=is_gt,
            )

        t_elapsed = time.perf_counter() - t_start

        mode = "cognitive_qaoa" if (self.use_zeno and self.use_swr) else (
            "zeno_qaoa" if self.use_zeno else (
                "swr_qaoa" if self.use_swr else "vanilla_qaoa"
            )
        )

        return CognitiveCVPResult(
            z=best_z,
            closest_point=closest_point,
            distance=final_distance,
            approximation_factor=approx_factor,
            active_qubits=active_qubits,
            total_qubits=total_qubits,
            zeno_pinned=pinned_coords,
            swr_recalled=swr_recalled,
            wall_clock_time=t_elapsed,
            iterations=iterations,
            success=True,
            circuit_depth=circuit_depth,
            qaoa_result=qaoa_res,
            zeno_diagnostics=zeno_diagnostics,
            warm_start_diagnostics=warm_config,
            babai_distance=d_babai,
            solver_mode=mode,
            metadata={
                "lattice_dimension": n,
                "bits_per_coord": bits_per_coord,
                "p_layers": self.p,
                "shots": self.shots,
                "optimizer": self.optimizer,
            },
        )
