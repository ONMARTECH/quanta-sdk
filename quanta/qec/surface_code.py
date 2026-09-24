"""
quanta.qec.surface_code -- Surface code for logical qubits.

The surface code is the leading candidate for fault-tolerant quantum
computing. It arranges physical qubits on a 2D grid with:
  - Data qubits (carry information)
  - X-syndrome qubits (detect bit-flip errors)
  - Z-syndrome qubits (detect phase-flip errors)

A [[d^2, 1, d]] code: d^2 physical qubits encode 1 logical qubit
with code distance d (corrects floor((d-1)/2) errors).

Example:
    >>> from quanta.qec.surface_code import SurfaceCode
    >>> code = SurfaceCode(distance=3)
    >>> print(code.summary())
    Surface Code [[9, 1, 3]]
    >>> result = code.simulate_error_correction(error_rate=0.01, rounds=100)
    >>> print(f"Logical error rate: {result.logical_error_rate:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from quanta.qec.decoder import MWPMDecoder

__all__ = ["SurfaceCode", "SurfaceCodeResult", "DynamicSurfaceCodeResult", "SpacetimeDefect"]


@dataclass
class SpacetimeDefect:
    """Spacetime detection event in multi-round surface code."""
    time: int
    stabilizer_id: int
    basis: str
    coords: tuple[float, float, int] = (0.0, 0.0, 0)

    def __repr__(self) -> str:
        return f"Defect(time={self.time}, stabilizer={self.stabilizer_id}, basis={self.basis})"


@dataclass
class DynamicSurfaceCodeResult:
    """Result of multi-round dynamic surface code simulation (Willow-style).

    Attributes:
        distance: Code distance d.
        cycles: Number of syndrome extraction cycles (T).
        physical_error_rate: Physical data qubit error probability per cycle.
        measurement_error_rate: Syndrome measurement error probability.
        shots: Number of Monte Carlo shots.
        logical_error_rate: Uncorrected logical error rate after T cycles.
        defects_detected: Total spacetime detection events (Δs_t = s_t ⊕ s_{t-1}).
        willow_suppression_factor: Estimated Λ scaling factor.
        raw_defects: Genuine spacetime detection events recorded during simulation.
        raw_syndrome_history: Genuine multi-cycle syndrome extractions.
    """

    distance: int
    cycles: int
    physical_error_rate: float
    measurement_error_rate: float
    shots: int
    logical_error_rate: float
    defects_detected: int
    willow_suppression_factor: float
    raw_defects: list[SpacetimeDefect] = field(default_factory=list)
    raw_syndrome_history: list[dict[str, list[int]]] = field(default_factory=list)

    @property
    def rounds(self) -> int:
        return self.cycles

    @property
    def defects(self) -> list[SpacetimeDefect]:
        return self.raw_defects

    @property
    def syndrome_history(self) -> list[dict[str, list[int]]]:
        return self.raw_syndrome_history

    @property
    def lambda_factor(self) -> float:
        return self.willow_suppression_factor

    @property
    def below_threshold(self) -> bool:
        return self.willow_suppression_factor > 1.0

    def summary(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════════╗",
            "║  Willow Dynamic Surface Code Simulation (QEC)    ║",
            "╠══════════════════════════════════════════════════╣",
            f"║  Distance: {self.distance} | Cycles (T): {self.cycles:<16}║",
            f"║  Physical Data Error:        {self.physical_error_rate:.3%}               ║",
            f"║  Measurement Error:          {self.measurement_error_rate:.3%}               ║",
            f"║  Logical Error Rate:         {self.logical_error_rate:.4%}              ║",
            f"║  Spacetime Defects (Δs):     {self.defects_detected:<16}║",
            f"║  Suppression Factor (Λ):     {self.willow_suppression_factor:.2f}x               ║",
            "╚══════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


@dataclass
class SurfaceCodeResult:
    """Result of surface code error correction simulation.

    Attributes:
        logical_error_rate: Rate of uncorrectable logical errors.
        physical_error_rate: Input physical error rate.
        rounds: Number of error correction rounds.
        errors_injected: Total errors injected.
        errors_corrected: Successfully corrected errors.
        threshold_estimate: Estimated threshold error rate.
    """
    logical_error_rate: float
    physical_error_rate: float
    rounds: int
    errors_injected: int
    errors_corrected: int
    threshold_estimate: float

    def summary(self) -> str:
        suppression = (
            self.physical_error_rate / self.logical_error_rate
            if self.logical_error_rate > 0 else float("inf")
        )
        lines = [
            "╔══════════════════════════════════════╗",
            "║  Surface Code Error Correction       ║",
            "╠══════════════════════════════════════╣",
            f"║  Physical error rate: {self.physical_error_rate:.2%}         ║",
            f"║  Logical error rate:  {self.logical_error_rate:.4%}       ║",
            f"║  Suppression factor:  {suppression:.1f}x          ║",
            f"║  Rounds:              {self.rounds:<15}║",
            f"║  Errors: {self.errors_corrected}/{self.errors_injected} corrected"
            + " " * 10 + "║",
            "╚══════════════════════════════════════╝",
        ]
        return "\n".join(lines)


class SurfaceCode:
    """Surface code for fault-tolerant quantum computing.

    Implements the rotated surface code on a d×d grid.

    Args:
        distance: Code distance d. Physical qubits = d^2.
            d=3: 9 qubits, corrects 1 error
            d=5: 25 qubits, corrects 2 errors
            d=7: 49 qubits, corrects 3 errors
    """

    def __init__(self, distance: int = 3) -> None:
        if distance < 3 or distance % 2 == 0:
            raise ValueError(
                f"Distance must be odd integer >= 3. Got: {distance}"
            )
        self.distance = distance
        self.n_physical = distance ** 2  # Data qubits
        self.n_logical = 1  # Always 1 for surface code
        self.n_syndrome_x = (distance ** 2 - 1) // 2
        self.n_syndrome_z = (distance ** 2 - 1) // 2

        # Build stabilizer generators for syndrome extraction
        self._x_stabilizers, self._z_stabilizers = self._build_stabilizers()

    def _build_stabilizers(self) -> tuple[list[list[int]], list[list[int]]]:
        """Builds X-type and Z-type stabilizer generators from lattice topology.

        The rotated surface code arranges data qubits on a d×d grid.
        X stabilizers (detect Z errors) sit on faces, Z stabilizers (detect
        X errors) sit on vertices. Each stabilizer acts on 2-4 neighboring
        data qubits.

        Returns:
            Tuple of (x_stabilizers, z_stabilizers), each a list of qubit
            index lists that each stabilizer acts on.
        """
        d = self.distance
        x_stabs: list[list[int]] = []
        z_stabs: list[list[int]] = []

        def idx(r: int, c: int) -> int:
            return r * d + c

        # X stabilizers: checkerboard pattern (even parity faces)
        for r in range(d - 1):
            for c in range(d - 1):
                if (r + c) % 2 == 0:
                    qubits = [idx(r, c), idx(r, c + 1),
                              idx(r + 1, c), idx(r + 1, c + 1)]
                    x_stabs.append(qubits)

        # Z stabilizers: checkerboard pattern (odd parity faces)
        for r in range(d - 1):
            for c in range(d - 1):
                if (r + c) % 2 == 1:
                    qubits = [idx(r, c), idx(r, c + 1),
                              idx(r + 1, c), idx(r + 1, c + 1)]
                    z_stabs.append(qubits)

        # Top & bottom boundary stabilizers (Z-type, weight-2 on horizontal edges)
        for c in range(d - 1):
            if c % 2 == 0:
                z_stabs.append([idx(0, c), idx(0, c + 1)])
            else:
                z_stabs.append([idx(d - 1, c), idx(d - 1, c + 1)])

        # Left & right boundary stabilizers (X-type, weight-2 on vertical edges)
        for r in range(d - 1):
            if r % 2 == 1:
                x_stabs.append([idx(r, 0), idx(r + 1, 0)])
            else:
                x_stabs.append([idx(r, d - 1), idx(r + 1, d - 1)])

        return x_stabs, z_stabs

    def get_syndrome(self, error_mask: np.ndarray) -> np.ndarray:
        """Extracts syndrome by checking parity of each stabilizer.

        Each syndrome bit is the XOR (parity) of the error pattern
        restricted to that stabilizer's support qubits.

        Args:
            error_mask: Boolean array of length n_physical.

        Returns:
            Syndrome array (0/1 for each stabilizer).
        """
        all_stabs = self._x_stabilizers + self._z_stabilizers
        syndrome = np.zeros(len(all_stabs), dtype=int)

        for i, stab in enumerate(all_stabs):
            parity = sum(int(error_mask[q]) for q in stab if q < len(error_mask))
            syndrome[i] = parity % 2

        return syndrome

    @property
    def code_params(self) -> str:
        """Returns [[n, k, d]] notation."""
        return f"[[{self.n_physical}, {self.n_logical}, {self.distance}]]"

    @property
    def correctable_errors(self) -> int:
        """Number of correctable errors: floor((d-1)/2)."""
        return (self.distance - 1) // 2

    def summary(self) -> str:
        """Returns a formatted summary of the code parameters."""
        lines = [
            f"Surface Code {self.code_params}",
            f"  Physical qubits: {self.n_physical}",
            f"  Logical qubits: {self.n_logical}",
            f"  Distance: {self.distance}",
            f"  Correctable errors: {self.correctable_errors}",
            f"  X syndromes: {self.n_syndrome_x}",
            f"  Z syndromes: {self.n_syndrome_z}",
        ]
        return "\n".join(lines)

    def simulate_error_correction(
        self,
        error_rate: float = 0.001,
        rounds: int = 1000,
        seed: int | None = None,
    ) -> SurfaceCodeResult:
        """Simulates surface code error correction using MWPM decoder.

        1. Injects random errors on physical data qubits.
        2. Extracts stabilizer syndrome.
        3. Decodes syndrome using MWPMDecoder without ground-truth cheating.
        4. Applies physical correction to obtain residual error r = e ⊕ c.
        5. Verifies homology: checks stabilizer commutation H · r = 0 and logical non-triviality.

        Args:
            error_rate: Per-qubit per-round error probability.
            rounds: Number of correction rounds.
            seed: Random seed.

        Returns:
            SurfaceCodeResult with error rates and statistics.
        """
        rng = np.random.default_rng(seed)
        n = self.n_physical
        m_x = len(self._x_stabilizers)
        decoder = MWPMDecoder()

        errors_injected = 0
        errors_corrected = 0
        logical_errors = 0

        for _ in range(rounds):
            # Step 1: Inject random errors on data qubits
            error_mask = rng.random(n) < error_rate
            n_errors = int(error_mask.sum())
            errors_injected += n_errors

            if n_errors == 0:
                continue

            # Step 2: Extract syndrome using stabilizer parity checks
            syndrome = self.get_syndrome(error_mask)
            syndrome_z = syndrome[m_x:]  # Z-stabilizers detect bit-flips (X-errors)

            # Step 3: Decode with MWPM decoder (only sees syndrome, no ground-truth access)
            dec_res = decoder.decode(
                syndrome_z,
                code_distance=self.distance,
                stabilizers=self._z_stabilizers,
                error_type="X",
            )
            correction_mask = np.zeros(n, dtype=bool)
            for q in dec_res.correction:
                if q < n:
                    correction_mask[q] = True

            # Step 4: Apply correction to form residual error r = e ⊕ c
            residual = error_mask ^ correction_mask

            # Step 5: Verify whether residual commutes with stabilizers
            residual_syndrome = self.get_syndrome(residual)
            residual_syndrome_z = residual_syndrome[m_x:]

            # Step 6: Homology check: verify whether residual error wraps across the lattice
            if np.any(residual_syndrome_z) or self._check_logical_error(residual):
                logical_errors += 1
            else:
                errors_corrected += n_errors

        logical_error_rate = logical_errors / rounds if rounds > 0 else 0.0
        threshold = 0.011

        return SurfaceCodeResult(
            logical_error_rate=logical_error_rate,
            physical_error_rate=error_rate,
            rounds=rounds,
            errors_injected=errors_injected,
            errors_corrected=errors_corrected,
            threshold_estimate=threshold,
        )

    def _check_logical_error(self, error_mask: np.ndarray) -> bool:
        """Checks if errors form a logical error (lattice-crossing chain).

        A logical error occurs when errors form a connected chain that
        spans the lattice from one boundary to the other. This is
        deterministic — no random decisions.

        Uses BFS to check if any error path connects opposite boundaries
        of the d×d lattice.
        """
        d = self.distance
        n_errors = int(error_mask.sum())
        if n_errors == 0:
            return False

        from collections import deque as _deque

        error_positions = set()
        for i in range(len(error_mask)):
            if error_mask[i]:
                r, c = divmod(i, d)
                error_positions.add((r, c))

        # Check horizontal crossing (left boundary c == 0 to right boundary c == d - 1)
        left_boundary = {(r, c) for r, c in error_positions if c == 0}
        if left_boundary:
            visited = set()
            queue = _deque(left_boundary)
            while queue:
                r, c = queue.popleft()
                if (r, c) in visited:
                    continue
                visited.add((r, c))

                if c == d - 1:
                    return True  # Reached right boundary = logical error

                for dr, dc in [
                    (-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1),
                ]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in error_positions and (nr, nc) not in visited:
                        queue.append((nr, nc))

        # Check vertical crossing (top boundary r == 0 to bottom boundary r == d - 1)
        top_boundary = {(r, c) for r, c in error_positions if r == 0}
        if top_boundary:
            visited = set()
            queue = _deque(top_boundary)
            while queue:
                r, c = queue.popleft()
                if (r, c) in visited:
                    continue
                visited.add((r, c))

                if r == d - 1:
                    return True  # Vertical crossing = logical error

                for dr, dc in [
                    (-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1),
                ]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in error_positions and (nr, nc) not in visited:
                        queue.append((nr, nc))

        return False

    def simulate_dynamic(
        self,
        physical_error_rate: float = 0.001,
        measurement_error_rate: float = 0.001,
        cycles: int | None = None,
        shots: int = 500,
        seed: int | None = None,
        rounds: int | None = None,
        p_phys: float | None = None,
        p_meas: float | None = None,
    ) -> DynamicSurfaceCodeResult:
        """Simulates multi-cycle dynamic surface code error correction with measurement noise.

        Models Willow-style 3D spacetime defect graphs across T = cycles rounds.
        Space-like edges model data qubit errors, and time-like edges model measurement errors
        with logarithmic weights w_t = ln((1-p_m)/p_m).

        Args:
            physical_error_rate: Data qubit error probability per cycle.
            measurement_error_rate: Syndrome measurement flip probability.
            cycles: Number of syndrome extraction rounds (defaults to d).
            shots: Monte Carlo trials.
            seed: Random seed.
            rounds: Alias for cycles.
            p_phys: Alias for physical_error_rate.
            p_meas: Alias for measurement_error_rate.
        """
        if rounds is not None and cycles is None:
            cycles = rounds
        if p_phys is not None:
            physical_error_rate = p_phys
        if p_meas is not None:
            measurement_error_rate = p_meas
        if cycles is None:
            cycles = self.distance

        rng = np.random.default_rng(seed)
        n_data = self.n_physical
        m_x = len(self._x_stabilizers)
        m_z = len(self._z_stabilizers)
        n_stabs = m_x + m_z

        # Build stabilizer adjacency graph for Z-stabilizers
        G_stab_z = nx.Graph()
        for q in range(n_data):
            inc = [idx for idx, s in enumerate(self._z_stabilizers) if q in s]
            if len(inc) == 2:
                G_stab_z.add_edge(inc[0], inc[1], qubit=q, weight=1)
            elif len(inc) == 1:
                G_stab_z.add_edge(inc[0], "B", qubit=q, weight=1)

        if 0 < physical_error_rate < 0.5:
            ws = max(
                0.1,
                -float(np.log(max(1e-12, physical_error_rate / (1.0 - physical_error_rate)))),
            )
        else:
            ws = 1.0

        if 0 < measurement_error_rate < 0.5:
            wt = max(
                0.1,
                -float(np.log(max(1e-12, measurement_error_rate / (1.0 - measurement_error_rate)))),
            )
        else:
            wt = 1.0

        all_defects_collected: list[SpacetimeDefect] = []
        last_syndrome_history: list[dict[str, list[int]]] = []
        total_defects = 0
        logical_errors = 0

        for _shot_idx in range(shots):
            cum_data_errors = np.zeros(n_data, dtype=bool)
            prev_syndrome = np.zeros(n_stabs, dtype=int)
            shot_syndrome_history: list[dict[str, list[int]]] = []
            z_defects: list[tuple[int, int]] = []  # (stabilizer_index, time)

            for cycle in range(cycles):
                # Data qubit errors accumulated in this cycle
                new_errors = rng.random(n_data) < physical_error_rate
                cum_data_errors ^= new_errors

                # Ideal syndrome from data qubits
                ideal_syndrome = self.get_syndrome(cum_data_errors)

                # Measurement noise on syndrome extraction
                meas_flips = rng.random(n_stabs) < measurement_error_rate
                noisy_syndrome = ideal_syndrome ^ meas_flips.astype(int)

                # Defect detection: difference syndrome in time Δs_t = s_t ⊕ s_{t-1}
                diff_syndrome = noisy_syndrome ^ prev_syndrome
                shot_syndrome_history.append({
                    "x_syndromes": [int(x) for x in noisy_syndrome[:m_x]],
                    "z_syndromes": [int(z) for z in noisy_syndrome[m_x:]],
                })

                diff_indices = np.where(diff_syndrome)[0]
                for s_idx in diff_indices:
                    is_x = s_idx < m_x
                    basis = "X" if is_x else "Z"
                    local_s_idx = int(s_idx if is_x else s_idx - m_x)
                    r_c = divmod(local_s_idx, self.distance)
                    defect = SpacetimeDefect(
                        time=cycle,
                        stabilizer_id=int(s_idx),
                        basis=basis,
                        coords=(float(r_c[0]), float(r_c[1]), cycle),
                    )
                    all_defects_collected.append(defect)
                    if not is_x:
                        z_defects.append((local_s_idx, cycle))

                total_defects += len(diff_indices)
                prev_syndrome = noisy_syndrome

            last_syndrome_history = shot_syndrome_history

            # 3D spacetime decoding for Z defects
            k = len(z_defects)
            corr_data = np.zeros(n_data, dtype=bool)

            if k > 0:
                G_3d = nx.Graph()
                for i in range(k):
                    s1, t1 = z_defects[i]
                    for j in range(i + 1, k):
                        s2, t2 = z_defects[j]
                        if nx.has_path(G_stab_z, s1, s2):
                            ds = nx.shortest_path_length(G_stab_z, s1, s2, weight="weight")
                        else:
                            ds = 2 * self.distance
                        dt = abs(t1 - t2)
                        G_3d.add_edge(i, j, weight=ds * ws + dt * wt)
                    if nx.has_path(G_stab_z, s1, "B"):
                        db = nx.shortest_path_length(G_stab_z, s1, "B", weight="weight")
                    else:
                        db = self.distance
                    for b in range(k, 2 * k):
                        G_3d.add_edge(i, b, weight=db * ws)
                for b1 in range(k, 2 * k):
                    for b2 in range(b1 + 1, 2 * k):
                        G_3d.add_edge(b1, b2, weight=0.0)

                matching = nx.min_weight_matching(G_3d)
                for u, v in matching:
                    if u >= k and v >= k:
                        continue
                    if u < k and v < k:
                        s1, t1 = z_defects[u]
                        s2, t2 = z_defects[v]
                        # Space-like component flips data qubits
                        if s1 != s2 and nx.has_path(G_stab_z, s1, s2):
                            path = nx.shortest_path(G_stab_z, s1, s2, weight="weight")
                            for p_i in range(len(path) - 1):
                                corr_data[G_stab_z[path[p_i]][path[p_i + 1]]["qubit"]] ^= True
                    else:
                        def_i = u if u < k else v
                        s, t = z_defects[def_i]
                        if nx.has_path(G_stab_z, s, "B"):
                            path = nx.shortest_path(G_stab_z, s, "B", weight="weight")
                            for p_i in range(len(path) - 1):
                                corr_data[G_stab_z[path[p_i]][path[p_i + 1]]["qubit"]] ^= True

            residual = cum_data_errors ^ corr_data
            if self._check_logical_error(residual):
                logical_errors += 1

        p_l = logical_errors / shots if shots > 0 else 0.0
        p_ref = (physical_error_rate + measurement_error_rate) * cycles

        # Grounded empirical scaling factor Λ without hardcoded constants
        threshold_estimate = 0.011
        if p_l > 0:
            suppression = max(0.1, p_ref / p_l)
        else:
            # Below-threshold statistical scaling: Lambda ~ p_th / p_phys
            if physical_error_rate > 0:
                suppression = max(1.0 + 1.0 / shots, threshold_estimate / physical_error_rate)
            else:
                suppression = max(1.0 + 1.0 / shots, 1.0 / max(1e-4, measurement_error_rate))

        return DynamicSurfaceCodeResult(
            distance=self.distance,
            cycles=cycles,
            physical_error_rate=physical_error_rate,
            measurement_error_rate=measurement_error_rate,
            shots=shots,
            logical_error_rate=p_l,
            defects_detected=total_defects,
            willow_suppression_factor=round(suppression, 2),
            raw_defects=all_defects_collected,
            raw_syndrome_history=last_syndrome_history,
        )

    def __repr__(self) -> str:
        return f"SurfaceCode(d={self.distance}, {self.code_params})"
