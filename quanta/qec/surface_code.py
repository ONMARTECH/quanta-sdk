"""
quanta.qec.surface_code -- Rotated surface code for fault-tolerant logical qubits.

The rotated [[d^2, 1, d]] surface code arranges d^2 physical data qubits on a
2D grid with (d^2 - 1)/2 weight-2/4 X-stabilizers and (d^2 - 1)/2 weight-2/4
Z-stabilizers, encoding 1 logical qubit with exact GF(2) homology operators
X_bar (top row) and Z_bar (left column).

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

from quanta.qec.decoder import DecoderBase, MWPMDecoder, UnionFindDecoder

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
    """Result of multi-round dynamic surface code simulation (3D phenomenological).

    Attributes:
        distance: Code distance d.
        cycles: Number of noisy syndrome extraction cycles (T).
        physical_error_rate: Physical data qubit error probability per cycle.
        measurement_error_rate: Syndrome measurement error probability.
        shots: Number of Monte Carlo shots.
        logical_error_rate: Uncorrected logical error rate after T cycles + readout.
        defects_detected: Total spacetime detection events (Δs_t = s_t ⊕ s_{t-1}).
        willow_suppression_factor: Estimated Λ scaling factor.
        raw_defects: Genuine spacetime detection events recorded during simulation.
        raw_syndrome_history: Genuine multi-cycle syndrome extractions.
        threshold_estimate: Phenomenological 3D surface code threshold (~0.029).
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
    threshold_estimate: float = 0.029

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
        threshold_estimate: Code-capacity threshold error rate (~0.103 bitflip).
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
            if self.logical_error_rate > 0
            else float("inf")
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
            + " " * 10
            + "║",
            "╚══════════════════════════════════════╝",
        ]
        return "\n".join(lines)


class SurfaceCode:
    """Rotated [[d^2, 1, d]] surface code with exact GF(2) homology verification.

    Args:
        distance: Odd code distance d >= 3. Physical data qubits = d^2.
    """

    def __init__(self, distance: int = 3) -> None:
        if distance < 3 or distance % 2 == 0:
            raise ValueError(
                f"Distance must be odd integer >= 3. Got: {distance}"
            )
        self.distance = distance
        self.n_physical = distance**2
        self.n_logical = 1
        self.n_syndrome_x = (distance**2 - 1) // 2
        self.n_syndrome_z = (distance**2 - 1) // 2

        self._x_stabilizers, self._z_stabilizers = self._build_stabilizers()

        # Binary check matrices over GF(2)
        self.H_X = np.zeros((self.n_syndrome_x, self.n_physical), dtype=np.int8)
        for i, s in enumerate(self._x_stabilizers):
            self.H_X[i, s] = 1

        self.H_Z = np.zeros((self.n_syndrome_z, self.n_physical), dtype=np.int8)
        for i, s in enumerate(self._z_stabilizers):
            self.H_Z[i, s] = 1

        # Minimal logical operators on the rotated d×d lattice:
        #   X_bar = row 0 (top horizontal chain, weight d)
        #   Z_bar = col 0 (left vertical chain, weight d)
        self.logical_x = np.zeros(self.n_physical, dtype=np.int8)
        self.logical_x[[c for c in range(distance)]] = 1

        self.logical_z = np.zeros(self.n_physical, dtype=np.int8)
        self.logical_z[[r * distance for r in range(distance)]] = 1

        # Algebraic verification of CSS commutativity & symplectic logical algebra
        if np.any((self.H_X @ self.H_Z.T) % 2 != 0):
            raise RuntimeError("Rotated surface code CSS commutation H_X H_Z^T = 0 failed.")
        if np.any((self.H_Z @ self.logical_x) % 2 != 0):
            raise RuntimeError("Logical X operator fails H_Z @ X_bar == 0 (mod 2).")
        if np.any((self.H_X @ self.logical_z) % 2 != 0):
            raise RuntimeError("Logical Z operator fails H_X @ Z_bar == 0 (mod 2).")
        if int((self.logical_x @ self.logical_z) % 2) != 1:
            raise RuntimeError("Logical operators fail <X_bar, Z_bar> == 1 (mod 2).")

    def _build_stabilizers(self) -> tuple[list[list[int]], list[list[int]]]:
        """Builds X-type and Z-type stabilizer generators on the rotated d×d grid."""
        d = self.distance
        x_stabs: list[list[int]] = []
        z_stabs: list[list[int]] = []

        def idx(r: int, c: int) -> int:
            return r * d + c

        for r in range(d - 1):
            for c in range(d - 1):
                if (r + c) % 2 == 0:
                    x_stabs.append(
                        [idx(r, c), idx(r, c + 1), idx(r + 1, c), idx(r + 1, c + 1)]
                    )
                else:
                    z_stabs.append(
                        [idx(r, c), idx(r, c + 1), idx(r + 1, c), idx(r + 1, c + 1)]
                    )

        # Top & bottom weight-2 Z-type boundary stabilizers
        for c in range(d - 1):
            if c % 2 == 0:
                z_stabs.append([idx(0, c), idx(0, c + 1)])
            else:
                z_stabs.append([idx(d - 1, c), idx(d - 1, c + 1)])

        # Left & right weight-2 X-type boundary stabilizers
        for r in range(d - 1):
            if r % 2 == 1:
                x_stabs.append([idx(r, 0), idx(r + 1, 0)])
            else:
                x_stabs.append([idx(r, d - 1), idx(r + 1, d - 1)])

        return x_stabs, z_stabs

    def get_syndrome(self, error_mask: np.ndarray) -> np.ndarray:
        """Extracts [X-syndrome, Z-syndrome] parity vector for a bit-vector."""
        err = np.zeros(self.n_physical, dtype=np.int8)
        k = min(len(error_mask), self.n_physical)
        err[:k] = np.asarray(error_mask[:k], dtype=np.int8) & 1
        syn_x = (self.H_X @ err) % 2
        syn_z = (self.H_Z @ err) % 2
        return np.concatenate([syn_x, syn_z]).astype(int)

    def get_css_syndrome(
        self, x_error_mask: np.ndarray, z_error_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extracts (x_syndrome detecting Z-errors, z_syndrome detecting X-errors)."""
        x_err = np.asarray(x_error_mask, dtype=np.int8) & 1
        z_err = np.asarray(z_error_mask, dtype=np.int8) & 1
        syn_x = ((self.H_X @ z_err) % 2).astype(int)
        syn_z = ((self.H_Z @ x_err) % 2).astype(int)
        return syn_x, syn_z

    @property
    def code_params(self) -> str:
        """Returns [[n, k, d]] notation."""
        return f"[[{self.n_physical}, {self.n_logical}, {self.distance}]]"

    @property
    def correctable_errors(self) -> int:
        """Number of guaranteed correctable errors: floor((d-1)/2)."""
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

    def _check_logical_error(
        self, error_mask: np.ndarray, error_type: str = "X"
    ) -> bool:
        """Exact GF(2) homology test for residual Pauli errors.

        An X-type residual r_X is a logical failure iff:
          - H_Z r_X != 0 (mod 2)  [uncorrected Z-syndrome defect], OR
          - <r_X, Z_bar> == 1 (mod 2)  [non-trivial logical X homology cycle].

        Because H_Z H_X^T = 0 and H_X Z_bar = 0 over GF(2), EVERY product of
        X-stabilizers satisfies both conditions identically and has 0% false
        positive rate. Symmetrically for Z-type residuals against H_X and X_bar.
        """
        r = np.asarray(error_mask, dtype=np.int8).ravel() & 1
        if len(r) < self.n_physical:
            padded = np.zeros(self.n_physical, dtype=np.int8)
            padded[: len(r)] = r
            r = padded
        elif len(r) > self.n_physical:
            r = r[: self.n_physical]

        if not np.any(r):
            return False

        if error_type.upper() == "Z":
            if np.any((self.H_X @ r) % 2 != 0):
                return True
            return int((self.logical_x @ r) % 2) == 1

        if np.any((self.H_Z @ r) % 2 != 0):
            return True
        return int((self.logical_z @ r) % 2) == 1

    def simulate_error_correction(
        self,
        error_rate: float = 0.001,
        rounds: int = 1000,
        seed: int | None = None,
        channel: str = "bitflip",
        decoder: str | DecoderBase = "mwpm",
    ) -> SurfaceCodeResult:
        """Simulates code-capacity surface code error correction with exact homology.

        Args:
            error_rate: Per-qubit error probability p.
            rounds: Number of Monte Carlo rounds.
            seed: Random seed.
            channel: 'bitflip' ('X'), 'phaseflip' ('Z'), 'depolarizing', or 'independent_xz'.
            decoder: 'mwpm', 'union_find', or a DecoderBase instance.
        """
        rng = np.random.default_rng(seed)
        n = self.n_physical

        if isinstance(decoder, DecoderBase):
            dec_obj: DecoderBase = decoder
        elif decoder.lower() in ("uf", "union_find", "union-find"):
            dec_obj = UnionFindDecoder()
        else:
            dec_obj = MWPMDecoder()

        ch = channel.lower()
        if ch in ("bitflip", "x", "phaseflip", "z", "independent_xz"):
            threshold = 0.103
        elif ch == "depolarizing":
            threshold = 0.155
        else:
            raise ValueError(
                f"Unknown channel '{channel}'. Use 'bitflip', 'phaseflip', "
                f"'depolarizing', or 'independent_xz'."
            )

        errors_injected = 0
        errors_corrected = 0
        logical_errors = 0

        for _ in range(rounds):
            if ch in ("bitflip", "x"):
                x_err = rng.random(n) < error_rate
                z_err = np.zeros(n, dtype=bool)
                n_errors = int(x_err.sum())
            elif ch in ("phaseflip", "z"):
                x_err = np.zeros(n, dtype=bool)
                z_err = rng.random(n) < error_rate
                n_errors = int(z_err.sum())
            elif ch == "depolarizing":
                u = rng.random(n)
                x_err = (u < 2.0 * error_rate / 3.0)  # X (p/3) or Y (p/3)
                z_err = (u >= error_rate / 3.0) & (u < error_rate)  # Y (p/3) or Z (p/3)
                n_errors = int((x_err | z_err).sum())
            else:  # independent_xz
                x_err = rng.random(n) < error_rate
                z_err = rng.random(n) < error_rate
                n_errors = int((x_err | z_err).sum())

            errors_injected += n_errors
            if n_errors == 0:
                continue

            failed = False

            if np.any(x_err):
                syn_z = ((self.H_Z @ x_err.astype(np.int8)) % 2).astype(bool)
                dec_x = dec_obj.decode(
                    syn_z,
                    code_distance=self.distance,
                    stabilizers=self._z_stabilizers,
                    error_type="X",
                )
                corr_x = np.zeros(n, dtype=bool)
                for q in dec_x.correction:
                    if q < n:
                        corr_x[q] = True
                res_x = x_err ^ corr_x
                if self._check_logical_error(res_x, "X"):
                    failed = True

            if not failed and np.any(z_err):
                syn_x = ((self.H_X @ z_err.astype(np.int8)) % 2).astype(bool)
                dec_z = dec_obj.decode(
                    syn_x,
                    code_distance=self.distance,
                    stabilizers=self._x_stabilizers,
                    error_type="Z",
                )
                corr_z = np.zeros(n, dtype=bool)
                for q in dec_z.correction:
                    if q < n:
                        corr_z[q] = True
                res_z = z_err ^ corr_z
                if self._check_logical_error(res_z, "Z"):
                    failed = True

            if failed:
                logical_errors += 1
            else:
                errors_corrected += n_errors

        logical_error_rate = logical_errors / rounds if rounds > 0 else 0.0

        return SurfaceCodeResult(
            logical_error_rate=logical_error_rate,
            physical_error_rate=error_rate,
            rounds=rounds,
            errors_injected=errors_injected,
            errors_corrected=errors_corrected,
            threshold_estimate=threshold,
        )

    @staticmethod
    def _build_2d_stab_graph(
        stabilizers: list[list[int]], n_data: int
    ) -> tuple[nx.Graph, dict[int | str, dict[int | str, int]], dict[int | str, dict[int | str, list[int | str]]]]:
        """Builds stabilizer graph and all-pairs shortest paths for 3D matching."""
        G_stab = nx.Graph()
        for idx in range(len(stabilizers)):
            G_stab.add_node(idx)
        G_stab.add_node("B")
        for q in range(n_data):
            inc = [idx for idx, s in enumerate(stabilizers) if q in s]
            if len(inc) == 2 and not G_stab.has_edge(inc[0], inc[1]):
                G_stab.add_edge(inc[0], inc[1], qubit=q, weight=1)
            elif len(inc) == 1 and not G_stab.has_edge(inc[0], "B"):
                G_stab.add_edge(inc[0], "B", qubit=q, weight=1)
        dists = {u: dict(nx.single_source_shortest_path_length(G_stab, u)) for u in G_stab.nodes}
        paths = {u: dict(nx.single_source_shortest_path(G_stab, u)) for u in G_stab.nodes}
        return G_stab, dists, paths

    def _decode_3d_spacetime(
        self,
        defects: list[tuple[int, int]],
        G_stab: nx.Graph,
        dists: dict[int | str, dict[int | str, int]],
        paths: dict[int | str, dict[int | str, list[int | str]]],
        ws: float,
        wt: float,
    ) -> np.ndarray:
        """Matches 3D spacetime defects (s_idx, t) and projects space-like edges mod 2."""
        corr_data = np.zeros(self.n_physical, dtype=bool)
        k = len(defects)
        if k == 0:
            return corr_data

        G_3d = nx.Graph()
        big_d = 2 * self.distance
        for i in range(k):
            s1, t1 = defects[i]
            for j in range(i + 1, k):
                s2, t2 = defects[j]
                ds = dists[s1].get(s2, big_d)
                dt = abs(t1 - t2)
                G_3d.add_edge(i, j, weight=ds * ws + dt * wt)

            db = dists[s1].get("B", self.distance)
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
                s1, _ = defects[u]
                s2, _ = defects[v]
                if s1 != s2 and s2 in paths[s1]:
                    sp = paths[s1][s2]
                    for step in range(len(sp) - 1):
                        q = int(G_stab[sp[step]][sp[step + 1]]["qubit"])
                        corr_data[q] = not corr_data[q]
            else:
                real_idx = u if u < k else v
                s1, _ = defects[real_idx]
                if "B" in paths[s1]:
                    sp = paths[s1]["B"]
                    for step in range(len(sp) - 1):
                        q = int(G_stab[sp[step]][sp[step + 1]]["qubit"])
                        corr_data[q] = not corr_data[q]

        return corr_data

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
        channel: str = "bitflip",
    ) -> DynamicSurfaceCodeResult:
        """Simulates multi-cycle phenomenological 3D surface code QEC.

        Executes `cycles` rounds of noisy data accumulation + noisy syndrome
        extraction, followed by a mandatory final noiseless destructive data-readout
        syndrome difference round at `t = cycles` before 3D Blossom matching so
        measurement flips at `t = cycles - 1` pair across time instead of injecting
        boundary physical errors.
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
        m_x = self.n_syndrome_x
        m_z = self.n_syndrome_z
        n_stabs = m_x + m_z
        ch = channel.lower()

        G_stab_z, dists_z, paths_z = self._build_2d_stab_graph(self._z_stabilizers, n_data)
        G_stab_x, dists_x, paths_x = self._build_2d_stab_graph(self._x_stabilizers, n_data)

        if physical_error_rate <= 0.0:
            ws = 1e6
        elif physical_error_rate < 0.5:
            ws = max(
                0.1,
                -float(np.log(max(1e-12, physical_error_rate / (1.0 - physical_error_rate)))),
            )
        else:
            ws = 0.05

        if measurement_error_rate <= 0.0:
            wt = 1e6
        elif measurement_error_rate < 0.5:
            wt = max(
                0.1,
                -float(
                    np.log(max(1e-12, measurement_error_rate / (1.0 - measurement_error_rate)))
                ),
            )
        else:
            wt = 0.05

        all_defects_collected: list[SpacetimeDefect] = []
        last_syndrome_history: list[dict[str, list[int]]] = []
        total_defects = 0
        logical_errors = 0

        for _shot_idx in range(shots):
            cum_x_errors = np.zeros(n_data, dtype=bool)
            cum_z_errors = np.zeros(n_data, dtype=bool)
            prev_syndrome = np.zeros(n_stabs, dtype=int)
            shot_syndrome_history: list[dict[str, list[int]]] = []
            z_defects: list[tuple[int, int]] = []
            x_defects: list[tuple[int, int]] = []

            for cycle in range(cycles):
                if ch == "depolarizing":
                    u = rng.random(n_data)
                    cum_x_errors ^= u < 2.0 * physical_error_rate / 3.0
                    cum_z_errors ^= (u >= physical_error_rate / 3.0) & (
                        u < physical_error_rate
                    )
                    syn_x, syn_z = self.get_css_syndrome(cum_x_errors, cum_z_errors)
                    ideal_syndrome = np.concatenate([syn_x, syn_z])
                elif ch in ("independent_xz",):
                    cum_x_errors ^= rng.random(n_data) < physical_error_rate
                    cum_z_errors ^= rng.random(n_data) < physical_error_rate
                    syn_x, syn_z = self.get_css_syndrome(cum_x_errors, cum_z_errors)
                    ideal_syndrome = np.concatenate([syn_x, syn_z])
                else:  # bitflip (default)
                    cum_x_errors ^= rng.random(n_data) < physical_error_rate
                    ideal_syndrome = self.get_syndrome(cum_x_errors)

                meas_flips = rng.random(n_stabs) < measurement_error_rate
                noisy_syndrome = ideal_syndrome ^ meas_flips.astype(int)

                diff_syndrome = noisy_syndrome ^ prev_syndrome
                shot_syndrome_history.append(
                    {
                        "x_syndromes": [int(x) for x in noisy_syndrome[:m_x]],
                        "z_syndromes": [int(z) for z in noisy_syndrome[m_x:]],
                    }
                )

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
                    if is_x:
                        x_defects.append((local_s_idx, cycle))
                    else:
                        z_defects.append((local_s_idx, cycle))

                total_defects += len(diff_indices)
                prev_syndrome = noisy_syndrome

            # Mandatory final noiseless destructive data-readout syndrome difference at t = cycles
            if ch in ("depolarizing", "independent_xz"):
                fin_syn_x, fin_syn_z = self.get_css_syndrome(cum_x_errors, cum_z_errors)
                final_ideal_syndrome = np.concatenate([fin_syn_x, fin_syn_z])
            else:
                final_ideal_syndrome = self.get_syndrome(cum_x_errors)

            final_diff = final_ideal_syndrome ^ prev_syndrome
            for s_idx in np.where(final_diff)[0]:
                if s_idx < m_x:
                    x_defects.append((int(s_idx), cycles))
                else:
                    z_defects.append((int(s_idx - m_x), cycles))

            last_syndrome_history = shot_syndrome_history

            corr_x = self._decode_3d_spacetime(z_defects, G_stab_z, dists_z, paths_z, ws, wt)
            res_x = cum_x_errors ^ corr_x
            failed = self._check_logical_error(res_x, "X")

            if not failed and ch in ("depolarizing", "independent_xz"):
                corr_z = self._decode_3d_spacetime(
                    x_defects, G_stab_x, dists_x, paths_x, ws, wt
                )
                res_z = cum_z_errors ^ corr_z
                failed = self._check_logical_error(res_z, "Z")

            if failed:
                logical_errors += 1

        logical_error_rate = logical_errors / shots if shots > 0 else 0.0

        p_eff = max(physical_error_rate, 1e-6)
        per_cycle_err = logical_error_rate / max(1, cycles)
        if per_cycle_err == 0:
            p_th = 0.029
            lambda_est = float(min(100.0, max(1.1, p_th / p_eff)))
        elif per_cycle_err < p_eff:
            exponent = 2.0 / (self.distance + 1)
            lambda_est = float((p_eff / max(per_cycle_err, 1e-9)) ** exponent)
            lambda_est = max(1.01, lambda_est)
        else:
            lambda_est = float(p_eff / per_cycle_err)

        return DynamicSurfaceCodeResult(
            distance=self.distance,
            cycles=cycles,
            physical_error_rate=physical_error_rate,
            measurement_error_rate=measurement_error_rate,
            shots=shots,
            logical_error_rate=logical_error_rate,
            defects_detected=total_defects,
            willow_suppression_factor=round(lambda_est, 4),
            raw_defects=all_defects_collected[:200],
            raw_syndrome_history=last_syndrome_history,
            threshold_estimate=0.029,
        )
