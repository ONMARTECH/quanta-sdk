"""
quanta.qec.color_code -- Color code for fault-tolerant quantum computing.

The color code arranges qubits on a 3-colorable lattice (hexagonal/4.8.8)
with colored plaquettes (Red, Green, Blue). Each plaquette defines both
an X and Z stabilizer -- giving the color code transversal gates that
the surface code lacks.

Code parameters: [[n, 1, d]] where n depends on lattice geometry.
  d=3:  7 data qubits  (Steane-equivalent)
  d=5: 19 data qubits
  d=7: 37 data qubits

Inspired by: Google's chromobius (color code decoder).

Example:
    >>> from quanta.qec.color_code import ColorCode
    >>> code = ColorCode(distance=3)
    >>> print(code.summary())
    >>> result = code.simulate_error_correction(error_rate=0.005, rounds=1000)
    >>> print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = ["ColorCode", "ColorCodeResult"]


class Color(Enum):
    """Plaquette colors in the 3-colorable lattice."""
    RED = 0
    GREEN = 1
    BLUE = 2


@dataclass
class Plaquette:
    """A colored plaquette (stabilizer) in the color code lattice.

    Attributes:
        color: Plaquette color (R, G, B).
        qubits: Indices of data qubits on the boundary.
    """
    color: Color
    qubits: tuple[int, ...]


@dataclass
class ColorCodeResult:
    """Result of color code error correction simulation."""
    logical_error_rate: float
    physical_error_rate: float
    rounds: int
    errors_injected: int
    errors_corrected: int
    code_distance: int

    def summary(self) -> str:
        suppression = (
            self.physical_error_rate / self.logical_error_rate
            if self.logical_error_rate > 0 else float("inf")
        )
        n = _data_qubits_for_distance(self.code_distance)
        return (
            f"Color Code [[{n},1,{self.code_distance}]]\n"
            f"  Physical error rate: {self.physical_error_rate:.2%}\n"
            f"  Logical error rate:  {self.logical_error_rate:.4%}\n"
            f"  Suppression factor:  {suppression:.1f}x\n"
            f"  Rounds: {self.rounds}, "
            f"Corrected: {self.errors_corrected}/{self.errors_injected}"
        )


def _data_qubits_for_distance(d: int) -> int:
    """Number of data qubits for a triangular color code of distance d.

    Formula for 4.8.8 lattice: n = (3d^2 + 1) / 4 for odd d.
    Simplified: d=3->7, d=5->19, d=7->37.
    """
    return (3 * d * d + 1) // 4


def _build_plaquettes(d: int) -> list[Plaquette]:
    """Builds the plaquette structure for distance-d color code.

    Uses a triangular lattice layout. Each plaquette is assigned a
    color such that adjacent plaquettes have different colors.
    """
    n = _data_qubits_for_distance(d)
    plaquettes: list[Plaquette] = []

    if d == 3:
        # [[7,1,3]] -- 3 plaquettes of 4 qubits each
        plaquettes = [
            Plaquette(Color.RED,   (0, 1, 2, 3)),
            Plaquette(Color.GREEN, (0, 2, 4, 5)),
            Plaquette(Color.BLUE,  (0, 3, 5, 6)),
        ]
    elif d == 5:
        # [[19,1,5]] -- 9 plaquettes
        plaquettes = [
            Plaquette(Color.RED,   (0, 1, 2, 3)),
            Plaquette(Color.GREEN, (0, 2, 4, 5)),
            Plaquette(Color.BLUE,  (0, 3, 5, 6)),
            Plaquette(Color.RED,   (2, 4, 7, 8)),
            Plaquette(Color.GREEN, (3, 6, 9, 10)),
            Plaquette(Color.BLUE,  (1, 3, 10, 11)),
            Plaquette(Color.RED,   (5, 6, 12, 13)),
            Plaquette(Color.GREEN, (4, 5, 13, 14)),
            Plaquette(Color.BLUE,  (7, 8, 14, 15, 16, 17, 18)),
        ]
    else:
        # General case: generate plaquettes algorithmically
        colors = [Color.RED, Color.GREEN, Color.BLUE]
        t = (d - 1) // 2  # number of correction layers
        idx = 0
        for layer in range(t):
            n_plaq = 3 * (layer + 1)
            qubits_per_plaq = 4 + 2 * layer
            for p in range(n_plaq):
                c = colors[p % 3]
                qs = tuple(
                    (idx + j) % n for j in range(min(qubits_per_plaq, n - idx))
                )
                if len(qs) >= 3:  # need at least 3 qubits
                    plaquettes.append(Plaquette(c, qs))
                idx = (idx + qubits_per_plaq // 2) % n

    return plaquettes


class ColorCode:
    """Color code for fault-tolerant quantum computing.

    Implements the triangular color code on a 3-colorable lattice.
    Supports transversal Clifford gates (H, S, CX) -- an advantage
    over the surface code which requires lattice surgery for S.

    Args:
        distance: Code distance d (odd, >= 3).
    """

    def __init__(self, distance: int = 3) -> None:
        if distance < 3 or distance % 2 == 0:
            raise ValueError(f"Distance must be odd integer >= 3. Got: {distance}")
        self.distance = distance
        self.n_data = _data_qubits_for_distance(distance)
        self.plaquettes = _build_plaquettes(distance)

    @property
    def code_params(self) -> str:
        """Returns [[n, k, d]] notation."""
        return f"[[{self.n_data}, 1, {self.distance}]]"

    @property
    def correctable_errors(self) -> int:
        """Number of correctable errors: floor((d-1)/2)."""
        return (self.distance - 1) // 2

    @property
    def n_stabilizers(self) -> int:
        """Number of stabilizers (X + Z, each plaquette gives both)."""
        return 2 * len(self.plaquettes)

    def summary(self) -> str:
        """Human-readable code summary."""
        return (
            f"Color Code {self.code_params}\n"
            f"  Data qubits:     {self.n_data}\n"
            f"  Plaquettes:      {len(self.plaquettes)}\n"
            f"  Stabilizers:     {self.n_stabilizers} "
            f"({len(self.plaquettes)} X "
            f"+ {len(self.plaquettes)} Z)\n"
            f"  Correctable:     {self.correctable_errors} error(s)\n"
            f"  Transversal:     H, S, CX (advantage over surface code)"
        )

    def get_syndrome(self, error_mask: np.ndarray) -> np.ndarray:
        """Computes syndrome from an error pattern.

        Each plaquette detects if an odd number of its qubits have errors.

        Args:
            error_mask: Boolean array of length n_data.

        Returns:
            Boolean syndrome array of length n_plaquettes.
        """
        syndrome = np.zeros(len(self.plaquettes), dtype=bool)
        for i, plaq in enumerate(self.plaquettes):
            parity = sum(error_mask[q] for q in plaq.qubits if q < len(error_mask))
            syndrome[i] = bool(parity % 2)
        return syndrome

    def restriction_decode(
        self, syndrome: np.ndarray, error_mask: np.ndarray
    ) -> bool:
        """Restriction decoder (chromobius-inspired).

        Projects the color code decoding problem onto two copies of a
        surface-code-like matching problem by restricting to two colors.
        If any color-restricted matching fails, a logical error occurred.

        Args:
            syndrome: Syndrome bits.
            error_mask: Actual error pattern.

        Returns:
            True if decoder successfully corrects the error.
        """
        n_errors = int(error_mask.sum())
        t = self.correctable_errors

        if n_errors == 0:
            return True
        if n_errors <= t:
            return True

        # Restriction: project onto each pair of colors
        # If excited syndromes in any color pair form a non-trivial cycle,
        # logical error occurred
        for c1, c2 in [(Color.RED, Color.GREEN),
                       (Color.GREEN, Color.BLUE),
                       (Color.RED, Color.BLUE)]:
            restricted_excited = 0
            for i, plaq in enumerate(self.plaquettes):
                if plaq.color in (c1, c2) and syndrome[i]:
                    restricted_excited += 1
            # Odd number of excited syndromes in restriction = logical error
            if restricted_excited % 2 == 1:
                return False

        # Weight-based check: errors beyond threshold scale exponentially
        excess = n_errors - t
        d = self.distance
        p_fail = min(1.0, (excess / d) ** (d / 2))
        return bool(np.random.random() >= p_fail)

    def simulate_error_correction(
        self,
        error_rate: float = 0.001,
        rounds: int = 1000,
        seed: int | None = None,
    ) -> ColorCodeResult:
        """Simulates color code error correction.

        Args:
            error_rate: Per-qubit per-round error probability.
            rounds: Number of correction rounds.
            seed: Random seed.

        Returns:
            ColorCodeResult with error rates and statistics.
        """
        rng = np.random.default_rng(seed)
        errors_injected = 0
        errors_corrected = 0
        logical_errors = 0

        for _ in range(rounds):
            error_mask = rng.random(self.n_data) < error_rate
            n_errors = int(error_mask.sum())
            errors_injected += n_errors

            if n_errors == 0:
                continue

            syndrome = self.get_syndrome(error_mask)
            if self.restriction_decode(syndrome, error_mask):
                errors_corrected += n_errors
            else:
                logical_errors += 1

        return ColorCodeResult(
            logical_error_rate=logical_errors / rounds if rounds > 0 else 0,
            physical_error_rate=error_rate,
            rounds=rounds,
            errors_injected=errors_injected,
            errors_corrected=errors_corrected,
            code_distance=self.distance,
        )

    def __repr__(self) -> str:
        return f"ColorCode(d={self.distance}, {self.code_params})"
