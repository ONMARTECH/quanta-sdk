"""
quanta.simulator.pauli_frame -- Pauli frame simulator for Clifford circuits.

A fast, vectorized simulator for stabilizer circuits using the Aaronson-Gottesman
tableau formalism (2004). Instead of tracking the full 2^n statevector, we track
the stabilizer and destabilizer generators in an integer tableau of shape (2n, 2n+1).

This achieves the Gottesman-Knill theorem: Clifford-only circuits are simulated in
polynomial time O(n^2), vectorized with SIMD NumPy column slicing for millions of gates/sec.

Supported gates: H, S, SDG, SX, SXdg, X, Y, Z, CX (CNOT), CZ, SWAP, ECR, iSWAP, I
Implements the SimulatorBackend interface.

Example:
    >>> from quanta.simulator.pauli_frame import PauliFrameSimulator
    >>> sim = PauliFrameSimulator(num_qubits=3)
    >>> sim.apply("H", (0,))
    >>> sim.apply("CX", (0, 1))
    >>> sim.apply("CX", (0, 2))
    >>> counts = sim.sample(shots=1000, seed=42)
    >>> print(counts)  # {'000': ~500, '111': ~500}
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quanta.core.types import QuantaError
from quanta.simulator.base import SimulatorBackend

__all__ = ["PauliFrameSimulator"]


class PauliFrameSimulator(SimulatorBackend):
    """Vectorized Pauli frame simulator for Clifford circuits.

    Uses the stabilizer tableau (Aaronson-Gottesman, 2004):
    - 2n generators stored as a (2n) x (2n+1) binary matrix
    - Rows 0..n-1: destabilizers
    - Rows n..2n-1: stabilizers
    - Each row: [x0..xn-1 | z0..zn-1 | phase]

    Memory: O(n^2) bits (vs O(2^n) for statevector)
    Speed: Vectorized O(n) per gate (vs O(2^n) for statevector)

    Args:
        num_qubits: Number of qubits.
        seed: Random seed for sampling reproducibility.
    """

    def __init__(self, num_qubits: int, seed: int | None = None) -> None:
        self.num_qubits = num_qubits
        self.n = num_qubits
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Tableau: (2n) rows x (2n+1) columns
        # Columns: [x0..xn-1, z0..zn-1, phase]
        self._tab = np.zeros((2 * num_qubits, 2 * num_qubits + 1), dtype=np.int8)

        # Initialize: destabilizer i = X_i, stabilizer i = Z_i
        for i in range(num_qubits):
            self._tab[i, i] = 1                            # destab[i] = X_i
            self._tab[i + num_qubits, num_qubits + i] = 1  # stab[i] = Z_i

        self._measured_qubits: tuple[int, ...] | None = None

    def _x_col(self, row: int, qubit: int) -> int:
        return int(self._tab[row, qubit])

    def _z_col(self, row: int, qubit: int) -> int:
        return int(self._tab[row, self.n + qubit])

    def _phase(self, row: int) -> int:
        return int(self._tab[row, 2 * self.n])

    # ── SimulatorBackend Interface ──

    def apply(
        self,
        gate_name: str,
        qubits: tuple[int, ...],
        params: tuple[float, ...] = (),
    ) -> None:
        """Applies a Clifford gate to the stabilizer tableau."""
        name = gate_name.upper()
        if name in ("H", "HADAMARD"):
            self.h(qubits[0])
        elif name == "S":
            self.s(qubits[0])
        elif name in ("SDG", "S_DAGGER"):
            self.sdg(qubits[0])
        elif name == "SX":
            self.sx(qubits[0])
        elif name in ("SXDG", "SX_DAGGER"):
            self.sxdg(qubits[0])
        elif name == "X":
            self.x(qubits[0])
        elif name == "Y":
            self.y(qubits[0])
        elif name == "Z":
            self.z(qubits[0])
        elif name in ("CX", "CNOT"):
            self.cx(qubits[0], qubits[1])
        elif name == "CZ":
            self.cz(qubits[0], qubits[1])
        elif name == "SWAP":
            self.swap(qubits[0], qubits[1])
        elif name == "ECR":
            self.ecr(qubits[0], qubits[1])
        elif name == "ISWAP":
            self.iswap(qubits[0], qubits[1])
        elif name in ("I", "ID", "IDENTITY"):
            pass
        else:
            raise ValueError(
                f"Gate '{gate_name}' is not a Clifford gate. "
                "PauliFrameSimulator only supports Clifford gates: "
                "H, S, SDG, SX, SXdg, X, Y, Z, CX, CZ, SWAP, ECR, iSWAP, I."
            )

    # ── Vectorized Gate Operations ──

    def h(self, qubit: int) -> None:
        """Hadamard gate: X <-> Z, phase update for Y (vectorized)."""
        n = self.n
        xi = self._tab[:, qubit].copy()
        zi = self._tab[:, n + qubit].copy()
        self._tab[:, 2 * n] ^= (xi & zi)
        self._tab[:, qubit] = zi
        self._tab[:, n + qubit] = xi

    def s(self, qubit: int) -> None:
        """S gate: X -> +Y, Y -> -X, Z -> +Z (vectorized)."""
        n = self.n
        xi = self._tab[:, qubit]
        zi = self._tab[:, n + qubit]
        self._tab[:, 2 * n] ^= (xi & zi)
        self._tab[:, n + qubit] ^= xi

    def sdg(self, qubit: int) -> None:
        """S† gate: X -> -Y, Y -> +X, Z -> +Z (vectorized)."""
        n = self.n
        xi = self._tab[:, qubit]
        zi = self._tab[:, n + qubit]
        self._tab[:, 2 * n] ^= (xi & (1 - zi))
        self._tab[:, n + qubit] ^= xi

    def sx(self, qubit: int) -> None:
        """SX (√X) gate: X -> +X, Y -> +Z, Z -> -Y (vectorized)."""
        n = self.n
        xi = self._tab[:, qubit]
        zi = self._tab[:, n + qubit]
        self._tab[:, 2 * n] ^= ((1 - xi) & zi)
        self._tab[:, qubit] ^= zi

    def sxdg(self, qubit: int) -> None:
        """SX† (√X†) gate: X -> +X, Y -> -Z, Z -> +Y (vectorized)."""
        n = self.n
        xi = self._tab[:, qubit]
        zi = self._tab[:, n + qubit]
        self._tab[:, 2 * n] ^= (xi & zi)
        self._tab[:, qubit] ^= zi

    def x(self, qubit: int) -> None:
        """Pauli X: Z -> -Z, Y -> -Y (vectorized)."""
        self._tab[:, 2 * self.n] ^= self._tab[:, self.n + qubit]

    def y(self, qubit: int) -> None:
        """Pauli Y: X -> -X, Z -> -Z (vectorized)."""
        n = self.n
        self._tab[:, 2 * n] ^= (self._tab[:, qubit] ^ self._tab[:, n + qubit])

    def z(self, qubit: int) -> None:
        """Pauli Z: X -> -X, Y -> -Y (vectorized)."""
        self._tab[:, 2 * self.n] ^= self._tab[:, qubit]

    def cx(self, control: int, target: int) -> None:
        """CNOT gate (vectorized)."""
        n = self.n
        xc = self._tab[:, control]
        zc = self._tab[:, n + control]
        xt = self._tab[:, target]
        zt = self._tab[:, n + target]
        # Phase update: r += x_c * z_t * (x_t XOR z_c XOR 1)
        self._tab[:, 2 * n] ^= (xc & zt & (xt ^ zc ^ 1))
        # X propagation: x_t ^= x_c
        self._tab[:, target] ^= xc
        # Z propagation: z_c ^= z_t
        self._tab[:, n + control] ^= zt

    def cz(self, q1: int, q2: int) -> None:
        """CZ gate: H(q2) . CX(q1, q2) . H(q2) (vectorized)."""
        self.h(q2)
        self.cx(q1, q2)
        self.h(q2)

    def swap(self, q1: int, q2: int) -> None:
        """SWAP gate (vectorized column swap)."""
        n = self.n
        self._tab[:, [q1, q2]] = self._tab[:, [q2, q1]]
        self._tab[:, [n + q1, n + q2]] = self._tab[:, [n + q2, n + q1]]

    def ecr(self, q1: int, q2: int) -> None:
        """IBM Heron native ECR gate: (X ⊗ I)(S ⊗ I) CX(q1, q2) (I ⊗ SX) up to global phase."""
        self.sx(q2)
        self.cx(q1, q2)
        self.s(q1)
        self.x(q1)

    def iswap(self, q1: int, q2: int) -> None:
        """iSWAP gate: (S ⊗ S) . CZ(q1, q2) . SWAP(q1, q2)."""
        self.swap(q1, q2)
        self.cz(q1, q2)
        self.s(q1)
        self.s(q2)

    def measure(self, *qubits: int) -> None:
        """Records qubits to measure."""
        self._measured_qubits = qubits

    def inject_error(self, qubit: int, error: str) -> None:
        """Injects a Pauli error on a qubit."""
        if error == "X":
            self.x(qubit)
        elif error == "Y":
            self.y(qubit)
        elif error == "Z":
            self.z(qubit)
        else:
            raise ValueError(f"Unknown error type: {error}. Use 'X', 'Y', or 'Z'.")

    def apply_noise(
        self,
        noise_model: Any,
        qubits: tuple[int, ...],
        rng: np.random.Generator | None = None,
    ) -> None:
        """Applies stochastic Pauli noise channels directly to the stabilizer tableau."""
        from quanta.simulator.noise import BitFlip, Depolarizing, PhaseFlip, ReadoutError

        active_rng = rng if rng is not None else self._rng
        for channel in getattr(noise_model, "channels", ()):
            if isinstance(channel, ReadoutError):
                continue
            if isinstance(channel, Depolarizing):
                for q in qubits:
                    if active_rng.random() < channel.probability:
                        pauli = ("X", "Y", "Z")[int(active_rng.integers(0, 3))]
                        self.inject_error(q, pauli)
            elif isinstance(channel, BitFlip):
                for q in qubits:
                    if active_rng.random() < channel.probability:
                        self.x(q)
            elif isinstance(channel, PhaseFlip):
                for q in qubits:
                    if active_rng.random() < channel.probability:
                        self.z(q)
            else:
                raise QuantaError(
                    f"PauliFrameSimulator only supports Pauli noise channels; "
                    f"got {type(channel).__name__}."
                )

    # ── Measurement & Row Operations ──

    def _measure_qubit(self, qubit: int, rng: np.random.Generator | None) -> int:
        """Measures a single qubit, collapsing the stabilizer state.

        When rng is None, deterministic choice 0 is selected on random branches
        (used internally to find an exact anchor bitstring in probabilities()).
        """
        n = self.n

        # Check if any stabilizer anticommutes with Z_qubit
        stab_x = self._tab[n : 2 * n, qubit]
        nonzeros = np.nonzero(stab_x)[0]

        if len(nonzeros) > 0:
            # Random outcome
            p = n + int(nonzeros[0])
            # Row-reduce: for all other commuting rows that anticommute with Z_qubit, multiply by row p
            rows_to_mult = [
                i for i in range(2 * n)
                if i != p and i != (p - n) and self._tab[i, qubit]
            ]
            for i in rows_to_mult:
                self._rowmult(i, p)

            # Move stabilizer p to destabilizer position p - n
            self._tab[p - n] = self._tab[p].copy()
            # Set stabilizer to ±Z_qubit with random (or anchor 0) phase
            self._tab[p] = 0
            self._tab[p, n + qubit] = 1
            outcome = 0 if rng is None else int(rng.integers(0, 2))
            self._tab[p, 2 * n] = outcome
            return outcome
        else:
            # Deterministic outcome
            scratch = np.zeros(2 * n + 1, dtype=np.int8)
            scratch[n + qubit] = 1  # Start with +Z_qubit
            destab_x = self._tab[:n, qubit]
            for i in np.nonzero(destab_x)[0]:
                self._rowmult_scratch(scratch, int(i) + n)
            return int(scratch[2 * n])

    def _rowmult(self, target: int, source: int) -> None:
        """Multiplies row target by row source in the tableau (Aaronson-Gottesman PRA 70, 052328)."""
        n = self.n
        x1 = self._tab[target, :n]
        z1 = self._tab[target, n : 2 * n]
        x2 = self._tab[source, :n]
        z2 = self._tab[source, n : 2 * n]

        # Exact Aaronson-Gottesman phase exponent g(x1, z1, x2, z2) for (1,1) = +Y = +iXZ:
        # X * P (x1=1, z1=0): z2 * (2 * x2 - 1)   [XY = +iZ -> +1, XZ = -iY -> -1]
        # Z * P (x1=0, z1=1): x2 * (1 - 2 * z2)   [ZX = +iY -> +1, ZY = -iX -> -1]
        # Y * P (x1=1, z1=1): z2 - x2             [YZ = +iX -> +1, YX = -iZ -> -1]
        c_x = (x1 == 1) & (z1 == 0)
        c_z = (x1 == 0) & (z1 == 1)
        c_y = (x1 == 1) & (z1 == 1)
        contribs = c_x * (z2 * (2 * x2 - 1)) + c_z * (x2 * (1 - 2 * z2)) + c_y * (z2 - x2)
        phase_contrib = int(np.sum(contribs, dtype=np.int64))

        total_i_exp = (2 * int(self._tab[target, 2 * n]) + 2 * int(self._tab[source, 2 * n]) + phase_contrib) % 4
        self._tab[target, 2 * n] = 1 if total_i_exp == 2 else 0

        self._tab[target, : 2 * n] ^= self._tab[source, : 2 * n]

    def _rowmult_scratch(self, scratch: np.ndarray, source: int) -> None:
        """Multiplies scratch row by tableau row (Aaronson-Gottesman PRA 70, 052328)."""
        n = self.n
        x1 = scratch[:n]
        z1 = scratch[n : 2 * n]
        x2 = self._tab[source, :n]
        z2 = self._tab[source, n : 2 * n]

        c_x = (x1 == 1) & (z1 == 0)
        c_z = (x1 == 0) & (z1 == 1)
        c_y = (x1 == 1) & (z1 == 1)
        contribs = c_x * (z2 * (2 * x2 - 1)) + c_z * (x2 * (1 - 2 * z2)) + c_y * (z2 - x2)
        phase_contrib = int(np.sum(contribs, dtype=np.int64))

        total_i_exp = (2 * int(scratch[2 * n]) + 2 * int(self._tab[source, 2 * n]) + phase_contrib) % 4
        scratch[2 * n] = 1 if total_i_exp == 2 else 0

        scratch[: 2 * n] ^= self._tab[source, : 2 * n]

    def sample(self, shots: int = 1024, seed: int | None = None) -> dict[str, int]:
        """Samples measurement outcomes.

        Args:
            shots: Number of measurement samples.
            seed: Random seed.

        Returns:
            Dict of {bitstring: count}.
        """
        rng = np.random.default_rng(seed) if seed is not None else self._rng

        measured = list(range(self.n))
        if self._measured_qubits is not None:
            measured = list(self._measured_qubits)

        # Save tableau state so we can restore after each shot
        tab_backup = self._tab.copy()

        counts: dict[str, int] = {}
        for _ in range(shots):
            self._tab = tab_backup.copy()
            bits = []
            for q in measured:
                bits.append(str(self._measure_qubit(q, rng)))
            bitstring = "".join(bits)
            counts[bitstring] = counts.get(bitstring, 0) + 1

        # Restore original state
        self._tab = tab_backup
        return counts

    def probabilities(self) -> np.ndarray:
        """Returns exact analytical Born probabilities for all 2^n computational basis states.

        Every stabilizer state has uniform probability 2^{-k} on an affine GF(2)
        subspace x_0 + rowspace(X_stab) of dimension k = rank_GF2(X_stab), and 0
        everywhere else.

        Raises:
            QuantaError: If num_qubits > 26 (dense probability array would exceed memory).
        """
        n = self.n
        if n > 26:
            raise QuantaError(
                f"PauliFrameSimulator.probabilities() requires allocating 2^{n} floats; "
                "maximum supported is 26 qubits. Use sample() for large Clifford circuits."
            )
        if n == 0:
            return np.array([1.0], dtype=np.float64)

        # 1. Extract GF(2) echelon basis of X-block of stabilizer generators (rows n..2n-1)
        pivots: dict[int, int] = {}
        for r in range(n):
            mask = 0
            row_x = self._tab[n + r, :n]
            for q in range(n):
                if row_x[q]:
                    mask |= 1 << (n - 1 - q)
            while mask:
                bit = mask.bit_length()
                if bit in pivots:
                    mask ^= pivots[bit]
                else:
                    pivots[bit] = mask
                    break
        basis_masks = list(pivots.values())

        # 2. Obtain one exact valid basis bitstring x_0 without mutating _tab or _rng
        tab_backup = self._tab.copy()
        try:
            x0_mask = 0
            for q in range(n):
                bit_val = self._measure_qubit(q, rng=None)
                if bit_val:
                    x0_mask |= 1 << (n - 1 - q)
        finally:
            self._tab = tab_backup

        # 3. Expand affine subspace x_0 XOR span(basis_masks) in O(2^k)
        indices = np.array([x0_mask], dtype=np.int64)
        for b in basis_masks:
            indices = np.concatenate((indices, indices ^ b))

        probs = np.zeros(1 << n, dtype=np.float64)
        probs[indices] = 2.0 ** (-len(basis_masks))
        return probs

    @property
    def state(self) -> np.ndarray:
        """Returns the current stabilizer tableau."""
        return self._tab.copy()

    @state.setter
    def state(self, value: Any) -> None:
        """Sets the stabilizer tableau."""
        arr = np.asarray(value, dtype=np.int8)
        if arr.shape != (2 * self.n, 2 * self.n + 1):
            raise ValueError(
                f"Tableau shape mismatch: expected {(2 * self.n, 2 * self.n + 1)}, got {arr.shape}"
            )
        self._tab = arr.copy()

    @property
    def max_qubits(self) -> int:
        return 100000

    def reset(self) -> None:
        """Resets the simulator to |00...0> state while preserving seed."""
        self.__init__(self.n, seed=self._seed)  # type: ignore[misc]

    def __repr__(self) -> str:
        return f"PauliFrameSimulator(n={self.n})"
