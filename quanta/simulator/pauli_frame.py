"""
quanta.simulator.pauli_frame -- Pauli frame simulator for Clifford circuits.

A fast simulator for stabilizer circuits using the Pauli frame formalism.
Instead of tracking the full 2^n statevector, we track only the Pauli
frame: n qubits need 2n bits (X and Z components).

This is equivalent to the Stabilizer Simulator (Gottesman-Knill theorem):
Clifford-only circuits can be simulated in O(n^2) time.

Supported gates: H, S, X, Y, Z, CX (CNOT), CZ, SWAP
These are exactly the gates needed for QEC circuit simulation.

Inspired by: Google's Stim (C++ pauli frame tracker).

Example:
    >>> from quanta.simulator.pauli_frame import PauliFrameSimulator
    >>> sim = PauliFrameSimulator(num_qubits=3)
    >>> sim.h(0)
    >>> sim.cx(0, 1)
    >>> sim.cx(0, 2)
    >>> counts = sim.sample(shots=1000, seed=42)
    >>> print(counts)  # {'000': ~500, '111': ~500}
"""

from __future__ import annotations

import numpy as np

__all__ = ["PauliFrameSimulator"]


class PauliFrameSimulator:
    """Pauli frame simulator for Clifford circuits.

    Uses the stabilizer tableau (Aaronson-Gottesman, 2004):
    - 2n generators stored as a (2n) x (2n+1) binary matrix
    - Rows 0..n-1: destabilizers
    - Rows n..2n-1: stabilizers
    - Each row: [x0..xn-1 | z0..zn-1 | phase]

    Memory: O(n^2) bits (vs O(2^n) for statevector)
    Speed: O(n) per gate (vs O(2^n) for statevector)

    Args:
        num_qubits: Number of qubits.
    """

    def __init__(self, num_qubits: int) -> None:
        self.n = num_qubits
        # Tableau: (2n) rows x (2n+1) columns
        # Columns: [x0..xn-1, z0..zn-1, phase]
        self._tab = np.zeros((2 * num_qubits, 2 * num_qubits + 1), dtype=np.int8)

        # Initialize: destabilizer i = X_i, stabilizer i = Z_i
        for i in range(num_qubits):
            self._tab[i, i] = 1                      # destab[i] = X_i
            self._tab[i + num_qubits, num_qubits + i] = 1  # stab[i] = Z_i

        self._measured_qubits: tuple[int, ...] | None = None

    def _x_col(self, row: int, qubit: int) -> int:
        return self._tab[row, qubit]

    def _z_col(self, row: int, qubit: int) -> int:
        return self._tab[row, self.n + qubit]

    def _phase(self, row: int) -> int:
        return self._tab[row, 2 * self.n]

    def h(self, qubit: int) -> None:
        """Hadamard gate: X<->Z, phase update for Y."""
        for i in range(2 * self.n):
            xi = self._tab[i, qubit]
            zi = self._tab[i, self.n + qubit]
            # Phase: XZ -> -ZX (Y -> -Y)
            self._tab[i, 2 * self.n] ^= xi & zi
            # Swap X and Z
            self._tab[i, qubit] = zi
            self._tab[i, self.n + qubit] = xi

    def s(self, qubit: int) -> None:
        """S gate: X -> Y (XZ), Z -> Z."""
        for i in range(2 * self.n):
            xi = self._tab[i, qubit]
            zi = self._tab[i, self.n + qubit]
            self._tab[i, 2 * self.n] ^= xi & zi
            self._tab[i, self.n + qubit] = xi ^ zi

    def x(self, qubit: int) -> None:
        """Pauli X: Z -> -Z, Y -> -Y."""
        for i in range(2 * self.n):
            self._tab[i, 2 * self.n] ^= self._tab[i, self.n + qubit]

    def y(self, qubit: int) -> None:
        """Pauli Y: X -> -X, Z -> -Z."""
        for i in range(2 * self.n):
            xi = self._tab[i, qubit]
            zi = self._tab[i, self.n + qubit]
            self._tab[i, 2 * self.n] ^= xi ^ zi

    def z(self, qubit: int) -> None:
        """Pauli Z: X -> -X, Y -> -Y."""
        for i in range(2 * self.n):
            self._tab[i, 2 * self.n] ^= self._tab[i, qubit]

    def cx(self, control: int, target: int) -> None:
        """CNOT gate."""
        for i in range(2 * self.n):
            xc = self._tab[i, control]
            zc = self._tab[i, self.n + control]
            xt = self._tab[i, target]
            zt = self._tab[i, self.n + target]
            # Phase: r += x_c * z_t * (x_t XOR z_c XOR 1)
            self._tab[i, 2 * self.n] ^= xc & zt & (xt ^ zc ^ 1)
            # X propagation: x_t ^= x_c
            self._tab[i, target] ^= xc
            # Z propagation: z_c ^= z_t
            self._tab[i, self.n + control] ^= zt

    def cz(self, q1: int, q2: int) -> None:
        """CZ gate: H(q2) . CX(q1,q2) . H(q2)."""
        self.h(q2)
        self.cx(q1, q2)
        self.h(q2)

    def swap(self, q1: int, q2: int) -> None:
        """SWAP gate: 3 CNOTs."""
        self.cx(q1, q2)
        self.cx(q2, q1)
        self.cx(q1, q2)

    def measure(self, *qubits: int) -> None:
        """Records qubits to measure."""
        self._measured_qubits = qubits

    def inject_error(self, qubit: int, error: str) -> None:
        """Injects a Pauli error on a qubit.

        Args:
            qubit: Target qubit index.
            error: Error type: "X", "Y", or "Z".
        """
        if error == "X":
            self.x(qubit)
        elif error == "Y":
            self.y(qubit)
        elif error == "Z":
            self.z(qubit)
        else:
            raise ValueError(f"Unknown error type: {error}. Use 'X', 'Y', or 'Z'.")

    def _measure_qubit(self, qubit: int, rng: np.random.Generator) -> int:
        """Measures a single qubit, collapsing the stabilizer state.

        Returns 0 or 1.
        """
        n = self.n

        # Check if any stabilizer anticommutes with Z_qubit
        # (has X component on the measured qubit)
        anticommuting = -1
        for i in range(n, 2 * n):
            if self._tab[i, qubit]:  # X component set
                anticommuting = i
                break

        if anticommuting >= 0:
            # Random outcome
            p = anticommuting
            # Row-reduce: for all other rows that anticommute, multiply by row p
            for i in range(2 * n):
                if i != p and self._tab[i, qubit]:
                    self._rowmult(i, p)

            # Move destabilizer to stabilizer position
            self._tab[p - n] = self._tab[p].copy()
            # Set stabilizer to Z_qubit with random phase
            self._tab[p] = 0
            self._tab[p, n + qubit] = 1
            outcome = int(rng.integers(0, 2))
            self._tab[p, 2 * n] = outcome
            return outcome
        else:
            # Deterministic outcome
            # Need to compute the outcome from the destabilizers
            # The outcome is determined by the product of stabilizers
            scratch = np.zeros(2 * n + 1, dtype=np.int8)
            scratch[n + qubit] = 1  # Start with Z_qubit

            for i in range(n):
                if self._tab[i, qubit]:  # destabilizer anticommutes
                    # Multiply scratch by stabilizer[i + n]
                    self._rowmult_scratch(scratch, i + n)

            return int(scratch[2 * n])

    def _rowmult(self, target: int, source: int) -> None:
        """Multiplies row target by row source in the tableau."""
        n = self.n
        # Phase: compute phase of product
        phase_contrib = 0
        for j in range(n):
            x1, z1 = self._tab[target, j], self._tab[target, n + j]
            x2, z2 = self._tab[source, j], self._tab[source, n + j]
            # Pauli product phase contribution
            if x2 or z2:
                if x1 == 0 and z1 == 0:
                    pass  # I * P = P
                elif x1 == 1 and z1 == 0:
                    phase_contrib += z2 * (1 - 2 * x2)  # X * P
                elif x1 == 0 and z1 == 1:
                    phase_contrib += x2 * (2 * z2 - 1)  # Z * P
                else:
                    phase_contrib += z2 - x2  # Y * P

        self._tab[target, 2 * n] ^= self._tab[source, 2 * n]
        if phase_contrib % 4 == 2 or phase_contrib % 4 == -2:
            self._tab[target, 2 * n] ^= 1

        for j in range(2 * n):
            self._tab[target, j] ^= self._tab[source, j]

    def _rowmult_scratch(self, scratch: np.ndarray, source: int) -> None:
        """Multiplies scratch row by tableau row."""
        n = self.n
        phase_contrib = 0
        for j in range(n):
            x1, z1 = scratch[j], scratch[n + j]
            x2, z2 = self._tab[source, j], self._tab[source, n + j]
            if x2 or z2:
                if x1 == 0 and z1 == 0:
                    pass
                elif x1 == 1 and z1 == 0:
                    phase_contrib += z2 * (1 - 2 * x2)
                elif x1 == 0 and z1 == 1:
                    phase_contrib += x2 * (2 * z2 - 1)
                else:
                    phase_contrib += z2 - x2

        scratch[2 * n] ^= self._tab[source, 2 * n]
        if phase_contrib % 4 == 2 or phase_contrib % 4 == -2:
            scratch[2 * n] ^= 1

        for j in range(2 * n):
            scratch[j] ^= self._tab[source, j]

    def sample(self, shots: int = 1024, seed: int | None = None) -> dict[str, int]:
        """Samples measurement outcomes.

        Args:
            shots: Number of measurement samples.
            seed: Random seed.

        Returns:
            Dict of {bitstring: count}.
        """
        rng = np.random.default_rng(seed)

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

    def __repr__(self) -> str:
        return f"PauliFrameSimulator(n={self.n})"
