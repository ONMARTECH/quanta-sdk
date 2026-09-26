"""quanta.primitives.cvp — Closest Vector Problem (CVP) Hamiltonian mapping.

Translates lattice Closest Vector Problem instances into exact Quadratic Unconstrained
Binary Optimization (QUBO) and Ising spin Hamiltonians for quantum variational solvers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class CVPHamiltonian:
    """Ising and QUBO Hamiltonian representation of a lattice CVP instance.

    Given a lattice basis B in R^(m x n) and target vector t in R^m, represents the
    Closest Vector Problem:
        min_{z in Z^n} ||B z - t||^2

    Coordinates z_i are parameterized using K bits:
        z_i = z_i^min + sum_{k=0}^{K-1} 2^k x_{i, k},  where x_{i, k} in {0, 1}.

    Attributes:
        basis: Lattice basis matrix B of shape (m, n).
        target: Target vector t of shape (m,).
        bits_per_coord: Number of binary qubits allocated per coordinate.
        center: Integer reference center vector in Z^n.
        z_min: Lower bound integer coordinate vector in Z^n.
        num_qubits: Total number of qubits N = n * bits_per_coord.
        dim: Lattice dimension n.
        qubo_matrix: Upper-triangular QUBO matrix Q of shape (N, N).
        qubo_offset: Scalar constant energy offset C_offset.
        gram_matrix: Symmetric positive-definite Gram matrix G = B^T B of shape (n, n).
    """

    basis: np.ndarray
    target: np.ndarray
    bits_per_coord: int = 3
    center: np.ndarray | None = None
    z_min: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.basis = np.asarray(self.basis, dtype=np.float64)
        self.target = np.asarray(self.target, dtype=np.float64).reshape(-1)

        m, n = self.basis.shape
        if len(self.target) != m:
            raise ValueError(
                f"Target dimension {len(self.target)} does not match basis rows {m}."
            )
        if self.bits_per_coord < 1:
            raise ValueError(
                f"bits_per_coord must be >= 1, got {self.bits_per_coord}."
            )

        self.dim: int = n
        self.num_qubits: int = n * self.bits_per_coord

        span = 1 << self.bits_per_coord
        half_span = 1 << (self.bits_per_coord - 1)

        if self.z_min is not None:
            self.z_min = np.asarray(self.z_min, dtype=np.int64).reshape(-1)
            if len(self.z_min) != n:
                raise ValueError(
                    f"z_min length {len(self.z_min)} must match dimension {n}."
                )
            if self.center is None:
                self.center = self.z_min + half_span
            else:
                self.center = np.asarray(self.center, dtype=np.int64).reshape(-1)
                if len(self.center) != n:
                    raise ValueError(
                        f"center length {len(self.center)} must match dimension {n}."
                    )
        elif self.center is not None:
            self.center = np.asarray(self.center, dtype=np.int64).reshape(-1)
            if len(self.center) != n:
                raise ValueError(
                    f"center length {len(self.center)} must match dimension {n}."
                )
            self.z_min = self.center - half_span
        else:
            self.center = np.zeros(n, dtype=np.int64)
            self.z_min = -half_span * np.ones(n, dtype=np.int64)

        self.z_max: np.ndarray = self.z_min + (span - 1)

        # Precompute algebraic matrices
        self.gram_matrix: np.ndarray = self.basis.T @ self.basis
        self._linear_target: np.ndarray = self.basis.T @ self.target
        self._c0: float = float(self.target @ self.target)

        # Decoding matrix W: z(x) = W @ x + z_min
        self._decoding_matrix: np.ndarray = self._build_decoding_matrix()

        # Build QUBO matrix and offset
        self.qubo_matrix, self.qubo_offset = self._build_qubo()

        # Build Ising couplings, fields, and offset
        self._ising_j, self._ising_h, self._ising_offset = self._build_ising()

        # Cache Pauli list representation
        self._pauli_list: list[tuple[str, float]] | None = None
        self._diagonal_energies: np.ndarray | None = None

    def _build_decoding_matrix(self) -> np.ndarray:
        """Constructs affine transformation matrix W mapping x to z - z_min."""
        w = np.zeros((self.dim, self.num_qubits), dtype=np.float64)
        k = self.bits_per_coord
        for i in range(self.dim):
            for bit in range(k):
                w[i, i * k + bit] = float(1 << bit)
        return w

    def _build_qubo(self) -> tuple[np.ndarray, float]:
        """Constructs upper-triangular QUBO matrix Q and scalar offset C_offset.

        Energy formula:
            ||B z(x) - t||^2 = x^T Q x + C_offset
        """
        assert self.z_min is not None
        w = self._decoding_matrix
        g = self.gram_matrix
        z0 = self.z_min.astype(np.float64)
        y = self._linear_target

        # A = W^T G W (symmetric)
        a = w.T @ g @ w

        # b = 2 W^T (G z0 - y)
        b = 2.0 * (w.T @ (g @ z0 - y))

        # c_base = ||B z0 - t||^2
        c_base = float(z0 @ (g @ z0) - 2.0 * (y @ z0) + self._c0)

        n_q = self.num_qubits
        q = np.zeros((n_q, n_q), dtype=np.float64)

        for p in range(n_q):
            # Linear terms absorbed into diagonal: x_p^2 = x_p
            q[p, p] = a[p, p] + b[p]
            for q_idx in range(p + 1, n_q):
                # Bilinear terms: symmetric entries combined into upper triangle
                q[p, q_idx] = 2.0 * a[p, q_idx]

        return q, c_base

    def _build_ising(
        self,
    ) -> tuple[dict[tuple[int, int], float], dict[int, float], float]:
        """Constructs Ising two-body couplings J_ij, linear fields h_i, and offset.

        Mapping: x_p = (1 - Z_p) / 2
        """
        q = self.qubo_matrix
        n_q = self.num_qubits

        j_dict: dict[tuple[int, int], float] = {}
        for p in range(n_q):
            for q_idx in range(p + 1, n_q):
                val = 0.25 * q[p, q_idx]
                if abs(val) > 1e-15:
                    j_dict[(p, q_idx)] = float(val)

        h_dict: dict[int, float] = {}
        for p in range(n_q):
            sum_cross = 0.0
            for other in range(n_q):
                if other < p:
                    sum_cross += q[other, p]
                elif other > p:
                    sum_cross += q[p, other]
            hp = -0.5 * q[p, p] - 0.25 * sum_cross
            h_dict[p] = float(hp)

        sum_diag = float(np.trace(q))
        sum_upper = float(np.sum(np.triu(q, k=1)))
        offset_ising = float(self.qubo_offset + 0.5 * sum_diag + 0.25 * sum_upper)

        return j_dict, h_dict, offset_ising

    def to_qubo(self) -> tuple[np.ndarray, float]:
        """Returns the upper-triangular QUBO matrix Q and constant scalar offset.

        Returns:
            Tuple of (Q, offset) such that x^T Q x + offset = ||B z(x) - t||^2.
        """
        return self.qubo_matrix.copy(), float(self.qubo_offset)

    def to_ising(self) -> tuple[dict[tuple[int, int], float], dict[int, float], float]:
        """Returns Ising spin representation (J, h, offset).

        Returns:
            Tuple of:
                - J: Dictionary mapping (i, j) with i < j to coupling J_ij.
                - h: Dictionary mapping qubit i to local longitudinal field h_i.
                - offset: Constant scalar energy offset.
        """
        return dict(self._ising_j), dict(self._ising_h), float(self._ising_offset)

    def to_pauli_list(self) -> list[tuple[str, float]]:
        """Generates list of (pauli_string, coeff) compatible with Quanta Estimator.

        Returns:
            List of Pauli strings and coefficients representing the cost Hamiltonian.
        """
        if self._pauli_list is not None:
            return list(self._pauli_list)

        n_q = self.num_qubits
        terms: list[tuple[str, float]] = []

        # 2-qubit terms J_ij Z_i Z_j
        for (i, j), j_val in self._ising_j.items():
            chars = ["I"] * n_q
            chars[i] = "Z"
            chars[j] = "Z"
            terms.append(("".join(chars), j_val))

        # 1-qubit terms h_i Z_i
        for i, h_val in self._ising_h.items():
            if abs(h_val) > 1e-15:
                chars = ["I"] * n_q
                chars[i] = "Z"
                terms.append(("".join(chars), h_val))

        # Constant offset term offset * I
        if abs(self._ising_offset) > 1e-15:
            terms.append(("I" * n_q, self._ising_offset))

        self._pauli_list = terms
        return list(terms)

    def bitstring_to_x(self, bitstring: str) -> np.ndarray:
        """Converts a binary string to a 1D numpy array."""
        if len(bitstring) != self.num_qubits:
            raise ValueError(
                f"Bitstring length {len(bitstring)} does not match {self.num_qubits} qubits."
            )
        return np.array([int(ch) for ch in bitstring], dtype=np.int64)

    def x_to_bitstring(self, x: np.ndarray | list[int]) -> str:
        """Converts a binary vector to a bitstring."""
        arr = np.asarray(x, dtype=np.int64).reshape(-1)
        if len(arr) != self.num_qubits:
            raise ValueError(
                f"Binary array length {len(arr)} does not match {self.num_qubits} qubits."
            )
        return "".join(str(int(bit)) for bit in arr)

    def x_to_z(self, x: np.ndarray | list[int] | str) -> np.ndarray:
        """Decodes binary array or bitstring into integer coordinate vector z."""
        if isinstance(x, str):
            x_arr = self.bitstring_to_x(x)
        else:
            x_arr = np.asarray(x, dtype=np.int64).reshape(-1)

        z = self._decoding_matrix @ x_arr + self.z_min
        return np.asarray(np.round(z), dtype=np.int64)

    def z_to_x(self, z: np.ndarray | list[int]) -> np.ndarray | None:
        """Encodes integer coordinate vector z into binary vector x, or None if out of bounds."""
        z_arr = np.asarray(z, dtype=np.int64).reshape(-1)
        if len(z_arr) != self.dim:
            raise ValueError(f"Vector z length {len(z_arr)} does not match dimension {self.dim}.")

        assert self.z_min is not None
        delta = z_arr - self.z_min
        k = self.bits_per_coord
        max_val = (1 << k) - 1

        if np.any(delta < 0) or np.any(delta > max_val):
            return None

        x = np.zeros(self.num_qubits, dtype=np.int64)
        for i in range(self.dim):
            val = int(delta[i])
            for bit in range(k):
                x[i * k + bit] = (val >> bit) & 1

        return x

    def decode_sample(
        self, sample: str | list[int] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Decodes a measurement sample into lattice coordinates, lattice point, and distance.

        Args:
            sample: Bitstring (e.g. '0110') or binary sequence.

        Returns:
            Tuple of:
                - z: Integer coordinate vector in Z^n.
                - v: Real lattice vector v = B @ z in R^m.
                - distance: Direct Euclidean distance ||v - t||_2.
        """
        z = self.x_to_z(sample)
        v = self.basis @ z
        dist = float(np.linalg.norm(v - self.target))
        return z, v, dist

    def evaluate_distance(self, z: np.ndarray | list[int]) -> float:
        """Computes direct Euclidean distance ||B z - t||_2 for integer vector z."""
        z_arr = np.asarray(z, dtype=np.float64).reshape(-1)
        v = self.basis @ z_arr
        return float(np.linalg.norm(v - self.target))

    def evaluate_squared_distance(self, z: np.ndarray | list[int]) -> float:
        """Computes direct squared Euclidean distance ||B z - t||_2^2 for integer vector z."""
        z_arr = np.asarray(z, dtype=np.float64).reshape(-1)
        diff = self.basis @ z_arr - self.target
        return float(diff @ diff)

    def qubo_energy(self, sample: str | list[int] | np.ndarray) -> float:
        """Computes QUBO objective energy x^T Q x + C_offset."""
        if isinstance(sample, str):
            x = self.bitstring_to_x(sample).astype(np.float64)
        else:
            x = np.asarray(sample, dtype=np.float64).reshape(-1)
        return float(x @ (self.qubo_matrix @ x) + self.qubo_offset)

    def ising_energy(self, sample: str | list[int] | np.ndarray) -> float:
        """Computes Ising energy sum J_ij Z_i Z_j + sum h_i Z_i + offset."""
        if isinstance(sample, str):
            x = self.bitstring_to_x(sample)
        else:
            x = np.asarray(sample, dtype=np.int64).reshape(-1)

        # Spins Z_p = 1 - 2 * x_p
        spins = 1.0 - 2.0 * x.astype(np.float64)

        energy = float(self._ising_offset)
        for (i, j), j_val in self._ising_j.items():
            energy += j_val * spins[i] * spins[j]
        for i, h_val in self._ising_h.items():
            energy += h_val * spins[i]

        return float(energy)

    def diagonal_energy_vector(self) -> np.ndarray:
        """Returns vectorized 1D array of energies for all 2^N basis states."""
        if self._diagonal_energies is not None:
            return self._diagonal_energies

        n_q = self.num_qubits
        dim_space = 1 << n_q

        # Construct all binary bitstrings efficiently
        # basis_indices shape (dim_space, 1)
        basis_indices = np.arange(dim_space, dtype=np.int64)[:, None]
        # bit_shifts shape (1, n_q): bit p is (basis_indices >> (n_q - 1 - p)) & 1
        bit_shifts = np.arange(n_q - 1, -1, -1, dtype=np.int64)[None, :]
        x_all = ((basis_indices >> bit_shifts) & 1).astype(np.float64)  # (dim_space, n_q)

        # Compute x @ Q @ x^T along axis 1
        # Q_x shape (dim_space, n_q)
        q_x = x_all @ self.qubo_matrix.T
        energies = np.sum(x_all * q_x, axis=1) + self.qubo_offset

        self._diagonal_energies = energies.astype(np.float64)
        return self._diagonal_energies

    def to_cost_function(self, squared: bool = True) -> Callable[[int], float]:
        """Returns an integer-indexed cost function for diagonal statevector simulation.

        Args:
            squared: If True, returns squared distance ||Bz - t||^2 (exact QUBO).
                If False, returns Euclidean distance ||Bz - t||.

        Returns:
            Callable mapping basis state index in [0, 2^N - 1] to cost.
        """
        diag = self.diagonal_energy_vector()
        if squared:
            return lambda idx: float(diag[idx])
        return lambda idx: float(np.sqrt(max(0.0, diag[idx])))
