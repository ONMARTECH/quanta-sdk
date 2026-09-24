"""
quanta.layer3.hamiltonian -- Hamiltonian simulation.

Simulates time evolution of quantum systems under a Hamiltonian:
  |psi(t)> = exp(-i * H * t) |psi(0)>

Key for: molecular simulation, material science, physics research.

Supports Pauli Hamiltonians (like VQE) and Trotterized evolution.

Example:
    >>> from quanta.layer3.hamiltonian import evolve, molecular_hamiltonian
    >>> # Simulate H2 molecule
    >>> H = molecular_hamiltonian("H2")
    >>> result = evolve(H, num_qubits=2, time=1.0, steps=10)
    >>> print(f"Energy: {result.energy:.4f}")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "evolve", "molecular_hamiltonian", "EvolutionResult",
    "HamiltonianSpec", "spectral_unitary_evolution",
    "suzuki_trotter_step", "trotter_evolve",
    "magnus_step", "evolve_time_dependent",
]


# -- Pauli matrices --
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


@dataclass
class HamiltonianSpec:
    """Hamiltonian specification from Pauli terms."""
    name: str
    terms: list[tuple[str, float]]
    num_qubits: int
    description: str = ""


@dataclass
class EvolutionResult:
    """Result of Hamiltonian time evolution.

    Attributes:
        final_state: Final statevector.
        energy: Expectation value <psi|H|psi>.
        energy_history: Energy at each time step.
        time: Total evolution time.
    """
    final_state: np.ndarray
    energy: float
    energy_history: list[float]
    time: float

    def summary(self) -> str:
        lines = [
            "=== Hamiltonian Evolution ===",
            f"  Time: {self.time:.4f}",
            f"  Final energy: {self.energy:.6f}",
            f"  Steps: {len(self.energy_history)}",
        ]
        return "\n".join(lines)


# -- Pre-defined molecular Hamiltonians (simplified) --

_MOLECULES: dict[str, HamiltonianSpec] = {
    "H2": HamiltonianSpec(
        name="H2 (Hydrogen molecule)",
        terms=[
            ("II", -0.8105),
            ("IZ", 0.1721),
            ("ZI", -0.2257),
            ("ZZ", 0.1716),
            ("XX", 0.0454),
        ],
        num_qubits=2,
        description="Minimal STO-3G basis H2 at 0.735A bond length",
    ),
    "LiH": HamiltonianSpec(
        name="LiH (Lithium hydride)",
        terms=[
            ("IIII", -7.4983),
            ("IIIZ", 0.3435),
            ("IIZI", -0.4347),
            ("IZII", 0.5716),
            ("ZIII", 0.0910),
            ("IIZZ", 0.1209),
            ("IZIZ", 0.0594),
            ("ZIIZ", 0.0615),
            ("IZZI", 0.0346),
            ("ZIZI", -0.0545),
            ("ZZII", 0.0737),
            ("IIXX", 0.0178),
            ("IXIX", 0.0178),
            ("XIXI", -0.0089),
            ("XXII", 0.0649),
        ],
        num_qubits=4,
        description="Simplified STO-3G basis LiH",
    ),
    "HeH+": HamiltonianSpec(
        name="HeH+ (Helium hydride ion)",
        terms=[
            ("II", -1.4627),
            ("IZ", 0.3435),
            ("ZI", -0.3895),
            ("ZZ", 0.1810),
            ("XX", 0.0512),
            ("YY", 0.0512),
        ],
        num_qubits=2,
        description="Minimal basis HeH+ cation",
    ),
}


def molecular_hamiltonian(molecule: str) -> HamiltonianSpec:
    """Returns a pre-defined molecular Hamiltonian.

    Available molecules: H2, LiH, HeH+

    Args:
        molecule: Molecule name (case-insensitive).

    Returns:
        HamiltonianSpec with Pauli terms and metadata.
    """
    key = molecule.upper().replace(" ", "")
    # Try exact match first, then without special chars
    if key not in _MOLECULES:
        # Try common aliases
        aliases = {"HEH+": "HeH+", "HEH": "HeH+", "H2": "H2", "LIH": "LiH"}
        key = aliases.get(key, key)
    if key not in _MOLECULES:
        available = ", ".join(_MOLECULES.keys())
        raise ValueError(
            f"Unknown molecule: {molecule}. Available: {available}"
        )
    return _MOLECULES[key]


def _build_matrix(
    terms: list[tuple[str, float]], n: int
) -> np.ndarray:
    """Builds full matrix from Pauli terms."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for pauli_str, coeff in terms:
        padded = pauli_str.ljust(n, "I")
        term = np.array([[1.0]], dtype=complex)
        for ch in padded:
            term = np.kron(term, _PAULI[ch])
        H += coeff * term
    return H


def _build_term_matrices(
    terms: list[tuple[str, float]], n: int
) -> list[np.ndarray]:
    """Builds individual matrix terms from Pauli terms."""
    matrices: list[np.ndarray] = []
    for pauli_str, coeff in terms:
        padded = pauli_str.ljust(n, "I")
        term = np.array([[1.0]], dtype=complex)
        for ch in padded:
            term = np.kron(term, _PAULI[ch])
        matrices.append(coeff * term)
    return matrices


def spectral_unitary_evolution(H: np.ndarray, dt: float) -> np.ndarray:
    """Computes U(dt) = exp(-i * H * dt) via exact spectral decomposition.

    For Hermitian H, H = V diag(lambda) V^dagger where lambda in R.
    Then U = V diag(exp(-i * lambda * dt)) V^dagger.
    This guarantees exact machine-precision unitarity:
        ||U^dagger U - I||_inf < 10^-14.

    Args:
        H: Hermitian Hamiltonian matrix of shape (D, D).
        dt: Time evolution step.

    Returns:
        Unitary evolution matrix U of shape (D, D).
    """
    H_herm = (H + H.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(H_herm)
    phases = np.exp(-1j * eigenvalues * dt)
    return eigenvectors @ np.diag(phases) @ eigenvectors.conj().T


def _matrix_exp(A: np.ndarray) -> np.ndarray:
    """Matrix exponential using spectral decomposition or expm.

    For anti-Hermitian A = -1j * H (where A + A^H = 0), performs exact
    spectral decomposition guaranteeing machine-precision unitarity:
        ||U^dagger U - I||_inf < 10^-14.
    For Hermitian matrices A = A^H, computes V diag(exp(lambda)) V^H.
    For general non-normal matrices, falls back to scipy.linalg.expm.
    """
    scale = max(float(np.max(np.abs(A))), 1e-14)
    anti_herm_diff = float(np.max(np.abs(A + A.conj().T)))
    herm_diff = float(np.max(np.abs(A - A.conj().T)))

    if anti_herm_diff / scale < 1e-10:
        # A is skew-Hermitian: A = -1j * K where K = 1j * A is Hermitian
        K = 1j * A
        K_herm = (K + K.conj().T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(K_herm)
        phases = np.exp(-1j * eigenvalues)
        return eigenvectors @ np.diag(phases) @ eigenvectors.conj().T
    elif herm_diff / scale < 1e-10:
        # A is Hermitian
        A_herm = (A + A.conj().T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(A_herm)
        return eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.conj().T
    else:
        from scipy.linalg import expm
        return expm(A)


def suzuki_trotter_step(
    term_matrices: list[np.ndarray],
    dt: float,
    order: int = 2,
) -> np.ndarray:
    """Computes a single Suzuki-Trotter step for non-commuting Hamiltonian terms.

    Supports:
        - order 1: Lie-Trotter product U = prod_k exp(-i H_k dt)
        - order 2: Strang splitting:
            U = prod_{k=1}^{m-1} exp(-i H_k dt/2) exp(-i H_m dt) prod_{k=m-1}^1 exp(-i H_k dt/2)
        - order 4: Suzuki fractal decomposition with p = 1 / (4 - 4^(1/3))

    Args:
        term_matrices: List of Hermitian matrices H_k where H = sum_k H_k.
        dt: Time step.
        order: Trotter order (1, 2, or 4).

    Returns:
        Unitary operator matrix for one step.
    """
    if not term_matrices:
        raise ValueError("term_matrices list cannot be empty.")

    m = len(term_matrices)
    dim = term_matrices[0].shape[0]

    if m == 1:
        return spectral_unitary_evolution(term_matrices[0], dt)

    if order == 1:
        U = np.eye(dim, dtype=complex)
        for H_k in term_matrices:
            U = spectral_unitary_evolution(H_k, dt) @ U
        return U

    elif order == 2:
        # Forward half steps for k = 0 to m-2
        U = np.eye(dim, dtype=complex)
        for k in range(m - 1):
            U = spectral_unitary_evolution(term_matrices[k], dt / 2.0) @ U
        # Center full step for k = m-1
        U = spectral_unitary_evolution(term_matrices[m - 1], dt) @ U
        # Backward half steps for k = m-2 down to 0
        for k in range(m - 2, -1, -1):
            U = spectral_unitary_evolution(term_matrices[k], dt / 2.0) @ U
        return U

    elif order == 4:
        # Suzuki fractal construction
        p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        S2_p = suzuki_trotter_step(term_matrices, p * dt, order=2)
        S2_mid = suzuki_trotter_step(term_matrices, (1.0 - 4.0 * p) * dt, order=2)
        return S2_p @ S2_p @ S2_mid @ S2_p @ S2_p

    else:
        raise ValueError(f"Unsupported Trotter order {order}. Must be 1, 2, or 4.")


def trotter_evolve(
    hamiltonian: HamiltonianSpec | list[tuple[str, float]],
    num_qubits: int | None = None,
    time: float = 1.0,
    steps: int = 20,
    order: int = 2,
    initial_state: np.ndarray | None = None,
) -> EvolutionResult:
    """Simulates time evolution using Suzuki-Trotter decomposition."""
    if isinstance(hamiltonian, HamiltonianSpec):
        terms = hamiltonian.terms
        n = hamiltonian.num_qubits
    else:
        terms = hamiltonian
        n = num_qubits or max(len(t[0]) for t in terms)

    term_mats = _build_term_matrices(terms, n)
    H_full = _build_matrix(terms, n)
    dim = 2 ** n
    dt = time / steps

    if initial_state is not None:
        state = initial_state.copy().astype(complex)
    else:
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

    U_step = suzuki_trotter_step(term_mats, dt, order=order)
    energy_history = []

    for _step in range(steps):
        energy = float(np.real(state.conj() @ H_full @ state))
        energy_history.append(energy)
        state = U_step @ state

    final_energy = float(np.real(state.conj() @ H_full @ state))
    energy_history.append(final_energy)

    return EvolutionResult(
        final_state=state,
        energy=final_energy,
        energy_history=energy_history,
        time=time,
    )


def magnus_step(
    H_func: Callable[[float], np.ndarray],
    t: float,
    dt: float,
    order: int = 4,
) -> np.ndarray:
    """Computes unitary time evolution step U(t + dt, t) via Magnus expansion.

    Guarantees strict unitarity for time-dependent Hamiltonians.
    - Order 2 (midpoint Magnus):
        Omega_1 = -i * dt * H(t + dt / 2)
        U = exp(Omega_1)
    - Order 4 (2-point Gauss-Legendre Magnus):
        c1 = 1/2 - sqrt(3)/6, c2 = 1/2 + sqrt(3)/6
        H1 = H(t + c1*dt), H2 = H(t + c2*dt)
        Omega = -i * (dt/2) * (H1 + H2) - (dt^2 * sqrt(3) / 12) * [H2, H1]
        U = exp(Omega)
    """
    if order == 2:
        H_mid = H_func(t + dt / 2.0)
        return spectral_unitary_evolution(H_mid, dt)
    elif order == 4:
        c1 = 0.5 - np.sqrt(3.0) / 6.0
        c2 = 0.5 + np.sqrt(3.0) / 6.0
        H1 = H_func(t + c1 * dt)
        H2 = H_func(t + c2 * dt)
        # Commutator [H2, H1] = H2 @ H1 - H1 @ H2
        comm = H2 @ H1 - H1 @ H2
        # Omega is purely anti-Hermitian
        omega = -1j * (dt / 2.0) * (H1 + H2) - (dt ** 2 * np.sqrt(3.0) / 12.0) * comm
        return _matrix_exp(omega)
    else:
        raise ValueError(f"Magnus order {order} not supported. Use order 2 or 4.")


def evolve_time_dependent(
    H_func: Callable[[float], np.ndarray],
    num_qubits: int,
    t_span: tuple[float, float],
    steps: int = 100,
    initial_state: np.ndarray | None = None,
    order: int = 4,
) -> EvolutionResult:
    """Simulates time evolution under a time-dependent Hamiltonian H(t)."""
    t0, t1 = t_span
    total_time = t1 - t0
    dt = total_time / steps
    dim = 2 ** num_qubits

    if initial_state is not None:
        state = initial_state.copy().astype(complex)
    else:
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

    energy_history = []
    current_time = t0

    for _step in range(steps):
        H_now = H_func(current_time)
        energy = float(np.real(state.conj() @ H_now @ state))
        energy_history.append(energy)

        U = magnus_step(H_func, current_time, dt, order=order)
        state = U @ state
        current_time += dt

    H_final = H_func(t1)
    final_energy = float(np.real(state.conj() @ H_final @ state))
    energy_history.append(final_energy)

    return EvolutionResult(
        final_state=state,
        energy=final_energy,
        energy_history=energy_history,
        time=total_time,
    )


def evolve(
    hamiltonian: HamiltonianSpec | list[tuple[str, float]],
    num_qubits: int | None = None,
    time: float = 1.0,
    steps: int = 20,
    initial_state: np.ndarray | None = None,
    order: int | None = None,
    method: str = "exact",
) -> EvolutionResult:
    """Simulates time evolution under a Hamiltonian.

    Supports exact spectral evolution as well as Suzuki-Trotter integrators.

    Args:
        hamiltonian: HamiltonianSpec or list of (pauli, coeff) terms.
        num_qubits: Number of qubits (auto-detected from spec).
        time: Total evolution time.
        steps: Evolution steps.
        initial_state: Initial state (default: |0...0>).
        order: Optional Trotter order (1, 2, or 4).
        method: "exact" (spectral decomposition) or "trotter".

    Returns:
        EvolutionResult with final state and energy history.
    """
    if method in ("trotter", "suzuki") or (order is not None and order > 1):
        return trotter_evolve(
            hamiltonian=hamiltonian,
            num_qubits=num_qubits,
            time=time,
            steps=steps,
            order=order or 2,
            initial_state=initial_state,
        )

    if isinstance(hamiltonian, HamiltonianSpec):
        terms = hamiltonian.terms
        n = hamiltonian.num_qubits
    else:
        terms = hamiltonian
        n = num_qubits or max(len(t[0]) for t in terms)

    H_mat = _build_matrix(terms, n)
    dim = 2 ** n
    dt = time / steps

    # Initial state
    if initial_state is not None:
        state = initial_state.copy().astype(complex)
    else:
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

    # Time evolution operator for one step via exact spectral decomposition
    U_step = spectral_unitary_evolution(H_mat, dt)

    energy_history = []

    for _step in range(steps):
        energy = float(np.real(state.conj() @ H_mat @ state))
        energy_history.append(energy)
        state = U_step @ state

    final_energy = float(np.real(state.conj() @ H_mat @ state))
    energy_history.append(final_energy)

    return EvolutionResult(
        final_state=state,
        energy=final_energy,
        energy_history=energy_history,
        time=time,
    )
