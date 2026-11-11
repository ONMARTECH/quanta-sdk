"""
quanta.layer3.hamiltonian -- Quantum Hamiltonian simulation & O(m * 2^n) Pauli expectation.

Simulates time evolution ``|psi(t)> = exp(-i * H * t) |psi(0)>`` with exact
spectral decomposition, Suzuki-Trotter splitting (orders 1, 2, 4), Gauss-Legendre
Magnus integrators, and fast ``O(m * 2^n)`` bitwise Pauli expectation evaluation
(replacing ``O(m * 4^n)`` Kronecker loops in energy measurements).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from quanta.layer3.vqe import (
    _compile_pauli_masks,
    _eval_compiled_pauli_expectation,
    pauli_expectation,
)

__all__ = [
    "evolve",
    "molecular_hamiltonian",
    "EvolutionResult",
    "HamiltonianSpec",
    "spectral_unitary_evolution",
    "suzuki_trotter_step",
    "trotter_evolve",
    "magnus_step",
    "evolve_time_dependent",
    "pauli_expectation",
]


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
    """Result of Hamiltonian time evolution."""

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
    """Returns a pre-defined molecular Hamiltonian (H2, LiH, HeH+)."""
    key = molecule.upper().replace(" ", "")
    if key not in _MOLECULES:
        aliases = {"HEH+": "HeH+", "HEH": "HeH+", "H2": "H2", "LIH": "LiH"}
        key = aliases.get(key, key)
    if key not in _MOLECULES:
        available = ", ".join(_MOLECULES.keys())
        raise ValueError(
            f"Unknown molecule: {molecule}. Available: {available}"
        )
    return _MOLECULES[key]


def _apply_pauli_term(
    flip_mask: int,
    z_mask: int,
    phase: complex,
    state: np.ndarray,
    basis_indices: np.ndarray,
) -> np.ndarray:
    """Applies (coeff * P)|state> in O(2^n) bitwise operations."""
    if flip_mask == 0 and z_mask == 0:
        return phase * state
    out = state
    if z_mask != 0:
        x = (basis_indices & np.uint64(z_mask)).astype(np.uint64, copy=True)
        x ^= x >> 32
        x ^= x >> 16
        x ^= x >> 8
        x ^= x >> 4
        x ^= x >> 2
        x ^= x >> 1
        signs = 1.0 - 2.0 * (x & 1).astype(np.float64)
        out = signs * out
    if flip_mask != 0:
        out = out[basis_indices ^ np.uint64(flip_mask)]
    return phase * out


def _apply_pauli_rotation(
    flip_mask: int,
    z_mask: int,
    unit_phase: complex,
    theta: float,
    state: np.ndarray,
    basis_indices: np.ndarray,
) -> np.ndarray:
    """Applies exp(-i * theta * P)|state> = cos(theta)|state> - i*sin(theta)*(P|state>) in O(2^n)
    because every normalized Pauli string satisfies P^2 = I!
    """
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    p_state = _apply_pauli_term(flip_mask, z_mask, unit_phase, state, basis_indices)
    return c * state - 1j * s * p_state


def _build_matrix(terms: list[tuple[str, float]], n: int) -> np.ndarray:
    """Builds full 2^n x 2^n Hermitian matrix using O(m * 2^n) column generation (no O(4^n) kron!)."""
    dim = 1 << n
    compiled = _compile_pauli_masks(terms, n)
    basis_indices = np.arange(dim, dtype=np.uint64)
    H = np.zeros((dim, dim), dtype=complex)
    rows = basis_indices

    for flip_mask, z_mask, phase in compiled:
        cols = rows
        dest_rows = rows ^ np.uint64(flip_mask)
        if z_mask == 0:
            H[dest_rows, cols] += phase
        else:
            x = (cols & np.uint64(z_mask)).astype(np.uint64, copy=True)
            x ^= x >> 32
            x ^= x >> 16
            x ^= x >> 8
            x ^= x >> 4
            x ^= x >> 2
            x ^= x >> 1
            signs = 1.0 - 2.0 * (x & 1).astype(np.float64)
            H[dest_rows, cols] += phase * signs
    return H


def _build_term_matrices(
    terms: list[tuple[str, float]], n: int
) -> list[np.ndarray]:
    """Builds individual Hermitian term matrices via O(2^n) sparse column mapping."""
    return [_build_matrix([term], n) for term in terms]


def spectral_unitary_evolution(H: np.ndarray, dt: float) -> np.ndarray:
    """Computes U(dt) = exp(-i * H * dt) via exact Hermitian spectral decomposition."""
    H_herm = (H + H.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(H_herm)
    phases = np.exp(-1j * eigenvalues * dt)
    return (eigenvectors * phases) @ eigenvectors.conj().T


def _matrix_exp(A: np.ndarray) -> np.ndarray:
    """Matrix exponential using spectral decomposition or scipy.linalg.expm."""
    scale = max(float(np.max(np.abs(A))), 1e-14)
    anti_herm_diff = float(np.max(np.abs(A + A.conj().T)))
    herm_diff = float(np.max(np.abs(A - A.conj().T)))

    if anti_herm_diff / scale < 1e-10:
        K = 1j * A
        K_herm = (K + K.conj().T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(K_herm)
        phases = np.exp(-1j * eigenvalues)
        return (eigenvectors * phases) @ eigenvectors.conj().T
    if herm_diff / scale < 1e-10:
        A_herm = (A + A.conj().T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(A_herm)
        return (eigenvectors * np.exp(eigenvalues)) @ eigenvectors.conj().T

    from scipy.linalg import expm

    return expm(A)


def suzuki_trotter_step(
    term_matrices: list[np.ndarray],
    dt: float,
    order: int = 2,
) -> np.ndarray:
    """Computes a single Suzuki-Trotter step unitary matrix (order 1, 2, or 4)."""
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

    if order == 2:
        U = np.eye(dim, dtype=complex)
        for k in range(m - 1):
            U = spectral_unitary_evolution(term_matrices[k], dt / 2.0) @ U
        U = spectral_unitary_evolution(term_matrices[m - 1], dt) @ U
        for k in range(m - 2, -1, -1):
            U = spectral_unitary_evolution(term_matrices[k], dt / 2.0) @ U
        return U

    if order == 4:
        p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        S2_p = suzuki_trotter_step(term_matrices, p * dt, order=2)
        S2_mid = suzuki_trotter_step(term_matrices, (1.0 - 4.0 * p) * dt, order=2)
        return S2_p @ S2_p @ S2_mid @ S2_p @ S2_p

    raise ValueError(f"Unsupported Trotter order {order}. Must be 1, 2, or 4.")


def _trotter_state_step(
    compiled_unit: list[tuple[int, int, complex, float]],
    dt: float,
    order: int,
    state: np.ndarray,
    basis_indices: np.ndarray,
) -> np.ndarray:
    """Matrix-free O(m * 2^n) Suzuki-Trotter state evolution step using P^2 = I!"""
    m = len(compiled_unit)
    if m == 0:
        return state
    if m == 1:
        f_m, z_m, u_ph, coeff = compiled_unit[0]
        return _apply_pauli_rotation(f_m, z_m, u_ph, coeff * dt, state, basis_indices)

    if order == 1:
        cur = state
        for f_m, z_m, u_ph, coeff in compiled_unit:
            cur = _apply_pauli_rotation(
                f_m, z_m, u_ph, coeff * dt, cur, basis_indices
            )
        return cur

    if order == 2:
        cur = state
        for k in range(m - 1):
            f_m, z_m, u_ph, coeff = compiled_unit[k]
            cur = _apply_pauli_rotation(
                f_m, z_m, u_ph, 0.5 * coeff * dt, cur, basis_indices
            )
        f_m, z_m, u_ph, coeff = compiled_unit[m - 1]
        cur = _apply_pauli_rotation(f_m, z_m, u_ph, coeff * dt, cur, basis_indices)
        for k in range(m - 2, -1, -1):
            f_m, z_m, u_ph, coeff = compiled_unit[k]
            cur = _apply_pauli_rotation(
                f_m, z_m, u_ph, 0.5 * coeff * dt, cur, basis_indices
            )
        return cur

    if order == 4:
        p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        cur = _trotter_state_step(compiled_unit, p * dt, 2, state, basis_indices)
        cur = _trotter_state_step(compiled_unit, p * dt, 2, cur, basis_indices)
        cur = _trotter_state_step(
            compiled_unit, (1.0 - 4.0 * p) * dt, 2, cur, basis_indices
        )
        cur = _trotter_state_step(compiled_unit, p * dt, 2, cur, basis_indices)
        return _trotter_state_step(compiled_unit, p * dt, 2, cur, basis_indices)

    raise ValueError(f"Unsupported Trotter order {order}. Must be 1, 2, or 4.")


def trotter_evolve(
    hamiltonian: HamiltonianSpec | list[tuple[str, float]],
    num_qubits: int | None = None,
    time: float = 1.0,
    steps: int = 20,
    order: int = 2,
    initial_state: np.ndarray | None = None,
) -> EvolutionResult:
    """Simulates Suzuki-Trotter evolution in O(steps * m * 2^n) without any O(4^n) matrices."""
    if isinstance(hamiltonian, HamiltonianSpec):
        terms = hamiltonian.terms
        n = hamiltonian.num_qubits
    else:
        terms = hamiltonian
        n = num_qubits or max(len(t[0]) for t in terms)

    compiled = _compile_pauli_masks(terms, n)
    unit_compiled: list[tuple[int, int, complex, float]] = []
    for (pauli_str, coeff), (f_m, z_m, _) in zip(terms, compiled, strict=True):
        y_cnt = pauli_str.ljust(n, "I")[:n].count("Y")
        unit_compiled.append((f_m, z_m, 1j**y_cnt, float(coeff)))

    dim = 1 << n
    basis_indices = np.arange(dim, dtype=np.uint64)
    dt = time / steps

    if initial_state is not None:
        state = initial_state.copy().astype(complex)
    else:
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

    energy_history: list[float] = []
    for _step in range(steps):
        energy = _eval_compiled_pauli_expectation(compiled, state, basis_indices)
        energy_history.append(energy)
        state = _trotter_state_step(unit_compiled, dt, order, state, basis_indices)

    final_energy = _eval_compiled_pauli_expectation(compiled, state, basis_indices)
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
    """Computes unitary time evolution step U(t + dt, t) via Magnus expansion."""
    if order == 2:
        H_mid = H_func(t + dt / 2.0)
        return spectral_unitary_evolution(H_mid, dt)
    if order == 4:
        c1 = 0.5 - np.sqrt(3.0) / 6.0
        c2 = 0.5 + np.sqrt(3.0) / 6.0
        H1 = H_func(t + c1 * dt)
        H2 = H_func(t + c2 * dt)
        comm = H2 @ H1 - H1 @ H2
        omega = -1j * (dt / 2.0) * (H1 + H2) - (dt**2 * np.sqrt(3.0) / 12.0) * comm
        return _matrix_exp(omega)
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
    dim = 1 << num_qubits

    if initial_state is not None:
        state = initial_state.copy().astype(complex)
    else:
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

    energy_history: list[float] = []
    current_time = t0

    for _step in range(steps):
        H_now = H_func(current_time)
        energy_history.append(float(np.real(state.conj() @ H_now @ state)))
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
    """Simulates time evolution under a Pauli Hamiltonian."""
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

    compiled = _compile_pauli_masks(terms, n)
    dim = 1 << n
    basis_indices = np.arange(dim, dtype=np.uint64)
    H_mat = _build_matrix(terms, n)
    dt = time / steps

    if initial_state is not None:
        state = initial_state.copy().astype(complex)
    else:
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

    U_step = spectral_unitary_evolution(H_mat, dt)
    energy_history: list[float] = []

    for _step in range(steps):
        energy = _eval_compiled_pauli_expectation(compiled, state, basis_indices)
        energy_history.append(energy)
        state = U_step @ state

    final_energy = _eval_compiled_pauli_expectation(compiled, state, basis_indices)
    energy_history.append(final_energy)

    return EvolutionResult(
        final_state=state,
        energy=final_energy,
        energy_history=energy_history,
        time=time,
    )
