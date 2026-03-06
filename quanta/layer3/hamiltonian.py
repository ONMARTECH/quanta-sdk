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

from dataclasses import dataclass

import numpy as np

from quanta.simulator.statevector import StateVectorSimulator

__all__ = [
    "evolve", "molecular_hamiltonian", "EvolutionResult",
    "HamiltonianSpec",
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


def evolve(
    hamiltonian: HamiltonianSpec | list[tuple[str, float]],
    num_qubits: int | None = None,
    time: float = 1.0,
    steps: int = 20,
    initial_state: np.ndarray | None = None,
) -> EvolutionResult:
    """Simulates time evolution under a Hamiltonian.

    Uses first-order Trotter decomposition:
      exp(-iHt) ≈ (exp(-iH*dt))^steps

    Args:
        hamiltonian: HamiltonianSpec or list of (pauli, coeff) terms.
        num_qubits: Number of qubits (auto-detected from spec).
        time: Total evolution time.
        steps: Trotter steps (more = more accurate).
        initial_state: Initial state (default: |0...0>).

    Returns:
        EvolutionResult with final state and energy history.
    """
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

    # Time evolution operator for one step
    U_step = _matrix_exp(-1j * H_mat * dt)

    energy_history = []

    for step in range(steps):
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


def _matrix_exp(A: np.ndarray) -> np.ndarray:
    """Matrix exponential using eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(
        (A + A.conj().T) / 2  # Ensure Hermitian
    )
    return eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.conj().T
