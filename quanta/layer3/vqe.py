"""
quanta.layer3.vqe -- Variational Quantum Eigensolver (VQE).

Finds the ground-state energy of a Hamiltonian using a hybrid
quantum-classical optimization loop. The quantum part prepares
a parameterized ansatz and measures, the classical part optimizes.

This is the core algorithm for:
  - Molecular energy calculations (quantum chemistry)
  - Material science simulations
  - Physics ground-state problems

Example:
    >>> from quanta.layer3.vqe import vqe
    >>> # Find ground state of a simple 2-qubit Hamiltonian
    >>> H = [("ZZ", 0.5), ("Z", -1.0), ("X", 0.25)]
    >>> result = vqe(num_qubits=2, hamiltonian=H, layers=3)
    >>> print(f"Ground state energy: {result.energy:.4f}")
"""

from __future__ import annotations

import numpy as np

from quanta.core.circuit import CircuitDefinition, circuit, CircuitBuilder
from quanta.core.gates import H, RX, RY, RZ, CX, GATE_REGISTRY
from quanta.core.measure import measure
from quanta.simulator.statevector import StateVectorSimulator

__all__ = ["vqe", "VQEResult"]


# -- Pauli matrices --
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


class VQEResult:
    """Result of a VQE computation.

    Attributes:
        energy: Estimated ground-state energy.
        optimal_params: Best parameter values found.
        history: Energy values during optimization.
        num_iterations: Total optimization steps.
    """

    def __init__(
        self,
        energy: float,
        optimal_params: np.ndarray,
        history: list[float],
    ) -> None:
        self.energy = energy
        self.optimal_params = optimal_params
        self.history = history
        self.num_iterations = len(history)

    def __repr__(self) -> str:
        return (
            f"VQEResult(energy={self.energy:.6f}, "
            f"iterations={self.num_iterations})"
        )

    def summary(self) -> str:
        """Pretty summary of VQE result."""
        lines = [
            "=== VQE Result ===",
            f"  Ground state energy: {self.energy:.6f}",
            f"  Iterations: {self.num_iterations}",
            f"  Convergence: {abs(self.history[-1] - self.history[-2]):.2e}"
            if len(self.history) > 1 else "",
        ]
        return "\n".join(lines)


def build_hamiltonian_matrix(
    terms: list[tuple[str, float]],
    num_qubits: int,
) -> np.ndarray:
    """Builds full Hamiltonian matrix from Pauli terms.

    Args:
        terms: List of (pauli_string, coefficient).
            E.g. [("ZZ", 0.5), ("XI", -0.3), ("IZ", 0.8)]
        num_qubits: Number of qubits.

    Returns:
        2^n x 2^n Hermitian matrix.
    """
    dim = 2 ** num_qubits
    H_mat = np.zeros((dim, dim), dtype=complex)

    for pauli_str, coeff in terms:
        # Pad with I if needed
        padded = pauli_str.ljust(num_qubits, "I")

        # Build tensor product
        term = np.array([[1.0]], dtype=complex)
        for ch in padded:
            term = np.kron(term, _PAULI[ch])

        H_mat += coeff * term

    return H_mat


def _expectation_value(
    hamiltonian: np.ndarray,
    statevector: np.ndarray,
) -> float:
    """Computes <psi|H|psi>."""
    return float(np.real(statevector.conj() @ hamiltonian @ statevector))


def _build_ansatz_state(
    num_qubits: int,
    layers: int,
    params: np.ndarray,
) -> np.ndarray:
    """Builds ansatz statevector from parameters.

    Hardware-efficient ansatz:
      - RY on each qubit (per layer)
      - RZ on each qubit (per layer)
      - CNOT chain entanglement (per layer)
    """
    sim = StateVectorSimulator(num_qubits)
    idx = 0

    for layer in range(layers):
        # Rotation layer
        for q in range(num_qubits):
            sim.apply("RY", (q,), (params[idx],))
            idx += 1
            sim.apply("RZ", (q,), (params[idx],))
            idx += 1

        # Entanglement layer (CNOT chain)
        for q in range(num_qubits - 1):
            sim.apply("CX", (q, q + 1))

    return sim.state


def vqe(
    num_qubits: int,
    hamiltonian: list[tuple[str, float]],
    layers: int = 2,
    max_iter: int = 200,
    learning_rate: float = 0.1,
    seed: int | None = None,
) -> VQEResult:
    """Variational Quantum Eigensolver.

    Finds ground-state energy of a Hamiltonian using parameterized
    quantum circuits and classical optimization.

    Args:
        num_qubits: Number of qubits.
        hamiltonian: Pauli terms as [(string, coeff), ...].
            E.g. [("ZZ", 0.5), ("Z", -1.0), ("X", 0.25)]
        layers: Ansatz circuit depth.
        max_iter: Maximum optimization iterations.
        learning_rate: Step size for parameter updates.
        seed: Random seed.

    Returns:
        VQEResult with energy, optimal params, and history.

    Example:
        >>> result = vqe(2, [("ZZ", 1.0), ("XI", 0.5), ("IX", 0.5)])
        >>> print(f"Energy: {result.energy:.4f}")
    """
    rng = np.random.default_rng(seed)

    # Build Hamiltonian matrix
    H_mat = build_hamiltonian_matrix(hamiltonian, num_qubits)

    # Exact ground state (for validation)
    eigenvalues = np.linalg.eigvalsh(H_mat)
    exact_ground = float(eigenvalues[0])

    # Number of parameters: 2 * num_qubits * layers (RY + RZ per qubit per layer)
    num_params = 2 * num_qubits * layers
    params = rng.uniform(-np.pi, np.pi, size=num_params)

    history: list[float] = []
    best_energy = float("inf")
    best_params = params.copy()

    for iteration in range(max_iter):
        # Forward pass
        state = _build_ansatz_state(num_qubits, layers, params)
        energy = _expectation_value(H_mat, state)
        history.append(energy)

        if energy < best_energy:
            best_energy = energy
            best_params = params.copy()

        # Parameter-shift gradient estimation
        gradients = np.zeros(num_params)
        shift = np.pi / 2

        for i in range(num_params):
            params_plus = params.copy()
            params_plus[i] += shift
            state_plus = _build_ansatz_state(num_qubits, layers, params_plus)
            e_plus = _expectation_value(H_mat, state_plus)

            params_minus = params.copy()
            params_minus[i] -= shift
            state_minus = _build_ansatz_state(num_qubits, layers, params_minus)
            e_minus = _expectation_value(H_mat, state_minus)

            gradients[i] = (e_plus - e_minus) / 2

        # Update
        params -= learning_rate * gradients

        # Early stopping
        if iteration > 10 and abs(history[-1] - history[-2]) < 1e-8:
            break

    return VQEResult(
        energy=best_energy,
        optimal_params=best_params,
        history=history,
    )
