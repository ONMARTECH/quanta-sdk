"""
quanta.layer3.vqe -- Variational Quantum Eigensolver (VQE).

Evaluates Pauli Hamiltonian expectation values ``<psi|H|psi>`` in ``O(m * 2^n)``
bitwise index-flip and phase operations WITHOUT building dense ``2^n x 2^n``
Kronecker matrices during variational optimization.
"""

from __future__ import annotations

import numpy as np

from quanta.simulator.factory import create_simulator

__all__ = ["vqe", "VQEResult", "build_hamiltonian_matrix", "pauli_expectation"]


_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


class VQEResult:
    """Result of a VQE computation."""

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
        lines = [
            "=== VQE Result ===",
            f"  Ground state energy: {self.energy:.6f}",
            f"  Iterations: {self.num_iterations}",
            f"  Convergence: {abs(self.history[-1] - self.history[-2]):.2e}"
            if len(self.history) > 1
            else "",
        ]
        return "\n".join(lines)


def build_hamiltonian_matrix(
    terms: list[tuple[str, float]],
    num_qubits: int,
) -> np.ndarray:
    """Builds the explicit 2^n x 2^n Hermitian matrix from Pauli terms (for exact diag/testing)."""
    dim = 1 << num_qubits
    H_mat = np.zeros((dim, dim), dtype=complex)

    for pauli_str, coeff in terms:
        padded = pauli_str.ljust(num_qubits, "I")
        term = np.array([[1.0]], dtype=complex)
        for ch in padded:
            term = np.kron(term, _PAULI[ch])
        H_mat += coeff * term

    return H_mat


def _compile_pauli_masks(
    terms: list[tuple[str, float]], num_qubits: int
) -> list[tuple[int, int, complex]]:
    """Precompiles each Pauli term into (flip_mask, z_mask, coeff * i^{y_count}) in MSB order.

    Using ``Y = i * X * Z``, for any computational basis state ``|x>``:
        P |x> = (i^{y_count}) * (-1)^{popcount(x & z_mask)} |x ^ flip_mask>.
    """
    compiled: list[tuple[int, int, complex]] = []
    for pauli_str, coeff in terms:
        padded = pauli_str.ljust(num_qubits, "I")
        flip_mask = 0
        z_mask = 0
        y_count = 0
        for q, ch in enumerate(padded[:num_qubits]):
            bit = 1 << (num_qubits - 1 - q)
            if ch == "X":
                flip_mask |= bit
            elif ch == "Y":
                flip_mask |= bit
                z_mask |= bit
                y_count += 1
            elif ch == "Z":
                z_mask |= bit
            elif ch != "I":
                raise ValueError(f"Unknown Pauli operator character: {ch!r}")
        phase = complex(coeff) * (1j**y_count)
        compiled.append((flip_mask, z_mask, phase))
    return compiled


def _popcount_parity_signs(masked: np.ndarray) -> np.ndarray:
    """Vectorized (-1)^{popcount(masked)} over uint64 basis indices in O(2^n)."""
    x = masked.astype(np.uint64, copy=True)
    x ^= x >> 32
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return 1.0 - 2.0 * (x & 1).astype(np.float64)


def _eval_compiled_pauli_expectation(
    compiled_terms: list[tuple[int, int, complex]],
    statevector: np.ndarray,
    basis_indices: np.ndarray,
) -> float:
    """Evaluates <psi|H|psi> in O(m * 2^n) without any 4^n matrix construction."""
    probs: np.ndarray | None = None
    total = 0.0
    for flip_mask, z_mask, phase in compiled_terms:
        if flip_mask == 0 and z_mask == 0:
            total += float(np.real(phase))
            continue
        if flip_mask == 0:
            if probs is None:
                probs = np.abs(statevector) ** 2
            signs = _popcount_parity_signs(basis_indices & np.uint64(z_mask))
            total += float(np.real(phase) * np.dot(signs, probs))
        else:
            partner = statevector[basis_indices ^ np.uint64(flip_mask)]
            if z_mask == 0:
                overlap = np.vdot(partner, statevector)
            else:
                signs = _popcount_parity_signs(basis_indices & np.uint64(z_mask))
                overlap = np.vdot(partner, signs * statevector)
            total += float(np.real(phase * overlap))
    return total


def pauli_expectation(
    terms: list[tuple[str, float]],
    statevector: np.ndarray,
    num_qubits: int,
) -> float:
    """Public O(m * 2^n) Pauli expectation evaluator ``<psi|H|psi>``."""
    compiled = _compile_pauli_masks(terms, num_qubits)
    basis_indices = np.arange(1 << num_qubits, dtype=np.uint64)
    return _eval_compiled_pauli_expectation(compiled, statevector, basis_indices)


def _expectation_value(
    hamiltonian: np.ndarray | list[tuple[str, float]],
    statevector: np.ndarray,
) -> float:
    """Computes <psi|H|psi> for either Pauli term list (O(m*2^n)) or dense matrix."""
    if isinstance(hamiltonian, list):
        n = int(round(np.log2(len(statevector))))
        return pauli_expectation(hamiltonian, statevector, n)
    return float(np.real(statevector.conj() @ hamiltonian @ statevector))


def _build_ansatz_state(
    num_qubits: int,
    layers: int,
    params: np.ndarray,
) -> np.ndarray:
    """Hardware-efficient RY + RZ + linear CX ansatz."""
    sim = create_simulator(num_qubits)
    idx = 0
    for _layer in range(layers):
        for q in range(num_qubits):
            sim.apply("RY", (q,), (float(params[idx]),))
            idx += 1
            sim.apply("RZ", (q,), (float(params[idx]),))
            idx += 1
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
    """Variational Quantum Eigensolver using O(m * 2^n) bitwise Pauli expectations."""
    rng = np.random.default_rng(seed)
    compiled_terms = _compile_pauli_masks(hamiltonian, num_qubits)
    basis_indices = np.arange(1 << num_qubits, dtype=np.uint64)

    num_params = 2 * num_qubits * layers
    params = rng.uniform(-np.pi, np.pi, size=num_params)

    history: list[float] = []
    best_energy = float("inf")
    best_params = params.copy()
    shift = np.pi / 2.0

    # Adam moments for reliable convergence within max_iter
    m_state = np.zeros(num_params, dtype=float)
    v_state = np.zeros(num_params, dtype=float)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for iteration in range(max_iter):
        state = _build_ansatz_state(num_qubits, layers, params)
        energy = _eval_compiled_pauli_expectation(
            compiled_terms, state, basis_indices
        )
        history.append(energy)

        if energy < best_energy:
            best_energy = energy
            best_params = params.copy()

        gradients = np.zeros(num_params, dtype=float)
        for i in range(num_params):
            orig = params[i]

            params[i] = orig + shift
            state_plus = _build_ansatz_state(num_qubits, layers, params)
            e_plus = _eval_compiled_pauli_expectation(
                compiled_terms, state_plus, basis_indices
            )

            params[i] = orig - shift
            state_minus = _build_ansatz_state(num_qubits, layers, params)
            e_minus = _eval_compiled_pauli_expectation(
                compiled_terms, state_minus, basis_indices
            )

            params[i] = orig
            gradients[i] = 0.5 * (e_plus - e_minus)

        t = iteration + 1
        m_state = beta1 * m_state + (1.0 - beta1) * gradients
        v_state = beta2 * v_state + (1.0 - beta2) * (gradients**2)
        m_hat = m_state / (1.0 - beta1**t)
        v_hat = v_state / (1.0 - beta2**t)
        params -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        if iteration > 10 and abs(history[-1] - history[-2]) < 1e-9:
            break

    # Final check after last update
    final_state = _build_ansatz_state(num_qubits, layers, params)
    final_energy = _eval_compiled_pauli_expectation(
        compiled_terms, final_state, basis_indices
    )
    if final_energy < best_energy:
        best_energy = final_energy
        best_params = params.copy()

    return VQEResult(
        energy=best_energy,
        optimal_params=best_params,
        history=history,
    )
