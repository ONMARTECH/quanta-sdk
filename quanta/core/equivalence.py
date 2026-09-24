"""
quanta.core.equivalence — Circuit equivalence checking.


Example:
    >>> from quanta.core.equivalence import circuits_equivalent
    >>> circuits_equivalent(bell_v1, bell_v2)
    True
"""

from __future__ import annotations

import numpy as np

from quanta.core.circuit import CircuitDefinition
from quanta.dag.dag_circuit import DAGCircuit
from quanta.simulator.statevector import StateVectorSimulator

# ── Public API ──
__all__ = [
    "circuits_equivalent",
    "get_unitary",
    "unitaries_equivalent",
    "fidelity",
]

def get_unitary(circuit: CircuitDefinition) -> np.ndarray:
    """Computes the unitary matrix of a circuit.


    Args:
        circuit: Circuit defined with @circuit.

    Returns:
        2^n × 2^n unitary matrix.

    """
    builder = circuit.build()
    dag = DAGCircuit.from_builder(builder)
    n = dag.num_qubits
    dim = 2 ** n

    unitary = np.zeros((dim, dim), dtype=complex)

    for col in range(dim):
        # Inject basis state |col⟩
        sim = StateVectorSimulator(n)
        basis = np.zeros(dim, dtype=complex)
        basis[col] = 1.0
        sim.state = basis

        for op in dag.op_nodes():
            sim.apply(op.gate_name, op.qubits, op.params)

        unitary[:, col] = sim.state

    return unitary

def unitaries_equivalent(
    u1: np.ndarray, u2: np.ndarray, atol: float = 1e-12
) -> bool:
    """Checks if two unitary matrices are equivalent up to global phase.

    Uses the normalized Hilbert-Schmidt inner product fidelity:
        F_HS(U1, U2) = (1 / 2^n) * |Tr(U1^dagger @ U2)|
    Two unitaries are equivalent if and only if:
        1. F_HS >= 1.0 - atol
        2. The global phase factor has unit magnitude: ||phase| - 1.0| < atol
        3. U2 matches U1 * phase pointwise within tolerance.

    Args:
        u1: First unitary matrix.
        u2: Second unitary matrix.
        atol: Machine-precision tolerance (default 1e-12).

    Returns:
        True if equivalent up to global phase, False otherwise.
    """
    if u1.shape != u2.shape or u1.ndim != 2 or u1.shape[0] != u1.shape[1]:
        return False

    dim = u1.shape[0]
    hs_prod = np.trace(u1.conj().T @ u2)
    fid = float(np.abs(hs_prod)) / float(dim)

    if fid < 1.0 - atol or fid > 1.0 + atol:
        return False

    # Verify unit-magnitude phase factor
    phase = hs_prod / (fid * dim)
    if abs(abs(phase) - 1.0) >= atol:
        return False

    max_diff = float(np.max(np.abs(u1 * phase - u2)))
    return max_diff <= max(atol * 10.0, 1e-10)

def circuits_equivalent(
    circuit_a: CircuitDefinition,
    circuit_b: CircuitDefinition,
    atol: float = 1e-12,
) -> bool:
    """Checks if two circuits are equivalent.


    Args:

    Returns:
    """
    if circuit_a.num_qubits != circuit_b.num_qubits:
        return False

    u_a = get_unitary(circuit_a)
    u_b = get_unitary(circuit_b)

    return unitaries_equivalent(u_a, u_b, atol=atol)

def fidelity(
    circuit_a: CircuitDefinition,
    circuit_b: CircuitDefinition,
) -> float:
    """Fidelity score between two circuits.

    F = |Tr(U_a† · U_b)| / 2^n


    Args:
        circuit_a: Referans devre.

    Returns:
        Fidelity skoru [0, 1].
    """
    u_a = get_unitary(circuit_a)
    u_b = get_unitary(circuit_b)

    dim = u_a.shape[0]
    trace = np.abs(np.trace(u_a.conj().T @ u_b))

    return float(trace / dim)
