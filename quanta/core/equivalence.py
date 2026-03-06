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
    """Circuitnin unitär matrisini hesaplar.


    Args:
        circuit: Circuit defined with @circuit.

    Returns:
        2^n × 2^n unitär matris.

    """
    builder = circuit.build()
    dag = DAGCircuit.from_builder(builder)
    n = dag.num_qubits
    dim = 2 ** n

    unitary = np.zeros((dim, dim), dtype=complex)

    for col in range(dim):
        # Temel durum |col⟩'i girdiye koy
        sim = StateVectorSimulator(n)
        sim._state = np.zeros(dim, dtype=complex)
        sim._state[col] = 1.0

        for op in dag.op_nodes():
            sim.apply(op.gate_name, op.qubits, op.params)

        unitary[:, col] = sim.state

    return unitary

def unitaries_equivalent(
    u1: np.ndarray, u2: np.ndarray, atol: float = 1e-8
) -> bool:
    """Checks if two unitary matrices are equivalent up to global phase.


    Args:

    Returns:
    """
    if u1.shape != u2.shape:
        return False

    for i in range(u1.shape[0]):
        for j in range(u1.shape[1]):
            if abs(u1[i, j]) > atol and abs(u2[i, j]) > atol:
                phase = u2[i, j] / u1[i, j]
                u1_adjusted = u1 * phase
                return np.allclose(u1_adjusted, u2, atol=atol)

    return np.allclose(u1, u2, atol=atol)

def circuits_equivalent(
    circuit_a: CircuitDefinition,
    circuit_b: CircuitDefinition,
    atol: float = 1e-8,
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

    return trace / dim
