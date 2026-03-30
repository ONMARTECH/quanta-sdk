"""
quanta.simulator.router -- Automatic simulator selection based on circuit analysis.

Analyzes circuit structure to choose the optimal simulator backend:

    ┌─────────────────────────────────────────────────┐
    │              Circuit Analysis                    │
    │                                                  │
    │  Clifford only? ──→ PauliFrameSimulator (1000+) │
    │  n ≤ 27?       ──→ StateVectorSimulator (exact) │
    │  Sparse?       ──→ SparseSimulator (40-50)      │
    │  Low χ?        ──→ MPSSimulator (100+)          │
    │  Otherwise     ──→ Error + recommendation       │
    └─────────────────────────────────────────────────┘

Example:
    >>> from quanta.simulator.router import select_simulator
    >>> sim = select_simulator(num_qubits=100, gate_names=["H", "CX", "RZ"])
    MPSSimulator(qubits=100, chi_max=64, ...)
"""

from __future__ import annotations

from quanta.simulator.base import SimulatorBackend

__all__ = ["select_simulator", "analyze_circuit"]

# Gates that are purely Clifford
_CLIFFORD_GATES = frozenset({
    "H", "S", "Sdg", "X", "Y", "Z",
    "CX", "CNOT", "CZ", "SWAP",
    "I",
})


def analyze_circuit(
    gate_names: list[str] | None = None,
    num_qubits: int = 0,
) -> dict[str, object]:
    """Analyzes circuit properties to guide simulator selection.

    Args:
        gate_names: List of gate names in the circuit.
        num_qubits: Number of qubits.

    Returns:
        Dict with analysis results:
        - is_clifford: True if all gates are Clifford.
        - gate_count: Total gate count.
        - has_parametric: True if any parametric gates present.
        - recommended_method: Suggested simulator method.
    """
    if gate_names is None:
        gate_names = []

    is_clifford = all(g in _CLIFFORD_GATES for g in gate_names)
    has_parametric = any(g.startswith("R") or g in ("U1", "U2", "U3") for g in gate_names)

    if is_clifford and num_qubits > 27:
        method = "clifford"
    elif num_qubits <= 27:
        method = "dense"
    elif num_qubits <= 50:
        method = "sparse"
    else:
        method = "mps"

    return {
        "is_clifford": is_clifford,
        "gate_count": len(gate_names),
        "has_parametric": has_parametric,
        "num_qubits": num_qubits,
        "recommended_method": method,
    }


def select_simulator(
    num_qubits: int,
    gate_names: list[str] | None = None,
    seed: int | None = None,
    chi_max: int = 64,
) -> SimulatorBackend:
    """Selects the optimal simulator based on circuit analysis.

    Args:
        num_qubits: Number of qubits.
        gate_names: Gate names used in the circuit (for Clifford detection).
        seed: Random seed.
        chi_max: MPS bond dimension (only used if MPS selected).

    Returns:
        Configured SimulatorBackend instance.
    """
    analysis = analyze_circuit(gate_names, num_qubits)
    method = analysis["recommended_method"]

    if method == "clifford":
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        return PauliFrameSimulator(num_qubits)  # type: ignore[return-value]

    if method == "dense":
        from quanta.simulator.statevector import StateVectorSimulator
        return StateVectorSimulator(num_qubits, seed=seed)

    if method == "sparse":
        from quanta.simulator.sparse import SparseSimulator
        return SparseSimulator(num_qubits, seed=seed)

    if method == "mps":
        from quanta.simulator.mps import MPSSimulator
        return MPSSimulator(num_qubits, seed=seed, chi_max=chi_max)

    # Should never happen
    from quanta.simulator.factory import create_simulator
    return create_simulator(num_qubits, seed=seed)
