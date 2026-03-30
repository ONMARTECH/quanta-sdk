"""
quanta.simulator.factory -- Simulator creation and selection.

Central factory for creating simulator instances. Layer 3 modules
should use this instead of directly importing specific simulators.

Usage:
    from quanta.simulator.factory import create_simulator
    sim = create_simulator(num_qubits=20)                  # auto-select
    sim = create_simulator(num_qubits=20, method="dense")  # explicit
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quanta.simulator.base import SimulatorBackend

if TYPE_CHECKING:
    pass

__all__ = ["create_simulator"]

# Simulator methods registry
METHODS = ("auto", "dense", "sparse", "mps", "clifford")


def create_simulator(
    num_qubits: int,
    method: str = "auto",
    seed: int | None = None,
    **kwargs: object,
) -> SimulatorBackend:
    """Creates the best simulator for the given qubit count.

    Args:
        num_qubits: Number of qubits to simulate.
        method: Simulator method. One of:
            - "auto": Automatic selection based on qubit count.
            - "dense": Full statevector (max ~27 qubits).
            - "sparse": Sparse statevector (max ~50 qubits).
            - "mps": Matrix Product State (100+ qubits, low entanglement).
            - "clifford": Pauli frame / stabilizer (1000+ qubits, Clifford only).
        seed: Random seed for reproducibility.
        **kwargs: Backend-specific parameters (e.g., chi_max for MPS).

    Returns:
        SimulatorBackend instance ready for gate application.

    Raises:
        ValueError: If method is unknown or qubit count exceeds limits.
    """
    if method not in METHODS:
        raise ValueError(
            f"Unknown simulator method: {method!r}. "
            f"Available: {METHODS}"
        )

    if method == "auto":
        method = _select_method(num_qubits)

    if method == "dense":
        from quanta.simulator.statevector import StateVectorSimulator
        return StateVectorSimulator(num_qubits, seed=seed)

    if method == "sparse":
        # Phase 2: Will be implemented
        try:
            from quanta.simulator.sparse import SparseSimulator
            return SparseSimulator(num_qubits, seed=seed, **kwargs)
        except ImportError:
            # Fallback to dense if sparse not yet available
            from quanta.simulator.statevector import StateVectorSimulator
            return StateVectorSimulator(num_qubits, seed=seed)

    if method == "mps":
        # Phase 3: Will be implemented
        try:
            from quanta.simulator.mps import MPSSimulator
            return MPSSimulator(num_qubits, seed=seed, **kwargs)
        except ImportError:
            raise ValueError(
                "MPS simulator not yet available. "
                "Use method='dense' for ≤27 qubits."
            ) from None

    if method == "clifford":
        from quanta.simulator.pauli_frame import PauliFrameSimulator
        return PauliFrameSimulator(num_qubits)  # type: ignore[return-value]

    raise ValueError(f"Unhandled method: {method!r}")


def _select_method(num_qubits: int) -> str:
    """Auto-selects the best simulator method.

    Strategy:
        ≤ 27 qubits → dense (exact, fast)
        28-50 qubits → sparse (if available, else error)
        > 50 qubits → mps (if available, else error)
    """
    if num_qubits <= 27:
        return "dense"

    # Try sparse for medium circuits
    try:
        from quanta.simulator.sparse import SparseSimulator  # noqa: F401
        if num_qubits <= 50:
            return "sparse"
    except ImportError:
        pass

    # Try MPS for large circuits
    try:
        from quanta.simulator.mps import MPSSimulator  # noqa: F401
        return "mps"
    except ImportError:
        pass

    raise ValueError(
        f"No simulator available for {num_qubits} qubits. "
        f"Dense supports ≤27. Install sparse/MPS extensions for larger circuits."
    )
