"""quanta.simulator — Quantum simulators.

Available backends:
    - StateVectorSimulator: Dense statevector (≤ 27 qubits, exact)
    - SparseSimulator: Sparse statevector (≤ 50 qubits, sparse circuits)
    - MPSSimulator: Matrix Product State (100+ qubits, low entanglement)
    - PauliFrameSimulator: Stabilizer tableau (1000+ qubits, Clifford only)

Factory:
    create_simulator(n, method="auto") → auto-selects best backend
    select_simulator(n, gate_names=[...]) → circuit-aware selection
"""

from quanta.simulator.base import SimulatorBackend
from quanta.simulator.factory import create_simulator
from quanta.simulator.mlx import MLXSimulator, is_mlx_available
from quanta.simulator.router import select_simulator

__all__ = [
    "SimulatorBackend",
    "MLXSimulator",
    "is_mlx_available",
    "create_simulator",
    "select_simulator",
]
