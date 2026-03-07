"""
Quanta -- Clean, concise, and multi-paradigm quantum computing SDK.

3-layer abstraction:
  - Layer 3 (Declarative): State intent, SDK solves
  - Layer 2 (Algorithmic): Write circuits with @circuit
  - Layer 1 (Physical): Optimize with DAG + compiler

Quick Start:
    >>> from quanta import circuit, H, CX, measure, run
    >>> @circuit(qubits=2)
    ... def bell(q):
    ...     H(q[0])
    ...     CX(q[0], q[1])
    ...     return measure(q)
    >>> result = run(bell, shots=1024)
    >>> print(result)
"""

__version__ = "0.6.0"

# -- Core API --
from quanta.core.gates import (
    H, X, Y, Z, S, T,
    CX, CZ, SWAP, CCX,
    RX, RY, RZ,
)
from quanta.core.circuit import circuit
from quanta.core.measure import measure
from quanta.core.custom_gate import custom_gate
from quanta.runner import run, sweep

__all__ = [
    # Gates
    "H", "X", "Y", "Z", "S", "T",
    "CX", "CZ", "SWAP", "CCX",
    "RX", "RY", "RZ",
    # Custom gates
    "custom_gate",
    # Circuit
    "circuit",
    # Measurement
    "measure",
    # Execution
    "run", "sweep",
]
