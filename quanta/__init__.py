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

__version__ = "0.8.0"

from quanta.core.circuit import circuit
from quanta.core.custom_gate import custom_gate
from quanta.core.gates import (
    CCX,
    CX,
    CY,
    CZ,
    RC3X,
    RCCX,
    RX,
    RXX,
    RY,
    RZ,
    RZZ,
    SDG,
    SWAP,
    SX,
    TDG,
    H,
    # New IBM-parity gates
    I,
    P,
    S,
    SXdg,
    T,
    U,
    X,
    Y,
    Z,
)
from quanta.core.measure import measure
from quanta.runner import run, sweep

__all__ = [
    # Gates (original)
    "H", "X", "Y", "Z", "S", "T",
    "CX", "CY", "CZ", "SWAP", "CCX",
    "RX", "RY", "RZ",
    # Gates (new)
    "I", "SDG", "TDG", "P", "SX", "SXdg",
    "U", "RXX", "RZZ", "RCCX", "RC3X",
    # Custom gates
    "custom_gate",
    # Circuit
    "circuit",
    # Measurement
    "measure",
    # Execution
    "run", "sweep",
]
