"""


3 qubit:

Circuit:
    q[0]: ──RY(θ)──────●──H──M──
                        │
    q[1]: ────────H──●──X─────M──
                     │
    q[2]: ───────────X───────────
"""

import numpy as np

from quanta import CX, RY, H, circuit, measure, run
from quanta.visualize import draw


@circuit(qubits=3)
def teleportation(q):
    """Quantum teleportation circuit.

    """
    RY(np.pi / 3)(q[0])

    H(q[1])
    CX(q[1], q[2])

    CX(q[0], q[1])
    H(q[0])

    return measure(q)

if __name__ == "__main__":
    print(draw(teleportation))
    print()

    result = run(teleportation, shots=4096, seed=42)
    print(result.summary())
