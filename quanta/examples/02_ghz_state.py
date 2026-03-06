"""


Circuit:
    q[0]: ──H──●─────M──
               │
    q[1]: ─────X──●──M──
                  │
    q[2]: ────────X──M──
"""

from quanta import circuit, H, CX, measure, run
from quanta.visualize import draw

@circuit(qubits=3)
def ghz_state(q):
    """GHZ state: (|000⟩ + |111⟩) / √2."""
    H(q[0])
    CX(q[0], q[1])
    CX(q[1], q[2])
    return measure(q)

if __name__ == "__main__":
    print(draw(ghz_state))
    print()

    result = run(ghz_state, shots=4096, seed=42)
    print(result.summary())
