"""

  - %50 ihtimalle |00⟩
  - %50 ihtimalle |11⟩


Circuit:
    q[0]: ──H──●──M──
               │
    q[1]: ─────X──M──
"""

from quanta import CX, H, circuit, measure, run
from quanta.visualize import draw


@circuit(qubits=2)
def bell_state(q):
    """Bell state devresi: |Φ+⟩ = (|00⟩ + |11⟩) / √2."""
    H(q[0])           # Superposition: |0⟩ → (|0⟩ + |1⟩)/√2
    CX(q[0], q[1])    # Entanglement: |00⟩+|10⟩ → |00⟩+|11⟩
    return measure(q)

if __name__ == "__main__":
    print("═══ Bell State Circuitsi ═══\n")
    print(draw(bell_state))
    print()

    result = run(bell_state, shots=4096, seed=42)
    print(result.summary())
