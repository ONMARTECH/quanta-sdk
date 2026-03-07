"""


  2. Oracle: Hedef duruma faz ekle
  4. Tekrarla: √N kez

    q[0]: ──H──[Oracle]──[Diffusion]──M──
    q[1]: ──H──[Oracle]──[Diffusion]──M──
"""


from quanta import CZ, H, X, circuit, measure, run
from quanta.visualize import draw


@circuit(qubits=2)
def grover_2qubit(q):
    """Grover search: finds |11⟩ in a 4-element space.

    2 qubit = 4 durum. Hedef: |11⟩.
    """
    H(q[0])
    H(q[1])

    CZ(q[0], q[1])

    H(q[0])
    H(q[1])
    X(q[0])
    X(q[1])
    CZ(q[0], q[1])
    X(q[0])
    X(q[1])
    H(q[0])
    H(q[1])

    return measure(q)

@circuit(qubits=3)
def grover_3qubit(q):
    """Grover search: finds |111⟩ in an 8-element space.

    3 qubit = 8 durum. Hedef: |111⟩.
    """
    H(q[0])
    H(q[1])
    H(q[2])

    from quanta import CCX
    H(q[2])
    CCX(q[0], q[1], q[2])
    H(q[2])

    H(q[0])
    H(q[1])
    H(q[2])
    X(q[0])
    X(q[1])
    X(q[2])
    H(q[2])
    CCX(q[0], q[1], q[2])
    H(q[2])
    X(q[0])
    X(q[1])
    X(q[2])
    H(q[0])
    H(q[1])
    H(q[2])

    return measure(q)

if __name__ == "__main__":
    print(draw(grover_2qubit))
    result = run(grover_2qubit, shots=1024, seed=42)
    print(result.summary())
    print(f"\n→ Found: |{result.most_frequent}⟩\n")

    print(draw(grover_3qubit))
    result = run(grover_3qubit, shots=1024, seed=42)
    print(result.summary())
    print(f"\n→ Found: |{result.most_frequent}⟩ (success: %{result.probabilities[result.most_frequent]*100:.0f})")
