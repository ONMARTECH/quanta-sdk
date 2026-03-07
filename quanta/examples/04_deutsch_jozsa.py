"""

Bir fonksiyonun "sabit" mi (hep 0 veya hep 1) yoksa

    q[0]: ──H──[Oracle]──H──M──
    q[1]: ──H──[Oracle]──H──M──
    q[2]: ──X──H──[Oracle]──────

"""

from quanta import CX, H, X, circuit, measure, run
from quanta.visualize import draw


@circuit(qubits=3)
def deutsch_jozsa_balanced(q):
    """Deutsch-Jozsa: Dengeli oracle ile (f(x) = x₁ XOR x₂).

    """
    X(q[2])
    H(q[0])
    H(q[1])
    H(q[2])

    # Dengeli oracle: f(x) = x₁ XOR x₂
    CX(q[0], q[2])
    CX(q[1], q[2])

    # Geri Hadamard
    H(q[0])
    H(q[1])

    return measure(q[0], q[1])

@circuit(qubits=3)
def deutsch_jozsa_constant(q):
    """Deutsch-Jozsa: Sabit oracle ile (f(x) = 0 her zaman).

    """
    X(q[2])
    H(q[0])
    H(q[1])
    H(q[2])


    H(q[0])
    H(q[1])

    return measure(q[0], q[1])

if __name__ == "__main__":
    print("═══ Deutsch-Jozsa: Dengeli Oracle ═══\n")
    print(draw(deutsch_jozsa_balanced))
    result = run(deutsch_jozsa_balanced, shots=1024, seed=42)
    print(result.summary())
    print(f"\n→ Fonksiyon: {answer}\n")

    print("═══ Deutsch-Jozsa: Sabit Oracle ═══\n")
    print(draw(deutsch_jozsa_constant))
    result = run(deutsch_jozsa_constant, shots=1024, seed=42)
    print(result.summary())
    print(f"\n→ Fonksiyon: {answer}")
