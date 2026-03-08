"""
Example 04: Deutsch-Jozsa Algorithm

Determines whether a function is "constant" (always 0 or always 1)
or "balanced" (outputs 0 for half the inputs, 1 for the other half).

    q[0]: ──H──[Oracle]──H──M──
    q[1]: ──H──[Oracle]──H──M──
    q[2]: ──X──H──[Oracle]──────

"""

from quanta import CX, H, X, circuit, measure, run
from quanta.visualize import draw


@circuit(qubits=3)
def deutsch_jozsa_balanced(q):
    """Deutsch-Jozsa: Balanced oracle (f(x) = x₁ XOR x₂).

    """
    X(q[2])
    H(q[0])
    H(q[1])
    H(q[2])

    # Balanced oracle: f(x) = x₁ XOR x₂
    CX(q[0], q[2])
    CX(q[1], q[2])

    # Final Hadamard
    H(q[0])
    H(q[1])

    return measure(q[0], q[1])

@circuit(qubits=3)
def deutsch_jozsa_constant(q):
    """Deutsch-Jozsa: Constant oracle (f(x) = 0 always).

    """
    X(q[2])
    H(q[0])
    H(q[1])
    H(q[2])


    H(q[0])
    H(q[1])

    return measure(q[0], q[1])

if __name__ == "__main__":
    print("═══ Deutsch-Jozsa: Balanced Oracle ═══\n")
    print(draw(deutsch_jozsa_balanced))
    result = run(deutsch_jozsa_balanced, shots=1024, seed=42)
    print(result.summary())
    answer = "BALANCED" if result.most_frequent != "00" else "CONSTANT"
    print(f"\n→ Function: {answer}\n")

    print("═══ Deutsch-Jozsa: Constant Oracle ═══\n")
    print(draw(deutsch_jozsa_constant))
    result = run(deutsch_jozsa_constant, shots=1024, seed=42)
    print(result.summary())
    answer = "BALANCED" if result.most_frequent != "00" else "CONSTANT"
    print(f"\n→ Function: {answer}")
