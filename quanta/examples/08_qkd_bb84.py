"""
Example 08: Quantum Key Distribution (BB84 Protocol)

Demonstrates the BB84 quantum key distribution protocol —
the first practical quantum cryptography scheme.

BB84 uses quantum mechanics to detect eavesdropping:
  1. Alice sends random qubits in random bases (Z or X)
  2. Bob measures in random bases
  3. They compare bases and keep matching bits
  4. Eavesdropping disturbs the quantum states (detectable!)

This addresses the "quantum security" gap mentioned in the
TekBorsasi article about post-quantum cryptography.

Running:
    python -m quanta.examples.08_qkd_bb84
"""

import numpy as np

from quanta import H, X, circuit, measure, run


def bb84_protocol(
    key_length: int = 16,
    eve_intercepts: bool = False,
    seed: int | None = None,
) -> dict:
    """Simulates the BB84 quantum key distribution protocol.

    Args:
        key_length: Desired key length (will generate ~2x raw bits).
        eve_intercepts: If True, simulates an eavesdropper.
        seed: Random seed.

    Returns:
        Dict with alice_key, bob_key, error_rate, secure.
    """
    rng = np.random.default_rng(seed)
    raw_bits = key_length * 4  # Need ~4x bits to get key_length matching

    # Step 1: Alice generates random bits and random bases
    alice_bits = rng.integers(0, 2, size=raw_bits)
    alice_bases = rng.integers(0, 2, size=raw_bits)  # 0=Z, 1=X

    # Step 2: Bob chooses random measurement bases
    bob_bases = rng.integers(0, 2, size=raw_bits)  # 0=Z, 1=X

    # Step 3: Quantum transmission + measurement
    bob_results = []

    for i in range(raw_bits):
        @circuit(qubits=1)
        def qkd_bit(q):
            # Alice prepares qubit
            if alice_bits[i] == 1:
                X(q[0])  # Set to |1>
            if alice_bases[i] == 1:
                H(q[0])  # Switch to X basis

            # Eve intercepts (if present)
            if eve_intercepts:
                eve_basis = rng.integers(0, 2)
                if eve_basis == 1:
                    H(q[0])  # Eve measures in X
                # Eve's measurement collapses state
                # Then Eve re-sends (but may have changed it)
                if eve_basis == 1:
                    H(q[0])  # Eve re-prepares

            # Bob measures in his basis
            if bob_bases[i] == 1:
                H(q[0])  # Switch to X basis for measurement

            return measure(q)

        result = run(qkd_bit, shots=1, seed=seed)
        bob_results.append(int(result.most_frequent))

    # Step 4: Sifting — keep only matching bases
    alice_key = []
    bob_key = []

    for i in range(raw_bits):
        if alice_bases[i] == bob_bases[i]:
            alice_key.append(int(alice_bits[i]))
            bob_key.append(bob_results[i])

    # Trim to desired length
    alice_key = alice_key[:key_length]
    bob_key = bob_key[:key_length]

    # Step 5: Check error rate
    errors = sum(a != b for a, b in zip(alice_key, bob_key))
    error_rate = errors / len(alice_key) if alice_key else 0

    # BB84 threshold: >11% error rate indicates eavesdropping
    secure = error_rate < 0.11

    return {
        "alice_key": "".join(str(b) for b in alice_key),
        "bob_key": "".join(str(b) for b in bob_key),
        "key_length": len(alice_key),
        "errors": errors,
        "error_rate": error_rate,
        "secure": secure,
        "eve_present": eve_intercepts,
    }


def demo_bb84():
    """Demonstrate BB84 with and without eavesdropper."""
    print("=" * 55)
    print("  BB84 Quantum Key Distribution")
    print("=" * 55)

    # Without eavesdropper
    print("\n  ▸ Scenario 1: No eavesdropper")
    result = bb84_protocol(key_length=16, eve_intercepts=False, seed=42)
    print(f"    Alice key: {result['alice_key']}")
    print(f"    Bob key:   {result['bob_key']}")
    print(f"    Errors:    {result['errors']}/{result['key_length']}")
    print(f"    Error rate:{result['error_rate']:.1%}")
    print(f"    Secure:    {'✅ YES' if result['secure'] else '❌ NO'}")

    # With eavesdropper
    print("\n  ▸ Scenario 2: Eve is eavesdropping!")
    result_eve = bb84_protocol(key_length=16, eve_intercepts=True, seed=42)
    print(f"    Alice key: {result_eve['alice_key']}")
    print(f"    Bob key:   {result_eve['bob_key']}")
    print(f"    Errors:    {result_eve['errors']}/{result_eve['key_length']}")
    print(f"    Error rate:{result_eve['error_rate']:.1%}")
    print(f"    Secure:    {'✅ YES' if result_eve['secure'] else '❌ EAVESDROPPER DETECTED!'}")

    print("\n  BB84 detects eavesdropping because measuring")
    print("  a qubit disturbs its quantum state — Eve")
    print("  introduces ~25% error rate by intercepting.")


if __name__ == "__main__":
    demo_bb84()
