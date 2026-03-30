"""
quanta.qec.codes — Quantum error correction codes.


Supported codes:
  - RepetitionCode [[n,1,n]]: N-repetition code

Notation: [[n, k, d]]
  n = physical qubits

Example:
    >>> from quanta.qec.codes import BitFlipCode
    >>> code = BitFlipCode()
    >>> encoded = code.encode()     # Encoding circuit
    >>> print(code.info)
"""

from __future__ import annotations

from dataclasses import dataclass

from quanta.core.circuit import CircuitDefinition, circuit
from quanta.core.gates import CX, H
from quanta.core.measure import measure

# ── Public API ──
__all__ = ["QECCode", "BitFlipCode", "PhaseFlipCode", "SteaneCode", "ShorCode",
           "correct_error"]

@dataclass(frozen=True)
class CodeInfo:
    """Error correction code info.

    Attributes:
        name: Code name.
        d: Code distance.
    """

    name: str
    n: int  # physical qubits
    k: int  # logical qubits
    d: int  # distance

    @property
    def correctable_errors(self) -> int:
        """Number of correctable errors: ⌊(d-1)/2⌋."""
        return (self.d - 1) // 2

    def __repr__(self) -> str:
        return (
            f"[[{self.n},{self.k},{self.d}]] {self.name} — "
            f"{self.correctable_errors} errors correctable"
        )

class QECCode:
    """Base class for error correction codes."""

    @property
    def info(self) -> CodeInfo:
        raise NotImplementedError

    def encode(self) -> CircuitDefinition:
        raise NotImplementedError

    def decode(self) -> CircuitDefinition:
        """Decoding circuit (inverse of encode)."""
        raise NotImplementedError

    def syndrome_measure(self) -> CircuitDefinition:
        raise NotImplementedError

    def lookup_table(self) -> dict[str, str]:
        """Maps syndrome bitstring to error location.

        Returns:
            Dict mapping syndrome → correction description.
        """
        raise NotImplementedError


def correct_error(code: QECCode, syndrome: str) -> str:
    """Given a syndrome measurement result, returns correction action.

    Args:
        code: The QEC code used.
        syndrome: Measured syndrome bitstring.

    Returns:
        Description of correction to apply.

    Example:
        >>> code = BitFlipCode()
        >>> correct_error(code, "11")
        'Apply X on qubit 0'
    """
    table = code.lookup_table()
    return table.get(syndrome, "No error detected")


class BitFlipCode(QECCode):
    """[[3,1,3]] Bit-flip repetition code.

    Encodes 1 logical qubit into 3 physical qubits.
    Distance d=3 → can correct ⌊(3-1)/2⌋ = 1 bit-flip error.

    Encoding: |ψ⟩ = α|0⟩ + β|1⟩  →  α|000⟩ + β|111⟩
    """

    @property
    def info(self) -> CodeInfo:
        return CodeInfo("BitFlip", n=3, k=1, d=3)

    def encode(self) -> CircuitDefinition:
        """Bit-flip encoding circuit.

        q[1], q[2]: Ancilla (redundancy) qubits

        Circuit: q[0]──●──●──
                     │  │
               q[1]──X──│──
                        │
               q[2]─────X──
        """
        @circuit(qubits=3)
        def encode_bitflip(q):
            CX(q[0], q[1])
            CX(q[0], q[2])
        return encode_bitflip

    def decode(self) -> CircuitDefinition:
        """Bit-flip decoding circuit (inverse of encode)."""
        @circuit(qubits=3)
        def decode_bitflip(q):
            CX(q[0], q[2])
            CX(q[0], q[1])
        return decode_bitflip

    def syndrome_measure(self) -> CircuitDefinition:
        """Syndrome measurement: error detection with 2 ancilla qubits.

        q[0-2]: Code qubits
        q[3-4]: Syndrome qubits

          00 → no error
          01 → error on q[2]
          10 → error on q[1]
          11 → error on q[0]
        """
        @circuit(qubits=5)
        def syndrome_bitflip(q):
            # Syndrome 1: q[0] XOR q[1]
            CX(q[0], q[3])
            CX(q[1], q[3])
            # Syndrome 2: q[0] XOR q[2]
            CX(q[0], q[4])
            CX(q[2], q[4])
            return measure(q[3], q[4])
        return syndrome_bitflip

    def lookup_table(self) -> dict[str, str]:
        return {
            "00": "No error detected",
            "01": "Apply X on qubit 2",
            "10": "Apply X on qubit 1",
            "11": "Apply X on qubit 0",
        }

class PhaseFlipCode(QECCode):
    """[[3,1,3]] Phase-flip repetition code.

    Encodes 1 logical qubit into 3 physical qubits in the Hadamard basis.
    Distance d=3 → can correct 1 phase-flip error.

    Encoding: |ψ⟩ → α|+++⟩ + β|---⟩
    """

    @property
    def info(self) -> CodeInfo:
        return CodeInfo("PhaseFlip", n=3, k=1, d=3)

    def encode(self) -> CircuitDefinition:
        @circuit(qubits=3)
        def encode_phaseflip(q):
            CX(q[0], q[1])
            CX(q[0], q[2])
            H(q[0])
            H(q[1])
            H(q[2])
        return encode_phaseflip

    def decode(self) -> CircuitDefinition:
        """Phase-flip decoding circuit."""
        @circuit(qubits=3)
        def decode_phaseflip(q):
            H(q[0])
            H(q[1])
            H(q[2])
            CX(q[0], q[2])
            CX(q[0], q[1])
        return decode_phaseflip

    def lookup_table(self) -> dict[str, str]:
        return {
            "00": "No error detected",
            "01": "Apply Z on qubit 2",
            "10": "Apply Z on qubit 1",
            "11": "Apply Z on qubit 0",
        }

class SteaneCode(QECCode):
    """[[7,1,3]] Steane code.


      X: X₁X₃X₅X₇, X₂X₃X₆X₇, X₄X₅X₆X₇
      Z: Z₁Z₃Z₅Z₇, Z₂Z₃Z₆Z₇, Z₄Z₅Z₆Z₇
    """

    @property
    def info(self) -> CodeInfo:
        return CodeInfo("Steane", n=7, k=1, d=3)

    def encode(self) -> CircuitDefinition:
        """Steane code encoding circuit.

        q[1-6]: Ancilla (redundancy) qubits

        """
        @circuit(qubits=7)
        def encode_steane(q):
            H(q[3])
            H(q[4])
            H(q[5])

            CX(q[0], q[3])
            CX(q[0], q[4])
            CX(q[0], q[5])

            CX(q[3], q[1])
            CX(q[3], q[6])
            CX(q[4], q[2])
            CX(q[4], q[6])
            CX(q[5], q[1])
            CX(q[5], q[2])
        return encode_steane

    def syndrome_measure(self) -> CircuitDefinition:
        """Steane syndrome measurement: 6 syndrome qubits (3 X + 3 Z).

        13 qubit circuit: 7 code + 6 syndrome.
        """
        @circuit(qubits=13)
        def syndrome_steane(q):
            for s, targets in [(7, [0, 2, 4, 6]), (8, [1, 2, 5, 6]), (9, [3, 4, 5, 6])]:
                for t in targets:
                    CX(q[t], q[s])

            for s, targets in [(10, [0, 2, 4, 6]), (11, [1, 2, 5, 6]), (12, [3, 4, 5, 6])]:
                H(q[s])
                for t in targets:
                    CX(q[s], q[t])
                H(q[s])

            return measure(q[7], q[8], q[9], q[10], q[11], q[12])
        return syndrome_steane


class ShorCode(QECCode):
    """[[9,1,3]] Shor code.

    First quantum error correction code. Protects against
    arbitrary single-qubit errors (bit-flip + phase-flip).

    Combines 3-qubit bit-flip and 3-qubit phase-flip repetition codes.
    Encoding: |ψ⟩ = α|0⟩ + β|1⟩ →
        α(|000⟩+|111⟩)(|000⟩+|111⟩)(|000⟩+|111⟩)/2√2 +
        β(|000⟩-|111⟩)(|000⟩-|111⟩)(|000⟩-|111⟩)/2√2
    """

    @property
    def info(self) -> CodeInfo:
        return CodeInfo("Shor", n=9, k=1, d=3)

    def encode(self) -> CircuitDefinition:
        """Shor 9-qubit encoding circuit."""
        @circuit(qubits=9)
        def encode_shor(q):
            # Phase-flip encoding (outer): q[0] → q[0], q[3], q[6]
            CX(q[0], q[3])
            CX(q[0], q[6])
            H(q[0])
            H(q[3])
            H(q[6])
            # Bit-flip encoding (inner): each group of 3
            CX(q[0], q[1])
            CX(q[0], q[2])
            CX(q[3], q[4])
            CX(q[3], q[5])
            CX(q[6], q[7])
            CX(q[6], q[8])
        return encode_shor

    def decode(self) -> CircuitDefinition:
        """Shor 9-qubit decoding circuit (inverse of encode)."""
        @circuit(qubits=9)
        def decode_shor(q):
            # Reverse bit-flip
            CX(q[6], q[8])
            CX(q[6], q[7])
            CX(q[3], q[5])
            CX(q[3], q[4])
            CX(q[0], q[2])
            CX(q[0], q[1])
            # Reverse phase-flip
            H(q[6])
            H(q[3])
            H(q[0])
            CX(q[0], q[6])
            CX(q[0], q[3])
        return decode_shor

    def lookup_table(self) -> dict[str, str]:
        return {
            "0000": "No error detected",
            "0001": "Apply X on qubit 8",
            "0010": "Apply X on qubit 7",
            "0011": "Apply X on qubit 6",
            "0100": "Apply X on qubit 5",
            "1000": "Apply X on qubit 2",
        }
