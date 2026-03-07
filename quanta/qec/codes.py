"""
quanta.qec.codes — Quantum error correction codes.


Desteklenen kodlar:
  - RepetitionCode [[n,1,n]]: N-tekrar kodu

Notasyon: [[n, k, d]]
  n = fiziksel qubit

Example:
    >>> from quanta.qec.codes import BitFlipCode
    >>> code = BitFlipCode()
    >>> encoded = code.encode()     # Kodlama devresi
    >>> print(code.info)
"""

from __future__ import annotations

from dataclasses import dataclass

from quanta.core.circuit import CircuitDefinition, circuit
from quanta.core.gates import CX, H
from quanta.core.measure import measure

# ── Public API ──
__all__ = ["QECCode", "BitFlipCode", "PhaseFlipCode", "SteaneCode"]

@dataclass(frozen=True)
class CodeInfo:
    """Error correction code info.

    Attributes:
        name: Code name.
        d: Kod mesafesi.
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

    def syndrome_measure(self) -> CircuitDefinition:
        raise NotImplementedError

class BitFlipCode(QECCode):
    """[[3,1,1]] Bit-flip kodu.


    Kodlama: |ψ⟩ = α|0⟩ + β|1⟩  →  α|000⟩ + β|111⟩
    """

    @property
    def info(self) -> CodeInfo:
        return CodeInfo("BitFlip", n=3, k=1, d=1)

    def encode(self) -> CircuitDefinition:
        """Bit-flip kodlama devresi.

        q[1], q[2]: Yedek qubit'ler

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

    def syndrome_measure(self) -> CircuitDefinition:
        """Syndrome measurement: error detection with 2 ancilla qubits.

        q[0-2]: Kod qubit'leri
        q[3-4]: Sendrom qubit'leri

          00 → no error
          01 → error on q[2]
          10 → error on q[1]
          11 → error on q[0]
        """
        @circuit(qubits=5)
        def syndrome_bitflip(q):
            # Sendrom 1: q[0] XOR q[1]
            CX(q[0], q[3])
            CX(q[1], q[3])
            # Sendrom 2: q[0] XOR q[2]
            CX(q[0], q[4])
            CX(q[2], q[4])
            return measure(q[3], q[4])
        return syndrome_bitflip

class PhaseFlipCode(QECCode):
    """[[3,1,1]] Faz-flip kodu.


    Kodlama: |ψ⟩ → α|+++⟩ + β|---⟩
    """

    @property
    def info(self) -> CodeInfo:
        return CodeInfo("PhaseFlip", n=3, k=1, d=1)

    def encode(self) -> CircuitDefinition:
        @circuit(qubits=3)
        def encode_phaseflip(q):
            CX(q[0], q[1])
            CX(q[0], q[2])
            H(q[0])
            H(q[1])
            H(q[2])
        return encode_phaseflip

class SteaneCode(QECCode):
    """[[7,1,3]] Steane code.


      X: X₁X₃X₅X₇, X₂X₃X₆X₇, X₄X₅X₆X₇
      Z: Z₁Z₃Z₅Z₇, Z₂Z₃Z₆Z₇, Z₄Z₅Z₆Z₇
    """

    @property
    def info(self) -> CodeInfo:
        return CodeInfo("Steane", n=7, k=1, d=3)

    def encode(self) -> CircuitDefinition:
        """Steane code kodlama devresi.

        q[1-6]: Yedek qubit'ler

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

        13 qubit devre: 7 kod + 6 sendrom.
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
