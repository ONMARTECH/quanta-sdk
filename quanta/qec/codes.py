"""
quanta.qec.codes -- Quantum Error Correction code definitions.

Implements classic QEC codes with complete stabilizer syndrome extraction
and single-error Pauli recovery:
  - BitFlipCode   [[3, 1, 1]] -- Corrects 1 X (bit-flip) error
  - PhaseFlipCode [[3, 1, 1]] -- Corrects 1 Z (phase-flip) error
  - SteaneCode    [[7, 1, 3]] -- CSS Hamming-7 code: corrects arbitrary single X/Y/Z
  - ShorCode      [[9, 1, 3]] -- Degenerate 9-qubit code: 6 ZZ + 2 XXXXXX checks
"""

from __future__ import annotations

from dataclasses import dataclass

from quanta.core.circuit import CircuitDefinition, circuit
from quanta.core.gates import CCX, CX, H, X, Y, Z
from quanta.core.measure import measure

__all__ = [
    "CodeInfo",
    "CodeParameters",
    "QECCode",
    "BitFlipCode",
    "PhaseFlipCode",
    "SteaneCode",
    "ShorCode",
    "correct_error",
]


@dataclass(frozen=True)
class CodeInfo:
    """[[n, k, d]] metadata of a QEC code."""

    name: str
    n: int
    k: int
    d: int

    @property
    def correctable_errors(self) -> int:
        """Maximum guaranteed correctable errors: floor((d-1)/2)."""
        return (self.d - 1) // 2

    def __repr__(self) -> str:
        return f"{self.name} [[{self.n},{self.k},{self.d}]]"


@dataclass(frozen=True)
class CodeParameters:
    """[[n, k, d]] parameters of a QEC code."""

    n: int
    k: int
    d: int

    @property
    def correctable_errors(self) -> int:
        return (self.d - 1) // 2

    def __repr__(self) -> str:
        return f"[[{self.n}, {self.k}, {self.d}]]"


class QECCode:
    """Base class for quantum error correction codes."""

    @property
    def info(self) -> CodeInfo:
        raise NotImplementedError

    def encode(self) -> CircuitDefinition:
        raise NotImplementedError

    def decode(self) -> CircuitDefinition:
        raise NotImplementedError

    def syndrome_measure(self) -> CircuitDefinition:
        raise NotImplementedError

    def lookup_table(self) -> dict[str, str]:
        raise NotImplementedError


def correct_error(code: QECCode, syndrome: str) -> str:
    """Looks up error diagnosis string from a QECCode syndrome table."""
    return code.lookup_table().get(syndrome, "No error detected")


class BitFlipCode(QECCode):
    """[[3, 1, 3]] 3-qubit bit-flip code (Z0Z1, Z0Z2 checks; bit-flip distance 3)."""

    params = CodeParameters(n=3, k=1, d=3)

    @property
    def info(self) -> CodeInfo:
        return CodeInfo("BitFlip", n=3, k=1, d=3)

    def encode(self) -> CircuitDefinition:
        @circuit(qubits=3)
        def bitflip_encode(q):
            CX(q[0], q[1])
            CX(q[0], q[2])
            return measure(q)

        return bitflip_encode

    def syndrome_measure(self) -> CircuitDefinition:
        @circuit(qubits=5)
        def bitflip_syndrome(q):
            CX(q[0], q[3])
            CX(q[1], q[3])
            CX(q[0], q[4])
            CX(q[2], q[4])
            return measure(q[3], q[4])

        return bitflip_syndrome

    def decode(self) -> CircuitDefinition:
        @circuit(qubits=3)
        def bitflip_decode(q):
            CX(q[0], q[1])
            CX(q[0], q[2])
            CCX(q[1], q[2], q[0])
            return measure(q[0])

        return bitflip_decode

    def lookup_table(self) -> dict[str, str]:
        return {
            "00": "No error detected",
            "11": "X on qubit 0",
            "10": "X on qubit 1",
            "01": "X on qubit 2",
        }


class PhaseFlipCode(QECCode):
    """[[3, 1, 3]] 3-qubit phase-flip code (X0X1, X0X2 checks; phase-flip distance 3)."""

    params = CodeParameters(n=3, k=1, d=3)

    @property
    def info(self) -> CodeInfo:
        return CodeInfo("PhaseFlip", n=3, k=1, d=3)

    def encode(self) -> CircuitDefinition:
        @circuit(qubits=3)
        def phaseflip_encode(q):
            CX(q[0], q[1])
            CX(q[0], q[2])
            H(q[0])
            H(q[1])
            H(q[2])
            return measure(q)

        return phaseflip_encode

    def syndrome_measure(self) -> CircuitDefinition:
        """Measures X0X1 on q[3] and X0X2 on q[4]."""

        @circuit(qubits=5)
        def phaseflip_syndrome(q):
            H(q[3])
            H(q[4])
            CX(q[3], q[0])
            CX(q[3], q[1])
            CX(q[4], q[0])
            CX(q[4], q[2])
            H(q[3])
            H(q[4])
            return measure(q[3], q[4])

        return phaseflip_syndrome

    def decode(self) -> CircuitDefinition:
        @circuit(qubits=3)
        def phaseflip_decode(q):
            H(q[0])
            H(q[1])
            H(q[2])
            CX(q[0], q[1])
            CX(q[0], q[2])
            CCX(q[1], q[2], q[0])
            return measure(q[0])

        return phaseflip_decode

    def lookup_table(self) -> dict[str, str]:
        return {
            "00": "No error detected",
            "11": "Z on qubit 0",
            "10": "Z on qubit 1",
            "01": "Z on qubit 2",
        }


class SteaneCode(QECCode):
    """[[7, 1, 3]] Steane CSS code based on the [7, 4, 3] Hamming code H_7.

    Corrects ALL 21 single-qubit Pauli errors (X, Y, Z on q[0]..q[6]).
    """

    params = CodeParameters(n=7, k=1, d=3)

    @property
    def info(self) -> CodeInfo:
        return CodeInfo("Steane", n=7, k=1, d=3)

    def encode(self) -> CircuitDefinition:
        @circuit(qubits=7)
        def steane_encode(q):
            CX(q[0], q[1])
            CX(q[0], q[2])
            H(q[3])
            H(q[4])
            H(q[5])
            CX(q[3], q[0])
            CX(q[3], q[1])
            CX(q[3], q[6])
            CX(q[4], q[0])
            CX(q[4], q[2])
            CX(q[4], q[6])
            CX(q[5], q[1])
            CX(q[5], q[2])
            CX(q[5], q[6])
            return measure(q)

        return steane_encode

    def decode(self) -> CircuitDefinition:
        @circuit(qubits=7)
        def steane_decode(q):
            CX(q[5], q[6])
            CX(q[5], q[2])
            CX(q[5], q[1])
            CX(q[4], q[6])
            CX(q[4], q[2])
            CX(q[4], q[0])
            CX(q[3], q[6])
            CX(q[3], q[1])
            CX(q[3], q[0])
            H(q[3])
            H(q[4])
            H(q[5])
            CX(q[0], q[2])
            CX(q[0], q[1])
            return measure(q[0])

        return steane_decode

    def syndrome_measure(self) -> CircuitDefinition:
        @circuit(qubits=13)
        def steane_syndrome(q):
            for idx in [0, 2, 4, 6]:
                CX(q[idx], q[7])
            for idx in [1, 2, 5, 6]:
                CX(q[idx], q[8])
            for idx in [3, 4, 5, 6]:
                CX(q[idx], q[9])

            H(q[10])
            H(q[11])
            H(q[12])
            for idx in [0, 2, 4, 6]:
                CX(q[10], q[idx])
            for idx in [1, 2, 5, 6]:
                CX(q[11], q[idx])
            for idx in [3, 4, 5, 6]:
                CX(q[12], q[idx])
            H(q[10])
            H(q[11])
            H(q[12])

            return measure(q[7], q[8], q[9], q[10], q[11], q[12])

        return steane_syndrome

    def decode_syndrome(self, syndrome: str) -> dict[int, str]:
        """Maps 6-bit (or 3-bit bit-flip) Steane syndrome to {qubit: Pauli} recovery."""
        s = syndrome.strip()
        if len(s) == 3:
            loc = int(s[0]) + 2 * int(s[1]) + 4 * int(s[2])
            return {} if loc == 0 else {loc - 1: "X"}
        if len(s) != 6 or any(ch not in "01" for ch in s):
            raise ValueError(f"Steane syndrome must be 6 bits (or 3 bits), got '{syndrome}'")

        x_loc = int(s[0]) + 2 * int(s[1]) + 4 * int(s[2])
        z_loc = int(s[3]) + 2 * int(s[4]) + 4 * int(s[5])

        if x_loc == 0 and z_loc == 0:
            return {}
        if x_loc > 0 and z_loc == 0:
            return {x_loc - 1: "X"}
        if x_loc == 0 and z_loc > 0:
            return {z_loc - 1: "Z"}
        if x_loc == z_loc:
            return {x_loc - 1: "Y"}
        return {x_loc - 1: "X", z_loc - 1: "Z"}

    def lookup_table(self) -> dict[str, str]:
        table: dict[str, str] = {
            "000": "No error detected",
            "000000": "No error detected",
        }
        for loc in range(1, 8):
            b_str = f"{loc & 1}{(loc >> 1) & 1}{(loc >> 2) & 1}"
            table[b_str] = f"X on qubit {loc - 1}"

        for x_loc in range(8):
            sx = f"{x_loc & 1}{(x_loc >> 1) & 1}{(x_loc >> 2) & 1}"
            for z_loc in range(8):
                sz = f"{z_loc & 1}{(z_loc >> 1) & 1}{(z_loc >> 2) & 1}"
                key = sx + sz
                if x_loc == 0 and z_loc == 0:
                    table[key] = "No error detected"
                elif x_loc > 0 and z_loc == 0:
                    table[key] = f"X on qubit {x_loc - 1}"
                elif x_loc == 0 and z_loc > 0:
                    table[key] = f"Z on qubit {z_loc - 1}"
                elif x_loc == z_loc:
                    table[key] = f"Y on qubit {x_loc - 1}"
                else:
                    table[key] = f"X on qubit {x_loc - 1} and Z on qubit {z_loc - 1}"
        return table

    def verify_single_pauli_recovery(self, pauli: str, qubit: int) -> float:
        """End-to-end statevector verification on Steane [[7, 1, 3]] across all 21 Paulis."""
        import numpy as np

        from quanta.simulator.statevector import StateVectorSimulator

        sim = StateVectorSimulator(13)
        sim.apply("RY", (0,), (1.1,))
        sim.apply("RZ", (0,), (0.7,))

        ref = StateVectorSimulator(1)
        ref.apply("RY", (0,), (1.1,))
        ref.apply("RZ", (0,), (0.7,))
        psi_target = ref.state.copy()

        # Encode
        sim.apply("CX", (0, 1))
        sim.apply("CX", (0, 2))
        for hq in (3, 4, 5):
            sim.apply("H", (hq,))
        for tgt in (0, 1, 6):
            sim.apply("CX", (3, tgt))
        for tgt in (0, 2, 6):
            sim.apply("CX", (4, tgt))
        for tgt in (1, 2, 6):
            sim.apply("CX", (5, tgt))

        # Inject single-qubit Pauli error
        p_up = pauli.upper()
        if p_up in ("X", "Y", "Z"):
            sim.apply(p_up, (qubit,))

        # Measure 6 CSS syndromes onto q[7..12]
        for idx in (0, 2, 4, 6):
            sim.apply("CX", (idx, 7))
        for idx in (1, 2, 5, 6):
            sim.apply("CX", (idx, 8))
        for idx in (3, 4, 5, 6):
            sim.apply("CX", (idx, 9))

        for anc in (10, 11, 12):
            sim.apply("H", (anc,))
        for idx in (0, 2, 4, 6):
            sim.apply("CX", (10, idx))
        for idx in (1, 2, 5, 6):
            sim.apply("CX", (11, idx))
        for idx in (3, 4, 5, 6):
            sim.apply("CX", (12, idx))
        for anc in (10, 11, 12):
            sim.apply("H", (anc,))

        probs = sim.probabilities()
        nz_idx = int(np.argmax(probs))
        syndrome_6 = format(nz_idx, "013b")[7:13]

        recovery = self.decode_syndrome(syndrome_6)
        for q_rec, gate_name in recovery.items():
            sim.apply(gate_name, (q_rec,))

        # Unencode
        for tgt in (6, 2, 1):
            sim.apply("CX", (5, tgt))
        for tgt in (6, 2, 0):
            sim.apply("CX", (4, tgt))
        for tgt in (6, 1, 0):
            sim.apply("CX", (3, tgt))
        for hq in (3, 4, 5):
            sim.apply("H", (hq,))
        sim.apply("CX", (0, 2))
        sim.apply("CX", (0, 1))

        sv_2d = sim.state.reshape(2, 1 << 12)
        rho_0 = sv_2d @ sv_2d.conj().T
        return float(np.real(psi_target.conj() @ rho_0 @ psi_target))


class ShorCode(QECCode):
    """[[9, 1, 3]] Shor 9-qubit degenerate stabilizer code (6 ZZ + 2 XXXXXX checks)."""

    params = CodeParameters(n=9, k=1, d=3)

    @property
    def info(self) -> CodeInfo:
        return CodeInfo("Shor", n=9, k=1, d=3)

    def encode(self) -> CircuitDefinition:
        @circuit(qubits=9)
        def shor_encode(q):
            CX(q[0], q[3])
            CX(q[0], q[6])
            H(q[0])
            H(q[3])
            H(q[6])
            CX(q[0], q[1])
            CX(q[0], q[2])
            CX(q[3], q[4])
            CX(q[3], q[5])
            CX(q[6], q[7])
            CX(q[6], q[8])
            return measure(q)

        return shor_encode

    def syndrome_measure(self) -> CircuitDefinition:
        """17-qubit circuit (9 data + 8 ancillas) extracting full 8-bit syndrome."""

        @circuit(qubits=17)
        def shor_syndrome(q):
            CX(q[0], q[9])
            CX(q[1], q[9])
            CX(q[0], q[10])
            CX(q[2], q[10])
            CX(q[3], q[11])
            CX(q[4], q[11])
            CX(q[3], q[12])
            CX(q[5], q[12])
            CX(q[6], q[13])
            CX(q[7], q[13])
            CX(q[6], q[14])
            CX(q[8], q[14])

            H(q[15])
            H(q[16])
            for idx in (0, 1, 2, 3, 4, 5):
                CX(q[15], q[idx])
            for idx in (0, 1, 2, 6, 7, 8):
                CX(q[16], q[idx])
            H(q[15])
            H(q[16])

            return measure(q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16])

        return shor_syndrome

    def decode(self) -> CircuitDefinition:
        @circuit(qubits=9)
        def shor_decode(q):
            CX(q[0], q[1])
            CX(q[0], q[2])
            CCX(q[1], q[2], q[0])

            CX(q[3], q[4])
            CX(q[3], q[5])
            CCX(q[4], q[5], q[3])

            CX(q[6], q[7])
            CX(q[6], q[8])
            CCX(q[7], q[8], q[6])

            H(q[0])
            H(q[3])
            H(q[6])

            CX(q[0], q[3])
            CX(q[0], q[6])
            CCX(q[3], q[6], q[0])
            return measure(q[0])

        return shor_decode

    def decode_syndrome(self, syndrome: str) -> dict[int, str]:
        """Decodes an 8-bit Shor stabilizer syndrome into {qubit: Pauli} recovery."""
        s = syndrome.strip()
        if len(s) != 8 or any(ch not in "01" for ch in s):
            raise ValueError(f"Shor syndrome must be an 8-bit binary string, got '{syndrome}'")

        pair_to_offset = {"11": 0, "10": 1, "01": 2}
        x_qubits: list[int] = []
        for blk in range(3):
            bits = s[2 * blk : 2 * blk + 2]
            if bits in pair_to_offset:
                x_qubits.append(3 * blk + pair_to_offset[bits])

        phase_bits = s[6:8]
        phase_to_block = {"11": 0, "10": 1, "01": 2}
        z_block = phase_to_block.get(phase_bits)

        recovery: dict[int, str] = {}
        for q_idx in x_qubits:
            recovery[q_idx] = "X"

        if z_block is not None:
            same_block_q = [q_idx for q_idx in x_qubits if q_idx // 3 == z_block]
            if same_block_q:
                recovery[same_block_q[0]] = "Y"
            else:
                recovery[3 * z_block] = "Z"

        return recovery

    def lookup_table(self) -> dict[str, str]:
        """Complete 8-bit stabilizer lookup table (+ legacy 4-bit keys)."""
        table: dict[str, str] = {
            "0000": "No error detected",
            "0001": "X on qubit 8",
            "0010": "X on qubit 7",
            "0011": "X on qubit 6",
            "0100": "X on qubit 5",
            "1000": "X on qubit 4",
            "1100": "X on qubit 3",
            "00000000": "No error detected",
        }

        pair_for_offset = {0: "11", 1: "10", 2: "01"}
        phase_for_block = {0: "11", 1: "10", 2: "01"}

        for q_idx in range(9):
            blk, off = divmod(q_idx, 3)
            zz_bits = ["00", "00", "00"]
            zz_bits[blk] = pair_for_offset[off]
            zz_prefix = "".join(zz_bits)

            table[zz_prefix + "00"] = f"X on qubit {q_idx}"
            table[zz_prefix + phase_for_block[blk]] = f"Y on qubit {q_idx}"

        for blk, p_bits in phase_for_block.items():
            q0 = 3 * blk
            table["000000" + p_bits] = (
                f"Z in block {blk} (correct with Z on qubit {q0})"
            )

        return table

    def verify_single_pauli_recovery(self, pauli: str, qubit: int) -> float:
        """End-to-end statevector verification on Shor [[9, 1, 3]] across all 27 Paulis."""
        import numpy as np

        from quanta.simulator.statevector import StateVectorSimulator

        sim = StateVectorSimulator(17)
        sim.apply("RY", (0,), (1.2,))
        sim.apply("RZ", (0,), (0.8,))

        ref = StateVectorSimulator(1)
        ref.apply("RY", (0,), (1.2,))
        ref.apply("RZ", (0,), (0.8,))
        psi_target = ref.state.copy()

        sim.apply("CX", (0, 3))
        sim.apply("CX", (0, 6))
        for hq in (0, 3, 6):
            sim.apply("H", (hq,))
        for ctrl in (0, 3, 6):
            sim.apply("CX", (ctrl, ctrl + 1))
            sim.apply("CX", (ctrl, ctrl + 2))

        p_up = pauli.upper()
        if p_up in ("X", "Y", "Z"):
            sim.apply(p_up, (qubit,))

        zz_pairs = [
            (0, 1, 9),
            (0, 2, 10),
            (3, 4, 11),
            (3, 5, 12),
            (6, 7, 13),
            (6, 8, 14),
        ]
        for d1, d2, anc in zz_pairs:
            sim.apply("CX", (d1, anc))
            sim.apply("CX", (d2, anc))

        sim.apply("H", (15,))
        sim.apply("H", (16,))
        for idx in (0, 1, 2, 3, 4, 5):
            sim.apply("CX", (15, idx))
        for idx in (0, 1, 2, 6, 7, 8):
            sim.apply("CX", (16, idx))
        sim.apply("H", (15,))
        sim.apply("H", (16,))

        probs = sim.probabilities()
        nz_idx = int(np.argmax(probs))
        syndrome_8 = format(nz_idx, "017b")[9:17]

        recovery = self.decode_syndrome(syndrome_8)
        for q_rec, gate_name in recovery.items():
            sim.apply(gate_name, (q_rec,))

        for ctrl in (0, 3, 6):
            sim.apply("CX", (ctrl, ctrl + 2))
            sim.apply("CX", (ctrl, ctrl + 1))
        for hq in (0, 3, 6):
            sim.apply("H", (hq,))
        sim.apply("CX", (0, 6))
        sim.apply("CX", (0, 3))

        sv_2d = sim.state.reshape(2, 1 << 16)
        rho_0 = sv_2d @ sv_2d.conj().T
        return float(np.real(psi_target.conj() @ rho_0 @ psi_target))


_ = (X, Y, Z)
