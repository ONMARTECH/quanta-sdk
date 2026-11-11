# Noise and Quantum Error Correction

> Tested with: Quanta SDK v1.3.0

## What You'll Learn

Understand why quantum errors happen and how stabilizer and qLDPC error correction codes protect computations.

## Prerequisites

- [03 — Simulation](03-simulation.md)

## Why QEC?

Real quantum hardware has ~0.1–1% gate error and ~1–2% readout error. Without error correction, a 100-gate circuit has a ~60% chance of at least one physical error. QEC codes detect error syndromes without collapsing logical information and actively correct them.

## Available Codes

| Code | Class | Notation | Physical Qubits | Logical Qubits | Correctable Errors |
|------|-------|----------|-----------------|----------------|-------------------|
| BitFlip | `BitFlipCode` | `[[3,1,3]]` | 3 | 1 | 1 bit-flip ($X$) |
| PhaseFlip | `PhaseFlipCode` | `[[3,1,3]]` | 3 | 1 | 1 phase-flip ($Z$) |
| Steane | `SteaneCode` | `[[7,1,3]]` | 7 | 1 | 1 arbitrary Pauli |
| Shor | `ShorCode` | `[[9,1,3]]` | 9 | 1 | 1 arbitrary Pauli |
| Color ($d=3$) | `ColorCode(distance=3)` | `[[7,1,3]]` | 7 ($d=3$ triangular) | 1 | $\lfloor(d-1)/2\rfloor$ + transversal Clifford |
| Surface | `SurfaceCode(distance=d)` | `[[d²,1,d]]` | $d^2$ data qubits | 1 | $\lfloor(d-1)/2\rfloor$ |
| Gross qLDPC | `BivariateBicycleCode` | `[[144,12,12]]` | 144 | 12 | High-rate BP-OSD decoding |

## BitFlip & Shor Codes — Stabilizer Basics

```python
from quanta.qec.codes import BitFlipCode, ShorCode

bitflip = BitFlipCode()
print(f"Code: {bitflip.info}")
# Code: BitFlip [[3,1,3]]

enc = bitflip.encode()
print(f"BitFlip encoding circuit: {enc.num_qubits} qubits")

shor = ShorCode()
print(f"Code: {shor.info}")
# Code: Shor [[9,1,3]]
```

## Steane, Color & Bivariate Bicycle Codes

```python
from quanta.qec.codes import SteaneCode
from quanta.qec.color_code import ColorCode
from quanta.qec.qldpc import BivariateBicycleCode

steane = SteaneCode()
print(f"Code: {steane.info}")
# Code: Steane [[7,1,3]]

enc = steane.encode()
print(f"Steane encoding: {enc.num_qubits} physical qubits")

color = ColorCode(distance=3)
print(f"Color code (d=3): [[{color.n_data}, {color.n_logical}, {color.distance}]]")
# Color code (d=3): [[7, 1, 3]]

gross = BivariateBicycleCode.gross_144_12_12()
print(f"Gross qLDPC: [[{gross.n}, {gross.k}, {gross.d}]]")
# Gross qLDPC: [[144, 12, 12]]
```

## Understanding Code Parameters

```
[[n, k, d]]
  n = physical qubits needed
  k = logical qubits encoded
  d = code distance (minimum weight of undetectable logical error)

Correctable arbitrary errors = ⌊(d - 1) / 2⌋
```

**Example**: Steane `[[7,1,3]]` uses 7 physical qubits to encode 1 logical qubit with distance 3, correcting $\lfloor(3-1)/2\rfloor = 1$ arbitrary single-qubit Pauli error. Gross `[[144,12,12]]` encodes **12 logical qubits** in 144 physical qubits ($12\times$ higher encoding rate than a $d=12$ surface code).

## QEC Decoders

| Decoder | Class | Complexity | Best For |
|---------|-------|-----------|----------|
| Edmonds Blossom MWPM | `MWPMDecoder` | $O(V^3)$ worst-case | Optimal surface-code graph matching |
| Union-Find | `UnionFindDecoder` | $O(n \cdot \alpha(n))$ | Fast real-time surface-code decoding |
| BP-OSD | `BPOSDDecoder` | Belief propagation + OSD | High-rate qLDPC codes (`BivariateBicycleCode`) |

## Noise + QEC Motivation

Observe how physical bit-flip noise degrades an unencoded Bell state:

```python
from quanta import circuit, H, CX, measure, run
from quanta.simulator.noise import NoiseModel, BitFlip

# Without QEC: 5% bit-flip noise corrupts entangled parity
noise = NoiseModel().add(BitFlip(probability=0.05))

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell, shots=4096, noise=noise, seed=42)
p_error = 1 - (result.counts.get("00", 0) + result.counts.get("11", 0)) / 4096
print(f"Error rate without QEC: {p_error:.1%}")
```

## Try It Yourself

1. Compare `BitFlipCode` vs `PhaseFlipCode` vs `ShorCode` — why does Shor need 9 qubits?
2. Run a Bell state with different noise levels — at what point does fidelity drop below 90%?
3. Explore [FTQC & qLDPC Decoding](../cookbook/ftqc-decoding.md) for active syndrome decoding with `MWPMDecoder` and `BPOSDDecoder`.

## What's Next

→ [08 — MCP AI Integration](08-mcp-ai.md): Use Quanta from Claude, GPT, and other AI assistants
