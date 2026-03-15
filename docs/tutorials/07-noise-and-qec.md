# Noise and Quantum Error Correction

> Tested with: Quanta SDK v0.8.1

## What You'll Learn

Understand why quantum errors happen and how error correction codes protect computations.

## Prerequisites

- [03 — Simulation](03-simulation.md)

## Why QEC?

Real quantum hardware has ~0.1-1% gate error and ~1-2% readout error. Without error correction, a 100-gate circuit has ~60% chance of at least one error. QEC codes detect and correct these errors.

## Available Codes

| Code | Notation | Physical Qubits | Logical Qubits | Correctable Errors |
|------|----------|-----------------|----------------|-------------------|
| BitFlip | [[3,1,3]] | 3 | 1 | 1 bit-flip |
| PhaseFlip | [[3,1,3]] | 3 | 1 | 1 phase-flip |
| Steane | [[7,1,3]] | 7 | 1 | 1 arbitrary |
| Surface | [[d²,1,d]] | d² | 1 | ⌊(d-1)/2⌋ |
| Color | transversal | varies | 1 | distance-based |

## BitFlip Code — Simplest Example

```python
from quanta.qec.codes import BitFlipCode

code = BitFlipCode()
print(f"Code: {code.info}")
# [[3,1,3]] BitFlip — 1 errors correctable

# Build the encoding circuit
enc = code.encode()
print(f"Encoding circuit: {enc.num_qubits} qubits")
```

## Steane Code — Full Error Correction

```python
from quanta.qec.codes import SteaneCode

steane = SteaneCode()
print(f"Code: {steane.info}")
# [[7,1,3]] Steane — 1 errors correctable

enc = steane.encode()
print(f"Steane encoding: {enc.num_qubits} physical qubits")
```

## Understanding Code Parameters

```
[[n, k, d]]
  n = physical qubits needed
  k = logical qubits encoded
  d = code distance (minimum weight of undetectable error)
  
Correctable errors = ⌊(d-1)/2⌋
```

**Example**: Steane [[7,1,3]] uses 7 physical qubits to encode 1 logical qubit with distance 3, correcting 1 arbitrary error.

## QEC Decoders

| Decoder | Complexity | Best For |
|---------|-----------|----------|
| MWPM | O(n³) | High accuracy |
| Union-Find | O(n·α(n)) | Real-time decoding |

## Noise + QEC Pipeline

The full pipeline: encode → apply gates → inject noise → decode → measure:

```python
from quanta import circuit, H, CX, measure, run
from quanta.simulator.noise import NoiseModel, BitFlip

# Without QEC: noise corrupts the result
noise = NoiseModel().add(BitFlip(probability=0.05))

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

# Noisy result — observe error states
result = run(bell, shots=4096, noise=noise)
p_error = 1 - (result.counts.get("00", 0) + result.counts.get("11", 0)) / 4096
print(f"Error rate without QEC: {p_error:.1%}")
```

## Try It Yourself

1. Compare `BitFlipCode` vs `PhaseFlipCode` — when would you use each?
2. Run a Bell state with different noise levels — at what point does fidelity drop below 90%?
3. Look at `SteaneCode.encode()` — how many gates does it use?

## What's Next

→ [08 — MCP AI Integration](08-mcp-ai.md): Use Quanta from Claude, GPT, and other AI assistants
