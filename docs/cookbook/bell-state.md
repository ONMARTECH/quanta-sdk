# Bell State — Quick Recipe

> Create and measure the simplest entangled state in quantum computing.

## What It Does

A Bell state is a maximally entangled two-qubit state. When measured, both qubits
always collapse to the **same value** — either both `|00⟩` or both `|11⟩`,
each with 50% probability. This is the foundation of quantum teleportation,
superdense coding, and many quantum algorithms.

## The Circuit

```
q0: ─ H ─ ● ─ M ─
          │
q1: ──── CX ─ M ─
```

1. **H** on qubit 0 → creates superposition `(|0⟩ + |1⟩)/√2`
2. **CX** (CNOT) → entangles qubit 1 with qubit 0
3. **Measure** → collapses both qubits simultaneously

## Code

```python
from quanta import circuit, H, CX, measure, run

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell, shots=1024)
print(result.summary())
```

## Expected Output

```
Measurement Results (1024 shots):
  |00⟩ : ████████████████████ 512 (50.0%)
  |11⟩ : ████████████████████ 512 (50.0%)
```

> **Key insight:** You'll never see `|01⟩` or `|10⟩` — that's entanglement!
> The qubits are correlated, not independent.

## Try Next

- Add noise: `run(bell, shots=1024, noise=NoiseModel().add(Depolarizing(p=0.01)))`
- Try the GHZ state: extend to 3+ qubits with more CX gates
- Visualize: `from quanta.visualize_svg import draw_svg; draw_svg(bell, "bell.html")`

## See Also

- [Tutorial 01 — Getting Started](../tutorials/01-getting-started.md)
- [Tutorial 02 — Gates & Circuits](../tutorials/02-gates-and-circuits.md)
