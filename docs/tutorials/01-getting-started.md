# Getting Started with Quanta SDK

> Tested with: Quanta SDK v0.8.1

## What You'll Learn

In 5 minutes you'll install Quanta, build your first quantum circuit, run it, and interpret the results.

## Prerequisites

- Python 3.10+
- No quantum knowledge required

## Step 1 — Install

```bash
pip install quanta-sdk
```

Verify:

```bash
python -c "import quanta; print(f'Quanta {quanta.__version__} ready')"
```

## Step 2 — Your First Circuit (Bell State)

A Bell State creates two qubits that are perfectly correlated — measuring one instantly determines the other.

```python
from quanta import circuit, H, CX, measure, run

@circuit(qubits=2)
def bell_state(q):
    H(q[0])          # Put qubit 0 in superposition
    CX(q[0], q[1])   # Entangle qubit 0 and 1
    return measure(q) # Measure both

result = run(bell_state, shots=1024)
print(result.summary())
```

**Expected output:**

```
  |00⟩ : ████████████████████ ~512 (50.0%)
  |11⟩ : ████████████████████ ~512 (50.0%)
```

You'll see approximately 50% `|00⟩` and 50% `|11⟩` — never `|01⟩` or `|10⟩`. That's entanglement!

## Step 3 — Understanding the Result

```python
from quanta import circuit, H, CX, measure, run

@circuit(qubits=2)
def bell_state(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell_state, shots=1024)

# Access raw counts
print(result.counts)        # {'00': ~512, '11': ~512}

# Most likely outcome
print(result.most_frequent)          # '00' or '11'

# Circuit metadata
print(result.gate_count)     # 2
print(result.depth)          # 2
```

## Step 4 — A 3-Qubit GHZ State

Scale up — create a 3-qubit entangled state:

```python
from quanta import circuit, H, CX, measure, run

@circuit(qubits=3)
def ghz(q):
    H(q[0])
    CX(q[0], q[1])
    CX(q[1], q[2])
    return measure(q)

result = run(ghz, shots=1024)
print(result.summary())
# |000⟩ ≈ 50%, |111⟩ ≈ 50%
```

## Step 5 — Parametric Gates

Gates can take angle parameters:

```python
from quanta import circuit, RY, measure, run
import math

@circuit(qubits=1)
def rotation(q):
    RY(math.pi / 4)(q[0])  # Rotate by π/4 around Y-axis
    return measure(q)

result = run(rotation, shots=1024)
print(result.summary())
# |0⟩ ≈ 85%, |1⟩ ≈ 15%
```

## Try It Yourself

1. Change `RY(math.pi / 4)` to `RY(math.pi / 2)` — what happens to the probabilities?
2. Add a second qubit and entangle them with CX
3. Try `run(bell_state, shots=10000)` — do the probabilities get closer to 50/50?

## What's Next

→ [02 — Gates and Circuits](02-gates-and-circuits.md): Learn all 31 quantum gates
