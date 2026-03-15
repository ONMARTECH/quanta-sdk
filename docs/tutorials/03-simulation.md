# Simulation and Noise

> Tested with: Quanta SDK v0.8.1

## What You'll Learn

Use three simulators, add realistic noise, and measure simulation fidelity.

## Prerequisites

- [02 — Gates and Circuits](02-gates-and-circuits.md)

## Three Simulators

| Simulator | Max Qubits | Use Case |
|-----------|-----------|----------|
| **Statevector** | 27 | Default — full quantum state |
| **Density Matrix** | 13 | Mixed states + noise channels |
| **Pauli Frame** | 1,000+ | Clifford-only circuits |

### Statevector (Default)

Every `run()` call uses the statevector simulator automatically:

```python
from quanta import circuit, H, CX, measure, run

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell, shots=1024)
print(result.statevector[:4])  # Complex amplitudes
```

### Density Matrix

For mixed states and noise analysis:

```python
from quanta.simulator.density_matrix import DensityMatrixSimulator
import numpy as np

sim = DensityMatrixSimulator(2)
sim.apply("H", (0,))
sim.apply("CX", (0, 1))

# Access state and purity
purity = sim.purity
print(f"Purity: {purity:.2f}")  # 1.0 = pure state
probs = sim.probabilities()
print(f"Probabilities: {probs.round(2)}")
```

## Adding Noise

Real quantum computers have errors. Simulate them:

```python
from quanta import circuit, H, CX, measure, run
from quanta.simulator.noise import NoiseModel, Depolarizing

# Create noise model: 1% error per gate
noise = NoiseModel().add(Depolarizing(probability=0.01))

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

# Run with noise
noisy_result = run(bell, shots=1024, noise=noise)
print(noisy_result.summary())
# Now you'll see small amounts of |01⟩ and |10⟩ — these are errors
```

### All 7 Noise Channels

```python
from quanta import circuit, H, CX, measure, run
from quanta.simulator.noise import (
    NoiseModel,
    Depolarizing,      # Random Pauli error
    BitFlip,           # X error with probability
    PhaseFlip,         # Z error with probability
    AmplitudeDamping,  # Energy relaxation (T1)
    ReadoutError,      # Measurement errors
)

# Realistic IBM-like noise
ibm_noise = (NoiseModel()
    .add(Depolarizing(probability=0.001))
    .add(AmplitudeDamping(gamma=0.0005))
    .add(ReadoutError(p0_to_1=0.015, p1_to_0=0.015))
)

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell, shots=4096, noise=ibm_noise)
print(result.summary())
```

### ReadoutError — Post-Measurement Noise

ReadoutError is special — it flips measurement bits *after* the quantum computation:

```python
from quanta import circuit, H, CX, measure, run
from quanta.simulator.noise import NoiseModel, ReadoutError

# p0_to_1 = probability of reading 1 when state is 0
# p1_to_0 = probability of reading 0 when state is 1
noise = NoiseModel().add(ReadoutError(p0_to_1=0.05, p1_to_0=0.05))

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell, shots=10000, noise=noise)
print(result.summary())
# Even a perfect Bell state now shows some |01⟩ and |10⟩
```

## Try It Yourself

1. Increase depolarizing noise to `probability=0.1` — what happens?
2. Add `ReadoutError(p0_to_1=0.1, p1_to_0=0.01)` — which direction is more error-prone?
3. Build a 5-qubit GHZ state — does noise affect it more than 2 qubits?

## What's Next

→ [04 — Algorithms](04-algorithms.md): Grover, QAOA, VQE, Shor — one function call each
