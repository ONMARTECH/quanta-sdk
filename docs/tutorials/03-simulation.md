# Simulation and Noise

> Tested with: Quanta SDK v1.3.0

## What You'll Learn

Explore Quanta's simulation backends, inject all 7 physical noise channels, and analyze open-system mixed states.

## Prerequisites

- [02 — Gates and Circuits](02-gates-and-circuits.md)

## Simulator Overview

Quanta routes circuits across specialized simulation backends depending on circuit structure, qubit count, and hardware acceleration:

| Simulator | Scale | Use Case |
|-----------|-------|----------|
| **Statevector** (`statevector`) | Up to ~27 qubits | Default exact pure-state amplitudes (`np.einsum`) |
| **Density Matrix** (`density_matrix`) | Up to ~13 qubits | Exact Kraus mixed states $\rho$ & purity analysis |
| **Pauli Frame** (`pauli_frame`) | 1,000+ qubits | Ultra-fast $O(n)$ Clifford & stabilizer QEC circuits |
| **MPS Tensor Network** (`mps`) | 200+ qubits | Bounded-entanglement 1D/shallow circuits with configurable $\chi_{\max}$ |
| **Sparse Statevector** (`sparse`) | 30–50+ qubits | Circuits with sparse support in the computational basis |
| **Apple Silicon Metal** (`mlx`) | Up to ~28 qubits | Unified-memory Metal GPU statevector engine on Apple Silicon |
| **Lindblad Master Eq.** (`lindblad`) | Open quantum systems | Continuous-time $\dot{\rho} = -i[H, \rho] + \sum_k \mathcal{D}[L_k]\rho$ evolution |

### Statevector (Default)

Every `run()` call uses the statevector simulator automatically unless another method or noise model is specified:

```python
from quanta import circuit, H, CX, measure, run

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell, shots=1024, seed=42)
print(result.statevector[:4])  # Complex amplitudes
```

### Density Matrix

For mixed states and exact purity/entropy analysis:

```python
from quanta.simulator.density_matrix import DensityMatrixSimulator

sim = DensityMatrixSimulator(2)
sim.apply("H", (0,))
sim.apply("CX", (0, 1))

# Access state and purity
purity = sim.purity
print(f"Purity: {purity:.2f}")  # 1.00 = pure state
probs = sim.probabilities()
print(f"Probabilities: {probs.round(2)}")
```

## Adding Noise

Real quantum computers suffer from decoherence, gate imperfections, crosstalk, and readout errors. Simulate them with `NoiseModel`:

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
noisy_result = run(bell, shots=1024, noise=noise, seed=42)
print(noisy_result.summary())
# Now you'll see small amounts of |01⟩ and |10⟩ — these are errors
```

### All 7 Noise Channels

Quanta provides **7 composable physical noise channels** (`Depolarizing`, `BitFlip`, `PhaseFlip`, `AmplitudeDamping`, `T2Relaxation`, `Crosstalk`, and `ReadoutError`):

```python
from quanta import circuit, H, CX, measure, run
from quanta.simulator.noise import (
    NoiseModel,
    Depolarizing,      # 1. Symmetric Pauli (X, Y, Z) error
    BitFlip,           # 2. Pauli-X bit-flip channel
    PhaseFlip,         # 3. Pauli-Z phase-flip channel
    AmplitudeDamping,  # 4. Energy relaxation (T1 decay |1⟩ → |0⟩)
    T2Relaxation,      # 5. Pure dephasing (T2 phase relaxation)
    Crosstalk,         # 6. Neighbour-qubit ZZ parasitic coupling
    ReadoutError,      # 7. Asymmetric classical measurement bit-flip
)

# Realistic superconducting processor noise profile combining all 7 channels
hardware_noise = (
    NoiseModel()
    .add(Depolarizing(probability=0.001))
    .add(BitFlip(probability=0.0005))
    .add(PhaseFlip(probability=0.0005))
    .add(AmplitudeDamping(gamma=0.0008))
    .add(T2Relaxation(gamma=0.0012))
    .add(Crosstalk(probability=0.002))
    .add(ReadoutError(p0_to_1=0.015, p1_to_0=0.02))
)

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell, shots=4096, noise=hardware_noise, seed=42)
print(result.summary())
```

### ReadoutError — Post-Measurement Noise

`ReadoutError` flips classical measurement bits *after* state collapse according to asymmetric confusion probabilities:

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

result = run(bell, shots=10000, noise=noise, seed=42)
print(result.summary())
# Even a perfect Bell state now shows ~9.5% |01⟩ and |10⟩ readout flips
```

## Try It Yourself

1. Increase depolarizing noise to `probability=0.1` — what happens?
2. Add `ReadoutError(p0_to_1=0.1, p1_to_0=0.01)` — which direction is more error-prone?
3. Build a 5-qubit GHZ state — does noise affect it more than 2 qubits?

## What's Next

→ [04 — Algorithms](04-algorithms.md): Grover, QAOA, VQE, Shor — one function call each
