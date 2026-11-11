# Coming from Cirq

> Tested with: Quanta SDK v1.3.0

## Why Switch?

| Feature | Cirq | Quanta |
|---------|------|--------|
| Install size | ~300 MB | ~5 MB (Python + NumPy core) |
| IBM Hardware | ❌ (Google only) | ✅ (direct REST API) |
| Core dependencies | 15+ packages | 1 (`numpy>=1.24`) |
| Learning curve | Moderate (`Moment`, `Circuit`, `Simulator`) | Gentle (3 imports to start) |
| Gate count | 60+ | 34 (IBM Heron + Google + IonQ native) |
| AI Integration | ❌ | ✅ (25 MCP tools for Claude/GPT/Gemini) |
| QEC | Basic | Dual-track FTQC (Edmonds Blossom MWPM + Gross qLDPC) |

## Side-by-Side: 12 Common Patterns

### 1. Create a Circuit

```python
# ── Cirq ──
import cirq
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit([
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key="result"),
])
```

```python
# ── Quanta ──
from quanta import circuit, H, CX, measure

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

print(bell.build().num_qubits)
```

### 2. Run a Circuit

```python
# ── Cirq ──
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1024)
print(result.histogram(key="result"))
```

```python
# ── Quanta ──
from quanta import circuit, H, CX, measure, run

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell, shots=1024, seed=42)
print(result.counts)  # {'00': 515, '11': 509}
```

### 3. Apply Rotation Gates

```python
# ── Cirq ──
import cirq
import numpy as np
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit([
    cirq.rx(np.pi / 4)(q0),
    cirq.ry(np.pi / 3)(q1),
    cirq.rz(np.pi / 6)(q0),
])
```

```python
# ── Quanta ──
import numpy as np
from quanta import circuit, RX, RY, RZ, measure, run

@circuit(qubits=2)
def rotations(q):
    RX(np.pi / 4)(q[0])
    RY(np.pi / 3)(q[1])
    RZ(np.pi / 6)(q[0])
    return measure(q)

print(run(rotations, shots=256, seed=42).counts)
```

### 4. Noise Simulation

```python
# ── Cirq ──
noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(p=0.01))
noisy_sim = cirq.DensityMatrixSimulator(noise=noise)
result = noisy_sim.run(circuit, repetitions=1024)
```

```python
# ── Quanta ──
from quanta import circuit, H, CX, measure, run
from quanta.simulator.noise import NoiseModel, Depolarizing

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

noise = NoiseModel().add(Depolarizing(probability=0.01))
result = run(bell, shots=1024, noise=noise, seed=42)
print(result.counts)
```

### 5. Statevector Access

```python
# ── Cirq ──
result = cirq.Simulator().simulate(circuit)
print(result.final_state_vector)
```

```python
# ── Quanta ──
from quanta import circuit, H, CX, measure, run

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell, shots=1, seed=42)
print(result.statevector)  # Always available
```

### 6. Circuit Visualization

```python
# ── Cirq ──
print(circuit)
from cirq.contrib.svg import SVGCircuit
SVGCircuit(circuit)
```

```python
# ── Quanta ──
from quanta import circuit, H, CX, measure
from quanta.visualize import draw
from quanta.visualize_svg import to_html

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

print(draw(bell))  # ASCII diagram
html = to_html(bell, title="Bell State")  # Standalone SVG/HTML string
```

### 7. Custom Gate

```python
# ── Cirq ──
class MyGate(cirq.Gate):
    def _num_qubits_(self): return 1
    def _unitary_(self):
        return np.array([[0, 1], [1, 0]])
circuit.append(MyGate()(q0))
```

```python
# ── Quanta ──
import numpy as np
from quanta import circuit, custom_gate, measure, run

MyGate = custom_gate("MyGate", np.array([[0, 1], [1, 0]], dtype=complex))

@circuit(qubits=1)
def demo(q):
    MyGate(q[0])
    return measure(q)

print(run(demo, shots=64, seed=42).most_frequent)  # '1'
```

### 8. Grover Search

```python
# ── Cirq ──
# No built-in Grover — must implement manually
# (build oracle + diffuser circuits by hand)
```

```python
# ── Quanta ──
from quanta.layer3.search import search

result = search(num_bits=3, target=5, seed=42)
print(result.most_frequent)  # '101' (= 5)
```

### 9. VQE

```python
# ── Cirq ──
import cirq
from scipy.optimize import minimize
# Build ansatz manually, define cost function,
# run optimizer loop with cirq.Simulator()
```

```python
# ── Quanta ──
from quanta.layer3.vqe import vqe

result = vqe(num_qubits=2, hamiltonian=[("ZZ", -1.05), ("XX", 0.39)], seed=42)
print(f"{result.energy:.4f}")
```

### 10. QAOA Optimization

```python
# ── Cirq ──
# Manual implementation: cost unitaries, mixer unitaries,
# parameter optimization loop (~80 lines of code)
```

```python
# ── Quanta ──
from quanta.layer3.optimize import optimize

edges = [(0, 1), (1, 2), (2, 3)]
result = optimize(
    num_bits=4,
    cost=lambda x: sum(1 for i, j in edges if ((x >> i) & 1) != ((x >> j) & 1)),
    minimize=False,
    layers=2,
    seed=42,
)
print(result.best_bitstring, result.best_cost)
```

### 11. QEC Surface Code

```python
# ── Cirq ──
# Use cirq-google's experimental QEC module
# Limited to Google hardware topologies
```

```python
# ── Quanta ──
from quanta.qec.surface_code import SurfaceCode

code = SurfaceCode(distance=3)
result = code.simulate_error_correction(error_rate=0.01, seed=42)
print(f"Logical error rate: {result.logical_error_rate:.4%}")
```

### 12. Export to QASM

```python
# ── Cirq ──
from cirq.qasm_output import QasmOutput
qasm = QasmOutput(circuit, (q0, q1))
print(str(qasm))
```

```python
# ── Quanta ──
from quanta import circuit, H, CX, measure
from quanta.export.qasm import to_qasm

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

qasm_str = to_qasm(bell)  # Outputs OPENQASM 3.0
print(qasm_str.splitlines()[0])
```

## Key Differences

| Cirq | Quanta |
|------|--------|
| `cirq.LineQubit.range(n)` | `@circuit(qubits=n)` |
| `cirq.H(q)` | `H(q[0])` |
| `cirq.rx(angle)(q)` | `RX(angle)(q[0])` |
| `cirq.CNOT(q0, q1)` | `CX(q[0], q[1])` |
| `cirq.Simulator().run()` | `run(circuit, shots=N)` |
| `cirq.DensityMatrixSimulator` | Built-in noise & density-matrix support |
| Google hardware only | IBM Quantum + IonQ + Google + local backends |

## Cirq Concepts → Quanta

| Cirq Concept | Quanta Equivalent |
|-------------|-------------------|
| `Moment` | Automatic DAG layer tracking |
| `Circuit` | `@circuit` decorator |
| `LineQubit` / `GridQubit` | `q[i]` (index-based) |
| `Simulator` | `run()` (auto-selected) |
| `cirq.measure()` | `measure(q)` or `measure(q[0], q[1])` |
| `InsertStrategy` | Automatic DAG scheduling |
| `noise.ConstantQubitNoiseModel` | `NoiseModel().add(...)` |

## Getting Help

- [Tutorials](../tutorials/01-getting-started.md) — Structured learning path
- [Architecture](../ARCHITECTURE_EN.md) — How Quanta is designed
- [Features](../FEATURES_EN.md) — Complete feature reference
- [GitHub Issues](https://github.com/ONMARTECH/quanta-sdk/issues) — Bug reports
