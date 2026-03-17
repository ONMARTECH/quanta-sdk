# Coming from Cirq

> Tested with: Quanta SDK v0.8.1

## Why Switch?

| Feature | Cirq | Quanta |
|---------|------|--------|
| Install size | ~300 MB | ~5 MB (Python + NumPy only) |
| IBM Hardware | ❌ (Google only) | ✅ (direct REST API) |
| Dependencies | 15+ packages | 2 (numpy, python-dotenv) |
| Learning curve | Moderate (Moment, Circuit, Simulator) | Gentle (3 imports to start) |
| Gate count | 60+ | 25 (IBM Heron parity) |
| AI Integration | ❌ | ✅ (16 MCP tools for Claude/GPT) |
| QEC | Basic | Surface code + Color code + decoders |

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

# ── Quanta ──
from quanta import circuit, H, CX, measure
@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)
```

### 2. Run a Circuit

```python
# ── Cirq ──
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1024)
print(result.histogram(key="result"))

# ── Quanta ──
from quanta import run
result = run(bell, shots=1024)
print(result.counts)  # {'00': 512, '11': 512}
```

### 3. Apply Rotation Gates

```python
# ── Cirq ──
import numpy as np
circuit = cirq.Circuit([
    cirq.rx(np.pi/4)(q0),
    cirq.ry(np.pi/3)(q1),
    cirq.rz(np.pi/6)(q0),
])

# ── Quanta ──
RX(np.pi/4)(q[0])
RY(np.pi/3)(q[1])
RZ(np.pi/6)(q[0])
```

### 4. Noise Simulation

```python
# ── Cirq ──
noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(p=0.01))
noisy_sim = cirq.DensityMatrixSimulator(noise=noise)
result = noisy_sim.run(circuit, repetitions=1024)

# ── Quanta ──
from quanta.simulator.noise import NoiseModel, Depolarizing
noise = NoiseModel().add(Depolarizing(p=0.01))
result = run(bell, shots=1024, noise=noise)
```

### 5. Statevector Access

```python
# ── Cirq ──
result = cirq.Simulator().simulate(circuit)
print(result.final_state_vector)

# ── Quanta ──
result = run(bell, shots=1)
print(result.statevector)  # Always available
```

### 6. Circuit Visualization

```python
# ── Cirq ──
print(circuit)
# or
from cirq.contrib.svg import SVGCircuit
SVGCircuit(circuit)

# ── Quanta ──
from quanta.visualize import draw
draw(bell)  # ASCII diagram
# Or SVG:
from quanta.visualize_svg import draw_svg
draw_svg(bell, "circuit.html")
```

### 7. Custom Gate

```python
# ── Cirq ──
class MyGate(cirq.Gate):
    def _num_qubits_(self): return 1
    def _unitary_(self):
        return np.array([[0, 1], [1, 0]])
circuit.append(MyGate()(q0))

# ── Quanta ──
from quanta import custom_gate
custom_gate("MyGate", np.array([[0, 1], [1, 0]]))
# Then use: MyGate(q[0])
```

### 8. Grover Search

```python
# ── Cirq ──
# No built-in Grover — must implement manually
# (build oracle + diffuser circuits by hand)

# ── Quanta ──
from quanta.layer3.search import search
result = search(target=5, n_qubits=3)
print(result.found_value)  # 5
```

### 9. VQE

```python
# ── Cirq ──
import cirq
from scipy.optimize import minimize
# Build ansatz manually, define cost function,
# run optimizer loop with cirq.Simulator()
# ~50 lines of boilerplate

# ── Quanta ──
from quanta.layer3.vqe import vqe
result = vqe([(-1.05, "ZZ"), (0.39, "XX")], n_qubits=2)
print(result.energy)
```

### 10. QAOA Optimization

```python
# ── Cirq ──
# Manual implementation: cost unitaries, mixer unitaries,
# parameter optimization loop
# ~80 lines of code

# ── Quanta ──
from quanta.layer3.optimize import optimize
result = optimize(edges=[(0,1),(1,2),(2,3)], p=2)
```

### 11. QEC Surface Code

```python
# ── Cirq ──
# Use cirq-google's experimental QEC module
# Limited to Google hardware topologies

# ── Quanta ──
from quanta.qec.surface_code import SurfaceCode
code = SurfaceCode(distance=3)
result = code.simulate_error_correction(error_rate=0.01)
print(f"Logical error rate: {result.logical_error_rate:.4%}")
```

### 12. Export to QASM

```python
# ── Cirq ──
from cirq.qasm_output import QasmOutput
qasm = QasmOutput(circuit, (q0, q1))
print(str(qasm))

# ── Quanta ──
from quanta.export.qasm import to_qasm
qasm_str = to_qasm(bell)
# Both output OPENQASM 3.0
```

## Key Differences

| Cirq | Quanta |
|------|--------|
| `cirq.LineQubit.range(n)` | `@circuit(qubits=n)` |
| `cirq.H(q)` | `H(q[0])` |
| `cirq.rx(angle)(q)` | `RX(angle)(q[0])` |
| `cirq.CNOT(q0, q1)` | `CX(q[0], q[1])` |
| `cirq.Simulator().run()` | `run(circuit, shots=N)` |
| `cirq.DensityMatrixSimulator` | Built-in noise support |
| Google hardware only | IBM Quantum + simulators |

## Cirq Concepts → Quanta

| Cirq Concept | Quanta Equivalent |
|-------------|-------------------|
| `Moment` | Automatic depth tracking |
| `Circuit` | `@circuit` decorator |
| `LineQubit` / `GridQubit` | `q[i]` (index-based) |
| `Simulator` | `run()` (auto-selected) |
| `cirq.measure()` | `measure(q)` |
| `InsertStrategy` | Automatic DAG scheduling |
| `noise.ConstantQubitNoiseModel` | `NoiseModel().add(...)` |

## Getting Help

- [Tutorials](../tutorials/01-getting-started.md) — Structured learning path
- [Architecture](../ARCHITECTURE_EN.md) — How Quanta is designed  
- [Features](../FEATURES_EN.md) — Complete feature reference
- [GitHub Issues](https://github.com/ONMARTECH/quanta-sdk/issues) — Bug reports
