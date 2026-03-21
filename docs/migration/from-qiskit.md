# Coming from Qiskit

> Tested with: Quanta SDK v0.8.1

## Why Switch?

| Feature | Qiskit | Quanta |
|---------|--------|--------|
| Install size | ~500 MB | ~5 MB (Python + NumPy only) |
| IBM Hardware | ✅ (via Runtime) | ✅ (direct REST API, no Qiskit needed) |
| Dependencies | 20+ packages | 2 (numpy, python-dotenv) |
| Learning curve | Steep (Primitives, Transpiler, Provider) | Gentle (3 imports to start) |
| Gate count | 50+ | 31 (IBM Heron parity + Google/IonQ native) |
| AI Integration | ❌ | ✅ (16 MCP tools for Claude/GPT) |

## Side-by-Side: 15 Common Patterns

### 1. Create a Circuit

```python
# ── Qiskit ──
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

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
# ── Qiskit ──
from qiskit.primitives import StatevectorSampler
sampler = StatevectorSampler()
job = sampler.run([qc], shots=1024)
result = job.result()

# ── Quanta ──
from quanta import run
result = run(bell, shots=1024)
print(result.counts)  # {'00': 512, '11': 512}
```

### 3. Apply Gates

```python
# ── Qiskit ──
qc.rx(3.14/4, 0)
qc.ry(3.14/3, 1)
qc.rz(3.14/6, 0)

# ── Quanta ──
RX(3.14/4)(q[0])    # Note: gate(angle)(qubit) syntax
RY(3.14/3)(q[1])
RZ(3.14/6)(q[0])
```

### 4. Noise Simulation

```python
# ── Qiskit ──
from qiskit_aer.noise import NoiseModel, depolarizing_error
noise = NoiseModel()
noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ['h', 'x'])

# ── Quanta ──
from quanta.simulator.noise import NoiseModel, Depolarizing
noise = NoiseModel().add(Depolarizing(p=0.01))
result = run(bell, shots=1024, noise=noise)
```

### 5. Export to QASM

```python
# ── Qiskit ──
from qiskit.qasm3 import dumps
qasm_str = dumps(qc)

# ── Quanta ──
from quanta.export.qasm import to_qasm
qasm_str = to_qasm(bell)
# Both output OPENQASM 3.0
```

### 6. Grover Search

```python
# ── Qiskit ──
from qiskit.circuit.library import GroverOperator
from qiskit_algorithms import Grover, AmplificationProblem
oracle = QuantumCircuit(3)
oracle.cz(0, 2)
problem = AmplificationProblem(oracle)
grover = Grover(sampler=sampler)
result = grover.amplify(problem)

# ── Quanta ──
from quanta.layer3.search import search
result = search(target=5, n_qubits=3)
print(result.found_value)  # 5
```

### 7. VQE

```python
# ── Qiskit ──
from qiskit_algorithms import VQE
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import StatevectorEstimator
ansatz = EfficientSU2(2, reps=1)
vqe = VQE(StatevectorEstimator(), ansatz, COBYLA())
result = vqe.compute_minimum_eigenvalue(hamiltonian)

# ── Quanta ──
from quanta.layer3.vqe import vqe
result = vqe([(-1.05, "ZZ"), (0.39, "XX")], n_qubits=2)
print(result.energy)
```

### 8. Run on IBM Hardware

```python
# ── Qiskit ──
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
service = QiskitRuntimeService(channel="ibm_quantum", token="...")
backend = service.least_busy(min_num_qubits=2)
sampler = SamplerV2(backend)
job = sampler.run([qc])
result = job.result()

# ── Quanta ──
from quanta.backends.ibm_rest import IBMRestBackend
backend = IBMRestBackend(backend_name="ibm_torino")
result = run(bell, backend=backend, shots=4096)
```

### 9. QAOA Optimization

```python
# ── Qiskit ──
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
qaoa = QAOA(sampler, COBYLA(), reps=2)
result = qaoa.compute_minimum_eigenvalue(cost_operator)

# ── Quanta ──
from quanta.layer3.optimize import optimize
result = optimize(edges=[(0,1),(1,2),(2,3)], p=2)
```

### 10. Circuit Visualization

```python
# ── Qiskit ──
qc.draw("mpl")

# ── Quanta ──
from quanta.visualize import draw
draw(bell)  # ASCII diagram
# Or SVG:
from quanta.visualize_svg import draw_svg
draw_svg(bell, "circuit.html")
```

### 11. Statevector Access

```python
# ── Qiskit ──
from qiskit.quantum_info import Statevector
sv = Statevector.from_instruction(qc)
print(sv.data)

# ── Quanta ──
result = run(bell, shots=1)
print(result.statevector)  # Always available
```

### 12. Custom Gate

```python
# ── Qiskit ──
from qiskit.circuit.library import UnitaryGate
import numpy as np
gate = UnitaryGate(np.array([[0,1],[1,0]]))
qc.append(gate, [0])

# ── Quanta ──
from quanta import custom_gate
custom_gate("MyGate", np.array([[0,1],[1,0]]))
# Then use: MyGate(q[0])
```

### 13. Measurement

```python
# ── Qiskit ──
qc.measure_all()  # Measure everything
# or
qc.measure([0], [0])  # Selective

# ── Quanta ──
return measure(q)        # Measure all
return measure(q[0:2])   # Measure qubits 0,1 only
```

### 14. Transpilation

```python
# ── Qiskit ──
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
isa_circuit = pm.run(qc)

# ── Quanta ──
from quanta.compiler.pipeline import compile_circuit
dag = compile_circuit(bell, target="heron")
# Automatic: SWAP insertion, gate translation, optimization
```

### 15. QML Classification

```python
# ── Qiskit (via qiskit-machine-learning) ──
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
vqc = VQC(feature_map=ZZFeatureMap(2), ansatz=RealAmplitudes(2),
           optimizer=COBYLA(), sampler=sampler)
vqc.fit(X_train, y_train)

# ── Quanta ──
from quanta.layer3.qml import QuantumClassifier
clf = QuantumClassifier(n_qubits=2, n_layers=2)
clf.fit(X_train, y_train, epochs=30)
predictions = clf.predict(X_test)
```

## Key Differences to Remember

| Qiskit | Quanta |
|--------|--------|
| `QuantumCircuit(n)` | `@circuit(qubits=n)` |
| `qc.h(0)` | `H(q[0])` |
| `qc.rx(angle, 0)` | `RX(angle)(q[0])` |
| `Sampler().run()` | `run(circuit, shots=N)` |
| `NoiseModel()` | `NoiseModel().add(...)` |
| `QiskitRuntimeService` | `IBMRestBackend()` |
| Multiple packages | Single `pip install quanta-sdk` |

## Getting Help

- [Tutorials](../tutorials/01-getting-started.md) — Structured learning path
- [Architecture](../ARCHITECTURE_EN.md) — How Quanta is designed
- [Features](../FEATURES_EN.md) — Complete feature reference
- [GitHub Issues](https://github.com/ONMARTECH/quanta-sdk/issues) — Bug reports
