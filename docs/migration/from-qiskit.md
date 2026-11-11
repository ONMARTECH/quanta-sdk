# Coming from Qiskit

> Tested with: Quanta SDK v1.3.0

## Why Switch?

| Feature | Qiskit | Quanta |
|---------|--------|--------|
| Install size | ~500 MB | ~5 MB (Python + NumPy core) |
| IBM Hardware | ✅ (via Runtime) | ✅ (direct REST API, no Qiskit needed) |
| Core dependencies | 20+ packages | 1 (`numpy>=1.24`) |
| Learning curve | Steep (Primitives, Transpiler, Provider) | Gentle (3 imports to start) |
| Gate count | 50+ | 34 (IBM Heron + Google + IonQ native) |
| AI Integration | ❌ | ✅ (25 MCP tools for Claude/GPT/Gemini) |

## Side-by-Side: 15 Common Patterns

### 1. Create a Circuit

```python
# ── Qiskit ──
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()
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
# ── Qiskit ──
from qiskit.primitives import StatevectorSampler
sampler = StatevectorSampler()
job = sampler.run([qc], shots=1024)
result = job.result()
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

### 3. Apply Gates

```python
# ── Qiskit ──
qc.rx(3.14 / 4, 0)
qc.ry(3.14 / 3, 1)
qc.rz(3.14 / 6, 0)
```

```python
# ── Quanta ──
import math
from quanta import circuit, RX, RY, RZ, measure, run

@circuit(qubits=2)
def rotations(q):
    RX(math.pi / 4)(q[0])  # Note: gate(angle)(qubit) syntax
    RY(math.pi / 3)(q[1])
    RZ(math.pi / 6)(q[0])
    return measure(q)

print(run(rotations, shots=256, seed=42).counts)
```

### 4. Noise Simulation

```python
# ── Qiskit ──
from qiskit_aer.noise import NoiseModel, depolarizing_error
noise = NoiseModel()
noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["h", "x"])
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

### 5. Export to QASM

```python
# ── Qiskit ──
from qiskit.qasm3 import dumps
qasm_str = dumps(qc)
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

qasm_str = to_qasm(bell)  # Both output OPENQASM 3.0
print(qasm_str.splitlines()[0])
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
```

```python
# ── Quanta ──
from quanta.layer3.search import search

result = search(num_bits=3, target=5, seed=42)
print(result.most_frequent)  # '101' (= 5)
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
```

```python
# ── Quanta ──
from quanta.layer3.vqe import vqe

result = vqe(num_qubits=2, hamiltonian=[("ZZ", -1.05), ("XX", 0.39)], seed=42)
print(f"{result.energy:.4f}")
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
```

```python
# ── Quanta ── (live execution requires IBM_API_KEY)
from quanta.backends.ibm_rest import IBMRestBackend

backend = IBMRestBackend(backend_name="ibm_torino")
print(backend.name)
# result = run(bell, backend=backend, shots=4096)
```

### 9. QAOA Optimization

```python
# ── Qiskit ──
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
qaoa = QAOA(sampler, COBYLA(), reps=2)
result = qaoa.compute_minimum_eigenvalue(cost_operator)
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

### 10. Circuit Visualization

```python
# ── Qiskit ──
qc.draw("mpl")
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

### 11. Statevector Access

```python
# ── Qiskit ──
from qiskit.quantum_info import Statevector
sv = Statevector.from_instruction(qc)
print(sv.data)
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

### 12. Custom Gate

```python
# ── Qiskit ──
from qiskit.circuit.library import UnitaryGate
import numpy as np
gate = UnitaryGate(np.array([[0, 1], [1, 0]]))
qc.append(gate, [0])
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

### 13. Measurement

```python
# ── Qiskit ──
qc.measure_all()      # Measure everything
qc.measure([0], [0])  # Selective
```

```python
# ── Quanta ──
from quanta import circuit, H, CX, measure, run

@circuit(qubits=3)
def selective_measure(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q[0], q[1])  # Measure qubits 0 and 1 only (measure(q) measures all)

print(run(selective_measure, shots=256, seed=42).counts)
```

### 14. Transpilation

```python
# ── Qiskit ──
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
isa_circuit = pm.run(qc)
```

```python
# ── Quanta ──
from quanta import circuit, H, CX, measure
from quanta.dag.dag_circuit import DAGCircuit
from quanta.compiler.pipeline import CompilerPipeline
from quanta.compiler.passes.translate import TranslateToTarget
from quanta.compiler.passes.optimize import CancelInverses, MergeRotations

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

dag = DAGCircuit.from_builder(bell.build())
pipeline = CompilerPipeline([
    TranslateToTarget("ibm_heron"),
    CancelInverses(),
    MergeRotations(),
])
compiled = pipeline.run(dag)
print(f"Compiled gates: {compiled.gate_count()}, depth: {compiled.depth()}")
```

### 15. QML Classification

```python
# ── Qiskit (via qiskit-machine-learning) ──
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
vqc = VQC(feature_map=ZZFeatureMap(2), ansatz=RealAmplitudes(2),
          optimizer=COBYLA(), sampler=sampler)
vqc.fit(X_train, y_train)
```

```python
# ── Quanta ──
import numpy as np
from quanta.layer3.qml import QuantumClassifier

X_train = np.array([[0.1, 0.2], [0.8, 0.9], [0.2, 0.1], [0.9, 0.8]])
y_train = np.array([0, 1, 0, 1])
X_test = np.array([[0.15, 0.25], [0.85, 0.85]])

clf = QuantumClassifier(n_qubits=2, n_layers=2, seed=42)
clf.fit(X_train, y_train, epochs=25)
predictions = clf.predict(X_test)
print(predictions)
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
