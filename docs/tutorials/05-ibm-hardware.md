# IBM Quantum Hardware

> Tested with: Quanta SDK v0.8.1

## What You'll Learn

Submit circuits to real IBM quantum hardware, poll job results, and compare simulator vs hardware output.

## Prerequisites

- [04 — Algorithms](04-algorithms.md)
- An IBM Quantum account ([quantum.ibm.com](https://quantum.ibm.com))

## Step 1 — Set Up Credentials

Create a `.env` file in your project root:

```
IBM_API_KEY=your_api_key_here
```

Or set the environment variable:

```bash
export IBM_API_KEY=your_api_key_here
```

> **Security**: `.env` is automatically gitignored by Quanta SDK.

## Step 2 — List Available Backends

```python
# ── Quanta ── (requires IBM_API_KEY)
# from quanta.backends.ibm_rest import IBMRestBackend
# backends = IBMRestBackend.available_backends()
# for b in backends:
#     print(f"{b['name']}: {b['num_qubits']} qubits, status={b['status']}")
print("IBM backend listing requires API key — see docs")
```

## Step 3 — Submit a Job

```python
# ── Quanta IBM hardware submission ──
# from quanta import circuit, H, CX, measure, run
# from quanta.backends.ibm_rest import IBMRestBackend
#
# @circuit(qubits=2)
# def bell(q):
#     H(q[0])
#     CX(q[0], q[1])
#     return measure(q)
#
# backend = IBMRestBackend(backend_name="ibm_torino")
# result = run(bell, backend=backend, shots=4096)
# print(result.summary())
print("IBM job submission requires API key and queue time")
```

## Step 4 — Multi-Backend Support

Quanta supports 4 backends:

| Backend | Class | Use Case |
|---------|-------|----------|
| **Local** | (default) | Development, testing, education |
| **IBM Quantum** | `IBMRestBackend` | Superconducting qubits (Heron r3) |
| **IonQ** | `IonQBackend` | Trapped-ion (high fidelity) |
| **Google** | `GoogleBackend` | Sycamore processor |

```python
# All backends use the same run() API:
# result = run(circuit, backend=backend, shots=N)
#
# IBM:
# backend = IBMRestBackend(backend_name="ibm_torino")
#
# IonQ:
# backend = IonQBackend(target="simulator")  # or "qpu"
#
# Google:
# backend = GoogleBackend(processor_id="weber")
print("Multi-backend: IBM, IonQ, Google — same run() API")
```

## Step 5 — Compiler Pipeline

Before running on hardware, circuits are automatically compiled:

```python
from quanta.compiler.pipeline import CompilerPipeline
from quanta.dag.dag_circuit import DAGCircuit
from quanta import circuit, H, CX, measure

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

dag = DAGCircuit.from_builder(bell.build())
pipeline = CompilerPipeline()
compiled = pipeline.run(dag)
print(f"Compiled: {compiled.gate_count()} gates, depth {compiled.depth()}")
print(pipeline.summary())
```

## IBM Quantum Tips

1. **Use `ibm_sherbrooke`** for 127-qubit experiments
2. **Use `ibm_torino`** for latest Heron r3 (156 qubits)
3. **Start small**: Test with 2-5 qubits first
4. **Queue times**: Expect 1-30 minutes depending on system load
5. **Transpilation**: Quanta automatically translates to {CX, RZ, SX, X}

## What's Next

→ [06 — Quantum Machine Learning](06-qml.md): Deep-dive into QuantumClassifier
