# Quanta SDK

A clean, modular, and quantum-native quantum computing SDK.

## Vision

Quanta is designed to eliminate the complexity of existing quantum SDKs
(Qiskit, Cirq, PennyLane). Instead of adapting classical computing logic
to quantum, it naturally embraces quantum principles.

## 3-Layer Architecture

```
Layer 3 — Declarative      "What do you want?"
├── search()                Quantum search (auto Grover)
├── optimize()              Combinatorial optimization (QAOA)
└── MultiAgentSystem        Multi-agent decision modeling

Layer 2 — Algorithmic      "How to build the circuit?"
├── @circuit + gates        H, CX, RZ, CCX...
├── measure()               Flexible measurement
└── run()                   Single command execution

Layer 1 — Physical         "How will it run on hardware?"
├── DAG engine              Topological sort, parallelism
├── Compiler                Optimization + transpilation
├── Simulator               Statevector + noise model
├── QEC                     Error correction codes
└── Export                  OpenQASM 3.0 output
```

## Quick Start

### Layer 2: Gate-based programming

```python
from quanta import circuit, H, CX, measure, run

@circuit(qubits=2)
def bell(q):
    H(q[0])           # Superposition
    CX(q[0], q[1])    # Entanglement
    return measure(q)

result = run(bell, shots=1024)
print(result.summary())
```

### Layer 3: No gate knowledge required!

```python
from quanta.layer3.search import search

# Find 13 in a 16-element space — quantum automatic
result = search(num_bits=4, target=13, shots=1024)
print(f"Found: {result.most_frequent}")  # → 1101
```

### Multi-Agent Decision Modeling

```python
from quanta.layer3.agent import Agent, MultiAgentSystem

system = MultiAgentSystem([
    Agent("customer", ["buy", "skip"]),
    Agent("competitor", ["discount", "hold_price"]),
])
system.interact("customer", "competitor", strength=0.7)
result = system.simulate(shots=1024)
print(result.summary())
```

## Installation

```bash
pip install -e ".[dev]"
```

## Testing

```bash
pytest                    # 98 tests, 0.33 seconds
pytest --tb=short -v      # Verbose output
```

## Project Structure

```
quanta/
├── core/           Core types, gates, circuit decorator
├── dag/            DAG engine (topological sort, layers)
├── compiler/       Optimization + transpilation pipeline
├── simulator/      Statevector simulator + noise
├── backends/       Backend abstraction (local, cloud)
├── layer3/         Declarative API (search, optimize, agent)
├── export/         OpenQASM 3.0 output
├── qec/            Quantum error correction codes
├── examples/       Example algorithms
└── docs/           Documentation (TR + EN)
```

## Code Standards

- **Max 330 lines/file** (300 + 10% tolerance)
- **~30% comment/documentation ratio**
- **Modular**: Single responsibility per file
- **Type-safe**: Full type hints
- **Immutable**: Frozen dataclasses
- **Tested**: 98 tests

## License

MIT
