# Quick Start

Build and run your first quantum circuit in under a minute.

## 1. Create a Bell State

```python
from quanta import circuit, H, CX, measure, run

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell, shots=1024)
print(result)
```

**Output:**
```
Result(counts={'00': 512, '11': 512}, shots=1024, qubits=2)
```

## 2. Run Grover's Search

```python
from quanta.layer3.search import grover_search

result = grover_search(
    oracle_type="value",
    target=5,
    num_qubits=4,
)
print(f"Found: {result.found_value}")
```

## 3. Visualize a Circuit

```python
from quanta.visualize_svg import to_html

html = to_html(bell, title="Bell State", dark_mode=True)
with open("bell.html", "w") as f:
    f.write(html)
```

## 4. Use with AI (MCP)

```bash
fastmcp install quanta/mcp_server.py --name "Quanta"
```

Then ask Claude: *"Create a 3-qubit GHZ state and explain the results"*

## Next Steps

- [Gates & Circuits Tutorial](../tutorials/02-gates-and-circuits.md)
- [API Reference](../api/core/circuit.md)
- [Migration from Qiskit](../migration/from-qiskit.md)
