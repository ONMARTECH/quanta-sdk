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

result = run(bell, shots=1024, seed=42)
print(result.counts)
```

**Output:**
```
{'00': 515, '11': 509}
```

*(Tip: `print(result)` renders a formatted ASCII histogram and state summary box.)*

## 2. Run Grover's Search

```python
from quanta.layer3.search import search

result = search(num_bits=4, target=5, seed=42)
print(f"Found: {result.most_frequent} (queries={result.oracle_queries})")
```

**Output:**
```
Found: 0101 (queries=3)
```

## 3. Visualize a Circuit

```python
from pathlib import Path
import tempfile
from quanta import circuit, H, CX, measure
from quanta.visualize_svg import to_html

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

html = to_html(bell, title="Bell State", dark_mode=True)
out_path = Path(tempfile.gettempdir()) / "bell.html"
out_path.write_text(html, encoding="utf-8")
print(f"Generated {len(html)} bytes of standalone SVG/HTML")
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
