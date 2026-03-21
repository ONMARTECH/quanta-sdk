---
hide:
  - navigation
---

# ⚛️ Quanta SDK

**AI-native quantum computing SDK for Python**

*The quantum runtime built for AI agents, researchers, and production workloads*

---

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Get up and running in 5 minutes with your first quantum circuit.

    [:octicons-arrow-right-24: Getting Started](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **Tutorials**

    ---

    8 step-by-step tutorials from basics to IBM hardware.

    [:octicons-arrow-right-24: Tutorials](tutorials/01-getting-started.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Auto-generated from docstrings — every class, function, and module.

    [:octicons-arrow-right-24: API Reference](api/core/circuit.md)

-   :material-swap-horizontal:{ .lg .middle } **Migration Guides**

    ---

    Coming from Qiskit, PennyLane, or Cirq? We've got you covered.

    [:octicons-arrow-right-24: Migration](migration/from-qiskit.md)

</div>

---

## Why Quanta?

| Feature | Quanta | Qiskit | Cirq | PennyLane |
|---------|--------|--------|------|-----------|
| **MCP Server** | ✅ 16 tools | ❌ | ❌ | ❌ |
| **Dependencies** | 1 (numpy) | 20+ | 10+ | 10+ |
| **IBM Hardware** | Built-in REST | Via Provider | No | Via plugin |
| **QEC** | 5 codes, 2 decoders | No | No | No |
| **Install time** | ~2s | ~60s | ~30s | ~30s |

## Install

```bash
pip install quanta-sdk
```

## Hello Quantum

```python
from quanta import circuit, H, CX, measure, run

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell, shots=1000)
print(result)  # {'00': ~500, '11': ~500}
```

## MCP AI Integration

```bash
# Add to Claude Desktop
fastmcp install quanta/mcp_server.py --name "Quanta Quantum SDK"
```

16 tools · 5 resources · 4 guided prompts — ready for Claude, GPT, and other AI assistants.
