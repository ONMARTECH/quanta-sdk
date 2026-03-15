# MCP AI Integration

> Tested with: Quanta SDK v0.8.1

## What You'll Learn

Connect Quanta to Claude, GPT, and other AI assistants using the Model Context Protocol (MCP).

## Prerequisites

- [01 — Getting Started](01-getting-started.md)
- Claude Desktop or a MCP-compatible AI client

## What is MCP?

MCP (Model Context Protocol) lets AI assistants call tools directly. Quanta exposes 14 quantum tools that any MCP client can use:

| Tool | Description |
|------|-------------|
| `run_circuit` | Build and run any circuit |
| `create_bell_state` | One-click Bell state |
| `grover_search` | Quantum search |
| `shor_factor` | Factor integers |
| `simulate_noise` | Noisy simulation |
| `list_gates` | Show all 25 gates |
| `explain_result` | AI-friendly result explanation |
| `monte_carlo_price` | Option pricing |
| `qaoa_optimize` | Combinatorial optimization |
| `cluster_data` | Quantum clustering |
| `run_on_ibm` | Submit to IBM hardware |
| `ibm_backends` | List IBM backends |
| `ibm_job_result` | Poll IBM job |
| `draw_circuit` | ASCII circuit diagram |

## Setup — Claude Desktop

Install the MCP server locally:

```bash
# Install Quanta and fastmcp
pip install quanta-sdk fastmcp

# Register with Claude Desktop
fastmcp install quanta/mcp_server.py --name "Quanta Quantum SDK"
```

Then in Claude Desktop, you can say:

> "Create a Bell state and explain the results"

Claude will call `create_bell_state` → `explain_result` automatically.

## Setup — Cloud Deployment

For always-on MCP access:

```bash
# Run as SSE server (Cloud Run, Lambda)
python -m quanta.mcp_server --transport sse --port 8080
```

## Example AI Conversations

### "Factor the number 15"
→ AI calls `shor_factor(N=15)` → Returns `{factors: [3, 5]}`

### "Search for value 7 in a 4-qubit space"
→ AI calls `grover_search(target=7, num_bits=4)` → Returns results

### "Price a European call option with spot=100, strike=105"
→ AI calls `monte_carlo_price(...)` → Returns quantum vs classical prices

### "Show me the noise effect on a Bell state at 5% error"
→ AI calls `simulate_noise(circuit="bell", error_rate=0.05)` → Returns fidelity

## SDK Info via MCP

```python
# The MCP server exposes SDK metadata
import json
sdk_info = {
    "name": "quanta-sdk",
    "version": "0.8.1",
    "tools": 14,
    "gates": 25,
    "algorithms": 12,
}
print(json.dumps(sdk_info, indent=2))
```

## Try It Yourself

1. Install the MCP server in Claude Desktop and ask it to explain quantum entanglement with a live Bell state
2. Ask Claude to compare noisy vs ideal simulation
3. Try "Factor 21 using Shor's algorithm" in conversation

## Learning Path Complete! 🎉

You've completed all 8 tutorials:

| Tutorial | Topic |
|----------|-------|
| 01 | Getting Started — first circuit |
| 02 | Gates and Circuits — all 25 gates |
| 03 | Simulation — noise and fidelity |
| 04 | Algorithms — Grover, QAOA, VQE, Shor |
| 05 | IBM Hardware — real quantum computers |
| 06 | QML — quantum machine learning |
| 07 | QEC — error correction |
| 08 | MCP — AI integration |

**Next steps:**
- [Cookbook](../cookbook/) — Copy-paste recipes for common tasks
- [Migration Guide](../migration/from-qiskit.md) — Coming from Qiskit?
- [GitHub](https://github.com/ONMARTECH/quanta-sdk) — Contribute!
