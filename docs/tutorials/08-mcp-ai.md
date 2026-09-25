# MCP AI Integration

> Tested with: Quanta SDK v1.2.0-production

## What You'll Learn

Connect Quanta to Claude, GPT, and other AI assistants using the Model Context Protocol (MCP).

## Prerequisites

- [01 — Getting Started](01-getting-started.md)
- Claude Desktop or any MCP-compatible AI client

## What is MCP?

MCP (Model Context Protocol) lets AI assistants discover and call tools directly. Quanta exposes **23 native quantum tools** that any MCP client can use:

| Tool | Category | Description |
|------|----------|-------------|
| `run_circuit` | Core Circuit | Build and simulate any quantum circuit |
| `create_bell_state` | Core Circuit | Instant Bell state preparation |
| `list_gates` | Core Circuit | Inspect all 31 native quantum gates |
| `draw_circuit` | Core Circuit | Visual ASCII / SVG circuit diagram |
| `optimize_circuit` | Compilation | Multi-pass compiler optimization |
| `transpile_for_target` | Compilation | Target architecture transpilation |
| `grover_search` | Algorithms | Quadratic search algorithm |
| `shor_factor` | Algorithms | Integer factorization via period finding |
| `qaoa_optimize` | Algorithms | Combinatorial QAOA solver |
| `monte_carlo_price` | Finance | Amplitude estimation option pricing |
| `option_greeks` | Finance | Quantum Monte Carlo Greeks evaluation |
| `cluster_data` | Machine Learning | Swap-test quantum k-means clustering |
| `qml_classify` | Machine Learning | Differentiable quantum classification |
| `simulate_noise` | Noise & Systems | 7-channel Kraus open system simulation |
| `surface_code_simulate` | 2026 FTQC | Stabilizer surface code simulation |
| `compare_decoders` | 2026 FTQC | Benchmark Edmonds Blossom vs heuristics |
| `qec_diagnose` | 2026 FTQC | Error syndrome diagnosis and extraction |
| `estimate_fault_tolerant_cost` | 2026 FTQC | Physical qubit overhead estimation |
| `quanta_reasoning_eval` | Cognitive | Biomorphic Zeno cognitive decision arbitration |
| `explain_result` | AI Utilities | AI-native structured measurement analysis |
| `run_on_ibm` | Hardware | Submit job to IBM Quantum cloud |
| `ibm_backends` | Hardware | Enumerate real quantum processors |
| `ibm_job_result` | Hardware | Poll remote quantum hardware execution |

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
| 02 | Gates and Circuits — all 31 gates |
| 03 | Simulation — noise and fidelity |
| 04 | Algorithms — Grover, QAOA, VQE, Shor |
| 05 | IBM Hardware — real quantum computers |
| 06 | QML — quantum machine learning |
| 07 | QEC — error correction |
| 08 | MCP — AI integration |

**Next steps:**
- [Cookbook](../cookbook/index.md) — Copy-paste recipes for common tasks
- [Migration Guide](../migration/from-qiskit.md) — Coming from Qiskit?
- [GitHub](https://github.com/ONMARTECH/quanta-sdk) — Contribute!
