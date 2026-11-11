# MCP AI Integration

> Tested with: Quanta SDK v1.3.0

## What You'll Learn

Connect Quanta to Claude, GPT, Gemini, and other AI assistants using the Model Context Protocol (MCP).

## Prerequisites

- [01 — Getting Started](01-getting-started.md)
- Claude Desktop or any MCP-compatible AI client

## What is MCP?

MCP (Model Context Protocol) lets AI assistants discover and call tools directly. Quanta exposes **25 native quantum tools** that any MCP client can invoke:

| Tool | Category | Description |
|------|----------|-------------|
| `run_circuit` | Core Circuit | Build and simulate any quantum circuit |
| `mps_circuit_simulate` | Core Circuit | 200+ qubit Matrix Product State (MPS) tensor network simulation |
| `create_bell_state` | Core Circuit | Instant Bell state preparation |
| `list_gates` | Core Circuit | Inspect all 34 native quantum gates |
| `draw_circuit` | Core Circuit | Visual SVG/HTML circuit diagram |
| `optimize_circuit` | Compilation | Multi-pass compiler optimization |
| `transpile_for_target` | Compilation | Target architecture transpilation (`ibm_heron`, `ionq_aria`, `google_sycamore`) |
| `grover_search` | Algorithms | Quadratic search algorithm |
| `shor_factor` | Algorithms | Integer factorization via semiclassical period finding |
| `qaoa_optimize` | Algorithms | Combinatorial QAOA solver |
| `monte_carlo_price` | Finance | Amplitude estimation option pricing |
| `option_greeks` | Finance | Option Greeks ($\Delta, \Gamma, \nu, \Theta, \rho$) evaluation |
| `portfolio_optimize` | Finance | Combinatorial portfolio optimization via QAOA |
| `cluster_data` | Machine Learning | Swap-test quantum k-means clustering |
| `qml_classify` | Machine Learning | Differentiable quantum classification |
| `simulate_noise` | Noise & Systems | 7-channel Kraus open-system simulation |
| `surface_code_simulate` | 2026 FTQC | Stabilizer surface code simulation |
| `compare_decoders` | 2026 FTQC | Benchmark Edmonds Blossom MWPM vs Union-Find |
| `qec_diagnose` | 2026 FTQC | Error syndrome diagnosis and extraction |
| `estimate_fault_tolerant_cost` | 2026 FTQC | Physical qubit & runtime overhead estimation |
| `quanta_reasoning_eval` | Cognitive | Biomorphic Zeno cognitive decision arbitration |
| `explain_result` | AI Utilities | Structured measurement interpretation |
| `run_on_ibm` | Hardware | Submit job to IBM Quantum cloud |
| `ibm_backends` | Hardware | Enumerate real quantum processors |
| `ibm_job_result` | Hardware | Poll remote quantum hardware execution |

## Setup — Claude Desktop

Install the MCP server locally:

```bash
# Install Quanta with MCP support
pip install "quanta-sdk[mcp]"

# Register with Claude Desktop
fastmcp install quanta/mcp_server.py --name "Quanta Quantum SDK"
```

Then in Claude Desktop, you can say:

> "Create a Bell state and explain the results"

Claude will call `create_bell_state` → `explain_result` automatically.

## Setup — Cloud / SSE Deployment

For always-on MCP access:

```bash
# Run as SSE server (Cloud Run, container, etc.)
python -m quanta.mcp_server --transport sse --port 8080
```

## Example AI Conversations

### "Factor the number 15"
→ AI calls `shor_factor(number=15)` → Returns `{"factors": [3, 5], ...}`

### "Search for value 7 in a 4-qubit space"
→ AI calls `grover_search(num_qubits=4, target=7)` → Returns amplified bitstring `'0111'`

### "Price a European call option with spot=100, strike=105"
→ AI calls `monte_carlo_price(S0=100.0, K=105.0)` → Returns quantum vs classical prices

### "Show me the noise effect on a Bell state at 5% error"
→ AI calls `simulate_noise(noise_type="depolarizing", probability=0.05)` → Returns noisy counts & fidelity

## SDK Info via MCP

```python
# The MCP server exposes SDK metadata via quanta://info
import json

sdk_info = {
    "name": "quanta-sdk",
    "version": "1.3.0",
    "tools": 25,
    "gates": 34,
    "noise_channels": 7,
}
print(json.dumps(sdk_info, indent=2))
```

## Try It Yourself

1. Install the MCP server in Claude Desktop and ask it to explain quantum entanglement with a live Bell state
2. Ask your AI assistant to compare noisy vs ideal simulation using `simulate_noise`
3. Try *"Optimize a 2-asset selection from 4 candidates using `portfolio_optimize`"*

## Core Learning Path Complete — Explore Advanced Tutorials!

You've completed the 8 core tutorials:

| Tutorial | Topic |
|----------|-------|
| [01](01-getting-started.md) | Getting Started — first circuit |
| [02](02-gates-and-circuits.md) | Gates and Circuits — all 34 gates |
| [03](03-simulation.md) | Simulation — 7 noise channels & backends |
| [04](04-algorithms.md) | Algorithms — Grover, QAOA, VQE, Shor |
| [05](05-ibm-hardware.md) | IBM Hardware — real quantum computers |
| [06](06-qml.md) | QML — quantum machine learning |
| [07](07-noise-and-qec.md) | QEC — stabilizer & qLDPC error correction |
| [08](08-mcp-ai.md) | MCP — 25 AI tools integration |

### Real-World & Frontier Deep Dives (`09a`–`12`)

- [09a — Shor & RSA Cryptography](09a-shor-cryptography.md)
- [09b — QAOA Logistics & Routing](09b-qaoa-logistics.md)
- [09c — QML Credit Card Fraud Detection](09c-qml-fraud.md)
- [10a — VQE Molecular Chemistry ($H_2$)](10a-vqe-molecular.md)
- [10b — Bell Inequality & CHSH Violation](10b-bell-inequality.md)
- [10c — Surface Code QEC Threshold](10c-qec-threshold.md)
- [11 — Google Willow Dynamic Surface Code](11-willow-dynamic-qec.md)
- [12 — Agentic Circuit Auditing via MCP](12-agentic-circuit-auditing.md)

**More resources:**
- [Cookbook](../cookbook/index.md) — Copy-paste recipes for common tasks
- [Migration Guide](../migration/from-qiskit.md) — Coming from Qiskit, Cirq, or PennyLane?
- [GitHub](https://github.com/ONMARTECH/quanta-sdk) — Source & issues
