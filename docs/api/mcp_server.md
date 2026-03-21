# MCP Server

Quanta's MCP server exposes 16 tools, 5 resources, and 4 guided prompts
for AI assistants (Claude, GPT, etc.).

## Installation

```bash
pip install "quanta-sdk[mcp]"
fastmcp install quanta/mcp_server.py --name "Quanta Quantum SDK"
```

## Tools

| Tool | Category | Description |
|------|----------|-------------|
| `run_circuit` | Core | Execute quantum circuit code |
| `create_bell_state` | Education | Quick Bell state |Φ+⟩ |
| `grover_search` | Research | Grover's search algorithm |
| `shor_factor` | Research | Shor's factoring algorithm |
| `simulate_noise` | Research | Run circuit with noise model |
| `draw_circuit` | Education | SVG circuit diagram |
| `list_gates` | Education | All 31 quantum gates |
| `explain_result` | Education | Interpret measurements |
| `monte_carlo_price` | Business | Quantum Monte Carlo option pricing |
| `qaoa_optimize` | Research | QAOA combinatorial optimization |
| `cluster_data` | Business | Quantum clustering |
| `run_on_ibm` | Hardware | Run on IBM Quantum hardware |
| `ibm_backends` | Hardware | List IBM quantum computers |
| `ibm_job_result` | Hardware | Poll job status & fetch results |
| `surface_code_simulate` | Research | Surface code QEC simulation |
| `compare_decoders` | Research | Compare MWPM vs Union-Find decoders |

## Resources

| URI | Description |
|-----|-------------|
| `quanta://info` | SDK version and capabilities |
| `quanta://examples` | Example quantum circuits |
| `quanta://noise-profiles` | 7 noise channel specifications |
| `quanta://gate-catalog` | 31 gates with categories |
| `quanta://backend-specs` | IBM hardware specifications |

## Prompts

| Prompt | Description |
|--------|-------------|
| `grover-tutorial` | Step-by-step Grover's search |
| `option-pricing` | Quantum Monte Carlo finance workflow |
| `circuit-debug` | Systematic circuit debugging |
| `qec-intro` | Interactive QEC exploration |
