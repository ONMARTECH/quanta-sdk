# MCP Server

Quanta's MCP server exposes **25 tools**, 5 resources, and 4 guided prompts
for AI assistants (Claude, GPT, Gemini, etc.).

## Installation

```bash
pip install "quanta-sdk[mcp]"
fastmcp install quanta/mcp_server.py --name "Quanta Quantum SDK"
```

## Tools (25 Tools)

| Tool | Category | Description |
|------|----------|-------------|
| `run_circuit` | Core | Execute quantum circuit code |
| `mps_circuit_simulate` | Core | Simulate 200+ qubit circuits via Matrix Product States (MPS) |
| `create_bell_state` | Education | Quick Bell state \|Φ+⟩ |
| `grover_search` | Research | Grover's search algorithm |
| `shor_factor` | Research | Shor's factoring algorithm |
| `simulate_noise` | Research | Run circuit with noise model |
| `draw_circuit` | Education | SVG circuit diagram |
| `list_gates` | Education | All 34 quantum gates |
| `explain_result` | Education | Interpret measurements |
| `monte_carlo_price` | Business | Quantum Monte Carlo option pricing |
| `option_greeks` | Business | Compute option Greeks (Δ,Γ,ν,Θ,ρ) |
| `portfolio_optimize` | Business | Combinatorial portfolio optimization via QAOA |
| `qaoa_optimize` | Research | QAOA combinatorial optimization |
| `cluster_data` | Business | Quantum clustering |
| `qml_classify` | Business | Quantum machine learning classification |
| `optimize_circuit` | Core | Compile and optimize quantum circuits |
| `transpile_for_target` | Compilation | Transpile circuit for specific hardware topology |
| `run_on_ibm` | Hardware | Run on IBM Quantum hardware |
| `ibm_backends` | Hardware | List IBM quantum computers |
| `ibm_job_result` | Hardware | Poll job status & fetch results |
| `surface_code_simulate` | 2026 FTQC | Surface code QEC simulation |
| `compare_decoders` | 2026 FTQC | Compare Edmonds Blossom vs heuristics |
| `qec_diagnose` | 2026 FTQC | Diagnose errors from syndrome measurement |
| `estimate_fault_tolerant_cost` | 2026 FTQC | Estimate FTQC physical qubit & runtime overhead |
| `quanta_reasoning_eval` | Cognitive | Biomorphic Zeno cognitive decision arbitration |

## Resources

| URI | Description |
|-----|-------------|
| `quanta://info` | SDK version and capabilities |
| `quanta://examples` | Example quantum circuits |
| `quanta://noise-profiles` | 7 noise channel specifications |
| `quanta://gate-catalog` | 34 gates with categories |
| `quanta://backend-specs` | IBM hardware specifications |

## Prompts

| Prompt | Description |
|--------|-------------|
| `grover-tutorial` | Step-by-step Grover's search |
| `option-pricing` | Quantum Monte Carlo finance workflow |
| `circuit-debug` | Systematic circuit debugging |
| `qec-intro` | Interactive QEC exploration |
