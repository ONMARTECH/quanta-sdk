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

-   :material-file-document-outline:{ .lg .middle } **Architecture Whitepaper**

    ---

    Comprehensive RevTeX / Markdown manuscript detailing Quanta's 5 core paradigms.

    [:octicons-arrow-right-24: Read Whitepaper](papers/quanta_framework_paper.md)

-   :material-book-open-variant:{ .lg .middle } **Tutorials & Cookbooks**

    ---

    16 step-by-step tutorials from Bell states to Google Willow QEC and Agentic Auditing.

    [:octicons-arrow-right-24: Explore Tutorials](tutorials/01-getting-started.md)

-   :material-brain:{ .lg .middle } **Theoretical Foundations**

    ---

    6 rigorous academic monographs on biomorphic quantum resonance, CSF shielding, and Lie algebras.

    [:octicons-arrow-right-24: Theory Monographs](theory/quantum_brain_frontiers.md)

-   :material-microscope:{ .lg .middle } **2026 Scientific Audit**

    ---

    Peer-reviewed evaluation of FTQC, Edmonds Blossom MWPM, and September 2026 frontiers.

    [:octicons-arrow-right-24: Read Audit](scientific_audit_september_2026.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Auto-generated from docstrings — every class, function, and module.

    [:octicons-arrow-right-24: API Reference](api/core/circuit.md)

</div>

---

## Why Quanta?

| Dimension | **Quanta SDK** | Qiskit | Cirq | PennyLane |
|-----------|----------------|--------|------|-----------|
| **MCP AI Server** | ✅ **23 tools** (Autonomous AI loop) | ❌ | ❌ | ❌ |
| **Dependencies** | **0** (Pure Python/NumPy) | 20+ packages | 10+ packages | 10+ packages |
| **Hardware Acceleration** | **Apple Metal / MLX Zero-Copy** (52.09×) | C++/Rust (Aer) | C++ (qsim) | JAX/Torch C++ |
| **2026 FTQC Engine** | **Dual-Track (MWPM + Gross qLDPC)** | External | No | No |
| **Continuous Autograd**| **Daleckii-Krein Fréchet** ($9.99\times 10^{-16}$) | Finite Diff | Manual | Parameter-Shift / AD |
| **Automated Tests** | **2,076 Passing Tests** | 5000+ | 3000+ | 2500+ |
| **Install Time** | **~2 seconds** | ~60s | ~30s | ~30s |

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
# Add to Claude Desktop or any MCP Client
fastmcp install quanta/mcp_server.py --name "Quanta Quantum SDK"
```

**23 tools · 5 resources · 4 guided prompts** — ready for Claude, Gemini, GPT, and autonomous agent loops.

---

## Authorship & Identity Metadata

- **Lead Author & Principal Architect**: Abdullah Enes SARI (ORCID: [0000-0002-8827-0587](https://orcid.org/0000-0002-8827-0587))
- **Affiliation**: ONMARTECH Quantum Computing Initiative (`info@onmartech.com`)
- **Permanent Software DOI**: [10.5281/zenodo.22952779](https://doi.org/10.5281/zenodo.22952779)

