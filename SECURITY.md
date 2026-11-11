# Security Policy

## Supported Versions

| Version       | Supported |
| ------------- | --------- |
| 1.3.x         | ✅        |
| 1.2.x         | ✅        |
| < 1.2         | ❌        |

## Reporting a Vulnerability

If you discover a security vulnerability in Quanta SDK, please report it responsibly.

### How to Report

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email: **info@onmartech.com**

Include:
1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix release**: Within 30 days for critical issues

### Scope

The following are in scope:
- Sandbox escape or code execution vulnerabilities in `quanta.mcp_server` (`_validate_code` / `_SAFE_BUILTINS`)
- Dependency vulnerabilities (`numpy`, `scipy`, `networkx`, `certifi`, `fastmcp`, `torch`)
- API token exposure or credential leakage in hardware backends (`IBM`, `IonQ`, `Google`) or `QuantaConfig`
- Unsafe parsing or deserialization in OpenQASM 2.0/3.0 import (`quanta.export.qasm_import`)

The following are out of scope:
- Statistical variance in shot-based quantum simulation
- Performance tuning on classical RAM-bounded statevectors
- Issues in standalone example scripts

## Security Best Practices

When using cloud quantum hardware backends or the MCP server:
- **Never hardcode API keys or tokens** in source code, notebooks, or version-controlled files.
- Provide hardware credentials strictly via environment variables or user-restricted config files (`chmod 600`):
  - `IBM_API_KEY` / `IBM_QUANTUM_TOKEN` (and optional `IBM_INSTANCE_CRN`) for IBM Quantum REST & Runtime execution (`run_on_ibm`, `ibm_backends`, `ibm_job_result`).
  - `IONQ_API_KEY` for IonQ cloud QPU and simulator execution (`IonQBackend`).
  - `GOOGLE_CLOUD_PROJECT` (with Application Default Credentials) for Google Quantum Engine.
- Rotate hardware API keys periodically and scope tokens to least-privilege workspace roles.

## Dependencies

Quanta SDK maintains an audited, modular dependency footprint:
- **Core & Layer-3 Runtime (`pip install quanta-sdk`)**: `numpy>=1.24`, `scipy>=1.10`, `networkx>=3.0`, `certifi>=2024.0` (powers simulators, Cartan/KAK compiler synthesis, MWPM/BP-OSD QEC decoders, variational/QP Layer-3 solvers, and TLS-verified REST backends).
- **Optional Extras**:
  - `mcp`: `fastmcp>=3.0`, `scikit-learn>=1.0`, `python-dotenv>=1.0`
  - `qml`: `scikit-learn>=1.0`
  - `torch`: `torch>=2.0`
  - `metal`: `mlx>=0.20`, `mlx-metal>=0.20` (Apple Silicon macOS ARM64)
  - `gpu`: `jax[cuda12]`, `cupy-cuda12x`, `cuquantum-python-cu12`
  - `docs`: `mkdocs>=1.6`, `mkdocs-material>=9.5`, `mkdocstrings[python]>=0.25`
- **Development (`dev`)**: `pytest`, `pytest-cov`, `hypothesis`, `ruff`, `mypy`, `pyyaml`, `pillow`, `matplotlib`

We monitor dependencies for known CVEs and ship security patches promptly.
