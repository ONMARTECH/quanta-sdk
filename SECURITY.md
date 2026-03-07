# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.6.x   | :white_check_mark: |
| 0.5.x   | :white_check_mark: |
| < 0.5   | :x:                |

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
- Code execution vulnerabilities in the simulator
- Dependency vulnerabilities (numpy)
- API token exposure in backend modules (Google, IBM, IonQ)
- Unsafe deserialization in QASM import

The following are out of scope:
- Quantum algorithm correctness (not a security issue)
- Performance issues
- Issues in example scripts

## Security Best Practices

When using hardware backends:
- **Never hardcode API tokens** in source code
- Use environment variables (`IBM_QUANTUM_TOKEN`, `GOOGLE_CLOUD_PROJECT`)
- Rotate tokens regularly
- Use the minimum required permissions

## Dependencies

Quanta SDK has a minimal dependency footprint:
- **Required**: `numpy` (only runtime dependency)
- **Optional**: `jax`, `cupy` (GPU acceleration)
- **Dev**: `pytest`, `ruff`, `mypy` (development only)

We monitor dependencies for known vulnerabilities and update promptly.
