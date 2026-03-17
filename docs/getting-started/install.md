# Installation

## Quick Install

```bash
pip install quanta-sdk
```

## With MCP Server

```bash
pip install "quanta-sdk[mcp]"
```

## Development

```bash
git clone https://github.com/ONMARTECH/quanta-sdk.git
cd quanta-sdk
pip install -e ".[dev]"
```

## Requirements

- Python 3.10+
- NumPy ≥ 1.24

## Optional Dependencies

| Extra | Packages | Purpose |
|-------|----------|---------|
| `mcp` | FastMCP ≥ 3.0 | MCP server for AI assistants |
| `dev` | pytest, ruff, mypy | Development & testing |
| `gpu` | JAX, CuPy | GPU-accelerated simulation |
