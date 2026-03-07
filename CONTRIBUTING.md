# Contributing to Quanta SDK

Thank you for your interest in contributing to Quanta! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ONMARTECH/quanta-sdk.git
cd quanta-sdk

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install in development mode with all extras
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests with coverage
pytest

# Run a specific test file
pytest tests/test_core.py -v

# Run without coverage (faster)
pytest --no-cov --tb=short
```

## Code Quality

### Linting

We use [Ruff](https://docs.astral.sh/ruff/) for linting. All PRs must pass lint:

```bash
# Check for lint errors
ruff check quanta/

# Auto-fix what's possible
ruff check quanta/ --fix
```

### Type Checking

We use [mypy](https://mypy-lang.org/) for static type analysis:

```bash
mypy quanta/ --ignore-missing-imports
```

### Code Style

- **Language**: All code, comments, and docstrings must be in **English**.
- **Line length**: 100 characters max.
- **Docstrings**: Google-style docstrings for all public functions.
- **Type hints**: Required for all public API functions.

## Project Structure

```
quanta/
├── core/          # Circuit, gates, types, qubits
├── dag/           # DAG circuit representation
├── compiler/      # Optimization passes
├── simulator/     # Statevector, density matrix, noise
├── layer3/        # High-level APIs (search, optimize, factor, etc.)
├── qec/           # Quantum error correction codes
├── backends/      # Hardware backends (Google, IBM, IonQ)
├── export/        # QASM export
├── examples/      # Runnable examples
└── benchmark/     # Benchpress adapter
```

## Making Changes

1. **Fork** the repository and create a feature branch.
2. **Write tests** for your changes in `tests/`.
3. **Run the full test suite** — all tests must pass with ≥80% coverage.
4. **Run lint** — zero Ruff errors required.
5. **Submit a PR** against `main`.

## What to Contribute

### Good First Issues

- Add missing docstrings to internal functions
- Improve test coverage for specific modules
- Add new quantum algorithm examples to `examples/`

### Feature Ideas

- New noise channels (crosstalk, T1/T2 relaxation)
- Additional QEC codes
- Jupyter notebook integration
- New Layer 3 algorithms

## Reporting Bugs

Please open a [GitHub Issue](https://github.com/ONMARTECH/quanta-sdk/issues) with:

1. Python version and OS
2. Minimal reproducible example
3. Expected vs. actual behavior
4. Full traceback (if applicable)

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
