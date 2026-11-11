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
# Check for lint errors across source, tests, benchmarks, and scripts
ruff check quanta/ tests/ benchmarks/ scripts/

# Auto-fix what's possible
ruff check quanta/ tests/ benchmarks/ scripts/ --fix
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
├── core/          # @circuit, @quantum, 34 built-in gates, custom_gate, types
├── dag/           # Directed acyclic graph (DAGCircuit) IR
├── compiler/      # Optimization, Heavy-Hex/grid routing, ISA translation & Cartan/KAK SU(4) synthesis (synthesis.py)
├── simulator/     # Statevector, MPS, PauliFrame, DensityMatrix, Sparse, Apple MLX, Lindblad & Kraus noise
├── gradients.py   # Parameter-shift (2- & 4-term), reverse-mode adjoint_gradient & Mari-Bromley-Killoran QFIM
├── primitives/    # IBM V2-compatible Sampler & Estimator primitives
├── torch/         # PyTorch QuantumLayer, ContinuousResonantLayer, Daleckii-Krein autograd & Lie algebra
├── qml/           # Scikit-learn compatible Classifier, Regressor, feature maps & ansatz presets
├── layer3/        # Declarative algorithms (Grover, Semiclassical Shor, QAOA, VQE, QSVM, IQAE, Finance, Clustering, Entity Resolution)
├── qec/           # SurfaceCode, 6.6.6 ColorCode, BivariateBicycle qLDPC (BP-OSD), MWPM/Union-Find decoders & magic-state distillation
├── lattice/       # LLL reduction, Babai nearest-plane & lattice cryptography utilities
├── solvers/       # Hybrid lattice-QAOA Closest Vector Problem (CVP) solvers
├── cognitive/     # Quantum-inspired decision arbitration, Noetherian manifold & telemetry dashboard
├── backends/      # Hardware & REST backends (IBM Heron, IonQ Aria, Google Sycamore/Willow, Local)
├── export/        # OpenQASM 2.0/3.0 export (qasm.py) and safe AST import (qasm_import.py)
├── mcp_server.py  # FastMCP AI server exposing 25 quantum tools, 5 resources & 4 prompts
├── cli.py         # Command-line interface (`quanta` & `quanta-mcp`)
├── examples/      # Runnable end-to-end scripts
└── benchmark/     # QASMBench & Benchpress adapters
```

## Making Changes

1. **Fork** the repository and create a feature branch.
2. **Write tests** for your changes in `tests/`.
3. **Run the full test suite** — all tests must pass with ≥80% coverage.
4. **Run lint & docs build** — zero Ruff errors (`ruff check quanta/ tests/ benchmarks/ scripts/`) and strict docs pass (`mkdocs build --strict`).
5. **Submit a PR** against `main`.

## What to Contribute

### Good First Issues

- Add unit and property-based Hypothesis tests for edge-case circuit topologies
- Expand cookbook tutorials and domain-specific examples in `docs/cookbook/`
- Add hardware topology presets and calibration benchmarks

### Feature Ideas

- C++/Rust native acceleration for high-order BP-OSD qLDPC matrix inversion kernels
- Multi-node MPI distributed tensor-network contraction for high-entanglement 2D circuits
- ZX-calculus phase-polynomial simplification passes in `quanta/compiler/passes/`
- OpenQASM 3.0 `defcal` pulse-level schedule export and hardware timing validation

## Reporting Bugs

Please open a [GitHub Issue](https://github.com/ONMARTECH/quanta-sdk/issues) with:

1. Python version and OS
2. Minimal reproducible example
3. Expected vs. actual behavior
4. Full traceback (if applicable)

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
