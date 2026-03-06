# Quanta SDK

A clean, modular quantum computing SDK for Python.

## Overview

Quanta provides a 3-layer architecture for quantum computing:

- **Layer 3** (Declarative): `search()`, `optimize()`, `vqe()`, `factor()`, `resolve()` -- use quantum without knowing gates
- **Layer 2** (Circuit): `@circuit`, H, CX, RZ, `measure()`, `run()` -- standard circuit programming
- **Layer 1** (Physical): DAG, compiler, routing, simulator, QEC, QASM -- hardware optimization

## Quick Start

```python
from quanta import circuit, H, CX, measure, run

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell, shots=1024)
print(result.summary())
```

## Examples

11 demo scripts covering Bell states, GHZ, teleportation, Deutsch-Jozsa, Grover, VQE, portfolio optimization, QKD, Shor, QSVM, and entity resolution.

```bash
python -m quanta.examples.01_bell_state
python -m quanta.examples.11_entity_resolution
```

## Installation

```bash
git clone https://github.com/ONMARTECH/quanta-sdk.git
cd quanta-sdk
pip install -e ".[dev]"
pytest
```

## Documentation

See the `docs/` directory for detailed documentation:

- [Architecture](ARCHITECTURE_EN.md)
- [Features](FEATURES_EN.md)
- [Comparison](COMPARISON_EN.md)
- [Installation](INSTALL_TR.md)

## Author

Abdullah Enes SARI -- ONMARTECH

info@onmartech.com

## License

Apache License 2.0
