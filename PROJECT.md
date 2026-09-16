# Project: Quanta SDK Pillar 2 (`quanta.torch`)

## Architecture
Pillar 2 introduces a production-ready, autograd-differentiable PyTorch native quantum extension to Quanta SDK. It unifies standard discrete variational quantum circuits with brain-inspired continuous-time quantum resonance, non-local quantum coherence, and simultaneous non-sequential state evolution grounded in foundational physics (Einstein EPR non-locality, Continuous-Time Quantum Walks) and modern quantum neuroscience (Penrose-Hameroff Orch-OR, Fisher Posner molecules, endogenous biophotons).

```
                     PyTorch User Application / nn.Sequential
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
  quanta.torch.QuantumLayer                        quanta.torch.ContinuousResonantLayer
  (Discrete Variational Circuit)                  (Continuous-Time Graph Hamiltonian Walk)
             │                                                     │
             ▼                                                     ▼
 _QuantumLayerFunction (autograd)                  _ContinuousResonantFunction (autograd)
  - Forward: Batch state simulation                 - Forward: exp(-i H(x,θ) t) |ψ0>
  - Backward: Analytical Parameter-Shift            - Backward: Matrix exp Fréchet / Adjoint ODE
             │                                                     │
             └──────────────────────────┬──────────────────────────┘
                                        ▼
                            quanta.torch.ops & Backends
              CPU (StateVector) ── Apple Silicon (MLX / MPS GPU)
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---|---|---|---|
| 1 | Interdisciplinary Theoretical Whitepaper | Rigorous whitepaper in `docs/theory/continuous_quantum_neural_dynamics.md` synthesizing EPR, CTQW, open quantum systems, Orch-OR, Posner molecules, biophotons, and holistic resonance. | M1 | R1, spec_miner |
| 2 | Analytical Gradient Derivations & Proofs | Formal mathematical proofs for discrete parameter-shift rule, Ehrenfest time derivative, Duhamel/Wilcox Fréchet derivative, and Daleckii-Krein matrix spectral formula. | M1 | R1, spec_miner |
| 3 | Environment Setup & PyTorch Packaging | Add `torch` optional dependency to `pyproject.toml`, fix mypy Python 3.12 compatibility, install PyTorch 2.14 in `.venv`. | M4 | R2, explorer_env |
| 4 | QuantumLayer Module | `quanta.torch.QuantumLayer(nn.Module)` accepting batch inputs $(B, D_{\text{in}})$, variational weights $\theta$, and returning observable expectations $(B, D_{\text{out}})$. | M2 | R2, explorer_codebase |
| 5 | Custom PyTorch Autograd Function for QuantumLayer | `_QuantumLayerFunction(torch.autograd.Function)` executing batch forward circuit simulation and exact backward pass via analytical Parameter-Shift Rule for both weights and inputs. | M2 | R2, spec_miner |
| 6 | Quantum Native Operations & Pauli Utilities | `quanta.torch.ops` implementing Pauli tensor products, batch Hamiltonian construction, expectation value evaluations, and device transfers. | M2 | R2, explorer_codebase |
| 7 | ContinuousResonantLayer Module | `quanta.torch.ContinuousResonantLayer(nn.Module)` with graph Hamiltonian $H(x, \theta) = \sum J_{jk} (\sigma_j^x \sigma_k^x + \sigma_j^y \sigma_k^y) + \sum (h_j + W_j x_j) \sigma_j^z + \sum \omega_j \sigma_j^x$. | M3 | R3, spec_miner |
| 8 | Continuous Unitary State Evolution | Unitary matrix exponential state evolution $|\psi(t)\rangle = \exp(-i H(x, \theta) t) |\psi_0\rangle$ with unconditional norm preservation $\sum_i |a_i|^2 = 1.0 \pm 10^{-6}$. | M3 | R3, spec_miner |
| 9 | Simultaneous Multi-Observable Readout | Concurrent expectation readouts across all qubits ($\langle Z_j \rangle, \langle X_j \rangle$) modeling all-at-once holistic network resonance without sequential collapse. | M3 | R3, R1, spec_miner |
| 10 | Continuous Autograd Backward Engine | Analytical gradients with respect to coupling $J$, bias $h$, projection $W$, drive $\omega$, and time $t$ via matrix exponential spectral decomposition / adjoint state propagation. | M3 | R3, spec_miner |
| 11 | Top-Level Quanta Namespace Export | Expose `quanta.torch` conditionally in `quanta/__init__.py` when PyTorch is installed without breaking non-PyTorch environments. | M4 | R2, AC-PKG-01 |
| 12 | Apple Silicon Metal & MPS Acceleration Bridge | Ensure seamless execution and device transfer across CPU, Apple MPS (`torch.device("mps")`), and Quanta MLX Metal acceleration. | M4 | R2, R4, explorer_env |
| 13 | E2E Testing Suite (Tiers 1-4) | Systematic 4-tier test suite in `tests/test_torch_layer.py` and `tests/test_torch_continuous.py` covering features, boundary values, pairwise combinations, and real-world learning tasks. | M5 / Test Track | R4, test_infra |
| 14 | Adversarial Coverage Hardening (Tier 5) | White-box adversarial test suite probing extreme limits, degenerate spectra, zero couplings, and gradient edge cases reaching $\ge 90\%$ coverage. | M5 | R4, AC-QA-01 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Interdisciplinary Theoretical Foundation | `docs/theory/continuous_quantum_neural_dynamics.md` | None | DONE |
| M2 | PyTorch Native Quantum Layer & Autograd Shift Rule | `quanta/torch/layer.py`, `quanta/torch/ops.py`, `quanta/torch/__init__.py` | M1 | DONE |
| M3 | Continuous Quantum Resonance & Graph Dynamics | `quanta/torch/continuous.py`, Hamiltonian graph walks, multi-observable readout | M2 | DONE |
| M4 | Packaging, Top-level Namespace Integration & Metal Acceleration | `quanta/__init__.py`, `pyproject.toml`, Metal/MPS acceleration bridge | M2, M3 | DONE |
| M5 | E2E Test Suite Pass (Tiers 1-4) & Adversarial Coverage Hardening (Tier 5) | Pass 100% of test suite, 0 ruff errors, 0 mypy errors, $\ge 90\%$ coverage | M1, M2, M3, M4, Test Track | DONE |

## Interface Contracts
### `quanta.torch.QuantumLayer` ↔ PyTorch & Quanta Engine
```python
class QuantumLayer(torch.nn.Module):
    def __init__(
        self,
        num_qubits: int,
        circuit_fn: Callable[..., Any] | CircuitDefinition | str = "hardware_efficient",
        num_layers: int = 1,
        observables: list[str] | list[tuple[str, float]] | None = None,
        diff_method: str = "parameter-shift",
        device: str | torch.device | None = None,
    ) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape:  (B, D_in)
        # Output shape: (B, D_out)
        ...
```

### `quanta.torch.ContinuousResonantLayer` ↔ PyTorch & Network Hamiltonian
```python
class ContinuousResonantLayer(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        coupling_graph: torch.Tensor | list[tuple[int, int]] | str = "complete",
        observable_types: tuple[str, ...] = ("Z", "X"),
        initial_state: str | torch.Tensor = "zero",
        learnable_time: bool = True,
        initial_time: float = 1.0,
        device: str | torch.device | None = None,
    ) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape:  (B, in_features)
        # Output shape: (B, num_nodes * len(observable_types))
        ...
```

## Code Layout
```
quanta/
├── torch/
│   ├── __init__.py                  # Public exports: QuantumLayer, ContinuousResonantLayer
│   ├── layer.py                     # QuantumLayer(nn.Module) & _QuantumLayerFunction(autograd.Function)
│   ├── continuous.py                # ContinuousResonantLayer(nn.Module) & _ContinuousResonantFunction
│   └── ops.py                       # Pauli matrices, Kronecker products, Hamiltonian builder, expectation
tests/
├── test_torch_layer.py              # Discrete QuantumLayer tests (Tiers 1-4)
└── test_torch_continuous.py         # ContinuousResonantLayer tests (Tiers 1-4)
docs/
└── theory/
    └── continuous_quantum_neural_dynamics.md # Interdisciplinary theoretical whitepaper (DONE)
```
