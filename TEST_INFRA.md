# E2E Test Infra: Quanta SDK Pillar 2 (`quanta.torch`)

## Test Philosophy
- **Opaque-box & Requirement-driven**: Derived directly from `ORIGINAL_REQUEST.md` and user specifications, independent of implementation internals.
- **Methodology**: 4-Tier verification hierarchy:
  1. Tier 1: Feature Coverage (≥5 tests per feature).
  2. Tier 2: Boundary & Corner Cases (≥5 tests per feature: empty batch, extreme times $t \to 0, t \to \infty$, degenerate spectra, disconnected graphs).
  3. Tier 3: Cross-Feature Combinations (pairwise interactions: hybrid classical-quantum networks, multi-layer cascades, device migrations CPU ↔ MPS).
  4. Tier 4: Real-World Application Scenarios (XOR/parity non-linear classification, dynamic continuous regression, quantum neural resonance).
  5. Tier 5 (Hardening): Adversarial stress testing, fuzzing, and white-box gap elimination.

## Feature Inventory & Target Test Allocations
| # | Feature | Target Test File | Tier 1 (≥5) | Tier 2 (≥5) | Tier 3 (Pairwise) | Tier 4 (Workloads) |
|---|---|---|:---:|:---:|:---:|:---:|
| F1 | `QuantumLayer` Construction & Forward | `tests/test_torch_layer.py` | 5 | 5 | ✓ | ✓ |
| F2 | Analytical Parameter-Shift Weight Gradients | `tests/test_torch_layer.py` | 5 | 5 | ✓ | ✓ |
| F3 | Analytical Parameter-Shift Input Gradients | `tests/test_torch_layer.py` | 5 | 5 | ✓ | ✓ |
| F4 | `ContinuousResonantLayer` Forward & Readout | `tests/test_torch_continuous.py` | 5 | 5 | ✓ | ✓ |
| F5 | Unitary Norm Preservation ($\sum \|a_i\|^2 = 1.0 \pm 10^{-6}$) | `tests/test_torch_continuous.py` | 5 | 5 | ✓ | ✓ |
| F6 | Hamiltonian Parameter Autograd Gradients | `tests/test_torch_continuous.py` | 5 | 5 | ✓ | ✓ |
| F7 | Interaction Time Parameter Gradient ($t$) | `tests/test_torch_continuous.py` | 5 | 5 | ✓ | ✓ |
| F8 | Hybrid Optimization & Convergence | Both | 5 | 5 | ✓ | ✓ |
| F9 | Device Compatibility (CPU / Apple MPS / MLX) | Both | 5 | 5 | ✓ | ✓ |
| F10 | Top-level Namespace Export & Importability | Both | 5 | 5 | ✓ | ✓ |

## Test Architecture
- **Test Runner**: `unset VIRTUAL_ENV && uv run pytest tests/test_torch_layer.py tests/test_torch_continuous.py`
- **Lint Runner**: `unset VIRTUAL_ENV && uv run ruff check quanta/ tests/`
- **Mypy Runner**: `unset VIRTUAL_ENV && uv run mypy --python-version 3.12 quanta/torch`
- **Directory Layout**:
  - `tests/test_torch_layer.py`: Comprehensive test suite for discrete `QuantumLayer`.
  - `tests/test_torch_continuous.py`: Comprehensive test suite for `ContinuousResonantLayer`.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|---|---|---|
| 1 | Non-linear XOR / Parity Binary Classification with `QuantumLayer` | F1, F2, F3, F8 | Medium |
| 2 | Continuous Resonance Graph Walk Classification with `ContinuousResonantLayer` | F4, F5, F6, F7, F8 | High |
| 3 | Hybrid Neural Network (`nn.Linear` $\to$ `ContinuousResonantLayer` $\to$ `QuantumLayer` $\to$ `nn.Linear`) | F1, F3, F4, F8 | High |
| 4 | Ballistic Quantum Walk Verification on Ring & Line Graphs ($|\psi(t)\rangle$ spread) | F4, F5, F7 | Medium |
| 5 | Simultaneous Multi-Observable Entangled State Classification | F4, F6, F8 | High |

## Coverage Thresholds
- Tier 1: ≥5 per feature
- Tier 2: ≥5 per feature (where boundaries exist)
- Tier 3: Pairwise coverage of major feature interactions
- Tier 4: ≥5 realistic application scenarios
- Tier 5: Adversarial hardening achieving $\ge 90\%$ test coverage on `quanta/torch`.
