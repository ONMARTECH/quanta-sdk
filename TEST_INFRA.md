# E2E Test Infrastructure: CSF / ISF Biophysical Quantum Shielding Framework (`quanta.torch.brain`)

## 1. Test Philosophy & Engineering Principles

The test suite for the **Cerebrospinal Fluid (CSF / BOS) & Interstitial Fluid (ISF) Biophysical Quantum Shielding Framework** (Theorem 8) validates the biological cryostat and electromagnetic shielding enclosure protecting the brain's quantum computational minicolumns.

The test infrastructure adheres to six core engineering principles:
1. **Opaque-Box & Requirement-Driven**: Tests are derived strictly from first-principles biophysics, mathematical proofs (`docs/theory/csf_quantum_shielding.md`), and interface specifications (`PROJECT.md` / `ORIGINAL_REQUEST.md`), completely decoupled from internal implementation artifacts.
2. **Analytical Ground Truth Grounding**: Every numerical tolerance is derived from exact physical constants ($\epsilon_0, k_B, e, N_A$) and closed-form solutions (Poisson-Boltzmann screening, Stokes-Einstein-Debye Brownian tumbling, and Bloembergen-Purcell-Pound relaxation), avoiding heuristic thresholds.
3. **Physical Boundary & Constraint Enforcement**: Strict boundary rejection ($I \le 0, \epsilon_r \le 0, \eta \le 0, [\text{Para}] < 0, G < 0, T \le 0$) guarantees mathematical stability and prevents non-physical simulation states.
4. **Multi-Scale Clinical Pathology Verification**: Validates state transitions between healthy homeostasis and neuropathological breakdown (`normal`, `meningitis`, `hydrocephalus`, `lumbar_puncture_recovery`, `sleep_deprived`, `rem_sleep`), verifying both parameter shifts and functional dephasing attenuation.
5. **Autograd Differentiability & Inverse Modeling**: Rigorously verifies that the biophysical shielding equation $\kappa_{\text{CSF}}(\lambda_D, \eta, [\text{Para}], G)$ maintains a continuous computational graph with finite, non-zero gradients for learning and system identification.
6. **Device Portability & Hardware Guarding**: Enforces strict execution across CPU and Apple Silicon Metal (MPS), ensuring proper 32-bit float compliance on MPS and defensive rejection of unsupported 64-bit precision.

---

## 2. 5-Tier Verification Hierarchy

The testing architecture is organized into five complementary tiers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            5-TIER VERIFICATION MATRIX                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 1: Parameter Validation, Range Checking & Constructors                 │
│         - Physiological defaults, buffer/parameter allocation               │
│         - Negative/invalid input rejection, baseline reset, type resolution │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 2: Analytical Invariants & Theorem 8 Ground Truth                      │
│         - Debye length lambda_D approx 0.79 nm (Poisson-Boltzmann)          │
│         - BPP rotational correlation time tau_R approx 86.4 ps              │
│         - Extreme motional narrowing criterion omega_0 * tau_R << 10^-4     │
│         - Theorem 8 attenuation bound kappa_CSF <= 10^-3, G_CSF >= 10^3     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 3: Clinical Neuropathology Simulation & Mode Switching                 │
│         - State transitions: normal, meningitis, hydrocephalus,             │
│           lumbar_puncture_recovery, sleep_deprived, rem_sleep               │
│         - Quantitative parameter shifts and Lindblad dephasing modulation   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 4: PyTorch Layer Dynamics, Shapes & Property Preservation              │
│         - Tensor shapes: 1D unbatched, 2D batched, 3D sequential, empty B=0 │
│         - Unitary statevector norm preservation ||psi(t)||^2 = 1.0          │
│         - Readout expectation bounds [-1.0, 1.0], return_dict compatibility │
│         - Interoperability with BiomorphicBrain, Hippocampus, REMSleep      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 5: Autograd Backprop, Device Transfer (CPU/MPS) & Hardening            │
│         - Differentiable backpropagation through input and weights          │
│         - Trainable environment parameters with finite gradients            │
│         - Analytical vs. numerical gradcheck (||grad_auto - grad_fd||<10^-4)│
│         - Optimization in nn.Sequential with Adam                           │
│         - Apple Silicon Metal (MPS) float32 execution & float64 guard       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Feature Inventory & Test Allocation Matrix

| # | Feature | Target Test File | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Tier 5 | Total Tests |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| F1 | `CSFShieldedEnvironment` Initialization & Validation | `tests/test_csf_shielding.py` | 5 | — | — | — | — | 5 |
| F2 | Theorem 8 Analytical Invariants ($\lambda_D, \tau_R, \kappa, \mathcal{G}$) | `tests/test_csf_shielding.py` | — | 7 | — | — | — | 7 |
| F3 | Clinical Neuropathology Transitions & Dephasing Shifts | `tests/test_csf_shielding.py` | — | — | 6 | — | — | 6 |
| F4 | `CSFShieldedResonantLayer` Tensor Shapes & Empty Batch | `tests/test_csf_shielding.py` | 2 | — | — | 3 | — | 5 |
| F5 | Norm Preservation & Quantum Expectation Bounds | `tests/test_csf_shielding.py` | — | — | — | 2 | — | 2 |
| F6 | Ecosystem Interoperability (Brain, Hippocampus, REM Sleep) | `tests/test_csf_shielding.py` | — | — | — | 2 | 1 | 3 |
| F7 | Autograd Backprop, Numerical Gradcheck & Optimizer | `tests/test_csf_shielding.py` | — | — | — | — | 3 | 3 |
| F8 | Hardware Device Execution (CPU, Apple Silicon Metal MPS) | `tests/test_csf_shielding.py` | — | — | — | — | 2 | 2 |
| **Totals** | **Complete CSF Shielding Framework** | `tests/test_csf_shielding.py` | **7** | **7** | **6** | **7** | **6** | **33 Tests** |

---

## 4. Test Runner & Verification Instructions

### 4.1 Isolated Test Execution
To run the dedicated CSF Shielding test suite without coverage overhead:
```bash
unset VIRTUAL_ENV && uv run pytest tests/test_csf_shielding.py -v --no-cov
```

### 4.2 Targeted Tier Execution
To execute specific verification tiers using expression matching:
```bash
# Tier 1: Constructors & Range Checks
unset VIRTUAL_ENV && uv run pytest tests/test_csf_shielding.py -k "init or validation or constructor" -v --no-cov

# Tier 2: Theorem 8 Analytical Bounds
unset VIRTUAL_ENV && uv run pytest tests/test_csf_shielding.py -k "debye or bpp or theorem_8" -v --no-cov

# Tier 3: Clinical Neuropathology
unset VIRTUAL_ENV && uv run pytest tests/test_csf_shielding.py -k "clinical" -v --no-cov

# Tier 4: Shapes & Dynamics
unset VIRTUAL_ENV && uv run pytest tests/test_csf_shielding.py -k "shape or empty or norm or bounds" -v --no-cov

# Tier 5: Autograd, Gradcheck & MPS
unset VIRTUAL_ENV && uv run pytest tests/test_csf_shielding.py -k "autograd or gradcheck or optimizer or mps" -v --no-cov
```

### 4.3 Full Neural Brain Regression Suite
To run all quantum brain and resonance test suites:
```bash
unset VIRTUAL_ENV && uv run pytest tests/test_torch_brain.py tests/test_torch_continuous.py tests/test_csf_shielding.py -v --no-cov
```

### 4.4 Static Analysis & Type Checking
```bash
# Linting (0 errors required)
unset VIRTUAL_ENV && uv run ruff check quanta/ tests/ scripts/

# Type Checking (strict type annotations required)
unset VIRTUAL_ENV && uv run mypy quanta/torch tests/test_csf_shielding.py
```

---

## 5. Coverage Goals & Quality Gate Criteria

1. **Test Pass Rate**: Exactly 100% of authored tests must pass cleanly.
2. **Branch & Statement Coverage**: Minimum 90% statement coverage across `quanta/torch/brain.py` and new CSF shielding logic.
3. **Execution Latency**: Complete test suite `tests/test_csf_shielding.py` must execute in $< 2.0\,\text{s}$ on standard CPU hardware.
4. **Code Cleanliness**: Zero linter diagnostics under `ruff check` (rule sets `E`, `F`, `W`, `I`, `UP`, `B`, `SIM`).
5. **Static Typing**: Zero mypy errors under Python 3.12 with `disallow_untyped_defs = true`.
