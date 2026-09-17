# Project: CSF / ISF Biophysical Quantum Shielding Framework (`quanta.torch.brain`)

## Architecture
The Cerebrospinal Fluid (CSF / Beyin Omurilik Sıvısı - BOS) and Interstitial Fluid (ISF) framework models the macroscopic biophysical cryostat/shield surrounding the brain's quantum computational minicolumns.
The architecture comprises:
1. **Biophysical Environment (`CSFShieldedEnvironment`)**:
   Differentiable PyTorch module modeling the 5 physical shielding mechanisms:
   - Paramagnetic ion exclusion via BBB/BCSFB ($Fe^{3+}, Cu^{2+}, Mn^{2+} < 0.5\,\mu\text{M}$ vs plasma $25-55\,\mu\text{M}$).
   - Debye electrostatic screening ($\epsilon_r \approx 78.4, I \approx 0.15\,\text{M}, \lambda_D \approx 0.79\,\text{nm}$).
   - Hydrodynamic BPP motional narrowing ($\eta \approx 0.8\,\text{mPa}\cdot\text{s}, \tau_R \approx 86\,\text{ps}, T_2 \sim 10^4\,\text{s}$).
   - Acoustic/phonon damping & buoyant suspension ($1400\,\text{g} \to 50\,\text{g}, 96.5\%$ mass reduction).
   - Glymphatic clearance & entropic bath reset (astrocytic AQP4 convective flush restoring baseline $\Gamma_0$).
   - Clinical state simulations: `normal`, `meningitis`, `hydrocephalus`, `lumbar_puncture_recovery`, `sleep_deprived`, `rem_sleep`.
2. **Shielded Resonant Layer (`CSFShieldedResonantLayer`)**:
   Drop-in PyTorch neural layer combining continuous-time bipartite biomorphic Hamiltonian resonance ($H_{\text{total}} = H_L + H_R + H_{\text{callosum}}$) with CSF-shielded Lindblad dephasing attenuation factor $\kappa_{\text{CSF}} \le 10^{-3}$ and effective dephasing $\Gamma_{\text{eff}} = \kappa_{\text{CSF}} \Gamma_{\text{bare}}$.
3. **Interoperability Hooks**:
   - `BiomorphicResonantBrain`: optional integration of `CSFShieldedEnvironment`.
   - `NoisyHippocampalBuffer`: attenuation of phase diffusion $\sigma_\phi$ and amplitude noise $\sigma_{\text{noise}}$ by $\sqrt{\kappa_{\text{CSF}}}$ and Ebbinghaus memory decay rate by $\kappa_{\text{CSF}}$.
   - `QuantumREMSleep`: triggering nocturnal convective glymphatic flush mode resetting environmental entropy.
4. **Empirical Benchmarking & Figure 9**:
   - `scripts/benchmark_csf_shielding.py` producing 300 DPI 4-panel publication Figure 9 (`docs/paper/figures/fig9_csf_biophysical_shielding.png`) and telemetry in `docs/paper/benchmark_academic_data.json`.
5. **Academic Manuscript Integration**:
   - Section 7 in `docs/paper/biomorphic_quantum_resonance.tex` formalizing Theorem 8, Corollaries 8.1-8.3, and embedding Figure 9.
6. **Testing & Quality Assurance**:
   - `tests/test_csf_shielding.py` validating 5 tiers (32+ tests), autograd gradcheck, MPS device support, 0 ruff errors, 0 mypy errors.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Theory Monograph & Theorem 8 | Dedicated monograph `docs/theory/csf_quantum_shielding.md`, Theorem 8 proof, Corollaries 8.1–8.3, expansion of `quantum_brain_frontiers.md` | M1 | Survey (Spec Miner) |
| 2 | `CSFShieldedEnvironment` Module | Differentiable PyTorch module in `quanta/torch/brain.py` with physical parameters, analytical $\lambda_D, \tau_R, \kappa_{\text{CSF}}$, clinical condition switching, and autograd/MPS support | M2 | Survey (Code Explorer) |
| 3 | `CSFShieldedResonantLayer` Module | Continuous-time Hamiltonian graph resonance coupled with shielded Lindblad dephasing attenuation in `quanta/torch/brain.py` | M2 | Survey (Code Explorer) |
| 4 | Interoperability Hooks & Top-Level Exports | Hooking `CSFShieldedEnvironment` into `BiomorphicResonantBrain`, `NoisyHippocampalBuffer`, and `QuantumREMSleep`; exporting in `quanta/torch/__init__.py` | M2 | Survey (Code Explorer) |
| 5 | Benchmark Script & Publication Fig 9 | `scripts/benchmark_csf_shielding.py` (< 5s runtime, deterministic seed) generating 300 DPI 4-panel Figure 9 and updating `benchmark_academic_data.json` | M3 | Survey (Bench Explorer) |
| 6 | Academic LaTeX Manuscript Section 7 | Authoring Section 7 in `docs/paper/biomorphic_quantum_resonance.tex`, embedding Figure 9 with comprehensive caption, and integrating Theorem 8 & medical citations | M4 | Survey (Spec Miner) |
| 7 | Opaque-Box E2E Test Suite & Test Infra | Comprehensive test suite in `tests/test_csf_shielding.py` covering Tiers 1–4 (>30 test cases) and publishing `TEST_READY.md` | M5 | Survey (Bench Explorer) |
| 8 | Adversarial Coverage Hardening & Static Analysis | White-box stress-testing (Tier 5), MPS/CPU validation, numerical gradcheck, 0 ruff lint errors, 0 mypy type errors | M5 | Survey (Bench Explorer) |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Theory Monograph & Theorem 8 | `docs/theory/csf_quantum_shielding.md`, `docs/theory/quantum_brain_frontiers.md` | none | DONE |
| M2 | PyTorch Architecture Implementation | `quanta/torch/brain.py`, `quanta/torch/__init__.py` | M1 | DONE |
| M3 | Empirical Benchmarking & Publication Fig 9 | `scripts/benchmark_csf_shielding.py`, `docs/paper/figures/fig9_csf_biophysical_shielding.png`, `docs/paper/benchmark_academic_data.json` | M2 | DONE |
| M4 | Academic Manuscript LaTeX Section 7 | `docs/paper/biomorphic_quantum_resonance.tex` | M1, M3 | DONE |
| M5 | E2E Testing, Adversarial Hardening & Quality Gate | `tests/test_csf_shielding.py`, ruff, mypy, pytest | M2, M3 | DONE |

## Interface Contracts
### `CSFShieldedEnvironment`
- Constructor:
  ```python
  CSFShieldedEnvironment(
      ionic_strength: float = 0.155,       # M
      dielectric_constant: float = 78.5,   # dimensionless
      viscosity: float = 0.80,             # mPa*s
      paramagnetic_concentration: float = 0.40, # uM
      glymphatic_clearance_rate: float = 0.20,  # 1/h
      temperature: float = 310.15,         # K (37 C)
      learnable_params: bool = False,
      device: Optional[Union[str, torch.device]] = None,
      dtype: Optional[torch.dtype] = None,
  )
  ```
- Methods:
  - `compute_debye_length() -> torch.Tensor` ($\lambda_D = \sqrt{\frac{\epsilon_0 \epsilon_r k_B T}{2 N_A e^2 I}}$)
  - `compute_rotational_correlation_time() -> torch.Tensor` ($\tau_R = \frac{4 \pi \eta r_H^3}{3 k_B T}$)
  - `compute_attenuation_factor() -> torch.Tensor` ($\kappa_{\text{CSF}} = \kappa_{\text{elec}} \cdot \kappa_{\text{motional}} \cdot \kappa_{\text{para}} \cdot \kappa_{\text{glym}}$)
  - `apply_shielding(lindblad_gamma: Union[float, torch.Tensor]) -> torch.Tensor` ($\Gamma_{\text{eff}} = \kappa_{\text{CSF}} \Gamma_{\text{bare}}$)
  - `simulate_clinical_condition(condition_name: str) -> None` (`normal`, `meningitis`, `hydrocephalus`, `lumbar_puncture_recovery`, `sleep_deprived`, `rem_sleep`)

### `CSFShieldedResonantLayer`
- Constructor:
  ```python
  CSFShieldedResonantLayer(
      n_qubits: int = 4,
      dim_in: int = 4,
      dim_out: int = 4,
      interaction_time: float = 1.0,
      env: Optional[CSFShieldedEnvironment] = None,
      bare_dephasing_rate: float = 1.30,
      device: Optional[Union[str, torch.device]] = None,
      dtype: Optional[torch.dtype] = None,
  )
  ```
- Forward:
  `forward(x: torch.Tensor) -> torch.Tensor` where $x \in \mathbb{R}^{B \times \text{dim\_in}}$, returns expectation values $y \in [-1, 1]^{B \times \text{dim\_out}}$ damped by $\exp(-\Gamma_{\text{eff}} t / 2)$.

### `NoisyHippocampalBuffer` Integration
- `attach_csf_environment(env: CSFShieldedEnvironment) -> None`:
  Attenuates stochastic Lindblad phase diffusion and Gaussian noise by $\sqrt{\kappa_{\text{CSF}}}$ and Ebbinghaus memory decay rate by $\kappa_{\text{CSF}}$.

### `QuantumREMSleep` Integration
- `trigger_glymphatic_reset(env: CSFShieldedEnvironment) -> None`:
  Applies nocturnal convective AQP4 flush during sleep annealing, resetting environmental entropy and purging noise accumulation.

## Code Layout
- `quanta/torch/brain.py`: Core biophysical classes (`CSFShieldedEnvironment`, `CSFShieldedResonantLayer`, integration methods)
- `quanta/torch/__init__.py`: Public exports
- `docs/theory/csf_quantum_shielding.md`: Comprehensive theory monograph
- `docs/theory/quantum_brain_frontiers.md`: Monograph expansion with Theorem 8
- `scripts/benchmark_csf_shielding.py`: Publication benchmark & plotting script
- `docs/paper/figures/fig9_csf_biophysical_shielding.png`: 300 DPI 4-panel publication figure
- `docs/paper/benchmark_academic_data.json`: Benchmark telemetry data
- `docs/paper/biomorphic_quantum_resonance.tex`: Academic LaTeX paper Section 7
- `tests/test_csf_shielding.py`: Test suite across Tiers 1–5
