# Project: Interdisciplinary Biomorphic Memory & Multi-Perspective Cognitive Panel

## Architecture
Quanta SDK Cognitive Architecture enhancement integrating biomorphic quantum-inspired memory, fuzzy-trace cognitive consolidation, and interdisciplinary decision evaluation:
- **Biomorphic Memory Kinetics**: `FastBiomorphicMemory` (stdlib) and `CognitiveMemoryManager` (PyTorch) with Lindblad-Ebbinghaus dephasing, CSF dielectric attenuation ($\kappa_{\text{CSF}} = 1.6 \times 10^{-4}$), and calibrated biological time scaling ($\Delta t = 0.2$ for tool executions).
- **Fuzzy-Trace Cognitive Consolidation**: Brainerd & Reyna Fuzzy-Trace Theory (FTT) dividing memory into dephasing verbatim traces and persistent semantic gist traces. Pre-pruning synthesis before 70% threshold crystallizes actionable architectural/strategic resolutions into `category="semantic_gist"`, $S \ge 1.8$, $F=0.9998$, `is_core_anchor=True`.
- **Interdisciplinary Cognitive Panel**: Multi-criteria evaluation panel spanning Neurobiology (synaptic homeostasis, metabolic energy, sleep-wake consolidation), Psychiatry (anti-rumination bounds, cognitive flexibility vs perseveration, threat appraisal), and Sociology & Human Alignment (social framing, user cognitive fatigue reduction, collective intelligence coordination).
- **Quantum Decision Arbitration**: 6-qubit Hilbert space decision DAG traversal (`QuantumDecisionArbiter`) integrating 4D consequence vectors with CognitivePanelScore penalties and dynamic Zeno pinning.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Calibrated Micro-Step Tool Dephasing | Scale sub-turn tool dephasing to $\Delta t = 0.2$ in `quanta_subconscious_hook.py` and `FastBiomorphicMemory`, preserving transient decisions across $\ge 20$ tool steps | M1 | Survey R1 |
| 2 | Proportional Biological Age Evolution | Update `FastBiomorphicMemory.step(dt)` to advance age proportionally by `dt` instead of rigid integer increments | M1 | Survey R1 |
| 3 | PreInvocation Sub-Turn Decoupling | Decouple intermediate tool continuation sub-turns from user conversational turns in `quanta_subconscious_hook.py` | M1 | Survey R1 |
| 4 | Fuzzy-Trace Actionable Resolution Classifier | Deterministic heuristic classifying architectural/strategic resolutions and filtering procedural tool noise | M2 | Survey R2 |
| 5 | Semantic Gist Distillation Engine | Distill actionable resolutions into compact semantic gists (`category="semantic_gist"`, $S \ge 1.8$, $F=0.9998$, `is_core_anchor=True`) before 70% eviction | M2 | Survey R2 |
| 6 | Gist Integration in CognitiveMemoryManager | Add `record_semantic_gist`, `recall_semantic_gists`, and pre-pruning consolidation in `quanta/cognitive/memory.py` | M2 | Survey R2 |
| 7 | Gist Integration in FastBiomorphicMemory | Add `record_semantic_gist`, `recall_semantic_gists`, and pre-pruning consolidation in `scripts/hooks/quanta_subconscious_hook.py` | M2 | Survey R2 |
| 8 | SWR Replay Gist Logging | Display active semantic gists (`🧠 Özüt`) and crystallization events (`✨ Kristalleşen Özüt`) in SWR Replay hook messages | M2 | Survey R2 |
| 9 | Atomic State Mirror Persistence | Ensure `quanta_cognitive_state.json` atomic POSIX writers preserve `"semantic_gist"` records across workspace mirrors | M2 | Survey R2 |
| 10 | Neurobiological Evaluation Module | Dataclass evaluating synaptic saturation (SHY), metabolic energy budgeting, and sleep-wake consolidation affinity | M3 | Survey R3 |
| 11 | Psychiatric Evaluation Module | Dataclass evaluating anti-rumination risk, perseveration penalty vs cognitive flexibility, and threat appraisal | M3 | Survey R3 |
| 12 | Sociological Evaluation Module | Dataclass evaluating social context framing, user cognitive fatigue reduction, and collective coordination | M3 | Survey R3 |
| 13 | Extended ConsequenceVector | Incorporate `cognitive_panel: CognitivePanelScore` into `ConsequenceVector` with backwards-compatible `panel_weight` parameter | M3 | Survey R3 |
| 14 | TheoryOfMind Interdisciplinary Lens | Add `evaluate_cognitive_panel()`, `analyze_interdisciplinary()`, and interdisciplinary DreamSeeds to `TheoryOfMindAnalyzer` | M3 | Survey R3 |
| 15 | QuantumDecisionArbiter Integration | Integrate panel penalties into DAG rollout microglial pruning and multi-branch decision arbitration | M3 | Survey R3 |
| 16 | Comprehensive Interdisciplinary Test Suite | Implement comprehensive tests in `tests/test_cognitive_interdisciplinary.py` across 6 test classes | M4 | Survey R4 |
| 17 | Code Quality & Regression Invariants | Enforce Ruff (0 errors), Mypy (0 errors on target modules), Google-style docstrings (100-char limit), and 100% pass on 559+ tests | M4 | Survey R4 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Calibrated Biological Time & Calmed Decay Kinetics | Features 1, 2, 3 | None | DONE |
| M2 | Fuzzy-Trace Semantic Gist Extraction & Consolidation | Features 4, 5, 6, 7, 8, 9 | M1 | DONE |
| M3 | Multi-Disciplinary Cognitive Evaluation Panel | Features 10, 11, 12, 13, 14, 15 | None | IN_PROGRESS |
| M4 | Architectural Invariants, Quality Verification & Tests | Features 16, 17 | M1, M2, M3 | PLANNED |

## Interface Contracts

### M1: Memory Kinetics Contract
- `FastBiomorphicMemory.step(self, dt: float = 1.0) -> None`:
  `e["age"] += dt` for all engrams; Lindblad decay factor `math.exp(-eff_gamma * dt)`.
- `handle_post_tool_use(...) -> None`:
  Invokes `mem.step(dt=0.2)` on tool execution. `turn_count` is NOT incremented.
- `main()` in `scripts/hooks/quanta_subconscious_hook.py`:
  If non-debounced user conversational turn: `turn_count += 1`, `mem.step(dt=1.0)`. If continuation sub-turn after tool: do not double-step.
- Transient retention invariant: $S = 0.5$ engrams maintain $F \ge 0.885$ after 20 tool steps ($\Delta t_{\text{total}} = 4.0$).

### M2: Fuzzy-Trace Semantic Gist Contract
- `is_actionable_resolution(content: str, key: str = "") -> bool`:
  Returns True if `len(content) >= 20`, does not match procedural tool noise, and contains strategic/architectural keywords.
- `distill_semantic_gist(content: str, key: str = "") -> tuple[str, str]`:
  Strips preambles/parentheticals, produces `(gist_key, distilled_content)` with `gist_key` starting with `"gist_"`.
- `record_semantic_gist(key: str, content: str, salience: float = 1.85, is_core_anchor: bool = True) -> Any`:
  Records engram with `category="semantic_gist"`, $S \ge 1.80$, `is_core_anchor=True`, $F=0.9998$.
- `prune_obsolete(...)`:
  When transient engram decays below $F < 0.70$ and is actionable, crystallizes gist into core memory before detail eviction.
- SWR Replay message:
  Includes `🧠 Özüt: ...` for active gists and `✨ Kristalleşen Özüt: X karar` when new gists crystallize.

### M3: Cognitive Panel Contract
- `NeurobiologicalEvaluation(synaptic_saturation: float, energy_expenditure: float, sleep_consolidation_affinity: float)`:
  `aggregate_cost(weights=(0.40, 0.35, 0.25)) -> float` in $[0.0, 1.0]$.
- `PsychiatricEvaluation(rumination_risk: float, perseveration_penalty: float, threat_distortion: float)`:
  `aggregate_cost(weights=(0.40, 0.35, 0.25)) -> float` in $[0.0, 1.0]$.
- `SociologicalEvaluation(social_misalignment: float, user_fatigue_impact: float, coordination_friction: float)`:
  `aggregate_cost(weights=(0.35, 0.40, 0.25)) -> float` in $[0.0, 1.0]$.
- `CognitivePanelScore(neurobiology, psychiatry, sociology)`:
  `aggregate_penalty(discipline_weights=(0.30, 0.35, 0.35)) -> float` in $[0.0, 1.0]$.
- `ConsequenceVector`:
  `weighted_cost(weights=(0.25, 0.35, 0.25, 0.15), panel_weight=0.0) -> float`.
  When `panel_weight == 0.0`, returns exact 4D cost. When `panel_weight > 0.0`, interpolates $(1 - w_p) C_{\text{4D}} + w_p C_{\text{panel}}$.
- `TheoryOfMindAnalyzer`:
  `evaluate_cognitive_panel(messages, project_state=None, context=None) -> CognitivePanelScore`.
  `analyze_interdisciplinary(messages, project_state=None) -> tuple[float, list[DreamSeed], CognitivePanelScore]`.

### M4: Verification Contract
- `tests/test_cognitive_interdisciplinary.py` contains 6 test classes:
  1. `TestNeurobiologicalEvaluation`
  2. `TestPsychiatricEvaluation`
  3. `TestSociologicalEvaluation`
  4. `TestMultiCriteriaCognitivePanel`
  5. `TestInterdisciplinaryArbitration`
  6. `TestInterdisciplinaryInvariantsAndRegressions`
- Verification commands:
  - `.venv/bin/ruff check quanta/ scripts/hooks/` $\to$ 0 errors
  - `.venv/bin/mypy quanta/cognitive/arbiter.py quanta/cognitive/tom_analyzer.py quanta/cognitive/memory.py --ignore-missing-imports` $\to$ 0 errors
  - `.venv/bin/pytest tests/test_cognitive_interdisciplinary.py` $\to$ 100% pass
  - `.venv/bin/pytest tests/test_cognitive_*.py tests/test_subconscious_*.py` $\to$ 100% pass (0 regressions on existing 559+ tests).

## Code Layout
- `scripts/hooks/quanta_subconscious_hook.py`: Runtime hook for Antigravity, `FastBiomorphicMemory`, SWR Replay, micro-step scaling, and gist crystallization.
- `quanta/cognitive/memory.py`: PyTorch `CognitiveMemoryManager`, Lindblad phase diffusion, fuzzy-trace semantic gist extraction, and synaptic pruning.
- `quanta/cognitive/arbiter.py`: `QuantumDecisionArbiter`, `ConsequenceVector`, `CognitivePanelScore`, `NeurobiologicalEvaluation`, `PsychiatricEvaluation`, `SociologicalEvaluation`.
- `quanta/cognitive/tom_analyzer.py`: `TheoryOfMindAnalyzer`, interdisciplinary evaluation, conversational cadence and fatigue analysis.
- `quanta/cognitive/consolidation.py`: `SubconsciousConsolidator` (SHY downscaling and microglial sleep consolidation).
- `tests/test_cognitive_interdisciplinary.py`: Comprehensive test suite for all interdisciplinary and fuzzy-trace functionality.
