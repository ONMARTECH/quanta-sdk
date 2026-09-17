# Quanta Cognitive Architecture: Subconscious Mind-Wandering & Anticipatory Prospection Engine
# End-to-End (E2E) Test Infrastructure & Verification Specification

- **Project**: Quanta SDK (`quanta.cognitive`) — Pillar 3
- **Author**: `teamwork_preview_test_writer_test_track_1` (QA / Specialist Archetype)
- **Status**: Published & Authoritative
- **Target Version**: Quanta SDK v1.1.0 (Darwin / Apple Silicon / Linux / POSIX)
- **Scope**: Tiers 1–4 Requirement-Driven Opaque-Box Test Suites

---

## 1. Executive Summary & Verification Strategy

The Quanta Subconscious Mind-Wandering and Anticipatory Prospection Engine introduces an autonomous, unprompted background cognitive cycle executing on Apple Silicon Efficiency cores (`QOS_CLASS_BACKGROUND`). To ensure maximum scientific validity, biomorphic fidelity, memory isolation, and system safety, this document formalizes the complete End-to-End (E2E) Test Infrastructure.

Testing follows rigorous black-box / opaque-box software testing methodologies:
1. **Category-Partition Method (Ostrand & Balcer)**: Systematic decomposition of input domains and environment states into disjoint, equivalence partitions.
2. **Boundary Value Analysis (BVA)**: Exhaustive testing of mathematical inflection points, asymptotic limits, and resource constraints ($\lambda(t)$, $\mathcal{F}_{\text{fatigue}} \to 1$, $T_{\text{idle\_min}}$, token ceilings $\le 2500$, turn limits $\le 5$, cosine similarity $> 0.95$).
3. **Pairwise Combinatorial Testing (All-Pairs)**: Covering orthogonal environment configurations (OS, thermal state, battery level, CPU load, ToM urgency).
4. **Statistical Goodness-of-Fit Verification**: Validation of non-homogeneous Poisson point processes via the **Time-Rescaling Theorem** and Kolmogorov-Smirnov test ($p > 0.05$).
5. **Real-World Fault Tolerance & Preemption**: Empirical verification of $< 20\,\text{ms}$ preemption latency when user interactive turns or hardware interrupts occur.

---

## 2. Test Suite Architecture Across Tiers 1–4

```
+---------------------------------------------------------------------------------------+
| TIER 4: END-TO-END AUTONOMOUS CYCLE (`tests/test_e2e_subconscious.py`)                |
| Idle -> Poisson Pulse -> ToM Seed -> Headless Dialectic -> SWR Engram -> Hook Output  |
+-------------------------------------------+-------------------------------------------+
                                            |
         +----------------------------------+----------------------------------+
         |                                                                     |
         v                                                                     v
+---------------------------------------------------+ +---------------------------------------------------+
| TIER 3B: ISOLATED HEADLESS DIALECTIC              | | TIER 3A: THEORY OF MIND (ToM) ANALYZER            |
| (`tests/test_mind_wander.py`)                     | | (`tests/test_tom_analyzer.py`)                    |
| - DMN Generative Dreamer vs Zeno Arbiter          | | - Cadence tracking & hesitation heuristics        |
| - Anti-rumination cosine check (> 0.95)           | | - Milestone urgency detection                     |
| - Synthetic noradrenaline reset & abort           | | - Speculative DreamSeed synthesis                 |
| - Hard caps (turns <= 5, tokens <= 2500)          | |                                                   |
+---------------------------------------------------+ +---------------------------------------------------+
         |                                                                     |
         +----------------------------------+----------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| TIER 2: STOCHASTIC POISSON SPINDLE SCHEDULER (`tests/test_poisson_trigger.py`)        |
| - Hazard rate lambda(t) computation & sigmoid relaxation                              |
| - Refractory period gating & dead-time enforcement                                    |
| - Exponential inter-arrival renewal sampling                                          |
| - Kolmogorov-Smirnov statistical test via Time-Rescaling Theorem                      |
+---------------------------------------------------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| TIER 1: DARWIN MACH KERNEL QoS & QUIESCENCE (`tests/test_darwin_idle.py`)             |
| - Mach thread QoS (QOS_CLASS_BACKGROUND = 0x09) & IOPOL_THROTTLE                      |
| - Host CPU load statistics & instantaneous idle ratio calculation                     |
| - Apple Silicon thermal pressure telemetry (Nominal, Fair, Serious, Critical)         |
| - IOKit battery / power source telemetry & cross-platform fallbacks                   |
+---------------------------------------------------------------------------------------+
```

---

## 3. Test File Inventory & Responsibility Matrix

| Test File | Test Tier | Target Modules | Primary Contracts & Invariants Tested |
|---|---|---|---|
| `tests/test_darwin_idle.py` | Tier 1 | `quanta.cognitive.darwin_idle` | `set_background_qos()`, `get_cpu_quiescence()`, `get_thermal_state()`, `is_on_battery()`, `is_system_idle()`, ctypes Mach bindings, error code handling, Linux/Windows graceful fallbacks. |
| `tests/test_poisson_trigger.py` | Tier 2 | `quanta.cognitive.poisson_trigger` | `PoissonSpindleTrigger.compute_rate()`, `sample_next_interval()`, `should_trigger()`, refractory gating, exponential renewal distribution, Kolmogorov-Smirnov statistical test ($D_n < D_{\text{crit}}$). |
| `tests/test_tom_analyzer.py` | Tier 3A | `quanta.cognitive.tom_analyzer` | `TheoryOfMindAnalyzer.analyze_conversation()`, `DreamSeed`, conversational cadence shifts, lexical hesitation heuristics ("maybe", "..."), milestone urgency ("deploy", "deadline"), urgency multiplier $\mathcal{S}_{\text{ToM}} \in [0.2, 5.0]$. |
| `tests/test_mind_wander.py` | Tier 3B | `quanta.cognitive.mind_wander` | `MindWanderEngine.execute_dream_cycle()`, `DreamInsight`, headless dialectic (DMN Dreamer vs Zeno Critic), zero prompt leakage, anti-rumination cosine similarity $> 0.95$, synthetic noradrenaline reset, token cap $\le 2500$, turn cap $\le 5$, preemption interrupt. |
| `tests/test_e2e_subconscious.py` | Tier 4 | Complete Subconscious Pipeline | Full pipeline integration: user idle $\to$ Poisson trigger $\to$ ToM seed $\to$ Antigravity headless dialectic $\to$ SWR consolidation into `quanta_cognitive_state.json` $\to$ PreInvocation hook delivery $\to$ $< 20\,\text{ms}$ preemption interrupt. |

---

## 4. Category-Partition Test Specifications (Ostrand & Balcer)

### 4.1 Tier 1: Darwin Mach Kernel QoS & Quiescence (`tests/test_darwin_idle.py`)

#### Partition 1.1: Operating System Platform
- **P1.1.1 [Darwin / macOS]**: Native Apple Silicon or Intel Darwin (`platform.system() == "Darwin"`). Real `ctypes` Mach/IOKit symbols bound via `ctypes.CDLL(None)`.
- **P1.1.2 [Linux Fallback]**: Simulated Linux (`platform.system() == "Linux"`). Fallback to `/proc/stat` for CPU ticks, `os.nice(19)` for background scheduling, AC power default.
- **P1.1.3 [Windows Fallback]**: Simulated Windows (`platform.system() == "Windows"`). Fallback to `GetSystemTimes`, idle priority class, AC power default.

#### Partition 1.2: Thread QoS Assignment
- **P1.2.1 [Background QoS Request]**: `set_background_qos()` invoked. Passes `QOS_CLASS_BACKGROUND = 0x09` and relative priority `0` to `pthread_set_qos_class_self_np`. Returns `True`.
- **P1.2.2 [Disk I/O Throttling]**: `setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_PROCESS, IOPOL_THROTTLE)` invoked. Sets throttle policy `3`. Returns `True`.
- **P1.2.3 [Invalid Priority Error]**: Relative priority $> 0$ or $< -15$. Function returns error code `22` (`EINVAL`). Handled cleanly without process crash.

#### Partition 1.3: Host CPU Load & Quiescence
- **P1.3.1 [Quiescent State]**: Delta idle ticks $\ge 85\%$, user + system + nice $\le 15\%$. Returns quiescence ratio $\ge 0.85$.
- **P1.3.2 [Moderate Load State]**: Delta idle ticks $= 50\%$, user + system $= 50\%$. Returns quiescence ratio $\approx 0.50$.
- **P1.3.3 [Heavy Load State]**: Delta idle ticks $\le 10\%$. Returns quiescence ratio $\le 0.10$.
- **P1.3.4 [Zero / Negative Tick Delta]**: Delta ticks $\le 0$ (clock glitch, overflow, or instantaneous sample). Clamped to default `1.0` (idle) or `0.0` (load) without division-by-zero exception.

#### Partition 1.4: Thermal Pressure Telemetry
- **P1.4.1 [Nominal (Level 0)]**: No thermal throttling. Full speed permitted.
- **P1.4.2 [Fair (Level 1)]**: Minor elevation. Fans may activate. Mind-wandering permitted under standard threshold.
- **P1.4.3 [Serious (Level 2)]**: Active throttling. `is_system_idle()` must return `False` when `max_thermal = 1`.
- **P1.4.4 [Critical (Level 3)]**: Severe emergency. All subconscious cycles suppressed.

#### Partition 1.5: Power Source & Battery
- **P1.5.1 [AC Power Connected]**: `is_on_battery() == False`. Unlimited background execution allowed.
- **P1.5.2 [Battery Power Normal (> 20%)]**: `is_on_battery() == True`, battery $= 80\%$. Subconscious execution permitted.
- **P1.5.3 [Battery Power Critical (< 20%)]**: `is_on_battery() == True`, battery $= 15\%$. System idle check should reject background execution to prevent battery drain.

---

### 4.2 Tier 2: Non-Homogeneous Poisson Spindle Scheduler (`tests/test_poisson_trigger.py`)

#### Partition 2.1: Hazard Rate $\lambda(t)$ Parameters
$$\lambda(t) = \lambda_0 \cdot \sigma\left(\frac{t - T_{\text{idle\_min}}}{\tau}\right) \cdot (1 - \mathcal{F}_{\text{fatigue}}) \cdot \mathcal{S}_{\text{ToM}}$$
- **P2.1.1 [Idle Time Below Gate ($t < T_{\text{idle\_min}}$)]**: $t = 5\,\text{s}$, $T_{\text{idle\_min}} = 15\,\text{s}$. Sigmoid evaluates to $\approx 0.119$. Rate is suppressed.
- **P2.1.2 [Idle Time At Gate ($t = T_{\text{idle\_min}}$)]**: $t = 15\,\text{s}$. Sigmoid evaluates to exactly $0.50$.
- **P2.1.3 [Idle Time Well Above Gate ($t \gg T_{\text{idle\_min}}$)]**: $t = 60\,\text{s}$. Sigmoid saturates to $\approx 1.0$.
- **P2.1.4 [Fatigue Scaling ($\mathcal{F} \in [0, 1)$)]**:
  - $\mathcal{F} = 0.0$: Full rate ($1 - \mathcal{F} = 1.0$).
  - $\mathcal{F} = 0.5$: Half rate ($1 - \mathcal{F} = 0.5$).
  - $\mathcal{F} \to 1.0$: Rate drops to $0.0$.
- **P2.1.5 [ToM Urgency Multiplier ($\mathcal{S}_{\text{ToM}} \in [0.2, 5.0]$)]**:
  - Low urgency ($\mathcal{S}_{\text{ToM}} = 0.2$): Subconscious rate reduced $5\times$.
  - Baseline urgency ($\mathcal{S}_{\text{ToM}} = 1.0$): Standard rate.
  - Critical urgency ($\mathcal{S}_{\text{ToM}} = 5.0$): Rapid spindle bursts ($5\times$ acceleration).

#### Partition 2.2: Refractory Period Gating
- **P2.2.1 [Within Absolute Dead-Time ($\Delta t < T_{\text{refr}}$)]**: $\Delta t = 4\,\text{s}$, $T_{\text{refr}} = 10\,\text{s}$. `should_trigger()` returns `False` unconditionally.
- **P2.2.2 [At Refractory Boundary ($\Delta t = T_{\text{refr}}$)]**: $\Delta t = 10\,\text{s}$. Gating opens.
- **P2.2.3 [Post-Refractory Epoch ($\Delta t > T_{\text{refr}}$)]**: $\Delta t = 30\,\text{s}$. Stochastic renewal process governs triggering.

#### Partition 2.3: Exponential Renewal Sampling
- **P2.3.1 [Stochastic Interval Generation]**: $\Delta t = -\frac{\ln(U)}{\lambda(t)}$ with $U \sim \text{Uniform}(0, 1)$. Sample mean matches $\mathbb{E}[\Delta t] = \frac{1}{\lambda}$ within statistical tolerance.
- **P2.3.2 [Extreme Uniform Random Samples]**:
  - $U \to 1.0$: $\Delta t \to 0.0$.
  - $U \to 0.001$: $\Delta t$ large, finite.
  - $\lambda \le 0$: Safe handling (returns infinity or large delay, no division by zero).

---

### 4.3 Tier 3A: Theory of Mind (ToM) Analyzer (`tests/test_tom_analyzer.py`)

#### Partition 3.1: Conversational Cadence
- **P3.1.1 [Rapid Conversational Rhythm]**: Inter-turn delays $< 3\,\text{s}$. Indicates active user flow; urgency remains baseline.
- **P3.1.2 [Hesitant Long Pause]**: Inter-turn delay $> 60\,\text{s}$ following a complex technical question. Urgency elevated ($\mathcal{S}_{\text{ToM}} > 1.5$).

#### Partition 3.2: Sociolinguistic Hesitation & Uncertainty Markers
- **P3.2.1 [Explicit Epistemic Doubt]**: Message contains "not sure", "confused", "maybe", "perhaps", "could it be", "I wonder". Boosts urgency and tags doubt in DreamSeed.
- **P3.2.2 [Punctuation & Ellipsis Markers]**: Trailing "..." or multiple question marks "???". Suggests cognitive stall.
- **P3.2.3 [Confident Imperative Turns]**: "Run build", "git push", "clean up". Urgency remains nominal ($\mathcal{S}_{\text{ToM}} \approx 1.0$).

#### Partition 3.3: Milestone & Project State Urgency
- **P3.3.1 [Active Milestone Impasse]**: Project state indicates failed tests, compile error, or upcoming deadline keywords ("deadline", "deploy", "prod", "critical"). Urgency boosted to $\mathcal{S}_{\text{ToM}} \ge 2.5$.
- **P3.3.2 [Clean Project State]**: All tests passing, zero uncommitted changes. Urgency remains standard.

#### Partition 3.4: DreamSeed Synthesis
- **P3.4.1 [Speculative Question Formulation]**: Extracted `DreamSeed.speculative_question` contains actionable hypotheses addressing the user's latent doubts.
- **P3.4.2 [Context Keys Extraction]**: Relevant module paths and symbols identified from conversation history.
- **P3.4.3 [Empty / Degenerate History]**: Zero messages or trivial greetings ("hi"). Emits fallback or empty dream seed list without exceptions.

---

### 4.4 Tier 3B: Isolated Headless Dialectic Engine (`tests/test_mind_wander.py`)

#### Partition 4.1: Multi-Agent Personas & Dialectic Dynamics
- **P4.1.1 [The Generative Dreamer (DMN Incubator)]**: Generates exploratory hypotheses, cross-module associations, speculative refactor plans.
- **P4.1.2 [The Evaluative Arbiter (Zeno Prefrontal Critic)]**: Critiques feasibility, flags regressions, checks against memory engrams.
- **P4.1.3 [Consensus Synthesis]**: Dreamer and Arbiter reach agreement. Returns consolidated `DreamInsight`.

#### Partition 4.2: Psychiatric Anti-Rumination Safeguards
- **P4.2.1 [Healthy Divergent Thought]**: Cosine similarity between consecutive turns $\le 0.85$. Dialectic proceeds normally.
- **P4.2.2 [Rumination Loop Detected ($\cos > 0.95$)]**: Consecutive turns repeat identical semantic statevector ($> 0.95$). Triggers synthetic noradrenaline reset:
  - Sets `anti_rumination_reset_occurred = True`.
  - Injects noradrenaline intervention prompt.
  - Boosts exploration drive.
- **P4.2.3 [Persistent Rumination Hard Abort]**: Two consecutive similarity violations. Dialectic aborts immediately with `StopReason.RUMINATION_DETECTED` to prevent token burn.

#### Partition 4.3: Bounded Resource Ceilings
- **P4.3.1 [Turn Cap Enforcement]**: Turns taken must never exceed $5$ turns per cycle ($N_{\text{turns}} \le 5$).
- **P4.3.2 [Token Cap Enforcement]**: Cumulative tokens must never exceed $2500$ tokens per cycle ($N_{\text{tokens}} \le 2500$).

#### Partition 4.4: Preemption Interruption
- **P4.4.1 [Zero-Latency User Preemption]**: `preemption_check()` callback returns `True` at turn 1, 2, or 3. Mind-wandering cycle halts cleanly in $< 20\,\text{ms}$, returning `None` or checkpointed state.

---

### 4.5 Tier 4: End-to-End Autonomous Subconscious Cycle (`tests/test_e2e_subconscious.py`)

#### Partition 5.1: Pipeline Data Flow & Memory Consolidation
- **P5.1.1 [Full Forward Cycle]**:
  1. Darwin host quiescence confirmed (`is_system_idle() == True`).
  2. Poisson trigger fires (`should_trigger() == True`).
  3. ToM analyzer extracts high-value `DreamSeed`.
  4. Mind wander engine executes dialectic and synthesizes `DreamInsight`.
  5. SWR memory consolidator stores engram in `CognitiveMemoryManager` with category `"subconscious_dream"`.
  6. Engram persistence verified in `quanta_cognitive_state.json`.
  7. PreInvocation hook (`scripts/hooks/quanta_subconscious_hook.py`) is invoked with mock user turn JSON; outputs `injectSteps` with ephemeral message: `[Quanta Bilişsel Çıpa | Subconscious Insight]`.
- **P5.1.2 [Mid-Cycle Preemption & Graceful Abort]**: User interactive turn arrives during dialectic incubation. Subconscious cycle preempts within $< 20\,\text{ms}$, cleans up lock/PID, and yields thread.

---

## 5. Boundary Value Analysis (BVA) Reference Matrix

| Parameter / Boundary | Minimum Valid | Inflection / Nominal | Maximum Valid | Edge / Violation Handling |
|---|---|---|---|---|
| **CPU Quiescence Ratio** | $0.00$ (100% busy) | $0.70$ (default threshold) | $1.00$ (100% idle) | Clamped to $[0.0, 1.0]$. Zero delta returns $1.0$. |
| **Thermal Pressure Level** | $0$ (Nominal) | $1$ (Fair) | $3$ (Critical) | Values $> 1$ block `is_system_idle()` if `max_thermal=1`. |
| **Battery Percentage** | $0\%$ | $20\%$ (quiescence boundary) | $100\%$ | Battery $< 20\%$ suppresses subconscious cycle. |
| **Poisson Idle Gate $T_{\text{idle\_min}}$** | $0.0\,\text{s}$ | $15.0\,\text{s}$ | $300.0\,\text{s}$ | Sigmoid $\sigma((t - T_{\text{idle}})/\tau)$: at $t = T_{\text{idle}}$, factor is exactly $0.5$. |
| **Fatigue Factor $\mathcal{F}$** | $0.00$ (rested) | $0.50$ (moderate) | $0.999$ (exhausted) | $\lambda \to 0$ as $\mathcal{F} \to 1.0$. Clamped to $[0.0, 1.0)$. |
| **ToM Urgency Multiplier $\mathcal{S}_{\text{ToM}}$** | $0.20$ (trivial) | $1.00$ (standard) | $5.00$ (urgent) | Clamped to $[0.2, 5.0]$. |
| **Refractory Period $T_{\text{refr}}$** | $0.0\,\text{s}$ | $10.0\,\text{s}$ | $60.0\,\text{s}$ | Hard block if $\Delta t_{\text{last\_dream}} < T_{\text{refr}}$. |
| **Turn Limit $N_{\text{turns}}$** | $1$ | $3$ | $5$ | Hard halt if turns exceed $5$. |
| **Token Budget $N_{\text{tokens}}$** | $10$ | $800$ | $2500$ | Hard halt if tokens exceed $2500$. |
| **Thought Cosine Similarity $\mathcal{C}_k$** | $-1.00$ | $0.50$ | $0.95$ | If $\mathcal{C}_k > 0.95$, trigger noradrenaline reset; repeat $\to$ abort. |
| **Preemption Latency** | $0.0\,\text{ms}$ | $< 1.0\,\text{ms}$ (empirical) | $< 20.0\,\text{ms}$ (spec limit) | Abort flag checked per generation step / turn. |

---

## 6. Pairwise Combinatorial Test Matrix

To guarantee robust cross-feature coverage without factorial explosion, 16 canonical orthogonal test frames cover the system state space:

| Frame # | Platform | CPU Idle | Thermal State | Power Source | ToM Urgency | Fatigue | Expected Trigger? | Dialectic Action |
|---|---|---|---|---|---|---|---|---|
| **TF-01** | Darwin | 92% (High) | 0 (Nominal) | AC Power | 1.0 (Normal) | 0.0 (None) | **YES** | Standard Dialectic |
| **TF-02** | Darwin | 88% (High) | 0 (Nominal) | Battery 85% | 3.5 (High) | 0.2 (Low) | **YES** | Accelerated Incubation |
| **TF-03** | Darwin | 60% (Low) | 0 (Nominal) | AC Power | 1.0 (Normal) | 0.0 (None) | **NO** (CPU busy) | Suppressed |
| **TF-04** | Darwin | 95% (High) | 2 (Serious) | AC Power | 4.0 (High) | 0.1 (Low) | **NO** (Thermal) | Suppressed |
| **TF-05** | Darwin | 90% (High) | 0 (Nominal) | Battery 12% | 5.0 (Critical) | 0.0 (None) | **NO** (Low battery) | Suppressed |
| **TF-06** | Darwin | 93% (High) | 0 (Nominal) | AC Power | 0.2 (Minimal) | 0.9 (Fatigued) | **NO** (Low rate) | Suppressed |
| **TF-07** | Darwin | 91% (High) | 1 (Fair) | AC Power | 2.5 (Elevated) | 0.1 (Low) | **YES** | Incubation Allowed |
| **TF-08** | Darwin | 94% (High) | 0 (Nominal) | AC Power | 1.0 (Normal) | 0.0 (None) | **YES** | Rumination ($\cos > 0.95$) $\to$ Reset |
| **TF-09** | Darwin | 94% (High) | 0 (Nominal) | AC Power | 1.0 (Normal) | 0.0 (None) | **YES** | Rumination Repeat $\to$ Hard Abort |
| **TF-10** | Darwin | 92% (High) | 0 (Nominal) | AC Power | 1.0 (Normal) | 0.0 (None) | **YES** | Token cap exceeded $\to$ Halt |
| **TF-11** | Darwin | 92% (High) | 0 (Nominal) | AC Power | 1.0 (Normal) | 0.0 (None) | **YES** | Turn cap $= 5 \to$ Halt |
| **TF-12** | Darwin | 90% (High) | 0 (Nominal) | AC Power | 2.0 (Elevated) | 0.0 (None) | **YES** | Mid-cycle preemption $\to$ Yield |
| **TF-13** | Linux | 88% (High) | 0 (Nominal) | AC Power | 1.0 (Normal) | 0.0 (None) | **YES** (Fallback) | Standard Dialectic |
| **TF-14** | Linux | 40% (Low) | 0 (Nominal) | AC Power | 2.0 (Elevated) | 0.0 (None) | **NO** (CPU busy) | Suppressed |
| **TF-15** | Windows | 91% (High) | 0 (Nominal) | AC Power | 1.0 (Normal) | 0.0 (None) | **YES** (Fallback) | Standard Dialectic |
| **TF-16** | Windows | 90% (High) | 2 (Serious) | AC Power | 1.0 (Normal) | 0.0 (None) | **NO** (Thermal) | Suppressed |

---

## 7. Statistical Testing via Time-Rescaling Theorem & Kolmogorov-Smirnov

For non-homogeneous Poisson validation in `tests/test_poisson_trigger.py`:
1. **Time-Rescaling Transform**:
   Given event arrival times $0 < t_1 < t_2 < \dots < t_n$:
   $$\Lambda_k = \int_{t_{k-1}}^{t_k} \lambda(s) \, ds$$
   By Theorem 1, $\Lambda_k \sim_{\text{i.i.d.}} \text{Exponential}(1)$.
2. **Probability Integral Transform (PIT)**:
   $$u_k = 1 - \exp(-\Lambda_k) \sim_{\text{i.i.d.}} \text{Uniform}(0, 1)$$
3. **Kolmogorov-Smirnov Statistic**:
   $$D_n = \sup_{u \in [0, 1]} |S_n(u) - u|$$
   - For $n = 100$, critical value $D_{0.05} = \frac{1.36}{\sqrt{100}} = 0.136$.
   - The test asserts $D_n < 0.136$ with $p$-value $> 0.05$, confirming statistical conformity with biomorphic Poisson renewal theory.

---

## 8. Test Execution & Quality Gates

### Execution Commands
```bash
# Targeted E2E & Subconscious suite execution
/Users/aes/Antigravity\ Projects/Alfa/quanta/.venv/bin/pytest \
  tests/test_darwin_idle.py \
  tests/test_poisson_trigger.py \
  tests/test_tom_analyzer.py \
  tests/test_mind_wander.py \
  tests/test_e2e_subconscious.py \
  -o addopts="-v --tb=short"

# Ruff code quality & lint verification
/Users/aes/Antigravity\ Projects/Alfa/quanta/.venv/bin/ruff check \
  tests/test_darwin_idle.py \
  tests/test_poisson_trigger.py \
  tests/test_tom_analyzer.py \
  tests/test_mind_wander.py \
  tests/test_e2e_subconscious.py
```

### Mandatory Quality Gate Standards
- 0 lint errors on `ruff check` (100-character line length, PEP 8, Google docstrings).
- 100% pass rate across all test suites.
- Strict isolation: tests must not create persistent uncleaned files or leave rogue daemon processes.
- Opaque-box integrity: all test assertions check genuine logic against the specification contracts.
