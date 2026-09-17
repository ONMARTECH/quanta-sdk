# Project: Quanta Cognitive Architecture Pillar 3
# Autonomous Biomorphic Subconscious Mind-Wandering & Anticipatory Prospection Engine

## Architecture

The Subconscious Mind-Wandering & Anticipatory Prospection Engine operates as an autonomous, biomorphic background cognitive process that models mammalian default mode network (DMN) daydreaming, thalamocortical sleep spindles, synaptic homeostasis downscaling, and sociological Theory of Mind. It executes strictly within Apple Silicon Efficiency cores (`QOS_CLASS_BACKGROUND`) without consuming foreground interactive CPU/GPU resources or exceeding the ~20W biomorphic thermodynamic budget.

### Subsystem Decomposition & Data Flow:

```
[ Darwin Host Quiescence ] ──> [ Poisson Spindle Generator ] ──> [ Dream Pulse Trigger ]
  (Mach host_info, thermal,          (lambda(t), refractory,               │
   battery, E-core QoS)               ToM urgency modulation)              │
                                                                           ▼
[ Conversation Cadence & History ] ──> [ Theory of Mind Analyzer ] ──> [ Speculative Dream Seed ]
                                                                           │
                                                                           ▼
                                                             [ Headless Dialectic Engine ]
                                                             - Generative Dreamer (DMN)
                                                             - Evaluative Arbiter (Zeno)
                                                             - Anti-Rumination Check (cos > 0.95)
                                                             - Headless Antigravity Conversation
                                                                           │
                                                                           ▼
                                                             [ SWR Memory Consolidation ]
                                                             - SWR Replay & Synaptic Downscaling
                                                             - quanta_cognitive_state.json
                                                             - Preemption Interrupt (<20ms)
                                                                           │
                                                                           ▼
[ User Interactive Turn ] ───────────> [ PreInvocation Hook ] ───────> [ Pre-incubated Insight Delivery ]
```

---

## Feature Inventory

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | R1 Comprehensive Theoretical Treatise | Formal mathematical and biophysical foundation in `docs/theory/subconscious_mind_wandering_and_dmn.md` | M0 | Survey 2 & Request |
| 2 | Darwin Mach Thread QoS Assignment | Native `pthread_set_qos_class_self_np(0x09, 0)` enforcing `QOS_CLASS_BACKGROUND` on Apple Silicon E-cores | M1 | Survey 1 |
| 3 | Darwin Disk I/O Throttling | Native `setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_PROCESS, IOPOL_THROTTLE)` | M1 | Survey 1 |
| 4 | Mach CPU Host Statistics | Instantaneous CPU idle tick tracking via `host_statistics64(HOST_CPU_LOAD_INFO)` | M1 | Survey 1 |
| 5 | Apple Silicon Thermal Telemetry | Thermal pressure monitoring via `notify(3)` / `kIOPlatformThermalNotificationKey` / `NSProcessInfo` | M1 | Survey 1 |
| 6 | Battery & Power State Telemetry | Power source detection via `IOKit` / `IOPowerSources.h` | M1 | Survey 1 |
| 7 | Cross-Platform Hardware Fallback | Graceful stubs for Linux (`/proc/stat`, nice 19) and Windows (`GetSystemTimes`, idle priority) | M1 | Survey 1 |
| 8 | Non-Homogeneous Poisson Rate $\lambda(t)$ | Dynamic rate $\lambda(t) = \lambda_0 \cdot \sigma((t - T_{\text{idle}})/\tau) \cdot (1 - \mathcal{F}_{\text{fatigue}}) \cdot \mathcal{S}_{\text{ToM}}$ | M2 | Survey 2 |
| 9 | Exponential Inter-Arrival Sampling | Inverse transform stochastic sampling $\Delta t \sim -\ln(U)/\lambda(t)$ with refractory gating | M2 | Survey 2 |
| 10 | Theory of Mind Latent Needs Extractor | Conversational cadence, hesitation, and epistemic uncertainty analyzer | M2 | Survey 2 |
| 11 | Speculative Dream Seed Generator | Synthesizes high-utility speculative questions for background deliberation | M2 | Survey 2 |
| 12 | Headless Antigravity Dialectic Engine | Headless `Agent` + `Conversation` dual-persona deliberation with zero prompt leakage | M3 | Survey 3 |
| 13 | Dual Personas (DMN Dreamer & Zeno Critic)| Generative exploratory incubator vs. Evaluative prefrontal arbiter with virtual rollouts | M3 | Survey 3 |
| 14 | Psychiatric Anti-Rumination Guard | Cosine similarity > 0.95 detection triggering synthetic noradrenaline reset | M3 | Survey 2 & Request |
| 15 | Bounded Execution Budget | Hard turn limit ($\le 5$ turns) and token limit ($\le 2500$ tokens per cycle) | M3 | Request |
| 16 | SWR Memory Engram Consolidation | Integrates consensus into `CognitiveMemoryManager` (`quanta_cognitive_state.json`) | M4 | Survey 3 |
| 17 | Instant Preemption Reflex | User turn or hardware interrupt aborts/checkpoints subconscious run in $< 20\,\text{ms}$ | M4 | Survey 1 & 3 |
| 18 | Background Subconscious Daemon Manager | Daemon lifecycle manager with PID tracking, signal handling, and status logging | M4 | Survey 3 |
| 19 | CLI Command Suite | `quanta dream start/stop/status/inspect` commands | M4 | Survey 3 & Request |
| 20 | PreInvocation Subconscious Hook | `scripts/hooks/quanta_subconscious_hook.py` serving pre-incubated insights on user turns | M4 | Survey 3 & Request |
| 21 | Opaque-Box E2E Test Suite | 4-Tier requirement-driven E2E test suite published with `TEST_READY.md` | Test Track | Dual Track |
| 22 | Kolmogorov-Smirnov Goodness-of-Fit | Statistical validation of Poisson renewal intervals via Time-Rescaling Theorem | M5 | Survey 2 |
| 23 | Preemption Latency Benchmarking | Verification of interruption latency $< 50\,\text{ms}$ (target $< 20\,\text{ms}$) | M5 | Survey 1 & 3 |
| 24 | Package Quality Gates | >90% test coverage across `quanta.cognitive`, 0 ruff lint errors, 0 mypy type errors | M5 | Request |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M0 | R1 Theoretical Treatise | `docs/theory/subconscious_mind_wandering_and_dmn.md` | Survey Complete | DONE |
| M1 | Darwin Mach QoS & Quiescence Monitor | `quanta/cognitive/darwin_idle.py` | Survey Complete | DONE |
| M2 | Poisson Trigger & Theory of Mind Analyzer | `quanta/cognitive/poisson_trigger.py`, `quanta/cognitive/tom_analyzer.py` | M0, M1 | DONE |
| M3 | Headless Dialectics & Mind Wander Engine | `quanta/cognitive/mind_wander.py` | M2 | DONE |
| M4 | SWR Consolidation, Daemon, CLI & Hook | `quanta/cognitive/consolidation.py`, `quanta/cognitive/daemon.py`, `scripts/hooks/quanta_subconscious_hook.py`, `quanta/cli.py` | M3 | PLANNED |
| M5 | Final Milestone: 100% E2E Pass & Benchmarks | Pass full E2E suite, empirical benchmarks (KS, latency), >90% coverage, 0 lint/type errors | M0-M4, TEST_READY.md | PLANNED |

---

## Interface Contracts

### 1. `quanta.cognitive.darwin_idle`
```python
def set_background_qos() -> bool:
    """Enforce QOS_CLASS_BACKGROUND (0x09) and IOPOL_THROTTLE."""

def get_cpu_quiescence() -> float:
    """Return CPU idle ratio in [0.0, 1.0] from host_statistics64."""

def get_thermal_state() -> int:
    """Return Apple Silicon thermal state (0=Nominal, 1=Fair, 2=Serious, 3=Critical)."""

def is_on_battery() -> bool:
    """Return True if system is running on battery power."""

def is_system_idle(idle_threshold: float = 0.70, max_thermal: int = 1) -> bool:
    """Check if host is quiet enough for subconscious mind-wandering."""
```

### 2. `quanta.cognitive.poisson_trigger`
```python
class PoissonSpindleTrigger:
    def __init__(self, lambda_0: float = 0.1, idle_min: float = 15.0, tau: float = 5.0, refractory_sec: float = 10.0): ...
    def compute_rate(self, current_time: float, last_active_time: float, fatigue: float, tom_urgency: float) -> float: ...
    def sample_next_interval(self, rate: float) -> float: ...
    def should_trigger(self, current_time: float, last_active_time: float, last_dream_time: float, fatigue: float, tom_urgency: float) -> bool: ...
```

### 3. `quanta.cognitive.tom_analyzer`
```python
@dataclass
class DreamSeed:
    topic: str
    speculative_question: str
    urgency: float
    context_keys: list[str]

class TheoryOfMindAnalyzer:
    def analyze_conversation(self, messages: list[dict], project_state: dict | None = None) -> tuple[float, list[DreamSeed]]:
        """Return (tom_urgency, dream_seeds)."""
```

### 4. `quanta.cognitive.mind_wander`
```python
@dataclass
class DreamInsight:
    topic: str
    seed_question: str
    synthesis: str
    confidence: float
    turns_taken: int
    tokens_used: int
    anti_rumination_reset_occurred: bool

class MindWanderEngine:
    def __init__(self, max_turns: int = 5, max_tokens: int = 2500, rumination_threshold: float = 0.95): ...
    def execute_dream_cycle(self, seed: DreamSeed, preemption_check: Callable[[], bool]) -> DreamInsight | None:
        """Run headless dialectic with DMN and Zeno personas."""
```

### 5. `quanta.cognitive.consolidation`
```python
class SubconsciousConsolidator:
    def __init__(self, state_file: Path | str = "quanta_cognitive_state.json"): ...
    def consolidate_insight(self, insight: DreamInsight) -> bool:
        """SWR replay into CognitiveMemoryManager engrams with category='subconscious_dream'."""
```

### 6. `quanta.cognitive.daemon`
```python
class SubconsciousDaemon:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def status(self) -> dict: ...
    def inspect(self) -> list[dict]: ...
    def interrupt_immediate(self) -> None: ...
```

---

## Code Layout

```
quanta/
├── __init__.py
├── cli.py                               # CLI entry points (quanta dream ...)
├── cognitive/
│   ├── __init__.py                      # Export all subconscious public APIs
│   ├── memory.py                        # CognitiveMemoryManager (existing)
│   ├── arbiter.py                       # QuantumDecisionArbiter (existing)
│   ├── middleware.py                    # QuantaCognitiveMiddleware (existing)
│   ├── darwin_idle.py                   # Mach QoS, E-core, thermal, CPU quiescence
│   ├── poisson_trigger.py               # Stochastic Poisson spindle scheduler
│   ├── tom_analyzer.py                  # Theory of Mind & Dream Seed extractor
│   ├── mind_wander.py                   # Isolated headless dialectic & anti-rumination
│   ├── consolidation.py                 # SWR memory consolidation into state JSON
│   └── daemon.py                        # Background daemon manager & IPC preemption
docs/
└── theory/
    └── subconscious_mind_wandering_and_dmn.md  # Comprehensive theoretical monograph
scripts/
└── hooks/
    └── quanta_subconscious_hook.py      # PreInvocation subconscious insight delivery hook
tests/
├── test_darwin_idle.py                  # Darwin Mach QoS & quiescence tests
├── test_poisson_trigger.py              # Poisson rate, renewal sampling, KS tests
├── test_tom_analyzer.py                 # Theory of Mind heuristics tests
├── test_mind_wander.py                  # Headless dialectic & anti-rumination tests
├── test_consolidation.py                # SWR memory consolidation & preemption tests
├── test_subconscious_daemon.py          # Daemon start/stop/status CLI & IPC tests
└── test_e2e_subconscious.py             # Full end-to-end integration test
```
