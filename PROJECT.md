# Project: Quanta SDK & Antigravity 5 Advanced Cognitive-Quantum Architectural Enhancements

## Architecture
Comprehensive architectural expansion implementing 5 advanced cognitive-quantum enhancements across Quanta SDK and Antigravity multi-agent systems, structured in a disciplined two-phase rollout:

- **Phase 1: Cognitive Memory & Multi-Agent Collaboration Infrastructure**
  - **Dynamic Context-Adaptive SWR Resonance (`quanta/cognitive/memory.py`)**:
    Enhances `CognitiveMemoryManager` with a contextual resonance scoring function $R(C, e) = 0.6 F_{\text{hilbert}}(C, e) + 0.4 J(C, e)$ computing quantum Hilbert-space overlap $|\langle \psi_C | \psi_e \rangle|^2$ and token Jaccard similarity. Unconditionally preserves 100% of Core Anchors (`is_core_anchor=True`, salience $S \ge 2.0$, or category $\in \{\text{'system\_core'}, \text{'semantic\_gist'}, \text{'constraint'}\}$), while dynamically pruning transient engrams to top-$K$ resonant elements, eliminating static prompt bloat and reducing transient prompt tokens by $\ge 40\%$ on off-topic tasks.
  - **Multi-Agent Shared Cognitive Blackboard (`quanta/cognitive/blackboard.py`)**:
    Provides a concurrency-safe shared blackboard allowing the orchestrator and subagents to exchange ephemeral constraints, findings, architectural decisions, and heartbeats. Uses a dedicated lockfile (`.lock`) with advisory locking (`fcntl.flock(LOCK_EX)`), write-to-tempfile, `os.fsync`, and POSIX-atomic `os.replace` to prevent race conditions, partial writes, or vault corruption under heavy multi-agent contention.
  - **Local Offline Vector Retrieval for Central Vaults (`quanta/cognitive/vault_index.py`)**:
    Zero-external-dependency local semantic and keyword indexer for `local vault storage knowledge_vault.json` and `assets_vault.json`. Uses QNLP complex unit statevector projections in $\mathbb{C}^{64}$ and batch matrix multiplication to achieve $<50\text{ms}$ query latency completely offline, with strict security isolation preventing access to `credentials_vault.json`.

- **Phase 2: Decision Arbiter Ablation & Tensor Network Quantum Simulation**
  - **Zeno Arbiter Telemetry & Programmatic Ablation Engine (`quanta/cognitive/ablation.py`)**:
    Automated scientific ablation engine benchmarking `QuantumDecisionArbiter` against naive classical baselines (Random, Greedy, Rule-Based) across adversarial hard decision instances. Measures marginal utility $\Delta U$, Pareto trade-offs, and consensus metrics, logging results atomically to `quanta_cognitive_telemetry.jsonl` in strict adherence to the Quanta Scientific Integrity Protocol .
  - **MPS Simulator Deep Integration & High-Qubit Verification (`quanta/simulator/mps.py`, `quanta/runner.py`, `quanta/mcp_server.py`)**:
    Deeply integrates `MPSSimulator` into Quanta's execution pipeline via `CircuitRunner`, supporting `method="mps"` with automatic routing for high-qubit circuits. Suppresses exponential $O(2^N)$ dense statevector materialization for $N > 20$, exposes MPS simulation to MCP tools, and provides automated scaling benchmarks on 15–30 qubit 1D circuits demonstrating memory containment ($<500\text{KB}$ vs $>17\text{GB}$ dense) and near-zero truncation error for low-entanglement states.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Context Resonance Scoring | $R(C, e) = 0.6 F_{\text{hilbert}} + 0.4 J$ in `CognitiveMemoryManager` | M1 | Phase 1 R1 |
| 2 | 100% Core Anchor Preservation | Unconditional retention of `is_core_anchor=True`, $S \ge 2.0$, and core categories | M1 | Phase 1 R1 |
| 3 | Dynamic Transient Engram Pruning | Filters transient engrams to top-$K$ resonant items, cutting off-topic tokens $\ge 40\%$ | M1 | Phase 1 R1 |
| 4 | SWR Prompt Formatting & Eviction APIs | `recall_context_adaptive_swr`, `format_context_adaptive_swr_prompt`, `prune_off_topic_transient` | M1 | Phase 1 R1 |
| 5 | Shared Cognitive Blackboard Module | `quanta/cognitive/blackboard.py` for multi-agent ephemeral coordination | M2 | Phase 1 R2 |
| 6 | Dedicated Lockfile Advisory Locking | `fcntl.flock(LOCK_EX)` on `<path>.lock` preventing lock-inode invalidation | M2 | Phase 1 R2 |
| 7 | POSIX Atomic Tempfile Replacement | Write-to-tempfile + `fsync` + `os.replace` for 100% atomic state updates | M2 | Phase 1 R2 |
| 8 | Blackboard Structured JSON Schema | Typed entries for constraints, findings, decisions, and agent heartbeats | M2 | Phase 1 R2 |
| 9 | Local Offline Vault Indexer | `quanta/cognitive/vault_index.py` with zero external network dependencies | M3 | Phase 1 R3 |
| 10 | Dual Vault Parser | Ingestion of `knowledge_vault.json` and `assets_vault.json` lists and dictionary items | M3 | Phase 1 R3 |
| 11 | QNLP Statevector + Lexical Hybrid Search | Vector cosine similarity + BM25/keyword ranking executed via local PyTorch | M3 | Phase 1 R3 |
| 12 | Strict Credential Vault Isolation | Hard blacklist preventing indexing or access to `credentials_vault.json` or secret keys | M3 | Phase 1 R3 |
| 13 | Sub-50ms Query Latency Guarantee | In-memory matrix evaluation achieving $<50\text{ms}$ query latency offline | M3 | Phase 1 R3 |
| 14 | Automated Programmatic Ablation Engine | `quanta/cognitive/ablation.py` comparing Zeno Arbiter vs classical baselines | M4 | Phase 2 R4 |
| 15 | 4-Way Comparative Baseline Variants | Random, Greedy, Rule-Based, and Quantum Zeno Arbiter implementations | M4 | Phase 2 R4 |
| 16 | Adversarial Hard Decision Testbed | Near-degenerate ties, Trojan traps, Pareto conflicts, cascading regret DAGs | M4 | Phase 2 R4 |
| 17 | Scientific Integrity Telemetry Logging | Thread-safe atomic JSONL append to `quanta_cognitive_telemetry.jsonl` | M4 | Phase 2 R4 |
| 18 | `CircuitRunner` Architecture & Auto-Routing | `quanta/runner.py` with `method="mps"` and routing based on qubit count | M5 | Phase 2 R5 |
| 19 | High-Qubit Statevector Safeguard | Omission of dense $O(2^N)$ statevector allocation when $N > 20$ in MPS results | M5 | Phase 2 R5 |
| 20 | MCP Server MPS Exposure | Support for `method="mps"` in `run_circuit` and dedicated `mps_circuit_simulate` | M5 | Phase 2 R5 |
| 21 | 15–30 Qubit 1D MPS Scaling Benchmarks | Automated benchmark suite demonstrating memory containment and low truncation error | M5 | Phase 2 R5 |
| 22 | Full Quanta Regression Gate | 0 regressions across all 2,634+ existing unit and cognitive tests | M6 | Acceptance |
| 23 | Sentinel Victory Handoff | Comprehensive handoff documentation for independent Victory Auditor | M6 | Acceptance |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Dynamic Context-Adaptive SWR Resonance | Features 1, 2, 3, 4 | Survey Complete | DONE |
| M2 | Multi-Agent Shared Cognitive Blackboard | Features 5, 6, 7, 8 | Survey Complete | DONE |
| M3 | Local Offline Vector Retrieval for Vaults | Features 9, 10, 11, 12, 13 | Survey Complete | DONE |
| M4 | Zeno Arbiter Telemetry & Programmatic Ablation Engine | Features 14, 15, 16, 17 | M1, M2 | DONE |
| M5 | MPS Simulator Deep Integration & Scaling Benchmarks | Features 18, 19, 20, 21 | Survey Complete | DONE |
| M6 | Full Regression Gate & Victory Handoff | Features 22, 23 | M1, M2, M3, M4, M5 | DONE |

## Interface Contracts

### M1: Dynamic Context-Adaptive SWR Resonance
- `quanta/cognitive/memory.py`:
  - `CognitiveMemoryManager`:
    - `compute_context_resonance(self, context: str, engram: dict[str, Any] | int, alpha: float = 0.6) -> float`
    - `recall_context_adaptive_swr(self, context: str, top_k_transient: int = 3, min_resonance: float = 0.12, min_fidelity: float = 0.40) -> dict[str, list[dict[str, Any]]]`
    - `format_context_adaptive_swr_prompt(self, context: str, top_k_transient: int = 3, min_resonance: float = 0.12) -> str`
    - `prune_off_topic_transient(self, context: str, min_resonance: float = 0.08, min_age_turns: float = 2.0) -> list[dict[str, Any]]`

### M2: Multi-Agent Shared Cognitive Blackboard
- `quanta/cognitive/blackboard.py`:
  - `CognitiveBlackboard`:
    - `__init__(self, storage_path: str | Path | None = None)`
    - `post_constraint(self, agent_id: str, key: str, rule: str, is_hard: bool = True, ttl: float | None = None) -> str`
    - `post_finding(self, agent_id: str, title: str, content: str, tags: list[str] | None = None, salience: float = 1.0) -> str`
    - `post_decision(self, agent_id: str, key: str, decision: str, rationale: str = "", status: str = "active") -> str`
    - `update_heartbeat(self, agent_id: str, status: str, metadata: dict[str, Any] | None = None) -> None`
    - `get_constraints(self, active_only: bool = True) -> list[dict[str, Any]]`
    - `get_findings(self, tag: str | None = None) -> list[dict[str, Any]]`
    - `get_decisions(self, status: str = "active") -> list[dict[str, Any]]`
    - `get_heartbeats(self) -> dict[str, dict[str, Any]]`
    - `atomic_modify(self, modifier_callable: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]`

### M3: Local Offline Vector Retrieval for Central Vaults
- `quanta/cognitive/vault_index.py`:
  - `LocalVaultIndex`:
    - `__init__(self, vault_dir: str | Path | None = None, auto_build: bool = True)`
    - `build_index(self, force_refresh: bool = False) -> int`
    - `search(self, query: str, top_k: int = 5, vault_filter: str | None = None) -> list[dict[str, Any]]`
    - `search_knowledge(self, query: str, top_k: int = 5) -> list[dict[str, Any]]`
    - `search_assets(self, query: str, top_k: int = 5) -> list[dict[str, Any]]`

### M4: Zeno Arbiter Telemetry & Programmatic Ablation Engine
- `quanta/cognitive/ablation.py`:
  - `ZenoAblationEngine`:
    - `__init__(self, telemetry_path: str | Path | None = None, seed: int = 42)`
    - `run_ablation_trial(self, instance: DecisionInstance, variant: str) -> AblationTrialResult`
    - `run_suite(self, instances: list[DecisionInstance] | None = None, runs_per_instance: int = 5) -> AblationReport`
    - `log_to_telemetry(self, report: AblationReport) -> None`
    - `generate_summary_table(self, report: AblationReport) -> str`

### M5: MPS Simulator Deep Integration & Scaling Benchmarks
- `quanta/runner.py`:
  - `CircuitRunner`:
    - `__init__(self, default_method: str = "auto", max_dense_qubits: int = 27)`
    - `run(self, circuit_or_dag: Any, shots: int = 1024, method: str = "auto", max_bond_dim: int = 64, cutoff: float = 1e-12, seed: int | None = None) -> CircuitResult`
- `quanta/mcp_server.py`:
  - Update `run_circuit` tool signature to accept `method: str = "statevector"`
  - Add `mps_circuit_simulate` tool supporting 15–30 qubit circuits
- `benchmarks/mps_scaling_benchmark.py`:
  - `run_mps_scaling_benchmark(qubits_range: list[int] = [15, 20, 25, 30]) -> dict[str, Any]`

## Code Layout
- `quanta/cognitive/memory.py`: Enhanced `CognitiveMemoryManager` with contextual SWR resonance.
- `quanta/cognitive/blackboard.py`: `CognitiveBlackboard` concurrency-safe atomic multi-agent blackboard.
- `quanta/cognitive/vault_index.py`: `LocalVaultIndex` zero-dependency offline PyTorch statevector search.
- `quanta/cognitive/ablation.py`: `ZenoAblationEngine` programmatic ablation comparing Zeno vs classical baselines.
- `quanta/simulator/mps.py`: Core `MPSSimulator` matrix product state simulator.
- `quanta/runner.py`: Enhanced `CircuitRunner` with MPS auto-routing and memory containment.
- `quanta/mcp_server.py`: MCP tool definitions exposing MPS simulation.
- `benchmarks/mps_scaling_benchmark.py`: 15–30 qubit automated scaling benchmarks.
- `tests/test_cognitive_phase1_memory_resonance.py`: SWR resonance and token reduction tests.
- `tests/test_cognitive_phase1_blackboard.py`: Blackboard multi-process/thread concurrency stress tests.
- `tests/test_cognitive_phase1_vault_index.py`: Local vault index latency (<50ms) and security tests.
- `tests/test_cognitive_phase2_zeno_ablation.py`: Programmatic ablation suite and telemetry tests.
- `tests/test_mps_runner_scaling.py`: CircuitRunner MPS integration and scaling tests.
