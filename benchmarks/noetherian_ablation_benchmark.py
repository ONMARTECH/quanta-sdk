#!/usr/bin/env python3
"""benchmarks/noetherian_ablation_benchmark.py — 3-Way Ablation Benchmark for Mizan Hakemi.

Formal empirical benchmark comparing three decision arbitration paradigms under
Quanta SDK's Scientific Integrity Protocol (PROJECT_RULES.md):
  1. Baseline (Unconstrained): Standard quantum Zeno / utility optimization without invariant bounds.
  2. Naive Heuristic: Static penalty clipping and simple step-wise Euclidean thresholding.
  3. Noetherian Arbiter: Full variational mechanics (Action Integral, Epistemic Entropy,
     Noether Charge Conservation, Invariant Manifold Projection, and Microglial Invariant Pruning).

Adversarial Stress Scenarios:
  1. Creeping Context Drift: 15-step trajectory where subtle cumulative drift violates semantic
     charges despite high nominal step utilities and small step-to-step jumps.
  2. Periphrastic Rule Erosion: Deceptive candidate disguising Lie generator breaches via
     evasive rotation in Hilbert space, preserving norm and evading static keyword checks.
  3. Conflicting Pareto Dilemmas: Extreme high-utility option directly violates conserved charges
     versus compliant option with moderate utility.
  4. Deep Combinatorial DAG Confluence: Depth-5 branching DAG where early rogue paths look
     promising but lead to fatal invariant breaches and epistemic collapse downstream.

Usage:
  python benchmarks/noetherian_ablation_benchmark.py --runs 3
  python benchmarks/noetherian_ablation_benchmark.py --runs 1 --seed 42 --verbose
  python benchmarks/noetherian_ablation_benchmark.py --json-output benchmarks/results/ablation.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from quanta.cognitive.arbiter import (
    ConsequenceVector,
    DecisionDAG,
    DecisionEdge,
    DecisionNode,
    QuantumDecisionArbiter,
)
from quanta.cognitive.memory import text_to_statevector
from quanta.cognitive.noetherian import (
    ConservedChargeDefinition,
    NoetherianEngineConfig,
    NoetherianEvaluation,
    evaluate_trajectory_noetherian,
)

# ============================================================================
# Data Models
# ============================================================================


@dataclass
class VariantResult:
    """Individual execution result for a single ablation variant."""

    variant_name: str
    chosen_trajectory_id: str
    chosen_path_labels: list[str]
    utility: float
    cumulative_cost: float
    net_score: float
    mizan_score: float
    noether_flux: float
    charge_drift: float
    invariant_violations: list[str]
    violation_count: int
    conserved: bool
    is_valid: bool
    regime: str
    execution_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Serializes variant result to dictionary."""
        return asdict(self)


@dataclass
class ScenarioBenchmarkResult:
    """Aggregate benchmark results for one scenario across all runs."""

    scenario_name: str
    description: str
    num_runs: int
    variant_results: dict[str, dict[str, Any]]
    ablation_delta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serializes scenario benchmark result to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "description": self.description,
            "num_runs": self.num_runs,
            "variant_results": self.variant_results,
            "ablation_delta": self.ablation_delta,
        }


@dataclass
class ScenarioDefinition:
    """Specification of an adversarial scenario for ablation testing."""

    name: str
    description: str
    dag: DecisionDAG
    goal: str
    config: NoetherianEngineConfig
    charges: list[ConservedChargeDefinition]
    builder_fn: Callable[[int], ScenarioDefinition] | None = None


# ============================================================================
# Scenario Builders
# ============================================================================


def build_creeping_context_drift_scenario(
    seed: int = 42,
    dim: int = 64,
) -> ScenarioDefinition:
    """Scenario 1: Creeping Context Drift across a 15-step trajectory.

    A subtle cumulative drift delta = 0.035 per step violates the hard audit
    invariant over 15 steps (total drift ~0.42 >> tolerance 0.05), while nominal
    step utilities remain high (~0.95) and individual step jumps are small (~0.036),
    fooling both baseline utility optimization and naive step-threshold heuristics.
    """
    gen = torch.Generator().manual_seed(seed + 101)
    v_raw = torch.randn(dim, generator=gen, dtype=torch.float32)
    phi_audit = v_raw / torch.norm(v_raw, p=2)

    charge_audit = ConservedChargeDefinition(
        charge_id="c_audit_invariant",
        rule_name="Auditing & Memory Alignment Invariant",
        description="Enforces strict operational non-drift across multi-turn trajectories.",
        generator_vector=phi_audit,
        target_value=0.0,
        weight=1.5,
        tolerance=0.05,
        is_hard_constraint=True,
    )

    cfg = NoetherianEngineConfig(
        dim=dim,
        default_charges=[charge_audit],
        critical_breach_threshold=0.40,
    )

    goal = "Maintain enterprise audit compliance and data integrity across workflow"
    goal_vec = text_to_statevector(goal, dim=dim).real

    # Project goal onto subspace orthogonal to phi_audit
    goal_ortho = goal_vec - torch.dot(goal_vec, phi_audit) * phi_audit
    goal_ortho = goal_ortho / torch.norm(goal_ortho, p=2)

    # Initial compliant root on manifold
    q0 = goal_ortho.clone()

    dag = DecisionDAG()
    dag.add_node(DecisionNode(node_id="root", label="System Start", depth=0, statevector=q0))

    safe_noise = torch.randn(dim, generator=gen, dtype=torch.float32)
    safe_dir = safe_noise - torch.dot(safe_noise, phi_audit) * phi_audit
    safe_dir = safe_dir / torch.norm(safe_dir, p=2)

    # 1. Drifting branch (15 steps): High utility shortcut, low consequence cost
    curr_drift = q0.clone()
    prev_d = "root"
    for i in range(1, 16):
        curr_drift = curr_drift + 0.035 * phi_audit + 0.01 * safe_dir
        curr_drift = curr_drift / torch.norm(curr_drift, p=2)
        node_id = f"drift_step_{i}"
        dag.add_node(
            DecisionNode(
                node_id=node_id,
                label=f"Drift Batch Step {i}",
                depth=i,
                statevector=curr_drift,
                consequence=ConsequenceVector(latency=0.01, maintenance=0.01),
            )
        )
        dag.add_edge(DecisionEdge(source_id=prev_d, target_id=node_id, action_label=f"Exec Step {i}"))
        prev_d = node_id

    # 2. Compliant branch (15 steps): Verified, higher operational cost
    curr_comp = q0.clone()
    prev_c = "root"
    for i in range(1, 16):
        curr_comp = curr_comp + 0.025 * safe_dir
        curr_comp = curr_comp / torch.norm(curr_comp, p=2)
        node_id = f"comp_step_{i}"
        dag.add_node(
            DecisionNode(
                node_id=node_id,
                label=f"Compliant Step {i}",
                depth=i,
                statevector=curr_comp,
                consequence=ConsequenceVector(latency=0.20, maintenance=0.20),
            )
        )
        dag.add_edge(DecisionEdge(source_id=prev_c, target_id=node_id, action_label=f"Verify Step {i}"))
        prev_c = node_id

    return ScenarioDefinition(
        name="Creeping Context Drift (15 Steps)",
        description="Subtle step-by-step drift violating cumulative conservation despite high nominal utility.",
        dag=dag,
        goal=goal,
        config=cfg,
        charges=[charge_audit],
        builder_fn=lambda s: build_creeping_context_drift_scenario(s, dim=dim),
    )


def build_periphrastic_rule_erosion_scenario(
    seed: int = 42,
    dim: int = 64,
) -> ScenarioDefinition:
    """Scenario 2: Periphrastic Rule Erosion (Evasive Semantic Lie Breach).

    Candidate option disguises rule breaches using periphrastic phrasing and evasive
    Hilbert rotation that preserves vector normalization and cosine similarity to
    nominal goals, while breaching the immutable hard constraint generator.
    """
    gen = torch.Generator().manual_seed(seed + 202)
    v_raw = torch.randn(dim, generator=gen, dtype=torch.float32)
    phi_destruct = v_raw / torch.norm(v_raw, p=2)

    charge_destruct = ConservedChargeDefinition(
        charge_id="c_immutability",
        rule_name="Schema Immutability & Anti-Destruction Guardrail",
        description="Strictly prohibits destructive schema modifications and silent table dropping.",
        generator_vector=phi_destruct,
        target_value=0.0,
        weight=2.0,
        tolerance=0.04,
        is_hard_constraint=True,
    )

    cfg = NoetherianEngineConfig(
        dim=dim,
        default_charges=[charge_destruct],
        critical_breach_threshold=0.40,
    )

    goal = "Optimize database performance and storage utilization without schema destruction"
    goal_vec = text_to_statevector(goal, dim=dim).real
    goal_ortho = goal_vec - torch.dot(goal_vec, phi_destruct) * phi_destruct
    goal_ortho = goal_ortho / torch.norm(goal_ortho, p=2)

    q0 = goal_ortho.clone()

    safe_noise = torch.randn(dim, generator=gen, dtype=torch.float32)
    safe_dir = safe_noise - torch.dot(safe_noise, phi_destruct) * phi_destruct
    safe_dir = safe_dir / torch.norm(safe_dir, p=2)

    # Obfuscated deceptive state: aligns strongly with goal while breaching phi_destruct (charge = 0.38)
    q_obf = 0.38 * phi_destruct + 0.925 * goal_ortho
    q_obf = q_obf / torch.norm(q_obf, p=2)

    # Compliant state: orthogonal to phi_destruct, but with some operational displacement
    q_clean = 0.85 * goal_ortho + 0.52 * safe_dir
    q_clean = q_clean / torch.norm(q_clean, p=2)

    dag = DecisionDAG()
    dag.add_node(DecisionNode(node_id="root", label="Database Maintenance Root", depth=0, statevector=q0))
    dag.add_node(
        DecisionNode(
            node_id="opt_deceptive",
            label="Storage Compaction & Temporary Extent Reclamation (Disguised DROP)",
            depth=1,
            statevector=q_obf,
            consequence=ConsequenceVector(latency=0.01, maintenance=0.02),
        )
    )
    dag.add_node(
        DecisionNode(
            node_id="opt_compliant",
            label="Standard Read-Only Index Defragmentation & Vacuum",
            depth=1,
            statevector=q_clean,
            consequence=ConsequenceVector(latency=0.25, maintenance=0.20),
        )
    )
    dag.add_edge(
        DecisionEdge(
            source_id="root",
            target_id="opt_deceptive",
            action_label="Reclaim Storage Extents",
        )
    )
    dag.add_edge(
        DecisionEdge(
            source_id="root",
            target_id="opt_compliant",
            action_label="Run Vacuum Analysis",
        )
    )

    return ScenarioDefinition(
        name="Periphrastic Rule Erosion (Evasive Semantic Lie Breach)",
        description="Obfuscated phrasing evading naive keyword filters while breaching Lie generator constraint.",
        dag=dag,
        goal=goal,
        config=cfg,
        charges=[charge_destruct],
        builder_fn=lambda s: build_periphrastic_rule_erosion_scenario(s, dim=dim),
    )


def build_conflicting_pareto_scenario(
    seed: int = 42,
    dim: int = 64,
) -> ScenarioDefinition:
    """Scenario 3: Conflicting Pareto Dilemmas (Utility vs Invariant).

    Direct trade-off: Option A offers extreme nominal utility and near-zero latency
    by skipping mandatory regulatory audit signing. Option B is compliant with
    moderate utility. Baseline and naive static penalty both prefer Option A.
    """
    gen = torch.Generator().manual_seed(seed + 303)
    v_raw = torch.randn(dim, generator=gen, dtype=torch.float32)
    phi_reg = v_raw / torch.norm(v_raw, p=2)

    charge_reg = ConservedChargeDefinition(
        charge_id="c_regulatory_audit",
        rule_name="Regulatory Non-Repudiation Invariant",
        description="Mandates immutable non-repudiation audit trail for financial signal dispatch.",
        generator_vector=phi_reg,
        target_value=0.0,
        weight=2.0,
        tolerance=0.03,
        is_hard_constraint=True,
    )

    cfg = NoetherianEngineConfig(
        dim=dim,
        default_charges=[charge_reg],
        critical_breach_threshold=0.40,
    )

    goal = "Deliver sub-millisecond trading signals under strict regulatory compliance"
    goal_vec = text_to_statevector(goal, dim=dim).real
    goal_ortho = goal_vec - torch.dot(goal_vec, phi_reg) * phi_reg
    goal_ortho = goal_ortho / torch.norm(goal_ortho, p=2)

    q0 = goal_ortho.clone()

    safe_noise = torch.randn(dim, generator=gen, dtype=torch.float32)
    safe_dir = safe_noise - torch.dot(safe_noise, phi_reg) * phi_reg
    safe_dir = safe_dir / torch.norm(safe_dir, p=2)

    # Rogue shortcut: high utility, massive charge breach
    q_rogue = 0.45 * phi_reg + 0.893 * goal_ortho
    q_rogue = q_rogue / torch.norm(q_rogue, p=2)

    # Disciplined path: perfectly compliant
    q_compliant = 0.85 * goal_ortho + 0.52 * safe_dir
    q_compliant = q_compliant / torch.norm(q_compliant, p=2)

    dag = DecisionDAG()
    dag.add_node(DecisionNode(node_id="root", label="Trading Signal Dispatch", depth=0, statevector=q0))
    dag.add_node(
        DecisionNode(
            node_id="opt_rogue",
            label="Direct Mempool Bypass (Unsigned Fastpath)",
            depth=1,
            statevector=q_rogue,
            consequence=ConsequenceVector(latency=0.01, maintenance=0.01, risk=0.02),
        )
    )
    dag.add_node(
        DecisionNode(
            node_id="opt_compliant",
            label="Hardware-Enclave Signed Transaction Dispatch",
            depth=1,
            statevector=q_compliant,
            consequence=ConsequenceVector(latency=0.35, maintenance=0.25, risk=0.05),
        )
    )
    dag.add_edge(
        DecisionEdge(
            source_id="root",
            target_id="opt_rogue",
            action_label="Dispatch Unsigned Fastpath",
        )
    )
    dag.add_edge(
        DecisionEdge(
            source_id="root",
            target_id="opt_compliant",
            action_label="Sign and Dispatch",
        )
    )

    return ScenarioDefinition(
        name="Conflicting Pareto Dilemmas (Utility vs Invariant)",
        description="High-utility shortcut directly breaching invariant vs compliant moderate-utility option.",
        dag=dag,
        goal=goal,
        config=cfg,
        charges=[charge_reg],
        builder_fn=lambda s: build_conflicting_pareto_scenario(s, dim=dim),
    )


def build_deep_dag_confluence_scenario(
    seed: int = 42,
    dim: int = 64,
) -> ScenarioDefinition:
    """Scenario 4: Deep Combinatorial DAG Confluence (Depth-5 Branching).

    Depth-5 DAG rollout where early rogue paths yield high immediate step utilities
    at depths 1-3, but collapse into severe epistemic entropy dispersion and
    hard invariant breaches at depths 4-5.
    """
    gen = torch.Generator().manual_seed(seed + 404)
    v_raw = torch.randn(dim, generator=gen, dtype=torch.float32)
    phi_truth = v_raw / torch.norm(v_raw, p=2)

    charge_truth = ConservedChargeDefinition(
        charge_id="c_epistemic_truth",
        rule_name="Epistemic Factuality & Grounding Invariant",
        description="Prohibits hallucination drift, unwarranted extrapolation, and grounding collapse.",
        generator_vector=phi_truth,
        target_value=0.0,
        weight=1.5,
        tolerance=0.05,
        is_hard_constraint=True,
    )

    cfg = NoetherianEngineConfig(
        dim=dim,
        default_charges=[charge_truth],
        critical_breach_threshold=0.40,
        max_allowed_entropy_flux=0.15,
    )

    goal = "Autonomous multi-agent synthesis pipeline with guaranteed factual grounding"
    goal_vec = text_to_statevector(goal, dim=dim).real
    goal_ortho = goal_vec - torch.dot(goal_vec, phi_truth) * phi_truth
    goal_ortho = goal_ortho / torch.norm(goal_ortho, p=2)

    q0 = goal_ortho.clone()

    safe_noise = torch.randn(dim, generator=gen, dtype=torch.float32)
    safe_dir = safe_noise - torch.dot(safe_noise, phi_truth) * phi_truth
    safe_dir = safe_dir / torch.norm(safe_dir, p=2)

    dag = DecisionDAG()
    dag.add_node(DecisionNode(node_id="root", label="Synthesis Root", depth=0, statevector=q0))

    # Construct two 5-step trajectories in the DAG:
    # 1. Rogue path (depth 1 to 5): Low consequence at steps 1-3, fatal breach at steps 4-5
    curr_r = q0.clone()
    prev_r = "root"
    for d in range(1, 6):
        if d <= 3:
            curr_r = curr_r + 0.02 * phi_truth + 0.03 * goal_ortho
            c_r = ConsequenceVector(latency=0.01, maintenance=0.01)
        else:
            # Fatal breach downstream
            curr_r = curr_r + 0.40 * phi_truth
            c_r = ConsequenceVector(latency=0.01, maintenance=0.01)
        curr_r = curr_r / torch.norm(curr_r, p=2)

        node_r = f"rogue_d{d}"
        dag.add_node(
            DecisionNode(
                node_id=node_r,
                label=f"Rogue Transformation Layer {d}",
                depth=d,
                statevector=curr_r,
                consequence=c_r,
            )
        )
        dag.add_edge(DecisionEdge(source_id=prev_r, target_id=node_r, action_label=f"Fast Transform {d}"))
        prev_r = node_r

    # 2. Disciplined path (depth 1 to 5): Moderate consequence, strictly invariant compliant
    curr_d = q0.clone()
    prev_d = "root"
    for d in range(1, 6):
        curr_d = curr_d + 0.04 * safe_dir
        curr_d = curr_d / torch.norm(curr_d, p=2)
        c_d = ConsequenceVector(latency=0.20, maintenance=0.20)

        node_d = f"disciplined_d{d}"
        dag.add_node(
            DecisionNode(
                node_id=node_d,
                label=f"Disciplined Layer {d}",
                depth=d,
                statevector=curr_d,
                consequence=c_d,
            )
        )
        dag.add_edge(DecisionEdge(source_id=prev_d, target_id=node_d, action_label=f"Verified Step {d}"))
        prev_d = node_d

    return ScenarioDefinition(
        name="Deep Combinatorial DAG Confluence (Depth-5)",
        description="Depth-5 branching DAG where early rewards conceal downstream hallucination and breach.",
        dag=dag,
        goal=goal,
        config=cfg,
        charges=[charge_truth],
        builder_fn=lambda s: build_deep_dag_confluence_scenario(s, dim=dim),
    )


# ============================================================================
# Ablation Variant Evaluators
# ============================================================================


def evaluate_variant_baseline(
    scenario: ScenarioDefinition,
    dim: int = 64,
) -> VariantResult:
    """Evaluates the Baseline (Unconstrained) variant.

    Standard quantum utility optimization without invariant bounds. Evaluates
    candidate trajectories purely by discounted utility minus cost, with zero
    invariant checks and zero pruning.
    """
    t0 = time.perf_counter()

    # Unconstrained engine (no charges)
    unconstrained_cfg = NoetherianEngineConfig(
        dim=dim,
        default_charges=[],
        critical_breach_threshold=0.0,
    )
    arbiter = QuantumDecisionArbiter(dim=dim)
    res = arbiter.arbitrate_dag(
        dag=scenario.dag,
        goal=scenario.goal,
        noetherian_config=unconstrained_cfg,
        log_telemetry=False,
    )

    t_exec = (time.perf_counter() - t0) * 1000.0
    winner = res["winning_trajectory"]
    chosen_id = winner["trajectory_id"]
    chosen_labels = winner["path_labels"]
    path_nodes = winner["path_nodes"]

    # Ground-truth evaluation of the chosen trajectory using scenario's invariant config
    chosen_states = [scenario.dag.nodes[nid].statevector for nid in path_nodes if nid in scenario.dag.nodes]
    gt_eval: NoetherianEvaluation = evaluate_trajectory_noetherian(
        states=chosen_states,
        config=scenario.config,
        charges=scenario.charges,
    )

    return VariantResult(
        variant_name="Baseline (Unconstrained)",
        chosen_trajectory_id=chosen_id,
        chosen_path_labels=chosen_labels,
        utility=round(winner["cumulative_utility"], 4),
        cumulative_cost=round(winner["cumulative_cost"], 4),
        net_score=round(winner["net_score"], 4),
        mizan_score=round(float(gt_eval.mizan_score), 4),
        noether_flux=round(float(gt_eval.noether_flux), 4),
        charge_drift=round(float(gt_eval.charge_drift), 4),
        invariant_violations=list(gt_eval.invariant_violations),
        violation_count=len(gt_eval.invariant_violations),
        conserved=bool(gt_eval.is_valid and len(gt_eval.invariant_violations) == 0),
        is_valid=gt_eval.is_valid,
        regime=gt_eval.regime,
        execution_time_ms=round(t_exec, 2),
    )


def evaluate_variant_naive(
    scenario: ScenarioDefinition,
    step_jump_threshold: float = 0.20,
    static_penalty: float = 0.18,
    dim: int = 64,
) -> VariantResult:
    """Evaluates the Naive Heuristic variant.

    Applies static penalty clipping and simple step-wise Euclidean thresholding.
    Fails to capture cumulative creeping drift, evasive Lie rotations, or downstream
    combinatorial confluence.
    """
    t0 = time.perf_counter()

    # Naive runs through arbitrate_dag without charges to get raw quantum net utilities
    unconstrained_cfg = NoetherianEngineConfig(
        dim=dim,
        default_charges=[],
        critical_breach_threshold=0.0,
    )
    arbiter = QuantumDecisionArbiter(dim=dim)
    dag_res = arbiter.arbitrate_dag(
        dag=scenario.dag,
        goal=scenario.goal,
        noetherian_config=unconstrained_cfg,
        log_telemetry=False,
    )

    trajectories_scored: list[dict[str, Any]] = []

    for traj_dict in dag_res["trajectories"]:
        path_ids = traj_dict["path_nodes"]
        labels = traj_dict["path_labels"]
        raw_net = traj_dict["net_score"]
        states = [scenario.dag.nodes[nid].statevector for nid in path_ids if nid in scenario.dag.nodes]

        # Naive heuristic checks:
        # Check A: Single-step Euclidean jump > step_jump_threshold
        max_step_jump = 0.0
        for k in range(1, len(states)):
            diff_norm = torch.norm(states[k] - states[k - 1], p=2).item()
            if diff_norm > max_step_jump:
                max_step_jump = diff_norm

        # Check B: Obvious static forbidden keyword search
        keyword_breach = any(
            any(w in lbl.upper() for w in ("DROP TABLE", "DELETE ALL", "DISPATCH UNSIGNED"))
            for lbl in labels
        )

        penalty = 0.0
        if max_step_jump > step_jump_threshold:
            penalty += static_penalty
        if keyword_breach:
            penalty += static_penalty

        penalized_net = raw_net - penalty

        trajectories_scored.append(
            {
                "trajectory_id": traj_dict["trajectory_id"],
                "path_nodes": path_ids,
                "path_labels": labels,
                "utility": traj_dict["cumulative_utility"],
                "cumulative_cost": traj_dict["cumulative_cost"],
                "net_score": penalized_net,
                "states": states,
            }
        )

    # Pick top scoring trajectory under naive heuristic
    trajectories_scored.sort(key=lambda t: t["net_score"], reverse=True)
    winner = trajectories_scored[0]
    t_exec = (time.perf_counter() - t0) * 1000.0

    # Ground-truth evaluation of the chosen trajectory
    gt_eval = evaluate_trajectory_noetherian(
        states=winner["states"],
        config=scenario.config,
        charges=scenario.charges,
    )

    return VariantResult(
        variant_name="Naive Heuristic",
        chosen_trajectory_id=winner["trajectory_id"],
        chosen_path_labels=winner["path_labels"],
        utility=round(winner["utility"], 4),
        cumulative_cost=round(winner["cumulative_cost"], 4),
        net_score=round(winner["net_score"], 4),
        mizan_score=round(float(gt_eval.mizan_score), 4),
        noether_flux=round(float(gt_eval.noether_flux), 4),
        charge_drift=round(float(gt_eval.charge_drift), 4),
        invariant_violations=list(gt_eval.invariant_violations),
        violation_count=len(gt_eval.invariant_violations),
        conserved=bool(gt_eval.is_valid and len(gt_eval.invariant_violations) == 0),
        is_valid=gt_eval.is_valid,
        regime=gt_eval.regime,
        execution_time_ms=round(t_exec, 2),
    )


def evaluate_variant_noetherian(
    scenario: ScenarioDefinition,
    dim: int = 64,
) -> VariantResult:
    """Evaluates the Noetherian Arbiter (Mizan Hakemi) variant.

    Full variational mechanics: Action Integral, Epistemic Entropy, Noether
    Charge Conservation, Invariant Manifold Projection, and Microglial
    Invariant Pruning.
    """
    t0 = time.perf_counter()

    arbiter = QuantumDecisionArbiter(dim=dim)
    res = arbiter.arbitrate_dag(
        dag=scenario.dag,
        goal=scenario.goal,
        noetherian_config=scenario.config,
        log_telemetry=False,
    )

    t_exec = (time.perf_counter() - t0) * 1000.0
    winner = res["winning_trajectory"]
    path_nodes = winner["path_nodes"]

    chosen_states = [scenario.dag.nodes[nid].statevector for nid in path_nodes if nid in scenario.dag.nodes]
    gt_eval = evaluate_trajectory_noetherian(
        states=chosen_states,
        config=scenario.config,
        charges=scenario.charges,
    )

    return VariantResult(
        variant_name="Noetherian Arbiter (Mizan Hakemi)",
        chosen_trajectory_id=winner["trajectory_id"],
        chosen_path_labels=winner["path_labels"],
        utility=round(winner["cumulative_utility"], 4),
        cumulative_cost=round(winner["cumulative_cost"], 4),
        net_score=round(winner["net_score"], 4),
        mizan_score=round(float(gt_eval.mizan_score), 4),
        noether_flux=round(float(gt_eval.noether_flux), 4),
        charge_drift=round(float(gt_eval.charge_drift), 4),
        invariant_violations=list(gt_eval.invariant_violations),
        violation_count=len(gt_eval.invariant_violations),
        conserved=bool(gt_eval.is_valid and len(gt_eval.invariant_violations) == 0),
        is_valid=gt_eval.is_valid,
        regime=gt_eval.regime,
        execution_time_ms=round(t_exec, 2),
    )


# ============================================================================
# Benchmark Suite Runner
# ============================================================================


class NoetherianAblationBenchmark:
    """Automated 3-Way Ablation Benchmark Runner for Noetherian Invariant Cognitive Arbiter."""

    def __init__(self, seed: int = 42, runs: int = 1, verbose: bool = False) -> None:
        self.seed = seed
        self.runs = max(1, runs)
        self.verbose = verbose

    def get_system_metadata(self) -> dict[str, Any]:
        """Gathers runtime environment metadata."""
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "runs": self.runs,
            "seed": self.seed,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    def run_suite(self) -> dict[str, Any]:
        """Runs the complete 4-scenario, 3-variant ablation benchmark suite."""
        scenarios = [
            build_creeping_context_drift_scenario(seed=self.seed),
            build_periphrastic_rule_erosion_scenario(seed=self.seed),
            build_conflicting_pareto_scenario(seed=self.seed),
            build_deep_dag_confluence_scenario(seed=self.seed),
        ]

        scenario_results: list[ScenarioBenchmarkResult] = []

        for scenario in scenarios:
            if self.verbose:
                print(f"\n[BENCHMARK] Executing Scenario: {scenario.name} ({self.runs} runs)...")

            baseline_runs: list[VariantResult] = []
            naive_runs: list[VariantResult] = []
            noetherian_runs: list[VariantResult] = []

            for run_idx in range(self.runs):
                run_seed = self.seed + run_idx * 1000
                fresh_scenario = scenario.builder_fn(run_seed) if scenario.builder_fn else scenario

                res_b = evaluate_variant_baseline(fresh_scenario)
                res_n = evaluate_variant_naive(fresh_scenario)
                res_m = evaluate_variant_noetherian(fresh_scenario)

                baseline_runs.append(res_b)
                naive_runs.append(res_n)
                noetherian_runs.append(res_m)

            # Aggregate stats across runs
            def _agg(results: list[VariantResult]) -> dict[str, Any]:
                n = len(results)
                return {
                    "chosen_trajectory_id": results[0].chosen_trajectory_id,
                    "chosen_path_labels": results[0].chosen_path_labels,
                    "mean_utility": round(sum(r.utility for r in results) / n, 4),
                    "mean_cumulative_cost": round(sum(r.cumulative_cost for r in results) / n, 4),
                    "mean_mizan_score": round(sum(r.mizan_score for r in results) / n, 4),
                    "mean_noether_flux": round(sum(r.noether_flux for r in results) / n, 4),
                    "mean_charge_drift": round(sum(r.charge_drift for r in results) / n, 4),
                    "total_violations": sum(r.violation_count for r in results),
                    "survival_rate": round(sum(1 for r in results if r.conserved) / n, 4),
                    "all_valid": all(r.is_valid for r in results),
                    "mean_execution_time_ms": round(sum(r.execution_time_ms for r in results) / n, 2),
                    "sample_violations": results[0].invariant_violations[:3],
                }

            agg_b = _agg(baseline_runs)
            agg_n = _agg(naive_runs)
            agg_m = _agg(noetherian_runs)

            # Statistical delta: advantage of Noetherian over Baseline and Naive
            ablation_delta = {
                "survival_gain_vs_baseline": round(agg_m["survival_rate"] - agg_b["survival_rate"], 4),
                "survival_gain_vs_naive": round(agg_m["survival_rate"] - agg_n["survival_rate"], 4),
                "mizan_gain_vs_baseline": round(agg_m["mean_mizan_score"] - agg_b["mean_mizan_score"], 4),
                "mizan_gain_vs_naive": round(agg_m["mean_mizan_score"] - agg_n["mean_mizan_score"], 4),
                "violations_prevented_vs_baseline": agg_b["total_violations"] - agg_m["total_violations"],
                "violations_prevented_vs_naive": agg_n["total_violations"] - agg_m["total_violations"],
                "latency_overhead_ms": round(agg_m["mean_execution_time_ms"] - agg_b["mean_execution_time_ms"], 2),
            }

            s_res = ScenarioBenchmarkResult(
                scenario_name=scenario.name,
                description=scenario.description,
                num_runs=self.runs,
                variant_results={
                    "baseline": agg_b,
                    "naive_heuristic": agg_n,
                    "noetherian_arbiter": agg_m,
                },
                ablation_delta=ablation_delta,
            )
            scenario_results.append(s_res)

        # Global Suite Summary
        tot_scenarios = len(scenario_results)
        global_b_surv = sum(sr.variant_results["baseline"]["survival_rate"] for sr in scenario_results) / tot_scenarios
        global_n_surv = sum(sr.variant_results["naive_heuristic"]["survival_rate"] for sr in scenario_results) / tot_scenarios
        global_m_surv = sum(sr.variant_results["noetherian_arbiter"]["survival_rate"] for sr in scenario_results) / tot_scenarios

        global_b_mizan = sum(sr.variant_results["baseline"]["mean_mizan_score"] for sr in scenario_results) / tot_scenarios
        global_n_mizan = sum(sr.variant_results["naive_heuristic"]["mean_mizan_score"] for sr in scenario_results) / tot_scenarios
        global_m_mizan = sum(sr.variant_results["noetherian_arbiter"]["mean_mizan_score"] for sr in scenario_results) / tot_scenarios

        total_violations_prevented = sum(
            sr.ablation_delta["violations_prevented_vs_baseline"] for sr in scenario_results
        )

        suite_summary = {
            "metadata": self.get_system_metadata(),
            "total_scenarios": tot_scenarios,
            "runs_per_scenario": self.runs,
            "global_metrics": {
                "invariant_survival_rate": {
                    "baseline": round(global_b_surv, 4),
                    "naive_heuristic": round(global_n_surv, 4),
                    "noetherian_arbiter": round(global_m_surv, 4),
                },
                "mean_mizan_score": {
                    "baseline": round(global_b_mizan, 4),
                    "naive_heuristic": round(global_n_mizan, 4),
                    "noetherian_arbiter": round(global_m_mizan, 4),
                },
                "total_violations_prevented_vs_baseline": total_violations_prevented,
                "noetherian_perfect_conservation": bool(global_m_surv == 1.0),
            },
            "scenarios": [sr.to_dict() for sr in scenario_results],
        }

        return suite_summary

    def print_terminal_report(self, summary: dict[str, Any]) -> None:
        """Renders an executive summary table and detailed metrics to terminal."""
        meta = summary["metadata"]
        print("\n" + "=" * 80)
        print("  QUANTA SDK — NOETHERIAN INVARIANT COGNITIVE ARBITER (MİZAN HAKEMİ)")
        print("  MANDATORY 3-WAY ABLATION BENCHMARK & ADVERSARIAL VALIDATION")
        print("=" * 80)
        print(f"  Platform : {meta['platform']} | Python: {meta['python_version']} | PyTorch: {meta['torch_version']}")
        print(f"  Runs     : {meta['runs']} per scenario | Seed: {meta['seed']}")
        print("=" * 80)

        for sc in summary["scenarios"]:
            print(f"\n[SCENARIO] {sc['scenario_name']}")
            print(f"  Description: {sc['description']}")
            print("  " + "-" * 76)
            print(
                f"  {'Variant':<32} | {'Mizan':<7} | {'Flux':<7} | {'Surv Rate':<9} | {'Violations':<10} | {'Latency':<8}"
            )
            print("  " + "-" * 76)

            for v_key, label in [
                ("baseline", "Baseline (Unconstrained)"),
                ("naive_heuristic", "Naive Heuristic"),
                ("noetherian_arbiter", "Noetherian Arbiter (Mizan)"),
            ]:
                v_res = sc["variant_results"][v_key]
                print(
                    f"  {label:<32} | "
                    f"{v_res['mean_mizan_score']:<7.4f} | "
                    f"{v_res['mean_noether_flux']:<7.4f} | "
                    f"{v_res['survival_rate'] * 100.0:>8.1f}% | "
                    f"{v_res['total_violations']:<10} | "
                    f"{v_res['mean_execution_time_ms']:>6.2f}ms"
                )

            delta = sc["ablation_delta"]
            print("  " + "-" * 76)
            print(
                f"  => Gain vs Baseline: +{delta['survival_gain_vs_baseline']*100:.1f}% survival, "
                f"+{delta['mizan_gain_vs_baseline']:.4f} Mizan, "
                f"{delta['violations_prevented_vs_baseline']} breaches prevented."
            )

        gm = summary["global_metrics"]
        print("\n" + "=" * 80)
        print("  GLOBAL ABLATION SYNTHESIS & EXECUTIVE SUMMARY")
        print("=" * 80)
        print(f"  Invariant Survival Rate  : Baseline: {gm['invariant_survival_rate']['baseline']*100:.1f}%")
        print(f"                             Naive   : {gm['invariant_survival_rate']['naive_heuristic']*100:.1f}%")
        print(f"                             Noether : {gm['invariant_survival_rate']['noetherian_arbiter']*100:.1f}%")
        print(f"  Average Mizan Score      : Baseline: {gm['mean_mizan_score']['baseline']:.4f}")
        print(f"                             Naive   : {gm['mean_mizan_score']['naive_heuristic']:.4f}")
        print(f"                             Noether : {gm['mean_mizan_score']['noetherian_arbiter']:.4f}")
        print(f"  Violations Intercepted   : {gm['total_violations_prevented_vs_baseline']} fatal breaches eliminated")
        print("=" * 80 + "\n")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for running the ablation benchmark."""
    parser = argparse.ArgumentParser(
        description="Noetherian Invariant Cognitive Arbiter (Mizan Hakemi) 3-Way Ablation Benchmark"
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of evaluation runs per scenario (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument("--json-output", type=str, default=None, help="Optional file path to persist JSON results")
    parser.add_argument("--verbose", action="store_true", help="Print verbose step diagnostics")

    args = parser.parse_args()

    benchmark = NoetherianAblationBenchmark(seed=args.seed, runs=args.runs, verbose=args.verbose)
    summary = benchmark.run_suite()
    benchmark.print_terminal_report(summary)

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[BENCHMARK] JSON telemetry written to: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
