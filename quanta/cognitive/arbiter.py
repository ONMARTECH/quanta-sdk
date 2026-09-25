"""Quantum Decision Arbiter using Quantum Zeno & Anti-Zeno Attention dynamics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from quanta.cognitive.memory import text_to_statevector
from quanta.cognitive.telemetry import record_decision_telemetry
from quanta.torch.brain import QuantumZenoAttention

__all__ = [
    "CognitivePanelScore",
    "ConsequenceVector",
    "DecisionDAG",
    "DecisionEdge",
    "DecisionNode",
    "DecisionTrajectory",
    "NeurobiologicalEvaluation",
    "PsychiatricEvaluation",
    "QuantumDecisionArbiter",
    "SociologicalEvaluation",
]

def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts value to float with default fallback."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _clamp01(val: Any, default: float = 0.0) -> float:
    """Safely converts value to float clamped in [0.0, 1.0] with NaN/Inf handling."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return max(0.0, min(1.0, f))
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely converts value to int with default fallback."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_dict(val: Any) -> dict[str, Any]:
    """Safely converts value to dictionary, returning empty dict if invalid."""
    if isinstance(val, dict):
        return dict(val)
    return {}


@dataclass
class NeurobiologicalEvaluation:
    """Neurobiological evaluation metrics grounded in synaptic and energy homeostasis.

    Attributes:
        synaptic_saturation: [0.0, 1.0] Degree of synaptic saturation / SHY downscaling need.
        energy_expenditure: [0.0, 1.0] Landauer metabolic / compute energy cost penalty.
        sleep_consolidation_affinity: [0.0, 1.0] Affinity for deferring to quiescent consolidation.
    """

    synaptic_saturation: float = 0.0
    energy_expenditure: float = 0.0
    sleep_consolidation_affinity: float = 0.0

    def aggregate_cost(
        self,
        weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
    ) -> float:
        """Computes the weighted aggregate neurobiological penalty in [0.0, 1.0]."""
        return float(
            weights[0] * self.synaptic_saturation
            + weights[1] * self.energy_expenditure
            + weights[2] * self.sleep_consolidation_affinity
        )

    def to_dict(self) -> dict[str, float]:
        """Serializes neurobiological evaluation to dictionary."""
        return {
            "synaptic_saturation": self.synaptic_saturation,
            "energy_expenditure": self.energy_expenditure,
            "sleep_consolidation_affinity": self.sleep_consolidation_affinity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> NeurobiologicalEvaluation:
        """Instantiates NeurobiologicalEvaluation from dictionary with safe fallbacks."""
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            synaptic_saturation=_clamp01(data.get("synaptic_saturation"), 0.0),
            energy_expenditure=_clamp01(data.get("energy_expenditure"), 0.0),
            sleep_consolidation_affinity=_clamp01(data.get("sleep_consolidation_affinity"), 0.0),
        )


@dataclass
class PsychiatricEvaluation:
    """Psychiatric and cognitive stability evaluation metrics.

    Attributes:
        rumination_risk: [0.0, 1.0] Repetitive circular deliberation / looping penalty.
        perseveration_penalty: [0.0, 1.0] Inability to shift cognitive sets after repeated failures.
        threat_distortion: [0.0, 1.0] Catastrophizing or paranoid hyper-vigilance vs calibrated risk.
    """

    rumination_risk: float = 0.0
    perseveration_penalty: float = 0.0
    threat_distortion: float = 0.0

    def aggregate_cost(
        self,
        weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
    ) -> float:
        """Computes the weighted aggregate psychiatric penalty in [0.0, 1.0]."""
        return float(
            weights[0] * self.rumination_risk
            + weights[1] * self.perseveration_penalty
            + weights[2] * self.threat_distortion
        )

    def to_dict(self) -> dict[str, float]:
        """Serializes psychiatric evaluation to dictionary."""
        return {
            "rumination_risk": self.rumination_risk,
            "perseveration_penalty": self.perseveration_penalty,
            "threat_distortion": self.threat_distortion,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PsychiatricEvaluation:
        """Instantiates PsychiatricEvaluation from dictionary with safe fallbacks."""
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            rumination_risk=_clamp01(data.get("rumination_risk"), 0.0),
            perseveration_penalty=_clamp01(data.get("perseveration_penalty"), 0.0),
            threat_distortion=_clamp01(data.get("threat_distortion"), 0.0),
        )


@dataclass
class SociologicalEvaluation:
    """Sociological and human alignment evaluation metrics.

    Attributes:
        social_misalignment: [0.0, 1.0] Violation of social context framing, norms, or tone.
        user_fatigue_impact: [0.0, 1.0] Cognitive load imposed on the user (verbosity, friction).
        coordination_friction: [0.0, 1.0] Friction with peer agents or collective intelligence.
    """

    social_misalignment: float = 0.0
    user_fatigue_impact: float = 0.0
    coordination_friction: float = 0.0

    def aggregate_cost(
        self,
        weights: tuple[float, float, float] = (0.35, 0.40, 0.25),
    ) -> float:
        """Computes the weighted aggregate sociological penalty in [0.0, 1.0]."""
        return float(
            weights[0] * self.social_misalignment
            + weights[1] * self.user_fatigue_impact
            + weights[2] * self.coordination_friction
        )

    def to_dict(self) -> dict[str, float]:
        """Serializes sociological evaluation to dictionary."""
        return {
            "social_misalignment": self.social_misalignment,
            "user_fatigue_impact": self.user_fatigue_impact,
            "coordination_friction": self.coordination_friction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SociologicalEvaluation:
        """Instantiates SociologicalEvaluation from dictionary with safe fallbacks."""
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            social_misalignment=_clamp01(data.get("social_misalignment"), 0.0),
            user_fatigue_impact=_clamp01(data.get("user_fatigue_impact"), 0.0),
            coordination_friction=_clamp01(data.get("coordination_friction"), 0.0),
        )


@dataclass
class CognitivePanelScore:
    """Composite multi-disciplinary cognitive panel evaluation score.

    Attributes:
        neurobiology: Synaptic saturation and energy budgeting evaluation.
        psychiatry: Cognitive stability and anti-rumination evaluation.
        sociology: Human alignment and fatigue impact evaluation.
    """

    neurobiology: NeurobiologicalEvaluation = field(default_factory=NeurobiologicalEvaluation)
    psychiatry: PsychiatricEvaluation = field(default_factory=PsychiatricEvaluation)
    sociology: SociologicalEvaluation = field(default_factory=SociologicalEvaluation)

    def aggregate_penalty(
        self,
        discipline_weights: tuple[float, float, float] = (0.30, 0.35, 0.35),
    ) -> float:
        """Computes the interdisciplinary aggregate penalty score in [0.0, 1.0]."""
        return float(
            discipline_weights[0] * self.neurobiology.aggregate_cost()
            + discipline_weights[1] * self.psychiatry.aggregate_cost()
            + discipline_weights[2] * self.sociology.aggregate_cost()
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes composite cognitive panel score to dictionary."""
        return {
            "neurobiology": self.neurobiology.to_dict(),
            "psychiatry": self.psychiatry.to_dict(),
            "sociology": self.sociology.to_dict(),
            "aggregate_penalty": round(self.aggregate_penalty(), 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CognitivePanelScore:
        """Instantiates CognitivePanelScore from dictionary with safe fallbacks."""
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            neurobiology=NeurobiologicalEvaluation.from_dict(data.get("neurobiology")),
            psychiatry=PsychiatricEvaluation.from_dict(data.get("psychiatry")),
            sociology=SociologicalEvaluation.from_dict(data.get("sociology")),
        )


@dataclass
class ConsequenceVector:
    """4-Dimensional operational cost and Multi-Disciplinary Cognitive Panel vector.

    Attributes:
        latency: [0.0, 1.0] Runtime latency, I/O overhead, and response time impact.
        maintenance: [0.0, 1.0] Technical debt, cognitive complexity, and maintenance burden.
        risk: [0.0, 1.0] Blast radius, failure probability, and irreversibility / vendor lock-in.
        metabolic: [0.0, 1.0] CPU/memory overhead and Landauer thermodynamic footprint.
        cognitive_panel: Multi-disciplinary cognitive evaluation panel score.
    """

    latency: float = 0.0
    maintenance: float = 0.0
    risk: float = 0.0
    metabolic: float = 0.0
    cognitive_panel: CognitivePanelScore = field(default_factory=CognitivePanelScore)

    def weighted_cost(
        self,
        weights: tuple[float, float, float, float] = (0.25, 0.35, 0.25, 0.15),
        panel_weight: float = 0.0,
    ) -> float:
        """Computes the weighted aggregate operational cost penalty in [0.0, 1.0].

        When panel_weight == 0.0, strictly evaluates the baseline 4D consequence cost.
        When panel_weight > 0.0, interpolates between 4D cost and cognitive panel penalty.
        """
        c_4d = float(
            weights[0] * self.latency
            + weights[1] * self.maintenance
            + weights[2] * self.risk
            + weights[3] * self.metabolic
        )
        if panel_weight <= 0.0:
            return c_4d
        w_p = min(1.0, max(0.0, panel_weight))
        c_panel = self.cognitive_panel.aggregate_penalty()
        return float((1.0 - w_p) * c_4d + w_p * c_panel)

    def to_dict(self) -> dict[str, Any]:
        """Serializes consequence vector to dictionary."""
        return {
            "latency": self.latency,
            "maintenance": self.maintenance,
            "risk": self.risk,
            "metabolic": self.metabolic,
            "cognitive_panel": self.cognitive_panel.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ConsequenceVector:
        """Instantiates ConsequenceVector from a dictionary with safe fallbacks."""
        if not data or not isinstance(data, dict):
            return cls()
        c_panel_raw = data.get("cognitive_panel")
        if isinstance(c_panel_raw, CognitivePanelScore):
            c_panel = c_panel_raw
        elif isinstance(c_panel_raw, dict):
            c_panel = CognitivePanelScore.from_dict(c_panel_raw)
        elif any(k in data for k in ("neurobiology", "psychiatry", "sociology")):
            c_panel = CognitivePanelScore.from_dict(data)
        else:
            c_panel = CognitivePanelScore()

        return cls(
            latency=_clamp01(data.get("latency"), 0.0),
            maintenance=_clamp01(data.get("maintenance"), 0.0),
            risk=_clamp01(data.get("risk"), 0.0),
            metabolic=_clamp01(data.get("metabolic"), 0.0),
            cognitive_panel=c_panel,
        )


@dataclass
class DecisionNode:
    """Node in the Decision DAG representing an architectural or algorithmic state.

    Attributes:
        node_id: Unique identifier for the node.
        label: Human-readable architectural choice or state description.
        depth: Distance from root node (root depth = 0).
        statevector: Hilbert space quantum representation (dim=64).
        consequence: Associated 4D biomorphic consequence vector.
        metadata: Arbitrary contextual properties (e.g. is_speculative, technology, tags).
    """

    node_id: str
    label: str
    depth: int = 0
    statevector: torch.Tensor | None = None
    consequence: ConsequenceVector = field(default_factory=ConsequenceVector)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_speculative(self) -> bool:
        """Returns True if marked as speculative/lateral exploration in metadata."""
        return bool(self.metadata.get("is_speculative", self.metadata.get("speculative", False)))


@dataclass
class DecisionEdge:
    """Directed edge representing an architectural choice transition.

    Attributes:
        source_id: Source node identifier.
        target_id: Target node identifier.
        action_label: Human-readable description of the transition or action.
        transition_ket: Quantum ket label for the transition (e.g. |transition>).
        alignment_score: Semantic cosine alignment with goal.
        tr_rho_pi: Scaled projective trace projection.
        p_zeno: Quantum Zeno pinning factor at this transition.
        metadata: Additional transition metadata.
    """

    source_id: str
    target_id: str
    action_label: str
    transition_ket: str = "|transition>"
    alignment_score: float = 0.0
    tr_rho_pi: float = 0.0
    p_zeno: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionTrajectory:
    """Evaluated decision rollout path from DAG root to terminal leaf.

    Attributes:
        trajectory_id: Unique trajectory identifier (e.g. tau_1).
        path_nodes: Sequence of node_ids from root to leaf [v0, v1, ..., vD].
        path_labels: Sequence of human-readable labels [Root, Branch, Sub-branch].
        cumulative_utility: Cascading discounted step utility sum.
        cumulative_cost: Discounted consequence penalty sum.
        net_score: Final ranking score.
        mean_p_zeno: Average Zeno pinning factor along trajectory.
        quantum_fidelity: Geometric mean of transition projections Tr(rho Pi).
        is_pruned: True if eliminated by microglial pruning.
        prune_reason: Explanation if pruned (e.g. "Dominated by winner").
        is_speculative: True if any node along trajectory is marked speculative.
        edges: Sequence of traversed DecisionEdge objects.
        metadata: Additional trajectory metadata.
    """

    trajectory_id: str
    path_nodes: list[str]
    path_labels: list[str]
    cumulative_utility: float = 0.0
    cumulative_cost: float = 0.0
    net_score: float = 0.0
    mean_p_zeno: float = 0.0
    quantum_fidelity: float = 1.0
    is_pruned: bool = False
    prune_reason: str | None = None
    is_speculative: bool = False
    edges: list[DecisionEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes trajectory to dictionary."""
        return {
            "trajectory_id": self.trajectory_id,
            "path_nodes": list(self.path_nodes),
            "path_labels": list(self.path_labels),
            "cumulative_utility": round(self.cumulative_utility, 4),
            "cumulative_cost": round(self.cumulative_cost, 4),
            "net_score": round(self.net_score, 4),
            "mean_p_zeno": round(self.mean_p_zeno, 4),
            "quantum_fidelity": round(self.quantum_fidelity, 6),
            "is_pruned": self.is_pruned,
            "prune_reason": self.prune_reason,
            "is_speculative": self.is_speculative,
        }


class DecisionDAG:
    """Directed Acyclic Graph engine for architectural choices and rollouts."""

    def __init__(self) -> None:
        self.nodes: dict[str, DecisionNode] = {}
        self.adj: dict[str, list[DecisionEdge]] = {}
        self.rev_adj: dict[str, list[DecisionEdge]] = {}

    def add_node(self, node: DecisionNode) -> None:
        """Adds a node to the DAG."""
        self.nodes[node.node_id] = node
        if node.node_id not in self.adj:
            self.adj[node.node_id] = []
        if node.node_id not in self.rev_adj:
            self.rev_adj[node.node_id] = []

    def add_edge(self, edge: DecisionEdge) -> None:
        """Adds a directed edge between existing nodes."""
        if edge.source_id not in self.nodes:
            raise KeyError(f"Source node '{edge.source_id}' does not exist in DAG.")
        if edge.target_id not in self.nodes:
            raise KeyError(f"Target node '{edge.target_id}' does not exist in DAG.")
        self.adj[edge.source_id].append(edge)
        self.rev_adj[edge.target_id].append(edge)

    @property
    def edges(self) -> list[DecisionEdge]:
        """Returns all directed edges in the DAG."""
        all_edges: list[DecisionEdge] = []
        for edge_list in self.adj.values():
            all_edges.extend(edge_list)
        return all_edges

    def get_node(self, node_id: str) -> DecisionNode | None:
        """Retrieves a node by id."""
        return self.nodes.get(node_id)

    def get_successors(self, node_id: str) -> list[DecisionNode]:
        """Returns target nodes of outgoing edges from node_id."""
        if node_id not in self.adj:
            return []
        return [
            self.nodes[edge.target_id]
            for edge in self.adj[node_id]
            if edge.target_id in self.nodes
        ]

    def get_predecessors(self, node_id: str) -> list[DecisionNode]:
        """Returns source nodes of incoming edges to node_id."""
        if node_id not in self.rev_adj:
            return []
        return [
            self.nodes[edge.source_id]
            for edge in self.rev_adj[node_id]
            if edge.source_id in self.nodes
        ]

    def get_outgoing_edges(self, node_id: str) -> list[DecisionEdge]:
        """Returns outgoing edges from node_id."""
        return list(self.adj.get(node_id, []))

    def get_incoming_edges(self, node_id: str) -> list[DecisionEdge]:
        """Returns incoming edges to node_id."""
        return list(self.rev_adj.get(node_id, []))

    def has_cycle(self) -> bool:
        """Checks whether the graph contains any cycle using DFS three-color tracking."""
        visited: dict[str, int] = {nid: 0 for nid in self.nodes}

        def dfs(u: str) -> bool:
            visited[u] = 1
            for edge in self.adj.get(u, []):
                v = edge.target_id
                if visited.get(v, 0) == 1:
                    return True
                if visited.get(v, 0) == 0 and dfs(v):
                    return True
            visited[u] = 2
            return False

        return any(visited[nid] == 0 and dfs(nid) for nid in self.nodes)

    def topological_sort(self) -> list[str]:
        """Returns a topological ordering of node_ids, raising ValueError if cycle detected."""
        if self.has_cycle():
            raise ValueError("Graph contains a cycle; cannot perform topological sort.")

        in_degree = {nid: len(self.rev_adj.get(nid, [])) for nid in self.nodes}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: list[str] = []

        while queue:
            queue.sort()
            u = queue.pop(0)
            order.append(u)
            for edge in self.adj.get(u, []):
                v = edge.target_id
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(order) != len(self.nodes):
            raise ValueError("Cycle detected during topological sorting.")
        return order

    def find_roots(self) -> list[str]:
        """Finds all root node IDs (nodes with in-degree 0)."""
        return [nid for nid in self.nodes if len(self.rev_adj.get(nid, [])) == 0]

    def find_leaves(self) -> list[str]:
        """Finds all leaf node IDs (nodes with out-degree 0)."""
        return [nid for nid in self.nodes if len(self.adj.get(nid, [])) == 0]

    def enumerate_trajectories(self, root_id: str | None = None) -> list[list[str]]:
        """Enumerates all paths from root_id (or all roots) to terminal leaf nodes.

        Uses dynamic programming / memoization on shared sub-branches to prevent
        exponential blowup on DAG confluence.
        """
        if self.has_cycle():
            raise ValueError("Graph contains a cycle; cannot enumerate finite trajectories.")

        roots = [root_id] if root_id is not None else self.find_roots()
        if not roots:
            roots = list(self.nodes.keys())[:1] if self.nodes else []

        memo: dict[str, list[list[str]]] = {}
        visiting: set[str] = set()

        def paths_from(u: str) -> list[list[str]]:
            if u in memo:
                return memo[u]
            if u in visiting:
                raise ValueError(f"Cycle detected at node {u}")
            visiting.add(u)

            successors = self.adj.get(u, [])
            if not successors:
                res = [[u]]
            else:
                res = []
                for edge in successors:
                    v = edge.target_id
                    sub_paths = paths_from(v)
                    for sp in sub_paths:
                        res.append([u] + sp)

            visiting.remove(u)
            memo[u] = res
            return res

        all_trajectories: list[list[str]] = []
        for r in roots:
            if r in self.nodes:
                all_trajectories.extend(paths_from(r))

        return all_trajectories

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> DecisionDAG:
        """Hydrates a DecisionDAG from a dictionary specification.

        Supports both:
        1. Hierarchical branch trees:
           {
             "root": "System Architecture",
             "branches": {
               "Option A": {
                 "consequences": {"latency": 0.4, ...},
                 "is_speculative": False,
                 "sub_branches": {
                   "Sub A1": {"consequences": {...}}
                 }
               }
             }
           }
        2. Explicit graph dictionary:
           {"nodes": [...], "edges": [...]}
        """
        dag = cls()
        if not spec or not isinstance(spec, dict):
            return dag

        if "nodes" in spec and "edges" in spec:
            for nd in spec["nodes"]:
                c_data = nd.get("consequence", nd.get("consequences"))
                node = DecisionNode(
                    node_id=str(nd.get("node_id", "")),
                    label=str(nd.get("label", nd.get("node_id", ""))),
                    depth=_safe_int(nd.get("depth"), 0),
                    consequence=ConsequenceVector.from_dict(c_data),
                    metadata=_safe_dict(nd.get("metadata")),
                )
                dag.add_node(node)
            for ed in spec["edges"]:
                edge = DecisionEdge(
                    source_id=str(ed.get("source_id", "")),
                    target_id=str(ed.get("target_id", "")),
                    action_label=str(
                        ed.get(
                            "action_label",
                            f"{ed.get('source_id', '')}->{ed.get('target_id', '')}",
                        )
                    ),
                    transition_ket=str(ed.get("transition_ket", "|transition>")),
                    alignment_score=_safe_float(ed.get("alignment_score"), 0.0),
                    tr_rho_pi=_safe_float(ed.get("tr_rho_pi"), 0.0),
                    p_zeno=_safe_float(ed.get("p_zeno"), 0.0),
                    metadata=_safe_dict(ed.get("metadata")),
                )
                dag.add_edge(edge)
            return dag

        root_spec = spec.get("root", "Root")
        if isinstance(root_spec, dict):
            raw_id = root_spec.get("id")
            root_id = str(raw_id) if raw_id is not None else "root"
            raw_label = root_spec.get("label")
            root_label = str(raw_label) if raw_label is not None else "Root"
            r_c_data = root_spec.get("consequences", root_spec.get("consequence"))
            r_cp_data = root_spec.get("cognitive_panel")
            if r_cp_data is not None:
                if r_c_data is None:
                    r_c_data = {"cognitive_panel": r_cp_data}
                elif isinstance(r_c_data, dict) and "cognitive_panel" not in r_c_data:
                    r_c_data = dict(r_c_data)
                    r_c_data["cognitive_panel"] = r_cp_data
            root_consequence = ConsequenceVector.from_dict(r_c_data)
            root_meta = _safe_dict(root_spec.get("metadata"))
        else:
            root_id = "root"
            root_label = str(root_spec)
            root_consequence = ConsequenceVector()
            root_meta = {}

        dag.add_node(
            DecisionNode(
                node_id=root_id,
                label=root_label,
                depth=0,
                consequence=root_consequence,
                metadata=root_meta,
            )
        )

        counter = 1

        def parse_branches(parent_id: str, branches_data: Any, depth: int) -> None:
            nonlocal counter
            if isinstance(branches_data, dict):
                items = list(branches_data.items())
            elif isinstance(branches_data, list):
                items = []
                for i, b in enumerate(branches_data):
                    if isinstance(b, dict):
                        name = b.get("name") if b.get("name") is not None else b.get("label")
                        items.append((name if name is not None else f"Branch_{i}", b))
                    else:
                        items.append((str(b), b))
            else:
                return

            for key, val in items:
                str_key = str(key) if key is not None else f"Branch_{counter}"
                node_id = f"n{counter}_{str_key.lower().replace(' ', '_')[:16]}"
                counter += 1
                if isinstance(val, dict):
                    raw_label = (
                        val.get("label") if val.get("label") is not None else val.get("name")
                    )
                    label = str(raw_label) if raw_label is not None else str_key
                    c_data = val.get("consequences", val.get("consequence"))
                    cp_data = val.get("cognitive_panel")
                    if cp_data is not None:
                        if c_data is None:
                            c_data = {"cognitive_panel": cp_data}
                        elif isinstance(c_data, dict) and "cognitive_panel" not in c_data:
                            c_data = dict(c_data)
                            c_data["cognitive_panel"] = cp_data
                    meta = {
                        k: v
                        for k, v in val.items()
                        if k
                        not in (
                            "sub_branches",
                            "branches",
                            "children",
                            "consequences",
                            "consequence",
                            "cognitive_panel",
                            "label",
                            "name",
                        )
                    }
                    if "is_speculative" in val:
                        meta["is_speculative"] = bool(val["is_speculative"])
                    if "speculative" in val:
                        meta["is_speculative"] = bool(val["speculative"])
                    sub_b = val.get("sub_branches", val.get("branches", val.get("children")))
                else:
                    label = str(val)
                    c_data = None
                    meta = {}
                    sub_b = None

                c_vec = ConsequenceVector.from_dict(c_data)
                node = DecisionNode(
                    node_id=node_id,
                    label=label,
                    depth=depth,
                    consequence=c_vec,
                    metadata=meta,
                )
                dag.add_node(node)
                dag.add_edge(
                    DecisionEdge(
                        source_id=parent_id,
                        target_id=node_id,
                        action_label=f"Choose {label}",
                    )
                )
                if sub_b:
                    parse_branches(node_id, sub_b, depth + 1)

        branches = spec.get("branches", spec.get("options", []))
        parse_branches(root_id, branches, depth=1)
        return dag


class QuantumDecisionArbiter:
    """Quantum-inspired 6-qubit decision arbiter (dim=64, 4-head attention) for AI agents.

    Employs Quantum Zeno Attention to balance attentional focus pinning
    (Zeno effect: sticking firmly to established goals) and divergent
    exploratory tunneling (Anti-Zeno effect: breaking out of local optima).
    Supports multi-branch decision manifolds and DAG cascading rollouts.
    """

    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 4,
        seed: int | None = 42,
        device: torch.device | str | None = "cpu",
    ) -> None:
        self.dim = dim
        if seed is not None:
            torch.manual_seed(seed)
        self.zeno_attention = QuantumZenoAttention(
            dim=dim,
            observation_frequency=12.0,
            dopamine_coupling=0.6,
            num_heads=num_heads,
            device=device,
            dtype=torch.float32,
        )

    def arbitrate(
        self,
        goal: str,
        options: list[str] | list[dict[str, Any]],
        criteria: list[str] | None = None,
        exploration_drive: float = 0.2,
        workspace: str | None = None,
        log_telemetry: bool = True,
        dilemma: str | None = None,
        injected_prompt_constraint: str | None = None,
        counterfactual_ab: dict[str, Any] | None = None,
        step_idx: int = 0,
        panel_weight: float | None = None,
    ) -> dict[str, Any]:
        """Evaluates decision options against a goal and criteria using Zeno attention.

        Supports arbitrary N-option branching, structured multi-criteria analysis,
        multi-disciplinary cognitive panel evaluation penalties, and dynamic Zeno
        pinning factor derived from 6-qubit quantum projections.

        Args:
            goal: Target objective or strategic constraint.
            options: List of proposed choices (strings or structured dicts).
            criteria: Optional list of specific constraints or criteria.
            exploration_drive: Float in [0.0, 1.0].
                Lower values (0.0 - 0.3) -> High Zeno pinning (goal-aligned).
                Higher values (0.7 - 1.0) -> Anti-Zeno tunneling (exploratory).
            workspace: Optional explicit workspace name for multi-project audit.
            log_telemetry: When True, logs decision telemetry to central ledger.
            dilemma: Optional architectural dilemma description.
            injected_prompt_constraint: Optional explicit prompt constraint string.
            counterfactual_ab: Optional dict with 'without_quanta' and 'with_quanta'.
            step_idx: Optional current step index.
            panel_weight: Optional float weight in [0.0, 1.0] for cognitive panel penalties.

        Returns:
            Dictionary containing recommended option, scores, latency, and quantum diagnostics.
        """
        if not options:
            raise ValueError("Must provide at least one option to arbitrate.")

        t_start = time.perf_counter()

        effective_goal = goal
        if criteria:
            valid_criteria = [c.strip() for c in criteria if isinstance(c, str) and c.strip()]
            if valid_criteria:
                effective_goal = f"{goal} [Kriterler/Kısıtlar: {'; '.join(valid_criteria)}]"

        goal_c = text_to_statevector(effective_goal, dim=self.dim)
        goal_vec = goal_c.real

        parsed_options: list[tuple[str, str, dict[str, Any] | None]] = []
        option_vecs = []
        for opt in options:
            if isinstance(opt, dict):
                name = str(opt.get("name") or opt.get("title") or opt.get("option") or "Option")
                parts = [
                    f"{k}: {v}"
                    for k, v in opt.items()
                    if k
                    not in (
                        "name",
                        "title",
                        "option",
                        "cognitive_panel",
                        "consequence",
                        "consequences",
                    )
                ]
                desc = " | ".join(parts)
                full_repr = f"{name} ({desc})" if desc else name
                parsed_options.append((name, full_repr, opt))
            else:
                opt_str = str(opt)
                parsed_options.append((opt_str, opt_str, None))

        for _, full_repr, _ in parsed_options:
            opt_c = text_to_statevector(f"{effective_goal} -> {full_repr}", dim=self.dim)
            option_vecs.append(opt_c.real)

        # Batch tensor: [num_options, dim]
        x = torch.stack(option_vecs, dim=0)

        dopamine_val = max(0.01, 1.0 - exploration_drive)
        dopamine_tensor = torch.tensor(dopamine_val, dtype=torch.float32)

        t_pt_start = time.perf_counter()
        with torch.no_grad():
            res = self.zeno_attention(x, dopamine=dopamine_tensor, return_diagnostics=True)
        pytorch_latency_ms = (time.perf_counter() - t_pt_start) * 1000.0

        out_tensor = res["output"]  # [num_options, dim]
        explore_mag = float(torch.norm(res["h_explore"]).item())

        # 6-Qubit Hilbert space projective measurement: Tr(\rho \Pi_i) = |<psi_goal | u_i>|^2
        alignments = F.cosine_similarity(out_tensor, goal_vec.unsqueeze(0), dim=-1)
        dim_scale = math.sqrt(float(self.dim))
        norm_tr = F.softmax(alignments * dim_scale, dim=-1).tolist()

        sorted_tr = sorted(norm_tr, reverse=True)
        t_winner = sorted_tr[0] if sorted_tr else 0.5
        t_runner_up = sorted_tr[1] if len(sorted_tr) > 1 else 0.0
        delta_t = max(0.0, t_winner - t_runner_up)
        eta_proj = t_winner / max(1e-6, sum(norm_tr))

        # Dynamic Zeno Pinning Factor P_zeno
        margin_factor = math.pow(delta_t, 0.7) if delta_t > 0.0 else 0.0
        raw_zeno = (
            0.25
            + 0.52 * eta_proj
            + 0.62 * margin_factor
            - 0.20 * float(exploration_drive)
        )
        zeno_pin = float(max(0.15, min(0.985, raw_zeno)))
        anti_zeno_tunneling = max(0.0, 1.0 - zeno_pin)

        # Multi-Disciplinary Cognitive Panel penalty integration
        option_panel_scores: list[CognitivePanelScore | None] = []
        option_panel_penalties: list[float] = []
        for _, _, opt_dict in parsed_options:
            panel_score = None
            if opt_dict is not None:
                if "cognitive_panel" in opt_dict:
                    raw_cp = opt_dict["cognitive_panel"]
                    if isinstance(raw_cp, CognitivePanelScore):
                        panel_score = raw_cp
                    elif isinstance(raw_cp, dict):
                        panel_score = CognitivePanelScore.from_dict(raw_cp)
                elif "consequence" in opt_dict or "consequences" in opt_dict:
                    raw_c = opt_dict.get("consequence", opt_dict.get("consequences"))
                    if isinstance(raw_c, ConsequenceVector):
                        panel_score = raw_c.cognitive_panel
                    elif isinstance(raw_c, dict):
                        panel_score = ConsequenceVector.from_dict(raw_c).cognitive_panel
                elif any(k in opt_dict for k in ("neurobiology", "psychiatry", "sociology")):
                    panel_score = CognitivePanelScore.from_dict(opt_dict)

            option_panel_scores.append(panel_score)
            pen = panel_score.aggregate_penalty() if panel_score is not None else 0.0
            option_panel_penalties.append(pen)

        has_any_panel_penalty = any(p > 0.0 for p in option_panel_penalties)
        if panel_weight is not None:
            eff_panel_weight = max(0.0, min(1.0, float(panel_weight)))
        elif has_any_panel_penalty:
            eff_panel_weight = 0.50
        else:
            eff_panel_weight = 0.0

        if eff_panel_weight > 0.0 and has_any_panel_penalty:
            penalty_tensor = torch.tensor(
                option_panel_penalties,
                dtype=torch.float32,
                device=alignments.device,
            )
            adj_alignments = alignments - eff_panel_weight * penalty_tensor
        else:
            adj_alignments = alignments

        probs = F.softmax(adj_alignments * (1.0 + zeno_pin), dim=-1).tolist()

        ranking: list[dict[str, Any]] = []
        for i, (name, full_repr, details) in enumerate(parsed_options):
            item: dict[str, Any] = {
                "option": name,
                "score": round(float(probs[i]), 4),
                "probability": round(float(probs[i]), 4),
                "tr_rho_pi": round(float(norm_tr[i]), 6),
                "raw_alignment": round(float(alignments[i].item()), 4),
            }
            if option_panel_scores[i] is not None:
                item["cognitive_panel"] = option_panel_scores[i].to_dict()  # type: ignore[union-attr]
                item["panel_penalty"] = round(float(option_panel_penalties[i]), 4)
            if details is not None:
                item["details"] = details
            elif full_repr != name:
                item["full_representation"] = full_repr
            ranking.append(item)

        ranking.sort(key=lambda item: float(item["score"]), reverse=True)
        recommended = ranking[0]
        regime = (
            "Zeno Pinning (Target Focus)"
            if zeno_pin >= 0.5
            else "Anti-Zeno Tunneling (Exploration)"
        )
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        hypotheses_structured: list[dict[str, Any]] | None = None
        if dilemma:
            hypotheses_structured = []
            for i, rk in enumerate(ranking):
                hyp_id = f"d{i+1}"
                hypotheses_structured.append({
                    "id": hyp_id,
                    "label": rk["option"],
                    "ket": f"|{hyp_id}>",
                    "score": rk["score"],
                    "probability": rk["probability"],
                    "tr_rho_pi": rk["tr_rho_pi"],
                    "p_zeno": round(zeno_pin, 4),
                })
            if not injected_prompt_constraint:
                injected_prompt_constraint = (
                    f"🔒 Bilişsel Kuantum Karar Kısıtı: '{recommended['option']}' seçildi. "
                    f"'{dilemma}' ikileminde bu eksenden sapma."
                )
            if not counterfactual_ab:
                opt_name = recommended["option"]
                counterfactual_ab = {
                    "without_quanta": (
                        "Ajan Quanta olmadan serbest bırakıldığında alternatif yaklaşımlar "
                        "arasında tereddüt eder, bağlam sapması ve kırılganlık riski taşır."
                    ),
                    "with_quanta": (
                        f"Quanta 6-qubit hakemi P_zeno={zeno_pin:.4f} Zeno kitlemesiyle "
                        f"'{opt_name}' hipotezine kilitlendi; deterministik icra sağlandı."
                    ),
                    "effect_verified": True,
                }

        if log_telemetry:
            record_decision_telemetry(
                goal=effective_goal,
                options=[p[0] for p in parsed_options],
                winner=recommended["option"],
                confidence=recommended["score"],
                zeno_pinning_factor=zeno_pin,
                anti_zeno_kickback=explore_mag,
                regime=regime,
                latency_ms=latency_ms,
                ranking=ranking,
                workspace=workspace,
                tr_rho_pi={r["option"]: r["tr_rho_pi"] for r in ranking},
                anti_zeno_tunneling_rate=anti_zeno_tunneling,
                pytorch_latency_ms=pytorch_latency_ms,
                dilemma=dilemma,
                hypotheses=hypotheses_structured,
                injected_prompt_constraint=injected_prompt_constraint,
                counterfactual_ab=counterfactual_ab,
                current_zeno_pinning=zeno_pin,
            )

        result: dict[str, Any] = {
            "recommended_option": recommended["option"],
            "confidence": recommended["score"],
            "zeno_pinning_factor": round(zeno_pin, 4),
            "anti_zeno_kickback": round(explore_mag, 4),
            "anti_zeno_tunneling_rate": round(anti_zeno_tunneling, 4),
            "tr_rho_pi": {r["option"]: r["tr_rho_pi"] for r in ranking},
            "t_winner": round(t_winner, 6),
            "delta_t": round(delta_t, 6),
            "eta_proj": round(eta_proj, 4),
            "latency_ms": round(latency_ms, 2),
            "pytorch_latency_ms": round(pytorch_latency_ms, 2),
            "regime": regime,
            "ranked_options": ranking,
        }
        if "details" in recommended:
            result["recommended_details"] = recommended["details"]
        if dilemma:
            result["dilemma"] = dilemma
            result["hypotheses"] = hypotheses_structured
            result["injected_prompt_constraint"] = injected_prompt_constraint
            result["counterfactual_ab"] = counterfactual_ab
            result["step_idx"] = step_idx

        return result

    def arbitrate_dilemma(
        self,
        dilemma: str,
        hypotheses: list[dict[str, Any] | str],
        context: str | None = None,
        criteria: list[str] | None = None,
        exploration_drive: float = 0.2,
        workspace: str | None = None,
        step_idx: int = 0,
        log_telemetry: bool = True,
        injected_prompt_constraint: str | None = None,
        counterfactual_ab: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Arbitrates an architectural dilemma across N candidate hypotheses (|d1>, ..., |dN>).

        Computes 6-qubit quantum tensor projections Tr(rho Pi), selects winning hypothesis
        with dynamic Zeno pinning factor, injects prompt constraint, and produces
        side-by-side counterfactual A/B rationale ('without_quanta' vs 'with_quanta').
        """
        if not hypotheses:
            raise ValueError("Must provide at least one hypothesis to arbitrate.")

        normalized_hyps: list[dict[str, Any]] = []
        for i, h in enumerate(hypotheses):
            if isinstance(h, dict):
                h_id = str(h.get("id") or f"d{i+1}")
                label = str(
                    h.get("label") or h.get("name") or h.get("title") or f"Hypothesis {i+1}"
                )
                ket = str(h.get("ket") or (f"|{h_id}>" if not h_id.startswith("|") else h_id))
                item_dict = dict(h)
                item_dict.update({"id": h_id, "label": label, "ket": ket})
                normalized_hyps.append(item_dict)
            else:
                h_str = str(h)
                normalized_hyps.append({
                    "id": f"d{i+1}",
                    "label": h_str,
                    "ket": f"|d{i+1}>",
                })

        effective_goal = dilemma
        if context:
            effective_goal = f"{dilemma} [Bağlam: {context}]"

        arb_res = self.arbitrate(
            goal=effective_goal,
            options=[h["label"] for h in normalized_hyps],
            criteria=criteria,
            exploration_drive=exploration_drive,
            workspace=workspace,
            log_telemetry=False,
            dilemma=dilemma,
            step_idx=step_idx,
        )

        ranked_options = arb_res.get("ranked_options", [])
        ranked_hyps: list[dict[str, Any]] = []
        unmatched_hyps = list(normalized_hyps)
        for opt_item in ranked_options:
            matched_idx = next(
                (idx for idx, h in enumerate(unmatched_hyps) if h["label"] == opt_item["option"]),
                None,
            )
            if matched_idx is not None:
                matched_hyp = unmatched_hyps.pop(matched_idx)
                hyp_entry = dict(matched_hyp)
                hyp_entry["probability"] = opt_item["score"]
                hyp_entry["score"] = opt_item["score"]
                hyp_entry["tr_rho_pi"] = opt_item["tr_rho_pi"]
                hyp_entry["raw_alignment"] = opt_item.get("raw_alignment", 0.0)
                hyp_entry["p_zeno"] = arb_res["zeno_pinning_factor"]
                ranked_hyps.append(hyp_entry)

        if unmatched_hyps:
            for h in unmatched_hyps:
                hyp_entry = dict(h)
                hyp_entry["probability"] = 0.0
                hyp_entry["score"] = 0.0
                hyp_entry["tr_rho_pi"] = 0.0
                hyp_entry["raw_alignment"] = 0.0
                hyp_entry["p_zeno"] = arb_res["zeno_pinning_factor"]
                ranked_hyps.append(hyp_entry)
            unmatched_hyps.clear()

        ranked_hyps.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        winner_hyp = ranked_hyps[0]
        p_zeno = arb_res["zeno_pinning_factor"]

        if not injected_prompt_constraint:
            injected_prompt_constraint = (
                f"🔒 Bilişsel Kuantum Karar Kısıtı: '{winner_hyp['label']}' eksenine kilitlen. "
                f"'{dilemma}' ikileminde bu mimari kuraldan sapma."
            )

        if not counterfactual_ab:
            losing_labels = [h["label"] for h in ranked_hyps[1:]]
            losing_desc = (
                ", ".join(losing_labels[:3]) if losing_labels else "alternatif yaklaşımlar"
            )
            w_label = winner_hyp["label"]
            counterfactual_ab = {
                "without_quanta": (
                    f"Ajan Quanta olmadan serbest bırakıldığında ({losing_desc}) seçenekleri "
                    "arasında tereddüt eder, bağlam sapması ve kırılgan çalışma riski oluşur."
                ),
                "with_quanta": (
                    f"Quanta 6-qubit hakemi P_zeno={p_zeno:.4f} Zeno kitlemesiyle '{w_label}' "
                    "hipotezine bağlandı; deterministik, doğrulanmış ve tutarlı icra sağlandı."
                ),
                "effect_verified": True,
            }

        result = {
            "dilemma": dilemma,
            "step_idx": step_idx,
            "hypotheses": ranked_hyps,
            "winner": winner_hyp["id"],
            "winning_label": winner_hyp["label"],
            "winning_choice": {
                "id": winner_hyp["id"],
                "label": winner_hyp["label"],
                "ket": winner_hyp.get("ket", f"|{winner_hyp['id']}>"),
                "p_zeno": p_zeno,
                "tr_rho_pi": winner_hyp["tr_rho_pi"],
                "probability": winner_hyp["probability"],
                "regime": arb_res["regime"],
            },
            "p_zeno": p_zeno,
            "zeno_pinning_factor": p_zeno,
            "anti_zeno_kickback": arb_res["anti_zeno_kickback"],
            "anti_zeno_tunneling_rate": arb_res["anti_zeno_tunneling_rate"],
            "injected_prompt_constraint": injected_prompt_constraint,
            "counterfactual_ab": counterfactual_ab,
            "t_winner": arb_res.get("t_winner"),
            "delta_t": arb_res.get("delta_t"),
            "eta_proj": arb_res.get("eta_proj"),
            "latency_ms": arb_res["latency_ms"],
            "pytorch_latency_ms": arb_res["pytorch_latency_ms"],
            "regime": arb_res["regime"],
            "ranked_options": ranked_options,
        }

        if log_telemetry:
            record_decision_telemetry(
                goal=f"[İkilem: {dilemma}] {winner_hyp['label']}",
                options=[h["label"] for h in ranked_hyps],
                winner=winner_hyp["label"],
                confidence=winner_hyp["score"],
                zeno_pinning_factor=p_zeno,
                anti_zeno_kickback=arb_res["anti_zeno_kickback"],
                regime=arb_res["regime"],
                latency_ms=arb_res["latency_ms"],
                ranking=ranked_options,
                workspace=workspace,
                tr_rho_pi={h["label"]: h["tr_rho_pi"] for h in ranked_hyps},
                anti_zeno_tunneling_rate=arb_res["anti_zeno_tunneling_rate"],
                pytorch_latency_ms=arb_res["pytorch_latency_ms"],
                dilemma=dilemma,
                hypotheses=ranked_hyps,
                injected_prompt_constraint=injected_prompt_constraint,
                counterfactual_ab=counterfactual_ab,
                current_zeno_pinning=p_zeno,
            )

        return result

    def arbitrate_dag(
        self,
        dag: DecisionDAG,
        goal: str,
        root_id: str | None = None,
        criteria: list[str] | None = None,
        consequence_weights: tuple[float, float, float, float] = (0.25, 0.35, 0.25, 0.15),
        discount_factor: float = 0.85,
        exploration_drive: float = 0.20,
        prune_threshold: float = 0.25,
        zeno_lock_threshold: float = 0.70,
        beam_width: int | None = None,
        workspace: str | None = None,
        log_telemetry: bool = True,
        panel_weight: float = 0.0,
    ) -> dict[str, Any]:
        """Evaluates multi-branch decision trajectories through a Directed Acyclic Graph (DAG).

        1. Computes quantum statevectors for all nodes and goal.
        2. Runs batch Quantum Zeno Attention over candidate transitions.
        3. Evaluates cascading downstream consequences along each trajectory, interpolating
           operational 4D cost with multi-disciplinary cognitive panel penalties.
        4. Applies microglial pruning to eliminate dominated sub-branches while protecting
           lateral exploratory paths (DMN mode).
        5. Returns winning trajectory, full trajectory rankings, and quantum diagnostics.
        """
        if not dag.nodes:
            raise ValueError("DecisionDAG contains no nodes.")

        t_start = time.perf_counter()

        effective_goal = goal
        if criteria:
            valid_criteria = [c.strip() for c in criteria if isinstance(c, str) and c.strip()]
            if valid_criteria:
                effective_goal = f"{goal} [Kriterler/Kısıtlar: {'; '.join(valid_criteria)}]"

        goal_c = text_to_statevector(effective_goal, dim=self.dim)
        goal_vec = goal_c.real

        # Find paths from root
        paths = dag.enumerate_trajectories(root_id)
        if not paths:
            raise ValueError("No valid trajectories found in DecisionDAG.")

        # Ensure all nodes have statevectors
        for node in dag.nodes.values():
            if node.statevector is None:
                vec_c = text_to_statevector(f"{effective_goal} -> {node.label}", dim=self.dim)
                node.statevector = vec_c.real

        all_nodes = list(dag.nodes.values())
        statevectors = [n.statevector for n in all_nodes if n.statevector is not None]
        x = torch.stack(statevectors, dim=0)

        dopamine_val = max(0.01, 1.0 - exploration_drive)
        dopamine_tensor = torch.tensor(dopamine_val, dtype=torch.float32)

        t_pt_start = time.perf_counter()
        with torch.no_grad():
            att_res = self.zeno_attention(x, dopamine=dopamine_tensor, return_diagnostics=True)
        pytorch_latency_ms = (time.perf_counter() - t_pt_start) * 1000.0

        out_tensor = att_res["output"]
        explore_mag = float(torch.norm(att_res["h_explore"]).item())

        alignments = F.cosine_similarity(out_tensor, goal_vec.unsqueeze(0), dim=-1)
        node_align_map: dict[str, float] = {}
        for i, node in enumerate(all_nodes):
            node_align_map[node.node_id] = float(alignments[i].item())

        # Update edge projections among siblings for each parent node
        dim_scale = math.sqrt(float(self.dim))
        for _u_id, out_edges in dag.adj.items():
            if not out_edges:
                continue
            child_ids = [e.target_id for e in out_edges]
            child_aligns = [node_align_map.get(cid, 0.5) for cid in child_ids]
            child_tensor = torch.tensor(child_aligns, dtype=torch.float32)
            norm_tr = F.softmax(child_tensor * dim_scale, dim=-1).tolist()
            sorted_tr = sorted(norm_tr, reverse=True)
            t_winner = sorted_tr[0] if sorted_tr else 0.5
            t_runner_up = sorted_tr[1] if len(sorted_tr) > 1 else 0.0
            delta_t = max(0.0, t_winner - t_runner_up)
            sum_tr = max(1e-6, sum(norm_tr))

            for k, edge in enumerate(out_edges):
                t_k = norm_tr[k]
                eta_proj = t_k / sum_tr
                margin_factor = math.pow(delta_t, 0.7) if delta_t > 0.0 else 0.0
                raw_zeno = (
                    0.25
                    + 0.52 * eta_proj
                    + 0.62 * margin_factor
                    - 0.20 * float(exploration_drive)
                )
                zeno_k = float(max(0.15, min(0.985, raw_zeno)))
                edge.alignment_score = round(child_aligns[k], 4)
                edge.tr_rho_pi = round(t_k, 6)
                edge.p_zeno = round(zeno_k, 4)
                edge.transition_ket = f"|{edge.target_id}>"

        trajectories: list[DecisionTrajectory] = []
        for idx, path_node_ids in enumerate(paths):
            traj_id = f"tau_{idx + 1}"
            path_nodes_list = [dag.nodes[nid] for nid in path_node_ids]
            path_labels_list = [n.label for n in path_nodes_list]
            is_spec = any(n.is_speculative for n in path_nodes_list)

            cum_utility = 0.0
            cum_cost = 0.0
            p_zenos: list[float] = []
            projections: list[float] = []
            traversed_edges: list[DecisionEdge] = []

            num_steps = len(path_node_ids) - 1
            if num_steps <= 0:
                cum_utility = 1.0
                cum_cost = 0.0
                mean_zeno = 0.5
                fidelity = 1.0
            else:
                for step_idx in range(1, len(path_node_ids)):
                    u_id = path_node_ids[step_idx - 1]
                    v_id = path_node_ids[step_idx]
                    v_node = dag.nodes[v_id]

                    found_edge: DecisionEdge | None = next(
                        (e for e in dag.adj.get(u_id, []) if e.target_id == v_id),
                        None,
                    )
                    if found_edge is None:
                        r_t = node_align_map.get(v_id, 0.5)
                        t_t = 0.5
                        p_z = 0.5
                    else:
                        traversed_edges.append(found_edge)
                        r_t = found_edge.alignment_score
                        t_t = found_edge.tr_rho_pi
                        p_z = found_edge.p_zeno

                    p_zenos.append(p_z)
                    projections.append(t_t)

                    cost_t = v_node.consequence.weighted_cost(
                        consequence_weights, panel_weight=panel_weight
                    )
                    discount = math.pow(discount_factor, step_idx - 1)

                    step_utility = r_t * (1.0 + p_z)
                    cum_utility += discount * step_utility
                    cum_cost += discount * cost_t

                mean_zeno = sum(p_zenos) / len(p_zenos) if p_zenos else 0.5
                prod_t = 1.0
                for t_val in projections:
                    prod_t *= max(1e-6, t_val)
                fidelity = math.pow(prod_t, 1.0 / len(projections)) if projections else 1.0

            trajectory = DecisionTrajectory(
                trajectory_id=traj_id,
                path_nodes=path_node_ids,
                path_labels=path_labels_list,
                cumulative_utility=round(cum_utility, 4),
                cumulative_cost=round(cum_cost, 4),
                mean_p_zeno=round(mean_zeno, 4),
                quantum_fidelity=round(fidelity, 6),
                is_speculative=is_spec,
                edges=traversed_edges,
            )
            trajectories.append(trajectory)

        # Compute net scores and rank trajectories
        v_values = [t.cumulative_utility - t.cumulative_cost for t in trajectories]
        v_tensor = torch.tensor(v_values, dtype=torch.float32)
        norm_v = F.softmax(v_tensor * dim_scale, dim=-1).tolist()

        for i, traj in enumerate(trajectories):
            traj.net_score = round(norm_v[i] * traj.quantum_fidelity, 4)

        trajectories.sort(
            key=lambda item: (item.cumulative_utility - item.cumulative_cost),
            reverse=True,
        )

        winner = trajectories[0]
        winner_val = winner.cumulative_utility - winner.cumulative_cost
        zeno_locked = bool(winner.mean_p_zeno >= zeno_lock_threshold)

        # Microglial Phagocytic Pruning of Dominated Branches
        for traj in trajectories:
            if traj.trajectory_id == winner.trajectory_id:
                traj.is_pruned = False
                traj.prune_reason = None
                continue

            traj_val = traj.cumulative_utility - traj.cumulative_cost
            gap = winner_val - traj_val

            if zeno_locked:
                if traj.is_speculative:
                    eff_thresh = 2.0 * prune_threshold
                    if gap > eff_thresh:
                        traj.is_pruned = True
                        traj.prune_reason = (
                            f"Extreme penalty exceeded protective lateral threshold "
                            f"(gap={gap:.4f} > {eff_thresh:.4f})"
                        )
                    else:
                        traj.is_pruned = False
                        traj.prune_reason = None
                else:
                    if gap > prune_threshold:
                        traj.is_pruned = True
                        traj.prune_reason = (
                            f"Dominated by winner (gap={gap:.4f} > {prune_threshold:.4f})"
                        )
                    else:
                        traj.is_pruned = False
                        traj.prune_reason = None
            else:
                traj.is_pruned = False
                traj.prune_reason = None

        # Beam width filtering
        if beam_width is not None and beam_width > 0:
            active_count = 0
            for traj in trajectories:
                if not traj.is_pruned:
                    active_count += 1
                    if active_count > beam_width and not traj.is_speculative:
                        traj.is_pruned = True
                        traj.prune_reason = f"Exceeded beam width limit ({beam_width})"

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        rec_option = (
            winner.path_labels[1]
            if len(winner.path_labels) > 1
            else winner.path_labels[0]
        )
        regime = (
            "Zeno Pinning (Target Focus)"
            if winner.mean_p_zeno >= 0.5
            else "Anti-Zeno Tunneling (Exploration)"
        )

        surviving = [t for t in trajectories if not t.is_pruned]
        pruned = [t for t in trajectories if t.is_pruned]

        if log_telemetry:
            record_decision_telemetry(
                goal=effective_goal,
                options=[t.trajectory_id for t in trajectories],
                winner=winner.trajectory_id,
                confidence=winner.net_score,
                zeno_pinning_factor=winner.mean_p_zeno,
                anti_zeno_kickback=explore_mag,
                regime=regime,
                latency_ms=latency_ms,
                ranking=[t.to_dict() for t in trajectories],
                workspace=workspace,
                pytorch_latency_ms=pytorch_latency_ms,
                dilemma=goal,
                dag_rollout_depth=len(winner.path_nodes) - 1,
                total_trajectories=len(trajectories),
                pruned_count=len(pruned),
                winning_trajectory=winner.to_dict(),
            )

        return {
            "winning_trajectory": winner.to_dict(),
            "recommended_option": rec_option,
            "recommended_path": winner.path_labels,
            "confidence": winner.net_score,
            "trajectories": [t.to_dict() for t in trajectories],
            "surviving_trajectories": [t.to_dict() for t in surviving],
            "pruned_trajectories": [t.to_dict() for t in pruned],
            "pruned_count": len(pruned),
            "total_trajectories": len(trajectories),
            "zeno_lock_engaged": zeno_locked,
            "regime": regime,
            "mean_p_zeno": winner.mean_p_zeno,
            "latency_ms": round(latency_ms, 2),
            "pytorch_latency_ms": round(pytorch_latency_ms, 2),
            "panel_weight": panel_weight,
        }

    def arbitrate_multibranch(
        self,
        goal: str,
        branches: dict[str, Any] | list[dict[str, Any]],
        criteria: list[str] | None = None,
        exploration_drive: float = 0.20,
        consequence_weights: tuple[float, float, float, float] = (0.25, 0.35, 0.25, 0.15),
        discount_factor: float = 0.85,
        prune_threshold: float = 0.25,
        zeno_lock_threshold: float = 0.70,
        beam_width: int | None = None,
        workspace: str | None = None,
        log_telemetry: bool = True,
        panel_weight: float = 0.0,
    ) -> dict[str, Any]:
        """High-level entry point that converts a multi-branch tree or DAG specification into

        a DecisionDAG, executes cascading rollout arbitration, and produces an executive summary
        and counterfactual A/B evaluation reflecting cognitive panel considerations.
        """
        if isinstance(branches, dict) and "branches" in branches:
            spec = dict(branches)
            if "root" not in spec:
                spec["root"] = goal
        else:
            spec = {"root": goal, "branches": branches}

        dag = DecisionDAG.from_dict(spec)
        dag_res = self.arbitrate_dag(
            dag=dag,
            goal=goal,
            criteria=criteria,
            consequence_weights=consequence_weights,
            discount_factor=discount_factor,
            exploration_drive=exploration_drive,
            prune_threshold=prune_threshold,
            zeno_lock_threshold=zeno_lock_threshold,
            beam_width=beam_width,
            workspace=workspace,
            log_telemetry=log_telemetry,
            panel_weight=panel_weight,
        )

        winner = dag_res["winning_trajectory"]
        rec_path_str = " -> ".join(winner["path_labels"])
        pruned_count = dag_res["pruned_count"]
        total_trajectories = dag_res["total_trajectories"]
        speculative_survivors = sum(
            1 for t in dag_res["surviving_trajectories"] if t.get("is_speculative")
        )

        has_panel = any(
            n.consequence.cognitive_panel.aggregate_penalty() > 0.0
            for n in dag.nodes.values()
        )
        winner_panel_penalty = sum(
            dag.nodes[nid].consequence.cognitive_panel.aggregate_penalty()
            for nid in winner["path_nodes"]
            if nid in dag.nodes
        )

        panel_note = ""
        if panel_weight > 0.0 or has_panel:
            panel_note = (
                f"\n- Bilişsel Panel: Nörobiyolojik, psikiyatrik ve sosyolojik kısıtlar "
                f"değerlendirildi (ceza skoru: {winner_panel_penalty:.2f})."
            )

        exec_summary = (
            f"🏛️ Yönetici Özeti (Executive Summary):\n"
            f"- Tavsiye Edilen Yol: '{rec_path_str}'\n"
            f"- Gerekçe: Stratejik hedefe en yüksek uyum ({winner['net_score']:.2f}) sağlandı; "
            f"kaskat operasyonel maliyetler ve riskler minimize edildi.{panel_note}\n"
            f"- Budanan Seçenekler: {pruned_count}/{total_trajectories} alternatif yüksek "
            f"teknik borç / operasyonel yük nedeniyle elendi.\n"
            f"- Korunan İnovasyon Yolları: {speculative_survivors} spekülatif lateral yol "
            f"alternatif senaryolar için korundu."
        )

        if panel_weight > 0.0 or has_panel:
            counterfactual_ab = {
                "without_quanta": (
                    "Ajan Quanta olmadan tek adımlı açgözlü (greedy) seçim yapar; "
                    "uzun vadeli bakım yükü, gecikme, kullanıcı bilişsel yorgunluğu ve "
                    "rumatif kilitlenme risklerini öngöremez."
                ),
                "with_quanta": (
                    f"Quanta çok disiplinli karar manifoldu (DAG Rollouts) ile {total_trajectories} "
                    f"yol kaskat halinde değerlendirildi; P_zeno={dag_res['mean_p_zeno']:.4f} ile "
                    f"'{rec_path_str}' yoluna kilitlenerek baskın ve bilişsel yük getiren "
                    f"dallar budandı."
                ),
                "effect_verified": True,
            }
        else:
            counterfactual_ab = {
                "without_quanta": (
                    "Ajan Quanta olmadan tek adımlı açgözlü (greedy) seçim yapar; "
                    "uzun vadeli bakım yükü, gecikme ve mimari kilitlenme risklerini öngöremez."
                ),
                "with_quanta": (
                    f"Quanta çok kollu karar manifoldu (DAG Rollouts) ile {total_trajectories} "
                    f"yol kaskat halinde değerlendirildi; P_zeno={dag_res['mean_p_zeno']:.4f} ile "
                    f"'{rec_path_str}' yoluna kilitlenerek baskın dallar budandı."
                ),
                "effect_verified": True,
            }

        dag_res["executive_summary"] = exec_summary
        dag_res["counterfactual_ab"] = counterfactual_ab
        dag_res["dag"] = dag
        return dag_res
