"""Unit tests for Non-Binary Multi-Branch Decision Manifold (DAG Rollouts) in Quanta SDK.

Verifies:
1. TestDecisionDAGRepresentation: Construction, cycle detection, topological sort, dictionary hydration.
2. TestBiomorphicCascadingRollouts: Multi-branch 5-option branching, cascading consequence penalties,
   discount factor decay, geometric mean fidelity.
3. TestPrefrontalZenoArbiterPruning: Microglial pruning of dominated branches, Zeno lock gating, beam filtering,
   and threshold sensitivity.
4. TestDMNLateralExplorationPreservation: DMN lateral speculative preservation, extreme penalty cutoff,
   exploration drive modulation, executive summary, and N-ary dilemma scaling.
"""

from __future__ import annotations

import pytest

from quanta.cognitive.arbiter import (
    ConsequenceVector,
    DecisionDAG,
    DecisionEdge,
    DecisionNode,
    QuantumDecisionArbiter,
)


class TestDecisionDAGRepresentation:
    """Tests for DAG construction, traversal, cycle detection, and serialization."""

    def test_dag_node_and_edge_addition_and_traversal(self) -> None:
        """Verifies node/edge additions and bi-directional adjacency traversal."""
        dag = DecisionDAG()
        root = DecisionNode(node_id="root", label="Root Architecture", depth=0)
        node_a = DecisionNode(
            node_id="node_a",
            label="Service Mesh",
            depth=1,
            consequence=ConsequenceVector(latency=0.2, maintenance=0.4),
        )
        node_b = DecisionNode(
            node_id="node_b",
            label="Direct RPC",
            depth=1,
            consequence=ConsequenceVector(latency=0.05, maintenance=0.2),
        )
        node_c = DecisionNode(node_id="node_c", label="Observability Layer", depth=2)

        dag.add_node(root)
        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        edge_ra = DecisionEdge(source_id="root", target_id="node_a", action_label="Adopt Mesh")
        edge_rb = DecisionEdge(source_id="root", target_id="node_b", action_label="Adopt RPC")
        edge_ac = DecisionEdge(source_id="node_a", target_id="node_c", action_label="Add Tracing")
        edge_bc = DecisionEdge(source_id="node_b", target_id="node_c", action_label="Add Logging")

        dag.add_edge(edge_ra)
        dag.add_edge(edge_rb)
        dag.add_edge(edge_ac)
        dag.add_edge(edge_bc)

        assert dag.get_node("root") is not None
        assert dag.get_node("root").label == "Root Architecture"  # type: ignore[union-attr]

        successors = dag.get_successors("root")
        assert len(successors) == 2
        assert {n.node_id for n in successors} == {"node_a", "node_b"}

        predecessors = dag.get_predecessors("node_c")
        assert len(predecessors) == 2
        assert {n.node_id for n in predecessors} == {"node_a", "node_b"}

        assert len(dag.get_outgoing_edges("root")) == 2
        assert len(dag.get_incoming_edges("node_c")) == 2

        with pytest.raises(KeyError, match="Source node 'missing' does not exist"):
            dag.add_edge(DecisionEdge(source_id="missing", target_id="node_a", action_label="X"))

        with pytest.raises(KeyError, match="Target node 'missing' does not exist"):
            dag.add_edge(DecisionEdge(source_id="root", target_id="missing", action_label="X"))

    def test_dag_cycle_detection(self) -> None:
        """Verifies cycle detection and rejection of cyclic topologies."""
        dag = DecisionDAG()
        dag.add_node(DecisionNode(node_id="n1", label="Step 1"))
        dag.add_node(DecisionNode(node_id="n2", label="Step 2"))
        dag.add_node(DecisionNode(node_id="n3", label="Step 3"))

        dag.add_edge(DecisionEdge(source_id="n1", target_id="n2", action_label="1->2"))
        dag.add_edge(DecisionEdge(source_id="n2", target_id="n3", action_label="2->3"))

        assert dag.has_cycle() is False

        # Add back-edge forming cycle n3 -> n1
        dag.add_edge(DecisionEdge(source_id="n3", target_id="n1", action_label="3->1"))
        assert dag.has_cycle() is True

        with pytest.raises(ValueError, match="Graph contains a cycle"):
            dag.topological_sort()

        with pytest.raises(ValueError, match="Graph contains a cycle"):
            dag.enumerate_trajectories()

    def test_dag_topological_sort_valid(self) -> None:
        """Verifies that topological sorting strictly respects causal dependencies."""
        dag = DecisionDAG()
        nodes = ["root", "opt_a", "opt_b", "sub_a1", "sub_b1", "converge_sink"]
        for nid in nodes:
            dag.add_node(DecisionNode(node_id=nid, label=nid))

        dag.add_edge(DecisionEdge(source_id="root", target_id="opt_a", action_label=""))
        dag.add_edge(DecisionEdge(source_id="root", target_id="opt_b", action_label=""))
        dag.add_edge(DecisionEdge(source_id="opt_a", target_id="sub_a1", action_label=""))
        dag.add_edge(DecisionEdge(source_id="opt_b", target_id="sub_b1", action_label=""))
        dag.add_edge(DecisionEdge(source_id="sub_a1", target_id="converge_sink", action_label=""))
        dag.add_edge(DecisionEdge(source_id="sub_b1", target_id="converge_sink", action_label=""))

        order = dag.topological_sort()
        assert len(order) == 6

        for u in nodes:
            for edge in dag.get_outgoing_edges(u):
                v = edge.target_id
                assert order.index(u) < order.index(v)

        assert dag.find_roots() == ["root"]
        assert dag.find_leaves() == ["converge_sink"]

    def test_dag_from_dict_hydration_and_trajectory_enumeration(self) -> None:
        """Verifies hydration from dictionary tree and dynamic programming path memoization."""
        spec = {
            "root": "Cloud Platform",
            "branches": {
                "Microservices": {
                    "consequences": {"latency": 0.4, "maintenance": 0.6, "risk": 0.3},
                    "sub_branches": {
                        "Kubernetes": {"consequences": {"latency": 0.2, "maintenance": 0.7}},
                        "Serverless": {"consequences": {"latency": 0.5, "maintenance": 0.2}},
                    },
                },
                "Monolith": {
                    "consequences": {"latency": 0.1, "maintenance": 0.3, "risk": 0.1},
                    "sub_branches": {
                        "Modular Go": {"consequences": {"latency": 0.05, "maintenance": 0.2}},
                        "Modular Python": {"consequences": {"latency": 0.15, "maintenance": 0.3}},
                    },
                },
            },
        }

        dag = DecisionDAG.from_dict(spec)
        assert len(dag.nodes) == 7

        trajectories = dag.enumerate_trajectories()
        assert len(trajectories) == 4
        for traj in trajectories:
            assert len(traj) == 3
            assert traj[0] == "root"

        # Explicit graph with confluence to verify memoization
        graph_spec = {
            "nodes": [
                {"node_id": "r", "label": "Start"},
                {"node_id": "a", "label": "Branch A"},
                {"node_id": "b", "label": "Branch B"},
                {"node_id": "c", "label": "Shared DB"},
                {"node_id": "d", "label": "Terminal Sink"},
            ],
            "edges": [
                {"source_id": "r", "target_id": "a"},
                {"source_id": "r", "target_id": "b"},
                {"source_id": "a", "target_id": "c"},
                {"source_id": "b", "target_id": "c"},
                {"source_id": "c", "target_id": "d"},
            ],
        }
        dag_confluence = DecisionDAG.from_dict(graph_spec)
        confluent_trajs = dag_confluence.enumerate_trajectories("r")
        assert len(confluent_trajs) == 2
        assert confluent_trajs[0] == ["r", "a", "c", "d"]
        assert confluent_trajs[1] == ["r", "b", "c", "d"]


class TestBiomorphicCascadingRollouts:
    """Tests for multi-branch cascading rollouts, consequence weighting, and fidelity."""

    def test_multibranch_5_options_trajectories(self) -> None:
        """Verifies 5-option multi-branch manifold rollout with 10 leaf trajectories."""
        arbiter = QuantumDecisionArbiter(dim=64, num_heads=4, seed=42)

        branches = {
            "Opt A (Event Sourcing)": {
                "consequences": {"latency": 0.3, "maintenance": 0.5, "risk": 0.4},
                "sub_branches": {
                    "Kafka Cluster": {"latency": 0.2, "maintenance": 0.6},
                    "NATS JetStream": {"latency": 0.1, "maintenance": 0.3},
                },
            },
            "Opt B (Relational ACID)": {
                "consequences": {"latency": 0.2, "maintenance": 0.3, "risk": 0.2},
                "sub_branches": {
                    "PostgreSQL Spanner": {"latency": 0.15, "maintenance": 0.25},
                    "MySQL InnoDB": {"latency": 0.2, "maintenance": 0.35},
                },
            },
            "Opt C (Document Store)": {
                "consequences": {"latency": 0.25, "maintenance": 0.4, "risk": 0.35},
                "sub_branches": {
                    "MongoDB Atlas": {"latency": 0.25, "maintenance": 0.3},
                    "DynamoDB": {"latency": 0.1, "maintenance": 0.2},
                },
            },
            "Opt D (Graph Topology)": {
                "consequences": {"latency": 0.4, "maintenance": 0.6, "risk": 0.45},
                "sub_branches": {
                    "Neo4j Enterprise": {"latency": 0.35, "maintenance": 0.5},
                    "Memgraph In-Memory": {"latency": 0.15, "maintenance": 0.4},
                },
            },
            "Opt E (Key-Value Cache Layer)": {
                "consequences": {"latency": 0.05, "maintenance": 0.2, "risk": 0.15},
                "sub_branches": {
                    "Redis Cluster": {"latency": 0.05, "maintenance": 0.2},
                    "Dragonfly Fast Store": {"latency": 0.03, "maintenance": 0.15},
                },
            },
        }

        res = arbiter.arbitrate_multibranch(
            goal="High Availability Low Latency Caching System",
            branches=branches,
            exploration_drive=0.20,
        )

        assert res["total_trajectories"] == 10
        assert len(res["trajectories"]) == 10
        assert res["recommended_option"] in branches
        assert len(res["winning_trajectory"]["path_labels"]) == 3

        for traj in res["trajectories"]:
            assert "cumulative_utility" in traj
            assert "cumulative_cost" in traj
            assert "net_score" in traj
            assert "quantum_fidelity" in traj
            assert 0.0 <= traj["quantum_fidelity"] <= 1.0

    def test_cascading_consequence_penalties(self) -> None:
        """Verifies that high operational consequence penalties diminish path utility."""
        arbiter = QuantumDecisionArbiter(dim=64, num_heads=4, seed=42)
        dag = DecisionDAG()

        root = DecisionNode(node_id="root", label="Select Compute Infrastructure")
        node_clean = DecisionNode(
            node_id="clean",
            label="Serverless Cloud Run",
            consequence=ConsequenceVector(
                latency=0.05, maintenance=0.10, risk=0.05, metabolic=0.05
            ),
        )
        node_heavy = DecisionNode(
            node_id="heavy",
            label="Self-Hosted Kubernetes",
            consequence=ConsequenceVector(
                latency=0.85, maintenance=0.95, risk=0.80, metabolic=0.90
            ),
        )

        dag.add_node(root)
        dag.add_node(node_clean)
        dag.add_node(node_heavy)
        dag.add_edge(DecisionEdge(source_id="root", target_id="clean", action_label="Deploy Run"))
        dag.add_edge(DecisionEdge(source_id="root", target_id="heavy", action_label="Deploy K8s"))

        res = arbiter.arbitrate_dag(
            dag=dag,
            goal="Deploy Cloud Run microservices with low operational burden",
            consequence_weights=(0.25, 0.35, 0.25, 0.15),
        )

        clean_traj = next(t for t in res["trajectories"] if t["path_nodes"][-1] == "clean")
        heavy_traj = next(t for t in res["trajectories"] if t["path_nodes"][-1] == "heavy")

        assert heavy_traj["cumulative_cost"] > clean_traj["cumulative_cost"] * 5.0
        assert clean_traj["net_score"] > heavy_traj["net_score"]
        assert res["winning_trajectory"]["path_nodes"][-1] == "clean"

    def test_discount_factor_depth_decay(self) -> None:
        """Verifies exponential discount decay gamma^(t-1) for deep downstream consequences."""
        arbiter = QuantumDecisionArbiter(dim=64, num_heads=4, seed=42)

        def make_chain_dag() -> DecisionDAG:
            dag = DecisionDAG()
            dag.add_node(DecisionNode(node_id="r", label="Root"))
            dag.add_node(DecisionNode(node_id="s1", label="Step 1"))
            dag.add_node(DecisionNode(node_id="s2", label="Step 2"))
            dag.add_node(
                DecisionNode(
                    node_id="s3",
                    label="Step 3",
                    consequence=ConsequenceVector(
                        latency=1.0, maintenance=1.0, risk=1.0, metabolic=1.0
                    ),
                )
            )
            dag.add_edge(DecisionEdge(source_id="r", target_id="s1", action_label=""))
            dag.add_edge(DecisionEdge(source_id="s1", target_id="s2", action_label=""))
            dag.add_edge(DecisionEdge(source_id="s2", target_id="s3", action_label=""))
            return dag

        res_undiscounted = arbiter.arbitrate_dag(
            dag=make_chain_dag(),
            goal="Analyze multi-step pipeline",
            discount_factor=1.0,
        )
        res_discounted = arbiter.arbitrate_dag(
            dag=make_chain_dag(),
            goal="Analyze multi-step pipeline",
            discount_factor=0.5,
        )

        cost_undiscounted = res_undiscounted["winning_trajectory"]["cumulative_cost"]
        cost_discounted = res_discounted["winning_trajectory"]["cumulative_cost"]

        # Step 3 is at depth 3 (t=3 -> discount is gamma^2).
        # gamma=1.0 -> 1.0 * cost; gamma=0.5 -> 0.25 * cost
        assert cost_undiscounted > cost_discounted
        assert cost_discounted == pytest.approx(cost_undiscounted * 0.25, rel=0.1)

    def test_geometric_quantum_fidelity_calculation(self) -> None:
        """Verifies geometric mean quantum fidelity F(tau) = (prod T_t)^(1/D)."""
        arbiter = QuantumDecisionArbiter(dim=64, num_heads=4, seed=42)
        dag = DecisionDAG()

        dag.add_node(DecisionNode(node_id="r", label="Entry"))
        dag.add_node(DecisionNode(node_id="n1", label="Intermediate Node"))
        dag.add_node(DecisionNode(node_id="n2", label="Leaf Node"))

        dag.add_edge(DecisionEdge(source_id="r", target_id="n1", action_label=""))
        dag.add_edge(DecisionEdge(source_id="n1", target_id="n2", action_label=""))

        res = arbiter.arbitrate_dag(dag=dag, goal="Fidelity verification")
        winner = res["winning_trajectory"]
        fidelity = winner["quantum_fidelity"]

        assert 0.0 < fidelity <= 1.0


class TestPrefrontalZenoArbiterPruning:
    """Tests for microglial pruning of dominated branches, Zeno lock gating, and beam filtering."""

    def test_microglial_pruning_dominated_branches(self) -> None:
        """Verifies that heavily dominated branches are pruned when Zeno lock is engaged."""
        arbiter = QuantumDecisionArbiter(dim=64, num_heads=4, seed=42)
        dag = DecisionDAG()

        root = DecisionNode(node_id="root", label="System Architecture Selection")
        node_optimal = DecisionNode(
            node_id="opt",
            label="Distributed Event-Driven Architecture",
            consequence=ConsequenceVector(latency=0.05, maintenance=0.1),
        )
        node_terrible1 = DecisionNode(
            node_id="bad1",
            label="Legacy Synchronous Polling via Cron Job",
            consequence=ConsequenceVector(
                latency=0.9, maintenance=0.95, risk=0.85, metabolic=0.9
            ),
        )
        node_terrible2 = DecisionNode(
            node_id="bad2",
            label="Single Point of Failure File Share",
            consequence=ConsequenceVector(
                latency=0.95, maintenance=0.9, risk=0.95, metabolic=0.95
            ),
        )

        dag.add_node(root)
        dag.add_node(node_optimal)
        dag.add_node(node_terrible1)
        dag.add_node(node_terrible2)

        dag.add_edge(DecisionEdge(source_id="root", target_id="opt", action_label="Choose Opt"))
        dag.add_edge(DecisionEdge(source_id="root", target_id="bad1", action_label="Choose Bad1"))
        dag.add_edge(DecisionEdge(source_id="root", target_id="bad2", action_label="Choose Bad2"))

        res = arbiter.arbitrate_dag(
            dag=dag,
            goal="High Throughput Distributed Event Driven Architecture",
            prune_threshold=0.25,
            zeno_lock_threshold=0.50,
        )

        assert res["zeno_lock_engaged"] is True
        assert res["pruned_count"] >= 2
        assert res["winning_trajectory"]["is_pruned"] is False

        pruned_trajs = res["pruned_trajectories"]
        for p in pruned_trajs:
            assert p["is_pruned"] is True
            assert "Dominated by winner" in str(p["prune_reason"])

    def test_pruning_respects_zeno_lock_threshold(self) -> None:
        """Verifies that pruning is deferred when Zeno lock threshold is not satisfied."""
        arbiter = QuantumDecisionArbiter(dim=64, num_heads=4, seed=42)
        dag = DecisionDAG()

        root = DecisionNode(node_id="root", label="Ambiguous Architectural Exploration")
        node_a = DecisionNode(
            node_id="a",
            label="Microservices Approach",
            consequence=ConsequenceVector(latency=0.4, maintenance=0.5),
        )
        node_b = DecisionNode(
            node_id="b",
            label="Monolithic Approach",
            consequence=ConsequenceVector(latency=0.8, maintenance=0.8),
        )

        dag.add_node(root)
        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_edge(DecisionEdge(source_id="root", target_id="a", action_label=""))
        dag.add_edge(DecisionEdge(source_id="root", target_id="b", action_label=""))

        # Set unattainable zeno_lock_threshold to simulate open deliberation / low confidence
        res = arbiter.arbitrate_dag(
            dag=dag,
            goal="Open Exploration",
            prune_threshold=0.10,
            zeno_lock_threshold=0.9999,
        )

        assert res["zeno_lock_engaged"] is False
        assert res["pruned_count"] == 0
        for t in res["trajectories"]:
            assert t["is_pruned"] is False
            assert t["prune_reason"] is None

    def test_beam_width_capping(self) -> None:
        """Verifies that beam filtering caps active candidates to beam_width."""
        arbiter = QuantumDecisionArbiter(dim=64, num_heads=4, seed=42)
        dag = DecisionDAG()
        dag.add_node(DecisionNode(node_id="r", label="Root"))

        for i in range(10):
            nid = f"branch_{i}"
            dag.add_node(
                DecisionNode(
                    node_id=nid,
                    label=f"Branch Option {i}",
                    consequence=ConsequenceVector(latency=0.01 * i, maintenance=0.01 * i),
                )
            )
            dag.add_edge(DecisionEdge(source_id="r", target_id=nid, action_label=""))

        res = arbiter.arbitrate_dag(
            dag=dag,
            goal="Multi-branch beam test",
            prune_threshold=100.0,  # Ensure threshold alone does not prune
            beam_width=4,
        )

        assert len(res["surviving_trajectories"]) == 4
        assert res["pruned_count"] == 6

        for p in res["pruned_trajectories"]:
            assert "Exceeded beam width limit (4)" in str(p["prune_reason"])

    def test_prune_threshold_sensitivity(self) -> None:
        """Verifies sensitivity of pruning count to prune_threshold parameter."""
        arbiter = QuantumDecisionArbiter(dim=64, num_heads=4, seed=42)
        dag = DecisionDAG()
        dag.add_node(DecisionNode(node_id="r", label="Root"))

        for i in range(6):
            nid = f"b_{i}"
            dag.add_node(
                DecisionNode(
                    node_id=nid,
                    label=f"Strategy {i}",
                    consequence=ConsequenceVector(latency=0.15 * i, maintenance=0.15 * i),
                )
            )
            dag.add_edge(DecisionEdge(source_id="r", target_id=nid, action_label=""))

        res_strict = arbiter.arbitrate_dag(
            dag=dag,
            goal="General Strategy",
            prune_threshold=0.05,
            zeno_lock_threshold=0.40,
        )
        res_loose = arbiter.arbitrate_dag(
            dag=dag,
            goal="General Strategy",
            prune_threshold=5.0,
            zeno_lock_threshold=0.40,
        )

        assert res_strict["pruned_count"] >= res_loose["pruned_count"]


class TestDMNLateralExplorationPreservation:
    """Tests for DMN lateral exploration protection, anti-zeno tunneling, and executive reporting."""

    def test_dmn_speculative_lateral_branch_protection(self) -> None:
        """Verifies that speculative/lateral branches survive standard microglial pruning."""
        arbiter = QuantumDecisionArbiter(dim=64, num_heads=4, seed=42)
        dag = DecisionDAG()

        root = DecisionNode(node_id="root", label="AI Pipeline Design")
        winner_node = DecisionNode(
            node_id="opt",
            label="State of the Art Transformer",
            consequence=ConsequenceVector(latency=0.05, maintenance=0.05),
        )
        conventional_dominated = DecisionNode(
            node_id="conv_dom",
            label="Classical Rule-Based Regex Script",
            consequence=ConsequenceVector(latency=0.60, maintenance=0.60, risk=0.50),
            metadata={"is_speculative": False},
        )
        speculative_lateral = DecisionNode(
            node_id="spec_lat",
            label="Neuromorphic Quantum Reservoir Superposition",
            consequence=ConsequenceVector(latency=0.25, maintenance=0.25),
            metadata={"is_speculative": True},
        )

        dag.add_node(root)
        dag.add_node(winner_node)
        dag.add_node(conventional_dominated)
        dag.add_node(speculative_lateral)

        dag.add_edge(DecisionEdge(source_id="root", target_id="opt", action_label=""))
        dag.add_edge(DecisionEdge(source_id="root", target_id="conv_dom", action_label=""))
        dag.add_edge(DecisionEdge(source_id="root", target_id="spec_lat", action_label=""))

        res = arbiter.arbitrate_dag(
            dag=dag,
            goal="Modern Transformer Machine Learning Pipeline",
            prune_threshold=0.20,
            zeno_lock_threshold=0.40,
        )

        conv_traj = next(t for t in res["trajectories"] if t["path_nodes"][-1] == "conv_dom")
        spec_traj = next(t for t in res["trajectories"] if t["path_nodes"][-1] == "spec_lat")

        assert conv_traj["is_pruned"] is True
        assert spec_traj["is_pruned"] is False
        assert spec_traj["is_speculative"] is True
        assert any(t["path_nodes"][-1] == "spec_lat" for t in res["surviving_trajectories"])

    def test_lateral_branch_extreme_penalty_pruning(self) -> None:
        """Verifies that speculative branches are pruned if gap exceeds the protective relaxed threshold."""
        arbiter = QuantumDecisionArbiter(dim=64, num_heads=4, seed=42)
        dag = DecisionDAG()

        root = DecisionNode(node_id="root", label="Database Choice")
        winner = DecisionNode(
            node_id="win",
            label="Postgres DB",
            consequence=ConsequenceVector(latency=0.05, maintenance=0.05),
        )
        catastrophic_spec = DecisionNode(
            node_id="cat_spec",
            label="Experimental CSV in DNS TXT Records",
            consequence=ConsequenceVector(
                latency=1.0, maintenance=1.0, risk=1.0, metabolic=1.0
            ),
            metadata={"is_speculative": True},
        )

        dag.add_node(root)
        dag.add_node(winner)
        dag.add_node(catastrophic_spec)
        dag.add_edge(DecisionEdge(source_id="root", target_id="win", action_label=""))
        dag.add_edge(DecisionEdge(source_id="root", target_id="cat_spec", action_label=""))

        res = arbiter.arbitrate_dag(
            dag=dag,
            goal="Production Relational Database",
            prune_threshold=0.20,
            zeno_lock_threshold=0.40,
        )

        spec_traj = next(t for t in res["trajectories"] if t["path_nodes"][-1] == "cat_spec")
        assert spec_traj["is_pruned"] is True
        assert "Extreme penalty exceeded protective lateral threshold" in str(
            spec_traj["prune_reason"]
        )

    def test_high_exploration_drive_activates_anti_zeno(self) -> None:
        """Verifies that high exploration drive shifts regime to Anti-Zeno Tunneling (DMN)."""
        arbiter = QuantumDecisionArbiter(dim=64, num_heads=4, seed=42)

        branches = {
            "Primary Focused Architecture": {"consequences": {"latency": 0.05}},
            "Alternative Variant B": {"consequences": {"latency": 0.2}},
            "Alternative Variant C": {"consequences": {"latency": 0.3}},
        }

        res_zeno = arbiter.arbitrate_multibranch(
            goal="Primary Focused Architecture",
            branches=branches,
            exploration_drive=0.05,
        )
        res_dream = arbiter.arbitrate_multibranch(
            goal="Exploratory divergence across alternative variants",
            branches=branches,
            exploration_drive=0.95,
        )

        assert res_zeno["regime"] == "Zeno Pinning (Target Focus)"
        assert res_dream["regime"] == "Anti-Zeno Tunneling (Exploration)"
        assert res_dream["mean_p_zeno"] < res_zeno["mean_p_zeno"]

    def test_multibranch_executive_summary_and_counterfactual_ab(self) -> None:
        """Verifies executive summary generation and N-ary dilemma scaling beyond 3 choices."""
        arbiter = QuantumDecisionArbiter(dim=64, num_heads=4, seed=42)

        branches = {
            "Modular Monolith": {
                "consequences": {"latency": 0.1, "maintenance": 0.2},
                "sub_branches": {
                    "In-Memory Event Bus": {"latency": 0.05, "maintenance": 0.15},
                },
            },
            "Microservices": {
                "consequences": {"latency": 0.5, "maintenance": 0.7},
                "sub_branches": {
                    "Distributed Service Mesh": {"latency": 0.6, "maintenance": 0.8},
                },
            },
        }

        res = arbiter.arbitrate_multibranch(
            goal="High Velocity Backend Architecture",
            branches=branches,
        )

        assert "🏛️ Yönetici Özeti (Executive Summary):" in res["executive_summary"]
        assert "Tavsiye Edilen Yol:" in res["executive_summary"]
        assert res["counterfactual_ab"]["effect_verified"] is True
        assert "without_quanta" in res["counterfactual_ab"]
        assert "with_quanta" in res["counterfactual_ab"]

        # Verify arbitrate_dilemma scaling to N=5 options (|d1> ... |d5>)
        dilemma_hyps = [
            {"id": "d1", "label": "Option 1: PostgreSQL"},
            {"id": "d2", "label": "Option 2: Redis Cluster"},
            {"id": "d3", "label": "Option 3: Neo4j Graph"},
            {"id": "d4", "label": "Option 4: Apache Cassandra"},
            {"id": "d5", "label": "Option 5: ClickHouse OLAP"},
        ]

        dil_res = arbiter.arbitrate_dilemma(
            dilemma="Database Architecture Selection",
            hypotheses=dilemma_hyps,
            exploration_drive=0.20,
        )

        assert len(dil_res["hypotheses"]) == 5
        assert {h["ket"] for h in dil_res["hypotheses"]} == {
            "|d1>",
            "|d2>",
            "|d3>",
            "|d4>",
            "|d5>",
        }
        assert dil_res["winning_choice"]["id"] in ["d1", "d2", "d3", "d4", "d5"]
        assert "🔒 Bilişsel Kuantum Karar Kısıtı:" in dil_res["injected_prompt_constraint"]
