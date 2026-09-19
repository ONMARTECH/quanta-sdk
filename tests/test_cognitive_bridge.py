"""Unit and Integration tests for Quanta Cognitive Bridge (Memory & Arbiter)."""

import pytest
import torch

from quanta.cognitive.arbiter import QuantumDecisionArbiter
from quanta.cognitive.memory import CognitiveMemoryManager, text_to_statevector


class TestTextToStatevector:
    def test_normalization_and_determinism(self) -> None:
        vec1 = text_to_statevector("Use PostgreSQL for production database", dim=16)
        vec2 = text_to_statevector("Use PostgreSQL for production database", dim=16)
        vec3 = text_to_statevector("Use MongoDB for document storage", dim=16)

        assert vec1.shape == (16,)
        assert vec1.is_complex()
        assert torch.allclose(vec1, vec2)
        assert not torch.allclose(vec1, vec3)

        norm = torch.linalg.norm(vec1)
        assert torch.isclose(norm, torch.tensor(1.0, dtype=norm.dtype), atol=1e-5)


class TestCognitiveMemoryManager:
    def test_record_and_recall_vital_context(self) -> None:
        mem = CognitiveMemoryManager(capacity=10, dim=16, enable_csf_shielding=True)

        # Store critical constraint (salience = 2.0)
        idx_crit = mem.record_decision(
            key="security_rule",
            content="Never log API keys or raw user tokens to console",
            salience=2.0,
            category="constraint",
        )
        assert idx_crit == 0

        # Store intermediate decision (salience = 0.5)
        idx_temp = mem.record_decision(
            key="temp_log",
            content="Cache query results in memory during testing",
            salience=0.5,
            category="temporary",
        )
        assert idx_temp == 1

        # Simulate 10 turns of conversation passing
        for _ in range(10):
            mem.step(dt=1.0)

        # Recall vital context
        recalled = mem.recall_vital_context(top_k=2)
        assert len(recalled) == 2

        # The critical security rule must rank first due to higher dopamine protection
        assert recalled[0]["key"] == "security_rule"
        assert recalled[0]["salience"] == 2.0
        assert recalled[0]["retention_fidelity"] > 0.0

        status = mem.get_status_summary()
        assert status["stored_engrams"] == 2
        assert status["csf_shielded"] is True
        assert status["mean_retention_fidelity"] > 0.0

    def test_conscious_forgetting_and_pruning(self) -> None:
        mem = CognitiveMemoryManager(capacity=10, dim=16)
        mem.record_decision(key="temp_key", content="Temporary value", salience=0.5)
        mem.record_decision(key="perm_key", content="Permanent value", salience=2.0)
        assert len(mem.buffer.buffer) == 2

        # Explicit conscious forgetting
        forgotten = mem.forget("temp_key")
        assert forgotten is True
        assert len(mem.buffer.buffer) == 1
        assert mem.buffer.buffer[0]["metadata"]["key"] == "perm_key"

        # Forgetting non-existent key returns False
        assert mem.forget("non_existent") is False
        assert mem.get_status_summary()["total_pruned_count"] == 1

    def test_decision_updating_and_superseding(self) -> None:
        mem = CognitiveMemoryManager(capacity=10, dim=16)
        mem.record_decision(key="db_engine", content="Use SQLite", salience=1.0)
        assert len(mem.buffer.buffer) == 1
        assert "SQLite" in mem.buffer.buffer[0]["metadata"]["content"]

        # Update decision supersedes old one
        mem.update_decision(key="db_engine", content="Switch to Cloud Spanner", salience=2.5)
        assert len(mem.buffer.buffer) == 1
        assert "Cloud Spanner" in mem.buffer.buffer[0]["metadata"]["content"]
        assert mem.buffer.buffer[0]["metadata"]["salience"] == 2.5
        assert mem.get_status_summary()["total_pruned_count"] == 1

    def test_active_synaptic_pruning_obsolete(self) -> None:
        mem = CognitiveMemoryManager(capacity=10, dim=16, enable_csf_shielding=True)
        mem.record_decision(key="critical_constraint", content="Never log secrets", salience=2.5)
        mem.record_decision(key="scratch_note", content="Temporary port 3005", salience=0.2)

        # Advance 25 turns to induce decay in low-salience note
        for _ in range(25):
            mem.step(dt=1.0)

        # Critical constraint should remain high
        recalled = mem.recall_vital_context(top_k=2)
        crit = next(r for r in recalled if r["key"] == "critical_constraint")
        assert crit["retention_fidelity"] > 0.95
        assert crit["status"] in ("pristine", "consolidated")

        # Active conscious pruning pass
        pruned = mem.prune_obsolete(fidelity_threshold=0.85, min_salience=0.5)
        assert len(pruned) >= 1
        assert pruned[0]["key"] == "scratch_note"

        # Only critical constraint should survive in memory buffer
        summary = mem.get_status_summary()
        assert summary["stored_engrams"] == 1
        assert "critical_constraint" in summary["active_keys"]
        assert "scratch_note" not in summary["active_keys"]

    def test_smart_capacity_eviction_protects_critical_rule(self) -> None:
        # Capacity is 2
        mem = CognitiveMemoryManager(capacity=2, dim=16)

        # Turn 0: store critical rule (salience = 3.0)
        mem.record_decision(key="vital_security", content="TLS 1.3 only", salience=3.0)
        # Turn 1: store intermediate note (salience = 0.8)
        mem.record_decision(key="note_1", content="Note 1", salience=0.8)
        assert len(mem.buffer.buffer) == 2

        # Turn 2: store note 2 (salience = 0.5) -> capacity exceeded!
        # Smart eviction should discard note_1 (lowest salience * fid), NOT vital_security!
        mem.record_decision(key="note_2", content="Note 2", salience=0.5)
        assert len(mem.buffer.buffer) == 2

        keys = [e["metadata"]["key"] for e in mem.buffer.buffer]
        assert "vital_security" in keys
        assert "note_2" in keys
        assert "note_1" not in keys



class TestQuantumDecisionArbiter:
    def test_arbitration_pinning_and_ranking(self) -> None:
        arb = QuantumDecisionArbiter(dim=16, num_heads=2)

        goal = "Optimize database query performance under high concurrency"
        options = [
            "Add clustered B-Tree index on tenant_id and created_at",
            "Use full-table sequential scan with regex matching",
            "Store all query rows in a flat JSON file on disk",
        ]

        # Low exploration -> Strong Zeno pinning
        res_pin = arb.arbitrate(goal, options, exploration_drive=0.1)

        assert "recommended_option" in res_pin
        assert res_pin["recommended_option"] in options
        assert res_pin["zeno_pinning_factor"] >= 0.5
        assert len(res_pin["ranked_options"]) == 3

        # Scores must sum to approximately 1.0
        total_score = sum(opt["score"] for opt in res_pin["ranked_options"])
        assert pytest.approx(total_score, abs=1e-2) == 1.0

        # High exploration -> Anti-Zeno tunneling
        res_tunnel = arb.arbitrate(goal, options, exploration_drive=0.9)
        assert res_tunnel["anti_zeno_kickback"] > 0.0

    def test_6qubit_default_arbitration(self) -> None:
        """Verifies the upgraded 6-qubit (dim=64, 4-head attention) default configuration."""
        arb = QuantumDecisionArbiter()
        assert arb.dim == 64
        assert arb.zeno_attention.num_heads == 4
        assert arb.zeno_attention.head_dim == 16

        goal = "High availability with zero downtime schema migrations"
        options = [
            "Blue-Green database deployment with dual-write proxy",
            "Direct in-place table rewrite during peak production hours",
            "Truncate and recreate schema on live cluster",
        ]
        res = arb.arbitrate(goal, options, exploration_drive=0.2)
        assert res["recommended_option"] == options[0]
        assert res["confidence"] > 0.35
        assert len(res["ranked_options"]) == 3

    def test_multi_option_with_criteria_and_effects(self) -> None:
        """Verifies multi-candidate (N=5) structured impact analysis with criteria."""
        arb = QuantumDecisionArbiter(dim=64, num_heads=4)

        goal = "Select cache storage architecture for high throughput user sessions"
        criteria = [
            "P99 latency under 5 milliseconds",
            "Zero data loss across cluster failover",
            "Minimal infrastructure maintenance cost",
        ]
        options = [
            {
                "name": "Redis Sentinel",
                "impact": "Low sub-millisecond latency, automatic failover",
                "risk": "Moderate memory cost",
            },
            {
                "name": "Local In-Memory",
                "impact": "Zero network latency, free",
                "risk": "Data lost on server restart, no shared state",
            },
            {
                "name": "PostgreSQL Table",
                "impact": "ACID compliance, persistent",
                "risk": "High database connection pool contention under load",
            },
            {
                "name": "Cloudflare KV",
                "impact": "Global edge deployment",
                "risk": "Eventual consistency write latency",
            },
            {
                "name": "Disk Flat File",
                "impact": "Simple filesystem writes",
                "risk": "Extremely slow IO blocking under concurrency",
            },
        ]

        res = arb.arbitrate(goal=goal, options=options, criteria=criteria, log_telemetry=False)

        assert len(res["ranked_options"]) == 5
        assert res["latency_ms"] < 50.0
        assert "recommended_option" in res
        assert "recommended_details" in res
        assert any(r["option"] == "Redis Sentinel" for r in res["ranked_options"])

