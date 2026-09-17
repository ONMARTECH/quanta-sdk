"""Unit tests for QuantaCognitiveMiddleware."""

from quanta.cognitive.middleware import QuantaCognitiveMiddleware


class TestQuantaCognitiveMiddleware:
    def test_initialization_and_recording(self) -> None:
        mw = QuantaCognitiveMiddleware(capacity=10, dim=16)
        idx = mw.record_constraint("jwt_rule", "Never log raw JWT", salience=2.5)
        assert idx == 0
        assert mw.turn_count == 0

        # Turn start step
        vitals = mw.on_turn_start(dt=1.0)
        assert mw.turn_count == 1
        assert len(vitals) == 1
        assert vitals[0]["key"] == "jwt_rule"

        # Anchor text generation
        anchor_text = mw.get_subconscious_anchor_text(top_k=3)
        assert anchor_text is not None
        assert "jwt_rule" in anchor_text
        assert "Never log raw JWT" in anchor_text

    def test_conscious_pruning_and_updating(self) -> None:
        mw = QuantaCognitiveMiddleware(capacity=10, dim=16, auto_prune=True)
        mw.record_constraint("core_db", "Use Postgres", salience=2.0)
        mw.record_constraint("temp_note", "Test port 3000", salience=0.2)

        # Update decision supersedes old one
        mw.update_decision("core_db", "Switch to Cloud Spanner", salience=2.5)
        assert len(mw.memory.buffer.buffer) == 2
        core_entry = next(e for e in mw.memory.buffer.buffer if e["metadata"]["key"] == "core_db")
        assert "Cloud Spanner" in core_entry["metadata"]["content"]

        # Run 30 turns to cause decay in low salience note
        for _ in range(30):
            mw.on_turn_start(dt=1.0)

        # Temp note with salience 0.2 should have decayed and been pruned
        keys = [e["metadata"]["key"] for e in mw.memory.buffer.buffer]
        assert "core_db" in keys
        assert "temp_note" not in keys

    def test_subconscious_arbitration(self) -> None:
        mw = QuantaCognitiveMiddleware(capacity=10, dim=16)
        goal = "Ensure low latency in real-time data streaming"
        options = [
            "Use distributed Kafka topic with partitioned consumer groups",
            "Periodically dump full database table to CSV every 5 minutes",
            "Store all streaming packets in local browser localStorage",
        ]

        result = mw.arbitrate_decision(goal, options, exploration_drive=0.1)
        assert "recommended_option" in result
        assert result["recommended_option"] in options
        assert result["zeno_pinning_factor"] >= 0.5
        assert len(result["ranked_options"]) == 3

    def test_telemetry(self) -> None:
        mw = QuantaCognitiveMiddleware(capacity=5, dim=16)
        mw.record_constraint("rule_a", "Constraint A", salience=2.0)
        mw.on_turn_start()

        tel = mw.get_telemetry()
        assert tel["turn_count"] == 1
        assert tel["stored_engrams"] == 1
        assert tel["csf_shielded"] is True
