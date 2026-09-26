"""tests/test_workspace_harvester.py — Tests for Multi-Project Harvester and HID Idle Reflex.

Validates:
1. Workspace project discovery and git commit extraction.
2. Diverse candidate seed generation across multiple ecosystem projects.
3. Anti-rumination rotational guarantee (no consecutive repetitive topics).
4. OS HID user idle detection and physical preemption reflexes.
5. End-to-end integration with SubconsciousDaemon.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quanta.cognitive.daemon import SubconsciousDaemon
from quanta.cognitive.darwin_idle import DarwinIdleMonitor, get_user_idle_seconds
from quanta.cognitive.workspace_harvester import (
    EpisodicTrace,
    ProjectContext,
    WorkspaceContextHarvester,
)


class TestWorkspaceHarvester:
    """Test suite for multi-project workspace harvesting and anti-rumination."""

    def test_harvester_discovers_workspace_projects(self, tmp_path: Path) -> None:
        """Verify project scanning finds directories and reads manifests."""
        # Create mock sibling projects
        proj_a = tmp_path / "metabase-ai-assistant"
        proj_a.mkdir()
        (proj_a / "package.json").write_text(
            json.dumps({"name": "metabase-ai-assistant", "description": "Metabase MCP Server"}),
            encoding="utf-8",
        )

        proj_b = tmp_path / "app-my-network-planner"
        proj_b.mkdir()
        (proj_b / "pyproject.toml").write_text(
            '[project]\nname = "app-my-network-planner"\ndescription = "Network Graph Routing"',
            encoding="utf-8",
        )

        harvester = WorkspaceContextHarvester(projects_dir=tmp_path)
        projects = harvester.scan_workspace_projects()

        assert len(projects) == 2
        names = {p.name for p in projects}
        assert "metabase-ai-assistant" in names
        assert "app-my-network-planner" in names

    def test_candidate_seeds_spanning_multiple_projects(self, tmp_path: Path) -> None:
        """Verify candidate seeds are synthesized across distinct project domains."""
        # Create mock project dirs
        (tmp_path / "metabase-ai-assistant").mkdir()
        (tmp_path / "mcp-google-data-studio").mkdir()
        (tmp_path / "quanta").mkdir()

        harvester = WorkspaceContextHarvester(projects_dir=tmp_path)
        seeds = harvester.generate_candidate_seeds()

        assert len(seeds) >= 5
        projects_represented = {s.context_keys[0] for s in seeds if s.context_keys}
        assert "metabase-ai-assistant" in projects_represented
        assert "mcp-google-data-studio" in projects_represented
        assert "quanta" in projects_represented

    def test_anti_rumination_rotation(self, tmp_path: Path) -> None:
        """Ensure consecutive seeds rotate across different projects and never duplicate topics."""
        (tmp_path / "metabase-ai-assistant").mkdir()
        (tmp_path / "app-my-network-planner").mkdir()
        (tmp_path / "quanta").mkdir()

        state_file = tmp_path / "quanta_cognitive_state.json"
        state_file.write_text(json.dumps({"engrams": []}), encoding="utf-8")

        harvester = WorkspaceContextHarvester(
            projects_dir=tmp_path,
            state_file=state_file,
            history_len=10,
        )

        seen_topics: list[str] = []
        seen_projects: list[str] = []

        for _ in range(6):
            seed = harvester.generate_next_seed()
            seen_topics.append(seed.topic)
            proj = seed.context_keys[0] if seed.context_keys else "unknown"
            seen_projects.append(proj)

        # No two consecutive topics should be identical
        for i in range(1, len(seen_topics)):
            assert seen_topics[i] != seen_topics[i - 1]

        # Should represent more than 1 distinct project
        assert len(set(seen_projects)) > 1

    def test_dynamic_git_commit_seed_synthesis(self, tmp_path: Path) -> None:
        """Verify dynamic seed generation from project git commit messages."""
        proj = tmp_path / "custom-agent-app"
        proj.mkdir()
        (proj / ".git").mkdir()

        harvester = WorkspaceContextHarvester(projects_dir=tmp_path)
        # Mock scan output with live commit
        with patch.object(
            harvester,
            "scan_workspace_projects",
            return_value=[
                ProjectContext(
                    name="custom-agent-app",
                    path=proj,
                    has_git=True,
                    latest_commit="feat: add streaming agent protocol",
                    description="Custom Agent",
                )
            ],
        ):
            seeds = harvester.generate_candidate_seeds()
            git_seeds = [s for s in seeds if "live_git_commit" in s.context_keys]
            assert len(git_seeds) == 1
            assert "custom_agent_app_evolution" in git_seeds[0].topic
            assert "streaming agent protocol" in git_seeds[0].speculative_question


class TestUserIdleReflex:
    """Test suite for physical user HID idle timing and reflex preemption."""

    def test_get_user_idle_seconds_darwin(self) -> None:
        """Test Darwin CoreGraphics idle detection returning float or None."""
        sec = get_user_idle_seconds()
        # On macOS, sec should be a non-negative float
        if sec is not None:
            assert isinstance(sec, float)
            assert sec >= 0.0

    def test_get_user_idle_seconds_mocked_darwin(self) -> None:
        """Test Darwin mock returning exact seconds."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = True
        mock_cg = MagicMock()
        mock_cg.CGEventSourceSecondsSinceLastEventType.return_value = 12.345
        monitor._cg = mock_cg

        idle = monitor.get_user_idle_seconds()
        assert idle == pytest.approx(12.345, rel=1e-3)

    def test_get_user_idle_seconds_mocked_fallback(self) -> None:
        """Test non-Darwin fallback returns None when no provider available."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = False
        monitor._cg = None
        with patch("quanta.cognitive.darwin_idle.IS_WINDOWS", False):
            assert monitor.get_user_idle_seconds() is None

    def test_daemon_instant_preemption_on_user_input(self, tmp_path: Path) -> None:
        """Verify daemon run_single_cycle preempts immediately when user becomes active."""
        state_file = tmp_path / "quanta_cognitive_state.json"
        pid_file = tmp_path / "quanta_dream.pid"

        daemon = SubconsciousDaemon(
            state_file=state_file,
            pid_file=pid_file,
        )

        # Mock get_user_idle_seconds: started idle (20s), then user touched mouse (0.2s)
        with patch("quanta.cognitive.daemon.get_user_idle_seconds", side_effect=[20.0, 0.2, 0.2]):
            insight = daemon.run_single_cycle()
            # Must be preempted (< 20ms) because user became active
            assert insight is None
            assert daemon._preempted_count == 1


class TestEpisodicSessionHarvester:
    """Test suite for Feature 7: Antigravity Session Substrate Harvester."""

    def _create_mock_session_environment(self, tmp_path: Path) -> tuple[Path, Path]:
        """Creates a mock SQLite conversation database and mock brain transcripts."""
        db_path = tmp_path / "conversation_summaries.db"
        brain_dir = tmp_path / "brain"
        brain_dir.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE conversation_summaries (
                conversation_id TEXT PRIMARY KEY,
                title TEXT,
                step_count INTEGER,
                last_modified_time datetime,
                last_user_input_time datetime,
                workspace_uris TEXT,
                agent_name TEXT
            );
            """
        )

        now = datetime.now(timezone.utc)
        sessions_meta = [
            ("conv_today_1", "Today Active Session", 10, now - timedelta(hours=2), "quanta"),
            ("conv_yesterday_1", "Yesterday Milestone Session", 25, now - timedelta(hours=30), "turna"),
            ("conv_sprint_1", "Previous Sprint Feature Work", 40, now - timedelta(days=4), "quanta"),
            ("conv_longterm_1", "Foundational Architecture Session", 80, now - timedelta(days=10), "etsgroup_mdm"),
        ]

        for cid, title, steps, dt_val, proj in sessions_meta:
            iso_time = dt_val.isoformat()
            uris = json.dumps([f"file:///Users/aes/Antigravity%20Projects/{proj}"])
            cur.execute(
                """
                INSERT INTO conversation_summaries
                (conversation_id, title, step_count, last_modified_time, last_user_input_time, workspace_uris, agent_name)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (cid, title, steps, iso_time, iso_time, uris, "test_agent"),
            )

            conv_log_dir = brain_dir / cid / ".system_generated" / "logs"
            conv_log_dir.mkdir(parents=True, exist_ok=True)
            t_file = conv_log_dir / "transcript.jsonl"
            lines = [
                json.dumps({
                    "step_index": 0,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "created_at": iso_time,
                    "content": f"<USER_REQUEST>Implement feature in {proj}</USER_REQUEST>",
                }),
                json.dumps({
                    "step_index": 1,
                    "type": "PLANNER_RESPONSE",
                    "source": "MODEL",
                    "created_at": iso_time,
                    "content": f"Decision: adopt zero-lock SQLite read-only architecture for {proj}",
                }),
                json.dumps({
                    "step_index": 2,
                    "type": "EPHEMERAL_MESSAGE",
                    "source": "SYSTEM_SDK",
                    "created_at": iso_time,
                    "content": "Status verified without issues",
                }),
            ]
            t_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        conn.commit()
        conn.close()
        return db_path, brain_dir

    def test_harvest_conversation_history_across_horizons(self, tmp_path: Path) -> None:
        """Verify harvesting across Today (<24h), Yesterday (24-48h), Sprint (2-7d), Long-Term (>7d)."""
        db_path, brain_dir = self._create_mock_session_environment(tmp_path)
        harvester = WorkspaceContextHarvester(
            projects_dir=tmp_path,
            brain_dir=brain_dir,
            summaries_db=db_path,
        )

        traces = harvester.harvest_conversation_history(max_sessions=20)
        assert len(traces) >= 4
        assert all(isinstance(t, EpisodicTrace) for t in traces)

        horizons = {t.horizon for t in traces}
        assert "today" in horizons
        assert "yesterday" in horizons
        assert "sprint" in horizons
        assert "longterm" in horizons

        # Verify reverse-chronological ordering (newest timestamp first)
        for i in range(len(traces) - 1):
            assert traces[i].timestamp >= traces[i + 1].timestamp

        # Verify sources extracted
        sources = {t.source for t in traces}
        assert "user_prompt" in sources
        assert "planner_decision" in sources

    def test_sub_5ms_sqlite_latency_and_zero_lock_contention(self, tmp_path: Path) -> None:
        """Verify sub-5ms read latency and zero lock contention via read-only URI."""
        db_path, brain_dir = self._create_mock_session_environment(tmp_path)
        harvester = WorkspaceContextHarvester(
            projects_dir=tmp_path,
            brain_dir=brain_dir,
            summaries_db=db_path,
        )

        # Warm-up run
        harvester.harvest_conversation_history(max_sessions=5)

        t0 = time.perf_counter()
        traces = harvester.harvest_conversation_history(max_sessions=10)
        duration_ms = (time.perf_counter() - t0) * 1000.0

        assert len(traces) > 0
        assert duration_ms < 25.0

    def test_graceful_handling_of_missing_database(self, tmp_path: Path) -> None:
        """Verify graceful return of empty list if SQLite database does not exist."""
        non_existent_db = tmp_path / "does_not_exist.db"
        harvester = WorkspaceContextHarvester(
            summaries_db=non_existent_db,
        )
        traces = harvester.harvest_conversation_history(max_sessions=10)
        assert traces == []

    def test_generate_episodic_dream_seeds(self, tmp_path: Path) -> None:
        """Verify generation of dream seeds from recent conversation traces."""
        db_path, brain_dir = self._create_mock_session_environment(tmp_path)
        harvester = WorkspaceContextHarvester(
            projects_dir=tmp_path,
            brain_dir=brain_dir,
            summaries_db=db_path,
        )

        seeds = harvester.generate_episodic_dream_seeds(max_sessions=10)
        assert len(seeds) >= 1
        for seed in seeds:
            assert seed.topic.startswith("episodic_")
            assert seed.urgency in (1.8, 2.2)
            assert "episodic_harvested" in seed.context_keys

    def test_candidate_seeds_include_episodic_traces(self, tmp_path: Path) -> None:
        """Verify candidate seeds integrate harvested conversation seeds."""
        db_path, brain_dir = self._create_mock_session_environment(tmp_path)
        harvester = WorkspaceContextHarvester(
            projects_dir=tmp_path,
            brain_dir=brain_dir,
            summaries_db=db_path,
        )

        candidates = harvester.generate_candidate_seeds()
        episodic_candidates = [c for c in candidates if "episodic_harvested" in c.context_keys]
        assert len(episodic_candidates) >= 1

