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
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quanta.cognitive.daemon import SubconsciousDaemon
from quanta.cognitive.darwin_idle import DarwinIdleMonitor, get_user_idle_seconds
from quanta.cognitive.workspace_harvester import (
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
