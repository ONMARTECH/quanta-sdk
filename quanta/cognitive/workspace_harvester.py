"""quanta/cognitive/workspace_harvester.py — Multi-Project Context Harvester.

Pillar 3 Subconscious Mind-Wandering Extension:
Scans the user's workspace ecosystem (/Users/aes/Antigravity Projects), discovers
sibling projects (metabase-ai-assistant, mcp-google-data-studio, app-my-network-planner,
etsgroup_mdm, ContentAI, bigquery-mcp-server, quanta, etc.), extracts real git commits,
recent tasks, and Antigravity brain transcripts, and synthesizes dynamic, diverse
DreamSeeds across multiple projects while strictly preventing cognitive rumination.
"""

from __future__ import annotations

import collections
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from quanta.cognitive.tom_analyzer import DreamSeed

logger = logging.getLogger(__name__)

# Default parent workspace directory on macOS
DEFAULT_PROJECTS_DIR = Path("/Users/aes/Antigravity Projects")
DEFAULT_BRAIN_DIR = Path("/Users/aes/.gemini/antigravity/brain")


@dataclass
class ProjectContext:
    """Structured context snapshot for a discovered workspace project."""

    name: str
    path: Path
    has_git: bool = False
    latest_commit: str = ""
    description: str = ""
    domain: str = "general"
    topics: list[str] = field(default_factory=list)


class WorkspaceContextHarvester:
    """Discovers and mines multi-project context across the Antigravity ecosystem.

    Provides a rich, rotating stream of DreamSeeds spanning disparate projects,
    grounded in actual git activity and recent conversation transcripts, ensuring
    subconscious mind-wandering explores diverse frontiers instead of perseverating.
    """

    KNOWN_PROJECT_SEEDS: dict[str, list[dict[str, str]]] = {
        "metabase-ai-assistant": [
            {
                "topic": "metabase_caching",
                "question": (
                    "How can Metabase MCP pre-cache semantic schema lineage "
                    "during background idle cycles?"
                ),
            },
            {
                "topic": "metabase_sql_templates",
                "question": (
                    "How can native SQL template-tag date filters be verified "
                    "against dialect AST before runtime?"
                ),
            },
            {
                "topic": "metabase_color_palettes",
                "question": (
                    "Can dbt meta brand color palettes be dynamically projected "
                    "onto Metabase chart cards without re-renders?"
                ),
            },
        ],
        "mcp-google-data-studio": [
            {
                "topic": "looker_schema_drift",
                "question": (
                    "How to detect calculated field blast radius and schema drift "
                    "in Looker Studio reports before deployment?"
                ),
            },
            {
                "topic": "d3_viz_caching",
                "question": (
                    "Can D3 custom visual components pre-render cached SVG frames "
                    "to improve perceived report latency?"
                ),
            },
            {
                "topic": "data_lineage_mapping",
                "question": (
                    "How can Mermaid data lineage be automatically generated from "
                    "complex multi-source Data Studio dashboards?"
                ),
            },
        ],
        "app-my-network-planner": [
            {
                "topic": "network_topology_routing",
                "question": (
                    "Can quantum Hamiltonian graph walks optimize multi-hop latency "
                    "in telecom network planning?"
                ),
            },
            {
                "topic": "traffic_matrix_clustering",
                "question": (
                    "How can spectral clustering partition high-throughput "
                    "cellular mesh networks to minimize packet loss?"
                ),
            },
        ],
        "etsgroup_mdm": [
            {
                "topic": "mdm_entity_resolution",
                "question": (
                    "How can biomorphic consensus arbitration resolve conflicting "
                    "customer identity records across disparate sources?"
                ),
            },
            {
                "topic": "golden_record_survivorship",
                "question": (
                    "How to maintain immutable audit lineage for Golden Record "
                    "survivorship rules across distributed pipelines?"
                ),
            },
        ],
        "customer-mdm": [
            {
                "topic": "cdp_realtime_stewardship",
                "question": (
                    "How can streaming change-data-capture (CDC) pipelines "
                    "reconcile customer identity graphs without locking?"
                ),
            },
        ],
        "ContentAI": [
            {
                "topic": "redis_prompt_freshness",
                "question": (
                    "How to optimize multi-tiered Redis caching and prompt "
                    "freshness validation without cache thundering?"
                ),
            },
        ],
        "bigquery-mcp-server": [
            {
                "topic": "bigquery_cost_guardrails",
                "question": (
                    "Can pre-flight dry-run AST cost estimation prevent runaway "
                    "cloud query expenditures in automated workflows?"
                ),
            },
        ],
        "quanta": [
            {
                "topic": "continuous_resonance",
                "question": (
                    "Can continuous-time quantum resonance layers maintain unitary "
                    "norm stability on Apple Silicon Metal MPS?"
                ),
            },
            {
                "topic": "landauer_efficiency",
                "question": (
                    "Does zero entropy production dS=0 hold during unitary "
                    "deliberation before projective consensus collapse?"
                ),
            },
            {
                "topic": "anti_rumination_reset",
                "question": (
                    "How can synthetic noradrenaline phase shifts kick the "
                    "Default Mode Network out of cognitive loops?"
                ),
            },
            {
                "topic": "quantum_rem_sleep",
                "question": (
                    "How does uncoupled Hamiltonian sleep annealing orthogonalize "
                    "memory subspaces to prevent catastrophic forgetting?"
                ),
            },
            {
                "topic": "darwin_qos_optimization",
                "question": (
                    "How to guarantee zero thread contention on Apple Silicon "
                    "Efficiency cores during background subconscious deliberation?"
                ),
            },
        ],
    }

    def __init__(
        self,
        projects_dir: Path | str | None = None,
        brain_dir: Path | str | None = None,
        state_file: Path | str | None = "quanta_cognitive_state.json",
        history_len: int = 20,
    ) -> None:
        """Initialize harvester with paths and anti-rumination history buffer.

        Args:
            projects_dir: Directory containing user workspace projects.
            brain_dir: Directory containing Antigravity session brain transcripts.
            state_file: Path to quanta_cognitive_state.json.
            history_len: Max recent topics retained to prevent repetition.
        """
        if projects_dir is not None:
            self.projects_dir = Path(projects_dir)
        elif DEFAULT_PROJECTS_DIR.exists():
            self.projects_dir = DEFAULT_PROJECTS_DIR
        else:
            # Fall back to parent of current working directory
            self.projects_dir = Path.cwd().parent

        self.brain_dir = Path(brain_dir) if brain_dir else DEFAULT_BRAIN_DIR
        self.state_file = Path(state_file) if state_file else Path("quanta_cognitive_state.json")

        self._recent_topics: collections.deque[str] = collections.deque(maxlen=history_len)
        self._project_rotation_idx: int = 0
        self._cached_projects: list[ProjectContext] = []
        self._last_scan_time: float = 0.0

    def scan_workspace_projects(self, force: bool = False) -> list[ProjectContext]:
        """Scan the projects directory for active workspaces and git commit metadata.

        Args:
            force: When True, bypasses cache and rescans filesystem.

        Returns:
            list[ProjectContext]: Discovered project snapshots.
        """
        if self._cached_projects and not force:
            return self._cached_projects

        if not self.projects_dir.exists():
            return []

        projects: list[ProjectContext] = []

        try:
            entries = sorted(self.projects_dir.iterdir())
        except OSError:
            return []

        for p in entries:
            if not p.is_dir() or p.name.startswith("."):
                continue

            git_dir = p / ".git"
            has_git = git_dir.exists()
            latest_commit = ""

            if has_git:
                try:
                    out = subprocess.check_output(
                        ["git", "-C", str(p), "log", "-n", "1", "--pretty=format:%s"],
                        stderr=subprocess.DEVNULL,
                        timeout=1.0,
                    )
                    latest_commit = out.decode("utf-8", errors="replace").strip()
                except Exception:
                    pass

            desc = ""
            pkg_json = p / "package.json"
            pyproject = p / "pyproject.toml"

            if pkg_json.exists():
                try:
                    with open(pkg_json, encoding="utf-8") as f:
                        data = json.load(f)
                        desc = str(data.get("description", ""))
                except Exception:
                    pass
            elif pyproject.exists():
                try:
                    with open(pyproject, encoding="utf-8") as f:
                        for line in f:
                            if line.strip().startswith("description ="):
                                desc = line.split("=", 1)[1].strip(" '\"\n")
                                break
                except Exception:
                    pass

            projects.append(
                ProjectContext(
                    name=p.name,
                    path=p,
                    has_git=has_git,
                    latest_commit=latest_commit,
                    description=desc,
                )
            )

        self._cached_projects = projects
        return projects

    def _get_crystallized_topics(self) -> set[str]:
        """Read set of topics already crystallized into persistent engrams."""
        if not self.state_file.exists():
            return set()
        try:
            with open(self.state_file, encoding="utf-8") as f:
                data = json.load(f)
            engrams = data.get("engrams", [])
            topics = set()
            for item in engrams:
                t = item.get("topic")
                if t:
                    topics.add(str(t))
                k = item.get("key", "")
                if k.startswith("insight_"):
                    topics.add(k[len("insight_") :])
            return topics
        except Exception:
            return set()

    def generate_candidate_seeds(self) -> list[DreamSeed]:
        """Synthesize candidate DreamSeeds across discovered projects and domains.

        Returns:
            list[DreamSeed]: Candidate seeds labeled with project context.
        """
        projects = self.scan_workspace_projects()
        project_names = {p.name: p for p in projects}
        seeds: list[DreamSeed] = []

        # 1. Seeds from known project blueprints
        for proj_name, blueprint_list in self.KNOWN_PROJECT_SEEDS.items():
            # Match against active projects or include if relevant
            p_ctx = project_names.get(proj_name)
            for item in blueprint_list:
                topic = item["topic"]
                question = item["question"]
                urgency = 2.0 if (p_ctx and p_ctx.has_git and p_ctx.latest_commit) else 1.5

                seeds.append(
                    DreamSeed(
                        topic=topic,
                        speculative_question=question,
                        urgency=urgency,
                        context_keys=[proj_name, topic, "workspace_harvested"],
                    )
                )

        # 2. Dynamic seeds derived from live git commits in other projects
        for p in projects:
            if p.has_git and p.latest_commit and p.name not in self.KNOWN_PROJECT_SEEDS:
                # Synthesize a speculative seed from the commit message
                commit_clean = p.latest_commit.replace('"', "").replace("'", "")
                # Truncate clean message
                if len(commit_clean) > 80:
                    commit_clean = commit_clean[:77] + "..."
                topic_name = f"{p.name.lower().replace(' ', '_').replace('-', '_')}_evolution"
                question = (
                    f"In project '{p.name}', how does '{commit_clean}' impact downstream "
                    f"architecture and system invariants?"
                )
                seeds.append(
                    DreamSeed(
                        topic=topic_name,
                        speculative_question=question,
                        urgency=2.2,
                        context_keys=[p.name, topic_name, "live_git_commit"],
                    )
                )

        return seeds

    def generate_next_seed(self) -> DreamSeed:
        """Select and return the next optimal DreamSeed, strictly avoiding rumination.

        Rotates across different projects, filters out recently visited topics,
        and ensures fresh, high-utility cognitive exploration.

        Returns:
            DreamSeed: The next speculative seed to deliberate on.
        """
        candidates = self.generate_candidate_seeds()
        if not candidates:
            # Absolute fallback if no projects discovered
            return DreamSeed(
                topic="subconscious_orchestration",
                speculative_question=(
                    "How can the subconscious daemon optimize background thread scheduling?"
                ),
                urgency=1.5,
                context_keys=["quanta", "subconscious_orchestration"],
            )

        crystallized = self._get_crystallized_topics()

        # Group candidates by primary project key
        by_project: dict[str, list[DreamSeed]] = collections.defaultdict(list)
        for seed in candidates:
            proj = seed.context_keys[0] if seed.context_keys else "general"
            by_project[proj].append(seed)

        project_list = sorted(by_project.keys())
        if not project_list:
            return candidates[0]

        # Select project via round-robin index
        num_projects = len(project_list)
        chosen_seed: DreamSeed | None = None

        for offset in range(num_projects):
            idx = (self._project_rotation_idx + offset) % num_projects
            proj = project_list[idx]
            proj_seeds = by_project[proj]

            # Filter candidates:
            # 1. Topic not in recent history
            # 2. Topic not already crystallized (or pick least recent)
            fresh_seeds = [
                s
                for s in proj_seeds
                if s.topic not in self._recent_topics and s.topic not in crystallized
            ]
            if not fresh_seeds:
                # Fallback 1: Topic not in recent history (even if crystallized previously)
                fresh_seeds = [s for s in proj_seeds if s.topic not in self._recent_topics]

            if fresh_seeds:
                chosen_seed = fresh_seeds[0]
                self._project_rotation_idx = (idx + 1) % num_projects
                break

        if chosen_seed is None:
            # If all topics are exhausted/recent, clear half the history and choose first available
            for _ in range(len(self._recent_topics) // 2):
                if self._recent_topics:
                    self._recent_topics.popleft()
            chosen_seed = candidates[self._project_rotation_idx % len(candidates)]
            self._project_rotation_idx = (self._project_rotation_idx + 1) % num_projects

        self._recent_topics.append(chosen_seed.topic)
        return chosen_seed
