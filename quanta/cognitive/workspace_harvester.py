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
    latest_commits: list[str] = field(default_factory=list)
    description: str = ""
    domain: str = "general"
    tech_stack: list[str] = field(default_factory=list)
    summary: str = ""
    challenges: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Sync latest_commit and latest_commits for backward compatibility."""
        if self.latest_commit and not self.latest_commits:
            self.latest_commits = [self.latest_commit]
        elif self.latest_commits and not self.latest_commit:
            self.latest_commit = self.latest_commits[0]


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
                "topic": "personal_crm_graph_clustering",
                "question": (
                    "How can spectral graph analysis partition professional contact "
                    "circles to detect isolated high-value relationships?"
                ),
            },
            {
                "topic": "ambient_touchpoint_timing",
                "question": (
                    "Can biomorphic Poisson spindle dynamics optimize ambient post-call "
                    "nudge timing without causing notification fatigue?"
                ),
            },
            {
                "topic": "dual_native_sync_integrity",
                "question": (
                    "How to guarantee atomic bi-directional sync parity between Android "
                    "Room SQLCipher and iOS SwiftData SQLite without conflict loops?"
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

    def _inspect_project(self, p: Path) -> ProjectContext:
        """Deeply inspect a workspace project directory for files, stack, and architecture."""
        git_dir = p / ".git"
        has_git = git_dir.exists()
        latest_commits: list[str] = []

        if has_git:
            try:
                out = subprocess.check_output(
                    ["git", "-C", str(p), "log", "-n", "3", "--pretty=format:%s"],
                    stderr=subprocess.DEVNULL,
                    timeout=1.0,
                )
                latest_commits = [
                    line.strip()
                    for line in out.decode("utf-8", errors="replace").splitlines()
                    if line.strip()
                ]
            except Exception:
                pass

        desc = ""
        tech_stack: list[str] = []
        summary = ""

        # 1. Tech stack detection
        if (p / "android").is_dir() or list(p.glob("*.gradle*")):
            tech_stack.append("Android (Kotlin, Compose, Room)")
        if (p / "ios").is_dir() or list(p.glob("*.swift")):
            tech_stack.append("iOS (Swift, SwiftData, SwiftUI)")

        pkg_json = p / "package.json"
        if pkg_json.exists():
            try:
                with open(pkg_json, encoding="utf-8") as f:
                    data = json.load(f)
                    desc = str(data.get("description", desc))
                    deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
                    for k in deps:
                        if "genkit" in k:
                            tech_stack.append("Firebase Genkit")
                        elif "react" in k and "React" not in tech_stack:
                            tech_stack.append("React")
                        elif "next" in k and "Next.js" not in tech_stack:
                            tech_stack.append("Next.js")
                        elif "d3" in k and "D3.js" not in tech_stack:
                            tech_stack.append("D3.js")
                        elif "firebase" in k and "Firebase" not in tech_stack:
                            tech_stack.append("Firebase")
            except Exception:
                pass

        pyproject = p / "pyproject.toml"
        if pyproject.exists():
            tech_stack.append("Python")
            try:
                with open(pyproject, encoding="utf-8") as f:
                    content = f.read()
                    for line in content.splitlines():
                        if line.strip().startswith("description ="):
                            desc = line.split("=", 1)[1].strip(" '\"\n")
                        if "torch" in line:
                            tech_stack.append("PyTorch")
                        if "mcp" in line:
                            tech_stack.append("MCP")
                        if "bigquery" in line:
                            tech_stack.append("BigQuery")
            except Exception:
                pass

        if (p / "Cargo.toml").exists():
            tech_stack.append("Rust")

        # 2. Deep document inspection (PROJECT.md, README.md, ORIGINAL_REQUEST.md)
        for doc_name in ("PROJECT.md", "README.md", "ORIGINAL_REQUEST.md"):
            doc_path = p / doc_name
            if doc_path.exists():
                try:
                    with open(doc_path, encoding="utf-8") as f:
                        lines = [f.readline() for _ in range(60)]
                        doc_text = "".join(lines).strip()
                        if doc_text:
                            summary = doc_text[:1200]
                            break
                except Exception:
                    pass

        if not desc and summary:
            first_line = summary.split("\n", 1)[0].replace("#", "").strip()
            desc = first_line[:120]

        return ProjectContext(
            name=p.name,
            path=p,
            has_git=has_git,
            latest_commits=latest_commits,
            description=desc,
            tech_stack=list(dict.fromkeys(tech_stack)),
            summary=summary,
        )

    def scan_workspace_projects(self, force: bool = False) -> list[ProjectContext]:
        """Scan the projects directory for active workspaces and deep metadata.

        Args:
            force: When True, bypasses cache and rescans filesystem.

        Returns:
            list[ProjectContext]: Discovered project snapshots with deep context.
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

            projects.append(self._inspect_project(p))

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
            list[DreamSeed]: Candidate seeds labeled with rich project context.
        """
        projects = self.scan_workspace_projects()
        project_map = {p.name: p for p in projects}
        seeds: list[DreamSeed] = []

        # 1. Seeds from known project blueprints enriched with live inspected data
        for proj_name, blueprint_list in self.KNOWN_PROJECT_SEEDS.items():
            p_ctx = project_map.get(proj_name)
            p_path = str(p_ctx.path) if p_ctx else ""
            p_sum = p_ctx.summary if p_ctx else ""
            p_stack = p_ctx.tech_stack if p_ctx else []

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
                        project_path=p_path,
                        project_summary=p_sum,
                        tech_stack=p_stack,
                    )
                )

        # 2. Dynamic seeds derived from live git commits and documents in other discovered projects
        for p in projects:
            if p.name not in self.KNOWN_PROJECT_SEEDS:
                # Synthesize speculative seeds from git commits or docs
                if p.has_git and p.latest_commit:
                    commit_clean = p.latest_commit.replace('"', "").replace("'", "")
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
                            project_path=str(p.path),
                            project_summary=p.summary,
                            tech_stack=p.tech_stack,
                        )
                    )
                elif p.summary:
                    topic_name = f"{p.name.lower().replace(' ', '_').replace('-', '_')}_architecture"
                    question = (
                        f"In project '{p.name}', what are the key architectural bottlenecks "
                        f"and scalability patterns for its tech stack: {', '.join(p.tech_stack) or 'general'}?"
                    )
                    seeds.append(
                        DreamSeed(
                            topic=topic_name,
                            speculative_question=question,
                            urgency=1.8,
                            context_keys=[p.name, topic_name, "document_mined"],
                            project_path=str(p.path),
                            project_summary=p.summary,
                            tech_stack=p.tech_stack,
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
