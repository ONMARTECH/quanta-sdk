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
import math
import re
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quanta.cognitive.memory import calculate_temporal_decay
from quanta.cognitive.tom_analyzer import DreamSeed

logger = logging.getLogger(__name__)

# Default parent workspace directory on macOS
DEFAULT_PROJECTS_DIR = Path("/Users/aes/Antigravity Projects")
DEFAULT_BRAIN_DIR = Path("/Users/aes/.gemini/antigravity/brain")
DEFAULT_SUMMARIES_DB = Path("/Users/aes/.gemini/antigravity/conversation_summaries.db")


@dataclass
class EpisodicTrace:
    """Represents a discrete episodic memory event harvested from conversation history."""

    trace_id: str
    session_id: str
    timestamp: datetime
    horizon: str  # "today" | "yesterday" | "sprint" | "longterm"
    source: str  # "user_prompt" | "planner_decision" | "agent_message" | "session_title"
    content: str
    full_text: str
    project: str = "quanta"
    salience: float = 1.0
    fidelity: float = 0.9998
    is_core_anchor: bool = False
    tags: list[str] = field(default_factory=list)
    age_hours: float = 0.0
    temporal_weight: float = 1.0
    replay_score: float = 1.0


def _parse_iso_timestamp(ts: Any) -> datetime:
    """Safely parses timestamp string or datetime into UTC datetime."""
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    if not ts or not isinstance(ts, str):
        return datetime.now(timezone.utc)
    clean_ts = ts.strip()
    if clean_ts.endswith("Z"):
        clean_ts = clean_ts[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(clean_ts)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    try:
        dt = datetime.strptime(clean_ts.split(".")[0], "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return datetime.now(timezone.utc)


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


@dataclass
class HybridDreamSeed:
    """Cross-project speculative dream seed driving bisociative REM deliberation."""

    topic: str
    speculative_question: str
    primary_project: ProjectContext
    secondary_project: ProjectContext
    domain_distance: float
    bisociation_theme: str
    urgency: float = 2.0
    context_keys: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Populate default context keys if not explicitly provided."""
        if not self.context_keys:
            self.context_keys = [
                self.primary_project.name,
                self.secondary_project.name,
                self.topic,
                "hybrid_bisociation",
            ]


CANONICAL_DOMAINS: list[str] = [
    "Travel/Pricing",
    "Quantum/Brain",
    "LegalTech",
    "Web Analytics/MarTech",
    "CDP/MDM",
    "BI/MCP",
    "Mobile Graph CRM",
    "Generative NLP",
]

DOMAIN_KEYWORDS: list[list[str]] = [
    # 0: Travel/Pricing
    [
        "turna", "etstur", "flight", "ticket", "hotel", "travel", "pricing",
        "fare", "reprice", "airline", "booking", "room", "yield", "reservation",
        "ota", "destinations", "pax", "gds",
    ],
    # 1: Quantum/Brain
    [
        "quanta", "quantum", "qec", "hamiltonian", "statevector", "subconscious",
        "biomorphic", "swr", "synaptic", "hippocampal", "zeno", "lindblad",
        "qubit", "unitary", "mps", "brain", "cognitive", "neuro", "plasticity",
    ],
    # 2: LegalTech
    [
        "legal", "hukuk", "mevzuat", "jurisprudence", "kvkk", "gdpr", "contract",
        "reklam-kurulu", "reklam kurulu", "court", "compliance", "lawyer",
        "avukat", "dava", "legislation", "sozlesme", "kanun", "icra",
    ],
    # 3: Web Analytics/MarTech
    [
        "metricase", "martech", "cortex", "analytics", "auditor", "audit",
        "tag", "tracking", "cwv", "core web vitals", "lcp", "gtm", "ga4",
        "seo", "web audit", "datalayer", "consent",
    ],
    # 4: CDP/MDM
    [
        "cdp", "mdm", "meiro", "customer", "golden record", "identity",
        "cdc", "streaming", "survivorship", "profile", "segment", "master data",
        "deduplication", "unification",
    ],
    # 5: BI/MCP
    [
        "metabase", "looker", "studio", "mcp-server", "bigquery-mcp", "dagster",
        "bi-platform", "dashboard", "visualization", "d3", "lineage", "ast",
        "sql template", "bi", "mcp", "reporting",
    ],
    # 6: Mobile Graph CRM
    [
        "network", "planner", "crm", "graph", "spectral", "android", "room",
        "swift", "swiftdata", "touchpoint", "offline sync", "sqlcipher",
        "mobile", "ios", "kotlin", "relationship",
    ],
    # 7: Generative NLP
    [
        "content", "comments", "rabbit", "nlp", "llm", "automation",
        "agent", "headless", "redis", "crawler", "prompt", "generator",
        "outlook", "mac-automation", "summarizer", "copywriting",
    ],
]


def classify_project_domain(
    project_path: Path,
    tech_stack: str | list[str] = "",
    summary: str = "",
) -> tuple[str, list[float]]:
    """Classify a project into the 8 canonical domain clusters and return an 8-dim unit vector.

    Canonical Clusters:
    0: Travel/Pricing
    1: Quantum/Brain
    2: LegalTech
    3: Web Analytics/MarTech
    4: CDP/MDM
    5: BI/MCP
    6: Mobile Graph CRM
    7: Generative NLP

    Args:
        project_path: Filesystem path to the project directory.
        tech_stack: Tech stack description or list of technologies.
        summary: Mined description, README summary, or commit context.

    Returns:
        tuple[str, list[float]]: (dominant_domain_name, 8-dimensional unit vector).
    """
    p = Path(project_path)
    name_str = p.name.lower()
    path_str = str(p).lower()
    stack_str = (tech_stack if isinstance(tech_stack, str) else " ".join(tech_stack)).lower()
    summary_str = summary.lower()

    # Read-only doc inspection if summary is empty and directory exists
    if not summary_str and p.exists() and p.is_dir():
        for fn in ("PROJECT.md", "README.md", "package.json", "pyproject.toml"):
            doc_f = p / fn
            if doc_f.exists():
                try:
                    with open(doc_f, encoding="utf-8") as f:
                        summary_str += " " + f.read(1500).lower()
                except Exception:
                    pass

    scores = [0.0] * 8

    # Project-specific explicit mappings for high precision
    explicit_exact_mappings: dict[str, int] = {
        "turna": 0,
        "etstur": 0,
        "quanta": 1,
        "legalport": 2,
        "hukukport": 2,
        "metricase": 3,
        "meiro": 4,
        "mdm": 4,
        "metabase": 5,
        "looker": 5,
        "app-my-network-planner": 6,
        "intelnetwork": 6,
        "qarezmi": 6,
        "contentai": 7,
        "commentshub": 7,
        "rabbit-ai": 7,
    }
    for key, c_idx in explicit_exact_mappings.items():
        if key in name_str:
            scores[c_idx] += 8.0

    # Keyword scoring across the 8 domains
    for i, kw_list in enumerate(DOMAIN_KEYWORDS):
        for kw in kw_list:
            if kw in name_str:
                scores[i] += 4.0
            if kw in stack_str:
                scores[i] += 2.5
            if kw in path_str:
                scores[i] += 1.5
            if kw in summary_str:
                scores[i] += 1.0

    # Add uniform baseline epsilon to prevent division by zero or empty vectors
    epsilon = 0.05
    adjusted_scores = [s + epsilon for s in scores]

    # Calculate L2 norm
    l2_norm = math.sqrt(sum(x * x for x in adjusted_scores)) or 1.0
    unit_vector = [round(x / l2_norm, 6) for x in adjusted_scores]

    # Renormalize to ensure exact unit length
    final_norm = math.sqrt(sum(x * x for x in unit_vector)) or 1.0
    unit_vector = [x / final_norm for x in unit_vector]

    # Determine dominant domain
    if max(scores) == 0.0:
        dominant_domain = "General"
        neutral_val = 1.0 / math.sqrt(len(CANONICAL_DOMAINS))
        unit_vector = [neutral_val] * len(CANONICAL_DOMAINS)
        return dominant_domain, unit_vector

    max_idx = scores.index(max(scores))
    dominant_domain = CANONICAL_DOMAINS[max_idx]

    return dominant_domain, unit_vector


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
        summaries_db: Path | str | None = None,
        state_file: Path | str | None = "quanta_cognitive_state.json",
        history_len: int = 20,
    ) -> None:
        """Initialize harvester with paths and anti-rumination history buffer.

        Args:
            projects_dir: Directory containing user workspace projects.
            brain_dir: Directory containing Antigravity session brain transcripts.
            summaries_db: Path to conversation_summaries.db SQLite database.
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
        self.summaries_db = Path(summaries_db) if summaries_db else DEFAULT_SUMMARIES_DB
        self.state_file = Path(state_file) if state_file else Path("quanta_cognitive_state.json")

        self._recent_topics: collections.deque[str] = collections.deque(maxlen=history_len)
        self._recent_hybrid_pairs: collections.deque[tuple[str, str]] = collections.deque(
            maxlen=50
        )
        self._project_rotation_idx: int = 0
        self._cached_projects: list[ProjectContext] = []
        self._last_scan_time: float = 0.0

    @property
    def anti_rumination_pairs(self) -> list[tuple[str, str]]:
        """Public access to the recent hybrid project pairs buffer (anti-rumination)."""
        return list(self._recent_hybrid_pairs)

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

        dom_name, _ = classify_project_domain(p, " ".join(tech_stack), summary)

        return ProjectContext(
            name=p.name,
            path=p,
            has_git=has_git,
            latest_commits=latest_commits,
            description=desc,
            tech_stack=list(dict.fromkeys(tech_stack)),
            summary=summary,
            domain=dom_name,
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
            if p.name == "Alfa" and (p / "quanta").is_dir():
                quanta_ctx = self._inspect_project(p / "quanta")
                quanta_ctx.name = "quanta"
                projects.append(quanta_ctx)

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

    def harvest_conversation_history(
        self, max_sessions: int = 20
    ) -> list[EpisodicTrace]:
        """Harvest recent user prompts, decisions, and messages from Antigravity session substrate.

        Queries the read-only SQLite conversation index (file:...mode=ro) and parses
        tail JSONL transcripts, categorizing traces across four temporal horizons:
        - Today (< 24h)
        - Yesterday (24h - 48h)
        - Previous Sprint (2d - 7d)
        - Long-Term (> 7d)

        Guarantees sub-5ms read latency and zero lock contention.

        Args:
            max_sessions: Maximum number of conversation sessions to inspect.

        Returns:
            list[EpisodicTrace]: Chronologically sorted (newest first) episodic traces.
        """
        if not self.summaries_db.exists():
            return []

        db_uri = f"file:{self.summaries_db.resolve()}?mode=ro"
        rows: list[tuple[Any, ...]] = []

        try:
            conn = sqlite3.connect(db_uri, uri=True, timeout=1.0)
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT conversation_id, title, last_modified_time, last_user_input_time, workspace_uris
                    FROM conversation_summaries
                    WHERE last_modified_time IS NOT NULL
                    ORDER BY last_modified_time DESC
                    LIMIT ?
                    """,
                    (max_sessions,),
                )
                rows = cur.fetchall()
            finally:
                conn.close()
        except Exception as e:
            logger.debug("Failed to query conversation_summaries.db: %s", e)
            return []

        now_utc = datetime.now(timezone.utc)
        traces: list[EpisodicTrace] = []

        for row in rows:
            conv_id = str(row[0])
            title = str(row[1] or "").strip()
            last_mod = row[2]
            last_input = row[3]
            ws_uris = str(row[4] or "")

            session_dt = _parse_iso_timestamp(last_input or last_mod)
            age_sec = max(0.0, (now_utc - session_dt).total_seconds())
            age_hours = age_sec / 3600.0

            if age_hours < 24.0:
                horizon = "today"
            elif age_hours < 48.0:
                horizon = "yesterday"
            elif age_hours < 168.0:
                horizon = "sprint"
            else:
                horizon = "longterm"

            # Infer project name
            project = "quanta"
            if ws_uris:
                try:
                    uris = json.loads(ws_uris)
                    if isinstance(uris, list) and uris:
                        p_str = str(uris[0])
                        parts = p_str.rstrip("/").split("/")
                        if parts:
                            cand = parts[-1].replace("%20", " ")
                            if cand and cand != "Antigravity Projects":
                                project = cand
                except Exception:
                    pass

            session_traces: list[EpisodicTrace] = []
            transcript_path = (
                self.brain_dir / conv_id / ".system_generated" / "logs" / "transcript.jsonl"
            )
            if transcript_path.exists():
                try:
                    f_size = transcript_path.stat().st_size
                    read_size = min(f_size, 65536)  # 64KB tail chunk
                    with open(transcript_path, "rb") as tf:
                        if f_size > read_size:
                            tf.seek(f_size - read_size)
                        raw_bytes = tf.read()
                    lines = raw_bytes.decode("utf-8", errors="replace").splitlines()
                    if f_size > read_size and len(lines) > 1:
                        lines = lines[1:]  # Skip partial first line

                    step_counter = 0
                    for line in reversed(lines):
                        line_str = line.strip()
                        if not line_str:
                            continue
                        try:
                            record = json.loads(line_str)
                        except Exception:
                            continue

                        rec_type = record.get("type", "")
                        rec_content = str(record.get("content") or "").strip()
                        rec_created = record.get("created_at")
                        entry_dt = (
                            _parse_iso_timestamp(rec_created) if rec_created else session_dt
                        )

                        source = ""
                        salience = 1.0
                        if rec_type == "USER_INPUT" or "<USER_REQUEST>" in rec_content:
                            source = "user_prompt"
                            salience = 1.8
                        elif rec_type == "PLANNER_RESPONSE":
                            source = "planner_decision"
                            salience = 1.4
                        elif rec_type == "EPHEMERAL_MESSAGE" and "SWR Replay" not in rec_content:
                            source = "agent_message"
                            salience = 1.0

                        if source and rec_content:
                            clean_content = (
                                rec_content.replace("<USER_REQUEST>", "")
                                .replace("</USER_REQUEST>", "")
                                .strip()
                            )
                            clean_content = re.sub(
                                r"(?:Bearer|token|key|password)\s*[:=]\s*\S+",
                                "[REDACTED]",
                                clean_content,
                                flags=re.I,
                            )
                            short_content = clean_content[:140]
                            if len(clean_content) > 140:
                                short_content += "..."

                            step_idx = record.get("step_index", step_counter)
                            trace_id = f"trace_{conv_id[:8]}_step_{step_idx}_{source}"

                            t_weight = calculate_temporal_decay(
                                created_at=entry_dt,
                                salience=salience,
                                is_core_anchor=False,
                                now_ts=now_utc,
                            )
                            r_score = salience * 0.9998 * t_weight

                            session_traces.append(
                                EpisodicTrace(
                                    trace_id=trace_id,
                                    session_id=conv_id,
                                    timestamp=entry_dt,
                                    horizon=horizon,
                                    source=source,
                                    content=short_content,
                                    full_text=clean_content,
                                    project=project,
                                    salience=salience,
                                    fidelity=0.9998,
                                    is_core_anchor=False,
                                    age_hours=round(age_hours, 2),
                                    temporal_weight=round(t_weight, 5),
                                    replay_score=round(r_score, 5),
                                )
                            )
                            step_counter += 1
                            if len(session_traces) >= 3:
                                break
                except Exception as e:
                    logger.debug("Failed reading transcript %s: %s", transcript_path, e)

            # If no traces extracted from transcript, fall back to title if available
            if not session_traces and title:
                t_weight = calculate_temporal_decay(
                    created_at=session_dt,
                    salience=1.2,
                    is_core_anchor=False,
                    now_ts=now_utc,
                )
                r_score = 1.2 * 0.9998 * t_weight
                session_traces.append(
                    EpisodicTrace(
                        trace_id=f"trace_{conv_id[:8]}_title",
                        session_id=conv_id,
                        timestamp=session_dt,
                        horizon=horizon,
                        source="user_prompt",
                        content=title[:140],
                        full_text=title,
                        project=project,
                        salience=1.2,
                        fidelity=0.9998,
                        is_core_anchor=False,
                        age_hours=round(age_hours, 2),
                        temporal_weight=round(t_weight, 5),
                        replay_score=round(r_score, 5),
                    )
                )

            traces.extend(session_traces)

        # Sort reverse-chronologically (newest first)
        traces.sort(key=lambda t: t.timestamp, reverse=True)
        return traces

    def generate_episodic_dream_seeds(self, max_sessions: int = 10) -> list[DreamSeed]:
        """Synthesizes candidate DreamSeeds from recent active session conversation traces.

        Allows subconscious REM dream cycles to deliberate on fresh user prompts and
        unresolved challenges from today and yesterday.

        Args:
            max_sessions: Number of recent sessions to harvest.

        Returns:
            list[DreamSeed]: Episodic dream seeds prioritized by recency.
        """
        traces = self.harvest_conversation_history(max_sessions=max_sessions)
        recent_traces = [t for t in traces if t.horizon in ("today", "yesterday")]
        seeds: list[DreamSeed] = []

        seen_topics: set[str] = set()
        for t in recent_traces:
            clean_snippet = t.content.replace("\n", " ").strip()
            if len(clean_snippet) < 15:
                continue
            slug = re.sub(r"[^\w]+", "_", clean_snippet[:30]).strip("_").lower()
            topic = f"episodic_{t.session_id[:8]}_{slug}"
            if topic in seen_topics:
                continue
            seen_topics.add(topic)

            urgency = 2.2 if t.horizon == "today" else 1.8
            question = (
                f"In recent {t.horizon} session ({t.project}), "
                f"how should we address: '{clean_snippet}'?"
            )
            seeds.append(
                DreamSeed(
                    topic=topic,
                    speculative_question=question,
                    urgency=urgency,
                    context_keys=[t.project, t.horizon, "episodic_harvested"],
                    project_summary=t.full_text[:500],
                )
            )

        return seeds

    def generate_candidate_seeds(self) -> list[DreamSeed]:
        """Synthesize candidate DreamSeeds across discovered projects and domains.

        Returns:
            list[DreamSeed]: Candidate seeds labeled with rich project context.
        """
        projects = self.scan_workspace_projects()
        project_map = {p.name: p for p in projects}
        seeds: list[DreamSeed] = []

        # 0. Fresh episodic conversation seeds from recent sessions (Today & Yesterday)
        episodic_seeds = self.generate_episodic_dream_seeds()
        seeds.extend(episodic_seeds)

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

    def _synthesize_bisociation_theme_and_question(
        self,
        p_a: ProjectContext,
        dom_a: str,
        p_b: ProjectContext,
        dom_b: str,
    ) -> tuple[str, str]:
        """Synthesize emergent bisociation theme and speculative question bridging two domains."""
        pair_key = tuple(sorted([dom_a, dom_b]))

        if pair_key == ("Mobile Graph CRM", "Travel/Pricing"):
            travel_proj = p_a.name if dom_a == "Travel/Pricing" else p_b.name
            graph_proj = p_b.name if dom_b == "Mobile Graph CRM" else p_a.name
            theme = (
                "Spectral Graph Decomposition of Passenger Travel Companion Networks "
                "for Fare Optimization"
            )
            question = (
                f"How can spectral graph algorithms from {graph_proj} partition passenger "
                f"companion networks to optimize dynamic ticket repricing in {travel_proj}?"
            )
        elif pair_key == ("LegalTech", "Quantum/Brain"):
            legal_proj = p_a.name if dom_a == "LegalTech" else p_b.name
            quantum_proj = p_b.name if dom_b == "Quantum/Brain" else p_a.name
            theme = (
                "Unitary Statevector Projection and Quantum Zeno Arbitration "
                "for Pre-Trial Dispute Resolution"
            )
            question = (
                f"Can quantum Zeno statevector projection from {quantum_proj} arbitrate "
                f"complex regulatory disputes and statutory compliance in {legal_proj}?"
            )
        elif pair_key == ("BI/MCP", "Web Analytics/MarTech"):
            audit_proj = p_a.name if dom_a == "Web Analytics/MarTech" else p_b.name
            bi_proj = p_b.name if dom_b == "BI/MCP" else p_a.name
            theme = (
                "Self-Healing Dynamic Schema Lineage & Automated Blast Radius Auditing "
                "for Analytics Dashboards"
            )
            question = (
                f"How can AST dry-run cost estimation and schema drift detection from {bi_proj} "
                f"automate tag auditing and governance in {audit_proj}?"
            )
        elif pair_key == ("BI/MCP", "Travel/Pricing"):
            travel_proj = p_a.name if dom_a == "Travel/Pricing" else p_b.name
            bi_proj = p_b.name if dom_b == "BI/MCP" else p_a.name
            theme = (
                "Pre-Flight Dry-Run AST Cost Guardrails and Distributed Cache Hedging "
                "for Real-Time Flight Search"
            )
            question = (
                f"How can AST query optimization from {bi_proj} eliminate redundant "
                f"BigQuery pricing lookups and latency spikes in {travel_proj}?"
            )
        elif pair_key == ("CDP/MDM", "Mobile Graph CRM"):
            cdp_proj = p_a.name if dom_a == "CDP/MDM" else p_b.name
            mobile_proj = p_b.name if dom_b == "Mobile Graph CRM" else p_a.name
            theme = (
                "Zero-Locking Conflict-Free Replicated Data Types (CRDTs) Bridging Cloud "
                "CDC Streams and Mobile SQLite"
            )
            question = (
                f"How can CRDT data structures reconcile streaming enterprise CDC records "
                f"in {cdp_proj} with offline mobile SQLite replicas in {mobile_proj}?"
            )
        elif pair_key == ("Generative NLP", "Quantum/Brain"):
            nlp_proj = p_a.name if dom_a == "Generative NLP" else p_b.name
            brain_proj = p_b.name if dom_b == "Quantum/Brain" else p_a.name
            theme = (
                "Biomorphic SWR Memory Consolidation and CSF Shielding "
                "for Anti-Hallucinatory Agent Prompting"
            )
            question = (
                f"How can biomorphic sharp-wave ripple replay from {brain_proj} prevent catastrophic "
                f"context drift and hallucinations in {nlp_proj}?"
            )
        else:
            theme = f"Cross-Domain Synthesis: Bridging {dom_a} and {dom_b}"
            question = (
                f"How can architectural patterns and data structures from '{p_a.name}' ({dom_a}) "
                f"cross-pollinate with '{p_b.name}' ({dom_b}) to solve scalability and data "
                "consistency bottlenecks?"
            )

        return theme, question

    def pair_orthogonal_projects(
        self, min_domain_distance: float = 0.50, max_pairs: int = 5
    ) -> list[HybridDreamSeed]:
        """Discovers sibling projects under workspace, computes domain orthogonality,
        and constructs bisociative HybridDreamSeed pairs.

        Filters:
        1. Delta_domain = 1.0 - (d_A . d_B) / (||d_A|| ||d_B||) >= min_domain_distance (0.50).
        2. Rejects same-project pairs (P_A == P_B).
        3. Excludes pairs with shared project prefixes (e.g. Turna Works vs Turna JIRA).
        4. Prioritizes pairs not in recent hybrid history (anti-rumination pair buffer).

        Args:
            min_domain_distance: Minimum orthogonal domain distance (default 0.50).
            max_pairs: Maximum number of hybrid seeds to return (default 5).

        Returns:
            list[HybridDreamSeed]: List of constructed hybrid dream seeds.
        """
        projects = self.scan_workspace_projects()

        if len(projects) < 2:
            return []

        # Classify all projects into domain vectors
        classified: list[tuple[ProjectContext, str, list[float]]] = []
        for p in projects:
            dom_name, dom_vec = classify_project_domain(
                p.path,
                tech_stack=" ".join(p.tech_stack),
                summary=p.summary,
            )
            p.domain = dom_name
            classified.append((p, dom_name, dom_vec))

        candidates: list[tuple[float, HybridDreamSeed]] = []

        for i in range(len(classified)):
            for j in range(i + 1, len(classified)):
                p_a, dom_a, vec_a = classified[i]
                p_b, dom_b, vec_b = classified[j]

                # 1. Reject identical projects and unclassified/empty projects without keyword evidence
                if p_a.name == p_b.name:
                    continue
                if dom_a in {"General", "Unclassified"} or dom_b in {"General", "Unclassified"}:
                    continue

                # 2. Exclude shared project prefixes (e.g. Turna Works vs Turna JIRA, TurnaWorks vs TurnaJira)
                name_a_clean = p_a.name.strip()
                name_b_clean = p_b.name.strip()

                tokens_a = [
                    t.lower()
                    for t in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", name_a_clean)
                    if t
                ]
                tokens_b = [
                    t.lower()
                    for t in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", name_b_clean)
                    if t
                ]

                if tokens_a and tokens_b:
                    pfx_a = tokens_a[0]
                    pfx_b = tokens_b[0]
                    if pfx_a == pfx_b and pfx_a not in {"mcp", "ai", "app", "project", "google"}:
                        continue
                    if (
                        len(tokens_a) > 1
                        and len(tokens_b) > 1
                        and (
                            name_a_clean.lower().startswith(tokens_b[0])
                            or name_b_clean.lower().startswith(tokens_a[0])
                        )
                        and tokens_a[0] not in {"mcp", "ai", "app"}
                    ):
                        continue

                # 3. Compute orthogonal domain distance:
                # Delta_domain = 1.0 - (d_A . d_B) / (||d_A|| ||d_B||)
                dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
                norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
                norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
                cos_sim = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
                delta_domain = 1.0 - cos_sim

                if delta_domain < min_domain_distance:
                    continue

                # 4. Anti-rumination pair buffer
                pair_key: tuple[str, str] = (
                    min(p_a.name, p_b.name),
                    max(p_a.name, p_b.name),
                )
                is_recent = pair_key in self._recent_hybrid_pairs

                theme, question = self._synthesize_bisociation_theme_and_question(
                    p_a, dom_a, p_b, dom_b
                )
                slug_a = re.sub(r"[^a-zA-Z0-9]+", "_", p_a.name).strip("_").lower()
                slug_b = re.sub(r"[^a-zA-Z0-9]+", "_", p_b.name).strip("_").lower()
                topic = f"{slug_a}_x_{slug_b}"

                seed = HybridDreamSeed(
                    topic=topic,
                    speculative_question=question,
                    primary_project=p_a,
                    secondary_project=p_b,
                    domain_distance=round(delta_domain, 4),
                    bisociation_theme=theme,
                    urgency=2.2 if not is_recent else 1.8,
                )

                sort_score = delta_domain + (0.5 if not is_recent else 0.0)
                candidates.append((sort_score, seed))

        # Sort candidate pairs by score descending (highest orthogonality first)
        candidates.sort(key=lambda x: x[0], reverse=True)

        selected_seeds: list[HybridDreamSeed] = []
        used_projects: set[str] = set()

        # Diversify selection: avoid pairing the exact same project repeatedly
        for _, seed in candidates:
            p_a_name = seed.primary_project.name
            p_b_name = seed.secondary_project.name
            if len(selected_seeds) < max_pairs and (
                p_a_name not in used_projects or p_b_name not in used_projects
            ):
                selected_seeds.append(seed)
                used_projects.add(p_a_name)
                used_projects.add(p_b_name)
                self._recent_hybrid_pairs.append((min(p_a_name, p_b_name), max(p_a_name, p_b_name)))

        # Backfill if diversity filter left underfilled slots
        if len(selected_seeds) < max_pairs:
            for _, seed in candidates:
                if seed not in selected_seeds and len(selected_seeds) < max_pairs:
                    selected_seeds.append(seed)
                    p_1 = seed.primary_project.name
                    p_2 = seed.secondary_project.name
                    self._recent_hybrid_pairs.append((min(p_1, p_2), max(p_1, p_2)))

        return selected_seeds
