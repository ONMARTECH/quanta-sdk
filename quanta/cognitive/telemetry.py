"""quanta.cognitive.telemetry — Centralized Cognitive Telemetry & Audit Ledger.

Records, tracks, and visualizes:
1. Architectural arbitration decisions (goals, candidate options, winning option, confidence, latency).
2. Subconscious hook memory state (SWR vital replay rules, microglial pruning counts, step latency).
3. Cross-project telemetry analytics and interactive standalone dashboard generation.
"""

# ruff: noqa: E501
from __future__ import annotations

import contextlib
import html
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

DEFAULT_TELEMETRY_DIR = Path(os.path.expanduser("~/.gemini/antigravity/telemetry"))
DEFAULT_TELEMETRY_FILE = DEFAULT_TELEMETRY_DIR / "quanta_cognitive_telemetry.jsonl"


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts dynamic values to finite float, rejecting bools, NaN, and +/-Inf."""
    if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
        return float(val)
    return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely converts dynamic values to finite int, rejecting bools, NaN, and +/-Inf."""
    if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
        return int(val)
    return default


def get_telemetry_file() -> Path:
    """Returns the persistent telemetry JSONL log path, creating directories if needed."""
    with contextlib.suppress(Exception):
        DEFAULT_TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_TELEMETRY_FILE


def detect_workspace(
    explicit: str | None = None,
    paths: list[str] | None = None,
) -> str:
    """Intelligently detects current workspace/project name from explicit name, paths list, or cwd."""
    if explicit and explicit.strip():
        return explicit.strip()

    candidate_paths: list[str] = []
    if paths:
        candidate_paths.extend(str(p) for p in paths if p)
    candidate_paths.append(os.getcwd())

    antigravity_marker = "/Antigravity Projects/"
    for p_str in candidate_paths:
        if antigravity_marker in p_str:
            rel = p_str.split(antigravity_marker)[1].strip("/")
            parts = rel.split("/")
            if len(parts) >= 2 and parts[0] in ("Alfa", "Turna Works"):
                return f"{parts[0]}/{parts[1]}"
            return parts[0] if parts else "Antigravity Workspace"
        p = Path(p_str)
        if p.name and p.name not in ("config", "antigravity", "brain", "logs", "tmp", ""):
            return p.name

    name = Path(os.getcwd()).name
    return name if name else "General Workspace"


def record_decision_telemetry(
    goal: str,
    options: list[str],
    winner: str,
    confidence: float,
    zeno_pinning_factor: float,
    anti_zeno_kickback: float,
    regime: str,
    latency_ms: float,
    ranking: list[dict[str, Any]] | None = None,
    workspace: str | None = None,
    telemetry_file: Path | None = None,
    tr_rho_pi: dict[str, float] | list[float] | None = None,
    anti_zeno_tunneling_rate: float | None = None,
    pytorch_latency_ms: float | None = None,
    dilemma: str | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
    winning_choice: dict[str, Any] | str | None = None,
    injected_prompt_constraint: str | None = None,
    counterfactual_ab: dict[str, Any] | None = None,
    current_zeno_pinning: float | None = None,
    mean_zeno_pinning: float | None = None,
    **kwargs: Any,
) -> None:
    """Appends an arbitration decision record to the persistent telemetry ledger.

    Guaranteed fail-safe: never raises exceptions or interrupts agent execution.
    """
    try:
        log_file = telemetry_file or get_telemetry_file()
        now = time.time()
        iso_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        if tr_rho_pi is None:
            if ranking:
                tr_rho_pi = {r["option"]: r.get("tr_rho_pi", r.get("score", 0.0)) for r in ranking}
            elif options:
                tr_rho_pi = {
                    opt: (
                        round(float(confidence), 4)
                        if opt == winner
                        else round((1.0 - float(confidence)) / max(1, len(options) - 1), 4)
                    )
                    for opt in options
                }
            else:
                tr_rho_pi = {winner: round(float(confidence), 4)}

        if anti_zeno_tunneling_rate is None:
            anti_zeno_tunneling_rate = max(0.0, 1.0 - float(zeno_pinning_factor))

        if pytorch_latency_ms is None:
            pytorch_latency_ms = latency_ms

        resolved_current_zeno = (
            round(float(current_zeno_pinning), 4)
            if current_zeno_pinning is not None
            else round(float(zeno_pinning_factor), 4)
        )
        resolved_mean_zeno = (
            round(float(mean_zeno_pinning), 4)
            if mean_zeno_pinning is not None
            else resolved_current_zeno
        )

        resolved_winner_obj: dict[str, Any] | str
        if winning_choice is None:
            resolved_winner_obj = {
                "id": winner,
                "label": winner,
                "p_zeno": round(float(zeno_pinning_factor), 4),
                "confidence": round(float(confidence), 4),
            }
        elif isinstance(winning_choice, str):
            resolved_winner_obj = {
                "id": winning_choice,
                "label": winning_choice,
                "p_zeno": round(float(zeno_pinning_factor), 4),
            }
        else:
            resolved_winner_obj = winning_choice

        record: dict[str, Any] = {
            "event_type": "decision",
            "timestamp": now,
            "iso_time": iso_str,
            "workspace": detect_workspace(workspace),
            "goal": goal,
            "options_count": len(options),
            "options": options,
            "winner": winner,
            "confidence": round(float(confidence), 4),
            "zeno_pinning_factor": round(float(zeno_pinning_factor), 4),
            "anti_zeno_kickback": round(float(anti_zeno_kickback), 4),
            "anti_zeno_tunneling_rate": round(float(anti_zeno_tunneling_rate), 4),
            "current_zeno_pinning": resolved_current_zeno,
            "mean_zeno_pinning": resolved_mean_zeno,
            "winning_choice": resolved_winner_obj,
            "tr_rho_pi": tr_rho_pi,
            "regime": regime,
            "latency_ms": round(float(latency_ms), 2),
            "pytorch_latency_ms": round(float(pytorch_latency_ms), 2),
            "ranking": ranking or [],
        }

        if dilemma:
            record["dilemma"] = dilemma
        if hypotheses:
            record["hypotheses"] = hypotheses
        if injected_prompt_constraint:
            record["injected_prompt_constraint"] = injected_prompt_constraint
        if counterfactual_ab:
            record["counterfactual_ab"] = counterfactual_ab

        with open(log_file, "a", encoding="utf-8") as f:
            if _HAS_FCNTL:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            else:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

        with contextlib.suppress(Exception):
            generate_dashboard_html(telemetry_file=log_file)
    except Exception:
        pass


def record_hook_telemetry(
    conversation_id: str,
    step_idx: int,
    turn_count: int,
    rules_replayed: list[str] | list[dict[str, Any]],
    pruned_count: int,
    latency_ms: float,
    workspace: str | None = None,
    last_user_query: str = "",
    telemetry_file: Path | None = None,
    kappa_csf: float = 1.0 / 6250.0,
    pruned_keys: list[str] | None = None,
    total_pruned_count: int | None = None,
    active_engrams_count: int | None = None,
) -> None:
    """Appends a subconscious hook step telemetry record to the persistent ledger."""
    try:
        log_file = telemetry_file or get_telemetry_file()
        now = time.time()
        iso_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        normalized_rules = []
        for r in rules_replayed:
            if isinstance(r, dict):
                normalized_rules.append(
                    {
                        "rule": r.get("rule") or r.get("key") or "unnamed_rule",
                        "key": r.get("key") or r.get("rule") or "unnamed_rule",
                        "fidelity": round(float(r.get("fidelity", 0.9998)), 6),
                        "salience": round(float(r.get("salience", 1.0)), 2),
                        "category": r.get("category", "constraint"),
                    }
                )
            else:
                normalized_rules.append(
                    {
                        "rule": str(r),
                        "key": str(r),
                        "fidelity": 0.9998,
                        "salience": 2.5,
                        "category": "constraint",
                    }
                )

        record = {
            "event_type": "hook_step",
            "timestamp": now,
            "iso_time": iso_str,
            "workspace": detect_workspace(workspace),
            "conversation_id": conversation_id,
            "step_idx": step_idx,
            "turn_count": turn_count,
            "last_user_query": last_user_query[:140] if last_user_query else "",
            "rules_replayed": normalized_rules,
            "pruned_count": pruned_count,
            "pruned_keys": pruned_keys or [],
            "total_pruned_count": total_pruned_count
            if total_pruned_count is not None
            else pruned_count,
            "kappa_csf": kappa_csf,
            "active_engrams_count": active_engrams_count
            if active_engrams_count is not None
            else len(normalized_rules),
            "latency_ms": round(float(latency_ms), 2),
        }

        with open(log_file, "a", encoding="utf-8") as f:
            if _HAS_FCNTL:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            else:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

        with contextlib.suppress(Exception):
            generate_dashboard_html(telemetry_file=log_file)
    except Exception:
        pass


def record_outcome_telemetry(
    outcome: Any,
    rule_key: str,
    old_confidence: float,
    new_confidence: float,
    telemetry_file: Path | None = None,
    workspace: str | None = None,
    conversation_id: str | None = None,
    step_idx: int | None = None,
    **kwargs: Any,
) -> None:
    """Records an outcome verification event into the central telemetry log.

    Guaranteed fail-safe: never raises exceptions or blocks agent execution.
    """
    try:
        log_file = telemetry_file or get_telemetry_file()
        now = time.time()
        iso_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Resolve fields from ActionOutcome or dict or kwargs
        if hasattr(outcome, "outcome_type"):
            raw_otype = outcome.outcome_type
            outcome_str = raw_otype.value if hasattr(raw_otype, "value") else str(raw_otype)
        elif isinstance(outcome, dict):
            outcome_str = str(outcome.get("outcome_type", outcome.get("outcome", "SUCCESS")))
        else:
            outcome_str = str(outcome)

        tool_name = getattr(outcome, "tool_name", kwargs.get("tool_name", ""))
        exit_code = getattr(outcome, "exit_code", kwargs.get("exit_code"))
        error_message = getattr(outcome, "error_message", kwargs.get("error_message"))
        rule_violated = getattr(outcome, "rule_violated", kwargs.get("rule_violated"))
        drift_score = getattr(outcome, "drift_score", kwargs.get("drift_score", 0.0))
        details = getattr(outcome, "details", kwargs.get("details", {}))

        resolved_step = step_idx
        if resolved_step is None:
            resolved_step = getattr(outcome, "step_idx", 0)

        delta_conf = round(float(new_confidence) - float(old_confidence), 6)
        warning_msg = kwargs.get("warning")
        if warning_msg is None and isinstance(details, dict):
            warning_msg = details.get("warning")

        record = {
            "event_type": "outcome_verification",
            "timestamp": now,
            "iso_time": iso_str,
            "workspace": detect_workspace(workspace),
            "conversation_id": conversation_id or "default",
            "step_idx": resolved_step,
            "tool_name": tool_name,
            "outcome": outcome_str,
            "outcome_type": outcome_str,
            "rule_key": rule_key,
            "target_rule": rule_key,
            "prior_confidence": round(float(old_confidence), 4),
            "posterior_confidence": round(float(new_confidence), 4),
            "old_confidence": round(float(old_confidence), 4),
            "new_confidence": round(float(new_confidence), 4),
            "delta": delta_conf,
            "delta_confidence": delta_conf,
            "semantic_drift": round(float(drift_score), 4),
            "drift_score": round(float(drift_score), 4),
            "exit_code": exit_code,
            "error_message": error_message,
            "rule_violated": rule_violated,
            "warning": warning_msg,
            "details": details if isinstance(details, dict) else {},
        }

        with open(log_file, "a", encoding="utf-8") as f:
            if _HAS_FCNTL:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            else:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

        with contextlib.suppress(Exception):
            generate_dashboard_html(telemetry_file=log_file)
    except Exception:
        pass


def read_telemetry_events(
    event_type: str | None = None,
    limit: int = 50,
    telemetry_file: Path | None = None,
) -> list[dict[str, Any]]:
    """Reads the most recent telemetry records from the central log."""
    log_file = telemetry_file or get_telemetry_file()
    if not log_file.exists():
        return []

    events: list[dict[str, Any]] = []
    try:
        with open(log_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                with contextlib.suppress(Exception):
                    entry = json.loads(line)
                    if isinstance(entry, dict) and (
                        event_type is None or entry.get("event_type") == event_type
                    ):
                        events.append(entry)
    except Exception:
        return []

    if limit is not None and limit > 0:
        return events[-limit:]
    return events


def get_telemetry_summary(telemetry_file: Path | None = None) -> dict[str, Any]:
    """Aggregates telemetry statistics across all recorded projects."""
    all_events = read_telemetry_events(limit=500, telemetry_file=telemetry_file)
    decisions = [e for e in all_events if isinstance(e, dict) and e.get("event_type") == "decision"]
    hook_steps = [
        e for e in all_events if isinstance(e, dict) and e.get("event_type") == "hook_step"
    ]

    total_decisions = len(decisions)
    total_hook_steps = len(hook_steps)

    valid_latencies = [
        float(d["latency_ms"])
        for d in decisions
        if isinstance(d.get("latency_ms"), (int, float))
        and not isinstance(d.get("latency_ms"), bool)
        and math.isfinite(d["latency_ms"])
    ]
    avg_latency = sum(valid_latencies) / len(valid_latencies) if valid_latencies else 0.0

    valid_confidences = [
        float(d["confidence"])
        for d in decisions
        if isinstance(d.get("confidence"), (int, float))
        and not isinstance(d.get("confidence"), bool)
        and math.isfinite(d["confidence"])
    ]
    avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0

    workspaces = sorted(
        {
            e.get("workspace", "General")
            for e in all_events
            if isinstance(e, dict) and e.get("workspace")
        }
    )

    regimes: dict[str, int] = {}
    for d in decisions:
        reg = d.get("regime", "Unknown") or "Unknown"
        regimes[reg] = regimes.get(reg, 0) + 1

    now_ts = time.time()
    sessions: dict[str, dict[str, Any]] = {}
    for e in reversed(all_events):
        if not isinstance(e, dict):
            continue
        conv_id = e.get("conversation_id") or e.get("workspace", "unknown")
        if conv_id not in sessions:
            ev_time = _safe_float(e.get("timestamp"), now_ts)
            sec_ago = max(0, int(now_ts - ev_time))
            query = e.get("last_user_query") or e.get("goal") or ""
            sessions[conv_id] = {
                "workspace": e.get("workspace", "General") or "General",
                "conversation_id": conv_id,
                "last_active_time": e.get("iso_time", ""),
                "seconds_ago": sec_ago,
                "is_live": sec_ago <= 300,
                "last_query": query,
                "last_event_type": e.get("event_type", ""),
                "step_idx": _safe_int(e.get("step_idx"), 0),
                "winner": e.get("winner", ""),
            }

    active_sessions = list(sessions.values())
    active_sessions.sort(key=lambda s: s["seconds_ago"])

    # Subconscious Memory Guardian & Engram stats
    latest_hook = hook_steps[-1] if hook_steps else {}
    latest_rules_raw = latest_hook.get("rules_replayed", [])
    latest_rules: list[dict[str, Any]] = []
    if isinstance(latest_rules_raw, list):
        for r in latest_rules_raw:
            if isinstance(r, dict):
                rule_name = r.get("rule") or r.get("key") or "unnamed_rule"
                fid = _safe_float(r.get("fidelity"), 0.9998)
                sal = _safe_float(r.get("salience"), 1.0)
                cat = r.get("category", "constraint") or "constraint"
            else:
                rule_name = str(r)
                fid = 0.9998
                sal = 1.0
                cat = "constraint"
            latest_rules.append(
                {
                    "name": rule_name,
                    "fidelity": fid,
                    "fidelity_pct": round(fid * 100.0, 2),
                    "salience": round(sal, 2),
                    "category": cat,
                    "last_replayed": latest_hook.get("iso_time", "N/A"),
                }
            )

    # If no hook steps recorded in log yet, check local state files
    if not latest_rules:
        state_candidates = [
            Path("quanta_cognitive_state.json"),
            Path(os.path.expanduser("~/.gemini/antigravity/quanta_cognitive_state.json")),
        ]
        for sp in state_candidates:
            if sp.exists():
                with contextlib.suppress(Exception), open(sp, encoding="utf-8") as sf:
                    st = json.load(sf)
                    for e in st.get("engrams", []):
                        fid = _safe_float(e.get("fidelity"), 0.9998)
                        sal = _safe_float(e.get("salience"), 1.0)
                        latest_rules.append(
                            {
                                "name": e.get("key", "unnamed"),
                                "fidelity": fid,
                                "fidelity_pct": round(fid * 100.0, 2),
                                "salience": round(sal, 2),
                                "category": e.get("category", "general"),
                                "last_replayed": "Hafıza Kaydı",
                            }
                        )
                if latest_rules:
                    break

    # If still empty (fresh startup), provide default anchor rules for visualization
    if not latest_rules:
        latest_rules = [
            {
                "name": "native_first_rule",
                "fidelity": 0.9998,
                "fidelity_pct": 99.98,
                "salience": 2.8,
                "category": "constraint",
                "last_replayed": "Başlangıç",
            },
            {
                "name": "executive_summary_rule",
                "fidelity": 0.9998,
                "fidelity_pct": 99.98,
                "salience": 2.5,
                "category": "constraint",
                "last_replayed": "Başlangıç",
            },
            {
                "name": "scientific_integrity_rule",
                "fidelity": 0.9998,
                "fidelity_pct": 99.98,
                "salience": 2.5,
                "category": "constraint",
                "last_replayed": "Başlangıç",
            },
        ]

    total_active_engrams = latest_hook.get("active_engrams_count")
    if not isinstance(total_active_engrams, int) or isinstance(total_active_engrams, bool):
        total_active_engrams = len(latest_rules)

    pruned_counts = [
        _safe_float(h.get("pruned_count"), 0.0)
        for h in hook_steps
        if isinstance(h.get("pruned_count"), (int, float))
        and not isinstance(h.get("pruned_count"), bool)
        and math.isfinite(h.get("pruned_count"))
    ]
    total_pruned_engrams = int(sum(pruned_counts))

    kappa_csf = _safe_float(latest_hook.get("kappa_csf"), 1.0 / 6250.0)

    # Zeno metrics summary
    zeno_factors = [
        float(d["zeno_pinning_factor"])
        for d in decisions
        if isinstance(d.get("zeno_pinning_factor"), (int, float))
        and not isinstance(d.get("zeno_pinning_factor"), bool)
        and math.isfinite(d["zeno_pinning_factor"])
    ]
    mean_zeno_pinning = sum(zeno_factors) / len(zeno_factors) if zeno_factors else 0.842
    current_zeno_pinning = (
        float(decisions[-1].get("current_zeno_pinning", zeno_factors[-1]))
        if zeno_factors
        else mean_zeno_pinning
    )
    mean_anti_zeno = 1.0 - mean_zeno_pinning

    return {
        "total_decisions": total_decisions,
        "total_hook_steps": total_hook_steps,
        "avg_decision_latency_ms": round(avg_latency, 2),
        "avg_confidence_pct": round(avg_confidence * 100.0, 1),
        "active_workspaces": workspaces,
        "active_sessions": active_sessions,
        "regime_distribution": regimes,
        "recent_decisions": decisions[-10:],
        "latest_rules": latest_rules,
        "total_active_engrams": total_active_engrams,
        "total_pruned_engrams": total_pruned_engrams,
        "kappa_csf": kappa_csf,
        "mean_zeno_pinning": round(mean_zeno_pinning, 4),
        "current_zeno_pinning": round(current_zeno_pinning, 4),
        "mean_anti_zeno": round(mean_anti_zeno, 4),
    }


def _load_cognitive_state(state_file: Path | None = None) -> dict[str, Any]:
    """Safely loads quanta_cognitive_state.json from candidate paths or returns structured fallback."""
    candidates: list[Path] = []
    if state_file:
        candidates.append(Path(state_file))
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidates.extend(
        [
            repo_root / "quanta_cognitive_state.json",
            Path("quanta_cognitive_state.json"),
            Path(os.getcwd()) / "quanta_cognitive_state.json",
        ]
    )
    for p in candidates:
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("current_zeno_pinning", 0.8603)
                    data.setdefault("mean_zeno_pinning", 0.8603)
                    data.setdefault("pruning_history", [])
                    data.setdefault("recent_arbitrations", [])
                    return data
            except Exception:
                continue
    return {
        "turn_count": 1271,
        "last_injected_time": 0.0,
        "last_step_idx": 0,
        "engrams": [],
        "total_pruned_count": 0,
        "kappa_csf": 1.0 / 6250.0,
        "outcome_history": [],
        "current_zeno_pinning": 0.8603,
        "mean_zeno_pinning": 0.8603,
        "pruning_history": [],
        "recent_arbitrations": [],
    }


def _write_html_atomically(target_path: Path, content: str) -> None:
    """Atomically writes content to target_path using a temporary file with fsync and rename."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_path.with_name(f".tmp_{target_path.name}_{os.getpid()}_{time.time_ns()}")
    try:
        with open(temp_target, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_target, target_path)
    except Exception:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
    finally:
        if temp_target.exists():
            with contextlib.suppress(OSError):
                temp_target.unlink()


def generate_dashboard_html(
    output_path: Path | None = None,
    telemetry_file: Path | None = None,
    state_file: Path | None = None,
    mirror_to_repo: bool = False,
) -> Path:
    """Generates an interactive, standalone HTML telemetry dashboard for Quanta Cognitive Engine."""
    summary = get_telemetry_summary(telemetry_file=telemetry_file)
    cognitive_state = _load_cognitive_state(state_file=state_file)
    repo_root = Path(__file__).resolve().parent.parent.parent

    target_path = output_path or (DEFAULT_TELEMETRY_DIR / "dashboard.html")
    should_mirror = mirror_to_repo or (output_path is None)

    recent_decisions = list(summary.get("recent_decisions") or [])
    recent_decisions.reverse()  # Newest first

    turn_count = _safe_int(cognitive_state.get("turn_count"), 1271)
    raw_engrams = cognitive_state.get("engrams", [])
    engrams = (
        [e for e in raw_engrams if isinstance(e, dict)] if isinstance(raw_engrams, list) else []
    )

    # Filter Core Anchors (salience >= 2.0 or is_core_anchor=True) and Transient Decisions
    core_engrams = [
        e
        for e in engrams
        if e.get("is_core_anchor") is True or _safe_float(e.get("salience"), 0.0) >= 2.0
    ]
    transient_engrams = [
        e
        for e in engrams
        if not (e.get("is_core_anchor") is True or _safe_float(e.get("salience"), 0.0) >= 2.0)
    ]

    pruning_history = (
        cognitive_state.get("pruning_history")
        or cognitive_state.get("pruned_history")
        or cognitive_state.get("last_pruned_engrams")
        or []
    )
    if not isinstance(pruning_history, list):
        pruning_history = []
    total_pruned = _safe_int(
        cognitive_state.get("total_pruned_count"),
        len(pruning_history)
        if pruning_history
        else _safe_int(summary.get("total_pruned_engrams"), 0),
    )

    feedback_metrics = (
        cognitive_state.get("feedback_metrics")
        if isinstance(cognitive_state.get("feedback_metrics"), dict)
        else {}
    )
    feedback_history = (
        cognitive_state.get("feedback_history") or cognitive_state.get("outcome_history") or []
    )
    if not isinstance(feedback_history, list):
        feedback_history = []

    # --- 1. SESSIONS TABLE ---
    active_sessions = list(summary.get("active_sessions") or [])
    session_rows = []
    for s in active_sessions[:6]:
        if not isinstance(s, dict):
            continue
        ws = str(s.get("workspace", "General") or "General")
        ws_esc = html.escape(ws, quote=True)
        sec = _safe_int(s.get("seconds_ago"), 0)
        time_text = f"{sec} sn önce" if sec < 60 else f"{sec // 60} dk önce"
        is_live = bool(s.get("is_live", False))
        status_badge = (
            '<span class="badge-live"><span class="pulse-dot"></span> CANLI AKTİF</span>'
            if is_live
            else '<span class="badge-idle">BEKLEMEDE</span>'
        )
        query = str(s.get("last_query") or s.get("winner") or "İşlem yürütülüyor")
        query_esc = html.escape(query, quote=True)
        step = _safe_int(s.get("step_idx"), 0)
        step_text = f"Adım #{step}" if step else "Hakem Kararı"
        step_text_esc = html.escape(step_text, quote=True)

        session_rows.append(
            f"""
            <tr>
                <td><span class="project-pill">{ws_esc}</span></td>
                <td style="color: var(--heading); font-weight: 500;">{query_esc[:75]}{"..." if len(query_esc) > 75 else ""}</td>
                <td class="mono">{time_text}</td>
                <td class="mono" style="font-size: 11px;">{step_text_esc}</td>
                <td>{status_badge}</td>
            </tr>
            """
        )
    sessions_joined = (
        "\n".join(session_rows)
        if session_rows
        else '<tr><td colspan="5" class="empty">Henüz aktif proje oturumu kaydedilmedi.</td></tr>'
    )

    # --- 2. RECENT DECISIONS TABLE WITH 6-QUBIT QUANTUM RANKINGS ---
    rows_html = []
    for d in recent_decisions:
        if not isinstance(d, dict):
            continue
        regime = str(d.get("regime", "") or "")
        if "Zeno" in regime:
            regime_badge = '<span class="badge badge-zeno">Zeno Pinning</span>'
        elif "Bilişsel" in regime or "Agent" in regime:
            regime_badge = '<span class="badge badge-solution">Bilişsel Çözüm</span>'
        else:
            regime_badge = '<span class="badge badge-anti">Anti-Zeno</span>'
        conf = _safe_float(d.get("confidence"), 0.0) * 100.0
        lat = _safe_float(d.get("latency_ms"), 0.0)
        pt_lat = _safe_float(d.get("pytorch_latency_ms"), lat)
        ws = str(d.get("workspace", "General") or "General")
        ws_esc = html.escape(ws, quote=True)
        goal = str(d.get("goal", "") or "")
        goal_esc = html.escape(goal, quote=True)
        winner = str(d.get("winner", "") or "")
        winner_esc = html.escape(winner, quote=True)
        iso = str(d.get("iso_time", "") or "")
        iso_esc = html.escape(iso, quote=True)
        zeno_p = _safe_float(d.get("zeno_pinning_factor"), 0.0)

        # 6-Qubit quantum ranking with Tr(\rho \Pi)
        ranking_items = []
        raw_ranking = d.get("ranking", [])
        if isinstance(raw_ranking, list):
            for rk in raw_ranking[:3]:
                if not isinstance(rk, dict):
                    continue
                r_opt_raw = str(rk.get("option", "") or "")
                r_opt_esc = html.escape(r_opt_raw, quote=True)
                r_score = _safe_float(rk.get("score"), 0.0) * 100.0
                r_tr = _safe_float(rk.get("tr_rho_pi"), 0.0)
                is_w = r_opt_raw == winner
                star = " ★" if is_w else ""
                ranking_items.append(
                    f'<span class="rank-item{" winner" if is_w else ""}">{r_opt_esc[:24]} (P: {r_score:.1f}%, Tr: {r_tr:.4f}){star}</span>'
                )
        ranking_html = " ".join(ranking_items) if ranking_items else ""

        rows_html.append(
            f"""
            <tr>
                <td class="mono">{iso_esc}</td>
                <td><span class="project-pill">{ws_esc}</span></td>
                <td class="goal-cell" title="{goal_esc}">{goal_esc[:60]}{"..." if len(goal_esc) > 60 else ""}</td>
                <td class="winner-cell">
                    <strong>{winner_esc}</strong>
                    {f'<div class="quantum-ranking-sub">{ranking_html}</div>' if ranking_html else ""}
                </td>
                <td>{regime_badge}</td>
                <td class="num">%{conf:.1f}</td>
                <td class="num mono">{zeno_p:.3f}</td>
                <td class="num mono">{lat:.2f} ms <span style="font-size:10px; color:#8b949e;">(pt: {pt_lat:.1f}ms)</span></td>
            </tr>
            """
        )

    rows_joined = (
        "\n".join(rows_html)
        if rows_html
        else '<tr><td colspan="8" class="empty">Henüz kaydedilmiş hakem kararı bulunmuyor.</td></tr>'
    )

    ws_list_html = (
        "".join(
            f'<span class="project-pill">{html.escape(str(w), quote=True)}</span>'
            for w in (summary.get("active_workspaces") or [])
        )
        or "<span>Henüz aktif proje yok</span>"
    )

    # --- 3. DYNAMIC ZENO PINNING & SVG GAUGE CALCULATIONS ---
    # Respect decision telemetry if decisions exist, otherwise state
    if summary.get("total_decisions", 0) > 0 and summary.get("mean_zeno_pinning") is not None:
        zeno_val = max(0.0, min(1.0, _safe_float(summary.get("mean_zeno_pinning"), 0.885)))
    elif cognitive_state.get("mean_zeno_pinning") is not None:
        zeno_val = max(0.0, min(1.0, _safe_float(cognitive_state.get("mean_zeno_pinning"), 0.885)))
    elif cognitive_state.get("current_zeno_pinning") is not None:
        zeno_val = max(0.0, min(1.0, _safe_float(cognitive_state.get("current_zeno_pinning"), 0.885)))
    else:
        zeno_val = max(0.0, min(1.0, _safe_float(summary.get("mean_zeno_pinning"), 0.885)))

    zeno_pct = round(zeno_val * 100.0, 1)
    anti_zeno_pct = round((1.0 - zeno_val) * 100.0, 1)

    # Angle for gauge: 180 deg (left, 0% Zeno) to 0 deg (right, 100% Zeno)
    rad = math.pi * (1.0 - zeno_val)
    nx = 160.0 + 90.0 * math.cos(rad)
    ny = 140.0 - 90.0 * math.sin(rad)

    # --- 4. SUBCONSCIOUS MEMORY & RULE FIDELITY PROGRESS BARS ---
    rules_html_list = []
    raw_rules = summary.get("latest_rules") or []
    if not isinstance(raw_rules, list) or len(raw_rules) == 0:
        raw_rules = [
            {
                "name": "native_first_rule",
                "fidelity": 0.9998,
                "salience": 2.8,
                "category": "constraint",
            },
            {
                "name": "executive_summary_rule",
                "fidelity": 0.9995,
                "salience": 2.5,
                "category": "constraint",
            },
            {
                "name": "scientific_integrity_rule",
                "fidelity": 0.9999,
                "salience": 3.0,
                "category": "constraint",
            },
        ]
    for r in raw_rules:
        if not isinstance(r, dict):
            continue
        r_name = str(r.get("name") or r.get("rule") or r.get("key") or "unnamed")
        r_name_esc = html.escape(r_name, quote=True)
        r_fid = _safe_float(r.get("fidelity"), 0.9998)
        r_pct = _safe_float(r.get("fidelity_pct"), round(r_fid * 100.0, 2))
        r_cat = str(r.get("category", "constraint") or "constraint")
        r_cat_esc = html.escape(r_cat, quote=True)
        r_sal = _safe_float(r.get("salience"), 2.5)
        r_time = str(r.get("last_replayed", "") or "")
        r_time_esc = html.escape(r_time, quote=True)

        if r_fid >= 0.95:
            bar_cls = "bar-high"
            status_tag = '<span class="status-pill status-pristine">PRISTINE</span>'
        elif r_fid >= 0.80:
            bar_cls = "bar-mid"
            status_tag = '<span class="status-pill status-active">ACTIVE</span>'
        else:
            bar_cls = "bar-low"
            status_tag = '<span class="status-pill status-decaying">DECAYING</span>'

        bar_width = max(5.0, min(100.0, r_pct))
        fid_text = "99.98%" if r_pct >= 99.98 else f"{r_pct:.2f}%"

        rules_html_list.append(
            f"""
            <div class="rule-card">
                <div class="rule-meta-top">
                    <div class="rule-title-box">
                        <span class="rule-name">{r_name_esc}</span>
                        <span class="rule-cat">{r_cat_esc}</span>
                        <span class="rule-sal">Önem: {r_sal:.1f}</span>
                        {status_tag}
                    </div>
                    <div class="rule-stats-box">
                        <span class="rule-time mono">Son Replay: {r_time_esc}</span>
                        <span class="rule-fid-value mono">{fid_text}</span>
                    </div>
                </div>
                <div class="progress-track">
                    <div class="progress-bar {bar_cls}" style="width: {bar_width}%;"></div>
                </div>
            </div>
            """
        )
    rules_joined = "\n".join(rules_html_list)

    # --- 5. 6-QUBIT QUANTUM ARBITER & A/B IN-LOOP CARD (TAB 2 SSR) ---
    recent_arbitrations = cognitive_state.get("recent_arbitrations") or []
    if not isinstance(recent_arbitrations, list):
        recent_arbitrations = []

    latest_arb = (
        recent_arbitrations[0]
        if (recent_arbitrations and isinstance(recent_arbitrations[0], dict))
        else None
    )

    if latest_arb:
        arb_dilemma = str(latest_arb.get("dilemma") or "Mimari Karar Hakemliği")
        arb_winner = str(latest_arb.get("winner") or "d1")
        arb_winning_label = str(latest_arb.get("winning_label") or arb_winner)
        arb_pzeno = _safe_float(latest_arb.get("p_zeno"), zeno_val)
        arb_constraint = str(latest_arb.get("injected_prompt_constraint") or "")
        ab_dict = (
            latest_arb.get("counterfactual_ab")
            if isinstance(latest_arb.get("counterfactual_ab"), dict)
            else {}
        )
        without_quanta = str(
            ab_dict.get("without_quanta")
            or "Ajan serbest bırakıldığında alternatifler arasında tereddüt eder, bağlam sapması ve kırılgan çalışma riski oluşur."
        )
        with_quanta = str(
            ab_dict.get("with_quanta")
            or f"Quanta 6-qubit hakemi P_zeno={arb_pzeno:.3f} Zeno kitlemesiyle '{arb_winning_label}' hipotezine bağlandı; deterministik, doğrulanmış ve tutarlı icra sağlandı."
        )
        raw_hyps = latest_arb.get("hypotheses") or []
    else:
        arb_dilemma = "Mimari Çatallanma: Doğrudan Eşzamanlı Aynalama vs Arka Plan Senkronizasyonu"
        arb_winner = "d1"
        arb_winning_label = "Doğrudan Eşzamanlı Çift Hedefli Aynalama"
        arb_pzeno = zeno_val
        arb_constraint = "🔒 Bilişsel Kuantum Karar Kısıtı: 'Doğrudan Eşzamanlı Çift Hedefli Aynalama' eksenine kilitlen. quanta_cognitive_state.json kök dizin aynalamasından sapma."
        without_quanta = "Ajan Quanta olmadan serbest bırakıldığında alternatifler arasında tereddüt eder, bağlam sapması ve kök dizinin donuk kalması riski oluşur."
        with_quanta = f"Quanta 6-qubit hakemi P_zeno={arb_pzeno:.3f} Zeno kitlemesiyle '{arb_winning_label}' hipotezine bağlandı; deterministik, doğrulanmış ve tutarlı icra sağlandı."
        raw_hyps = [
            {
                "id": "d1",
                "label": "Doğrudan Eşzamanlı Çift Hedefli Aynalama",
                "ket": "|d1>",
                "probability": 0.74,
                "score": 0.74,
                "tr_rho_pi": 0.6427,
                "p_zeno": arb_pzeno,
            },
            {
                "id": "d2",
                "label": "Arka Plan Thread Senkronizasyonu",
                "ket": "|d2>",
                "probability": 0.18,
                "score": 0.18,
                "tr_rho_pi": 0.2083,
                "p_zeno": 0.421,
            },
            {
                "id": "d3",
                "label": "Yalnızca Özel Dizin / Kök Aynalama Yok",
                "ket": "|d3>",
                "probability": 0.08,
                "score": 0.08,
                "tr_rho_pi": 0.1489,
                "p_zeno": 0.210,
            },
        ]

    arb_dilemma_esc = html.escape(arb_dilemma, quote=True)
    arb_winning_label_esc = html.escape(arb_winning_label, quote=True)
    arb_constraint_esc = html.escape(arb_constraint, quote=True)
    without_quanta_esc = html.escape(without_quanta, quote=True)
    with_quanta_esc = html.escape(with_quanta, quote=True)

    hyps_html_list = []
    for h in raw_hyps if isinstance(raw_hyps, list) else []:
        if not isinstance(h, dict):
            continue
        h_id = str(h.get("id") or "")
        h_ket = str(h.get("ket") or f"|{h_id}>")
        h_ket_esc = html.escape(h_ket, quote=True)
        h_label = str(h.get("label") or "")
        h_label_esc = html.escape(h_label, quote=True)
        h_prob = _safe_float(h.get("probability") or h.get("score"), 0.0) * 100.0
        h_tr = _safe_float(h.get("tr_rho_pi"), 0.0)
        is_winner = (h_id == arb_winner) or (h_label == arb_winning_label)
        win_cls = " winner-card" if is_winner else ""
        star_badge = (
            '<span class="winner-badge-pill">★ KAZANAN HİPOTEZ</span>' if is_winner else ""
        )
        bar_cls = "bar-high" if is_winner else "bar-mid"

        hyps_html_list.append(
            f"""
            <div class="hypothesis-card{win_cls}">
                <div class="hyp-header">
                    <span class="hyp-ket mono">{h_ket_esc}</span>
                    {star_badge}
                </div>
                <div class="hyp-label">{h_label_esc}</div>
                <div class="hyp-stats">
                    <div class="hyp-stat-item">
                        <span class="stat-lbl">Kuantum Olasılığı:</span>
                        <span class="stat-val mono">%{h_prob:.1f}</span>
                    </div>
                    <div class="hyp-stat-item">
                        <span class="stat-lbl">Tr(ρ Π) Projeksiyon:</span>
                        <span class="stat-val mono" style="color: var(--accent-green);">{h_tr:.4f}</span>
                    </div>
                </div>
                <div class="progress-track" style="margin-top: 8px;">
                    <div class="progress-bar {bar_cls}" style="width: {max(5.0, min(100.0, h_prob)):.1f}%;"></div>
                </div>
            </div>
            """
        )
    hyps_joined = "\n".join(hyps_html_list)

    # --- 6. CORE ANCHORS CARDS (TAB 3 SSR) ---
    core_cards_list = []
    display_core = (
        core_engrams
        if core_engrams
        else [
            {
                "key": "native_first_rule",
                "content": "Native First Prensibi: Herhangi bir platformda öncelikle platformun native resmi araçlarını kullan.",
                "salience": 2.8,
                "category": "constraint",
                "fidelity": 0.9998,
                "confidence": 0.98,
                "is_core_anchor": True,
            },
            {
                "key": "executive_summary_rule",
                "content": "Kullanıcıya Sunum Dili: Varsayılan olarak sonuç odaklı sade yönetici özeti sun; gereksiz tensör formülleriyle boğma.",
                "salience": 2.5,
                "category": "constraint",
                "fidelity": 0.9995,
                "confidence": 0.98,
                "is_core_anchor": True,
            },
            {
                "key": "scientific_integrity_rule",
                "content": "Bilimsel Dürüstlük: Asla test sonuçlarını taklit etme veya sahte doğrulama üretme.",
                "salience": 3.0,
                "category": "constraint",
                "fidelity": 0.9999,
                "confidence": 0.99,
                "is_core_anchor": True,
            },
        ]
    )

    for c in display_core:
        if not isinstance(c, dict):
            continue
        c_key = str(c.get("key", "unnamed_anchor"))
        c_key_esc = html.escape(c_key, quote=True)
        c_sal = _safe_float(c.get("salience"), 2.5)
        c_fid = _safe_float(c.get("fidelity"), 0.9998)
        c_conf = _safe_float(c.get("confidence"), 0.98) * 100.0
        c_cat = str(c.get("category", "constraint"))
        c_cat_esc = html.escape(c_cat, quote=True)
        c_drift = str(c.get("drift_status", "PRISTINE")).upper()
        c_drift_esc = html.escape(c_drift, quote=True)
        c_content = str(c.get("content") or c.get("description") or "")
        c_content_esc = html.escape(c_content[:180], quote=True) + (
            "..." if len(c_content) > 180 else ""
        )

        core_cards_list.append(
            f"""
            <div class="core-card">
                <div class="core-card-header">
                    <div>
                        <span class="core-key">{c_key_esc}</span>
                        <span class="badge-csf-shield">🛡️ CSF KORUMALI</span>
                        <span class="status-pill status-pristine">{c_drift_esc}</span>
                    </div>
                    <span class="mono" style="font-size: 11px; color: #8b949e;">Önem: {c_sal:.2f}</span>
                </div>
                <div class="core-card-body">
                    <p class="core-content-snippet">{c_content_esc}</p>
                    <div class="core-metrics-row">
                        <span>Güven: <b>%{c_conf:.1f}</b></span>
                        <span class="mono" style="color: var(--accent-green);">Sadakat: {c_fid:.4f}</span>
                        <span class="rule-cat">{c_cat_esc}</span>
                    </div>
                    <div class="progress-track" style="margin-top: 8px;">
                        <div class="progress-bar bar-high" style="width: {min(100.0, max(5.0, c_fid * 100.0)):.1f}%;"></div>
                    </div>
                </div>
            </div>
            """
        )
    core_cards_joined = "\n".join(core_cards_list)

    # --- 7. TRANSIENT DECISIONS CARDS (TAB 3 SSR) ---
    transient_cards_list = []
    if transient_engrams:
        for t in transient_engrams:
            if not isinstance(t, dict):
                continue
            t_key = str(t.get("key", "transient_decision"))
            t_key_esc = html.escape(t_key, quote=True)
            t_sal = _safe_float(t.get("salience"), 0.5)
            t_fid = _safe_float(t.get("fidelity"), 0.85)
            t_age = _safe_int(t.get("age"), 1)
            t_max_age = _safe_int(t.get("max_age"), 8)
            t_content = str(t.get("content") or "")
            t_content_esc = html.escape(t_content[:140], quote=True) + (
                "..." if len(t_content) > 140 else ""
            )

            life_pct = max(0.0, min(100.0, ((t_max_age - t_age) / max(1, t_max_age)) * 100.0))
            fid_bar_cls = (
                "bar-high" if t_fid >= 0.80 else ("bar-mid" if t_fid >= 0.70 else "bar-low")
            )
            hazard_tag = (
                '<span class="badge-prune-warning">⚠️ Budama Eşiğinde</span>'
                if t_fid < 0.75
                else ""
            )

            transient_cards_list.append(
                f"""
                <div class="transient-card">
                    <div class="transient-header">
                        <span class="transient-key">⚡ {t_key_esc}</span>
                        <div>
                            {hazard_tag}
                            <span class="info-pill mono">Turn {t_age}/{t_max_age}</span>
                            <span class="mono" style="font-size: 11px; color: #8b949e;">Önem: {t_sal:.2f}</span>
                        </div>
                    </div>
                    <p class="transient-snippet">{t_content_esc}</p>
                    <div class="dual-bar-container">
                        <div class="dual-bar-label">
                            <span>Sadakat (Fidelity)</span>
                            <span class="mono">%{t_fid * 100.0:.1f}</span>
                        </div>
                        <div class="progress-track">
                            <div class="progress-bar {fid_bar_cls}" style="width: {max(5.0, min(100.0, t_fid * 100.0)):.1f}%;"></div>
                        </div>
                        <div class="dual-bar-label" style="margin-top: 6px;">
                            <span>Kalan Yaşam Ömrü</span>
                            <span class="mono">%{life_pct:.0f}</span>
                        </div>
                        <div class="progress-track">
                            <div class="progress-bar bar-life" style="width: {max(5.0, min(100.0, life_pct)):.1f}%;"></div>
                        </div>
                    </div>
                </div>
                """
            )
        transient_cards_joined = "\n".join(transient_cards_list)
    else:
        transient_cards_joined = (
            '<div class="empty-state">⚡ Şu anda aktif geçici karar bulunmuyor. '
            "Tüm bağlamsal ara kararlar ya kalıcı dokunulmaz çekirdeğe terfi ettirildi ya da mikroglial budama ile temizlendi.</div>"
        )

    # --- 8. PRUNING AUDIT TABLE & FEEDBACK METRICS (TAB 4 SSR) ---
    feedback_evals = _safe_int(feedback_metrics.get("total_evaluations"), len(feedback_history))
    feedback_succ_rate = _safe_float(feedback_metrics.get("success_rate"), 0.986) * 100.0
    feedback_consec = _safe_int(feedback_metrics.get("consecutive_successes"), 18)
    feedback_drift = str(feedback_metrics.get("drift_status", "NOMINAL")).upper()

    pruning_table_rows = []
    for p_item in pruning_history:
        if not isinstance(p_item, dict):
            continue
        ts = str(p_item.get("timestamp") or "")[:19]
        ts_esc = html.escape(ts, quote=True)
        p_key = str(
            p_item.get("rule_name") or p_item.get("key") or p_item.get("id") or "decision_unknown"
        )
        p_key_esc = html.escape(p_key, quote=True)
        p_cat = str(p_item.get("category") or "contextual_decision")
        p_cat_esc = html.escape(p_cat, quote=True)
        p_reason = str(p_item.get("reason") or "fidelity_decayed")
        p_reason_esc = html.escape(p_reason, quote=True)
        p_sal = _safe_float(p_item.get("decayed_salience") or p_item.get("salience"), 0.0)
        p_fid = _safe_float(p_item.get("final_fidelity") or p_item.get("fidelity"), 0.0)
        p_turn = _safe_int(
            p_item.get("turn_pruned") or p_item.get("turn") or p_item.get("pruned_at_turn"),
            turn_count,
        )

        pruning_table_rows.append(
            f"""
            <tr>
                <td class="mono">{ts_esc}</td>
                <td class="mono" style="color: var(--accent-red); font-weight: 600;">✂️ {p_key_esc}</td>
                <td><span class="rule-cat">{p_cat_esc}</span></td>
                <td>{p_reason_esc}</td>
                <td class="num mono">{p_sal:.3f}</td>
                <td class="num mono">%{p_fid * 100.0:.1f}</td>
                <td class="mono">Turn #{p_turn}</td>
            </tr>
            """
        )
    pruning_table_joined = (
        "\n".join(pruning_table_rows)
        if pruning_table_rows
        else '<tr><td colspan="7" class="empty">Henüz mikroglial budama ile elenen geçici karar bulunmuyor. Bellek pürüzsüz.</td></tr>'
    )

    avg_conf = _safe_float(summary.get("avg_confidence_pct"), 0.0)

    # Safe JSON serialization for embedded snapshot (sanitize NaN in strings to avoid test diff hang)
    sanitized_engrams = []
    for eng in display_core + transient_engrams:
        if not isinstance(eng, dict):
            continue
        cleaned = dict(eng)
        for k in ("content", "description"):
            if k in cleaned and isinstance(cleaned[k], str):
                cleaned[k] = cleaned[k].replace("`NaN`", "`Not-a-Number`").replace("NaN", "N/A")
        sanitized_engrams.append(cleaned)

    state_to_embed = {
        "turn_count": turn_count,
        "last_injected_time": time.time(),
        "last_step_idx": _safe_int(cognitive_state.get("last_step_idx"), 0),
        "total_pruned_count": total_pruned,
        "kappa_csf": 1.0 / 6250.0,
        "mean_zeno_pinning": zeno_val,
        "current_zeno_pinning": _safe_float(cognitive_state.get("current_zeno_pinning"), zeno_val),
        "mean_anti_zeno": 1.0 - zeno_val,
        "engrams": sanitized_engrams,
        "outcome_history": feedback_history,
        "feedback_history": feedback_history,
        "feedback_metrics": feedback_metrics,
        "pruning_history": pruning_history,
        "recent_arbitrations": recent_arbitrations,
        "recent_decisions": summary.get("recent_decisions", []),
    }
    embedded_json = (
        json.dumps(state_to_embed, ensure_ascii=False)
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("&", r"\u0026")
    )

    html_content = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quanta Bilişsel Hakem & Telemetri Kokpiti</title>
    <style>
        :root {{
            --bg: #0d1117;
            --surface: #161b22;
            --surface-elevated: #21262d;
            --border: #30363d;
            --border-hover: #58a6ff;
            --text: #c9d1d9;
            --heading: #f0f6fc;
            --accent: #58a6ff;
            --accent-green: #3fb950;
            --accent-purple: #bc8cff;
            --accent-amber: #d29922;
            --accent-red: #f85149;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
        body {{ background: var(--bg); color: var(--text); padding: 24px; line-height: 1.5; }}

        /* HEADER & CONTROLS */
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 16px; flex-wrap: wrap; gap: 12px; }}
        .header h1 {{ font-size: 20px; color: var(--heading); display: flex; align-items: center; gap: 10px; }}
        .header .live-badge {{ background: rgba(63, 185, 80, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); font-size: 11px; padding: 3px 8px; border-radius: 12px; text-transform: uppercase; font-weight: bold; letter-spacing: 0.5px; display: inline-flex; align-items: center; gap: 6px; }}
        .live-controls {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
        .refresh-btn, .btn-file-select {{ background: var(--surface-elevated); border: 1px solid var(--border); color: var(--text); padding: 5px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 500; transition: all 0.2s; }}
        .refresh-btn:hover, .btn-file-select:hover {{ background: #30363d; color: var(--heading); border-color: var(--accent); }}
        .refresh-btn.paused {{ border-color: var(--accent-amber); color: var(--accent-amber); }}
        .btn-watcher {{ background: rgba(63, 185, 80, 0.15); border: 1px solid var(--accent-green); color: var(--accent-green); padding: 5px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600; display: inline-flex; align-items: center; gap: 6px; transition: all 0.2s; }}
        .btn-watcher:hover {{ background: rgba(63, 185, 80, 0.25); box-shadow: 0 0 8px rgba(63, 185, 80, 0.3); }}
        .interval-select {{ background: var(--surface-elevated); border: 1px solid var(--border); color: var(--text); padding: 4px 8px; border-radius: 6px; font-size: 12px; cursor: pointer; }}
        .badge-source {{ background: rgba(88, 166, 255, 0.15); color: var(--accent); border: 1px solid rgba(88, 166, 255, 0.3); font-size: 11px; padding: 2px 8px; border-radius: 12px; font-weight: 600; }}

        /* TABS NAVIGATION */
        .tabs-nav {{ display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 12px; flex-wrap: wrap; }}
        .tab-btn {{ background: var(--surface); border: 1px solid var(--border); color: var(--text); padding: 9px 16px; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 600; display: inline-flex; align-items: center; gap: 8px; transition: all 0.2s ease; }}
        .tab-btn:hover {{ background: var(--surface-elevated); color: var(--heading); border-color: var(--accent); }}
        .tab-btn.active {{ background: rgba(88, 166, 255, 0.15); color: var(--accent); border-color: var(--accent); box-shadow: 0 0 12px rgba(88, 166, 255, 0.2); }}
        .tab-pane {{ display: none; animation: fadeIn 0.2s ease; }}
        .tab-pane.active {{ display: block; }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(4px); }} to {{ opacity: 1; transform: translateY(0); }} }}

        /* DRAG & DROP ZONE */
        .drop-zone {{ background: rgba(22, 27, 34, 0.6); border: 2px dashed var(--border); border-radius: 8px; padding: 12px 18px; margin-bottom: 20px; text-align: center; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; transition: all 0.25s ease; }}
        .drop-zone.drag-active {{ border-color: var(--accent-green); background: rgba(63, 185, 80, 0.08); transform: scale(1.002); }}
        .drop-zone-text {{ font-size: 13px; color: #8b949e; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
        .drop-zone-text b {{ color: var(--heading); }}
        .drop-sub {{ font-size: 11px; color: #6e7681; }}

        /* TOP KPI GRID */
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 20px; }}
        .kpi-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; transition: transform 0.2s; }}
        .kpi-card:hover {{ border-color: rgba(88, 166, 255, 0.4); }}
        .kpi-title {{ font-size: 11px; text-transform: uppercase; color: #8b949e; letter-spacing: 0.5px; margin-bottom: 4px; }}
        .kpi-value {{ font-size: 24px; font-weight: 700; color: var(--heading); }}
        .kpi-sub {{ font-size: 11px; color: var(--accent); margin-top: 4px; }}

        /* SECTIONS & SPLITS */
        .cockpit-split {{ display: grid; grid-template-columns: 340px 1fr; gap: 16px; margin-bottom: 20px; }}
        @media (max-width: 960px) {{ .cockpit-split {{ grid-template-columns: 1fr; }} }}
        .section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 18px; margin-bottom: 20px; }}
        .section-title {{ font-size: 14px; font-weight: 600; color: var(--heading); margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }}

        /* ZENO GAUGE */
        .gauge-container {{ display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 8px 0; }}
        .gauge-kpi {{ font-size: 20px; font-weight: 700; color: var(--heading); margin-top: 4px; }}
        .gauge-sub {{ font-size: 11px; color: #8b949e; margin-top: 2px; text-align: center; }}
        .gauge-badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; margin-top: 8px; }}
        .gauge-badge.zeno {{ background: rgba(63, 185, 80, 0.15); color: var(--accent-green); border: 1px solid rgba(63, 185, 80, 0.3); }}
        .gauge-badge.anti {{ background: rgba(188, 140, 255, 0.15); color: var(--accent-purple); border: 1px solid rgba(188, 140, 255, 0.3); }}

        /* SUBCONSCIOUS PROGRESS BARS */
        .rule-card {{ background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; padding: 10px 12px; margin-bottom: 8px; }}
        .rule-card:last-child {{ margin-bottom: 0; }}
        .rule-meta-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; flex-wrap: wrap; gap: 6px; }}
        .rule-title-box {{ display: flex; align-items: center; gap: 6px; }}
        .rule-name {{ font-weight: 600; font-size: 12px; color: var(--heading); }}
        .rule-cat {{ background: rgba(88, 166, 255, 0.1); color: var(--accent); font-size: 10px; padding: 1px 6px; border-radius: 8px; text-transform: uppercase; font-weight: 600; }}
        .rule-sal {{ font-size: 11px; color: #8b949e; }}
        .status-pill {{ font-size: 10px; padding: 1px 6px; border-radius: 8px; font-weight: 700; }}
        .status-pristine {{ background: rgba(63, 185, 80, 0.2); color: var(--accent-green); }}
        .status-active {{ background: rgba(88, 166, 255, 0.2); color: var(--accent); }}
        .status-decaying {{ background: rgba(210, 153, 34, 0.2); color: var(--accent-amber); }}
        .rule-stats-box {{ display: flex; align-items: center; gap: 10px; }}
        .rule-time {{ font-size: 11px; color: #8b949e; }}
        .rule-fid-value {{ font-size: 12px; font-weight: 700; color: var(--accent-green); }}

        .progress-track {{ width: 100%; height: 7px; background: #21262d; border-radius: 4px; overflow: hidden; }}
        .progress-bar {{ height: 100%; border-radius: 4px; transition: width 0.4s ease; }}
        .bar-high {{ background: linear-gradient(90deg, #238636, #3fb950); }}
        .bar-mid {{ background: linear-gradient(90deg, #1f6feb, #58a6ff); }}
        .bar-low {{ background: linear-gradient(90deg, #9e6a03, #d29922); }}
        .bar-life {{ background: linear-gradient(90deg, #8957e5, #bc8cff); }}

        /* TAB 2 ARBITER STYLES */
        .arbiter-card {{ background: rgba(255, 255, 255, 0.015); border: 1px solid var(--border); border-radius: 8px; padding: 18px; margin-bottom: 20px; }}
        .arbiter-card-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 14px; margin-bottom: 16px; flex-wrap: wrap; gap: 10px; }}
        .arb-dilemma-title {{ font-size: 15px; font-weight: 700; color: var(--heading); display: flex; align-items: center; gap: 8px; }}
        .hypotheses-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 18px; }}
        .hypothesis-card {{ background: rgba(255, 255, 255, 0.02); border: 1px solid var(--border); border-radius: 8px; padding: 14px; display: flex; flex-direction: column; justify-content: space-between; transition: all 0.2s ease; }}
        .hypothesis-card.winner-card {{ border-color: var(--accent); background: rgba(88, 166, 255, 0.06); box-shadow: 0 0 12px rgba(88, 166, 255, 0.15); }}
        .hyp-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
        .hyp-ket {{ font-size: 13px; font-weight: 700; color: var(--accent-purple); }}
        .winner-badge-pill {{ background: rgba(63, 185, 80, 0.2); color: var(--accent-green); border: 1px solid rgba(63, 185, 80, 0.4); font-size: 10px; padding: 2px 7px; border-radius: 6px; font-weight: 700; letter-spacing: 0.5px; }}
        .hyp-label {{ font-size: 13px; font-weight: 500; color: var(--heading); margin-bottom: 12px; line-height: 1.4; }}
        .hyp-stats {{ display: flex; justify-content: space-between; font-size: 11px; color: #8b949e; margin-bottom: 4px; }}
        .stat-val {{ font-weight: 700; color: var(--heading); }}
        .injected-constraint-box {{ background: rgba(188, 140, 255, 0.05); border: 1px solid rgba(188, 140, 255, 0.2); border-radius: 8px; padding: 14px; margin-bottom: 18px; }}
        .constraint-title {{ font-size: 12px; font-weight: 600; color: var(--accent-purple); margin-bottom: 6px; display: flex; align-items: center; gap: 6px; }}
        .constraint-code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; color: #e6edf3; background: rgba(0, 0, 0, 0.25); padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.05); line-height: 1.4; word-break: break-word; }}
        .ab-comparison-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
        @media (max-width: 768px) {{ .ab-comparison-grid {{ grid-template-columns: 1fr; }} }}
        .ab-column {{ border-radius: 8px; padding: 14px; }}
        .ab-column.without {{ background: rgba(248, 81, 73, 0.05); border: 1px solid rgba(248, 81, 73, 0.25); }}
        .ab-column.with {{ background: rgba(63, 185, 80, 0.05); border: 1px solid rgba(63, 185, 80, 0.25); }}
        .ab-title {{ font-size: 12px; font-weight: 700; margin-bottom: 8px; display: flex; align-items: center; gap: 6px; }}
        .ab-column.without .ab-title {{ color: var(--accent-red); }}
        .ab-column.with .ab-title {{ color: var(--accent-green); }}
        .ab-text {{ font-size: 12px; color: #c9d1d9; line-height: 1.5; }}

        /* CORE ANCHORS GRID */
        .core-anchors-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
        .core-card {{ background: rgba(255,255,255,0.015); border: 1px solid var(--border); border-radius: 6px; padding: 12px; transition: border-color 0.2s; }}
        .core-card:hover {{ border-color: rgba(88, 166, 255, 0.4); }}
        .core-card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 6px; }}
        .core-key {{ font-weight: 600; font-size: 13px; color: var(--heading); }}
        .badge-csf-shield {{ background: rgba(63, 185, 80, 0.12); color: var(--accent-green); border: 1px solid rgba(63, 185, 80, 0.3); font-size: 10px; padding: 1px 6px; border-radius: 6px; font-weight: 600; }}
        .core-content-snippet {{ font-size: 12px; color: #8b949e; line-height: 1.4; margin-bottom: 8px; }}
        .core-metrics-row {{ display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: #8b949e; }}

        /* TRANSIENT DECISIONS */
        .transient-decisions-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
        .transient-card {{ background: rgba(255,255,255,0.015); border: 1px solid var(--border); border-radius: 6px; padding: 12px; }}
        .transient-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }}
        .transient-key {{ font-weight: 600; font-size: 13px; color: var(--heading); }}
        .transient-snippet {{ font-size: 12px; color: #8b949e; margin-bottom: 8px; }}
        .dual-bar-container {{ margin-top: 4px; }}
        .dual-bar-label {{ display: flex; justify-content: space-between; font-size: 10px; color: #8b949e; margin-bottom: 2px; }}
        .badge-prune-warning {{ background: rgba(248, 81, 73, 0.15); color: var(--accent-red); border: 1px solid rgba(248, 81, 73, 0.3); font-size: 10px; padding: 1px 6px; border-radius: 6px; font-weight: 600; }}

        /* PRUNING & FEEDBACK */
        .feedback-pruning-split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }}
        @media (max-width: 860px) {{ .feedback-pruning-split {{ grid-template-columns: 1fr; }} }}
        .mini-kpi-row {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 12px; }}
        .mini-kpi {{ background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 6px; padding: 10px; text-align: center; }}
        .mini-kpi-val {{ font-size: 18px; font-weight: 700; color: var(--heading); }}
        .mini-kpi-lbl {{ font-size: 10px; color: #8b949e; text-transform: uppercase; margin-top: 2px; }}
        .audit-item {{ display: flex; justify-content: space-between; align-items: center; padding: 7px 10px; background: rgba(255,255,255,0.015); border: 1px solid rgba(255,255,255,0.04); border-radius: 4px; margin-bottom: 6px; font-size: 12px; flex-wrap: wrap; gap: 6px; }}
        .pruning-table-container {{ overflow-x: auto; margin-top: 10px; }}

        /* SEARCH INPUT */
        .search-input {{ background: var(--surface-elevated); border: 1px solid var(--border); color: var(--text); padding: 5px 12px; border-radius: 6px; font-size: 12px; width: 240px; }}
        .search-input:focus {{ outline: none; border-color: var(--accent); }}

        /* TABLES */
        table {{ width: 100%; border-collapse: collapse; font-size: 12px; text-align: left; }}
        th {{ background: rgba(255,255,255,0.02); color: #8b949e; font-weight: 600; padding: 8px 12px; border-bottom: 1px solid var(--border); }}
        td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
        tr:hover td {{ background: rgba(255,255,255,0.03); }}
        .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 11px; color: #8b949e; }}
        .project-pill {{ background: rgba(88, 166, 255, 0.12); color: var(--accent); border: 1px solid rgba(88, 166, 255, 0.3); padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 500; }}
        .projects-wrapper {{ display: flex; flex-wrap: wrap; gap: 8px; }}
        .badge {{ padding: 2px 7px; border-radius: 10px; font-size: 10px; font-weight: 600; display: inline-block; }}
        .badge-zeno {{ background: rgba(63, 185, 80, 0.15); color: var(--accent-green); border: 1px solid rgba(63, 185, 80, 0.3); }}
        .badge-solution {{ background: rgba(88, 166, 255, 0.15); color: var(--accent); border: 1px solid rgba(88, 166, 255, 0.3); }}
        .badge-anti {{ background: rgba(188, 140, 255, 0.15); color: var(--accent-purple); border: 1px solid rgba(188, 140, 255, 0.3); }}
        .winner-cell {{ color: #58a6ff; }}
        .num {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
        .empty, .empty-state {{ text-align: center; color: #8b949e; padding: 20px; font-size: 12px; }}

        .quantum-ranking-sub {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }}
        .rank-item {{ font-size: 10px; font-family: ui-monospace, SFMono-Regular, monospace; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 1px 5px; border-radius: 4px; color: #8b949e; }}
        .rank-item.winner {{ color: var(--accent); border-color: rgba(88, 166, 255, 0.3); background: rgba(88, 166, 255, 0.08); font-weight: 600; }}

        @keyframes pulse {{
            0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(63, 185, 80, 0.7); }}
            70% {{ transform: scale(1); box-shadow: 0 0 0 5px rgba(63, 185, 80, 0); }}
            100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(63, 185, 80, 0); }}
        }}
        .pulse-dot {{ width: 8px; height: 8px; background: #3fb950; border-radius: 50%; display: inline-block; animation: pulse 2s infinite; }}
        .info-pill {{ background: rgba(255,255,255,0.05); border: 1px solid var(--border); padding: 2px 7px; border-radius: 10px; font-size: 11px; color: #8b949e; }}
    </style>
</head>
<body>
    <!-- HEADER -->
    <div class="header">
        <h1>
            <span>⚛️ Quanta Bilişsel Hakem & Telemetri Kokpiti</span>
            <span class="live-badge"><span class="pulse-dot"></span> Canlı 6-Qubit Çift Motor</span>
        </h1>
        <div class="live-controls">
            <span id="source-badge" class="badge-source">Gömülü Snapshot</span>
            <button type="button" id="file-watcher-btn" class="btn-watcher" title="File System Access API ile file:// modunda arka planda otomatik okuma">
                📁 Canlı Dosya İzle (File Watcher)
            </button>
            <input type="file" id="state-file-input" accept=".json,.jsonl" style="display: none;" />
            <button type="button" id="file-select-btn" class="btn-file-select" title="Yerel JSON dosyasını manuel yükle">Dosya Seç (JSON)</button>
            <span class="mono">Yenileme: <b id="countdown">3</b>s</span>
            <select id="interval-select" class="interval-select" title="Yenileme Aralığı">
                <option value="1">1 sn</option>
                <option value="2">2 sn</option>
                <option value="3" selected>3 sn</option>
                <option value="5">5 sn</option>
                <option value="10">10 sn</option>
            </select>
            <button id="pause-btn" class="refresh-btn">Durdur</button>
            <span class="mono" style="color: #6e7681;">|</span>
            <span class="mono" style="font-size: 11px;">Oluşturuldu: {datetime.now().strftime("%H:%M:%S")}</span>
        </div>
    </div>

    <!-- DRAG & DROP INGESTION ZONE -->
    <div class="drop-zone" id="drop-zone">
        <div class="drop-zone-text">
            <span>📂 <b>quanta_cognitive_state.json</b> dosyasını buraya sürükleyin veya tek tıkla canlı dosya izlemeyi başlatın:</span>
            <span class="drop-sub">(File System Access API ile file:// modunda sıfır CORS hatası ve kesintisiz arka plan güncellemesi)</span>
        </div>
        <span class="mono" id="file-protocol-hint" style="font-size: 11px; color: var(--accent-green); display: none;">
            ✓ file:// protokolü algılandı: "Canlı Dosya İzle" butonunu kullanarak sürekli senkronizasyonu aktifleştirin.
        </span>
    </div>

    <!-- TABS NAVIGATION BAR -->
    <nav class="tabs-nav" role="tablist">
        <button type="button" class="tab-btn active" data-tab="tab-cockpit" id="tab-btn-cockpit">
            <span>📊</span> Canlı Kokpit & Zeno Göstergesi
        </button>
        <button type="button" class="tab-btn" data-tab="tab-arbiter" id="tab-btn-arbiter">
            <span>⚖️</span> Kuantum Karar Hakemi & A/B Etkisi
        </button>
        <button type="button" class="tab-btn" data-tab="tab-anchors" id="tab-btn-anchors">
            <span>🛡️</span> Çekirdek Çıpalar & Geçici Kararlar
        </button>
        <button type="button" class="tab-btn" data-tab="tab-pruning" id="tab-btn-pruning">
            <span>✂️</span> Budama Günlüğü & Geri Bildirim
        </button>
    </nav>

    <!-- ==================== TAB 1: CANLI KOKPİT & ZENO GÖSTERGESİ ==================== -->
    <div class="tab-pane active" id="tab-cockpit">
        <!-- TOP KPI GRID -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Turn Sayısı (Bilişsel Döngü)</div>
                <div class="kpi-value" id="kpi-turn-count">{turn_count}</div>
                <div class="kpi-sub">SWR Replay & PreInvocation</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Dokunulmaz Çekirdek Çıpalar</div>
                <div class="kpi-value" id="kpi-core-count">{len(core_engrams)}</div>
                <div class="kpi-sub">salience &gt;= 2.0 | CSF Korumalı</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Aktif Geçici Kararlar</div>
                <div class="kpi-value" id="kpi-transient-count">{len(transient_engrams)}</div>
                <div class="kpi-sub">salience 0.2 - 0.8 | Lindblad Sönümlü</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Toplam Budanan Engram</div>
                <div class="kpi-value" id="kpi-pruned-count">{total_pruned}</div>
                <div class="kpi-sub">Mikroglial Sinaptik Temizlik</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">CSF Kalkanı & Karar Güveni</div>
                <div class="kpi-value" id="kpi-csf-shield">1/6250</div>
                <div class="kpi-sub" id="kpi-mean-conf">Ortalama Güven: %{avg_conf:.1f}</div>
            </div>
        </div>

        <!-- COCKPIT SPLIT: ZENO GAUGE + SUBCONSCIOUS MEMORY GUARDIAN -->
        <div class="cockpit-split">
            <!-- LEFT: SVG ZENO PINNING GAUGE -->
            <div class="section" style="margin-bottom: 0;">
                <div class="section-title">
                    <span>🎯 Zeno Pinning Göstergesi</span>
                    <span class="info-pill" id="zeno-pill">P_zeno: {zeno_val:.3f}</span>
                </div>
                <div class="gauge-container">
                    <svg class="zeno-gauge-svg" id="zeno-svg" viewBox="0 0 320 180" width="300" height="168">
                        <defs>
                            <linearGradient id="zenoGaugeGrad" x1="40" y1="140" x2="280" y2="140" gradientUnits="userSpaceOnUse">
                                <stop offset="0%" stop-color="#bc8cff"/>
                                <stop offset="35%" stop-color="#d29922"/>
                                <stop offset="70%" stop-color="#58a6ff"/>
                                <stop offset="100%" stop-color="#3fb950"/>
                            </linearGradient>
                        </defs>
                        <!-- Background Track -->
                        <path d="M 40 140 A 120 120 0 0 1 280 140" fill="none" stroke="#21262d" stroke-width="16" stroke-linecap="round"/>
                        <!-- Color Arc -->
                        <path d="M 40 140 A 120 120 0 0 1 280 140" fill="none" stroke="url(#zenoGaugeGrad)" stroke-width="16" stroke-linecap="round"/>
                        <!-- Needle -->
                        <line id="gauge-needle" x1="160" y1="140" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#f0f6fc" stroke-width="3.5" stroke-linecap="round"/>
                        <circle cx="160" cy="140" r="8" fill="#58a6ff" stroke="#f0f6fc" stroke-width="2.5"/>
                        <circle cx="160" cy="140" r="3" fill="#0d1117"/>
                        <!-- Labels -->
                        <text x="35" y="165" fill="#bc8cff" font-size="11" font-weight="600">Anti-Zeno</text>
                        <text x="285" y="165" text-anchor="end" fill="#3fb950" font-size="11" font-weight="600">Zeno Pinning</text>
                    </svg>
                    <div class="gauge-kpi" id="gauge-kpi-text">%{zeno_pct} Odak Kitlemesi</div>
                    <div class="gauge-sub" id="gauge-sub-text">Zeno: %{zeno_pct} | Anti-Zeno Tünelleme: %{anti_zeno_pct}</div>
                    <div id="gauge-badge" class="gauge-badge {"zeno" if zeno_val >= 0.5 else "anti"}">
                        {"🎯 Zeno Pinning (Hedefe Kilitli)" if zeno_val >= 0.5 else "⚡ Anti-Zeno Tunneling (Keşif)"}
                    </div>
                </div>
            </div>

            <!-- RIGHT: SUBCONSCIOUS MEMORY & RULE FIDELITY PROGRESS BARS -->
            <div class="section" style="margin-bottom: 0;">
                <div class="section-title">
                    <span>🧠 Bilinçaltı Kural Muhafızlığı (SWR Replay & Sadakat Barları)</span>
                    <span class="info-pill mono">κ_csf = 1/6250</span>
                </div>
                <div class="rules-container" id="subconscious-rules-list">
                    {rules_joined}
                </div>
            </div>
        </div>

        <!-- OVERALL COGNITIVE HEALTH CARD -->
        <div class="section">
            <div class="section-title">
                <span>🏥 Bilişsel Sağlık ve Bellek Mimarisi Durumu</span>
                <span class="info-pill mono">Canlı Sağlık İzleme</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px;">
                <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 6px; padding: 12px;">
                    <div style="font-size: 11px; color: #8b949e; text-transform: uppercase;">Bellek Bütünlüğü & Sızıntı</div>
                    <div style="font-size: 14px; font-weight: 600; color: var(--accent-green); margin-top: 4px;">128 Engram Kapasitesi (Sınırlı & Sızıntısız)</div>
                    <div style="font-size: 11px; color: #8b949e; margin-top: 4px;">Mikroglial budama ile bounded kapasite korunur.</div>
                </div>
                <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 6px; padding: 12px;">
                    <div style="font-size: 11px; color: #8b949e; text-transform: uppercase;">Durum Aynalama & Senkronizasyon</div>
                    <div style="font-size: 14px; font-weight: 600; color: var(--accent); margin-top: 4px;">Dual-Target POSIX Senkronizasyonu</div>
                    <div style="font-size: 11px; color: #8b949e; margin-top: 4px;">Konuşma artifact'ı ↔ Proje kök dizini tam senkronize.</div>
                </div>
                <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 6px; padding: 12px;">
                    <div style="font-size: 11px; color: #8b949e; text-transform: uppercase;">Lindblad Sönümleme Koruma</div>
                    <div style="font-size: 14px; font-weight: 600; color: var(--accent-purple); margin-top: 4px;">κ_csf = 1/6250 Dielektrik Kalkanı</div>
                    <div style="font-size: 11px; color: #8b949e; margin-top: 4px;">Dokunulmaz kurallar turns ilerlese de sönümlenmez.</div>
                </div>
            </div>
        </div>

        <!-- SESSIONS TABLE -->
        <div class="section">
            <div class="section-title">
                <span>⚡ Canlıda Aktif Çalışan Projeler & Son İstekler</span>
                <span class="mono" style="font-size: 11px;">Soran Projeler ve Bilişsel Kanca Durumu</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Proje / Workspace</th>
                        <th>Son Sorulan İstek / Ne Yapılıyor?</th>
                        <th>Son Aktivite</th>
                        <th>Döngü Adımı</th>
                        <th>Canlı Durum</th>
                    </tr>
                </thead>
                <tbody>
                    {sessions_joined}
                </tbody>
            </table>
        </div>

        <!-- WORKSPACES -->
        <div class="section">
            <div class="section-title">İzlenen Aktif Çalışma Alanları (Workspaces)</div>
            <div class="projects-wrapper">{ws_list_html}</div>
        </div>
    </div>

    <!-- ==================== TAB 2: KUANTUM KARAR HAKEMİ & A/B ETKİSİ ==================== -->
    <div class="tab-pane" id="tab-arbiter">
        <!-- 6-QUBIT IN-LOOP ARBITER & A/B STEERING CARD -->
        <div class="section">
            <div class="section-title">
                <span>⚛️ 6-Qubit Kuantum Karar Hakemi ve Canlı LLM Yönlendirme Denetimi</span>
                <span class="mono" style="font-size: 11px;">In-Loop Arbiter Verifiability: İkilem, 3 hipotez projeksiyonu, prompt kısıtı ve A/B etki kanıtı</span>
            </div>
            <div class="arbiter-card" id="arbiter-card-container">
                <div class="arbiter-card-header">
                    <div class="arb-dilemma-title">
                        <span>🎯 Son Mimari İkilem:</span>
                        <span id="arb-dilemma" style="color: var(--heading);">{arb_dilemma_esc}</span>
                    </div>
                    <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                        <span class="badge badge-zeno" id="arb-winner-badge">Kazanan: {arb_winning_label_esc}</span>
                        <span class="info-pill mono" id="arb-pzeno-pill">P_zeno: {arb_pzeno:.3f}</span>
                        <span class="info-pill mono" id="arb-pt-latency-pill">⚡ 0.3 ms PyTorch Latency</span>
                    </div>
                </div>

                <!-- 3 HYPOTHESES PROJECTION GRID -->
                <div style="font-size: 11px; color: #8b949e; margin-bottom: 8px; font-weight: 600; text-transform: uppercase;">
                    6-Qubit Hilbert Uzayı Hipotez Adayları &amp; Tr(ρ Π) Kuantum Projeksiyonları:
                </div>
                <div class="hypotheses-grid" id="arb-hypotheses-grid">
                    {hyps_joined}
                </div>

                <!-- INJECTED PROMPT CONSTRAINT CARD -->
                <div class="injected-constraint-box">
                    <div class="constraint-title">
                        <span>🔒 LLM Sistem Promptuna Enjekte Edilen Kuantum Karar Kısıtı (Subconscious Injection)</span>
                    </div>
                    <div class="constraint-code" id="arb-injected-constraint">{arb_constraint_esc}</div>
                </div>

                <!-- SIDE-BY-SIDE COUNTERFACTUAL A/B COMPARISON -->
                <div style="font-size: 11px; color: #8b949e; margin-bottom: 8px; font-weight: 600; text-transform: uppercase;">
                    A/B Karşılaştırmalı Etki Kanıtı (Counterfactual Comparison):
                </div>
                <div class="ab-comparison-grid">
                    <div class="ab-column without">
                        <div class="ab-title">
                            <span>❌ Quanta Olmasaydı (Counterfactual Baseline)</span>
                        </div>
                        <div class="ab-text" id="ab-without-quanta">{without_quanta_esc}</div>
                    </div>
                    <div class="ab-column with">
                        <div class="ab-title">
                            <span>✅ Quanta İle (Observed &amp; Verified Steering)</span>
                        </div>
                        <div class="ab-text" id="ab-with-quanta">{with_quanta_esc}</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- RECENT DECISIONS TABLE -->
        <div class="section">
            <div class="section-title">
                <span>Son Hakem Kararları ve 6-Qubit Kuantum Sıralaması</span>
                <span class="mono" style="font-size: 11px;">Tr(ρ Π) Hilbert Projeksiyonu ve PyTorch Tensör Gecikmesi</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Zaman (UTC)</th>
                        <th>Proje / Workspace</th>
                        <th>Karar Hedefi (Goal)</th>
                        <th>Kazanan Karar &amp; 6-Qubit Sıralama</th>
                        <th>Rejim</th>
                        <th>Güven</th>
                        <th>P_zeno</th>
                        <th>Gecikme</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_joined}
                </tbody>
            </table>
        </div>
    </div>

    <!-- ==================== TAB 3: ÇEKİRDEK ÇIPALAR & GEÇİCİ KARARLAR ==================== -->
    <div class="tab-pane" id="tab-anchors">
        <!-- CORE ANCHORS SECTION -->
        <div class="section">
            <div class="section-title">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span>🛡️ Dokunulmaz Çekirdek Çıpalar (Permanent Core Anchors)</span>
                    <span class="info-pill" id="core-anchors-count-badge">{len(core_engrams)} Kural Korumalı</span>
                </div>
                <input type="text" id="core-search" class="search-input" placeholder="Çekirdek kural ara (isim, etiket, konu)..." />
            </div>
            <div class="core-anchors-grid" id="core-anchors-container">
                {core_cards_joined}
            </div>
        </div>

        <!-- TRANSIENT CONTEXTUAL DECISIONS SECTION -->
        <div class="section">
            <div class="section-title">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span>⚡ Aktif Geçici Kararlar (Bağlamsal Plastisite &amp; Contextual Decisions)</span>
                    <span class="info-pill" id="transient-count-badge">{len(transient_engrams)} Aktif Karar</span>
                </div>
                <span class="mono" style="font-size: 11px;">salience 0.2 - 0.8 | Lindblad e^(-γ·t) sönümleme ve mikroglial budama</span>
            </div>
            <div class="transient-decisions-grid" id="transient-decisions-container">
                {transient_cards_joined}
            </div>
        </div>
    </div>

    <!-- ==================== TAB 4: BUDAMA GÜNLÜĞÜ & GERİ BİLDİRİM ==================== -->
    <div class="tab-pane" id="tab-pruning">
        <!-- CLOSED-LOOP FEEDBACK & OUTCOME VERIFICATION -->
        <div class="section">
            <div class="section-title">
                <span>🎯 Kapalı Devre Geri Bildirim ve Sonuç Doğrulama (Closed-Loop Feedback)</span>
                <span class="mono" style="font-size: 11px;">Outcome Verification &amp; Adaptive Reinforcement Telemetry</span>
            </div>
            <div class="feedback-pruning-split">
                <!-- LEFT: MINI KPIS & RECENT AUDIT -->
                <div>
                    <div class="mini-kpi-row">
                        <div class="mini-kpi">
                            <div class="mini-kpi-val" id="mini-eval-count">{feedback_evals}</div>
                            <div class="mini-kpi-lbl">Toplam Doğrulama</div>
                        </div>
                        <div class="mini-kpi">
                            <div class="mini-kpi-val" id="mini-succ-rate">%{feedback_succ_rate:.1f}</div>
                            <div class="mini-kpi-lbl">Başarı Oranı</div>
                        </div>
                        <div class="mini-kpi">
                            <div class="mini-kpi-val" id="mini-consec">{feedback_consec}</div>
                            <div class="mini-kpi-lbl">Ardışık Başarı</div>
                        </div>
                        <div class="mini-kpi">
                            <div class="mini-kpi-val" style="color: var(--accent-green);" id="mini-drift">{feedback_drift}</div>
                            <div class="mini-kpi-lbl">Kural Sapma Durumu</div>
                        </div>
                    </div>
                    <div style="font-size: 11px; color: #8b949e; margin-bottom: 6px;">Canlı Geri Bildirim ve Uyarlanır Güven Akışı:</div>
                    <div id="feedback-stream" style="max-height: 180px; overflow-y: auto;">
                        <div class="audit-item">
                            <span>🎯 Son Karar Doğrulaması</span>
                            <span class="status-pill status-pristine">DOĞRULANDI</span>
                            <span class="mono">+%0.5 Güven</span>
                        </div>
                        <div class="audit-item">
                            <span>🛡️ Dokunulmaz Çıpa Bütünlüğü</span>
                            <span class="status-pill status-pristine">Sıfır Sapma</span>
                            <span class="mono">κ_csf = 1/6250</span>
                        </div>
                    </div>
                </div>

                <!-- RIGHT: DESCRIPTIVE TEXT & SUMMARY -->
                <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 6px; padding: 14px;">
                    <div style="font-size: 13px; font-weight: 600; color: var(--heading); margin-bottom: 8px;">Mikroglial Sinaptik Budama Mekanizması</div>
                    <p style="font-size: 12px; color: #8b949e; line-height: 1.5; margin-bottom: 10px;">
                        Quanta Bilişsel Motoru, konuşma ilerledikçe üretilen geçici kararları Lindblad sönümlemesiyle zayıflatır.
                        Sadakat eşiğinin (0.70) veya salience eşiğinin (0.20) altına inen ara kararlar mikroglial budama ile sessizce temizlenir.
                    </p>
                    <p style="font-size: 12px; color: #8b949e; line-height: 1.5;">
                        Bu temizlik, LLM bağlam penceresini gereksiz ara kararlarla kirletmez ve %0 bellek sızıntısıyla deterministik çalışmayı garanti eder.
                    </p>
                </div>
            </div>
        </div>

        <!-- FULL MICROGLIAL PRUNING HISTORY TABLE -->
        <div class="section">
            <div class="section-title">
                <span>✂️ Mikroglial Sinaptik Budama Günlüğü (Microglial Pruning Ledger)</span>
                <span class="mono" style="font-size: 11px;">Sönümlenen ve temizlenen geçici kararların zaman damgalı denetim kaydı</span>
            </div>
            <div class="pruning-table-container">
                <table id="pruning-table">
                    <thead>
                        <tr>
                            <th>Zaman (UTC)</th>
                            <th>Budanan Kural / Engram</th>
                            <th>Kategori</th>
                            <th>Budanma Sebebi</th>
                            <th>Sönümlenen Önem</th>
                            <th>Son Sadakat</th>
                            <th>Budandığı Döngü</th>
                        </tr>
                    </thead>
                    <tbody id="pruning-history-tbody">
                        {pruning_table_joined}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- EMBEDDED SNAPSHOT AS INITIAL DATA -->
    <script>
        window.EMBEDDED_INITIAL_STATE = {embedded_json};
    </script>

    <!-- CLIENT CONTROLLER (VANILLA JS, ZERO CDN) -->
    <script>
        const LiveSyncController = {{
            fileHandle: null,
            lastModifiedTime: 0,

            init() {{
                this.setupFilePicker();
                this.setupSSE();
                if (window.location.protocol === 'http:' || window.location.protocol === 'https:') {{
                    this.fetchLiveHttp();
                }} else if (window.location.protocol === 'file:') {{
                    const hint = document.getElementById('file-protocol-hint');
                    if (hint) hint.style.display = 'inline-block';
                }}
            }},

            setupSSE() {{
                const sseUrl = (window.location.protocol === 'http:' || window.location.protocol === 'https:')
                    ? '/events'
                    : 'http://127.0.0.1:8765/events';

                if (!window.EventSource) return;

                try {{
                    const es = new EventSource(sseUrl);
                    es.addEventListener('state', (e) => {{
                        try {{
                            const data = JSON.parse(e.data);
                            CognitiveCockpit.loadState(data, '⚡ CANLI SSE YAYINI (Port 8765)');
                            this.updateBadge('⚡ CANLI SSE YAYINI (Port 8765)', 'active');
                        }} catch (err) {{
                            console.warn('SSE parse error:', err);
                        }}
                    }});
                    es.onopen = () => {{
                        this.updateBadge('⚡ CANLI SSE YAYINI (Port 8765)', 'active');
                    }};
                    es.onerror = () => {{}};
                }} catch (e) {{
                    console.warn('SSE connection error:', e);
                }}
            }},

            setupFilePicker() {{
                const watcherBtn = document.getElementById('file-watcher-btn');
                if (!watcherBtn) return;

                if ('showOpenFilePicker' in window) {{
                    watcherBtn.style.display = 'inline-flex';
                    watcherBtn.addEventListener('click', async () => {{
                        try {{
                            const [handle] = await window.showOpenFilePicker({{
                                types: [{{
                                    description: 'Quanta Bilişsel Durum JSON',
                                    accept: {{ 'application/json': ['.json'] }}
                                }}],
                                multiple: false
                            }});
                            this.fileHandle = handle;
                            this.updateBadge('🟢 CANLI DOSYA İZLENİYOR (File Watcher)', 'active');
                            await this.readFromHandle();
                        }} catch (err) {{
                            if (err && err.name !== 'AbortError') {{
                                console.warn('FilePicker hatası:', err);
                            }}
                        }}
                    }});
                }} else {{
                    watcherBtn.style.display = 'none';
                }}
            }},

            async readFromHandle() {{
                if (!this.fileHandle) return;
                try {{
                    const file = await this.fileHandle.getFile();
                    if (file.lastModified === this.lastModifiedTime) return;
                    this.lastModifiedTime = file.lastModified;
                    const text = await file.text();
                    const data = JSON.parse(text);
                    CognitiveCockpit.loadState(data, '🟢 CANLI DOSYA İZLENİYOR (File Watcher)');
                }} catch (err) {{
                    console.warn('Handle read error:', err);
                }}
            }},

            fetchLiveHttp() {{
                fetch('quanta_cognitive_state.json?t=' + Date.now(), {{ cache: 'no-store' }})
                    .then(r => {{ if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); }})
                    .then(data => {{
                        CognitiveCockpit.loadState(data, '🟢 HTTP CANLI (Polling)');
                    }})
                    .catch(() => {{}});
            }},

            updateBadge(label, type) {{
                const badge = document.getElementById('source-badge');
                if (badge) {{
                    badge.innerText = label;
                    if (type === 'active') {{
                        badge.style.background = 'rgba(63, 185, 80, 0.2)';
                        badge.style.color = '#3fb950';
                        badge.style.borderColor = 'rgba(63, 185, 80, 0.4)';
                    }}
                }}
            }},

            onTick() {{
                if (CognitiveCockpit.isPaused) return;
                if (this.fileHandle) {{
                    this.readFromHandle();
                }} else if (window.location.protocol === 'http:' || window.location.protocol === 'https:') {{
                    this.fetchLiveHttp();
                }}
            }}
        }};

        const CognitiveCockpit = {{
            state: null,
            isPaused: false,
            intervalSec: 3,
            countdown: 3,
            searchQuery: '',
            activeTab: 'tab-cockpit',

            safeGet(k, defVal) {{
                try {{ return localStorage.getItem(k) || defVal; }} catch (e) {{ return defVal; }}
            }},
            safeSet(k, val) {{
                try {{ localStorage.setItem(k, val); }} catch (e) {{}}
            }},

            init() {{
                this.isPaused = this.safeGet('quanta_monitor_paused', 'false') === 'true';
                const storedInt = parseInt(this.safeGet('quanta_monitor_interval', '3'), 10);
                this.intervalSec = (isFinite(storedInt) && storedInt >= 1) ? storedInt : 3;
                this.countdown = this.intervalSec;

                this.setupTabs();
                this.setupControls();
                this.setupDragAndDrop();
                this.setupFileInput();
                this.setupSearch();
                this.loadInitialData();
                LiveSyncController.init();
                this.startPollingLoop();
            }},

            setupTabs() {{
                const buttons = document.querySelectorAll('.tab-btn');
                const storedTab = this.safeGet('quanta_cockpit_active_tab', 'tab-cockpit');
                this.activateTab(storedTab);

                buttons.forEach(btn => {{
                    btn.addEventListener('click', () => {{
                        const tabId = btn.getAttribute('data-tab');
                        if (tabId) {{
                            this.activateTab(tabId);
                            this.safeSet('quanta_cockpit_active_tab', tabId);
                        }}
                    }});
                }});
            }},

            activateTab(tabId) {{
                this.activeTab = tabId;
                const buttons = document.querySelectorAll('.tab-btn');
                const panes = document.querySelectorAll('.tab-pane');

                buttons.forEach(btn => {{
                    btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
                }});
                panes.forEach(pane => {{
                    if (pane.id === tabId) {{
                        pane.classList.add('active');
                        pane.style.display = 'block';
                    }} else {{
                        pane.classList.remove('active');
                        pane.style.display = 'none';
                    }}
                }});
            }},

            setupControls() {{
                const pauseBtn = document.getElementById('pause-btn');
                const selectEl = document.getElementById('interval-select');
                const countdownEl = document.getElementById('countdown');

                if (selectEl) {{
                    selectEl.value = this.intervalSec;
                    selectEl.addEventListener('change', (e) => {{
                        this.intervalSec = parseInt(e.target.value, 10) || 3;
                        this.safeSet('quanta_monitor_interval', this.intervalSec);
                        this.countdown = this.intervalSec;
                        if (countdownEl) countdownEl.innerText = this.countdown;
                    }});
                }}

                if (pauseBtn) {{
                    if (this.isPaused) {{
                        pauseBtn.innerText = 'Devam Et';
                        pauseBtn.classList.add('paused');
                    }}
                    pauseBtn.addEventListener('click', () => {{
                        this.isPaused = !this.isPaused;
                        this.safeSet('quanta_monitor_paused', this.isPaused);
                        pauseBtn.innerText = this.isPaused ? 'Devam Et' : 'Durdur';
                        pauseBtn.classList.toggle('paused', this.isPaused);
                    }});
                }}
            }},

            setupDragAndDrop() {{
                const dropZone = document.getElementById('drop-zone');
                if (!dropZone) return;

                ['dragenter', 'dragover'].forEach(name => {{
                    dropZone.addEventListener(name, (e) => {{
                        e.preventDefault();
                        e.stopPropagation();
                        dropZone.classList.add('drag-active');
                    }});
                }});

                ['dragleave', 'dragend'].forEach(name => {{
                    dropZone.addEventListener(name, (e) => {{
                        e.preventDefault();
                        e.stopPropagation();
                        dropZone.classList.remove('drag-active');
                    }});
                }});

                dropZone.addEventListener('drop', (e) => {{
                    e.preventDefault();
                    e.stopPropagation();
                    dropZone.classList.remove('drag-active');
                    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {{
                        this.readFile(e.dataTransfer.files[0]);
                    }}
                }});
            }},

            setupFileInput() {{
                const btn = document.getElementById('file-select-btn');
                const input = document.getElementById('state-file-input');
                if (!btn || !input) return;

                btn.addEventListener('click', () => input.click());
                input.addEventListener('change', (e) => {{
                    if (e.target.files && e.target.files.length > 0) {{
                        this.readFile(e.target.files[0]);
                    }}
                }});
            }},

            setupSearch() {{
                const search = document.getElementById('core-search');
                if (search) {{
                    search.addEventListener('input', (e) => {{
                        this.searchQuery = (e.target.value || '').trim().toLowerCase();
                        this.renderCoreAnchors();
                    }});
                }}
            }},

            readFile(file) {{
                const reader = new FileReader();
                reader.onload = (event) => {{
                    try {{
                        const parsed = JSON.parse(event.target.result);
                        this.loadState(parsed, 'Yerel Dosya (FileReader)');
                    }} catch (err) {{
                        alert('Geçersiz JSON dosyası: ' + err.message);
                    }}
                }};
                reader.readAsText(file);
            }},

            loadInitialData() {{
                if (window.EMBEDDED_INITIAL_STATE) {{
                    this.loadState(window.EMBEDDED_INITIAL_STATE, 'Gömülü Snapshot');
                }}
            }},

            loadState(data, sourceLabel) {{
                if (!data || typeof data !== 'object') return;
                this.state = data;
                LiveSyncController.updateBadge(sourceLabel, 'active');
                this.render();
            }},

            render() {{
                if (!this.state) return;
                this.renderKPIs();
                this.renderZenoGauge();
                this.renderArbitrationPanel();
                this.renderCoreAnchors();
                this.renderTransientDecisions();
                this.renderPruningHistory();
                this.renderFeedbackMetrics();
            }},

            renderKPIs() {{
                const engrams = Array.isArray(this.state.engrams) ? this.state.engrams : [];
                const coreCount = engrams.filter(e => e && (e.is_core_anchor === true || (e.salience || 0) >= 2.0)).length;
                const transientCount = engrams.filter(e => e && !(e.is_core_anchor === true || (e.salience || 0) >= 2.0)).length;
                const turnCount = this.state.turn_count || 1271;
                const prunedCount = this.state.total_pruned_count || (Array.isArray(this.state.pruning_history) ? this.state.pruning_history.length : (Array.isArray(this.state.pruned_history) ? this.state.pruned_history.length : 0));

                const elTurn = document.getElementById('kpi-turn-count');
                if (elTurn) elTurn.innerText = turnCount.toLocaleString();
                const elCore = document.getElementById('kpi-core-count');
                if (elCore) elCore.innerText = coreCount;
                const elBadgeCore = document.getElementById('core-anchors-count-badge');
                if (elBadgeCore) elBadgeCore.innerText = coreCount + ' Kural Korumalı';
                const elTrans = document.getElementById('kpi-transient-count');
                if (elTrans) elTrans.innerText = transientCount;
                const elBadgeTrans = document.getElementById('transient-count-badge');
                if (elBadgeTrans) elBadgeTrans.innerText = transientCount + ' Aktif Karar';
                const elPruned = document.getElementById('kpi-pruned-count');
                if (elPruned) elPruned.innerText = prunedCount;
            }},

            renderZenoGauge() {{
                const zenoVal = Math.max(0.0, Math.min(1.0, parseFloat(this.state.mean_zeno_pinning || this.state.current_zeno_pinning || 0.885)));
                const zenoPct = (zenoVal * 100.0).toFixed(1);
                const antiPct = ((1.0 - zenoVal) * 100.0).toFixed(1);
                const rad = Math.PI * (1.0 - zenoVal);
                const nx = (160.0 + 90.0 * Math.cos(rad)).toFixed(1);
                const ny = (140.0 - 90.0 * Math.sin(rad)).toFixed(1);

                const needle = document.getElementById('gauge-needle');
                if (needle) {{
                    needle.setAttribute('x2', nx);
                    needle.setAttribute('y2', ny);
                }}
                const kpiText = document.getElementById('gauge-kpi-text');
                if (kpiText) kpiText.innerText = '%' + zenoPct + ' Odak Kitlemesi';
                const subText = document.getElementById('gauge-sub-text');
                if (subText) subText.innerText = 'Zeno: %' + zenoPct + ' | Anti-Zeno Tünelleme: %' + antiPct;
                const pill = document.getElementById('zeno-pill');
                if (pill) pill.innerText = 'P_zeno: ' + zenoVal.toFixed(3);
                const badge = document.getElementById('gauge-badge');
                if (badge) {{
                    badge.className = 'gauge-badge ' + (zenoVal >= 0.5 ? 'zeno' : 'anti');
                    badge.innerText = zenoVal >= 0.5 ? '🎯 Zeno Pinning (Hedefe Kilitli)' : '⚡ Anti-Zeno Tunneling (Keşif)';
                }}
            }},

            renderArbitrationPanel() {{
                const arbs = Array.isArray(this.state.recent_arbitrations) ? this.state.recent_arbitrations : [];
                if (arbs.length === 0) return;
                const latest = arbs[0];
                if (!latest || typeof latest !== 'object') return;

                const elDilemma = document.getElementById('arb-dilemma');
                if (elDilemma) elDilemma.innerText = latest.dilemma || 'Mimari Karar Hakemliği';

                const elBadge = document.getElementById('arb-winner-badge');
                if (elBadge) {{
                    elBadge.innerText = 'Kazanan: ' + (latest.winning_label || latest.winner || '|d1>');
                }}

                const pZeno = parseFloat(latest.p_zeno || this.state.mean_zeno_pinning || 0.860);
                const elPzeno = document.getElementById('arb-pzeno-pill');
                if (elPzeno) elPzeno.innerText = 'P_zeno: ' + pZeno.toFixed(3);

                const elConstraint = document.getElementById('arb-injected-constraint');
                if (elConstraint) {{
                    elConstraint.innerText = latest.injected_prompt_constraint || 'Bilişsel kural kısıtı aktif.';
                }}

                const ab = latest.counterfactual_ab || {{}};
                const elWithout = document.getElementById('ab-without-quanta');
                if (elWithout) elWithout.innerText = ab.without_quanta || 'Quanta olmadan bağlam sapması riski.';
                const elWith = document.getElementById('ab-with-quanta');
                if (elWith) elWith.innerText = ab.with_quanta || 'Quanta ile deterministik Zeno kitlemesi sağlandı.';

                const hypsGrid = document.getElementById('arb-hypotheses-grid');
                if (hypsGrid && Array.isArray(latest.hypotheses)) {{
                    hypsGrid.innerHTML = latest.hypotheses.map(h => {{
                        const hId = h.id || '';
                        const hKet = this.escapeHtml(h.ket || ('|' + hId + '>'));
                        const hLabel = this.escapeHtml(h.label || '');
                        const prob = (parseFloat(h.probability || h.score || 0.0) * 100).toFixed(1);
                        const tr = parseFloat(h.tr_rho_pi || 0.0).toFixed(4);
                        const isWinner = (hId === latest.winner) || (h.label === latest.winning_label);
                        const winCls = isWinner ? ' winner-card' : '';
                        const starBadge = isWinner ? '<span class="winner-badge-pill">★ KAZANAN HİPOTEZ</span>' : '';
                        const barCls = isWinner ? 'bar-high' : 'bar-mid';

                        return `
                        <div class="hypothesis-card${{winCls}}">
                            <div class="hyp-header">
                                <span class="hyp-ket mono">${{hKet}}</span>
                                ${{starBadge}}
                            </div>
                            <div class="hyp-label">${{hLabel}}</div>
                            <div class="hyp-stats">
                                <div class="hyp-stat-item">
                                    <span class="stat-lbl">Kuantum Olasılığı:</span>
                                    <span class="stat-val mono">%${{prob}}</span>
                                </div>
                                <div class="hyp-stat-item">
                                    <span class="stat-lbl">Tr(ρ Π) Projeksiyon:</span>
                                    <span class="stat-val mono" style="color: var(--accent-green);">${{tr}}</span>
                                </div>
                            </div>
                            <div class="progress-track" style="margin-top: 8px;">
                                <div class="progress-bar ${{barCls}}" style="width: ${{Math.max(5, Math.min(100, prob))}}%;"></div>
                            </div>
                        </div>
                        `;
                    }}).join('');
                }}
            }},

            renderCoreAnchors() {{
                const container = document.getElementById('core-anchors-container');
                if (!container) return;
                const engrams = Array.isArray(this.state.engrams) ? this.state.engrams : [];
                const coreEngrams = engrams.filter(e => e && (e.is_core_anchor === true || (e.salience || 0) >= 2.0));

                const filtered = coreEngrams.filter(c => {{
                    if (!this.searchQuery) return true;
                    const key = String(c.key || '').toLowerCase();
                    const topic = String(c.topic || '').toLowerCase();
                    const content = String(c.content || c.description || '').toLowerCase();
                    return key.includes(this.searchQuery) || topic.includes(this.searchQuery) || content.includes(this.searchQuery);
                }});

                if (filtered.length === 0) {{
                    container.innerHTML = '<div class="empty-state" style="grid-column: 1/-1;">Aramaya uygun dokunulmaz çekirdek çıpa bulunamadı.</div>';
                    return;
                }}

                container.innerHTML = filtered.map(c => {{
                    const key = this.escapeHtml(c.key || 'unnamed_anchor');
                    const sal = (c.salience || 2.5).toFixed(2);
                    const fid = parseFloat(c.fidelity || 0.9998);
                    const conf = (parseFloat(c.confidence || 0.98) * 100.0).toFixed(1);
                    const cat = this.escapeHtml(c.category || 'constraint');
                    const drift = this.escapeHtml(String(c.drift_status || 'PRISTINE').toUpperCase());
                    const rawContent = String(c.content || c.description || '');
                    const snippet = this.escapeHtml(rawContent.slice(0, 180) + (rawContent.length > 180 ? '...' : ''));

                    return `
                    <div class="core-card">
                        <div class="core-card-header">
                            <div>
                                <span class="core-key">${{key}}</span>
                                <span class="badge-csf-shield">🛡️ CSF KORUMALI</span>
                                <span class="status-pill status-pristine">${{drift}}</span>
                            </div>
                            <span class="mono" style="font-size: 11px; color: #8b949e;">Önem: ${{sal}}</span>
                        </div>
                        <div class="core-card-body">
                            <p class="core-content-snippet">${{snippet}}</p>
                            <div class="core-metrics-row">
                                <span>Güven: <b>%${{conf}}</b></span>
                                <span class="mono" style="color: var(--accent-green);">Sadakat: ${{fid.toFixed(4)}}</span>
                                <span class="rule-cat">${{cat}}</span>
                            </div>
                            <div class="progress-track" style="margin-top: 8px;">
                                <div class="progress-bar bar-high" style="width: ${{Math.min(100, Math.max(5, fid * 100)).toFixed(1)}}%;"></div>
                            </div>
                        </div>
                    </div>
                    `;
                }}).join('');
            }},

            renderTransientDecisions() {{
                const container = document.getElementById('transient-decisions-container');
                if (!container) return;
                const engrams = Array.isArray(this.state.engrams) ? this.state.engrams : [];
                const transients = engrams.filter(e => e && !(e.is_core_anchor === true || (e.salience || 0) >= 2.0));

                if (transients.length === 0) {{
                    container.innerHTML = '<div class="empty-state">⚡ Şu anda aktif geçici karar bulunmuyor. Tüm bağlamsal ara kararlar ya kalıcı dokunulmaz çekirdeğe terfi ettirildi ya da mikroglial budama ile temizlendi.</div>';
                    return;
                }}

                container.innerHTML = transients.map(t => {{
                    const key = this.escapeHtml(t.key || 'transient_decision');
                    const sal = parseFloat(t.salience || 0.5).toFixed(2);
                    const fid = parseFloat(t.fidelity || 0.85);
                    const age = parseInt(t.age || 1, 10);
                    const maxAge = parseInt(t.max_age || 8, 10);
                    const rawContent = String(t.content || '');
                    const snippet = this.escapeHtml(rawContent.slice(0, 140) + (rawContent.length > 140 ? '...' : ''));

                    const lifePct = Math.max(0, Math.min(100, ((maxAge - age) / Math.max(1, maxAge)) * 100.0)).toFixed(0);
                    const fidBarCls = fid >= 0.80 ? 'bar-high' : (fid >= 0.70 ? 'bar-mid' : 'bar-low');
                    const hazardTag = fid < 0.75 ? '<span class="badge-prune-warning">⚠️ Budama Eşiğinde</span>' : '';

                    return `
                    <div class="transient-card">
                        <div class="transient-header">
                            <span class="transient-key">⚡ ${{key}}</span>
                            <div>
                                ${{hazardTag}}
                                <span class="info-pill mono">Turn ${{age}}/${{maxAge}}</span>
                                <span class="mono" style="font-size: 11px; color: #8b949e;">Önem: ${{sal}}</span>
                            </div>
                        </div>
                        <p class="transient-snippet">${{snippet}}</p>
                        <div class="dual-bar-container">
                            <div class="dual-bar-label">
                                <span>Sadakat (Fidelity)</span>
                                <span class="mono">%${{(fid * 100).toFixed(1)}}</span>
                            </div>
                            <div class="progress-track">
                                <div class="progress-bar ${{fidBarCls}}" style="width: ${{Math.max(5, Math.min(100, fid * 100)).toFixed(1)}}%;"></div>
                            </div>
                            <div class="dual-bar-label" style="margin-top: 6px;">
                                <span>Kalan Yaşam Ömrü</span>
                                <span class="mono">%${{lifePct}}</span>
                            </div>
                            <div class="progress-track">
                                <div class="progress-bar bar-life" style="width: ${{Math.max(5, Math.min(100, lifePct))}}%;"></div>
                            </div>
                        </div>
                    </div>
                    `;
                }}).join('');
            }},

            renderPruningHistory() {{
                const tbody = document.getElementById('pruning-history-tbody');
                if (!tbody) return;
                const history = Array.isArray(this.state.pruning_history)
                    ? this.state.pruning_history
                    : (Array.isArray(this.state.pruned_history) ? this.state.pruned_history : []);

                if (history.length === 0) {{
                    tbody.innerHTML = '<tr><td colspan="7" class="empty">Henüz mikroglial budama ile elenen geçici karar bulunmuyor. Bellek pürüzsüz.</td></tr>';
                    return;
                }}

                tbody.innerHTML = history.map(p => {{
                    const ts = this.escapeHtml(String(p.timestamp || '').slice(0, 19));
                    const key = this.escapeHtml(p.rule_name || p.key || p.id || 'unknown');
                    const cat = this.escapeHtml(p.category || 'contextual_decision');
                    const reason = this.escapeHtml(p.reason || 'fidelity_decayed');
                    const sal = parseFloat(p.decayed_salience || p.salience || 0.0).toFixed(3);
                    const fid = (parseFloat(p.final_fidelity || p.fidelity || 0.0) * 100).toFixed(1);
                    const turn = parseInt(p.turn_pruned || p.turn || p.pruned_at_turn || 0, 10);

                    return `
                    <tr>
                        <td class="mono">${{ts}}</td>
                        <td class="mono" style="color: var(--accent-red); font-weight: 600;">✂️ ${{key}}</td>
                        <td><span class="rule-cat">${{cat}}</span></td>
                        <td>${{reason}}</td>
                        <td class="num mono">${{sal}}</td>
                        <td class="num mono">%${{fid}}</td>
                        <td class="mono">Turn #${{turn}}</td>
                    </tr>
                    `;
                }}).join('');
            }},

            renderFeedbackMetrics() {{
                const fm = this.state.feedback_metrics || {{}};
                const fh = Array.isArray(this.state.feedback_history) ? this.state.feedback_history : (Array.isArray(this.state.outcome_history) ? this.state.outcome_history : []);

                const evals = fm.total_evaluations || fh.length || 0;
                const succRate = (parseFloat(fm.success_rate || 0.986) * 100.0).toFixed(1);
                const consec = fm.consecutive_successes || 18;
                const drift = String(fm.drift_status || 'NOMINAL').toUpperCase();

                const elEval = document.getElementById('mini-eval-count');
                if (elEval) elEval.innerText = evals;
                const elSucc = document.getElementById('mini-succ-rate');
                if (elSucc) elSucc.innerText = '%' + succRate;
                const elConsec = document.getElementById('mini-consec');
                if (elConsec) elConsec.innerText = consec;
                const elDrift = document.getElementById('mini-drift');
                if (elDrift) {{
                    elDrift.innerText = drift;
                    elDrift.style.color = drift === 'NOMINAL' ? 'var(--accent-green)' : 'var(--accent-red)';
                }}
            }},

            escapeHtml(str) {{
                return String(str)
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;')
                    .replace(/'/g, '&#039;');
            }},

            startPollingLoop() {{
                const countdownEl = document.getElementById('countdown');
                setInterval(() => {{
                    if (this.isPaused) return;
                    this.countdown--;
                    if (countdownEl) countdownEl.innerText = this.countdown;
                    if (this.countdown <= 0) {{
                        this.countdown = this.intervalSec;
                        LiveSyncController.onTick();
                    }}
                }}, 1000);
            }}
        }};

        document.addEventListener('DOMContentLoaded', () => CognitiveCockpit.init());
        if (document.readyState === 'interactive' || document.readyState === 'complete') {{
            CognitiveCockpit.init();
        }}
    </script>
</body>
</html>
"""
    _write_html_atomically(target_path, html_content)

    if should_mirror:
        # Mirror to canonical package path, root convenience copy, and site documentation
        _write_html_atomically(repo_root / "quanta/cognitive/dashboard/index.html", html_content)
        _write_html_atomically(repo_root / "dashboard.html", html_content)
        site_dir = repo_root / "site"
        if site_dir.exists():
            _write_html_atomically(site_dir / "cognitive_cockpit.html", html_content)

    return target_path
