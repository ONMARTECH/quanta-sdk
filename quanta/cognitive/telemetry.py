"""quanta.cognitive.telemetry — Centralized Cognitive Telemetry & Audit Ledger.

Records, tracks, and visualizes:
1. Architectural arbitration decisions (goals, candidate options, winning option, confidence, latency).
2. Subconscious hook memory state (SWR vital replay rules, microglial pruning counts, step latency).
3. Cross-project telemetry analytics and interactive standalone dashboard generation.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import time
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
    try:
        DEFAULT_TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
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
                tr_rho_pi = {
                    r["option"]: r.get("tr_rho_pi", r.get("score", 0.0))
                    for r in ranking
                }
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

        record = {
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
            "tr_rho_pi": tr_rho_pi,
            "regime": regime,
            "latency_ms": round(float(latency_ms), 2),
            "pytorch_latency_ms": round(float(pytorch_latency_ms), 2),
            "ranking": ranking or [],
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
                normalized_rules.append({
                    "rule": r.get("rule") or r.get("key") or "unnamed_rule",
                    "key": r.get("key") or r.get("rule") or "unnamed_rule",
                    "fidelity": round(float(r.get("fidelity", 0.9998)), 6),
                    "salience": round(float(r.get("salience", 1.0)), 2),
                    "category": r.get("category", "constraint"),
                })
            else:
                normalized_rules.append({
                    "rule": str(r),
                    "key": str(r),
                    "fidelity": 0.9998,
                    "salience": 2.5,
                    "category": "constraint",
                })

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
            "total_pruned_count": total_pruned_count if total_pruned_count is not None else pruned_count,
            "kappa_csf": kappa_csf,
            "active_engrams_count": active_engrams_count if active_engrams_count is not None else len(normalized_rules),
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
                    if isinstance(entry, dict) and (event_type is None or entry.get("event_type") == event_type):
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
    hook_steps = [e for e in all_events if isinstance(e, dict) and e.get("event_type") == "hook_step"]

    total_decisions = len(decisions)
    total_hook_steps = len(hook_steps)

    valid_latencies = [
        float(d["latency_ms"])
        for d in decisions
        if isinstance(d.get("latency_ms"), (int, float))
        and not isinstance(d.get("latency_ms"), bool)
        and math.isfinite(d["latency_ms"])
    ]
    avg_latency = (
        sum(valid_latencies) / len(valid_latencies)
        if valid_latencies
        else 0.0
    )

    valid_confidences = [
        float(d["confidence"])
        for d in decisions
        if isinstance(d.get("confidence"), (int, float))
        and not isinstance(d.get("confidence"), bool)
        and math.isfinite(d["confidence"])
    ]
    avg_confidence = (
        sum(valid_confidences) / len(valid_confidences)
        if valid_confidences
        else 0.0
    )

    workspaces = sorted(
        {e.get("workspace", "General") for e in all_events if isinstance(e, dict) and e.get("workspace")}
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
            latest_rules.append({
                "name": rule_name,
                "fidelity": fid,
                "fidelity_pct": round(fid * 100.0, 2),
                "salience": round(sal, 2),
                "category": cat,
                "last_replayed": latest_hook.get("iso_time", "N/A"),
            })

    # If no hook steps recorded in log yet, check local state files
    if not latest_rules:
        state_candidates = [
            Path("quanta_cognitive_state.json"),
            Path(os.path.expanduser("~/.gemini/antigravity/quanta_cognitive_state.json")),
        ]
        for sp in state_candidates:
            if sp.exists():
                with contextlib.suppress(Exception):
                    with open(sp, encoding="utf-8") as sf:
                        st = json.load(sf)
                        for e in st.get("engrams", []):
                            fid = _safe_float(e.get("fidelity"), 0.9998)
                            sal = _safe_float(e.get("salience"), 1.0)
                            latest_rules.append({
                                "name": e.get("key", "unnamed"),
                                "fidelity": fid,
                                "fidelity_pct": round(fid * 100.0, 2),
                                "salience": round(sal, 2),
                                "category": e.get("category", "general"),
                                "last_replayed": "Hafıza Kaydı",
                            })
                if latest_rules:
                    break

    # If still empty (fresh startup), provide default anchor rules for visualization
    if not latest_rules:
        latest_rules = [
            {"name": "native_first_rule", "fidelity": 0.9998, "fidelity_pct": 99.98, "salience": 2.8, "category": "constraint", "last_replayed": "Başlangıç"},
            {"name": "executive_summary_rule", "fidelity": 0.9998, "fidelity_pct": 99.98, "salience": 2.5, "category": "constraint", "last_replayed": "Başlangıç"},
            {"name": "scientific_integrity_rule", "fidelity": 0.9998, "fidelity_pct": 99.98, "salience": 2.5, "category": "constraint", "last_replayed": "Başlangıç"},
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
    mean_zeno_pinning = (
        sum(zeno_factors) / len(zeno_factors) if zeno_factors else 0.842
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
        "mean_anti_zeno": round(mean_anti_zeno, 4),
    }


def generate_dashboard_html(
    output_path: Path | None = None,
    telemetry_file: Path | None = None,
) -> Path:
    """Generates an interactive, standalone HTML telemetry dashboard."""
    summary = get_telemetry_summary(telemetry_file=telemetry_file)
    recent_decisions = list(summary.get("recent_decisions") or [])
    recent_decisions.reverse()  # Newest first

    target_path = output_path or (DEFAULT_TELEMETRY_DIR / "dashboard.html")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. SESSIONS TABLE ---
    active_sessions = list(summary.get("active_sessions") or [])
    session_rows = []
    for s in active_sessions[:6]:
        if not isinstance(s, dict):
            continue
        ws = str(s.get("workspace", "General") or "General")
        ws_esc = html.escape(ws)
        sec = _safe_int(s.get("seconds_ago"), 0)
        time_text = f"{sec} sn önce" if sec < 60 else f"{sec // 60} dk önce"
        is_live = bool(s.get("is_live", False))
        status_badge = (
            '<span class="badge-live"><span class="pulse-dot"></span> CANLI AKTİF</span>'
            if is_live
            else '<span class="badge-idle">BEKLEMEDE</span>'
        )
        query = str(s.get("last_query") or s.get("winner") or "İşlem yürütülüyor")
        query_esc = html.escape(query)
        step = _safe_int(s.get("step_idx"), 0)
        step_text = f"Adım #{step}" if step else "Hakem Kararı"
        step_text_esc = html.escape(step_text)

        session_rows.append(
            f"""
            <tr>
                <td><span class="project-pill">{ws_esc}</span></td>
                <td style="color: var(--heading); font-weight: 500;">{query_esc[:75]}{'...' if len(query_esc)>75 else ''}</td>
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
        ws_esc = html.escape(ws)
        goal = str(d.get("goal", "") or "")
        goal_esc = html.escape(goal, quote=True)
        winner = str(d.get("winner", "") or "")
        winner_esc = html.escape(winner)
        iso = str(d.get("iso_time", "") or "")
        iso_esc = html.escape(iso)
        zeno_p = _safe_float(d.get("zeno_pinning_factor"), 0.0)

        # 6-Qubit quantum ranking with Tr(\rho \Pi)
        ranking_items = []
        raw_ranking = d.get("ranking", [])
        if isinstance(raw_ranking, list):
            for rk in raw_ranking[:3]:
                if not isinstance(rk, dict):
                    continue
                r_opt_raw = str(rk.get("option", "") or "")
                r_opt_esc = html.escape(r_opt_raw)
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
                <td class="goal-cell" title="{goal_esc}">{goal_esc[:60]}{'...' if len(goal_esc)>60 else ''}</td>
                <td class="winner-cell">
                    <strong>{winner_esc}</strong>
                    {f'<div class="quantum-ranking-sub">{ranking_html}</div>' if ranking_html else ''}
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

    ws_list_html = "".join(
        f'<span class="project-pill">{html.escape(str(w))}</span>' for w in (summary.get("active_workspaces") or [])
    ) or "<span>Henüz aktif proje yok</span>"

    # --- 3. ZENO PINNING SVG GAUGE CALCULATIONS ---
    zeno_val = max(0.0, min(1.0, _safe_float(summary.get("mean_zeno_pinning"), 0.842)))
    zeno_pct = round(zeno_val * 100.0, 1)
    anti_zeno_pct = round((1.0 - zeno_val) * 100.0, 1)

    # Angle for gauge: 180 deg (left, 0% Zeno) to 0 deg (right, 100% Zeno)
    rad = math.pi * (1.0 - zeno_val)
    nx = 160.0 + 90.0 * math.cos(rad)
    ny = 140.0 - 90.0 * math.sin(rad)

    # --- 4. SUBCONSCIOUS MEMORY & RULE FIDELITY PROGRESS BARS ---
    rules_html_list = []
    raw_rules = summary.get("latest_rules") or []
    if isinstance(raw_rules, list):
        for r in raw_rules:
            if not isinstance(r, dict):
                continue
            r_name = str(r.get("name", "unnamed") or "unnamed")
            r_name_esc = html.escape(r_name)
            r_fid = _safe_float(r.get("fidelity"), 0.9998)
            r_pct = _safe_float(r.get("fidelity_pct"), round(r_fid * 100.0, 2))
            r_cat = str(r.get("category", "constraint") or "constraint")
            r_cat_esc = html.escape(r_cat)
            r_sal = _safe_float(r.get("salience"), 2.5)
            r_time = str(r.get("last_replayed", "") or "")
            r_time_esc = html.escape(r_time)

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

    tot_dec = _safe_int(summary.get("total_decisions"), 0)
    avg_lat = _safe_float(summary.get("avg_decision_latency_ms"), 0.0)
    avg_conf = _safe_float(summary.get("avg_confidence_pct"), 0.0)
    tot_hooks = _safe_int(summary.get("total_hook_steps"), 0)
    tot_active = _safe_int(summary.get("total_active_engrams"), 0)
    tot_pruned = _safe_int(summary.get("total_pruned_engrams"), 0)

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
            --text: #c9d1d9;
            --heading: #f0f6fc;
            --accent: #58a6ff;
            --accent-green: #3fb950;
            --accent-purple: #bc8cff;
            --accent-amber: #d29922;
            --accent-red: #f85149;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
        body {{ background: var(--bg); color: var(--text); padding: 28px; line-height: 1.5; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 18px; margin-bottom: 24px; }}
        .header h1 {{ font-size: 22px; color: var(--heading); display: flex; align-items: center; gap: 10px; }}
        .header .live-badge {{ background: rgba(63, 185, 80, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); font-size: 11px; padding: 3px 8px; border-radius: 12px; text-transform: uppercase; font-weight: bold; letter-spacing: 0.5px; }}
        
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .kpi-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 18px; }}
        .kpi-title {{ font-size: 12px; text-transform: uppercase; color: #8b949e; letter-spacing: 0.5px; margin-bottom: 6px; }}
        .kpi-value {{ font-size: 26px; font-weight: 700; color: var(--heading); }}
        .kpi-sub {{ font-size: 12px; color: var(--accent); margin-top: 4px; }}

        .cockpit-split {{ display: grid; grid-template-columns: 360px 1fr; gap: 20px; margin-bottom: 24px; }}
        @media (max-width: 960px) {{ .cockpit-split {{ grid-template-columns: 1fr; }} }}

        .section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
        .section-title {{ font-size: 15px; font-weight: 600; color: var(--heading); margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }}
        
        /* ZENO GAUGE STYLES */
        .gauge-container {{ display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 10px 0; }}
        .gauge-kpi {{ font-size: 22px; font-weight: 700; color: var(--heading); margin-top: 4px; }}
        .gauge-sub {{ font-size: 12px; color: #8b949e; margin-top: 2px; text-align: center; }}
        .gauge-badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; margin-top: 8px; }}
        .gauge-badge.zeno {{ background: rgba(63, 185, 80, 0.15); color: var(--accent-green); border: 1px solid rgba(63, 185, 80, 0.3); }}
        .gauge-badge.anti {{ background: rgba(188, 140, 255, 0.15); color: var(--accent-purple); border: 1px solid rgba(188, 140, 255, 0.3); }}

        /* SUBCONSCIOUS PROGRESS BARS */
        .rule-card {{ background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; padding: 12px 14px; margin-bottom: 10px; }}
        .rule-card:last-child {{ margin-bottom: 0; }}
        .rule-meta-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 6px; }}
        .rule-title-box {{ display: flex; align-items: center; gap: 8px; }}
        .rule-name {{ font-weight: 600; font-size: 13px; color: var(--heading); }}
        .rule-cat {{ background: rgba(88, 166, 255, 0.1); color: var(--accent); font-size: 10px; padding: 1px 6px; border-radius: 8px; text-transform: uppercase; font-weight: 600; }}
        .rule-sal {{ font-size: 11px; color: #8b949e; }}
        .status-pill {{ font-size: 10px; padding: 1px 6px; border-radius: 8px; font-weight: 700; }}
        .status-pristine {{ background: rgba(63, 185, 80, 0.2); color: var(--accent-green); }}
        .status-active {{ background: rgba(88, 166, 255, 0.2); color: var(--accent); }}
        .status-decaying {{ background: rgba(210, 153, 34, 0.2); color: var(--accent-amber); }}
        .rule-stats-box {{ display: flex; align-items: center; gap: 12px; }}
        .rule-time {{ font-size: 11px; color: #8b949e; }}
        .rule-fid-value {{ font-size: 13px; font-weight: 700; color: var(--accent-green); }}

        .progress-track {{ width: 100%; height: 8px; background: #21262d; border-radius: 4px; overflow: hidden; }}
        .progress-bar {{ height: 100%; border-radius: 4px; transition: width 0.4s ease; }}
        .bar-high {{ background: linear-gradient(90deg, #238636, #3fb950); }}
        .bar-mid {{ background: linear-gradient(90deg, #1f6feb, #58a6ff); }}
        .bar-low {{ background: linear-gradient(90deg, #9e6a03, #d29922); }}

        /* QUANTUM RANKINGS IN TABLE */
        .quantum-ranking-sub {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }}
        .rank-item {{ font-size: 11px; font-family: ui-monospace, SFMono-Regular, monospace; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 1px 6px; border-radius: 4px; color: #8b949e; }}
        .rank-item.winner {{ color: var(--accent); border-color: rgba(88, 166, 255, 0.3); background: rgba(88, 166, 255, 0.08); font-weight: 600; }}

        .projects-wrapper {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
        th {{ background: rgba(255,255,255,0.02); color: #8b949e; font-weight: 600; padding: 10px 14px; border-bottom: 1px solid var(--border); }}
        td {{ padding: 12px 14px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
        tr:hover td {{ background: rgba(255,255,255,0.03); }}
        .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; color: #8b949e; }}
        .project-pill {{ background: rgba(88, 166, 255, 0.12); color: var(--accent); border: 1px solid rgba(88, 166, 255, 0.3); padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 500; }}
        .badge {{ padding: 3px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; display: inline-block; }}
        .badge-zeno {{ background: rgba(63, 185, 80, 0.15); color: var(--accent-green); border: 1px solid rgba(63, 185, 80, 0.3); }}
        .badge-solution {{ background: rgba(88, 166, 255, 0.15); color: var(--accent); border: 1px solid rgba(88, 166, 255, 0.3); }}
        .badge-anti {{ background: rgba(188, 140, 255, 0.15); color: var(--accent-purple); border: 1px solid rgba(188, 140, 255, 0.3); }}
        .winner-cell {{ color: #58a6ff; }}
        .num {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
        .empty {{ text-align: center; color: #8b949e; padding: 24px; }}
        
        @keyframes pulse {{
            0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(63, 185, 80, 0.7); }}
            70% {{ transform: scale(1); box-shadow: 0 0 0 6px rgba(63, 185, 80, 0); }}
            100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(63, 185, 80, 0); }}
        }}
        .pulse-dot {{
            width: 9px; height: 9px; background: #3fb950; border-radius: 50%;
            display: inline-block; animation: pulse 2s infinite;
        }}
        .live-controls {{
            display: flex; align-items: center; gap: 10px;
        }}
        .refresh-btn {{
            background: #21262d; border: 1px solid var(--border); color: var(--text);
            padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 12px;
            font-weight: 500; transition: all 0.2s;
        }}
        .refresh-btn:hover {{ background: #30363d; color: var(--heading); }}
        .refresh-btn.paused {{ border-color: var(--accent-amber); color: var(--accent-amber); }}
        .interval-select {{
            background: #21262d; border: 1px solid var(--border); color: var(--text);
            padding: 3px 6px; border-radius: 6px; font-size: 12px; cursor: pointer;
        }}
        .badge-live {{ background: rgba(63, 185, 80, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); font-size: 11px; padding: 2px 8px; border-radius: 12px; font-weight: 600; display: inline-flex; align-items: center; gap: 6px; }}
        .badge-idle {{ background: rgba(110, 118, 129, 0.15); color: #8b949e; border: 1px solid rgba(110, 118, 129, 0.3); font-size: 11px; padding: 2px 8px; border-radius: 12px; font-weight: 500; }}
        .info-pill {{ background: rgba(255,255,255,0.05); border: 1px solid var(--border); padding: 2px 8px; border-radius: 10px; font-size: 11px; color: #8b949e; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>
            <span>⚛️ Quanta Bilişsel Hakem & Telemetri Kokpiti</span>
            <span class="live-badge">Canlı 6-Qubit Çift Motor</span>
        </h1>
        <div class="live-controls">
            <span class="pulse-dot"></span>
            <span class="mono">Yenileme: <b id="countdown">3</b>s</span>
            <select id="interval-select" class="interval-select" title="Yenileme Aralığı">
                <option value="2">2 sn</option>
                <option value="3" selected>3 sn</option>
                <option value="5">5 sn</option>
                <option value="10">10 sn</option>
            </select>
            <button id="pause-btn" class="refresh-btn">Durdur</button>
            <span class="mono" style="margin-left: 6px; color: #6e7681;">|</span>
            <span class="mono" style="font-size: 11px; margin-left: 6px;">Oluşturuldu: {datetime.now().strftime("%H:%M:%S")}</span>
        </div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Toplam Karar Sayısı</div>
            <div class="kpi-value">{tot_dec}</div>
            <div class="kpi-sub">Arbitrasyon Çatallanması</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Ortalama Gecikme</div>
            <div class="kpi-value">{avg_lat:.2f} ms</div>
            <div class="kpi-sub">Apple Silicon / 6-Qubit Zeno</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Ortalama Karar Güveni</div>
            <div class="kpi-value">%{avg_conf:.1f}</div>
            <div class="kpi-sub">Dopaminerjik Odak Kitlemesi</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Bilinçaltı Kanca Adımları</div>
            <div class="kpi-value">{tot_hooks}</div>
            <div class="kpi-sub">SWR Replay & CSF Korumalı</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Aktif Engram Sayısı</div>
            <div class="kpi-value">{tot_active}</div>
            <div class="kpi-sub">Budanan: {tot_pruned} Engram</div>
        </div>
    </div>

    <!-- COCKPIT SPLIT: ZENO GAUGE + SUBCONSCIOUS MEMORY GUARDIAN -->
    <div class="cockpit-split">
        <!-- LEFT: SVG ZENO PINNING GAUGE -->
        <div class="section" style="margin-bottom: 0;">
            <div class="section-title">
                <span>🎯 Zeno Pinning Göstergesi</span>
                <span class="info-pill">P_zeno: {zeno_val:.3f}</span>
            </div>
            <div class="gauge-container">
                <svg class="zeno-gauge-svg" viewBox="0 0 320 180" width="300" height="168">
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
                    <line x1="160" y1="140" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#f0f6fc" stroke-width="3.5" stroke-linecap="round"/>
                    <circle cx="160" cy="140" r="8" fill="#58a6ff" stroke="#f0f6fc" stroke-width="2.5"/>
                    <circle cx="160" cy="140" r="3" fill="#0d1117"/>
                    <!-- Labels -->
                    <text x="35" y="165" fill="#bc8cff" font-size="11" font-weight="600">Anti-Zeno</text>
                    <text x="285" y="165" text-anchor="end" fill="#3fb950" font-size="11" font-weight="600">Zeno Pinning</text>
                </svg>
                <div class="gauge-kpi">%{zeno_pct} Odak Kitlemesi</div>
                <div class="gauge-sub">Zeno: %{zeno_pct} | Anti-Zeno Tünelleme: %{anti_zeno_pct}</div>
                <div class="gauge-badge {'zeno' if zeno_val >= 0.5 else 'anti'}">
                    {'🎯 Zeno Pinning (Hedefe Kilitli)' if zeno_val >= 0.5 else '⚡ Anti-Zeno Tunneling (Keşif)'}
                </div>
            </div>
        </div>

        <!-- RIGHT: SUBCONSCIOUS MEMORY & RULE FIDELITY PROGRESS BARS -->
        <div class="section" style="margin-bottom: 0;">
            <div class="section-title">
                <span>🧠 Bilinçaltı Kural Muhafızlığı (SWR Replay & Sadakat Barları)</span>
                <span class="info-pill mono">κ_csf = 1/6250</span>
            </div>
            <div class="rules-container">
                {rules_joined}
            </div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">
            <span>⚡ Canlıda Aktif Çalışan Projeler & Son İstekler</span>
            <span class="mono" style="font-size:12px;">Soran Projeler ve Bilişsel Kanca Durumu</span>
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

    <div class="section">
        <div class="section-title">İzlenen Aktif Çalışma Alanları (Workspaces)</div>
        <div class="projects-wrapper">{ws_list_html}</div>
    </div>

    <div class="section">
        <div class="section-title">
            <span>Son Hakem Kararları ve 6-Qubit Kuantum Sıralaması</span>
            <span class="mono" style="font-size:12px;">En son 10 karar gösteriliyor (Tr(ρ Π) Hilbert Projeksiyonu)</span>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Zaman (UTC)</th>
                    <th>Proje / Workspace</th>
                    <th>Karar Hedefi (Goal)</th>
                    <th>Kazanan Karar & 6-Qubit Sıralama</th>
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

    <script>
        function safeGet(k, defVal) {{
            try {{ return localStorage.getItem(k) || defVal; }} catch (e) {{ return defVal; }}
        }}
        function safeSet(k, val) {{
            try {{ localStorage.setItem(k, val); }} catch (e) {{}}
        }}

        let intervalSec = parseInt(safeGet('quanta_monitor_interval', '3'), 10);
        if (!isFinite(intervalSec) || intervalSec < 1) {{ intervalSec = 3; }}
        let isPaused = safeGet('quanta_monitor_paused', 'false') === 'true';
        let remaining = intervalSec;

        const countdownEl = document.getElementById('countdown');
        const pauseBtn = document.getElementById('pause-btn');
        const selectEl = document.getElementById('interval-select');

        if (selectEl) {{
            selectEl.value = intervalSec;
            selectEl.addEventListener('change', (e) => {{
                intervalSec = parseInt(e.target.value, 10);
                safeSet('quanta_monitor_interval', intervalSec);
                remaining = intervalSec;
                if (countdownEl) countdownEl.innerText = remaining;
            }});
        }}

        if (pauseBtn) {{
            if (isPaused) {{
                pauseBtn.innerText = 'Devam Et';
                pauseBtn.classList.add('paused');
            }}
            pauseBtn.addEventListener('click', () => {{
                isPaused = !isPaused;
                safeSet('quanta_monitor_paused', isPaused);
                pauseBtn.innerText = isPaused ? 'Devam Et' : 'Durdur';
                pauseBtn.classList.toggle('paused', isPaused);
            }});
        }}

        setInterval(() => {{
            if (isPaused) return;
            remaining--;
            if (countdownEl) countdownEl.innerText = remaining;
            if (remaining <= 0) {{
                window.location.reload();
            }}
        }}, 1000);
    </script>
</body>
</html>
"""
    temp_target = target_path.with_name(f".tmp_{target_path.name}_{os.getpid()}_{time.time_ns()}")
    try:
        with open(temp_target, "w", encoding="utf-8") as f:
            f.write(html_content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_target, target_path)
    except Exception:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    finally:
        if temp_target.exists():
            with contextlib.suppress(OSError):
                temp_target.unlink()

    return target_path
