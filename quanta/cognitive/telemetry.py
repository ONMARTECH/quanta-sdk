"""quanta.cognitive.telemetry — Centralized Cognitive Telemetry & Audit Ledger.

Records, tracks, and visualizes:
1. Architectural arbitration decisions (goals, candidate options, winning option, confidence, latency).
2. Subconscious hook memory state (SWR vital replay rules, microglial pruning counts, step latency).
3. Cross-project telemetry analytics and interactive standalone dashboard generation.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

DEFAULT_TELEMETRY_DIR = Path(os.path.expanduser("~/.gemini/antigravity/telemetry"))
DEFAULT_TELEMETRY_FILE = DEFAULT_TELEMETRY_DIR / "quanta_cognitive_telemetry.jsonl"


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
) -> None:
    """Appends an arbitration decision record to the persistent telemetry ledger.

    Guaranteed fail-safe: never raises exceptions or interrupts agent execution.
    """
    try:
        log_file = telemetry_file or get_telemetry_file()
        now = time.time()
        iso_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

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
            "regime": regime,
            "latency_ms": round(float(latency_ms), 2),
            "ranking": ranking or [],
        }

        with open(log_file, "a", encoding="utf-8") as f:
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
    rules_replayed: list[str],
    pruned_count: int,
    latency_ms: float,
    workspace: str | None = None,
    last_user_query: str = "",
    telemetry_file: Path | None = None,
) -> None:
    """Appends a subconscious hook step telemetry record to the persistent ledger."""
    try:
        log_file = telemetry_file or get_telemetry_file()
        now = time.time()
        iso_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        record = {
            "event_type": "hook_step",
            "timestamp": now,
            "iso_time": iso_str,
            "workspace": detect_workspace(workspace),
            "conversation_id": conversation_id,
            "step_idx": step_idx,
            "turn_count": turn_count,
            "last_user_query": last_user_query[:140] if last_user_query else "",
            "rules_replayed": rules_replayed,
            "pruned_count": pruned_count,
            "latency_ms": round(float(latency_ms), 2),
        }

        with open(log_file, "a", encoding="utf-8") as f:
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
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                with contextlib.suppress(Exception):
                    entry = json.loads(line)
                    if event_type is None or entry.get("event_type") == event_type:
                        events.append(entry)
    except Exception:
        return []

    return events[-limit:]


def get_telemetry_summary(telemetry_file: Path | None = None) -> dict[str, Any]:
    """Aggregates telemetry statistics across all recorded projects."""
    all_events = read_telemetry_events(limit=500, telemetry_file=telemetry_file)
    decisions = [e for e in all_events if e.get("event_type") == "decision"]
    hook_steps = [e for e in all_events if e.get("event_type") == "hook_step"]

    total_decisions = len(decisions)
    total_hook_steps = len(hook_steps)

    avg_latency = (
        sum(d.get("latency_ms", 0.0) for d in decisions) / total_decisions
        if total_decisions > 0
        else 0.0
    )
    avg_confidence = (
        sum(d.get("confidence", 0.0) for d in decisions) / total_decisions
        if total_decisions > 0
        else 0.0
    )

    workspaces = sorted(
        {e.get("workspace", "General") for e in all_events if e.get("workspace")}
    )

    regimes: dict[str, int] = {}
    for d in decisions:
        reg = d.get("regime", "Unknown")
        regimes[reg] = regimes.get(reg, 0) + 1

    now_ts = time.time()
    sessions: dict[str, dict[str, Any]] = {}
    for e in reversed(all_events):
        conv_id = e.get("conversation_id") or e.get("workspace", "unknown")
        if conv_id not in sessions:
            ev_time = e.get("timestamp", now_ts)
            sec_ago = max(0, int(now_ts - ev_time))
            query = e.get("last_user_query") or e.get("goal") or ""
            sessions[conv_id] = {
                "workspace": e.get("workspace", "General"),
                "conversation_id": conv_id,
                "last_active_time": e.get("iso_time", ""),
                "seconds_ago": sec_ago,
                "is_live": sec_ago <= 300,
                "last_query": query,
                "last_event_type": e.get("event_type", ""),
                "step_idx": e.get("step_idx", 0),
                "winner": e.get("winner", ""),
            }

    active_sessions = list(sessions.values())
    active_sessions.sort(key=lambda s: s["seconds_ago"])

    return {
        "total_decisions": total_decisions,
        "total_hook_steps": total_hook_steps,
        "avg_decision_latency_ms": round(avg_latency, 2),
        "avg_confidence_pct": round(avg_confidence * 100.0, 1),
        "active_workspaces": workspaces,
        "active_sessions": active_sessions,
        "regime_distribution": regimes,
        "recent_decisions": decisions[-10:],
    }


def generate_dashboard_html(
    output_path: Path | None = None,
    telemetry_file: Path | None = None,
) -> Path:
    """Generates an interactive, standalone HTML telemetry dashboard."""
    summary = get_telemetry_summary(telemetry_file=telemetry_file)
    recent_decisions = summary.get("recent_decisions", [])
    recent_decisions.reverse()  # Newest first

    target_path = output_path or (DEFAULT_TELEMETRY_DIR / "dashboard.html")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    active_sessions = summary.get("active_sessions", [])
    session_rows = []
    for s in active_sessions[:6]:
        ws = s.get("workspace", "General")
        sec = s.get("seconds_ago", 0)
        time_text = f"{sec} sn önce" if sec < 60 else f"{sec // 60} dk önce"
        is_live = s.get("is_live", False)
        status_badge = (
            '<span class="badge-live"><span class="pulse-dot"></span> CANLI AKTİF</span>'
            if is_live
            else '<span class="badge-idle">BEKLEMEDE</span>'
        )
        query = s.get("last_query") or s.get("winner") or "İşlem yürütülüyor"
        step = s.get("step_idx", 0)
        step_text = f"Adım #{step}" if step else "Hakem Kararı"

        session_rows.append(
            f"""
            <tr>
                <td><span class="project-pill">{ws}</span></td>
                <td style="color: var(--heading); font-weight: 500;">{query[:75]}{'...' if len(query)>75 else ''}</td>
                <td class="mono">{time_text}</td>
                <td class="mono" style="font-size: 11px;">{step_text}</td>
                <td>{status_badge}</td>
            </tr>
            """
        )
    sessions_joined = (
        "\n".join(session_rows)
        if session_rows
        else '<tr><td colspan="5" class="empty">Henüz aktif proje oturumu kaydedilmedi.</td></tr>'
    )

    rows_html = []
    for d in recent_decisions:
        regime = d.get("regime", "")
        if "Zeno" in regime:
            regime_badge = '<span class="badge badge-zeno">Zeno Pinning</span>'
        elif "Bilişsel" in regime or "Agent" in regime:
            regime_badge = '<span class="badge badge-solution">Bilişsel Çözüm</span>'
        else:
            regime_badge = '<span class="badge badge-anti">Anti-Zeno</span>'
        conf = d.get("confidence", 0.0) * 100.0
        lat = d.get("latency_ms", 0.0)
        ws = d.get("workspace", "General")
        goal = d.get("goal", "")
        winner = d.get("winner", "")
        iso = d.get("iso_time", "")

        rows_html.append(
            f"""
            <tr>
                <td class="mono">{iso}</td>
                <td><span class="project-pill">{ws}</span></td>
                <td class="goal-cell" title="{goal}">{goal[:65]}{'...' if len(goal)>65 else ''}</td>
                <td class="winner-cell"><strong>{winner}</strong></td>
                <td>{regime_badge}</td>
                <td class="num">{conf:.1f}%</td>
                <td class="num">{lat:.2f} ms</td>
            </tr>
            """
        )

    rows_joined = (
        "\n".join(rows_html)
        if rows_html
        else '<tr><td colspan="7" class="empty">Henüz kaydedilmiş hakem kararı bulunmuyor.</td></tr>'
    )

    ws_list_html = "".join(
        f'<span class="project-pill">{w}</span>' for w in summary.get("active_workspaces", [])
    ) or "<span>Henüz aktif proje yok</span>"

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
            --border: #30363d;
            --text: #c9d1d9;
            --heading: #f0f6fc;
            --accent: #58a6ff;
            --accent-green: #3fb950;
            --accent-purple: #bc8cff;
            --accent-amber: #d29922;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
        body {{ background: var(--bg); color: var(--text); padding: 28px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 18px; margin-bottom: 24px; }}
        .header h1 {{ font-size: 22px; color: var(--heading); display: flex; align-items: center; gap: 10px; }}
        .header .live-badge {{ background: rgba(63, 185, 80, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); font-size: 11px; padding: 3px 8px; border-radius: 12px; text-transform: uppercase; font-weight: bold; letter-spacing: 0.5px; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; margin-bottom: 28px; }}
        .kpi-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 18px; }}
        .kpi-title {{ font-size: 12px; text-transform: uppercase; color: #8b949e; letter-spacing: 0.5px; margin-bottom: 6px; }}
        .kpi-value {{ font-size: 26px; font-weight: 700; color: var(--heading); }}
        .kpi-sub {{ font-size: 12px; color: var(--accent); margin-top: 4px; }}
        .section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
        .section-title {{ font-size: 15px; font-weight: 600; color: var(--heading); margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; }}
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
            <div class="kpi-value">{summary.get("total_decisions", 0)}</div>
            <div class="kpi-sub">Arbitrasyon Çatallanması</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Ortalama Gecikme</div>
            <div class="kpi-value">{summary.get("avg_decision_latency_ms", 0.0):.2f} ms</div>
            <div class="kpi-sub">Apple Silicon / 6-Qubit Zeno</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Ortalama Karar Güveni</div>
            <div class="kpi-value">%{summary.get("avg_confidence_pct", 0.0):.1f}</div>
            <div class="kpi-sub">Dopaminerjik Odak Kitlemesi</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Bilinçaltı Kanca Adımları</div>
            <div class="kpi-value">{summary.get("total_hook_steps", 0)}</div>
            <div class="kpi-sub">SWR Replay & CSF Korumalı</div>
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
            <span>Son Hakem Kararları ve Değerlendirmeler</span>
            <span class="mono" style="font-size:12px;">En son 10 karar gösteriliyor</span>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Zaman (UTC)</th>
                    <th>Proje / Workspace</th>
                    <th>Karar Hedefi (Goal)</th>
                    <th>Kazanan Karar (Winner)</th>
                    <th>Rejim</th>
                    <th>Güven</th>
                    <th>Gecikme</th>
                </tr>
            </thead>
            <tbody>
                {rows_joined}
            </tbody>
        </table>
    </div>

    <script>
        let intervalSec = parseInt(localStorage.getItem('quanta_monitor_interval') || '3', 10);
        let isPaused = localStorage.getItem('quanta_monitor_paused') === 'true';
        let remaining = intervalSec;

        const countdownEl = document.getElementById('countdown');
        const pauseBtn = document.getElementById('pause-btn');
        const selectEl = document.getElementById('interval-select');

        if (selectEl) {{
            selectEl.value = intervalSec;
            selectEl.addEventListener('change', (e) => {{
                intervalSec = parseInt(e.target.value, 10);
                localStorage.setItem('quanta_monitor_interval', intervalSec);
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
                localStorage.setItem('quanta_monitor_paused', isPaused);
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
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return target_path
