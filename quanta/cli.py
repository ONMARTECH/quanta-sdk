"""quanta/cli.py — Command-Line Interface for the Quanta SDK.

Supports autonomous biomorphic subconscious mind-wandering daemon commands:
  quanta dream start [--idle-min SEC] [--lambda RATE] [--foreground]
  quanta dream stop
  quanta dream status
  quanta dream inspect [--limit N]
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from quanta.cognitive.daemon import SubconsciousDaemon


def _format_status_table(status: dict[str, Any]) -> str:
    """Formats the daemon status dictionary into a clean CLI display table."""
    lines = [
        "=" * 60,
        "   QUANTA SUBCONSCIOUS MIND-WANDERING DAEMON STATUS",
        "=" * 60,
        f"  Running:               {status.get('running', False)}",
        f"  Process ID (PID):      {status.get('pid', 'N/A')}",
        f"  Hardware Quiescent:    {status.get('system_idle', False)}",
        f"  Total Cycles:          {status.get('total_cycles', 0)}",
        f"  Consolidated Insights: {status.get('consolidated_insights', 0)}",
        f"  Preempted Cycles:      {status.get('preempted_cycles', 0)}",
        f"  Uptime (seconds):      {status.get('uptime_seconds', 0.0)}",
        f"  Quality of Service:    {status.get('qos', 'QOS_CLASS_BACKGROUND')}",
        f"  I/O Policy:            {status.get('io_policy', 'IOPOL_THROTTLE')}",
        f"  PID File:              {status.get('pid_file', '')}",
        f"  State Storage:         {status.get('state_file', '')}",
        "=" * 60,
    ]
    return "\n".join(lines)


def _format_insights_list(insights: list[dict[str, Any]]) -> str:
    """Formats inspected subconscious dream insights for human-readable CLI output."""
    if not insights:
        return "No subconscious dream insights consolidated yet."

    lines = [
        f"Found {len(insights)} consolidated subconscious dream insight(s):",
        "-" * 60,
    ]
    for idx, item in enumerate(insights, start=1):
        topic = item.get("topic", item.get("key", "unnamed"))
        synthesis = item.get("content", item.get("description", ""))
        fid = item.get("fidelity", 1.0)
        salience = item.get("salience", 1.0)
        conf = item.get("confidence", 1.0)
        tags = ", ".join(item.get("tags", []))

        lines.extend([
            f"[{idx}] Topic:      {topic}",
            f"    Synthesis:  {synthesis}",
            f"    Fidelity:   {fid:.4f} ({fid * 100.0:.2f}%)",
            f"    Salience:   {salience:.2f} | Confidence: {conf:.2f}",
            f"    Tags:       [{tags}]",
            "-" * 60,
        ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Constructs the root argument parser and subcommands for the Quanta CLI."""
    parser = argparse.ArgumentParser(
        prog="quanta",
        description="Quanta SDK — AI-native Quantum & Neuromorphic Computing Framework",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Command category to execute")

    # `quanta dream ...`
    dream_parser = subparsers.add_parser(
        "dream",
        help="Autonomous biomorphic subconscious mind-wandering daemon commands",
    )
    dream_subparsers = dream_parser.add_subparsers(
        dest="dream_action",
        help="Subconscious daemon action to perform",
    )

    # `quanta dream start`
    start_parser = dream_subparsers.add_parser("start", help="Start the subconscious daemon")
    start_parser.add_argument(
        "--idle-min",
        type=float,
        default=15.0,
        help="Minimum system idle seconds before dreaming initiates (default: 15.0)",
    )
    start_parser.add_argument(
        "--lambda",
        dest="lambda_rate",
        type=float,
        default=0.1,
        help="Poisson asymptotic spindle burst rate lambda_0 (default: 0.1)",
    )
    start_parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run daemon in the foreground instead of background thread",
    )
    start_parser.add_argument(
        "--pid-file",
        type=str,
        default="quanta_dream.pid",
        help="Path to PID tracking file (default: quanta_dream.pid)",
    )
    start_parser.add_argument(
        "--state-file",
        type=str,
        default="quanta_cognitive_state.json",
        help="Path to persistent state file (default: quanta_cognitive_state.json)",
    )

    # `quanta dream stop`
    stop_parser = dream_subparsers.add_parser("stop", help="Stop the subconscious daemon")
    stop_parser.add_argument(
        "--pid-file",
        type=str,
        default="quanta_dream.pid",
        help="Path to PID tracking file (default: quanta_dream.pid)",
    )
    stop_parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Graceful termination timeout in seconds (default: 5.0)",
    )

    # `quanta dream status`
    status_parser = dream_subparsers.add_parser(
        "status",
        help="Check status of the subconscious daemon",
    )
    status_parser.add_argument(
        "--pid-file",
        type=str,
        default="quanta_dream.pid",
        help="Path to PID tracking file (default: quanta_dream.pid)",
    )
    status_parser.add_argument(
        "--state-file",
        type=str,
        default="quanta_cognitive_state.json",
        help="Path to persistent state file (default: quanta_cognitive_state.json)",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Output status in raw JSON format",
    )

    # `quanta dream inspect`
    inspect_parser = dream_subparsers.add_parser(
        "inspect",
        help="Inspect consolidated dream insights",
    )
    inspect_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of insights to display (default: 10)",
    )
    inspect_parser.add_argument(
        "--state-file",
        type=str,
        default="quanta_cognitive_state.json",
        help="Path to persistent state file (default: quanta_cognitive_state.json)",
    )
    inspect_parser.add_argument(
        "--json",
        action="store_true",
        help="Output insights in raw JSON format",
    )

    # `quanta dream service`
    service_parser = dream_subparsers.add_parser(
        "service",
        help="Manage persistent background macOS LaunchAgent service",
    )
    service_subparsers = service_parser.add_subparsers(
        dest="service_action",
        help="Service action to perform",
    )
    service_subparsers.add_parser("install", help="Install and load macOS LaunchAgent service")
    service_subparsers.add_parser("uninstall", help="Unload and delete macOS LaunchAgent service")
    service_subparsers.add_parser("status", help="Check status of the LaunchAgent service")
    service_subparsers.add_parser("start", help="Start the LaunchAgent service")
    # `quanta monitor ...`
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Real-time telemetry and decision monitoring for Quanta Cognitive Arbiter",
    )
    monitor_parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Number of recent decisions to display (default: 15)",
    )
    monitor_parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Generate and output standalone interactive HTML telemetry dashboard",
    )
    monitor_parser.add_argument(
        "--json",
        action="store_true",
        help="Output telemetry summary in raw JSON format",
    )
    monitor_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Target output file path for the HTML dashboard (default: ~/.gemini/antigravity/telemetry/dashboard.html)",
    )
    monitor_parser.add_argument(
        "--watch",
        "-w",
        nargs="?",
        const=2,
        type=int,
        default=None,
        metavar="SEC",
        help="Live watch mode: continuously refresh output every SEC seconds (default: 2)",
    )

    # `quanta arbitrate ...`
    arbitrate_parser = subparsers.add_parser(
        "arbitrate",
        help="Execute 6-qubit Quantum Decision Arbitration between architectural choices",
    )
    arbitrate_parser.add_argument(
        "--goal",
        "-g",
        type=str,
        required=True,
        help="Goal or architectural objective to optimize for",
    )
    arbitrate_parser.add_argument(
        "--options",
        "-opt",
        type=str,
        required=True,
        help="Semicolon or comma separated list of candidate options/architectures",
    )
    arbitrate_parser.add_argument(
        "--criteria",
        "-c",
        type=str,
        default=None,
        help="Optional criteria or hard constraints to enforce",
    )
    arbitrate_parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Explicit project/workspace name",
    )
    arbitrate_parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw arbitration JSON result",
    )

    return parser


def handle_dream(args: argparse.Namespace) -> int:
    """Executes subconscious mind-wandering daemon CLI commands."""
    action = getattr(args, "dream_action", None)

    if action == "start":
        daemon = SubconsciousDaemon(
            state_file=args.state_file,
            pid_file=args.pid_file,
            idle_min=args.idle_min,
            lambda_0=args.lambda_rate,
        )
        if daemon.is_running():
            print(f"Subconscious daemon is already running (PID: {daemon._read_pid()}).")
            return 0

        if args.foreground:
            pid = os.getpid()
            print(f"Subconscious mind-wandering daemon started in foreground (PID: {pid}).")
            daemon.start(foreground=True)
            return 0
        else:
            cmd = [
                sys.executable,
                "-m",
                "quanta.cli",
                "dream",
                "start",
                "--foreground",
                "--idle-min",
                str(args.idle_min),
                "--lambda",
                str(args.lambda_rate),
                "--pid-file",
                str(args.pid_file),
                "--state-file",
                str(args.state_file),
            ]
            proc = subprocess.Popen(
                cmd,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Wait briefly for detached process to initialize and write its PID
            t_end = time.time() + 2.0
            while time.time() < t_end and not daemon.pid_file.exists():
                time.sleep(0.02)

            pid = daemon._read_pid() or proc.pid
            print(f"Subconscious mind-wandering daemon started in background (PID: {pid}).")
            return 0

    elif action == "stop":
        pid_file = getattr(args, "pid_file", "quanta_dream.pid")
        timeout = getattr(args, "timeout", 5.0)
        daemon = SubconsciousDaemon(pid_file=pid_file)
        if not daemon.is_running():
            print("Subconscious daemon is not running.")
            return 0

        daemon.stop(timeout=timeout)
        print("Subconscious mind-wandering daemon stopped successfully.")
        return 0

    elif action == "status":
        pid_file = getattr(args, "pid_file", "quanta_dream.pid")
        state_file = getattr(args, "state_file", "quanta_cognitive_state.json")
        daemon = SubconsciousDaemon(pid_file=pid_file, state_file=state_file)
        stat = daemon.status()
        if getattr(args, "json", False):
            print(json.dumps(stat, indent=2))
        else:
            print(_format_status_table(stat))
        return 0

    elif action == "inspect":
        limit = getattr(args, "limit", 10)
        state_file = getattr(args, "state_file", "quanta_cognitive_state.json")
        daemon = SubconsciousDaemon(state_file=state_file)
        insights = daemon.inspect(limit=limit)
        if getattr(args, "json", False):
            print(json.dumps(insights, indent=2))
        else:
            print(_format_insights_list(insights))
        return 0

    elif action == "service":
        return handle_service(args)

    else:
        print("Usage: quanta dream {start,stop,status,inspect,service} [options]")
        return 0


LAUNCHAGENT_LABEL = "com.quanta.mindwander"
LAUNCHAGENT_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHAGENT_LABEL}.plist"
DEFAULT_LOG_FILE = Path("/Users/aes/Antigravity Projects/Alfa/quanta/quanta_dream.log")
PYTHON_EXEC = Path("/Users/aes/Antigravity Projects/Alfa/quanta/.venv/bin/python")
WORKSPACE_DIR = Path("/Users/aes/Antigravity Projects/Alfa/quanta")


def handle_service(args: argparse.Namespace) -> int:
    """Manages the background macOS LaunchAgent service for autonomous mind-wandering."""
    action = getattr(args, "service_action", "status")

    if action == "install":
        LAUNCHAGENT_PLIST.parent.mkdir(parents=True, exist_ok=True)
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHAGENT_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{PYTHON_EXEC}</string>
        <string>-m</string>
        <string>quanta.cli</string>
        <string>dream</string>
        <string>start</string>
        <string>--foreground</string>
        <string>--idle-min</string>
        <string>15.0</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{WORKSPACE_DIR}</string>
    <key>StandardOutPath</key>
    <string>{DEFAULT_LOG_FILE}</string>
    <key>StandardErrorPath</key>
    <string>{DEFAULT_LOG_FILE}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ProcessType</key>
    <string>Background</string>
    <key>LowPriorityIO</key>
    <true/>
    <key>Nice</key>
    <integer>20</integer>
</dict>
</plist>
"""
        LAUNCHAGENT_PLIST.write_text(plist_content, encoding="utf-8")
        subprocess.run(["launchctl", "unload", str(LAUNCHAGENT_PLIST)], capture_output=True)
        res = subprocess.run(["launchctl", "load", str(LAUNCHAGENT_PLIST)], capture_output=True, text=True)
        if res.returncode == 0:
            print("🚀 [Servis Kuruldu]: Quanta Subconscious Mind-Wandering LaunchAgent aktif!")
            print(f"   📁 Plist: {LAUNCHAGENT_PLIST}")
            print(f"   📋 Log:   {DEFAULT_LOG_FILE}")
            print("   🌙 Bilgisayar boştayken otomatik rüya görecek ve projeleri zenginleştirecektir.")
            return 0
        else:
            print(f"⚠️ Servis yüklenirken hata: {res.stderr}")
            return 1

    elif action == "uninstall":
        subprocess.run(["launchctl", "unload", str(LAUNCHAGENT_PLIST)], capture_output=True)
        if LAUNCHAGENT_PLIST.exists():
            LAUNCHAGENT_PLIST.unlink()
        print("🛑 [Servis Kaldırıldı]: LaunchAgent servisi durduruldu ve silindi.")
        return 0

    elif action == "start":
        subprocess.run(["launchctl", "start", LAUNCHAGENT_LABEL], capture_output=True)
        print(f"▶️ [Servis Başlatıldı]: {LAUNCHAGENT_LABEL}")
        return 0

    elif action == "stop":
        subprocess.run(["launchctl", "stop", LAUNCHAGENT_LABEL], capture_output=True)
        print(f"⏹️ [Servis Durduruldu]: {LAUNCHAGENT_LABEL}")
        return 0

    elif action == "logs":
        if DEFAULT_LOG_FILE.exists():
            lines = DEFAULT_LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
            print("\n".join(lines[-40:]))
        else:
            print("Log dosyası henüz oluşmadı.")
        return 0

    elif action == "status":
        res = subprocess.run(["launchctl", "list"], capture_output=True, text=True)
        is_loaded = LAUNCHAGENT_LABEL in res.stdout
        print("=" * 60)
        print("   QUANTA SUBCONSCIOUS LAUNCHAGENT SERVICE STATUS")
        print("=" * 60)
        print(f"  Service Label:       {LAUNCHAGENT_LABEL}")
        print(f"  Loaded in launchd:   {'Evet (Aktif)' if is_loaded else 'Hayır'}")
        print(f"  Plist File:          {LAUNCHAGENT_PLIST} ({'Var' if LAUNCHAGENT_PLIST.exists() else 'Yok'})")
        print(f"  Log File:            {DEFAULT_LOG_FILE}")
        if DEFAULT_LOG_FILE.exists():
            print("\n  Son Log Çıktıları:")
            lines = DEFAULT_LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[-8:]:
                print(f"    {line}")
        print("=" * 60)
        return 0

    return 0


def handle_monitor(args: argparse.Namespace) -> int:
    """Displays real-time cognitive arbiter decisions, latency, and telemetry audit."""
    from quanta.cognitive.telemetry import (
        generate_dashboard_html,
        get_telemetry_summary,
        read_telemetry_events,
        _safe_float,
        _safe_int,
    )

    watch_interval = getattr(args, "watch", None)
    custom_out = Path(args.output).resolve() if getattr(args, "output", None) else None

    # Single-shot JSON export
    if getattr(args, "json", False) and watch_interval is None:
        summary = get_telemetry_summary()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    # Single-shot Dashboard export (without watch)
    if getattr(args, "dashboard", False) and watch_interval is None:
        path = generate_dashboard_html(output_path=custom_out)
        print("=" * 70)
        print("   ⚛️ QUANTA BİLİŞSEL KOKPİTİ: İNTERAKTİF HTML DASHBOARD")
        print("=" * 70)
        print(f"  Dosya Yolu: {path}")
        print(f"  Tarayıcıda Açmak İçin:")
        print(f"    open \"{path}\"")
        print("=" * 70)
        return 0

    CYAN = "\033[1;36m"
    GREEN = "\033[1;32m"
    PURPLE = "\033[1;35m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[1;34m"
    WHITE = "\033[1;37m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    def _render_once(is_live: bool = False, interval: int = 2) -> None:
        if getattr(args, "dashboard", False):
            generate_dashboard_html(output_path=custom_out)

        summary = get_telemetry_summary()
        if getattr(args, "json", False):
            if is_live:
                print(json.dumps(summary, ensure_ascii=False))
                sys.stdout.flush()
            else:
                print(json.dumps(summary, indent=2, ensure_ascii=False))
            return

        limit_val = getattr(args, "limit", 15) or 15
        events = read_telemetry_events(event_type="decision", limit=limit_val) or []

        now_str = datetime.now().strftime("%H:%M:%S")
        print("=" * 88)
        if is_live:
            print(f"   {CYAN}⚛️ QUANTA BİLİŞSEL HAKEM KOKPİTİ{RESET} [{GREEN}CANLI İZLEME: {interval}s{RESET} | {GRAY}Son Güncelleme: {now_str}{RESET}]")
        else:
            print(f"             {CYAN}⚛️ QUANTA BİLİŞSEL HAKEM & TELEMETRİ KOKPİTİ{RESET}")
        print("=" * 88)

        total_dec = _safe_int(summary.get("total_decisions"), 0)
        avg_lat = _safe_float(summary.get("avg_decision_latency_ms"), 0.0)
        avg_conf = _safe_float(summary.get("avg_confidence_pct"), 0.0)
        total_hooks = _safe_int(summary.get("total_hook_steps"), 0)
        active_engrams = _safe_int(summary.get("total_active_engrams"), 0)
        mean_zeno = _safe_float(summary.get("mean_zeno_pinning"), 0.842) * 100.0

        print(f"  {BOLD}Toplam Karar:{RESET}           {CYAN}{total_dec:<6}{RESET} │ {BOLD}Ortalama Gecikme:{RESET}       {WHITE}{avg_lat:.2f} ms{RESET}")
        print(f"  {BOLD}Ortalama Kuantum Güveni:{RESET} %{GREEN}{avg_conf:<5.1f}{RESET} │ {BOLD}Zeno Odak Kitlemesi:{RESET}   %{PURPLE}{mean_zeno:.1f}{RESET}")
        print(f"  {BOLD}SWR Replay Adımları:{RESET}    {YELLOW}{total_hooks:<6}{RESET} │ {BOLD}Aktif Engram Sayısı:{RESET}   {GREEN}{active_engrams}{RESET}")
        workspaces_list = summary.get("active_workspaces", [])
        projects_str = ", ".join(str(w) for w in workspaces_list) if workspaces_list else "N/A"
        print(f"  {BOLD}İzlenen Projeler:{RESET}       {BLUE}{projects_str}{RESET}")
        print("-" * 88)

        # Panel 1: Subconscious Rule Guardian
        print(f"  {PURPLE}🧠 BİLİNÇALTI KURAL MUHAFIZLIĞI (SWR REPLAY & ENGRAL SADAKATİ){RESET}")
        pruned_cnt = _safe_int(summary.get("total_pruned_engrams"), 0)
        print(f"  {GRAY}[Lindblad Kalkanı: κ_csf = 1/6250 | Mikroglial Budama: {pruned_cnt} engram elendi | Aktif: {active_engrams}]{RESET}")
        print(f"  {'Kural Adı':<28} {'Sadakat':<15} {'Durum':<11} {'Kategori':<12} {'Önem':<6} {'Son Replay'}")
        print("  " + "-" * 84)

        latest_rules = summary.get("latest_rules", [])
        if not latest_rules:
            print(f"  {GRAY}Aktif kurallar henüz kaydedilmedi.{RESET}")
        else:
            for r in latest_rules[:6]:
                if not isinstance(r, dict):
                    continue
                r_name = str(r.get("name", "unnamed"))[:26]
                r_fid = _safe_float(r.get("fidelity"), 0.9998)
                r_pct = _safe_float(r.get("fidelity_pct"), round(r_fid * 100.0, 2))
                r_cat = str(r.get("category", "constraint"))[:10]
                r_sal = _safe_float(r.get("salience"), 1.0)
                r_time = str(r.get("last_replayed", ""))[-12:]

                if r_fid >= 0.95:
                    fid_col = GREEN
                    stat_str = f"{GREEN}PRISTINE{RESET}"
                elif r_fid >= 0.80:
                    fid_col = CYAN
                    stat_str = f"{CYAN}ACTIVE{RESET}  "
                else:
                    fid_col = YELLOW
                    stat_str = f"{YELLOW}DECAYING{RESET}"

                fid_val_str = "99.98%" if r_pct >= 99.98 else f"{r_pct:.2f}%"
                r_pct_clamped = max(0.0, min(100.0, r_pct))
                bar_len = min(8, max(1, int(r_pct_clamped / 12.5)))
                bar_str = f"[{'█' * bar_len}{' ' * (8 - bar_len)}]"
                print(f"  {WHITE}{r_name:<28}{RESET} {fid_col}{fid_val_str:<7} {bar_str}{RESET} {stat_str} {GRAY}{r_cat:<12}{RESET} {r_sal:<6.1f} {r_time}")
        print("-" * 88)

        # Panel 2: Active Projects
        active_sessions = summary.get("active_sessions", [])
        if active_sessions:
            print(f"  {YELLOW}⚡ CANLIDA AKTİF PROJELER & SORULAR{RESET}")
            print(f"  {'Proje / Workspace':<25} {'Son Aktivite':<13} {'Durum':<10} {'Son İstek / Yapılan İş':<34}")
            print("  " + "-" * 84)
            for s in active_sessions[:5]:
                if not isinstance(s, dict):
                    continue
                ws = str(s.get("workspace", "General"))[:23]
                sec = _safe_int(s.get("seconds_ago"), 0)
                sec_str = f"{sec}s önce" if sec < 60 else f"{sec // 60}dk önce"
                status = f"{GREEN}🟢 CANLI{RESET}" if s.get("is_live") else f"{GRAY}⚪ BOŞTA{RESET}"
                query = str(s.get("last_query") or s.get("winner") or "İşlem yürütülüyor")
                query_str = query[:32]
                print(f"  {BLUE}{ws:<25}{RESET} {sec_str:<13} {status:<10} {WHITE}{query_str:<34}{RESET}")
            print("-" * 88)

        # Panel 3: Quantum Decisions
        print(f"  {CYAN}⚛️ 6-QUBIT KUANTUM KARARLARI & ZENO KİTLEMESİ{RESET}")
        if not events:
            print(f"  {GRAY}Henüz kaydedilmiş hakem kararı bulunmuyor.{RESET}")
        else:
            header = f"  {'Zaman (UTC)':<18} {'Proje / Workspace':<22} {'Kazanan Karar':<25} {'Güven':<8} {'P_zeno':<8} {'Gecikme':<8}"
            print(header)
            print("  " + "-" * 84)
            for ev in reversed(events):
                if not isinstance(ev, dict):
                    continue
                t_str = str(ev.get("iso_time", ""))[:17]
                ws = str(ev.get("workspace", "General"))[:20]
                winner = str(ev.get("winner", ""))[:24]
                conf_val = _safe_float(ev.get("confidence"), 0.0) * 100.0
                conf = f"%{conf_val:.1f}"
                zeno_val = _safe_float(ev.get("zeno_pinning_factor"), 0.0)
                zeno_p = f"{zeno_val:.3f}"
                lat_val = _safe_float(ev.get("latency_ms"), 0.0)
                lat = f"{lat_val:.2f}ms"
                print(f"  {t_str:<18} {BLUE}{ws:<22}{RESET} {GREEN}{winner:<25}{RESET} {conf:<8} {PURPLE}{zeno_p:<8}{RESET} {lat:<8}")
                goal = ev.get("goal", "")
                if goal:
                    print(f"    {GRAY}└─ Hedef: {str(goal)[:72]}{RESET}")
                ranking = ev.get("ranking", [])
                if isinstance(ranking, list) and ranking:
                    rank_str_parts = []
                    for r in ranking[:3]:
                        if not isinstance(r, dict):
                            continue
                        r_opt = str(r.get("option", ""))[:20]
                        r_sc = _safe_float(r.get("score"), 0.0) * 100.0
                        r_tr = _safe_float(r.get("tr_rho_pi"), 0.0)
                        star = " ★" if r_opt == winner[:20] else ""
                        rank_str_parts.append(f"{r_opt} (P={r_sc:.1f}%, Tr={r_tr:.4f}){star}")
                    if rank_str_parts:
                        print(f"    {CYAN}└─ Kuantum Sıralaması: {', '.join(rank_str_parts)}{RESET}")
        print("=" * 88)
        if is_live:
            print(f"  [Canlı Mod Aktif: Her {interval} sn'de bir yenilenir | Çıkmak için Ctrl+C]")

    if watch_interval is not None:
        interval = max(1, int(watch_interval))
        try:
            while True:
                if sys.stdout.isatty() and not getattr(args, "json", False):
                    # Clear terminal screen only for interactive TTY non-JSON output
                    sys.stdout.write("\033[2J\033[H")
                    sys.stdout.flush()
                try:
                    _render_once(is_live=True, interval=interval)
                except Exception as e:
                    print(f"  {YELLOW}[İzleme]: Veri okuma sırasında geçici hata (yeniden denenecek): {e}{RESET}")
                time.sleep(interval)
        except (KeyboardInterrupt, SystemExit):
            if not getattr(args, "json", False):
                sys.stdout.write(f"\n{RESET}  Canlı izleme durduruldu.\n")
                sys.stdout.flush()
            return 0

    _render_once(is_live=False)
    return 0


def handle_arbitrate(args: argparse.Namespace) -> int:
    """Executes 6-qubit Quantum Decision Arbitration and records telemetry."""
    from quanta.cognitive.arbiter import QuantumDecisionArbiter
    from quanta.cognitive.telemetry import detect_workspace

    goal = args.goal
    raw_opts = args.options
    if ";" in raw_opts:
        options = [o.strip() for o in raw_opts.split(";") if o.strip()]
    else:
        options = [o.strip() for o in raw_opts.split(",") if o.strip()]

    if len(options) < 2:
        print("Hata: Arbitrasyon için en az 2 seçenek belirtilmelidir.")
        return 1

    criteria = [c.strip() for c in args.criteria.split(";")] if args.criteria else None
    workspace = detect_workspace(args.workspace)

    arbiter = QuantumDecisionArbiter()
    result = arbiter.arbitrate(
        options=options,
        goal=goal,
        criteria=criteria,
        workspace=workspace,
        log_telemetry=True,
    )

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    winner = result["recommended_option"]
    conf = result["confidence"] * 100.0
    print("=" * 70)
    print("       ⚛️ QUANTA BİLİŞSEL KARAR HAKEMİ (6-QUBIT ZENO ARBITRATION)")
    print("=" * 70)
    print(f"  Proje / Workspace: {workspace}")
    print(f"  Karar Hedefi:      {goal}")
    print(f"  Kazanan Seçenek:   {winner}")
    print(f"  Karar Güveni:      %{conf:.1f}")
    print(f"  Dinamik Rejim:     {result['regime']}")
    print(f"  Gecikme:           {result['latency_ms']:.2f} ms")
    if result.get("ranked_options"):
        print("-" * 70)
        print("  Sıralama ve Olasılık Dağılımı:")
        for idx, r in enumerate(result["ranked_options"], start=1):
            star = " ★" if r["option"] == winner else ""
            score_pct = r.get("probability", r.get("score", 0.0)) * 100.0
            print(f"    {idx}. {r['option']:<35} (P={score_pct:.1f}%){star}")
    print("=" * 70)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point for the Quanta SDK."""
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)
    if args.subcommand == "dream":
        return handle_dream(args)
    elif args.subcommand == "monitor":
        return handle_monitor(args)
    elif args.subcommand == "arbitrate":
        return handle_arbitrate(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
