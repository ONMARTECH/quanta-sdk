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
        print("   ⚛️ QUANTA BİLİŞSEL DOKPİTİ: İNTERAKTİF HTML DASHBOARD")
        print("=" * 70)
        print(f"  Dosya Yolu: {path}")
        print(f"  Tarayıcıda Açmak İçin:")
        print(f"    open \"{path}\"")
        print("=" * 70)
        return 0

    def _render_once(is_live: bool = False, interval: int = 2) -> None:
        if getattr(args, "dashboard", False):
            generate_dashboard_html(output_path=custom_out)

        summary = get_telemetry_summary()
        events = read_telemetry_events(event_type="decision", limit=args.limit)

        now_str = datetime.now().strftime("%H:%M:%S")
        print("=" * 86)
        if is_live:
            print(f"   ⚛️ QUANTA BİLİŞSEL HAKEM KOKPİTİ [CANLI İZLEME: {interval}s | Son Güncelleme: {now_str}]")
        else:
            print("             ⚛️ QUANTA BİLİŞSEL HAKEM & TELEMETRİ KOKPİTİ")
        print("=" * 86)
        print(f"  Toplam Karar:           {summary['total_decisions']}")
        print(f"  Ortalama Gecikme:       {summary['avg_decision_latency_ms']:.2f} ms")
        print(f"  Ortalama Karar Güveni:  %{summary['avg_confidence_pct']:.1f}")
        print(f"  Bilinçaltı Kanca Adımı: {summary['total_hook_steps']}")
        projects_str = ", ".join(summary["active_workspaces"]) if summary["active_workspaces"] else "N/A"
        print(f"  İzlenen Projeler:       {projects_str}")
        print("-" * 86)

        active_sessions = summary.get("active_sessions", [])
        if active_sessions:
            print("  ⚡ CANLIDA AKTİF ÇALIŞAN PROJELER & SORULAN İSTEKLER:")
            print(f"  {'Proje / Workspace':<25} {'Son Aktivite':<13} {'Durum':<10} {'Son İstek / Yapılan İş':<34}")
            print("  " + "-" * 82)
            for s in active_sessions[:5]:
                ws = s.get("workspace", "General")[:23]
                sec = s.get("seconds_ago", 0)
                sec_str = f"{sec}s önce" if sec < 60 else f"{sec // 60}dk önce"
                status = "🟢 CANLI" if s.get("is_live") else "⚪ BOŞTA"
                query = s.get("last_query") or s.get("winner") or "İşlem yürütülüyor"
                query_str = query[:32]
                print(f"  {ws:<25} {sec_str:<13} {status:<10} {query_str:<34}")
            print("-" * 86)

        if not events:
            print("  Henüz kaydedilmiş hakem kararı bulunmuyor.")
        else:
            header = f"{'Zaman (UTC)':<20} {'Proje / Workspace':<24} {'Kazanan Karar':<25} {'Güven':<8} {'Gecikme':<8}"
            print(header)
            print("-" * 86)
            for ev in reversed(events):
                t_str = ev.get("iso_time", "")[:19]
                ws = ev.get("workspace", "General")[:22]
                winner = ev.get("winner", "")[:24]
                conf = f"%{ev.get('confidence', 0.0) * 100:.1f}"
                lat = f"{ev.get('latency_ms', 0.0):.2f}ms"
                print(f"{t_str:<20} {ws:<24} {winner:<25} {conf:<8} {lat:<8}")
                goal = ev.get("goal", "")
                if goal:
                    print(f"  └─ Hedef: {goal[:75]}")
        print("=" * 86)
        if is_live:
            print(f"  [Canlı Mod Aktif: Her {interval} sn'de bir yenilenir | Çıkmak için Ctrl+C]")

    if watch_interval is not None:
        interval = max(1, int(watch_interval))
        try:
            while True:
                # Clear terminal screen
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.flush()
                _render_once(is_live=True, interval=interval)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n  Canlı izleme durduruldu.\n")
            return 0

    _render_once(is_live=False)
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

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
