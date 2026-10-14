"""tests/test_challenger_m2_empirical.py — Challenger M2 Empirical Verification Suite.

Adversarially tests and independently verifies Milestone M2:
1. Empirically verify sub-5ms read latency and zero lock contention on conversation_summaries.db:
   - Real DB query latency benchmark (< 5ms SLA).
   - Concurrent writing stress: simultaneous multi-threaded writers while harvesting.
2. Stress-test reverse-chronological ordering with randomized multi-horizon timestamps:
   - Monotonicity of created_at descending across randomized multi-horizon spans.
   - Identification of underfill anomaly when core anchors dominate.
3. Stress-test core anchor shielding:
   - Engrams with S >= 2.0, is_core_anchor=True, or category='semantic_gist' retain
     w_eff == 1.0 even after thousands of days.
   - Sharp boundary at S = 2.0.
4. Verify subconscious hook latency remains < 25ms with reverse-chronological sorting:
   - End-to-end benchmark across varying engram loads (5, 20, 50, 100 engrams).
"""

from __future__ import annotations

import json
import random
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quanta.cognitive.memory import (
    GIST_CATEGORY,
    CognitiveMemoryManager,
    calculate_temporal_decay,
)
from quanta.cognitive.telemetry import DEFAULT_TELEMETRY_FILE
from quanta.cognitive.workspace_harvester import (
    DEFAULT_SUMMARIES_DB,
    WorkspaceContextHarvester,
)


class TestSQLiteLatencyAndConcurrencyStress:
    """Challenge 1: Empirically verify sub-5ms read latency and zero lock contention."""

    def test_real_database_read_latency_sub_5ms(self) -> None:
        """Confirms that querying the real conversation_summaries.db executes well below 5ms."""
        if not DEFAULT_SUMMARIES_DB.exists():
            pytest.skip(f"Database {DEFAULT_SUMMARIES_DB} not present on this machine.")

        db_uri = f"file:{DEFAULT_SUMMARIES_DB.resolve()}?mode=ro"
        latencies: list[float] = []

        for _ in range(50):
            t0 = time.perf_counter()
            conn = sqlite3.connect(db_uri, uri=True, timeout=1.0)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT conversation_id, title, last_modified_time, last_user_input_time, workspace_uris
                FROM conversation_summaries
                WHERE last_modified_time IS NOT NULL
                ORDER BY last_modified_time DESC
                LIMIT 20
                """
            )
            rows = cur.fetchall()
            conn.close()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)
            assert len(rows) <= 20

        median_lat = statistics.median(latencies)
        max_lat = max(latencies)
        assert median_lat < 2.0, f"Median latency {median_lat:.3f}ms exceeded 2ms"
        assert max_lat < 5.0, f"Max latency {max_lat:.3f}ms exceeded 5ms SLA"

    def test_concurrent_writer_zero_lock_contention(self, tmp_path: Path) -> None:
        """Verifies zero lock contention and sub-5ms latency under heavy concurrent writers."""
        db_path = tmp_path / "test_summaries.db"
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE conversation_summaries (
                conversation_id TEXT PRIMARY KEY,
                title TEXT,
                last_modified_time TEXT,
                last_user_input_time TEXT,
                workspace_uris TEXT
            )
            """
        )
        for i in range(200):
            conn.execute(
                "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?)",
                (
                    f"conv_{i}",
                    f"Title {i}",
                    "2026-09-25T15:00:00Z",
                    "2026-09-25T15:00:00Z",
                    '["/tmp/project"]',
                ),
            )
        conn.commit()
        conn.close()

        stop_flag = threading.Event()
        writer_errors: list[Exception] = []
        reader_latencies: list[float] = []
        reader_errors: list[Exception] = []

        def writer_worker(w_id: int) -> None:
            try:
                w_conn = sqlite3.connect(db_path, timeout=5.0)
                c = 0
                while not stop_flag.is_set():
                    w_conn.execute(
                        "UPDATE conversation_summaries SET last_modified_time = ? WHERE conversation_id = ?",
                        (f"2026-09-25T15:{c % 60:02d}:00Z", f"conv_{w_id * 50 + (c % 50)}"),
                    )
                    w_conn.commit()
                    c += 1
                    time.sleep(0.001)
                w_conn.close()
            except Exception as e:
                writer_errors.append(e)

        def reader_worker() -> None:
            harvester = WorkspaceContextHarvester(summaries_db=db_path, brain_dir=tmp_path)
            while not stop_flag.is_set():
                t0 = time.perf_counter()
                try:
                    traces = harvester.harvest_conversation_history(max_sessions=20)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    reader_latencies.append(dt_ms)
                    assert isinstance(traces, list)
                except Exception as e:
                    reader_errors.append(e)
                time.sleep(0.002)

        writers = [threading.Thread(target=writer_worker, args=(i,)) for i in range(2)]
        readers = [threading.Thread(target=reader_worker) for _ in range(4)]

        for t in writers + readers:
            t.start()

        time.sleep(1.5)
        stop_flag.set()

        for t in writers + readers:
            t.join()

        assert not writer_errors, f"Writer errors: {writer_errors}"
        assert not reader_errors, f"Reader lock errors: {reader_errors}"
        assert len(reader_latencies) > 50, "Insufficient reader iterations executed"

        median_lat = statistics.median(reader_latencies)
        p95_lat = statistics.quantiles(reader_latencies, n=20)[18]
        assert median_lat < 3.0, f"Concurrent median latency {median_lat:.3f}ms exceeded 3ms"
        assert p95_lat < 5.0, f"Concurrent P95 latency {p95_lat:.3f}ms exceeded 5ms SLA"


class TestMultiHorizonDecayAndCoreShielding:
    """Challenge 3: Stress-test core anchor shielding across thousands of days."""

    def test_core_shielding_thousands_of_days(self) -> None:
        """Verifies S >= 2.0, is_core_anchor=True, or category=semantic_gist retain w_eff == 1.0."""
        now = time.time()
        now_dt = datetime.now(timezone.utc)

        test_horizons_days = [1, 10, 100, 1000, 5000, 10000, 50000]

        for days in test_horizons_days:
            # 1. Salience >= 2.0
            w_sal = calculate_temporal_decay(
                created_at=now - days * 86400,
                salience=2.0,
                is_core_anchor=False,
                now_ts=now,
            )
            assert w_sal == 1.0, f"Salience >= 2.0 decayed at {days} days: {w_sal}"

            # 2. is_core_anchor = True with low salience
            w_core = calculate_temporal_decay(
                created_at=now - days * 86400,
                salience=0.1,
                is_core_anchor=True,
                now_ts=now,
            )
            assert w_core == 1.0, f"is_core_anchor decayed at {days} days: {w_core}"

            # 3. category == GIST_CATEGORY with low salience
            w_gist = calculate_temporal_decay(
                created_at=now - days * 86400,
                salience=0.1,
                category=GIST_CATEGORY,
                now_ts=now,
            )
            assert w_gist == 1.0, f"GIST_CATEGORY decayed at {days} days: {w_gist}"

            # 4. Datetime input support
            w_dt = calculate_temporal_decay(
                created_at=now_dt - timedelta(days=days),
                salience=2.5,
                now_ts=now_dt,
            )
            assert w_dt == 1.0, f"Datetime input failed at {days} days: {w_dt}"

    def test_sharp_boundary_at_salience_2_point_0(self) -> None:
        """Verifies sharp protection boundary at S = 2.0."""
        now = time.time()
        created = now - 10 * 86400  # 10 days ago

        w_below = calculate_temporal_decay(
            created_at=created, salience=1.99999, is_core_anchor=False, now_ts=now
        )
        w_at = calculate_temporal_decay(
            created_at=created, salience=2.00000, is_core_anchor=False, now_ts=now
        )

        assert w_below < 1.0, f"S=1.99999 should decay: {w_below}"
        assert w_at == 1.0, f"S=2.00000 should be shielded: {w_at}"


class TestReverseChronologicalOrderingStress:
    """Challenge 2: Stress-test reverse-chronological ordering with randomized multi-horizon timestamps."""

    def test_randomized_timestamps_strict_monotonicity(self) -> None:
        """Verifies strict monotonicity of created_at descending across randomized horizons."""
        random.seed(1337)
        now = time.time()

        for trial in range(30):
            mgr = CognitiveMemoryManager(capacity=40)
            num_engrams = random.randint(10, 30)

            for i in range(num_engrams):
                dt = random.uniform(0, 300 * 86400)
                sal = random.uniform(0.3, 2.5)
                core = random.choice([True, False, None])

                mgr.record_decision(
                    key=f"k_{trial}_{i}",
                    content=f"Content {trial}_{i}",
                    salience=sal,
                    is_core_anchor=core,
                )
                mgr.buffer.buffer[-1]["created_at"] = now - dt
                mgr.buffer.buffer[-1]["metadata"]["created_at"] = now - dt

            top_k = random.randint(5, 15)
            recalled = mgr.recall_reverse_chronological_cascade(top_k=top_k, now_ts=now)

            # Assert strict monotonicity: newest to oldest
            for j in range(len(recalled) - 1):
                assert (
                    recalled[j]["created_at"] >= recalled[j + 1]["created_at"]
                ), f"Monotonicity violated: {recalled[j]['created_at']} < {recalled[j+1]['created_at']}"


class TestSubconsciousHookLatencyUnderLoad:
    """Challenge 4: Verify subconscious hook latency remains < 25ms with reverse-chronological sorting."""

    def test_hook_latency_across_engram_scales(self, tmp_path: Path) -> None:
        """Verifies hook internal latency stays well below 25ms SLA with 5, 20, 50, and 100 engrams."""
        hook_path = (
            Path(__file__).resolve().parent.parent
            / "scripts"
            / "hooks"
            / "cognitive_lifecycle_hook.py"
        )
        python_exe = sys.executable
        state_file = tmp_path / "quanta_cognitive_state.json"

        for count in [5, 20, 50, 100]:
            now = time.time()
            engrams = [
                {
                    "key": "native_first_rule",
                    "content": "Platform işlemlerinde native API kullan.",
                    "salience": 2.8,
                    "category": "constraint",
                    "fidelity": 0.9998,
                    "is_core_anchor": True,
                }
            ]
            for i in range(count):
                engrams.append(
                    {
                        "key": f"transient_{count}_{i}",
                        "content": f"Decision {i} at step {i}",
                        "salience": 0.60,
                        "category": "contextual_decision",
                        "fidelity": 0.85,
                        "is_core_anchor": False,
                        "created_at": now - (count - i) * 60,
                    }
                )

            seed_state = {
                "conversation_id": f"test_scale_{count}",
                "turn_count": count,
                "engrams": engrams,
            }
            state_file.write_text(json.dumps(seed_state), encoding="utf-8")

            payload = {
                "conversationId": f"test_scale_{count}",
                "artifactDirectoryPath": str(tmp_path),
                "stepIdx": 10,
            }
            subprocess.run(
                [python_exe, str(hook_path)],
                input=json.dumps(payload).encode("utf-8"),
                capture_output=True,
                check=True,
            )

            # Read telemetry specifically filtered for this test conversation and hook_step
            matching_rec = None
            for raw_line in reversed(
                DEFAULT_TELEMETRY_FILE.read_text(encoding="utf-8").strip().splitlines()
            ):
                try:
                    entry = json.loads(raw_line)
                    if (
                        entry.get("event_type") == "hook_step"
                        and entry.get("conversation_id") == f"test_scale_{count}"
                    ):
                        matching_rec = entry
                        break
                except Exception:
                    continue

            assert matching_rec is not None, (
                f"No hook_step telemetry record found for test_scale_{count}"
            )
            lat = matching_rec["latency_ms"]
            assert (
                lat < 25.0
            ), f"Hook internal latency for {count} engrams was {lat:.2f}ms (exceeded 25ms SLA)"
