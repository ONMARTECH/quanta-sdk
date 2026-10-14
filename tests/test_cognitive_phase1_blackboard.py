"""Tests for Multi-Agent Shared Cognitive Blackboard (Milestone M2).

Verifies:
1. Concurrency safety: dedicated lockfile (<storage_path>.lock) using fcntl.flock(LOCK_EX).
2. POSIX atomic writes: writes to temporary file, fsync, atomic os.replace.
3. Multi-threading concurrent writes (8 threads x 20 updates = 160 updates) with zero data loss.
4. Multi-processing concurrent writes (4 processes x 15 updates = 60 updates) with zero data loss.
5. Mixed API concurrency (constraints, findings, decisions, heartbeats, entries) under high contention.
6. Zero JSON corruption, valid schema structure, valid monotonic revisioning, valid timestamps.
7. Core anchor immutability protection (PermissionError on unauthorized delete/overwrite).
8. Exception safety and transaction rollback without lock deadlocks or dangling temp files.
"""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from quanta.cognitive.blackboard import CognitiveBlackboard


# Top-level worker function required for multiprocessing 'spawn' context on macOS
def _mp_blackboard_worker(
    storage_path_str: str,
    worker_id: int,
    write_count: int,
    queue: Any,
) -> None:
    """Worker process writing updates with micro-jitter to induce lock contention."""
    try:
        bb = CognitiveBlackboard(storage_path_str)
        written_keys = []
        for step in range(write_count):
            key = f"proc_{worker_id}_step_{step}"
            payload = {
                "worker_id": worker_id,
                "step": step,
                "timestamp": time.time(),
                "message": f"Multiprocess payload from proc {worker_id} at step {step}",
            }
            # Micro-jitter to maximize lock collisions
            time.sleep(random.uniform(0.001, 0.003))
            bb.write_entry(key=key, value=payload, agent_id=f"proc_agent_{worker_id}")
            written_keys.append(key)

            # Periodic read
            if step % 5 == 0:
                _ = bb.read_all_entries()
        queue.put((worker_id, written_keys, None))
    except Exception as exc:
        queue.put((worker_id, [], str(exc)))


class TestCognitiveBlackboard:
    """Test battery verifying CognitiveBlackboard concurrency, atomicity, and schema integrity."""

    def test_multithreading_concurrency_zero_data_loss(self, tmp_path: Path) -> None:
        """Verifies 8 threads executing 20 concurrent updates each (160 total updates).

        Asserts:
        - 160/160 writes succeed with zero data loss.
        - Monotonic revision >= 160.
        - Zero JSON corruption on disk.
        - Valid ISO 8601 timestamps and last_writer field.
        """
        bb_file = tmp_path / "shared_blackboard_mt.json"
        blackboard = CognitiveBlackboard(file_path=bb_file)

        num_threads = 8
        writes_per_thread = 20
        total_expected_entries = num_threads * writes_per_thread

        def thread_worker(thread_id: int) -> list[str]:
            written = []
            local_bb = CognitiveBlackboard(file_path=bb_file)
            for step in range(writes_per_thread):
                key = f"thread_{thread_id}_entry_{step}"
                payload = {
                    "thread_id": thread_id,
                    "step": step,
                    "finding": f"Thread {thread_id} resolved constraint step {step}",
                    "timestamp": time.time(),
                }
                time.sleep(random.uniform(0.001, 0.003))
                local_bb.write_entry(
                    key=key,
                    value=payload,
                    agent_id=f"thread_agent_{thread_id}",
                )
                written.append(key)

                # Intermittent read contention
                if step % 4 == 0:
                    entries = local_bb.read_all_entries()
                    assert isinstance(entries, dict)
            return written

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(thread_worker, i) for i in range(num_threads)]
            all_written_keys = []
            for f in concurrent.futures.as_completed(futures):
                all_written_keys.extend(f.result())

        assert len(all_written_keys) == total_expected_entries

        # Validate final state
        final_entries = blackboard.read_all_entries()
        assert len(final_entries) == total_expected_entries, (
            f"Data loss detected: expected {total_expected_entries}, found {len(final_entries)}"
        )

        for expected_key in all_written_keys:
            assert expected_key in final_entries
            assert "finding" in final_entries[expected_key]

        # Verify raw JSON integrity on disk
        with open(bb_file, encoding="utf-8") as f:
            disk_data = json.load(f)

        assert disk_data["schema_version"] == "1.0"
        assert disk_data["revision"] >= total_expected_entries
        assert "created_at" in disk_data
        assert "updated_at" in disk_data
        # Verify timestamps are parseable ISO strings
        _ = datetime.fromisoformat(disk_data["created_at"])
        _ = datetime.fromisoformat(disk_data["updated_at"])

        # Dedicated lockfile must exist
        lockfile = tmp_path / "shared_blackboard_mt.json.lock"
        assert lockfile.exists()

    def test_multiprocessing_concurrency_zero_data_loss(self, tmp_path: Path) -> None:
        """Verifies 4 processes executing 15 concurrent updates each (60 total updates).

        Asserts:
        - 60/60 writes succeed with zero data loss across isolated processes.
        - Monotonic revision >= 60.
        - Lockfile mediation with fcntl.flock prevents any corruption or overwrites.
        """
        bb_file = tmp_path / "shared_blackboard_mp.json"
        blackboard = CognitiveBlackboard(file_path=bb_file)

        num_processes = 4
        writes_per_process = 15
        total_expected_entries = num_processes * writes_per_process

        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()

        processes = [
            ctx.Process(
                target=_mp_blackboard_worker,
                args=(str(bb_file), p_id, writes_per_process, queue),
            )
            for p_id in range(num_processes)
        ]

        for p in processes:
            p.start()

        results = []
        for _ in range(num_processes):
            results.append(queue.get())

        for p in processes:
            p.join(timeout=15.0)
            assert p.exitcode == 0, f"Process failed with exit code {p.exitcode}"

        all_written_keys = []
        for worker_id, keys, error in results:
            assert error is None, f"Worker {worker_id} encountered exception: {error}"
            assert len(keys) == writes_per_process
            all_written_keys.extend(keys)

        assert len(all_written_keys) == total_expected_entries

        # Verify blackboard content
        final_entries = blackboard.read_all_entries()
        assert len(final_entries) == total_expected_entries

        for expected_key in all_written_keys:
            assert expected_key in final_entries
            assert "message" in final_entries[expected_key]

        with open(bb_file, encoding="utf-8") as f:
            disk_data = json.load(f)

        assert disk_data["revision"] >= total_expected_entries
        assert len(disk_data["entries"]) == total_expected_entries

    def test_mixed_api_concurrency(self, tmp_path: Path) -> None:
        """Tests concurrent execution across distinct blackboard APIs simultaneously.

        Thread 0: Posts operational constraints.
        Thread 1: Posts research findings.
        Thread 2: Posts architectural decisions.
        Thread 3: Updates agent heartbeats.
        Thread 4: Writes arbitrary key-value entries.
        Thread 5: Continuous reader executing get_* and read_* calls.
        """
        bb_file = tmp_path / "shared_blackboard_mixed.json"
        blackboard = CognitiveBlackboard(file_path=bb_file)

        operations_per_worker = 20

        def post_constraints_worker() -> list[str]:
            bb = CognitiveBlackboard(file_path=bb_file)
            ids = []
            for i in range(operations_per_worker):
                time.sleep(random.uniform(0.001, 0.002))
                cid = bb.post_constraint(
                    agent_id="agent_constraints",
                    key=f"constraint_{i}",
                    rule=f"Rule description for constraint {i}",
                    is_hard=(i % 2 == 0),
                    ttl=100.0 if i % 2 == 0 else 0.001,  # half will expire fast
                )
                ids.append(cid)
            return ids

        def post_findings_worker() -> list[str]:
            bb = CognitiveBlackboard(file_path=bb_file)
            ids = []
            for i in range(operations_per_worker):
                time.sleep(random.uniform(0.001, 0.002))
                fid = bb.post_finding(
                    agent_id="agent_findings",
                    title=f"Finding Title {i}",
                    content=f"Detailed finding payload {i}",
                    tags=["quantum", "mps"] if i % 2 == 0 else ["memory", "swr"],
                    salience=1.0 + (i * 0.05),
                )
                ids.append(fid)
            return ids

        def post_decisions_worker() -> list[str]:
            bb = CognitiveBlackboard(file_path=bb_file)
            ids = []
            for i in range(operations_per_worker):
                time.sleep(random.uniform(0.001, 0.002))
                did = bb.post_decision(
                    agent_id="agent_decisions",
                    key=f"decision_{i}",
                    decision=f"Architectural choice {i}",
                    rationale=f"Rationale {i}",
                    status="active" if i % 2 == 0 else "superseded",
                )
                ids.append(did)
            return ids

        def heartbeat_worker() -> int:
            bb = CognitiveBlackboard(file_path=bb_file)
            for i in range(operations_per_worker):
                time.sleep(random.uniform(0.001, 0.002))
                bb.update_heartbeat(
                    agent_id=f"worker_agent_{i % 4}",
                    status="working" if i < operations_per_worker - 1 else "completed",
                    metadata={"iteration": i, "load": random.random()},
                )
            return operations_per_worker

        def entries_worker() -> list[str]:
            bb = CognitiveBlackboard(file_path=bb_file)
            keys = []
            for i in range(operations_per_worker):
                time.sleep(random.uniform(0.001, 0.002))
                k = f"entry_key_{i}"
                bb.write_entry(
                    key=k,
                    value={"data": i},
                    agent_id="agent_entries",
                )
                keys.append(k)
            return keys

        def continuous_reader() -> int:
            bb = CognitiveBlackboard(file_path=bb_file)
            read_count = 0
            for _ in range(operations_per_worker * 2):
                time.sleep(0.001)
                _ = bb.get_constraints(active_only=False)
                _ = bb.get_findings()
                _ = bb.get_decisions(status=None)
                _ = bb.get_heartbeats()
                _ = bb.read_all_entries()
                _ = bb.get_state()
                read_count += 1
            return read_count

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            f_constraints = executor.submit(post_constraints_worker)
            f_findings = executor.submit(post_findings_worker)
            f_decisions = executor.submit(post_decisions_worker)
            f_heartbeats = executor.submit(heartbeat_worker)
            f_entries = executor.submit(entries_worker)
            f_reader = executor.submit(continuous_reader)

            constraint_ids = f_constraints.result()
            finding_ids = f_findings.result()
            decision_ids = f_decisions.result()
            _ = f_heartbeats.result()
            entry_keys = f_entries.result()
            reads_done = f_reader.result()

        assert len(constraint_ids) == operations_per_worker
        assert len(finding_ids) == operations_per_worker
        assert len(decision_ids) == operations_per_worker
        assert len(entry_keys) == operations_per_worker
        assert reads_done > 0

        # Assert final consolidated state
        all_constraints = blackboard.get_constraints(active_only=False)
        assert len(all_constraints) == operations_per_worker

        all_findings = blackboard.get_findings()
        assert len(all_findings) == operations_per_worker

        all_decisions = blackboard.get_decisions(status=None)
        assert len(all_decisions) == operations_per_worker

        heartbeats = blackboard.get_heartbeats()
        assert len(heartbeats) == 4  # agents 0 to 3

        all_entries = blackboard.read_all_entries()
        assert len(all_entries) == operations_per_worker

        # Filtered queries
        mps_findings = blackboard.get_findings(tag="mps")
        assert len(mps_findings) == operations_per_worker // 2

        active_decisions = blackboard.get_decisions(status="active")
        superseded_decisions = blackboard.get_decisions(status="superseded")
        assert len(active_decisions) + len(superseded_decisions) == operations_per_worker

        # Total revisions must account for all operations
        total_ops = operations_per_worker * 5
        assert blackboard.revision >= total_ops

    def test_core_anchor_immutability_protection(self, tmp_path: Path) -> None:
        """Verifies that immutable core anchors cannot be overwritten or deleted without force."""
        bb_file = tmp_path / "shared_blackboard_anchors.json"
        blackboard = CognitiveBlackboard(file_path=bb_file)

        # 1. Write an immutable core anchor
        blackboard.write_entry(
            key="sys_core_anchor_01",
            value={"rule": "zero_leakage", "salience": 2.5},
            is_core_anchor=True,
            agent_id="system_init",
        )

        assert blackboard.read_entry("sys_core_anchor_01")["rule"] == "zero_leakage"

        # 2. Attempt unauthorized delete
        with pytest.raises(PermissionError, match="Cannot overwrite or delete immutable core anchor"):
            blackboard.delete_entry("sys_core_anchor_01", force_override=False)

        # Confirm entry still intact
        assert blackboard.read_entry("sys_core_anchor_01") is not None

        # 3. Attempt unauthorized overwrite
        with pytest.raises(PermissionError, match="Cannot overwrite or delete immutable core anchor"):
            blackboard.write_entry(
                key="sys_core_anchor_01",
                value={"rule": "compromised"},
                force_override=False,
            )

        assert blackboard.read_entry("sys_core_anchor_01")["rule"] == "zero_leakage"

        # 4. Authorized override with force_override=True succeeds
        blackboard.write_entry(
            key="sys_core_anchor_01",
            value={"rule": "updated_with_authorized_permission", "salience": 2.5},
            force_override=True,
        )
        assert (
            blackboard.read_entry("sys_core_anchor_01")["rule"]
            == "updated_with_authorized_permission"
        )

        # Authorized delete succeeds
        deleted = blackboard.delete_entry("sys_core_anchor_01", force_override=True)
        assert deleted is True
        assert blackboard.read_entry("sys_core_anchor_01") is None

    def test_atomic_modify_exception_safety_and_rollback(self, tmp_path: Path) -> None:
        """Verifies atomic_modify exception safety: on failure, state is preserved and lock released."""
        bb_file = tmp_path / "shared_blackboard_safe.json"
        blackboard = CognitiveBlackboard(file_path=bb_file)

        # Seed initial state
        blackboard.write_entry("safe_key", "initial_value")
        rev_before = blackboard.revision

        # Attempt failing modification
        def faulty_modifier(state: dict[str, Any]) -> None:
            state.setdefault("entries", {})["corrupted_key"] = "bad_value"
            raise RuntimeError("Intentional modifier failure midway")

        with pytest.raises(RuntimeError, match="Intentional modifier failure midway"):
            blackboard.atomic_modify(faulty_modifier)

        # Confirm disk state is unmodified
        assert blackboard.revision == rev_before
        assert blackboard.read_entry("safe_key") == "initial_value"
        assert blackboard.read_entry("corrupted_key") is None

        # Confirm lock was released by executing a successful write immediately
        blackboard.write_entry("recovery_key", "success")
        assert blackboard.read_entry("recovery_key") == "success"
        assert blackboard.revision == rev_before + 1

        # Check for zero dangling .tmp files
        tmp_files = list(tmp_path.glob("*.tmp*"))
        assert len(tmp_files) == 0

    def test_ttl_filtering_and_custom_metadata(self, tmp_path: Path) -> None:
        """Verifies TTL expiration on constraints and arbitrary custom metadata."""
        bb_file = tmp_path / "shared_blackboard_ttl.json"
        blackboard = CognitiveBlackboard(file_path=bb_file)

        # Post short-lived constraint (TTL = 0.05s) and permanent constraint
        c_short = blackboard.post_constraint(
            agent_id="test_agent",
            key="ephemeral_lock",
            rule="Lock while syncing",
            ttl=0.05,
        )
        c_perm = blackboard.post_constraint(
            agent_id="test_agent",
            key="permanent_rule",
            rule="Always adhere to integrity",
            ttl=None,
        )

        # Immediately both should be active
        active_c = blackboard.get_constraints(active_only=True)
        active_ids = {c["id"] for c in active_c}
        assert c_short in active_ids
        assert c_perm in active_ids

        # Sleep to expire short-lived constraint
        time.sleep(0.08)

        # Now only permanent should be active
        active_after = blackboard.get_constraints(active_only=True)
        active_after_ids = {c["id"] for c in active_after}
        assert c_short not in active_after_ids
        assert c_perm in active_after_ids

        # But all constraints are visible when active_only=False
        all_c = blackboard.get_constraints(active_only=False)
        assert len(all_c) == 2

        # Test custom metadata
        blackboard.update_custom_metadata(
            {"target_gpu": "cuda:0", "max_batch_size": 128},
            agent_id="orchestrator",
        )
        meta = blackboard.get_custom_metadata()
        assert meta["target_gpu"] == "cuda:0"
        assert meta["max_batch_size"] == 128

        # Test clear
        blackboard.clear()
        assert len(blackboard.get_constraints(active_only=False)) == 0
        assert len(blackboard.read_all_entries()) == 0
        assert blackboard.revision >= 4
        assert blackboard.get_revision() == blackboard.revision

    def test_edge_cases_and_recovery(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies empty files, corrupted JSON files, non-existent key deletion, and default paths."""
        bb_file = tmp_path / "edge_case_blackboard.json"

        # 1. Empty file handling
        bb_file.touch()
        bb = CognitiveBlackboard(file_path=bb_file)
        state = bb.get_state()
        assert state["schema_version"] == "1.0"
        assert state["revision"] == 0

        # 2. Corrupted JSON handling
        bb_file.write_text("NOT_VALID_JSON{::: broken", encoding="utf-8")
        corrupted_state = bb.get_state()
        assert corrupted_state["schema_version"] == "1.0"
        assert corrupted_state["revision"] == 0

        # Non-dict JSON handling
        bb_file.write_text("[1, 2, 3]", encoding="utf-8")
        list_state = bb.get_state()
        assert list_state["schema_version"] == "1.0"

        # 3. Non-existent key deletion returns False
        deleted = bb.delete_entry("ghost_key")
        assert deleted is False

        # 4. Default path initialization
        fake_home = tmp_path / "fake_home"
        monkeypatch.setenv("HOME", str(fake_home))
        default_bb = CognitiveBlackboard(file_path=None)
        assert "shared_blackboard.json" in str(default_bb.storage_path)
        assert default_bb.storage_path.parent.exists()

