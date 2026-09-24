"""Empirical Stress and Concurrency Test Suite for Quanta Subconscious Hook Mirroring.

Verifies:
1. _atomic_write_single_file:
   - High-throughput concurrent multithreaded writes & reads without empty/corrupted JSON.
   - High-throughput multiprocess concurrent writes & reads without empty/corrupted JSON.
   - Zero orphaned .tmp_* files left behind under all conditions (including exceptions and interruptions).
   - Clean handling of un-serializable objects and BaseException.
   - Automatic parent directory creation for deep paths.
2. _save_mirrored_state_atomically:
   - Synchronous dual-target atomic persistence across primary and mirror paths.
   - Hermetic test isolation guarding workspace root when in test environments.
   - Resilient fallback to /tmp when primary target directory is unwritable.
   - Independent fail-safe isolation when mirror target is unwritable (primary still succeeds).
   - Redundancy suppression when primary and mirror paths are identical.
   - Preservation of 53 production engrams when mirroring to workspace root.
"""

from __future__ import annotations

import copy
import json
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from scripts.hooks.quanta_subconscious_hook import (
    QUANTA_ROOT,
    _atomic_write_single_file,
    _save_mirrored_state_atomically,
)


@pytest.fixture
def real_state_path() -> Path:
    p = QUANTA_ROOT / "quanta_cognitive_state.json"
    assert p.exists(), f"quanta_cognitive_state.json not found at {p}"
    return p


@pytest.fixture
def real_state_dict(real_state_path: Path) -> dict[str, Any]:
    with open(real_state_path, encoding="utf-8") as f:
        return json.load(f)


class TestAtomicWriteSingleFileStress:
    """Stress tests and concurrency harness for _atomic_write_single_file."""

    def test_multithreaded_concurrent_writes_and_reads_zero_corruption(
        self, tmp_path: Path
    ) -> None:
        """Simulates rapid concurrent writes from multiple threads while concurrent readers poll."""
        target = tmp_path / "state_threaded.json"
        num_writers = 16
        writes_per_thread = 100

        stop_readers = False
        read_errors: list[str] = []
        read_count = 0
        write_errors: list[str] = []
        lock = threading.Lock()

        def reader_task() -> None:
            nonlocal read_count
            while not stop_readers:
                if target.exists():
                    try:
                        with open(target, encoding="utf-8") as f:
                            content = f.read()
                            if not content:
                                with lock:
                                    read_errors.append("EMPTY_FILE_READ: observed 0-byte file")
                                continue
                            data = json.loads(content)
                            if not isinstance(data, dict) or "thread_id" not in data:
                                with lock:
                                    read_errors.append(f"MALFORMED_DATA: {data}")
                                continue
                            with lock:
                                read_count += 1
                    except json.JSONDecodeError as exc:
                        with lock:
                            read_errors.append(f"JSON_DECODE_ERROR: {exc}")
                    except Exception as exc:
                        with lock:
                            read_errors.append(f"UNEXPECTED_READ_EXCEPTION: {type(exc).__name__}: {exc}")
                time.sleep(0.0002)

        def writer_task(tid: int) -> None:
            for i in range(writes_per_thread):
                state_data = {
                    "thread_id": tid,
                    "iteration": i,
                    "timestamp_ns": time.time_ns(),
                    "turn_count": 1000 + i,
                    "engrams": [
                        {"key": f"key_{tid}_{j}", "fidelity": 0.9998, "salience": 2.0}
                        for j in range(5)
                    ],
                    "payload": "adversarial_stress_padding_" * 50,
                }
                ok = _atomic_write_single_file(target, state_data)
                if not ok:
                    with lock:
                        write_errors.append(f"WRITE_FAILED: tid={tid} iter={i}")

        # Start 4 concurrent reader threads
        readers = [threading.Thread(target=reader_task, name=f"reader-{i}") for i in range(4)]
        for r in readers:
            r.start()

        # Start 16 concurrent writer threads
        with ThreadPoolExecutor(max_workers=num_writers) as executor:
            list(executor.map(writer_task, range(num_writers)))

        stop_readers = True
        for r in readers:
            r.join(timeout=5.0)

        # Assert zero write failures
        assert len(write_errors) == 0, f"Write errors occurred: {write_errors[:10]}"

        # Assert zero read errors / corruptions
        assert len(read_errors) == 0, f"Read errors observed during concurrency: {read_errors[:10]}"
        assert read_count > 0, "Readers performed 0 successful reads"

        # Assert target file exists and is valid
        assert target.exists()
        with open(target, encoding="utf-8") as f:
            final_data = json.load(f)
        assert isinstance(final_data, dict)
        assert "thread_id" in final_data
        assert len(final_data.get("engrams", [])) == 5

        # Assert ZERO orphaned .tmp_* files
        orphans = list(tmp_path.glob(".tmp_*"))
        assert len(orphans) == 0, f"Found orphaned temp files: {orphans}"

    def test_multiprocess_concurrent_subprocess_writes_and_reads(
        self, tmp_path: Path
    ) -> None:
        """Simulates rapid concurrent writes and reads from multiple OS subprocesses."""
        target = tmp_path / "state_multiprocess.json"

        worker_code = """
import sys
import json
import time
from pathlib import Path
from scripts.hooks.quanta_subconscious_hook import _atomic_write_single_file

target_path = Path(sys.argv[1])
worker_id = int(sys.argv[2])
iters = int(sys.argv[3])

for i in range(iters):
    payload = {
        "proc_id": worker_id,
        "iteration": i,
        "timestamp": time.time(),
        "turn_count": 2000 + i,
        "engrams": [
            {"key": f"proc_{worker_id}_rule_{k}", "fidelity": 0.9995}
            for k in range(8)
        ],
        "data": "x" * 1024,
    }
    if not _atomic_write_single_file(target_path, payload):
        sys.stderr.write(f"FAIL_WRITE: proc={worker_id} iter={i}\\n")
        sys.exit(1)
    time.sleep(0.001)

sys.exit(0)
"""
        reader_code = """
import sys
import json
import time
from pathlib import Path

target_path = Path(sys.argv[1])
duration_sec = float(sys.argv[2])
end_time = time.time() + duration_sec
read_count = 0

while time.time() < end_time:
    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                content = f.read()
                if not content:
                    sys.stderr.write("OBSERVED_EMPTY_FILE\\n")
                    sys.exit(2)
                data = json.loads(content)
                if not isinstance(data, dict) or "proc_id" not in data:
                    sys.stderr.write(f"OBSERVED_CORRUPT_DICT: {data}\\n")
                    sys.exit(3)
                read_count += 1
        except json.JSONDecodeError as exc:
            sys.stderr.write(f"OBSERVED_JSON_DECODE_ERROR: {exc}\\n")
            sys.exit(4)
        except Exception as exc:
            sys.stderr.write(f"OBSERVED_EXCEPTION: {type(exc).__name__}: {exc}\\n")
            sys.exit(5)
    time.sleep(0.0005)

if read_count == 0:
    sys.stderr.write("ZERO_READS\\n")
    sys.exit(6)

sys.exit(0)
"""
        writer_script = tmp_path / "proc_writer.py"
        writer_script.write_text(worker_code, encoding="utf-8")

        reader_script = tmp_path / "proc_reader.py"
        reader_script.write_text(reader_code, encoding="utf-8")

        num_procs = 8
        iters_per_proc = 25
        procs: list[subprocess.Popen] = []
        py_bin = sys.executable

        # Start 2 reader subprocesses
        readers: list[subprocess.Popen] = []
        for _ in range(2):
            rp = subprocess.Popen(
                [py_bin, str(reader_script), str(target), "4.0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            readers.append(rp)

        # Start 8 writer subprocesses
        for wid in range(num_procs):
            p = subprocess.Popen(
                [py_bin, str(writer_script), str(target), str(wid), str(iters_per_proc)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            procs.append(p)

        for p in procs:
            out, err = p.communicate(timeout=30)
            assert p.returncode == 0, f"Writer proc failed: {err.decode('utf-8')}"

        for rp in readers:
            out, err = rp.communicate(timeout=30)
            assert rp.returncode == 0, f"Reader proc failed: {err.decode('utf-8')}"

        # Verify state integrity
        with open(target, encoding="utf-8") as f:
            saved = json.load(f)
        assert isinstance(saved, dict)
        assert "proc_id" in saved
        assert len(saved.get("engrams", [])) == 8

        # Verify zero orphaned .tmp_* files
        orphans = list(tmp_path.glob(".tmp_*"))
        assert len(orphans) == 0, f"Orphaned files after multiprocess stress: {orphans}"

    def test_cleanup_on_serialization_exception(self, tmp_path: Path) -> None:
        """Verifies that if json.dump fails, the temp file is cleanly removed and returns False."""
        target = tmp_path / "unserializable_state.json"
        bad_data = {"lambda": lambda x: x, "valid_key": "valid_value"}

        ok = _atomic_write_single_file(target, bad_data)
        assert ok is False
        assert not target.exists()

        orphans = list(tmp_path.glob(".tmp_*"))
        assert len(orphans) == 0, f"Found orphaned temp files after serialization error: {orphans}"

    def test_cleanup_on_base_exception(self, tmp_path: Path) -> None:
        """Verifies that on BaseException (e.g. KeyboardInterrupt), temp file is deleted before re-raise."""
        target = tmp_path / "interrupt_state.json"
        data = {"turn": 1, "status": "active"}

        with (
            patch("json.dump", side_effect=KeyboardInterrupt("Simulated Ctrl+C")),
            pytest.raises(KeyboardInterrupt),
        ):
            _atomic_write_single_file(target, data)

        assert not target.exists()
        orphans = list(tmp_path.glob(".tmp_*"))
        assert len(orphans) == 0, f"Found orphaned temp files after KeyboardInterrupt: {orphans}"

    def test_creates_nonexistent_parent_directories(self, tmp_path: Path) -> None:
        """Verifies automatic creation of deeply nested parent directories."""
        deep_target = tmp_path / "level1" / "level2" / "level3" / "target.json"
        data = {"deep": True, "created": time.time()}

        ok = _atomic_write_single_file(deep_target, data)
        assert ok is True
        assert deep_target.exists()

        with open(deep_target, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["deep"] is True

        orphans = list(deep_target.parent.glob(".tmp_*"))
        assert len(orphans) == 0


class TestSaveMirroredStateAtomically:
    """Empirical verification for _save_mirrored_state_atomically."""

    def test_dual_write_success_and_sync(
        self, tmp_path: Path, real_state_dict: dict[str, Any]
    ) -> None:
        """Verifies dual atomic writing to primary and mirror test paths."""
        primary_dir = tmp_path / "artifact"
        mirror_dir = tmp_path / "workspace_mirror"

        primary_path = primary_dir / "quanta_cognitive_state.json"
        mirror_path = mirror_dir / "quanta_cognitive_state.json"

        test_data = copy.deepcopy(real_state_dict)
        test_data["test_marker"] = "dual_sync_verified"
        test_data["turn_count"] = 42

        prim_ok, mir_ok = _save_mirrored_state_atomically(
            primary_path=primary_path,
            mirror_path=mirror_path,
            state_dict=test_data,
            conv_id="test_dual_conv",
            is_test_env=False,
        )

        assert prim_ok is True
        assert mir_ok is True

        # Assert both files exist
        assert primary_path.exists()
        assert mirror_path.exists()

        # Assert bit-for-bit equivalence
        with open(primary_path, encoding="utf-8") as f1:
            prim_loaded = json.load(f1)
        with open(mirror_path, encoding="utf-8") as f2:
            mir_loaded = json.load(f2)

        assert prim_loaded == mir_loaded
        assert prim_loaded["test_marker"] == "dual_sync_verified"

        # Assert zero temp files left in either directory
        assert len(list(primary_dir.glob(".tmp_*"))) == 0
        assert len(list(mirror_dir.glob(".tmp_*"))) == 0

    def test_hermetic_isolation_guards_workspace_root(
        self, tmp_path: Path, real_state_path: Path, real_state_dict: dict[str, Any]
    ) -> None:
        """Guards real workspace root state file from mutation during test execution."""
        primary_path = tmp_path / "primary" / "state.json"

        initial_mtime = real_state_path.stat().st_mtime_ns
        with open(real_state_path, encoding="utf-8") as f:
            initial_content = f.read()

        mutated_data = copy.deepcopy(real_state_dict)
        mutated_data["MUTATION_ATTEMPT"] = "SHOULD_BE_BLOCKED"

        # 1. With is_test_env=True
        prim_ok, mir_ok = _save_mirrored_state_atomically(
            primary_path=primary_path,
            mirror_path=real_state_path,
            state_dict=mutated_data,
            conv_id="hermetic_test",
            is_test_env=True,
        )

        assert prim_ok is True
        assert mir_ok is False  # Mirroring to root MUST be skipped in test env

        # Verify real state file was NOT modified
        assert real_state_path.stat().st_mtime_ns == initial_mtime
        with open(real_state_path, encoding="utf-8") as f:
            current_content = f.read()
        assert current_content == initial_content
        assert "SHOULD_BE_BLOCKED" not in current_content

    def test_primary_unwritable_fallback_to_tmp(
        self, tmp_path: Path, real_state_dict: dict[str, Any]
    ) -> None:
        """Verifies that if primary_path directory is read-only, it falls back to /tmp."""
        ro_dir = tmp_path / "readonly_dir"
        ro_dir.mkdir()
        ro_primary = ro_dir / "unwritable_primary.json"

        # Make directory read-only
        ro_dir.chmod(stat.S_IREAD | stat.S_IEXEC)

        fallback_conv_id = f"test_fb_{int(time.time_ns())}"
        fallback_expected = Path(f"/tmp/quanta_cognitive_{fallback_conv_id}.json")

        try:
            prim_ok, mir_ok = _save_mirrored_state_atomically(
                primary_path=ro_primary,
                mirror_path=None,
                state_dict=real_state_dict,
                conv_id=fallback_conv_id,
                is_test_env=True,
            )

            assert prim_ok is True
            assert fallback_expected.exists(), f"Expected fallback file {fallback_expected} to exist"

            with open(fallback_expected, encoding="utf-8") as f:
                fb_data = json.load(f)
            assert "turn_count" in fb_data
            assert len(fb_data.get("engrams", [])) >= 53
        finally:
            ro_dir.chmod(stat.S_IRWXU)
            if fallback_expected.exists():
                fallback_expected.unlink()

    def test_mirror_unwritable_does_not_fail_primary(
        self, tmp_path: Path, real_state_dict: dict[str, Any]
    ) -> None:
        """Independent fail-safe: unwritable mirror does not abort or fail primary write."""
        primary_path = tmp_path / "valid_primary" / "state.json"

        ro_dir = tmp_path / "ro_mirror_dir"
        ro_dir.mkdir()
        ro_mirror = ro_dir / "unwritable_mirror.json"
        ro_dir.chmod(stat.S_IREAD | stat.S_IEXEC)

        try:
            prim_ok, mir_ok = _save_mirrored_state_atomically(
                primary_path=primary_path,
                mirror_path=ro_mirror,
                state_dict=real_state_dict,
                conv_id="test_indep_fail",
                is_test_env=False,
            )

            assert prim_ok is True
            assert mir_ok is False
            assert primary_path.exists()
        finally:
            ro_dir.chmod(stat.S_IRWXU)

    def test_identical_primary_and_mirror_avoids_redundant_write(
        self, tmp_path: Path, real_state_dict: dict[str, Any]
    ) -> None:
        """When primary and mirror paths are identical, avoids duplicate write."""
        single_path = tmp_path / "single_state.json"

        prim_ok, mir_ok = _save_mirrored_state_atomically(
            primary_path=single_path,
            mirror_path=single_path,
            state_dict=real_state_dict,
            conv_id="test_identical",
            is_test_env=False,
        )

        assert prim_ok is True
        assert mir_ok is False  # Mirror skipped to prevent duplicate I/O
        assert single_path.exists()

    def test_53_production_engrams_filtering_on_root_mirror(
        self, tmp_path: Path, real_state_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verifies that when mirror is workspace root, only the 53 production engrams are synced."""
        mock_root = tmp_path / "mock_repo"
        mock_root.mkdir()
        mock_root_file = mock_root / "quanta_cognitive_state.json"

        monkeypatch.setattr("scripts.hooks.quanta_subconscious_hook.QUANTA_ROOT", mock_root)

        primary_path = tmp_path / "artifact" / "quanta_cognitive_state.json"

        # Combine real 53 engrams with 10 ephemeral engrams
        mixed_data = copy.deepcopy(real_state_dict)
        prod_engrams = [
            e for e in mixed_data.get("engrams", [])
            if isinstance(e, dict)
            and (
                e.get("category") in ("subconscious_dream", "architecture_rfc")
                or str(e.get("key", "")).startswith("insight_")
                or e.get("key") == "architecture_rfc"
            )
        ]
        assert len(prod_engrams) == 53

        ephemeral_engrams = [
            {
                "key": f"ephemeral_decision_{i}",
                "content": f"Transient note {i}",
                "salience": 0.5,
                "fidelity": 0.85,
                "is_core_anchor": False,
            }
            for i in range(10)
        ]
        mixed_data["engrams"] = prod_engrams + ephemeral_engrams
        assert len(mixed_data["engrams"]) == 63

        # Must ensure is_hermetic_test_env() returns False for this mock root test
        monkeypatch.setattr(
            "scripts.hooks.quanta_subconscious_hook.is_hermetic_test_env",
            lambda payload=None: False,
        )

        prim_ok, mir_ok = _save_mirrored_state_atomically(
            primary_path=primary_path,
            mirror_path=mock_root_file,
            state_dict=mixed_data,
            conv_id="mock_prod_sync",
            is_test_env=False,
        )

        assert prim_ok is True
        assert mir_ok is True

        # Primary must have all 63 engrams
        with open(primary_path, encoding="utf-8") as f:
            prim_saved = json.load(f)
        assert len(prim_saved["engrams"]) == 63

        # Root mirror must strictly preserve the 53 production engrams
        with open(mock_root_file, encoding="utf-8") as f:
            root_saved = json.load(f)
        assert len(root_saved["engrams"]) == 53
        for e in root_saved["engrams"]:
            assert not e["key"].startswith("ephemeral_")


class TestHighConcurrencyDualMirrorStress:
    """Stress tests high concurrency dual mirroring with concurrent readers."""

    def test_rapid_concurrent_dual_mirroring_zero_leak(
        self, tmp_path: Path, real_state_dict: dict[str, Any]
    ) -> None:
        """12 concurrent writer threads syncing to both primary and mirror with 4 continuous readers."""
        primary_target = tmp_path / "primary_stress" / "quanta_cognitive_state.json"
        mirror_target = tmp_path / "mirror_stress" / "quanta_cognitive_state.json"

        num_threads = 12
        rounds_per_thread = 50
        stop = False
        read_errors: list[str] = []
        write_errors: list[str] = []
        successful_reads = 0
        lock = threading.Lock()

        def reader_worker(target: Path, name: str) -> None:
            nonlocal successful_reads
            while not stop:
                if target.exists():
                    try:
                        with open(target, encoding="utf-8") as f:
                            raw = f.read()
                            if not raw:
                                with lock:
                                    read_errors.append(f"{name}: EMPTY_FILE_READ")
                                continue
                            parsed = json.loads(raw)
                            if not isinstance(parsed, dict) or "engrams" not in parsed:
                                with lock:
                                    read_errors.append(f"{name}: CORRUPTED_PAYLOAD")
                                continue
                            with lock:
                                successful_reads += 1
                    except json.JSONDecodeError as err:
                        with lock:
                            read_errors.append(f"{name}: JSONDecodeError: {err}")
                    except Exception as err:
                        with lock:
                            read_errors.append(f"{name}: Exception: {err}")
                time.sleep(0.0003)

        def writer_worker(tid: int) -> None:
            for i in range(rounds_per_thread):
                state_copy = copy.deepcopy(real_state_dict)
                state_copy["writer_thread"] = tid
                state_copy["round"] = i
                state_copy["active_timestamp"] = time.time_ns()

                prim_ok, mir_ok = _save_mirrored_state_atomically(
                    primary_path=primary_target,
                    mirror_path=mirror_target,
                    state_dict=state_copy,
                    conv_id=f"stress_conv_{tid}",
                    is_test_env=False,
                )

                if not prim_ok or not mir_ok:
                    with lock:
                        write_errors.append(f"Dual write failed tid={tid} i={i} prim={prim_ok} mir={mir_ok}")

        readers = [
            threading.Thread(target=reader_worker, args=(primary_target, "primary_reader_1")),
            threading.Thread(target=reader_worker, args=(primary_target, "primary_reader_2")),
            threading.Thread(target=reader_worker, args=(mirror_target, "mirror_reader_1")),
            threading.Thread(target=reader_worker, args=(mirror_target, "mirror_reader_2")),
        ]
        for r in readers:
            r.start()

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            list(pool.map(writer_worker, range(num_threads)))

        stop = True
        for r in readers:
            r.join(timeout=5.0)

        assert len(write_errors) == 0, f"Write errors: {write_errors[:10]}"
        assert len(read_errors) == 0, f"Read errors: {read_errors[:10]}"
        assert successful_reads > 50, f"Low read count: {successful_reads}"

        # Verify zero leftover .tmp_* files
        prim_orphans = list(primary_target.parent.glob(".tmp_*"))
        mir_orphans = list(mirror_target.parent.glob(".tmp_*"))
        assert len(prim_orphans) == 0, f"Leftover temp files in primary dir: {prim_orphans}"
        assert len(mir_orphans) == 0, f"Leftover temp files in mirror dir: {mir_orphans}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
