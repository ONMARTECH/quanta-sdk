"""quanta.cognitive.blackboard — Concurrency-Safe Multi-Agent Cognitive Blackboard.

Provides an atomic, concurrency-safe shared substrate for orchestrator and subagents
to exchange ephemeral constraints, findings, architectural decisions, and heartbeats
without race conditions, file corruption, or credential vault exposure.

Concurrency Safety Architecture:
1. Dedicated lockfile `<storage_path>.lock` using kernel advisory locking (`fcntl.flock` with `LOCK_EX`).
2. In-process `threading.RLock` to guarantee reentrancy and thread-level mutual exclusion.
3. POSIX atomic writes: writes payload to unique temporary file in the same directory,
   forces disk sync via `file.flush()` + `os.fsync()`, and executes atomic rename via `os.replace()`.
4. Monotonic revisioning and structured JSON schema for constraints, findings, decisions,
   heartbeats, and custom metadata.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

DEFAULT_BLACKBOARD_DIR = Path(os.path.expanduser("~/.gemini/antigravity/blackboard"))
DEFAULT_BLACKBOARD_FILE = DEFAULT_BLACKBOARD_DIR / "shared_blackboard.json"


class CognitiveBlackboard:
    """Concurrency-safe shared cognitive blackboard for multi-agent coordination.

    Employs dedicated lockfiles (`.lock`), kernel-level `fcntl.flock(LOCK_EX)`,
    in-process reentrant locks, and POSIX-atomic temporary file replacement (`os.replace`)
    with `os.fsync` disk synchronizations.
    """

    _process_locks: dict[str, threading.RLock] = {}
    _process_locks_guard = threading.Lock()

    def __init__(self, file_path: str | Path | None = None) -> None:
        """Initializes the blackboard with the designated persistent storage file.

        Args:
            file_path: Optional custom path to JSON storage. Defaults to
                `~/.gemini/antigravity/blackboard/shared_blackboard.json`.
        """
        if file_path is None:
            self.storage_path = DEFAULT_BLACKBOARD_FILE.resolve()
        else:
            self.storage_path = Path(file_path).resolve()

        self.lock_path = Path(str(self.storage_path) + ".lock")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _get_process_lock(cls, resolved_path_str: str) -> threading.RLock:
        """Retrieves or creates a thread-safe reentrant lock for the given canonical file path."""
        with cls._process_locks_guard:
            if resolved_path_str not in cls._process_locks:
                cls._process_locks[resolved_path_str] = threading.RLock()
            return cls._process_locks[resolved_path_str]

    @contextlib.contextmanager
    def _acquire_lock(self, exclusive: bool = True):
        """Context manager acquiring both in-process RLock and kernel fcntl.flock on lockfile."""
        proc_lock = self._get_process_lock(str(self.storage_path))
        with proc_lock:
            if _HAS_FCNTL:
                lock_fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o666)
                try:
                    lock_flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                    fcntl.flock(lock_fd, lock_flags)
                    yield
                finally:
                    with contextlib.suppress(OSError):
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    with contextlib.suppress(OSError):
                        os.close(lock_fd)
            else:
                yield

    def _create_initial_state(self) -> dict[str, Any]:
        """Returns the canonical structured JSON schema template for a new blackboard."""
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "schema_version": "1.0",
            "revision": 0,
            "created_at": now_iso,
            "updated_at": now_iso,
            "last_writer": "system",
            "constraints": [],
            "findings": [],
            "decisions": [],
            "agent_heartbeats": {},
            "entries": {},
            "core_anchors": {},
            "custom_metadata": {},
        }

    def _read_state_unlocked(self) -> dict[str, Any]:
        """Reads JSON state from disk without lock acquisition (caller must hold lock)."""
        if not self.storage_path.exists():
            return self._create_initial_state()

        try:
            if self.storage_path.stat().st_size == 0:
                return self._create_initial_state()
            with open(self.storage_path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return self._create_initial_state()
        except (json.JSONDecodeError, OSError):
            return self._create_initial_state()

        # Enforce canonical schema defaults
        data.setdefault("schema_version", "1.0")
        data.setdefault("revision", 0)
        data.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        data.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
        data.setdefault("last_writer", "system")
        data.setdefault("constraints", [])
        data.setdefault("findings", [])
        data.setdefault("decisions", [])
        data.setdefault("agent_heartbeats", {})
        data.setdefault("entries", {})
        data.setdefault("core_anchors", {})
        data.setdefault("custom_metadata", {})
        return data

    def _write_state_unlocked(self, state: dict[str, Any]) -> None:
        """Atomically writes JSON state to storage_path via tmp file + fsync + os.replace."""
        tmp_file = self.storage_path.with_name(
            f"{self.storage_path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
        )
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, self.storage_path)
        finally:
            if tmp_file.exists():
                with contextlib.suppress(OSError):
                    tmp_file.unlink()

    def atomic_modify(self, modifier: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
        """Applies an in-place mutation to the blackboard state under an exclusive lock.

        Increments monotonic `revision`, updates `updated_at`, and commits via POSIX
        atomic rename with `os.fsync`. Releases lock cleanly even if modifier raises.

        Args:
            modifier: Callable taking the mutable dictionary state.

        Returns:
            A deep copy of the committed blackboard state.
        """
        with self._acquire_lock(exclusive=True):
            state = self._read_state_unlocked()
            modifier(state)
            state["revision"] = int(state.get("revision", 0)) + 1
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_state_unlocked(state)
            return copy.deepcopy(state)

    def get_state(self) -> dict[str, Any]:
        """Returns a snapshot of the current blackboard state under shared/exclusive lock."""
        with self._acquire_lock(exclusive=False):
            return copy.deepcopy(self._read_state_unlocked())

    @property
    def revision(self) -> int:
        """Returns current monotonic revision number."""
        with self._acquire_lock(exclusive=False):
            return int(self._read_state_unlocked().get("revision", 0))

    def get_revision(self) -> int:
        """Returns current monotonic revision number."""
        return self.revision

    # -------------------------------------------------------------------------
    # Structured Domain API: Constraints, Findings, Decisions, Heartbeats
    # -------------------------------------------------------------------------

    def post_constraint(
        self,
        agent_id: str,
        key: str,
        rule: str,
        is_hard: bool = True,
        ttl: float | None = None,
    ) -> str:
        """Posts an operational constraint to the shared blackboard.

        Args:
            agent_id: Identifier of the posting agent.
            key: Semantic identifier of the constraint.
            rule: Constraint description or rule text.
            is_hard: Whether the constraint is inviolable (hard) or advisory.
            ttl: Time-to-live in seconds. If None, constraint persists indefinitely.

        Returns:
            The unique constraint ID string.
        """
        cid = f"c_{uuid.uuid4().hex[:8]}"
        now = time.time()
        expires_at = (now + float(ttl)) if ttl is not None else None
        item = {
            "id": cid,
            "key": key,
            "rule": rule,
            "agent_id": agent_id,
            "is_hard": bool(is_hard),
            "created_at": now,
            "expires_at": expires_at,
        }

        def _mutate(state: dict[str, Any]) -> None:
            state.setdefault("constraints", []).append(item)
            state["last_writer"] = agent_id

        self.atomic_modify(_mutate)
        return cid

    def get_constraints(self, active_only: bool = True) -> list[dict[str, Any]]:
        """Retrieves constraints from the blackboard.

        Args:
            active_only: If True, filters out constraints whose TTL has expired.

        Returns:
            List of constraint dictionaries.
        """
        with self._acquire_lock(exclusive=False):
            state = self._read_state_unlocked()
            items = state.get("constraints", [])
            now = time.time()
            if active_only:
                return [
                    copy.deepcopy(c)
                    for c in items
                    if c.get("expires_at") is None or c["expires_at"] >= now
                ]
            return [copy.deepcopy(c) for c in items]

    def post_finding(
        self,
        agent_id: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        salience: float = 1.0,
    ) -> str:
        """Posts a research, audit, or exploration finding to the shared blackboard.

        Args:
            agent_id: Identifier of the posting agent.
            title: Short descriptive title of the finding.
            content: Detailed finding description or data.
            tags: Optional list of semantic topic tags.
            salience: Cognitive importance weight (default 1.0).

        Returns:
            The unique finding ID string.
        """
        fid = f"f_{uuid.uuid4().hex[:8]}"
        item = {
            "id": fid,
            "title": title,
            "content": content,
            "agent_id": agent_id,
            "tags": list(tags) if tags else [],
            "salience": float(salience),
            "created_at": time.time(),
        }

        def _mutate(state: dict[str, Any]) -> None:
            state.setdefault("findings", []).append(item)
            state["last_writer"] = agent_id

        self.atomic_modify(_mutate)
        return fid

    def get_findings(self, tag: str | None = None) -> list[dict[str, Any]]:
        """Retrieves findings, optionally filtered by tag.

        Args:
            tag: If specified, returns only findings containing this tag.

        Returns:
            List of finding dictionaries.
        """
        with self._acquire_lock(exclusive=False):
            state = self._read_state_unlocked()
            items = state.get("findings", [])
            if tag is not None:
                return [copy.deepcopy(f) for f in items if tag in f.get("tags", [])]
            return [copy.deepcopy(f) for f in items]

    def post_decision(
        self,
        agent_id: str,
        key: str,
        decision: str,
        rationale: str = "",
        status: str = "active",
    ) -> str:
        """Posts an architectural or implementation decision to the shared blackboard.

        Args:
            agent_id: Identifier of the deciding agent.
            key: Decision topic or identifier.
            decision: Chosen direction or decision text.
            rationale: Context or reason for the choice.
            status: Status string (e.g. "active", "superseded", "revoked").

        Returns:
            The unique decision ID string.
        """
        did = f"d_{uuid.uuid4().hex[:8]}"
        item = {
            "id": did,
            "key": key,
            "decision": decision,
            "rationale": rationale,
            "agent_id": agent_id,
            "status": status,
            "created_at": time.time(),
        }

        def _mutate(state: dict[str, Any]) -> None:
            state.setdefault("decisions", []).append(item)
            state["last_writer"] = agent_id

        self.atomic_modify(_mutate)
        return did

    def get_decisions(self, status: str | None = "active") -> list[dict[str, Any]]:
        """Retrieves decisions, optionally filtered by status.

        Args:
            status: If specified, returns only decisions matching this status.
                Pass None to retrieve all decisions regardless of status.

        Returns:
            List of decision dictionaries.
        """
        with self._acquire_lock(exclusive=False):
            state = self._read_state_unlocked()
            items = state.get("decisions", [])
            if status is not None:
                return [copy.deepcopy(d) for d in items if d.get("status") == status]
            return [copy.deepcopy(d) for d in items]

    def update_heartbeat(
        self,
        agent_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Updates or registers an agent liveness heartbeat on the blackboard.

        Args:
            agent_id: Unique agent identifier.
            status: Liveness status (e.g. 'working', 'idle', 'completed', 'blocked').
            metadata: Optional arbitrary telemetry dictionary.
        """
        heartbeat = {
            "status": status,
            "last_seen": time.time(),
            "metadata": metadata or {},
        }

        def _mutate(state: dict[str, Any]) -> None:
            state.setdefault("agent_heartbeats", {})[agent_id] = heartbeat
            state["last_writer"] = agent_id

        self.atomic_modify(_mutate)

    def get_heartbeats(self) -> dict[str, dict[str, Any]]:
        """Retrieves all registered agent heartbeats."""
        with self._acquire_lock(exclusive=False):
            state = self._read_state_unlocked()
            return copy.deepcopy(state.get("agent_heartbeats", {}))

    # -------------------------------------------------------------------------
    # Key-Value Entry & Core Anchor Protection API
    # -------------------------------------------------------------------------

    def write_entry(
        self,
        key: str,
        value: Any,
        agent_id: str = "anonymous",
        is_core_anchor: bool = False,
        force_override: bool = False,
    ) -> None:
        """Writes or updates an arbitrary key-value entry on the shared blackboard.

        If the entry is already tagged as an immutable core anchor, modifications
        without `force_override=True` will raise `PermissionError`.

        Args:
            key: Entry identifier key.
            value: JSON-serializable value payload.
            agent_id: Identifier of writer.
            is_core_anchor: If True, marks entry as immutable core anchor.
            force_override: If True, permits modifying existing core anchor.
        """
        def _mutate(state: dict[str, Any]) -> None:
            if state.get("core_anchors", {}).get(key) and not force_override:
                raise PermissionError(
                    f"Cannot overwrite or delete immutable core anchor: '{key}'"
                )
            state.setdefault("entries", {})[key] = value
            if is_core_anchor:
                state.setdefault("core_anchors", {})[key] = True
            state["last_writer"] = agent_id

        self.atomic_modify(_mutate)

    def read_entry(self, key: str, default: Any = None) -> Any:
        """Reads a specific key from entries dictionary, returning default if absent."""
        with self._acquire_lock(exclusive=False):
            state = self._read_state_unlocked()
            val = state.get("entries", {}).get(key, default)
            return copy.deepcopy(val)

    def read_all_entries(self) -> dict[str, Any]:
        """Returns a copy of all current key-value entries."""
        with self._acquire_lock(exclusive=False):
            state = self._read_state_unlocked()
            return copy.deepcopy(state.get("entries", {}))

    def delete_entry(self, key: str, force_override: bool = False) -> bool:
        """Deletes a key-value entry from the blackboard.

        Raises:
            PermissionError: If entry is an immutable core anchor and `force_override` is False.

        Returns:
            True if the key was deleted, False if key did not exist.
        """
        deleted = False

        def _mutate(state: dict[str, Any]) -> None:
            nonlocal deleted
            if state.get("core_anchors", {}).get(key) and not force_override:
                raise PermissionError(
                    f"Cannot overwrite or delete immutable core anchor: '{key}'"
                )
            if key in state.get("entries", {}):
                del state["entries"][key]
                state.get("core_anchors", {}).pop(key, None)
                deleted = True

        self.atomic_modify(_mutate)
        return deleted

    # -------------------------------------------------------------------------
    # Custom Metadata & Maintenance API
    # -------------------------------------------------------------------------

    def update_custom_metadata(
        self, metadata: dict[str, Any], agent_id: str = "anonymous"
    ) -> None:
        """Updates arbitrary custom metadata key-values on the blackboard."""
        def _mutate(state: dict[str, Any]) -> None:
            state.setdefault("custom_metadata", {}).update(metadata)
            state["last_writer"] = agent_id

        self.atomic_modify(_mutate)

    def get_custom_metadata(self) -> dict[str, Any]:
        """Retrieves the custom metadata dictionary."""
        with self._acquire_lock(exclusive=False):
            state = self._read_state_unlocked()
            return copy.deepcopy(state.get("custom_metadata", {}))

    def clear(self, reset_revision: bool = False) -> None:
        """Resets the blackboard to the empty canonical initial schema.

        Args:
            reset_revision: If True, resets the monotonic revision counter to 0.
                Otherwise, preserves monotonic counter continuity across clears.
        """
        def _mutate(state: dict[str, Any]) -> None:
            prev_rev = 0 if reset_revision else state.get("revision", 0)
            fresh = self._create_initial_state()
            state.clear()
            state.update(fresh)
            state["revision"] = prev_rev

        self.atomic_modify(_mutate)

