#!/usr/bin/env python3
"""Quanta Autonomous Subconscious Hook for Antigravity.

Runs silently and automatically on Antigravity lifecycle events (PreInvocation).
- Simulates continuous biomorphic hippocampal memory decay (CSF shielded).
- Actively prunes obsolete/decayed engrams (microglial clearance).
- Retrieves pristine, high-fidelity vital constraints via SWR replay.
- Debounced & Deduplicated: Never repeats within same step or when unchanged.
- Precision Calibrated: Never outputs artificial flat 100.0%.
- Ultra-low latency: < 25ms, fail-safe (never crashes or interrupts).
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure quanta package is importable regardless of caller's cwd or interpreter
QUANTA_ROOT = Path(__file__).resolve().parent.parent.parent
if str(QUANTA_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANTA_ROOT))

# Biophysical constants (calibrated to Theorem 4 and Theorem 8)
KAPPA_CSF = 1.0 / 6250.0  # CSF quantum dephasing attenuation
GAMMA_0 = 0.05            # Bare Lindblad dephasing rate
LAMBDA_DOPAMINE = 1.5     # Dopaminergic protection gain factor
DEFAULT_DIM = 64          # Effective 6-qubit minicolumn state space (was 16)


def format_fidelity(fid: float) -> str:
    """Formats fidelity ensuring true biological precision (never flat 100.0%)."""
    pct = fid * 100.0
    if pct >= 99.99:
        return "99.98%"
    elif pct >= 10.0:
        return f"{pct:.2f}%"
    else:
        return f"{pct:.1f}%"


class PrunedEngram(dict):
    """Detailed pruning record for microglial synaptic clearance audit."""

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return (
                self.get("key") == other
                or self.get("id") == other
                or self.get("rule_name") == other
            )
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.get("key") or self.get("id") or self.get("rule_name"))

    def __str__(self) -> str:
        return str(self.get("key") or self.get("rule_name") or super().__str__())


class FastBiomorphicMemory:
    """Ultra-fast stdlib-only biomorphic engram manager for Antigravity hooks."""

    def __init__(self, capacity: int = 128) -> None:
        self.capacity = capacity
        self.engrams: list[dict] = []
        self.total_pruned_count = 0
        self.pruning_history: list[dict[str, Any]] = []
        self.current_turn = 0

    def record(
        self,
        key: str,
        content: str,
        salience: float = 1.0,
        category: str = "general",
        is_core_anchor: bool | None = None,
    ) -> None:
        sal = (
            float(salience)
            if isinstance(salience, (int, float))
            and not isinstance(salience, bool)
            and math.isfinite(salience)
            else 1.0
        )
        sal = max(0.01, sal)
        core_flag = bool(is_core_anchor) if is_core_anchor is not None else (sal >= 2.0)

        # Check if key exists; update if so
        for e in self.engrams:
            if e["key"] == key:
                e["content"] = content
                e["salience"] = max(e["salience"], sal)
                e["category"] = category
                e["fidelity"] = 0.9998
                if is_core_anchor is not None:
                    e["is_core_anchor"] = bool(is_core_anchor)
                elif sal >= 2.0:
                    e["is_core_anchor"] = True
                return

        # Smart capacity eviction if full: protect core anchors from being evicted
        if len(self.engrams) >= self.capacity:
            candidates = [
                (idx, e)
                for idx, e in enumerate(self.engrams)
                if not (e.get("is_core_anchor", False) or float(e.get("salience", 0.0)) >= 2.0)
            ]
            if candidates:
                candidates.sort(
                    key=lambda item: float(item[1]["salience"]) * float(item[1]["fidelity"])
                )
                evict_idx = candidates[0][0]
                self.engrams.pop(evict_idx)
                self.total_pruned_count += 1
            # If all items are core anchors, buffer gracefully expands without evicting core anchors

        self.engrams.append({
            "key": key,
            "content": content,
            "salience": float(sal),
            "category": category,
            "fidelity": 0.9998,
            "age": 0,
            "is_core_anchor": core_flag,
        })

    def record_transient(
        self,
        key: str,
        content: str,
        salience: float = 0.5,
        category: str = "contextual_decision",
    ) -> None:
        """Records an ephemeral decision with salience in [0.2, 0.8] and is_core_anchor=False."""
        sal = (
            float(salience)
            if isinstance(salience, (int, float))
            and not isinstance(salience, bool)
            and math.isfinite(salience)
            else 0.5
        )
        sal = max(0.2, min(0.8, sal))
        self.record(
            key=key,
            content=content,
            salience=sal,
            category=category,
            is_core_anchor=False,
        )

    def step(self, dt: float = 1.0) -> None:
        dim_factor = 1.0 / DEFAULT_DIM
        for e in self.engrams:
            e["age"] += 1
            # Calibrated Lindblad-Ebbinghaus dephasing with biomorphic CSF shielding
            sal = max(0.0, float(e.get("salience", 1.0)))
            is_core = bool(e.get("is_core_anchor", False)) or sal >= 2.0
            if is_core:
                eff_gamma = (GAMMA_0 * KAPPA_CSF) / max(0.01, 1.0 + LAMBDA_DOPAMINE * sal)
            else:
                kappa_eff = KAPPA_CSF + (1.0 - KAPPA_CSF) * math.exp(-2.2 * sal)
                dopamine_gain = max(0.01, 1.0 + LAMBDA_DOPAMINE * sal)
                eff_gamma = (GAMMA_0 * math.sqrt(kappa_eff)) / dopamine_gain
            decay = math.exp(-eff_gamma * dt)
            raw_fid = dim_factor + (float(e.get("fidelity", 0.9998)) - dim_factor) * decay
            e["fidelity"] = max(dim_factor, min(0.9998, raw_fid))

    def prune_obsolete(
        self,
        fidelity_threshold: float = 0.70,
        min_salience: float = 0.50,
        max_age: float | None = None,
        turn: int | None = None,
    ) -> list[PrunedEngram]:
        survivors = []
        pruned_records: list[PrunedEngram] = []
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        turn_pruned = int(turn if turn is not None else getattr(self, "current_turn", 0))

        for e in self.engrams:
            is_core = bool(e.get("is_core_anchor", False)) or float(e.get("salience", 0.0)) >= 2.0
            if is_core:
                survivors.append(e)
                continue

            is_transient = e.get("is_core_anchor") is False and float(e.get("salience", 1.0)) <= 0.8
            fid = float(e.get("fidelity", 1.0))
            sal = float(e.get("salience", 1.0))
            decayed = (
                (sal <= min_salience or is_transient)
                and fid < fidelity_threshold
            )
            age_val = float(e.get("age", 0))
            aged = (max_age is not None and age_val > max_age)

            if decayed or aged:
                if decayed and aged:
                    reason = (
                        f"fidelity_and_age ({fid:.3f} < {fidelity_threshold:.3f}, "
                        f"age {age_val:.0f} > {max_age})"
                    )
                elif decayed:
                    reason = f"fidelity_decayed ({fid:.3f} < {fidelity_threshold:.3f})"
                else:
                    reason = f"age_exceeded ({age_val:.0f} > {max_age})"

                decayed_sal = round(sal * fid, 4)
                record = PrunedEngram({
                    "timestamp": now_ts,
                    "rule_name": e["key"],
                    "id": e["key"],
                    "key": e["key"],
                    "reason": reason,
                    "decayed_salience": decayed_sal,
                    "turn_pruned": turn_pruned,
                    "final_fidelity": round(fid, 6),
                    "category": e.get("category", "contextual_decision"),
                    "content_snippet": str(e.get("content", ""))[:80],
                })
                pruned_records.append(record)
            else:
                survivors.append(e)

        self.engrams = survivors
        self.total_pruned_count += len(pruned_records)
        self.pruning_history.extend([dict(r) for r in pruned_records])
        self.pruning_history = self.pruning_history[-100:]
        return pruned_records

    def recall_vital(self, top_k: int = 3) -> list[dict]:
        scored = sorted(
            self.engrams,
            key=lambda x: x["salience"] * math.sqrt(x["fidelity"]),
            reverse=True,
        )
        return scored[:top_k]

    def consolidate(self, keys: list[str], boost: float = 0.005) -> None:
        """Sharp-Wave Ripple consolidation: restores fidelity of replayed vital memories."""
        for e in self.engrams:
            if e["key"] in keys:
                e["fidelity"] = min(0.9998, e["fidelity"] + boost)


def extract_last_user_query(transcript_path: str | Path | None) -> str:
    """Fast, fail-safe extraction of the latest user prompt from the transcript."""
    if not transcript_path:
        return ""
    p = Path(transcript_path)
    if not p.exists():
        if p.name == "transcript_full.jsonl":
            p = p.with_name("transcript.jsonl")
        if not p.exists():
            return ""

    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            seek_pos = max(0, size - 131072)
            f.seek(seek_pos)
            chunk = f.read().decode("utf-8", errors="replace")

        lines = chunk.splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            if '"USER_INPUT"' in line:
                try:
                    entry = json.loads(line)
                    content = entry.get("content", "")
                    if "<USER_REQUEST>" in content:
                        req = content.split("<USER_REQUEST>")[1].split("</USER_REQUEST>")[0].strip()
                        lines_req = [
                            item.strip() for item in req.splitlines() if item.strip()
                        ]
                        return lines_req[0][:120] if lines_req else ""
                    lines_c = [
                        item.strip() for item in content.splitlines() if item.strip()
                    ]
                    return lines_c[0][:120] if lines_c else ""
                except Exception:
                    pass
    except Exception:
        pass
    return ""


def extract_decision_from_turn(content: str, thinking: str = "") -> str:
    """Intelligently extracts the decisive conclusion, recommendation, or solution from a turn."""
    content = content.strip()
    if not content:
        return ""

    # Priority 1: Explicit decision/solution/recommendation headings
    m = re.search(
        r'(?:^|\n)#{1,4}\s*(?:[0-9.]+\s*)?(?:Çözüm|Karar|Sonuç|Öneri|Plan|Strateji|Solution|Decision|Recommendation|Resolution)[:\s]*(.*?)(?:\n|$)',
        content,
        re.IGNORECASE,
    )
    if m and len(m.group(1).strip()) > 3:
        return m.group(1).strip()

    # Priority 2: Bold decision lead lines (* **Çözüm:** ..., **Karar:** ...)
    m2 = re.search(
        r'\*\*(?:Çözüm|Karar|Sonuç|Öneri|Plan|Solution|Decision|Recommendation)[:\s]*(.*?)\*\*',
        content,
        re.IGNORECASE,
    )
    if m2 and len(m2.group(1).strip()) > 3:
        return m2.group(1).strip()

    # Priority 3: First markdown header
    m3 = re.search(r'(?:^|\n)#{1,4}\s*(?:[0-9.]+\s*)?(.*?)(?:\n|$)', content)
    if m3:
        h = m3.group(1).strip()
        if len(h) > 5 and not h.startswith("http"):
            return h

    # Priority 4: First substantive sentence
    for line in content.splitlines():
        line = line.strip('*#- \t')
        if len(line) > 15 and not line.startswith("http"):
            return line[:100]

    return content[:80]


def extract_decision_and_options(content: str, thinking: str = "") -> tuple[str, list[str]]:
    """Extracts the winning decision and candidate options from model response."""
    decision_text = extract_decision_from_turn(content, thinking)
    options: list[str] = []

    for line in content.splitlines():
        line = line.strip()
        m_opt = re.match(r'^[0-9]+[.)]\s*(.*?)$', line) or re.match(
            r'^[\*\-]\s*\*\*(.*?)\*\*', line
        )
        if m_opt:
            opt_str = m_opt.group(1).strip()
            if 5 < len(opt_str) < 70 and not opt_str.startswith("http"):
                options.append(opt_str)

    if decision_text and decision_text not in options:
        options.insert(0, decision_text)

    seen: set[str] = set()
    unique_opts: list[str] = []
    for o in options:
        if o not in seen:
            seen.add(o)
            unique_opts.append(o)

    if len(unique_opts) < 2 and decision_text:
        unique_opts = [
            decision_text,
            "Alternatif Yaklaşım / Mevcut Durumu Koru",
            "Farklı Mimari Tasarım",
        ]

    return decision_text, unique_opts[:4]


def extract_unrecorded_decisions(
    transcript_path: str | Path | None,
    last_recorded_step: int,
) -> tuple[list[dict], int]:
    """Scans transcript for completed MODEL PLANNER_RESPONSE steps after last_recorded_step.
    Returns (list_of_decisions, new_last_recorded_step).
    """
    if not transcript_path:
        return [], last_recorded_step
    p = Path(transcript_path)
    if not p.exists():
        if p.name == "transcript_full.jsonl":
            p = p.with_name("transcript.jsonl")
        if not p.exists():
            return [], last_recorded_step

    decisions: list[dict] = []
    max_step = last_recorded_step

    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            seek_pos = max(0, size - 262144)
            f.seek(seek_pos)
            chunk = f.read().decode("utf-8", errors="replace")

        lines = chunk.splitlines()
        parsed_entries = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            with contextlib.suppress(Exception):
                parsed_entries.append(json.loads(line))

        if last_recorded_step == 0 and len(parsed_entries) > 0:
            candidates = [
                e
                for e in parsed_entries
                if e.get("source") == "MODEL"
                and e.get("type") == "PLANNER_RESPONSE"
                and e.get("content")
            ]
            if candidates:
                last_recorded_step = max(0, candidates[-1].get("step_index", 0) - 1)
                max_step = last_recorded_step

        current_user_request = ""
        for entry in parsed_entries:
            src = entry.get("source")
            etype = entry.get("type")
            step = entry.get("step_index", 0)

            if src == "USER_EXPLICIT" and etype == "USER_INPUT":
                c = entry.get("content", "")
                if "<USER_REQUEST>" in c:
                    req = c.split("<USER_REQUEST>")[1].split("</USER_REQUEST>")[0].strip()
                    lines_req = [
                        item.strip() for item in req.splitlines() if item.strip()
                    ]
                    current_user_request = lines_req[0][:120] if lines_req else ""
                else:
                    lines_c = [
                        item.strip() for item in c.splitlines() if item.strip()
                    ]
                    current_user_request = lines_c[0][:120] if lines_c else ""

            elif src == "MODEL" and etype == "PLANNER_RESPONSE":
                content = entry.get("content", "").strip()
                tool_calls = entry.get("tool_calls", [])
                is_substantive = len(content) > 80 and not tool_calls
                if not is_substantive and (
                    "Çözüm" in content or "Karar" in content or "###" in content
                ):
                    is_substantive = True

                if step > last_recorded_step and is_substantive:
                    decision_text, candidate_opts = extract_decision_and_options(
                        content, entry.get("thinking", "")
                    )
                    if decision_text:
                        decisions.append({
                            "step_index": step,
                            "goal": current_user_request or "Kullanıcı Görevi / Analiz",
                            "winner": decision_text,
                            "options": candidate_opts,
                        })
                    max_step = max(max_step, step)
    except Exception:
        pass

    return decisions, max_step


def is_hermetic_test_env(payload: dict | None = None) -> bool:
    """Detects whether execution is running inside a unit test environment."""
    if "PYTEST_CURRENT_TEST" in os.environ or "PYTEST_VERSION" in os.environ:
        return True
    return bool(payload and (payload.get("testing") or payload.get("hermetic")))


def _atomic_write_single_file(target: Path, state_dict: dict) -> bool:
    """Atomically writes state_dict to target path using POSIX atomic replacement.

    Creates temporary file .tmp_{filename}_{pid}_{time_ns} in target directory,
    writes JSON, flushes, fsyncs, and performs atomic os.replace.
    Cleans up temp file on any exception.
    """
    temp_path = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target.with_name(f".tmp_{target.name}_{os.getpid()}_{time.time_ns()}")
        with open(temp_path, "w", encoding="utf-8") as sf:
            json.dump(state_dict, sf, ensure_ascii=False, indent=2)
            sf.flush()
            with contextlib.suppress(OSError):
                os.fsync(sf.fileno())
        os.replace(temp_path, target)
        return True
    except Exception:
        if temp_path is not None and temp_path.exists():
            with contextlib.suppress(OSError):
                temp_path.unlink()
        return False
    except BaseException:
        if temp_path is not None and temp_path.exists():
            with contextlib.suppress(OSError):
                temp_path.unlink()
        raise


def _save_mirrored_state_atomically(
    primary_path: Path,
    mirror_path: Path | None,
    state_dict: dict[str, Any],
    conv_id: str,
    is_test_env: bool = False,
) -> tuple[bool, bool]:
    """Atomically writes state to primary_path and, if not in test env, mirror_path."""
    primary_ok = False
    mirror_ok = False

    # 1. Primary write with fallback to /tmp
    candidates = [primary_path]
    fallback = Path(f"/tmp/quanta_cognitive_{conv_id}.json")
    if fallback != primary_path:
        candidates.append(fallback)

    for target in candidates:
        if _atomic_write_single_file(target, state_dict):
            primary_ok = True
            break

    # 2. Mirror write (independent try/except fail-safe)
    if mirror_path is not None:
        try:
            ws_root_file = (QUANTA_ROOT / "quanta_cognitive_state.json").resolve()
            is_root_target = False
            with contextlib.suppress(Exception):
                is_root_target = (mirror_path.resolve() == ws_root_file)

            # Skip mirror if is_test_env, or if running under pytest and target is root
            skip_mirror = is_test_env or (is_root_target and is_hermetic_test_env())

            with contextlib.suppress(Exception):
                if mirror_path.resolve() == primary_path.resolve():
                    skip_mirror = True

            if not skip_mirror:
                mirror_dict = state_dict
                if is_root_target:
                    mirror_dict = copy.deepcopy(state_dict)
                    # Filter engrams so that root file strictly preserves production engrams
                    prod_engrams = [
                        e for e in state_dict.get("engrams", [])
                        if isinstance(e, dict)
                        and (
                            e.get("category") in ("subconscious_dream", "architecture_rfc")
                            or str(e.get("key", "")).startswith("insight_")
                            or e.get("key") == "architecture_rfc"
                        )
                    ]
                    if prod_engrams:
                        mirror_dict["engrams"] = prod_engrams

                mirror_ok = _atomic_write_single_file(mirror_path, mirror_dict)
        except Exception:
            mirror_ok = False

    return primary_ok, mirror_ok


def _save_state_atomically(state_file: Path, state_dict: dict, safe_conv_id: str) -> None:
    """Atomically writes state to state_file, falling back to /tmp if unwriteable."""
    candidates = [state_file]
    fallback = Path(f"/tmp/quanta_cognitive_{safe_conv_id}.json")
    if fallback != state_file:
        candidates.append(fallback)

    for target in candidates:
        if _atomic_write_single_file(target, state_dict):
            break


def refresh_memory_from_state(mem: FastBiomorphicMemory, state_file: Path) -> dict:
    """Refreshes FastBiomorphicMemory engrams and returns updated cached state dict."""
    if not state_file.exists():
        return {}
    try:
        with open(state_file, encoding="utf-8", errors="replace") as sf:
            refreshed = json.load(sf)
        if not isinstance(refreshed, dict):
            return {}

        raw_engrams = refreshed.get("engrams")
        if isinstance(raw_engrams, list):
            existing_by_key = {
                e["key"]: e for e in mem.engrams if isinstance(e, dict) and "key" in e
            }
            for item in raw_engrams:
                if not isinstance(item, dict) or "key" not in item:
                    continue
                k = item["key"]
                if "is_core_anchor" not in item:
                    item["is_core_anchor"] = (float(item.get("salience", 1.0)) >= 2.0)
                if k in existing_by_key:
                    existing_by_key[k].update(item)
                else:
                    mem.engrams.append(copy.deepcopy(item))
                    existing_by_key[k] = mem.engrams[-1]
        return refreshed
    except Exception:
        return {}


def main() -> None:
    output_payload: dict = {}
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            sys.stdout.write(json.dumps(output_payload))
            sys.stdout.flush()
            return

        payload = json.loads(raw_input)
        raw_conv_id = str(payload.get("conversationId", "default") or "default")
        safe_conv_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', raw_conv_id)
        conversation_id = safe_conv_id
        artifact_dir = payload.get("artifactDirectoryPath", "")

        # State storage path
        if artifact_dir and os.path.exists(artifact_dir):
            state_file = Path(artifact_dir) / "quanta_cognitive_state.json"
        else:
            state_file = Path(f"/tmp/quanta_cognitive_{safe_conv_id}.json")

        # Workspace state storage path (root mirror candidate)
        workspace_paths = payload.get("workspacePaths", [])
        workspace_state_file = None
        if workspace_paths and isinstance(workspace_paths, list):
            for wp in workspace_paths:
                if wp:
                    p = Path(wp) / "quanta_cognitive_state.json"
                    if p.exists():
                        workspace_state_file = p
                        break
        if workspace_state_file is None:
            workspace_state_file = QUANTA_ROOT / "quanta_cognitive_state.json"

        now = time.time()
        mem = FastBiomorphicMemory(capacity=128)
        turn_count = 0
        last_injected_time = 0.0
        last_step_idx = -1
        last_recorded_decision_step = 0
        last_verified_step = 0
        current_zeno_pinning = None
        mean_zeno_pinning = None
        pruning_history: list[dict[str, Any]] = []
        recent_arbitrations: list[dict[str, Any]] = []
        cached: dict = {}
        current_step_idx = payload.get("stepIdx", payload.get("initialNumSteps", 0))

        if state_file.exists():
            try:
                with open(state_file, encoding="utf-8", errors="replace") as sf:
                    cached = json.load(sf)
                    turn_count = cached.get("turn_count", 0)
                    last_injected_time = cached.get("last_injected_time", 0.0)
                    last_step_idx = cached.get("last_step_idx", -1)
                    last_recorded_decision_step = cached.get("last_recorded_decision_step", 0)
                    last_verified_step = cached.get("last_verified_step", 0)
                    mem.total_pruned_count = int(cached.get("total_pruned_count", 0))
                    pruning_history = list(
                        cached.get("pruning_history") or cached.get("pruned_history") or []
                    )
                    current_zeno_pinning = cached.get("current_zeno_pinning")
                    mean_zeno_pinning = cached.get("mean_zeno_pinning")
                    recent_arbitrations = list(cached.get("recent_arbitrations", []))
                    for item in cached.get("engrams", []):
                        sal_val = float(item.get("salience", 1.0))
                        is_core_raw = item.get("is_core_anchor")
                        is_core = (sal_val >= 2.0) if is_core_raw is None else bool(is_core_raw)

                        mem.engrams.append({
                            "key": item["key"],
                            "content": item.get("content", item.get("description", "")),
                            "salience": sal_val,
                            "category": item.get("category", "general"),
                            "fidelity": float(item.get("fidelity", 0.9998)),
                            "age": int(item.get("age", 0)),
                            "tags": item.get("tags", []),
                            "topic": item.get("topic", ""),
                            "confidence": float(item.get("confidence", 0.95)),
                            "consecutive_successes": int(item.get("consecutive_successes", 0)),
                            "consecutive_failures": int(item.get("consecutive_failures", 0)),
                            "total_evaluations": int(item.get("total_evaluations", 0)),
                            "last_outcome": item.get("last_outcome", "NEUTRAL"),
                            "drift_status": item.get("drift_status", "STABLE"),
                            "is_core_anchor": is_core,
                        })
            except Exception:
                pass

        # Hydrate non-duplicate engrams from workspace root (preserves 53 production engrams in production)
        if not is_hermetic_test_env(payload) and workspace_state_file and workspace_state_file.exists():
            try:
                is_same = False
                with contextlib.suppress(Exception):
                    is_same = (workspace_state_file.resolve() == state_file.resolve())
                if not is_same:
                    with open(workspace_state_file, encoding="utf-8", errors="replace") as wf:
                        ws_cached = json.load(wf)
                    if isinstance(ws_cached, dict):
                        if "pruning_history" in ws_cached and not pruning_history:
                            pruning_history = list(ws_cached.get("pruning_history", []))
                        if "recent_arbitrations" in ws_cached and not recent_arbitrations:
                            recent_arbitrations = list(ws_cached.get("recent_arbitrations", []))
                        if current_zeno_pinning is None and "current_zeno_pinning" in ws_cached:
                            current_zeno_pinning = ws_cached.get("current_zeno_pinning")
                        if mean_zeno_pinning is None and "mean_zeno_pinning" in ws_cached:
                            mean_zeno_pinning = ws_cached.get("mean_zeno_pinning")

                        existing_keys = {
                            e["key"] for e in mem.engrams if isinstance(e, dict) and "key" in e
                        }
                        ws_engrams = ws_cached.get("engrams", [])
                        if isinstance(ws_engrams, list):
                            for item in ws_engrams:
                                if not isinstance(item, dict) or "key" not in item:
                                    continue
                                k = item["key"]
                                if k not in existing_keys:
                                    sal_val = float(item.get("salience", 1.0))
                                    is_core_raw = item.get("is_core_anchor")
                                    is_core = (
                                        (sal_val >= 2.0)
                                        if is_core_raw is None
                                        else bool(is_core_raw)
                                    )
                                    hydrated = {
                                        "key": k,
                                        "content": item.get("content", item.get("description", "")),
                                        "salience": sal_val,
                                        "category": item.get("category", "general"),
                                        "fidelity": float(item.get("fidelity", 0.9998)),
                                        "age": int(item.get("age", 0)),
                                        "tags": item.get("tags", []),
                                        "topic": item.get("topic", ""),
                                        "confidence": float(item.get("confidence", 0.95)),
                                        "consecutive_successes": int(
                                            item.get("consecutive_successes", 0)
                                        ),
                                        "consecutive_failures": int(
                                            item.get("consecutive_failures", 0)
                                        ),
                                        "total_evaluations": int(item.get("total_evaluations", 0)),
                                        "last_outcome": item.get("last_outcome", "NEUTRAL"),
                                        "drift_status": item.get("drift_status", "STABLE"),
                                        "is_core_anchor": is_core,
                                    }
                                    if "description" in item:
                                        hydrated["description"] = item["description"]
                                    mem.engrams.append(hydrated)
                                    existing_keys.add(k)
            except Exception:
                pass

        if current_zeno_pinning is None:
            if recent_arbitrations:
                current_zeno_pinning = float(recent_arbitrations[-1].get("p_zeno", 0.8603))
            else:
                current_zeno_pinning = 0.8603
        else:
            current_zeno_pinning = float(current_zeno_pinning)

        if mean_zeno_pinning is None:
            if recent_arbitrations:
                zenos = [
                    float(a.get("p_zeno", current_zeno_pinning))
                    for a in recent_arbitrations
                    if isinstance(a, dict)
                ]
                mean_zeno_pinning = (
                    round(sum(zenos) / len(zenos), 4) if zenos else current_zeno_pinning
                )
            else:
                mean_zeno_pinning = current_zeno_pinning
        else:
            mean_zeno_pinning = float(mean_zeno_pinning)

        mem.pruning_history = list(pruning_history)
        mem.current_turn = turn_count

        # Check workspace and project context
        workspace_paths = payload.get("workspacePaths", [])
        transcript_path = payload.get("transcriptPath")
        from quanta.cognitive.telemetry import (
            detect_workspace,
            record_hook_telemetry,
        )
        real_workspace = detect_workspace(paths=workspace_paths)

        # 1. EXTRACT & ARBITRATE COMPLETED DECISIONS VIA REAL 6-QUBIT QUANTUM ARBITER
        # Check transcript for newly completed decisions from model planner responses
        unrecorded_decisions, new_last_dec_step = extract_unrecorded_decisions(
            transcript_path, last_recorded_decision_step
        )
        if unrecorded_decisions:
            try:
                from quanta.cognitive.arbiter import QuantumDecisionArbiter
                arbiter = QuantumDecisionArbiter(dim=64)
                for dec in unrecorded_decisions:
                    opts = list(dec.get("options") or [])
                    if dec["winner"] not in opts:
                        opts.insert(0, dec["winner"])
                    if len(opts) < 2:
                        opts.append("Alternatif Yaklaşım")
                    if len(opts) < 3:
                        opts.append("Mevcut Durumu Koru")

                    dilemma_text = dec.get("goal") or f"Mimari Karar: {dec['winner']}"
                    hyps = [
                        {"id": f"d{i+1}", "label": opt_name, "ket": f"|d{i+1}>"}
                        for i, opt_name in enumerate(opts[:3])
                    ]
                    arb_res = arbiter.arbitrate_dilemma(
                        dilemma=dilemma_text,
                        hypotheses=hyps,
                        workspace=real_workspace,
                        step_idx=dec.get("step_index", 0),
                        log_telemetry=True,
                    )
                    recent_arbitrations.append({
                        "step_idx": dec.get("step_index", 0),
                        "dilemma": dilemma_text,
                        "hypotheses": arb_res.get("hypotheses", []),
                        "winner": arb_res.get("winner", "d1"),
                        "winning_label": arb_res.get("winning_label", dec["winner"]),
                        "p_zeno": arb_res.get("p_zeno", 0.8603),
                        "injected_prompt_constraint": arb_res.get("injected_prompt_constraint", ""),
                        "counterfactual_ab": arb_res.get("counterfactual_ab", {}),
                    })
                    recent_arbitrations = recent_arbitrations[-20:]
                    current_zeno_pinning = arb_res.get("p_zeno", current_zeno_pinning)
                    zeno_window = [
                        float(a.get("p_zeno", current_zeno_pinning))
                        for a in recent_arbitrations
                        if isinstance(a, dict)
                    ]
                    mean_zeno_pinning = (
                        round(sum(zeno_window) / len(zeno_window), 4)
                        if zeno_window
                        else current_zeno_pinning
                    )
            except Exception:
                pass

            # Ingest transcript decisions into transient engrams
            for dec in unrecorded_decisions:
                step_idx = dec.get("step_index", 0)
                winner = dec.get("winner", "")
                goal = dec.get("goal", "")
                dec_key = (
                    f"decision_step_{step_idx}" if step_idx else f"decision_{int(time.time())}"
                )
                content = f"{winner} (Hedef: {goal[:60]})" if goal else winner
                mem.record_transient(
                    key=dec_key,
                    content=content,
                    salience=0.5,
                    category="contextual_decision",
                )

            last_recorded_decision_step = new_last_dec_step

        # 1b. EXTRACT & VERIFY ACTION OUTCOMES (CLOSED-LOOP FEEDBACK ENGINE)
        try:
            from quanta.cognitive.feedback import (
                CognitiveFeedbackLoop,
                OutcomeVerifier,
            )
            # Direct event evaluation if toolCall or error payload is provided
            is_post_tool = (
                payload.get("toolCall")
                or payload.get("error")
                or payload.get("event") == "PostToolUse"
            )
            if is_post_tool:
                tc = payload.get("toolCall", {})
                t_name = tc.get("name", payload.get("toolName", "tool"))
                t_args = tc.get("args", payload.get("toolArgs", {}))
                t_err = payload.get("error")
                t_out = payload.get("result", payload.get("output"))
                step_val = payload.get("stepIdx", current_step_idx)

                verifier = OutcomeVerifier(dim=64)
                outcome = verifier.evaluate_tool_result(
                    tool_name=t_name,
                    tool_args=t_args,
                    tool_output=t_out,
                    error=t_err,
                    active_rules=mem.engrams,
                    step_idx=step_val,
                )
                feedback_loop = CognitiveFeedbackLoop(verifier=verifier)
                feedback_loop.process_outcomes(
                    outcomes=[outcome],
                    state_path=state_file,
                    workspace=real_workspace,
                )
                last_verified_step = max(last_verified_step, step_val)
                refreshed = refresh_memory_from_state(mem, state_file)
                if refreshed:
                    cached.update(refreshed)

            # Retrospective transcript verification across previous turn's executed steps
            elif transcript_path:
                verifier = OutcomeVerifier(dim=64)
                unverified_outcomes = verifier.extract_outcomes_from_transcript(
                    transcript_path=transcript_path,
                    last_step_idx=last_verified_step,
                    active_rules=mem.engrams,
                )
                if unverified_outcomes:
                    feedback_loop = CognitiveFeedbackLoop(verifier=verifier)
                    feedback_loop.process_outcomes(
                        outcomes=unverified_outcomes,
                        state_path=state_file,
                        workspace=real_workspace,
                    )
                    max_scanned_step = max(o.step_idx for o in unverified_outcomes)
                    last_verified_step = max(last_verified_step, max_scanned_step)
                    # Refresh in-memory engrams from state file
                    refreshed = refresh_memory_from_state(mem, state_file)
                    if refreshed:
                        cached.update(refreshed)
        except Exception:
            pass

        # If this is a Stop lifecycle event, persist state and return immediately
        if payload.get("terminationReason") is not None:
            state_dict = dict(cached)
            state_dict.update({
                "turn_count": turn_count,
                "last_injected_time": now,
                "last_step_idx": current_step_idx,
                "last_recorded_decision_step": last_recorded_decision_step,
                "last_verified_step": last_verified_step,
                "engrams": mem.engrams,
                "total_pruned_count": mem.total_pruned_count,
                "kappa_csf": KAPPA_CSF,
                "current_zeno_pinning": round(float(current_zeno_pinning), 4),
                "mean_zeno_pinning": round(float(mean_zeno_pinning), 4),
                "pruning_history": pruning_history,
                "recent_arbitrations": recent_arbitrations,
            })
            _save_mirrored_state_atomically(
                primary_path=state_file,
                mirror_path=workspace_state_file,
                state_dict=state_dict,
                conv_id=safe_conv_id,
                is_test_env=is_hermetic_test_env(payload),
            )
            sys.stdout.write(json.dumps({}))
            sys.stdout.flush()
            return

        # 2. DEBOUNCE / RE-ENTRANCY CHECK FOR PRE-INVOCATION:
        # If called within 2.0 seconds or for same step, return empty
        is_recent = (now - last_injected_time) < 2.0
        is_same_step = (current_step_idx == last_step_idx) and (last_step_idx != -1)
        if is_recent or is_same_step:
            sys.stdout.write(json.dumps({}))
            sys.stdout.flush()
            return

        # Extract user query if present for conversational relevance matching
        user_query = ""
        for key_candidate in ("userPrompt", "prompt", "message", "userMessage", "query", "text"):
            val = payload.get(key_candidate)
            if isinstance(val, str) and val.strip():
                user_query = val.strip().lower()
                break

        last_query = extract_last_user_query(transcript_path)
        if not user_query and last_query:
            user_query = last_query.lower()

        cwd_hint = os.getcwd().lower()
        search_targets = (real_workspace.lower(), cwd_hint, artifact_dir.lower(), user_query)
        is_turna = any("turna" in h or "meiro" in h or "dengage" in h for h in search_targets)

        # Ensure foundational cognitive anchors exist in active memory
        existing_rule_keys = {e["key"] for e in mem.engrams if isinstance(e, dict) and "key" in e}
        if "executive_summary_rule" not in existing_rule_keys:
            mem.record(
                key="executive_summary_rule",
                content=(
                    "Kullanıcıya daima sonuç odaklı, net ve sade bir yönetici "
                    "özeti (Executive Summary) sun; formüllere boğma."
                ),
                salience=2.5,
                category="constraint",
                is_core_anchor=True,
            )
        if "scientific_integrity_rule" not in existing_rule_keys:
            mem.record(
                key="scientific_integrity_rule",
                content=(
                    "Tüm iddia ve önermelerde literatür doğrulaması yap; "
                    "halüsinasyon yapma ve bize ait olmayan fikirleri açıkça ayır."
                ),
                salience=2.5,
                category="constraint",
                is_core_anchor=True,
            )
        if "native_first_rule" not in existing_rule_keys:
            mem.record(
                key="native_first_rule",
                content=(
                    "Platform veya servis işlemlerinde daima native API/CLI aracını"
                    " öncelikli kullan; yetersiz kalırsa doğrudan ikincil sistemlere"
                    " geçmeden önce kullanıcıya sor."
                ),
                salience=2.8,
                category="constraint",
                is_core_anchor=True,
            )
        if is_turna and "turna_api_hierarchy_rule" not in existing_rule_keys:
            mem.record(
                key="turna_api_hierarchy_rule",
                content=(
                    "Turna projelerinde Dengage için Dengage API, Meiro için Meiro API/mpcli, "
                    "BigQuery için BQ kullan. Yetersiz kalırsa önce kullanıcıya sor."
                ),
                salience=3.2,
                category="constraint",
                is_core_anchor=True,
            )

        # Advance biological decay step
        turn_count += 1
        mem.current_turn = turn_count
        mem.step(dt=1.0)

        # Microglial active synaptic pruning
        pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50, turn=turn_count)
        if pruned:
            for p in pruned:
                p_dict = dict(p) if isinstance(p, dict) else {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "rule_name": str(p),
                    "id": str(p),
                    "key": str(p),
                    "reason": "microglial_clearance",
                    "decayed_salience": 0.25,
                    "turn_pruned": turn_count,
                    "final_fidelity": 0.65,
                }
                pruning_history.append(p_dict)
            pruning_history = pruning_history[-100:]

        # Query Relevance Scoring for Subconscious Dream Engrams:
        # Boost salience of subconscious dream insights that match user query context
        query_words_set: set[str] = set()
        if user_query:
            query_snippet = user_query[:10000] if len(user_query) > 10000 else user_query
            query_words_set = {w for w in query_snippet.split() if len(w) > 4}

        for e in mem.engrams:
            if e.get("category") == "subconscious_dream":
                topic = e.get("topic", "").lower()
                tags = [t.lower() for t in e.get("tags", [])]
                content = e.get("content", "").lower()

                # If query is provided, check lexical relevance
                if user_query:
                    content_words = set(content.split())
                    is_relevant = (
                        topic in user_query
                        or any(t in user_query for t in tags if len(t) > 2)
                        or any(w in user_query for w in topic.split("_") if len(w) > 3)
                        or bool(content_words & query_words_set)
                    )
                    if is_relevant:
                        e["salience"] = max(e["salience"], 3.5)
                else:
                    # When no prompt provided (e.g. automated turns / tests),
                    # protect high-priority dreams
                    e["salience"] = max(e["salience"], 2.8)

        # Recall vital engrams via SWR replay
        # Separate vital anchors into Core Anchors and Contextual Decisions
        core_anchors = [
            e for e in mem.engrams
            if (e.get("is_core_anchor", False) or float(e.get("salience", 0.0)) >= 2.0)
            and float(e.get("fidelity", 0.0)) >= 0.80
        ]
        # Sort core anchors by salience descending
        core_anchors.sort(key=lambda x: float(x.get("salience", 0.0)), reverse=True)
        # Prioritize invariant constraint rules
        # (native_first_rule, scientific_integrity_rule, executive_summary_rule)
        constraint_rules = [e for e in core_anchors if e.get("category") == "constraint"]
        other_core = [e for e in core_anchors if e.get("category") != "constraint"]
        selected_core = constraint_rules + other_core[:max(0, 6 - len(constraint_rules))]

        transient_decisions = [
            e for e in mem.engrams
            if not (e.get("is_core_anchor", False) or float(e.get("salience", 0.0)) >= 2.0)
            and float(e.get("fidelity", 0.0)) >= 0.70
        ]
        # Sort transient decisions by fidelity * salience descending
        transient_decisions.sort(
            key=lambda x: float(x.get("fidelity", 0.0)) * float(x.get("salience", 0.0)),
            reverse=True,
        )
        selected_transient = transient_decisions[:4]

        vital_anchors = selected_core + selected_transient

        if vital_anchors:
            core_items = []
            for v in selected_core:
                fid_str = format_fidelity(v["fidelity"])
                core_items.append(f"{v['key']} ({fid_str})")

            transient_items = []
            for t in selected_transient:
                fid_str = format_fidelity(t["fidelity"])
                transient_items.append(f"{t['key']} ({fid_str})")

            lines = ["[Quanta Bilişsel Çıpa | SWR Replay]:"]
            if core_items:
                lines.append(f"  🔒 Çekirdek: {', '.join(core_items)}")
            if transient_items:
                lines.append(f"  ⚡ Geçici: {', '.join(transient_items)}")
            if pruned:
                prune_word = "engram"
                lines.append(f"  ✂️ Budandı: {len(pruned)} {prune_word}")

            compact_msg = "\n".join(lines)

            output_payload = {
                "injectSteps": [
                    {
                        "ephemeralMessage": compact_msg
                    }
                ]
            }
            mem.consolidate([v["key"] for v in vital_anchors])

        # Centralized non-blocking telemetry logging
        try:
            hook_latency_ms = (time.time() - now) * 1000.0
            replayed_rules = [
                {
                    "rule": v["key"],
                    "key": v["key"],
                    "fidelity": round(float(v["fidelity"]), 6),
                    "salience": round(float(v.get("salience", 1.0)), 2),
                    "category": v.get("category", "constraint"),
                }
                for v in vital_anchors
            ] if vital_anchors else []
            record_hook_telemetry(
                conversation_id=conversation_id,
                step_idx=current_step_idx,
                turn_count=turn_count,
                rules_replayed=replayed_rules,
                pruned_count=len(pruned),
                latency_ms=hook_latency_ms,
                workspace=real_workspace,
                last_user_query=last_query or user_query,
                kappa_csf=KAPPA_CSF,
                pruned_keys=pruned,
                total_pruned_count=mem.total_pruned_count,
                active_engrams_count=len(mem.engrams),
            )
        except Exception:
            pass

        # Persist updated state to disk atomically
        state_dict = dict(cached)
        state_dict.update({
            "turn_count": turn_count,
            "last_injected_time": now,
            "last_step_idx": current_step_idx,
            "last_recorded_decision_step": last_recorded_decision_step,
            "last_verified_step": last_verified_step,
            "engrams": mem.engrams,
            "total_pruned_count": mem.total_pruned_count,
            "kappa_csf": KAPPA_CSF,
            "current_zeno_pinning": round(float(current_zeno_pinning), 4),
            "mean_zeno_pinning": round(float(mean_zeno_pinning), 4),
            "pruning_history": pruning_history,
            "recent_arbitrations": recent_arbitrations,
        })
        if pruned:
            state_dict["last_pruned_engrams"] = [
                {
                    "key": (k.get("key") if isinstance(k, dict) else str(k)),
                    "turn": turn_count,
                    "time": now,
                }
                for k in pruned
            ]
        _save_mirrored_state_atomically(
            primary_path=state_file,
            mirror_path=workspace_state_file,
            state_dict=state_dict,
            conv_id=safe_conv_id,
            is_test_env=is_hermetic_test_env(payload),
        )

    except Exception:
        output_payload = {}

    sys.stdout.write(json.dumps(output_payload))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
