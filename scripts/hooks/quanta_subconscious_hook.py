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
import json
import math
import os
import sys
import time
from pathlib import Path

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


class FastBiomorphicMemory:
    """Ultra-fast stdlib-only biomorphic engram manager for Antigravity hooks."""

    def __init__(self, capacity: int = 32) -> None:
        self.capacity = capacity
        self.engrams: list[dict] = []
        self.total_pruned_count = 0

    def record(
        self,
        key: str,
        content: str,
        salience: float = 1.0,
        category: str = "general",
    ) -> None:
        # Check if key exists; update if so
        for e in self.engrams:
            if e["key"] == key:
                e["content"] = content
                e["salience"] = max(e["salience"], salience)
                e["category"] = category
                e["fidelity"] = 0.9998
                return

        # Smart capacity eviction if full
        if len(self.engrams) >= self.capacity:
            self.engrams.sort(key=lambda x: x["salience"] * x["fidelity"])
            self.engrams.pop(0)
            self.total_pruned_count += 1

        self.engrams.append({
            "key": key,
            "content": content,
            "salience": float(salience),
            "category": category,
            "fidelity": 0.9998,
            "age": 0,
        })

    def step(self, dt: float = 1.0) -> None:
        dim_factor = 1.0 / DEFAULT_DIM
        for e in self.engrams:
            e["age"] += 1
            # Calibrated Lindblad-Ebbinghaus dephasing with CSF shielding
            eff_gamma = (GAMMA_0 * KAPPA_CSF) / (1.0 + LAMBDA_DOPAMINE * e["salience"])
            decay = math.exp(-eff_gamma * dt)
            e["fidelity"] = dim_factor + (e["fidelity"] - dim_factor) * decay

    def prune_obsolete(
        self,
        fidelity_threshold: float = 0.70,
        min_salience: float = 0.50,
    ) -> list[str]:
        survivors = []
        pruned_keys = []
        for e in self.engrams:
            if e["salience"] < min_salience and e["fidelity"] < fidelity_threshold:
                pruned_keys.append(e["key"])
            else:
                survivors.append(e)
        self.engrams = survivors
        self.total_pruned_count += len(pruned_keys)
        return pruned_keys

    def recall_vital(self, top_k: int = 3) -> list[dict]:
        scored = sorted(
            self.engrams,
            key=lambda x: x["salience"] * math.sqrt(x["fidelity"]),
            reverse=True,
        )
        return scored[:top_k]


def main() -> None:
    output_payload: dict = {}
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            sys.stdout.write(json.dumps(output_payload))
            sys.stdout.flush()
            return

        payload = json.loads(raw_input)
        conversation_id = payload.get("conversationId", "default")
        artifact_dir = payload.get("artifactDirectoryPath", "")

        # State storage path
        if artifact_dir and os.path.exists(artifact_dir):
            state_file = Path(artifact_dir) / "quanta_cognitive_state.json"
        else:
            state_file = Path(f"/tmp/quanta_cognitive_{conversation_id}.json")

        now = time.time()
        mem = FastBiomorphicMemory(capacity=32)
        turn_count = 0
        last_injected_time = 0.0
        last_step_idx = -1
        current_step_idx = payload.get("stepIdx", payload.get("initialNumSteps", 0))

        if state_file.exists():
            try:
                with open(state_file, encoding="utf-8") as sf:
                    cached = json.load(sf)
                    turn_count = cached.get("turn_count", 0)
                    last_injected_time = cached.get("last_injected_time", 0.0)
                    last_step_idx = cached.get("last_step_idx", -1)
                    for item in cached.get("engrams", []):
                        mem.engrams.append({
                            "key": item["key"],
                            "content": item.get("content", item.get("description", "")),
                            "salience": float(item.get("salience", 1.0)),
                            "category": item.get("category", "general"),
                            "fidelity": float(item.get("fidelity", 0.9998)),
                            "age": int(item.get("age", 0)),
                            "tags": item.get("tags", []),
                            "topic": item.get("topic", ""),
                            "confidence": float(item.get("confidence", 0.95)),
                        })
            except Exception:
                pass

        # 1. DEBOUNCE / RE-ENTRANCY CHECK:
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

        # Check workspace and project context
        workspace_hint = str(payload.get("workspaceDirectory", "")).lower()
        cwd_hint = os.getcwd().lower()
        is_turna = any("turna" in h or "meiro" in h or "dengage" in h for h in (workspace_hint, cwd_hint, artifact_dir.lower(), user_query))

        # If fresh conversation, initialize foundational cognitive anchors
        if len(mem.engrams) == 0:
            mem.record(
                key="executive_summary_rule",
                content=(
                    "Kullanıcıya daima sonuç odaklı, net ve sade bir yönetici "
                    "özeti (Executive Summary) sun; formüllere boğma."
                ),
                salience=2.5,
                category="constraint",
            )
            mem.record(
                key="scientific_integrity_rule",
                content=(
                    "Tüm iddia ve önermelerde literatür doğrulaması yap; "
                    "halüsinasyon yapma ve bize ait olmayan fikirleri açıkça ayır."
                ),
                salience=2.5,
                category="constraint",
            )
            mem.record(
                key="native_first_rule",
                content=(
                    "Platform veya servis işlemlerinde daima native API/CLI aracını öncelikli kullan; "
                    "yetersiz kalırsa doğrudan ikincil sistemlere geçmeden önce kullanıcıya sor."
                ),
                salience=2.8,
                category="constraint",
            )
            if is_turna:
                mem.record(
                    key="turna_api_hierarchy_rule",
                    content=(
                        "Turna projelerinde Dengage için Dengage API, Meiro için Meiro API/mpcli, "
                        "BigQuery için BQ kullan. Yetersiz kalırsa önce kullanıcıya sor."
                    ),
                    salience=3.2,
                    category="constraint",
                )

        # Advance biological decay step
        turn_count += 1
        mem.step(dt=1.0)

        # Microglial active synaptic pruning
        pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)

        # Query Relevance Scoring for Subconscious Dream Engrams:
        # Boost salience of subconscious dream insights that match user query context
        for e in mem.engrams:
            if e.get("category") == "subconscious_dream":
                topic = e.get("topic", "").lower()
                tags = [t.lower() for t in e.get("tags", [])]
                content = e.get("content", "").lower()

                # If query is provided, check lexical relevance
                if user_query:
                    is_relevant = (
                        topic in user_query
                        or any(t in user_query for t in tags if len(t) > 2)
                        or any(w in user_query for w in topic.split("_") if len(w) > 3)
                        or any(w in content for w in user_query.split() if len(w) > 4)
                    )
                    if is_relevant:
                        e["salience"] = max(e["salience"], 3.5)
                else:
                    # When no prompt provided (e.g. automated turns / tests),
                    # protect high-priority dreams
                    e["salience"] = max(e["salience"], 2.8)

        # Recall vital engrams via SWR replay (top_k=5 to capture rules and dreams)
        vital_anchors = [
            e for e in mem.recall_vital(top_k=5)
            if e["salience"] >= 1.5 and e["fidelity"] >= 0.80
        ]

        if vital_anchors:
            items = []
            for v in vital_anchors:
                fid_str = format_fidelity(v["fidelity"])
                items.append(f"{v['key']} ({fid_str})")

            prune_str = f" | Budandı: {len(pruned)}" if pruned else ""
            compact_msg = f"[Quanta Bilişsel Çıpa | SWR Replay]: {', '.join(items)}{prune_str}"

            output_payload = {
                "injectSteps": [
                    {
                        "ephemeralMessage": compact_msg
                    }
                ]
            }

        # Centralized non-blocking telemetry logging
        try:
            telemetry_dir = Path.home() / ".gemini" / "antigravity" / "telemetry"
            telemetry_dir.mkdir(parents=True, exist_ok=True)
            telemetry_file = telemetry_dir / "quanta_cognitive_telemetry.jsonl"
            hook_latency_ms = (time.time() - now) * 1000.0
            replayed_keys = [v["key"] for v in vital_anchors] if vital_anchors else []
            hook_entry = {
                "event_type": "hook_step",
                "timestamp": now,
                "iso_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now)),
                "workspace": Path(workspace_hint or os.getcwd()).name or "General",
                "conversation_id": conversation_id,
                "step_idx": current_step_idx,
                "turn_count": turn_count,
                "rules_replayed": replayed_keys,
                "pruned_count": len(pruned),
                "latency_ms": round(hook_latency_ms, 2),
            }
            with open(telemetry_file, "a", encoding="utf-8") as tf:
                tf.write(json.dumps(hook_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

        # Persist updated state to disk atomically
        try:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            temp_path = state_file.with_name(
                f".tmp_{state_file.name}_{os.getpid()}_{time.time_ns()}"
            )
            state_dict = {
                "turn_count": turn_count,
                "last_injected_time": now,
                "last_step_idx": current_step_idx,
                "engrams": mem.engrams,
            }
            with open(temp_path, "w", encoding="utf-8") as sf:
                json.dump(state_dict, sf, ensure_ascii=False, indent=2)
                sf.flush()
                os.fsync(sf.fileno())
            os.replace(temp_path, state_file)
        except Exception:
            if "temp_path" in locals() and temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()

    except Exception:
        output_payload = {}

    sys.stdout.write(json.dumps(output_payload))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
