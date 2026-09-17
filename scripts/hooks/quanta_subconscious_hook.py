#!/usr/bin/env python3
"""Quanta Autonomous Subconscious Hook for Antigravity.

Runs silently and automatically on Antigravity lifecycle events (PreInvocation).
- Simulates continuous biomorphic hippocampal memory decay (CSF shielded).
- Actively prunes obsolete/decayed engrams (microglial clearance).
- Retrieves pristine, high-fidelity vital constraints via SWR replay.
- Injects ephemeral memory anchors into the agent's subconscious prompt loop.
- Optimized for ultra-low latency: < 30ms cold, pure stdlib biomorphic engine.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

# Biophysical constants (calibrated to Theorem 4 and Theorem 8)
KAPPA_CSF = 1.0 / 6250.0  # CSF quantum dephasing attenuation
GAMMA_0 = 0.05  # Bare Lindblad dephasing rate
LAMBDA_DOPAMINE = 1.5  # Dopaminergic protection gain factor
DEFAULT_DIM = 16  # Effective minicolumn state space


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
                e["fidelity"] = 1.0
                return

        # Smart capacity eviction if full
        if len(self.engrams) >= self.capacity:
            # Evict item with lowest salience * fidelity
            self.engrams.sort(key=lambda x: x["salience"] * x["fidelity"])
            self.engrams.pop(0)
            self.total_pruned_count += 1

        self.engrams.append({
            "key": key,
            "content": content,
            "salience": float(salience),
            "category": category,
            "fidelity": 1.0,
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
        # SWR replay score: salience * sqrt(fidelity)
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

        mem = FastBiomorphicMemory(capacity=32)
        turn_count = 0

        if state_file.exists():
            try:
                with open(state_file, encoding="utf-8") as sf:
                    cached = json.load(sf)
                    turn_count = cached.get("turn_count", 0)
                    for item in cached.get("engrams", []):
                        mem.engrams.append({
                            "key": item["key"],
                            "content": item["content"],
                            "salience": float(item.get("salience", 1.0)),
                            "category": item.get("category", "general"),
                            "fidelity": float(item.get("fidelity", 1.0)),
                            "age": int(item.get("age", 0)),
                        })
            except Exception:
                pass

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

        # Advance biological decay step
        turn_count += 1
        mem.step(dt=1.0)

        # Microglial active synaptic pruning
        pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50)

        # Recall vital engrams via SWR replay
        vital_anchors = [
            e for e in mem.recall_vital(top_k=3)
            if e["salience"] >= 1.5 and e["fidelity"] >= 0.85
        ]

        if vital_anchors:
            anchor_lines = [
                "[Bilinçaltı Kuantum Hafıza Çıpası | SWR Replay & CSF Shielded]:"
            ]
            for v in vital_anchors:
                fid_pct = round(v["fidelity"] * 100, 2)
                anchor_lines.append(f" • [{v['key']}] (Sadakat: %{fid_pct}): {v['content']}")

            if pruned:
                anchor_lines.append(f" ✂ [Mikroglial Budama]: {len(pruned)} geçici not temizlendi.")

            output_payload = {
                "injectSteps": [
                    {
                        "ephemeralMessage": "\n".join(anchor_lines)
                    }
                ]
            }

        # Persist updated state to disk
        try:
            with open(state_file, "w", encoding="utf-8") as sf:
                state_dict = {"turn_count": turn_count, "engrams": mem.engrams}
                json.dump(state_dict, sf, ensure_ascii=False, indent=2)
        except Exception:
            pass

    except Exception:
        output_payload = {}

    sys.stdout.write(json.dumps(output_payload))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
