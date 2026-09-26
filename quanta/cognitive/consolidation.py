"""quanta/cognitive/consolidation.py — SWR Memory Consolidation Engine.

Implements Pillar 3 of the Quanta Cognitive Architecture:
Sharp-Wave Ripple (SWR, 150-250 Hz) memory replay and synaptic consolidation.
Crystallizes high-utility insights from subconscious dream cycles into the
persistent cognitive state (quanta_cognitive_state.json) via CognitiveMemoryManager.
Enforces Synaptic Homeostasis (SHY microglial downscaling & pruning) and atomic I/O.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch

from quanta.cognitive.memory import CognitiveMemoryManager
from quanta.cognitive.mind_wander import DreamInsight

logger = logging.getLogger(__name__)


class GlymphaticClearanceResult(dict[str, int]):
    """Result of metabolic glymphatic clearance during quiescent NREM sleep."""

    def __init__(
        self,
        *args: Any,
        files_cleared: int = 0,
        inhibitors_pruned: int = 0,
        compacted_engrams: int = 0,
        **kwargs: int,
    ) -> None:
        """Initialize GlymphaticClearanceResult.

        Args:
            *args: Optional positional dict or mapping.
            files_cleared: Count of stale temporary files cleared.
            inhibitors_pruned: Count of depotentiated inhibitors pruned.
            compacted_engrams: Count of compacted surviving engrams.
            **kwargs: Additional key-value pairs for backward compatibility.
        """
        if args and isinstance(args[0], dict):
            super().__init__(args[0], **kwargs)
        else:
            super().__init__(
                files_cleared=files_cleared,
                inhibitors_pruned=inhibitors_pruned,
                compacted_engrams=compacted_engrams,
                **kwargs,
            )

    @property
    def files_cleared(self) -> int:
        """Count of stale temporary files cleared."""
        return self.get("files_cleared", 0)

    @property
    def inhibitors_pruned(self) -> int:
        """Count of depotentiated inhibitors pruned."""
        return self.get("inhibitors_pruned", 0)

    @property
    def compacted_engrams(self) -> int:
        """Count of compacted surviving engrams."""
        return self.get("compacted_engrams", 0)


class SubconsciousConsolidator:
    """Consolidates subconscious dream insights into persistent cognitive engrams.

    Grounds insights in biological hippocampal replay (CA3-CA1 Sharp-Wave Ripples),
    applies Cerebrospinal Fluid (CSF) quantum shielding, enforces Synaptic
    Homeostasis (Tononi & Cirelli SHY downscaling), and guarantees atomic
    persistence to avoid corruption under concurrency.
    """

    def __init__(
        self,
        state_file: Path | str = "quanta_cognitive_state.json",
        capacity: int = 64,
        enable_csf_shielding: bool = True,
    ) -> None:
        """Initialize the consolidator with a backing state file and cognitive buffer.

        Args:
            state_file: Path to quanta_cognitive_state.json file.
            capacity: Maximum number of active engrams in the cognitive buffer.
            enable_csf_shielding: Enable CSF dielectric and paramagnetic attenuation.
        """
        self.state_file = Path(state_file)
        self.memory_manager = CognitiveMemoryManager(
            capacity=capacity,
            enable_csf_shielding=enable_csf_shielding,
            auto_prune=False,
        )
        self.turn_count = 0
        self.last_injected_time = 0.0
        self.last_step_idx = 0
        self._last_saved_mtime = 0.0
        self._lock = threading.Lock()
        self._load_existing_state()

    def _load_existing_state(self) -> None:
        """Loads and populates preexisting cognitive state if the file exists on disk."""
        if not self.state_file.exists():
            return

        try:
            self._last_saved_mtime = self.state_file.stat().st_mtime
            with open(self.state_file, encoding="utf-8") as f:
                data = json.load(f)

            self.turn_count = int(data.get("turn_count", 0))
            self.last_injected_time = float(data.get("last_injected_time", 0.0))
            self.last_step_idx = int(data.get("last_step_idx", 0))

            engrams = data.get("engrams", [])
            for item in engrams:
                key = item.get("key", "")
                content = item.get("content", item.get("description", ""))
                salience = float(item.get("salience", 1.0))
                category = item.get("category", "general")
                if key and content:
                    idx = self.memory_manager.record_decision(
                        key=key,
                        content=content,
                        salience=salience,
                        category=category,
                        overwrite=True,
                    )
                    # Restore extended metadata tags
                    if 0 <= idx < len(self.memory_manager.buffer.buffer):
                        target = self.memory_manager.buffer.buffer[idx]
                        target["metadata"]["tags"] = item.get("tags", [])
                        target["metadata"]["topic"] = item.get("topic", "")
                        target["metadata"]["description"] = content
                        if "confidence" in item:
                            target["metadata"]["confidence"] = float(item["confidence"])
                        if "v_inh" in item:
                            target["metadata"]["v_inh"] = float(item["v_inh"])
                        if "is_core_anchor" in item:
                            target["metadata"]["is_core_anchor"] = bool(item["is_core_anchor"])
        except Exception as e:
            logger.warning(
                "Could not load preexisting cognitive state from %s: %s", self.state_file, e
            )

    def _merge_external_state(self) -> None:
        """Reloads and merges external state from disk to integrate newly introduced engrams."""
        if not self.state_file.exists():
            return

        try:
            curr_mtime = self.state_file.stat().st_mtime
            if curr_mtime <= self._last_saved_mtime:
                return

            with open(self.state_file, encoding="utf-8") as f:
                data = json.load(f)

            disk_turn = int(data.get("turn_count", 0))
            if disk_turn > self.turn_count:
                self.turn_count = disk_turn

            disk_time = float(data.get("last_injected_time", 0.0))
            if disk_time > self.last_injected_time:
                self.last_injected_time = disk_time

            disk_step = int(data.get("last_step_idx", 0))
            if disk_step > self.last_step_idx:
                self.last_step_idx = disk_step

            # Map existing keys in buffer to their buffer engrams
            existing_engrams = {
                e.get("metadata", {}).get("key"): e
                for e in self.memory_manager.buffer.buffer
                if e.get("metadata", {}).get("key")
            }

            for item in data.get("engrams", []):
                key = item.get("key", "")
                content = item.get("content", item.get("description", ""))
                salience = float(item.get("salience", 1.0))
                category = item.get("category", "general")
                if not key or not content:
                    continue

                if key not in existing_engrams:
                    idx = self.memory_manager.record_decision(
                        key=key,
                        content=content,
                        salience=salience,
                        category=category,
                        overwrite=False,
                    )
                    if 0 <= idx < len(self.memory_manager.buffer.buffer):
                        target = self.memory_manager.buffer.buffer[idx]
                        target["metadata"]["tags"] = item.get("tags", [])
                        target["metadata"]["topic"] = item.get("topic", "")
                        target["metadata"]["description"] = content
                        if "confidence" in item:
                            target["metadata"]["confidence"] = float(item["confidence"])
                        if "v_inh" in item:
                            target["metadata"]["v_inh"] = float(item["v_inh"])
                        if "is_core_anchor" in item:
                            target["metadata"]["is_core_anchor"] = bool(item["is_core_anchor"])
                else:
                    curr = existing_engrams[key]
                    curr_meta = curr.get("metadata", {})
                    sal_diff = abs(float(curr.get("dopamine_tag", 1.0)) - salience)
                    if curr_meta.get("content") != content or sal_diff > 0.05:
                        idx = self.memory_manager.record_decision(
                            key=key,
                            content=content,
                            salience=salience,
                            category=category,
                            overwrite=True,
                        )
                        if 0 <= idx < len(self.memory_manager.buffer.buffer):
                            target = self.memory_manager.buffer.buffer[idx]
                            target["metadata"]["tags"] = item.get("tags", [])
                            target["metadata"]["topic"] = item.get("topic", "")
                            target["metadata"]["description"] = content
                            if "confidence" in item:
                                target["metadata"]["confidence"] = float(item["confidence"])
                            if "v_inh" in item:
                                target["metadata"]["v_inh"] = float(item["v_inh"])
                            if "is_core_anchor" in item:
                                target["metadata"]["is_core_anchor"] = bool(item["is_core_anchor"])
        except Exception as e:
            logger.warning(
                "Could not merge external cognitive state from %s: %s", self.state_file, e
            )

    def consolidate_insight(self, insight: DreamInsight) -> bool:
        """Consolidates a subconscious dream insight into cognitive memory with high fidelity.

        Uses CognitiveMemoryManager to store the insight with:
        - category="subconscious_dream"
        - description=insight.synthesis (content)
        - tags=["subconscious", "dream", insight.topic]
        - initial fidelity F >= 0.99
        - dopaminergic salience tag >= 2.0 (ensures protection under SHY)

        Invokes prune_obsolete() to maintain synaptic homeostasis and prevent
        associative saturation, then atomically commits the updated state to disk.

        Args:
            insight: The DreamInsight object produced by MindWanderEngine.

        Returns:
            True if consolidation and atomic state persistence succeeded.
        """
        with self._lock:
            # 0. Reload and merge external state to prevent lost updates
            self._merge_external_state()

            topic = getattr(insight, "topic", "general_insight")
            synthesis = getattr(insight, "synthesis", "")
            confidence = float(getattr(insight, "confidence", 0.95))
            seed_question = getattr(insight, "seed_question", "")
            turns_taken = int(getattr(insight, "turns_taken", 1))
            tokens_used = int(getattr(insight, "tokens_used", 100))

            key = f"insight_{topic}"
            # Salience >= 2.0 ensures preservation and immunity against automatic
            # microglial deletion
            salience = max(2.5, min(3.0, 2.0 + confidence * 0.8))
            tags = ["subconscious", "dream", topic]

            # 1. Store via CognitiveMemoryManager
            buf_idx = self.memory_manager.record_decision(
                key=key,
                content=synthesis,
                salience=salience,
                category="subconscious_dream",
                overwrite=True,
            )

            # Enrich underlying buffer engram with dream-specific metadata
            if 0 <= buf_idx < len(self.memory_manager.buffer.buffer):
                engram = self.memory_manager.buffer.buffer[buf_idx]
                engram["metadata"]["description"] = synthesis
                engram["metadata"]["tags"] = tags
                engram["metadata"]["topic"] = topic
                engram["metadata"]["seed_question"] = seed_question
                engram["metadata"]["confidence"] = confidence
                engram["metadata"]["turns_taken"] = turns_taken
                engram["metadata"]["tokens_used"] = tokens_used

            # 2. Synaptic Homeostasis: Prune obsolete/decayed low-salience memories
            self.memory_manager.prune_obsolete(
                fidelity_threshold=0.70,
                min_salience=0.50,
                max_age=20.0,
            )

            # 3. Advance turn count and serialize state
            self.turn_count += 1
            return self.save_state_atomic()

    def get_engrams_payload(self) -> list[dict[str, Any]]:
        """Extracts JSON-serializable representation of all engrams in the memory buffer."""
        payload: list[dict[str, Any]] = []
        for engram in self.memory_manager.buffer.buffer:
            psi_0 = engram["pristine_state"]
            psi_t = engram["degraded_state"]
            overlap = torch.vdot(psi_0, psi_t)
            fid = float((torch.abs(overlap) ** 2).item())

            meta = engram.get("metadata", {})
            key = meta.get("key", "unnamed")
            content = meta.get("content", "")
            description = meta.get("description", content)
            category = meta.get("category", "general")
            salience = float(engram.get("dopamine_tag", 1.0))
            age = int(engram.get("age", 0))
            tags = meta.get("tags", [])
            topic = meta.get("topic", "")
            confidence = meta.get("confidence", 1.0)

            entry: dict[str, Any] = {
                "key": key,
                "content": content,
                "description": description,
                "salience": round(salience, 3),
                "category": category,
                "fidelity": round(fid, 5),
                "age": age,
                "tags": tags,
                "topic": topic,
                "confidence": round(float(confidence), 3),
            }
            if "is_core_anchor" in meta:
                entry["is_core_anchor"] = bool(meta["is_core_anchor"])
            if "v_inh" in meta:
                entry["v_inh"] = round(float(meta["v_inh"]), 4)
            payload.append(entry)
        return payload

    def save_state_atomic(self) -> bool:
        """Atomically persists the cognitive state to `self.state_file`.

        Writes to a unique sibling temporary file, flushes and syncs to disk,
        and atomically renames to guarantee ACID durability against partial writes.

        Returns:
            True on successful atomic write.
        """
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.state_file.with_name(
                f".tmp_{self.state_file.name}_{os.getpid()}_{time.time_ns()}"
            )

            state_data = {
                "turn_count": self.turn_count,
                "last_injected_time": self.last_injected_time,
                "last_step_idx": self.last_step_idx,
                "engrams": self.get_engrams_payload(),
            }

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_path, self.state_file)
            self._last_saved_mtime = self.state_file.stat().st_mtime
            return True
        except Exception as e:
            logger.error("Failed atomic save of cognitive state to %s: %s", self.state_file, e)
            if "temp_path" in locals() and temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()
            return False

    def load_state(self) -> dict[str, Any]:
        """Loads and returns the current state dictionary from disk."""
        if not self.state_file.exists():
            return {"turn_count": 0, "engrams": []}
        with open(self.state_file, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
            return data

    def get_insights(self) -> list[dict[str, Any]]:
        """Returns all consolidated subconscious dream insights currently in memory."""
        all_engrams = self.get_engrams_payload()
        return [e for e in all_engrams if e.get("category") == "subconscious_dream"]

    def execute_glymphatic_clearance(
        self,
        max_stale_age_seconds: float = 60.0,
        persist: bool = True,
    ) -> GlymphaticClearanceResult:
        """Executes metabolic glymphatic clearance during quiescent NREM sleep.

        Purges stale .tmp_* files older than max_stale_age_seconds from the state directory,
        eliminates depotentiated unreinforced inhibitors (V_inh < 0.20 and S <= 0.50),
        and compacts persistent JSON cognitive state with zero LLM/CLI invocations.

        Args:
            max_stale_age_seconds: Maximum age in seconds before a .tmp_* file is unlinked.
            persist: Whether to save state atomically to disk (default: True).

        Returns:
            GlymphaticClearanceResult containing counts of 'files_cleared',
            'inhibitors_pruned', and 'compacted_engrams'.
        """
        files_cleared = 0
        parent_dir = self.state_file.parent
        if parent_dir.exists():
            now = time.time()
            try:
                for p in parent_dir.iterdir():
                    if p.is_file() and p.name.startswith(".tmp_"):
                        try:
                            mtime = p.stat().st_mtime
                            if (now - mtime) > max_stale_age_seconds:
                                p.unlink()
                                files_cleared += 1
                        except OSError:
                            pass
            except OSError:
                pass

        inhibitors_pruned = 0
        with self._lock:
            self._merge_external_state()

            survivors = []
            for engram in self.memory_manager.buffer.buffer:
                meta = engram.get("metadata", {})
                cat = meta.get("category", "general")
                d_tag = float(engram.get("dopamine_tag", 1.0))
                is_core = (
                    meta.get("is_core_anchor", False)
                    or d_tag >= 2.0
                    or cat in ("semantic_gist", "subconscious_dream")
                )
                is_inhibitor = cat in ("inhibitor", "anti_pattern") or "v_inh" in meta
                v_inh = float(meta.get("v_inh", 0.50)) if is_inhibitor else None

                # Depotentiated inhibitor condition: V_inh < 0.20 and salience <= 0.50, and not core anchor
                if is_inhibitor and v_inh is not None and v_inh < 0.20 and d_tag <= 0.50 and not is_core:
                    inhibitors_pruned += 1
                else:
                    # Clean duplicate tags and strip extra whitespace
                    if "tags" in meta and isinstance(meta["tags"], list):
                        seen = set()
                        deduped = []
                        for t in meta["tags"]:
                            t_clean = str(t).strip()
                            if t_clean and t_clean not in seen:
                                seen.add(t_clean)
                                deduped.append(t_clean)
                        meta["tags"] = deduped
                    survivors.append(engram)

            self.memory_manager.buffer.buffer = survivors
            self.memory_manager.total_pruned_count += inhibitors_pruned
            compacted_count = len(survivors)

            if persist:
                self.save_state_atomic()

        return GlymphaticClearanceResult(
            files_cleared=files_cleared,
            inhibitors_pruned=inhibitors_pruned,
            compacted_engrams=compacted_count,
        )

    def execute_synaptic_homeostasis(
        self,
        shy_factor: float = 0.90,
        persist: bool = True,
    ) -> dict[str, int]:
        """Applies proactive Tononi-Cirelli Synaptic Homeostasis (SHY) downscaling.

        Multiplicatively scales down saliences of non-core engrams:
            S_e <- S_e * shy_factor
        while crystallizing newly decayed actionable decisions into semantic gists
        via distill_semantic_gist before eviction.

        Args:
            shy_factor: Synaptic downscale scaling factor in [0.85, 0.95] (default 0.90).
            persist: Whether to save state atomically to disk (default: True).

        Returns:
            Dict containing 'downscaled_count', 'gists_crystallized', and 'pruned_count'.
        """
        downscaled_count = 0
        with self._lock:
            self._merge_external_state()

            # 1. Proactive downscaling across non-core engrams
            for engram in self.memory_manager.buffer.buffer:
                meta = engram.get("metadata", {})
                d_tag = float(engram.get("dopamine_tag", 1.0))
                cat = meta.get("category", "general")
                is_core = (
                    meta.get("is_core_anchor", False)
                    or d_tag >= 2.0
                    or cat in ("semantic_gist", "subconscious_dream")
                    or meta.get("is_system_core", False)
                    or engram.get("is_system_core", False)
                )
                if not is_core:
                    new_salience = max(0.20, d_tag * shy_factor)
                    engram["dopamine_tag"] = new_salience
                    downscaled_count += 1

            # 2. Prune obsolete engrams and crystallize newly decayed actionable decisions
            pruned_records = self.memory_manager.prune_obsolete(
                fidelity_threshold=0.70,
                min_salience=0.50,
                max_age=20.0,
                enable_gist_consolidation=True,
            )

            gists_crystallized = sum(
                1 for p in pruned_records if p.get("gist_crystallized")
            )
            pruned_count = len(pruned_records)

            if persist:
                self.save_state_atomic()

        return {
            "downscaled_count": downscaled_count,
            "gists_crystallized": gists_crystallized,
            "pruned_count": pruned_count,
        }

    def execute_hippocampal_neocortical_consolidation(self, persist: bool = True) -> int:
        """Executes silent hippocampal-to-neocortical memory consolidation.

        Refreshes degraded quantum statevectors (psi_t <- psi_0) of resilient engrams,
        resetting accumulated Lindblad dephasing and restoring fidelity to pristine baseline (F=1.0).

        Args:
            persist: Whether to save state atomically to disk (default: True).

        Returns:
            Count of engrams refreshed.
        """
        refreshed_count = 0
        with self._lock:
            self._merge_external_state()

            for engram in self.memory_manager.buffer.buffer:
                if "pristine_state" in engram:
                    engram["degraded_state"] = engram["pristine_state"].clone()
                    engram["age"] = 0
                    refreshed_count += 1

            if persist:
                self.save_state_atomic()

        return refreshed_count

    def apply_wake_sleep_plasticity(
        self,
        phase: Any,
        shy_factor: float = 0.90,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Applies Hinton Wake-Sleep synaptic plasticity and persists updated state atomically.

        Args:
            phase: SleepPhase enum or string ('NREM_SWS', 'REM_DREAM').
            shy_factor: Synaptic downscaling factor in SWS (default 0.90).
            persist: Whether to save state atomically to disk (default: True).

        Returns:
            Dictionary containing plasticity outcome metrics.
        """
        with self._lock:
            self._merge_external_state()
            result = self.memory_manager.apply_wake_sleep_plasticity(
                phase=phase,
                shy_factor=shy_factor,
            )
            if persist:
                self.save_state_atomic()
            return result

