"""Cognitive Memory Manager with Hippocampal SWR and Conscious Synaptic Pruning."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch

from quanta.torch.brain import CSFShieldedEnvironment, NoisyHippocampalBuffer


def text_to_statevector(text: str, dim: int = 64) -> torch.Tensor:
    """Deterministically transforms text into a normalized complex statevector in C^dim.

    Combines Quantum Natural Language Processing (QNLP) token phasor superposition
    with a SHA-256 global digest anchor. Words with shared semantic tokens interfere
    constructively in Hilbert space, while preserving exact determinism and unit norm.
    """
    clean_text = text.strip()
    if not clean_text:
        return torch.ones(dim, dtype=torch.complex64) / math.sqrt(dim)

    tokens = [
        w.strip().lower()
        for w in clean_text.replace(",", " ")
        .replace(".", " ")
        .replace(":", " ")
        .replace("-", " ")
        .replace("_", " ")
        .replace("/", " ")
        .split()
        if len(w.strip()) > 0
    ]
    if not tokens:
        tokens = [clean_text.lower()]

    reals = [0.0] * dim
    imags = [0.0] * dim

    # 1. Global digest anchor (breaks permutation symmetries)
    g_hash = hashlib.sha256(clean_text.encode("utf-8")).digest()
    for i in range(dim):
        b1 = g_hash[(i * 2) % len(g_hash)]
        b2 = g_hash[(i * 2 + 1) % len(g_hash)]
        reals[i] += ((b1 / 127.5) - 1.0) * 0.4
        imags[i] += ((b2 / 127.5) - 1.0) * 0.4

    # 2. QNLP Token phasor superposition
    for word in tokens:
        w_hash = hashlib.sha256(word.encode("utf-8")).digest()
        for i in range(dim):
            theta = (w_hash[i % len(w_hash)] / 255.0) * 2.0 * math.pi
            reals[i] += math.cos(theta)
            imags[i] += math.sin(theta)

    c_tensor = torch.complex(
        torch.tensor(reals, dtype=torch.float32), torch.tensor(imags, dtype=torch.float32)
    )
    norm = torch.linalg.norm(c_tensor)
    if norm > 1e-12:
        res: torch.Tensor = torch.as_tensor(c_tensor / norm)
        return res
    return c_tensor


class CognitiveMemoryManager:
    """Episodic and working memory manager for Antigravity AI agents.

    Integrates:
    1. Biologically realistic hippocampal buffering (CA3-CA1) with Lindblad phase diffusion.
    2. Five-tier Cerebrospinal Fluid (CSF) biophysical quantum shielding.
    3. Active biological forgetting (Ebbinghaus decay differentiated by dopamine salience).
    4. Conscious Synaptic Pruning: Microglia-inspired active eviction of obsolete memories.
    5. Priority-weighted capacity eviction to protect critical constraints from FIFO loss.
    """

    def __init__(
        self,
        capacity: int = 64,
        dim: int = 64,
        enable_csf_shielding: bool = True,
        auto_prune: bool = False,
        prune_threshold: float = 0.70,
        device: torch.device | str | None = "cpu",
    ) -> None:
        self.dim = dim
        self.auto_prune = auto_prune
        self.prune_threshold = prune_threshold
        self.total_pruned_count = 0
        self.csf_env = CSFShieldedEnvironment(device=device) if enable_csf_shielding else None
        self.buffer = NoisyHippocampalBuffer(
            capacity=capacity,
            noise_level=0.03,
            phase_diffusion_rate=0.015,
            temporal_decay_rate=0.005,
            dopamine_protection=2.0,
            csf_environment=self.csf_env,
            device=device,
            dtype=torch.float32,
        )

    def record_decision(
        self,
        key: str,
        content: str,
        salience: float = 1.0,
        category: str = "decision",
        overwrite: bool = True,
        is_core_anchor: bool | None = None,
    ) -> int:
        """Records a strategic decision, constraint, or user instruction.

        Args:
            key: Concise key or title (e.g. 'architecture_choice', 'db_constraint').
            content: Full text explanation or decision summary.
            salience: Priority tag (0.1-0.5: temporary, 1.0: normal, >=2.0: critical rule).
            category: Semantic category ('decision', 'constraint', 'temporary', etc.).
            overwrite: If True and an engram with `key` exists, consciously prunes the old one.
            is_core_anchor: Whether this engram is a permanent core anchor immune to pruning.
                If None, automatically resolves to True if salience >= 2.0, False otherwise.

        Returns:
            Buffer index of the stored engram.
        """
        if overwrite:
            self.forget(key)

        state = text_to_statevector(f"{key}:{content}", dim=self.dim)
        resolved_is_core = (
            bool(is_core_anchor) if is_core_anchor is not None else (float(salience) >= 2.0)
        )
        metadata = {
            "key": key,
            "content": content,
            "category": category,
            "salience": float(salience),
            "is_core_anchor": resolved_is_core,
        }

        # Smart priority eviction if buffer is at capacity
        if len(self.buffer.buffer) >= self.buffer.capacity:
            self._evict_least_salient()

        return self.buffer._store_single(state, dopamine_tag=salience, metadata=metadata)

    def record_transient_decision(
        self,
        key: str,
        content: str,
        salience: float = 0.5,
        category: str = "contextual_decision",
    ) -> int:
        """Records an ephemeral in-conversation decision with salience clamped to [0.2, 0.8].

        Transient decisions are marked with is_core_anchor=False, allowing them to naturally
        decay through Lindblad phase diffusion and be cleared by microglial synaptic pruning.

        Args:
            key: Concise key or title.
            content: Full text explanation or decision summary.
            salience: Priority tag (clamped to [0.2, 0.8], default: 0.5).
            category: Semantic category (default: 'contextual_decision').

        Returns:
            Buffer index of the stored engram.
        """
        clamped_salience = max(0.2, min(0.8, float(salience)))
        return self.record_decision(
            key=key,
            content=content,
            salience=clamped_salience,
            category=category,
            overwrite=True,
            is_core_anchor=False,
        )

    def update_decision(
        self,
        key: str,
        content: str,
        salience: float = 1.0,
        category: str = "decision",
        is_core_anchor: bool | None = None,
    ) -> int:
        """Consciously supersedes / updates an existing decision, preventing stale conflicts."""
        return self.record_decision(
            key=key,
            content=content,
            salience=salience,
            category=category,
            overwrite=True,
            is_core_anchor=is_core_anchor,
        )

    def forget(self, key: str) -> bool:
        """Explicitly and consciously forgets a memory by key (conscious invalidation).

        Args:
            key: The unique key of the decision or constraint to prune.

        Returns:
            True if one or more engrams were pruned, False if key was not found.
        """
        initial_len = len(self.buffer.buffer)
        self.buffer.buffer = [
            e for e in self.buffer.buffer if e.get("metadata", {}).get("key") != key
        ]
        pruned = initial_len - len(self.buffer.buffer)
        if pruned > 0:
            self.total_pruned_count += pruned
            return True
        return False

    def prune(self, key: str) -> bool:
        """Alias for `forget(key)`."""
        return self.forget(key)

    def prune_obsolete(
        self,
        fidelity_threshold: float = 0.70,
        min_salience: float = 0.5,
        max_age: float | None = None,
    ) -> list[dict[str, Any]]:
        """Active Synaptic Pruning: Clears decayed, expired, or unreinforced engrams.

        Inspired by sleep downscaling (Tononi & Cirelli) and microglial phagocytosis.
        Prunes engrams that have decayed below `fidelity_threshold` (for low/medium salience)
        or exceeded `max_age`. High-salience constraints and core anchors (is_core_anchor=True
        or salience >= 2.0) are permanently shielded against automatic pruning.

        Returns:
            List of dictionaries describing each pruned engram and the pruning reason.
        """
        if not self.buffer.buffer:
            return []

        survivors = []
        pruned_records = []

        for engram in self.buffer.buffer:
            psi_0 = engram["pristine_state"]
            psi_t = engram["degraded_state"]
            overlap = torch.vdot(psi_0, psi_t)
            fid = float((torch.abs(overlap) ** 2).item())
            d_tag = float(engram["dopamine_tag"])
            age = float(engram["age"])
            meta = engram.get("metadata") or {}
            key = meta.get("key", "unnamed")

            is_core = bool(meta.get("is_core_anchor", False) or d_tag >= 2.0)
            if is_core:
                survivors.append(engram)
                continue

            should_prune = False
            reason = ""
            is_transient = meta.get("is_core_anchor") is False and d_tag <= 0.8

            # Condition A: Decayed below threshold for low-salience or transient decisions
            if fid < fidelity_threshold and (d_tag <= min_salience or is_transient):
                should_prune = True
                reason = f"fidelity_decayed ({fid:.3f} < {fidelity_threshold})"
            # Condition B: Low-salience scratchpad or transient items exceeding maximum turn age
            elif max_age is not None and age > max_age and (d_tag <= min_salience or is_transient):
                should_prune = True
                reason = f"age_exceeded ({age:.1f} turns > {max_age})"
            # Condition C: Catastrophic dephasing even for normal items (fidelity < 0.35)
            elif fid < 0.35:
                should_prune = True
                reason = f"irrecoverable_dephasing ({fid:.3f} < 0.35)"

            if should_prune:
                pruned_records.append(
                    {
                        "key": key,
                        "content": meta.get("content", ""),
                        "salience": d_tag,
                        "age_turns": age,
                        "final_fidelity": round(fid, 5),
                        "reason": reason,
                        "is_core_anchor": False,
                    }
                )
            else:
                survivors.append(engram)

        self.buffer.buffer = survivors
        self.total_pruned_count += len(pruned_records)
        return pruned_records

    def _evict_least_salient(self) -> None:
        """Evicts lowest-scoring non-core engram (protects core anchors with salience >= 2.0)."""
        if not self.buffer.buffer:
            return

        candidate_indices = [
            idx
            for idx, e in enumerate(self.buffer.buffer)
            if not (
                (e.get("metadata") or {}).get("is_core_anchor", False)
                or float(e.get("dopamine_tag", 1.0)) >= 2.0
            )
        ]

        # If all engrams in buffer are core anchors, buffer expands gracefully to prevent FIFO loss
        if not candidate_indices:
            self.buffer.capacity = max(self.buffer.capacity, len(self.buffer.buffer) + 1)
            return

        min_idx = candidate_indices[0]
        min_score = float("inf")
        for idx in candidate_indices:
            engram = self.buffer.buffer[idx]
            psi_0 = engram["pristine_state"]
            psi_t = engram["degraded_state"]
            overlap = torch.vdot(psi_0, psi_t)
            fid = float((torch.abs(overlap) ** 2).item())
            score = float(engram["dopamine_tag"]) * fid
            if score < min_score:
                min_score = score
                min_idx = idx

        self.buffer.buffer.pop(min_idx)
        self.total_pruned_count += 1


    def step(
        self,
        dt: float = 1.0,
        auto_prune: bool | None = None,
        prune_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Simulates biological time progression, active forgetting & continuous dephasing.

        Args:
            dt: Biological elapsed turn duration (default: 1.0).
            auto_prune: Whether to execute active synaptic pruning after decay.
            prune_threshold: Retention threshold for automatic pruning.

        Returns:
            List of engrams pruned during this step (if auto_prune is active).
        """
        if not self.buffer.buffer or dt <= 0.0:
            return []

        kappa_base = (
            float(self.csf_env.compute_attenuation_factor().item())
            if self.csf_env is not None
            else 1.0
        )
        dev = self.buffer.current_device
        rdtype = self.buffer.current_real_dtype

        for engram in self.buffer.buffer:
            d_tag = float(engram["dopamine_tag"])
            # Active forgetting dynamics:
            # Low dopamine (D <= 0.4) experiences natural dephasing and thermal bath drift.
            # High dopamine (D >= 2.0) triggers synaptic tagging and CSF dielectric shielding.
            kappa_eff = kappa_base + (1.0 - kappa_base) * math.exp(-2.2 * d_tag)
            sqrt_k = math.sqrt(kappa_eff)

            sigma_noise = (0.05 / (1.0 + d_tag)) * sqrt_k * math.sqrt(dt)
            sigma_phi = (0.03 / (1.0 + d_tag)) * sqrt_k * math.sqrt(dt)
            gamma_drift = (0.015 / (1.0 + 2.0 * d_tag)) * math.sqrt(kappa_eff) * dt

            current_state = engram["degraded_state"]
            dim = current_state.shape[0]

            # 1. Lindblad Phase Diffusion
            if sigma_phi > 1e-9:
                delta_theta = torch.randn(dim, device=dev, dtype=rdtype) * sigma_phi
                phase_factor = torch.exp(1j * delta_theta)
                current_state = current_state * phase_factor

            # 2. Thermal Amplitude Jitter
            if sigma_noise > 1e-9:
                scale = sigma_noise / math.sqrt(2.0)
                noise_r = torch.randn(dim, device=dev, dtype=rdtype) * scale
                noise_i = torch.randn(dim, device=dev, dtype=rdtype) * scale
                xi = torch.complex(noise_r, noise_i)
                current_state = current_state + xi

            # 3. Lindblad Thermal Bath Depolarization Drift (Ebbinghaus asymptotic drift)
            if gamma_drift > 1e-9:
                decay_factor = math.exp(-gamma_drift)
                bath_r = torch.randn(dim, device=dev, dtype=rdtype)
                bath_i = torch.randn(dim, device=dev, dtype=rdtype)
                bath = torch.complex(bath_r, bath_i)
                bath_norm = torch.linalg.norm(bath)
                if bath_norm > 1e-12:
                    bath = bath / bath_norm
                    current_state = (
                        math.sqrt(decay_factor) * current_state
                        + math.sqrt(1.0 - decay_factor) * bath
                    )

            norm = torch.linalg.norm(current_state)
            if norm > 1e-12:
                current_state = current_state / norm

            engram["degraded_state"] = current_state
            engram["age"] += dt

        # Check if automatic pruning should run
        run_prune = self.auto_prune if auto_prune is None else auto_prune
        if run_prune:
            thresh = self.prune_threshold if prune_threshold is None else prune_threshold
            return self.prune_obsolete(fidelity_threshold=thresh)
        return []

    def recall_vital_context(
        self, top_k: int = 5, min_fidelity: float = 0.40
    ) -> list[dict[str, Any]]:
        """Recalls the most resilient memories via Sharp-Wave Ripple (SWR) replay.

        Evaluates real-time biological retention fidelity F(t) and dopaminergic tags
        to ensure high-priority constraints are never forgotten while degraded engrams
        are surfaced with transparent diagnostic fidelity.
        """
        if not self.buffer.buffer:
            return []

        results = []
        for idx, engram in enumerate(self.buffer.buffer):
            psi_0 = engram["pristine_state"]
            psi_t = engram["degraded_state"]
            overlap = torch.vdot(psi_0, psi_t)
            fidelity = float((torch.abs(overlap) ** 2).item())

            if fidelity < min_fidelity:
                continue

            status = (
                "pristine"
                if fidelity >= 0.999
                else (
                    "consolidated"
                    if fidelity >= 0.95
                    else ("decaying" if fidelity >= 0.70 else "obsolete")
                )
            )

            meta = engram.get("metadata") or {}
            results.append(
                {
                    "index": idx,
                    "key": meta.get("key", ""),
                    "content": meta.get("content", ""),
                    "category": meta.get("category", ""),
                    "salience": engram["dopamine_tag"],
                    "age_turns": round(float(engram["age"]), 1),
                    "retention_fidelity": round(fidelity, 5),
                    "retention_pct": round(fidelity * 100, 2),
                    "status": status,
                    "is_core_anchor": bool(
                        meta.get("is_core_anchor", float(engram["dopamine_tag"]) >= 2.0)
                        or float(engram["dopamine_tag"]) >= 2.0
                    ),
                }
            )

        results.sort(key=lambda x: x["salience"] * x["retention_fidelity"], reverse=True)
        return results[:top_k]

    def get_status_summary(self) -> dict[str, Any]:
        """Returns diagnostic metrics of the cognitive buffer."""
        mean_fid = self.buffer.get_mean_fidelity()
        active_keys = [e.get("metadata", {}).get("key", "") for e in self.buffer.buffer]
        return {
            "stored_engrams": len(self.buffer.buffer),
            "capacity": self.buffer.capacity,
            "mean_retention_fidelity": round(mean_fid, 5),
            "mean_retention_pct": round(mean_fid * 100, 2),
            "total_pruned_count": self.total_pruned_count,
            "csf_shielded": self.csf_env is not None,
            "active_keys": active_keys,
        }

    def save_state(self, filepath: str) -> None:
        """Serializes episodic engram buffer to persistent storage."""
        torch.save(
            {
                "buffer": self.buffer.buffer,
                "dim": self.dim,
                "total_pruned_count": self.total_pruned_count,
            },
            filepath,
        )

    def load_state(self, filepath: str) -> None:
        """Loads episodic engram buffer from persistent storage."""
        data = torch.load(filepath, map_location=self.buffer.current_device)
        self.buffer.buffer = data["buffer"]
        self.dim = data.get("dim", self.dim)
        self.total_pruned_count = data.get("total_pruned_count", 0)

