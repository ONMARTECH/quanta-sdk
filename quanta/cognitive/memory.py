"""Cognitive Memory Manager with Hippocampal SWR and Conscious Synaptic Pruning."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import torch

from quanta.torch.brain import CSFShieldedEnvironment, NoisyHippocampalBuffer

# Plasticity & Cognitive Immunity Invariant Constants
V_INH_MIN: float = 0.05
V_INH_MAX: float = 0.9998
V_INH_DEFAULT: float = 0.50
SALIENCE_INH_DEFAULT: float = 1.50
SALIENCE_MAX: float = 3.5
SALIENCE_FLOOR: float = 0.20
ETA_LTP: float = 0.25
DELTA_S_LTP: float = 0.35
LAMBDA_LTD: float = 0.30
DELTA_S_LTD: float = 0.35
GAMMA_INH: float = 0.02
SALIENCE_GIST_DEFAULT: float = 1.85
GIST_CATEGORY: str = "semantic_gist"


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


def is_actionable_resolution(content: str, key: str = "") -> bool:
    """Evaluates whether an ephemeral memory contains an actionable architectural resolution.

    Filters out operational tool chatter and requires architectural, strategic, or
    normative invariant markers (Brainerd & Reyna Fuzzy-Trace Theory).

    Args:
        content: Text content of the memory engram.
        key: Key/identifier of the engram.

    Returns:
        True if the content represents an actionable resolution suitable for
        fuzzy-trace semantic crystallization; False otherwise.
    """
    clean = content.strip().lower()
    if len(clean) < 20:
        return False

    # Exclude purely procedural tool noise
    noise_patterns = (
        r"^\[\w+\]",
        r"^(?:view_file|run_command|grep_search|find_by_name|list_dir|grep|find|cd|ls|cat|pwd)\b",
        r"\b(?:checked line|reading file|listing directory|command exited with code)\b",
    )
    if any(re.search(p, clean) for p in noise_patterns):
        return False

    # Check for strategic/architectural keywords & directives
    strategic_keywords = (
        "architect", "mimari", "strategy", "strateji", "solution", "çözüm",
        "decision", "karar", "invariant", "rule", "kural", "standard", "standart",
        "policy", "politika", "protocol", "protokol", "design", "tasarım",
        "pattern", "approach", "yaklaşım", "principle", "prensip", "consensus", "uzlaşı",
        "always", "never", "daima", "asla", "kullan", "use", "adopt", "benimse",
        "avoid", "kaçın", "enforce", "uygula", "optimize", "pin", "kilitle",
        "sqlite", "posix", "fsync", "waf", "cf.client.bot", "route", "sdk", "api",
    )
    combined = f"{key.lower()} {clean}" if key else clean
    return any(
        re.search(rf"\b{re.escape(kw)}", combined) if len(kw) <= 4 else kw in combined
        for kw in strategic_keywords
    )


def distill_semantic_gist(content: str, key: str = "") -> tuple[str, str]:
    """Distills an ephemeral decision into a crisp, permanent semantic gist.

    Removes conversational preambles, parenthetical targets, and step details.
    Produces a normalized gist key and a distilled summary statement (max 140 chars).

    Args:
        content: Verbatim text content of the decision engram.
        key: Original key of the decision engram (e.g. 'decision_step_42').

    Returns:
        tuple of (gist_key, distilled_content)
    """
    clean = content.strip()
    # Strip parenthetical goals (e.g. '(Hedef: ...)' or '(Goal: ...)')
    clean = re.sub(r"\((?:Hedef|Goal|Target):[^)]*\)", "", clean, flags=re.IGNORECASE).strip()
    # Strip leading preambles ('Karar: ', 'Çözüm: ', 'Sonuç: ', 'Solution: ', 'Decision: ', etc.)
    clean = re.sub(
        r"^(?:#{1,4}\s*)?(?:\*\*|\*)?"
        r"(?:Karar|Çözüm|Sonuç|Solution|Decision|Recommendation)[:\s]*(?:\*\*|\*)?",
        "",
        clean,
        flags=re.IGNORECASE,
    ).strip()
    # Normalize extra whitespace
    clean = re.sub(r"\s+", " ", clean).strip()

    # Generate gist key
    clean_key = re.sub(r"^decision_", "", key) if key else ""
    if not clean_key:
        slug_words = re.findall(r"[a-zA-Z0-9_]+", clean.lower())[:3]
        clean_key = "_".join(slug_words) if slug_words else "unnamed"

    gist_key = clean_key if clean_key.startswith("gist_") else f"gist_{clean_key}"
    gist_content = clean[:140].strip()
    if gist_content and not gist_content.endswith((".", "!", "?")):
        gist_content += "."
    return gist_key, gist_content


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

    def record_semantic_gist(
        self,
        key: str,
        content: str,
        salience: float = SALIENCE_GIST_DEFAULT,
        is_core_anchor: bool = True,
    ) -> int:
        """Records a crystallized semantic gist into core memory (Brainerd & Reyna FTT).

        Anchors overarching conclusion into core memory with high salience (S >= 1.8)
        and is_core_anchor=True, permanently protecting it against synaptic downscaling.

        Args:
            key: Unique gist identifier (e.g. 'gist_posix_atomic').
            content: Distilled invariant summary text.
            salience: Dopaminergic priority tag (clamped to >= 1.80).
            is_core_anchor: Whether to flag as an immutable core anchor (default: True).

        Returns:
            Buffer index of the stored engram.
        """
        sal = max(1.80, float(salience))
        return self.record_decision(
            key=key,
            content=content,
            salience=sal,
            category=GIST_CATEGORY,
            overwrite=True,
            is_core_anchor=is_core_anchor,
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

    def record_inhibitor(
        self,
        key: str,
        content: str,
        v_inh: float = V_INH_DEFAULT,
        salience: float = SALIENCE_INH_DEFAULT,
        context_tags: dict[str, Any] | None = None,
        category: str = "inhibitor",
    ) -> int:
        """Records an adaptive negative engram representing an anti-pattern or failed action.

        Args:
            key: Unique inhibitor identifier (e.g. 'inh_run_command_zsh_glob').
            content: Detailed description of failure mode, syntax violation, or anti-pattern.
            v_inh: Dynamic synaptic inhibitory weight in [0.05, 0.9998] (default: 0.50).
            salience: Noradrenergic threat salience clamped to [0.01, 3.5] (default: 1.50).
            context_tags: Contextual metadata (runtime, tool, library, os, error_type).
            category: Semantic category ('inhibitor' or 'anti_pattern', default: 'inhibitor').

        Returns:
            Buffer index of the stored inhibitory engram.
        """
        self.forget(key)

        clamped_v_inh = max(V_INH_MIN, min(V_INH_MAX, float(v_inh)))
        clamped_sal = max(0.01, min(SALIENCE_MAX, float(salience)))
        is_core = bool(clamped_sal >= 2.0)
        tags = dict(context_tags) if context_tags else {}

        state = text_to_statevector(f"{key}:{content}", dim=self.dim)
        metadata = {
            "key": key,
            "content": content,
            "category": category,
            "v_inh": clamped_v_inh,
            "salience": clamped_sal,
            "is_core_anchor": is_core,
            "context_tags": tags,
            "consecutive_failures": 1,
            "consecutive_successes": 0,
            "context_divergence": [],
        }

        if len(self.buffer.buffer) >= self.buffer.capacity:
            self._evict_least_salient()

        return self.buffer._store_single(state, dopamine_tag=clamped_sal, metadata=metadata)

    def potentiate_inhibitor(
        self,
        key: str,
        context_tags: dict[str, Any] | None = None,
        boost: float = ETA_LTP,
    ) -> float:
        """Long-Term Potentiation (LTP): Deepens inhibition weight V_inh upon repeated failure.

        Asymptotically deepens synaptic weight:
            V_inh <- min(0.9998, V_inh + boost * (0.9998 - V_inh))
        Increases threat salience and promotes to core anchor if salience >= 2.0.

        Args:
            key: Unique inhibitor identifier.
            context_tags: Optional execution context tags to update or verify.
            boost: Asymptotic learning rate eta_ltp (default: 0.25).

        Returns:
            Updated inhibitory synaptic weight V_inh in [0.05, 0.9998].
        """
        target_engram = None
        for engram in self.buffer.buffer:
            meta = engram.get("metadata") or {}
            if meta.get("key") == key:
                target_engram = engram
                break

        if target_engram is None:
            self.record_inhibitor(
                key=key,
                content=f"Inhibitory pattern for {key}",
                v_inh=V_INH_DEFAULT,
                salience=SALIENCE_INH_DEFAULT,
                context_tags=context_tags,
            )
            for engram in self.buffer.buffer:
                meta = engram.get("metadata") or {}
                if meta.get("key") == key:
                    target_engram = engram
                    break

        if target_engram is None:
            return V_INH_DEFAULT

        meta = target_engram.setdefault("metadata", {})
        k_fail = int(meta.get("consecutive_failures", 0)) + 1
        meta["consecutive_failures"] = k_fail
        meta["consecutive_successes"] = 0

        if context_tags:
            existing_tags = meta.setdefault("context_tags", {})
            existing_tags.update(context_tags)

        # Asymptotic potentiation: V_inh <- min(0.9998, V_inh + boost * (0.9998 - V_inh))
        curr_v = float(meta.get("v_inh", V_INH_DEFAULT))
        new_v = min(V_INH_MAX, curr_v + float(boost) * (V_INH_MAX - curr_v))
        meta["v_inh"] = new_v

        # Salience scaling (threat awakening): S <- min(3.5, S + 0.35 * (1 + 0.1 * k_fail))
        curr_sal = float(target_engram["dopamine_tag"])
        new_sal = min(SALIENCE_MAX, curr_sal + DELTA_S_LTP * (1.0 + 0.1 * k_fail))
        target_engram["dopamine_tag"] = new_sal
        meta["salience"] = new_sal

        # Critical threshold crossing: core anchor promotion
        if new_sal >= 2.0:
            meta["is_core_anchor"] = True

        # Fidelity consolidation: refresh degraded state representation to pristine
        target_engram["degraded_state"] = target_engram["pristine_state"].clone()
        target_engram["age"] = 0.0

        return new_v

    def depress_inhibitor(
        self,
        key: str,
        context_tags: dict[str, Any] | None = None,
        decay: float = LAMBDA_LTD,
    ) -> float:
        """Long-Term Depression (LTD): Relaxes V_inh and documents context divergence on success.

        Relaxes synaptic weight exponentially:
            V_inh <- max(0.05, V_inh * (1 - decay))
        Decreases salience and demotes from core anchor if salience falls below 2.0.

        Args:
            key: Unique inhibitor identifier.
            context_tags: The successful execution context tags to compute divergence against.
            decay: Multiplicative relaxation rate lambda_ltd (default: 0.30).

        Returns:
            Updated inhibitory synaptic weight V_inh in [0.05, 0.9998].
        """
        target_engram = None
        for engram in self.buffer.buffer:
            meta = engram.get("metadata") or {}
            if meta.get("key") == key:
                target_engram = engram
                break

        if target_engram is None:
            return 0.0

        meta = target_engram.setdefault("metadata", {})
        meta["consecutive_successes"] = int(meta.get("consecutive_successes", 0)) + 1
        meta["consecutive_failures"] = 0

        # Context divergence calculation: Delta C = {tag: (old, new) | old != new}
        if context_tags:
            existing_tags = meta.setdefault("context_tags", {})
            diff = {}
            for k in set(existing_tags.keys()) | set(context_tags.keys()):
                old_val = existing_tags.get(k)
                new_val = context_tags.get(k)
                if old_val != new_val:
                    diff[k] = (old_val, new_val)
            if diff:
                meta.setdefault("context_divergence", []).append(diff)
            existing_tags.update(context_tags)

        # Synaptic weight relaxation (multiplicative decay): V_inh <- max(0.05, V_inh * (1 - decay))
        curr_v = float(meta.get("v_inh", V_INH_DEFAULT))
        new_v = max(V_INH_MIN, curr_v * (1.0 - float(decay)))
        meta["v_inh"] = new_v

        # Salience relaxation
        curr_sal = float(target_engram["dopamine_tag"])
        new_sal = max(SALIENCE_FLOOR, curr_sal - DELTA_S_LTD)
        target_engram["dopamine_tag"] = new_sal
        meta["salience"] = new_sal

        # Demote core anchor if salience falls below threshold
        if new_sal < 2.0:
            meta["is_core_anchor"] = False

        return new_v

    def recall_inhibitors(
        self,
        top_k: int = 3,
        min_v_inh: float = 0.20,
    ) -> list[dict[str, Any]]:
        """Recalls active inhibitors ranked by (V_inh * salience).

        Args:
            top_k: Maximum number of active inhibitors to return.
            min_v_inh: Minimum inhibition weight threshold for activation (default: 0.20).

        Returns:
            List of active inhibitor dictionaries sorted descending by inhibition potency.
        """
        if not self.buffer.buffer:
            return []

        active_inhibitors = []
        for engram in self.buffer.buffer:
            meta = engram.get("metadata") or {}
            cat = meta.get("category", "")
            if cat not in ("inhibitor", "anti_pattern"):
                continue

            v_inh = float(meta.get("v_inh", V_INH_DEFAULT))
            if v_inh < min_v_inh:
                continue

            psi_0 = engram["pristine_state"]
            psi_t = engram["degraded_state"]
            overlap = torch.vdot(psi_0, psi_t)
            fid = float((torch.abs(overlap) ** 2).item())
            sal = float(engram["dopamine_tag"])

            active_inhibitors.append(
                {
                    "key": meta.get("key", ""),
                    "content": meta.get("content", ""),
                    "category": cat,
                    "v_inh": round(v_inh, 4),
                    "salience": round(sal, 2),
                    "fidelity": round(fid, 5),
                    "context_tags": dict(meta.get("context_tags", {})),
                    "consecutive_failures": int(meta.get("consecutive_failures", 0)),
                    "consecutive_successes": int(meta.get("consecutive_successes", 0)),
                    "context_divergence": list(meta.get("context_divergence", [])),
                    "is_core_anchor": bool(meta.get("is_core_anchor", False) or sal >= 2.0),
                }
            )

        active_inhibitors.sort(key=lambda x: x["v_inh"] * x["salience"], reverse=True)
        return active_inhibitors[:top_k]

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
        enable_gist_consolidation: bool = True,
    ) -> list[dict[str, Any]]:
        """Active Synaptic Pruning with Fuzzy-Trace Semantic Gist Extraction.

        Inspired by sleep downscaling (Tononi & Cirelli), microglial phagocytosis,
        and Brainerd & Reyna Fuzzy-Trace Theory (FTT). Prunes engrams that have decayed
        below `fidelity_threshold` (for low/medium salience) or exceeded `max_age`.
        Before pruning decayed transient decisions at fidelity_threshold, analyzes
        whether an actionable strategic/architectural resolution exists. If so,
        synthesizes and anchors a semantic gist (category='semantic_gist', S >= 1.8)
        into core memory while evicting the fine-grained verbatim engram.
        High-salience constraints and core anchors (is_core_anchor=True or salience >= 2.0)
        are permanently shielded against automatic pruning.

        Returns:
            List of dictionaries describing each pruned engram and the pruning reason.
        """
        if not self.buffer.buffer:
            return []

        survivors = []
        pruned_records = []
        gists_to_record: list[tuple[str, str, float]] = []

        for engram in self.buffer.buffer:
            psi_0 = engram["pristine_state"]
            psi_t = engram["degraded_state"]
            overlap = torch.vdot(psi_0, psi_t)
            fid = float((torch.abs(overlap) ** 2).item())
            d_tag = float(engram["dopamine_tag"])
            age = float(engram["age"])
            meta = engram.get("metadata") or {}
            key = meta.get("key", "unnamed")
            cat = meta.get("category", "")
            content = meta.get("content", "")

            is_core = bool(
                meta.get("is_core_anchor", False)
                or d_tag >= 2.0
                or cat == GIST_CATEGORY
            )
            if is_core:
                survivors.append(engram)
                continue

            should_prune = False
            reason = ""
            is_transient = meta.get("is_core_anchor") is False and d_tag <= 0.8
            is_inhibitor = cat in ("inhibitor", "anti_pattern")
            v_inh = float(meta.get("v_inh", V_INH_DEFAULT))

            # Condition 0: Depotentiated inhibitor clearance (V_inh < 0.20 and salience <= 0.50)
            if is_inhibitor and v_inh < 0.20 and d_tag <= 0.50:
                should_prune = True
                reason = (
                    f"depotentiated_inhibitor ({v_inh:.3f} < 0.20, salience {d_tag:.2f} <= 0.50)"
                )
            # Condition A: Decayed below threshold for low-salience or transient decisions
            elif fid < fidelity_threshold and (d_tag <= min_salience or is_transient):
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
                crystallized_key: str | None = None
                if (
                    enable_gist_consolidation
                    and is_transient
                    and not is_inhibitor
                    and is_actionable_resolution(content, key)
                ):
                    g_key, g_content = distill_semantic_gist(content, key)
                    gists_to_record.append((g_key, g_content, SALIENCE_GIST_DEFAULT))
                    crystallized_key = g_key
                    reason += f" (crystallized_to_{g_key})"

                pruned_records.append(
                    {
                        "key": key,
                        "content": content,
                        "salience": d_tag,
                        "age_turns": age,
                        "final_fidelity": round(fid, 5),
                        "reason": reason,
                        "is_core_anchor": False,
                        "gist_crystallized": crystallized_key is not None,
                        "gist_key": crystallized_key,
                    }
                )
            else:
                survivors.append(engram)

        self.buffer.buffer = survivors
        self.total_pruned_count += len(pruned_records)

        # Anchor newly crystallized semantic gists into core buffer
        for g_key, g_content, g_sal in gists_to_record:
            self.record_semantic_gist(key=g_key, content=g_content, salience=g_sal)

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
                or (e.get("metadata") or {}).get("category") == GIST_CATEGORY
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
            meta = engram.get("metadata") or {}
            # Unreinforced inhibitor decay: V_inh <- max(0.05, V_inh * exp(-0.02 * dt))
            if meta.get("category") in ("inhibitor", "anti_pattern"):
                v_curr = float(meta.get("v_inh", V_INH_DEFAULT))
                meta["v_inh"] = max(V_INH_MIN, v_curr * math.exp(-GAMMA_INH * dt))

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

    def recall_semantic_gists(
        self, top_k: int = 4, min_fidelity: float = 0.70
    ) -> list[dict[str, Any]]:
        """Recalls active crystallized semantic gists.

        Args:
            top_k: Maximum number of gists to return.
            min_fidelity: Minimum retention fidelity threshold.

        Returns:
            List of semantic gist dictionaries sorted by salience * fidelity descending.
        """
        vitals = self.recall_vital_context(top_k=self.buffer.capacity, min_fidelity=min_fidelity)
        gists = [v for v in vitals if v.get("category") == GIST_CATEGORY]
        return gists[:top_k]

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

