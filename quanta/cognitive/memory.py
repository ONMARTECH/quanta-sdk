"""Cognitive Memory Manager with Hippocampal SWR and Conscious Synaptic Pruning."""

from __future__ import annotations

import hashlib
import math
import re
import time
from datetime import datetime
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

# Multi-Horizon Temporal Decay Constants (Milestone M2)
LAMBDA_TODAY: float = 0.02  # per hour (< 24h)
LAMBDA_YESTERDAY: float = 0.05  # per hour (24h - 48h)
LAMBDA_SPRINT: float = 0.15  # per day (2d - 7d)
LAMBDA_LONGTERM: float = 0.05  # per day (> 7d)

# Anchored continuous piecewise exponential decay boundary weights
W_TODAY_BOUNDARY: float = 1.0
W_YESTERDAY_BOUNDARY: float = math.exp(-LAMBDA_TODAY * 24.0)  # ~0.61878
W_SPRINT_BOUNDARY: float = W_YESTERDAY_BOUNDARY * math.exp(-LAMBDA_YESTERDAY * 24.0)  # ~0.18637
W_LONGTERM_BOUNDARY: float = W_SPRINT_BOUNDARY * math.exp(-LAMBDA_SPRINT * 5.0)  # ~0.08803 (120h = 5d)


def calculate_temporal_decay(
    created_at: float | datetime,
    salience: float = 1.0,
    is_core_anchor: bool = False,
    category: str = "",
    now_ts: float | datetime | None = None,
) -> float:
    """Calculates anchored continuous piecewise exponential decay with core anchor protection.

    Biomorphic continuous decay formulation:
        - dt < 24h (Today):
            w = W_TODAY_BOUNDARY * exp(-lambda_today * dt_hours)
        - 24h <= dt < 48h (Yesterday):
            w = W_YESTERDAY_BOUNDARY * exp(-lambda_yesterday * (dt_hours - 24.0))
        - 48h <= dt < 168h (Sprint, 2d - 7d):
            w = W_SPRINT_BOUNDARY * exp(-lambda_sprint * ((dt_hours - 48.0) / 24.0))
        - dt >= 168h (Long-term, > 7d):
            w = W_LONGTERM_BOUNDARY * exp(-lambda_longterm * ((dt_hours - 168.0) / 24.0))

    Guarantees strict monotonicity: w(t1) > w(t2) for all 0 <= t1 < t2, eliminating
    boundary discontinuities and inversions across horizons.

    Core anchor protection:
        Engrams with salience >= 2.0 or is_core_anchor=True or category == 'semantic_gist'
        are shielded (w_eff = 1.0).

    Args:
        created_at: Creation timestamp (Unix epoch float or datetime).
        salience: Dopaminergic salience tag.
        is_core_anchor: Whether engram is an immutable core anchor.
        category: Semantic category (e.g. 'semantic_gist', 'constraint').
        now_ts: Current timestamp for time delta calculation (default: now).

    Returns:
        Effective temporal weight w_eff in (0.0, 1.0].
    """
    if float(salience) >= 2.0 or is_core_anchor or category == GIST_CATEGORY:
        return 1.0

    if now_ts is None:
        now_sec = time.time()
    elif isinstance(now_ts, datetime):
        now_sec = now_ts.timestamp()
    else:
        now_sec = float(now_ts)

    created_sec = (
        created_at.timestamp() if isinstance(created_at, datetime) else float(created_at)
    )

    dt_sec = max(0.0, now_sec - created_sec)
    dt_hours = dt_sec / 3600.0

    if dt_hours < 24.0:
        return float(W_TODAY_BOUNDARY * math.exp(-LAMBDA_TODAY * dt_hours))
    elif dt_hours < 48.0:
        return float(W_YESTERDAY_BOUNDARY * math.exp(-LAMBDA_YESTERDAY * (dt_hours - 24.0)))
    elif dt_hours < 168.0:
        dt_days_sprint = (dt_hours - 48.0) / 24.0
        return float(W_SPRINT_BOUNDARY * math.exp(-LAMBDA_SPRINT * dt_days_sprint))
    else:
        dt_days_longterm = (dt_hours - 168.0) / 24.0
        return float(W_LONGTERM_BOUNDARY * math.exp(-LAMBDA_LONGTERM * dt_days_longterm))



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


def extract_context_tokens(text: str) -> set[str]:
    """Extracts normalized, stopword-filtered alphanumeric tokens for lexical Jaccard similarity."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "not", "this", "that", "these", "those", "it",
        "its", "as", "if", "then", "else", "when", "up", "down", "out", "over",
        "under", "again", "further", "about", "into", "through", "before", "after",
        "above", "below", "between", "both", "few", "more", "most", "other", "some",
        "such", "no", "nor", "too", "very", "can", "will", "just", "should", "now",
        "bir", "ve", "veya", "ile", "de", "da", "bu", "şu", "o", "için", "ne",
        "gibi", "çok", "daha", "en", "kadar", "her", "ama", "fakat", "lakin", "ancak",
        "çünkü", "ki", "mi", "mu", "mü", "mı",
    }
    cleaned = (
        text.replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
        .replace(":", " ")
        .replace(".", " ")
        .replace(",", " ")
    )
    words = re.findall(r"[a-zA-Z0-9\u00C0-\u017F]+", cleaned.lower())
    return {w for w in words if len(w) > 1 and w not in stopwords}


class CognitiveMemoryManager:
    """Episodic and working memory manager for CognitiveAgent AI agents.

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
        created_at: float | datetime | None = None,
        is_system_core: bool = False,
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
            created_at: Optional creation timestamp (Unix epoch float or datetime).
                Defaults to current time.
            is_system_core: Whether this engram is an immutable system-level core anchor.

        Returns:
            Buffer index of the stored engram.
        """
        if overwrite:
            self.forget(key)

        state = text_to_statevector(f"{key}:{content}", dim=self.dim)
        resolved_system_core = bool(
            is_system_core
            or category in ("system_core", GIST_CATEGORY)
            or key.startswith("system_")
        )
        resolved_is_core = (
            True
            if resolved_system_core
            else (bool(is_core_anchor) if is_core_anchor is not None else (float(salience) >= 2.0))
        )
        if created_at is None:
            now_epoch = time.time()
        elif isinstance(created_at, datetime):
            now_epoch = created_at.timestamp()
        else:
            now_epoch = float(created_at)

        metadata = {
            "key": key,
            "content": content,
            "category": category,
            "salience": float(salience),
            "is_core_anchor": resolved_is_core,
            "is_system_core": resolved_system_core,
            "created_at": now_epoch,
            "last_replayed_at": 0.0,
            "replay_count": 0,
        }

        # Smart priority eviction if buffer is at capacity
        if len(self.buffer.buffer) >= self.buffer.capacity:
            self._evict_least_salient()

        idx = self.buffer._store_single(state, dopamine_tag=salience, metadata=metadata)
        if self.buffer.buffer:
            self.buffer.buffer[-1]["created_at"] = now_epoch
            self.buffer.buffer[-1]["is_system_core"] = resolved_system_core
        return idx

    def record_transient_decision(
        self,
        key: str,
        content: str,
        salience: float = 0.5,
        category: str = "contextual_decision",
        created_at: float | datetime | None = None,
    ) -> int:
        """Records an ephemeral in-conversation decision with salience clamped to [0.2, 0.8].

        Transient decisions are marked with is_core_anchor=False, allowing them to naturally
        decay through Lindblad phase diffusion and be cleared by microglial synaptic pruning.

        Args:
            key: Concise key or title.
            content: Full text explanation or decision summary.
            salience: Priority tag (clamped to [0.2, 0.8], default: 0.5).
            category: Semantic category (default: 'contextual_decision').
            created_at: Optional creation timestamp (Unix epoch float or datetime).

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
            created_at=created_at,
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
        now_epoch = time.time()
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
            "created_at": now_epoch,
            "last_replayed_at": 0.0,
            "replay_count": 0,
        }

        if len(self.buffer.buffer) >= self.buffer.capacity:
            self._evict_least_salient()

        idx = self.buffer._store_single(state, dopamine_tag=clamped_sal, metadata=metadata)
        if self.buffer.buffer:
            self.buffer.buffer[-1]["created_at"] = now_epoch
        return idx

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

    def apply_wake_sleep_plasticity(
        self,
        phase: Any,
        shy_factor: float = 0.90,
        fidelity_threshold: float = 0.70,
        min_salience_prune: float = 0.80,
    ) -> dict[str, Any]:
        """Applies Hinton Wake-Sleep biophysical synaptic plasticity.

        During SWS (phase == SleepPhase.NREM_SWS):
            - Proactively downscales non-core engrams: S_e <- max(0.20, S_e * shy_factor).
            - Prunes noisy ephemeral engrams (S <= 0.80 and (fidelity < 0.70 or older non-anchor items)).
            - Returns dict with 'phase', 'downscaled_count', and 'pruned_count'.

        During REM (phase == SleepPhase.REM_DREAM):
            - Selectively potentiates generalizing cross-domain bisociations and high-salience
              emergent RFC engrams: S >= 2.50, marking them as core anchors to prevent eviction.
            - Returns dict with 'phase' and 'potentiated_count'.

        Args:
            phase: SleepPhase enum or string ('NREM_SWS', 'REM_DREAM').
            shy_factor: Synaptic downscale factor for non-core engrams in SWS (default 0.90).
            fidelity_threshold: Fidelity cut-off for pruning degraded noise in SWS (default 0.70).
            min_salience_prune: Salience threshold below which noise is pruned in SWS (default 0.80).

        Returns:
            Dictionary containing plasticity outcome metrics.
        """
        phase_str = phase.value if hasattr(phase, "value") else str(phase)
        if "NREM" in phase_str or "SWS" in phase_str:
            downscaled_count = 0
            survivors = []
            pruned_count = 0

            for engram in self.buffer.buffer:
                d_tag = float(engram["dopamine_tag"])
                meta = engram.get("metadata") or {}
                cat = meta.get("category", "")
                age = float(engram.get("age", 0))

                is_core = bool(
                    meta.get("is_core_anchor", False)
                    or d_tag >= 2.0
                    or cat in (GIST_CATEGORY, "subconscious_dream")
                    or meta.get("is_system_core", False)
                    or engram.get("is_system_core", False)
                )

                if not is_core:
                    new_s = max(0.0, d_tag * shy_factor)
                    engram["dopamine_tag"] = new_s
                    downscaled_count += 1
                    current_salience = new_s
                else:
                    current_salience = d_tag

                # Compute fidelity
                psi_0 = engram["pristine_state"]
                psi_t = engram["degraded_state"]
                overlap = torch.vdot(psi_0, psi_t)
                fid = float((torch.abs(overlap) ** 2).item())

                # Check pruning of noisy ephemeral engrams:
                # S <= 0.80 and (fidelity < 0.70 or older non-anchor items or noise tags),
                # OR unreinforced non-core salience decayed to viability floor (S <= 0.30)
                is_noisy_ephemeral = (
                    not is_core
                    and (
                        current_salience <= 0.30
                        or (
                            current_salience <= min_salience_prune
                            and (
                                fid < fidelity_threshold
                                or age > 10.0
                                or cat in (
                                    "ephemeral",
                                    "noise",
                                    "scratchpad",
                                    "transient",
                                    "transient_task",
                                )
                                or any(
                                    t in ("noise", "ephemeral", "temp", "scratchpad")
                                    for t in (meta.get("tags") or [])
                                )
                            )
                        )
                    )
                )

                if is_noisy_ephemeral:
                    pruned_count += 1
                else:
                    survivors.append(engram)

            self.buffer.buffer = survivors
            self.total_pruned_count += pruned_count
            return {
                "phase": "NREM_SWS",
                "downscaled_count": downscaled_count,
                "pruned_count": pruned_count,
            }

        elif "REM" in phase_str:
            potentiated_count = 0
            dream_engram_indices = []
            for idx, engram in enumerate(self.buffer.buffer):
                meta = engram.get("metadata") or {}
                cat = meta.get("category", "")
                key = str(meta.get("key", ""))
                tags = meta.get("tags") or []
                if isinstance(tags, str):
                    tags = [tags]

                is_bisociation = (
                    cat
                    in (
                        "subconscious_dream",
                        "hybrid_dream",
                        "bisociation",
                        "rfc_hybrid",
                        "cross_domain",
                    )
                    or any(
                        t in ("bisociation", "hybrid", "cross_project", "rfc") for t in tags
                    )
                    or key.startswith(("rfc_", "hybrid_", "bisociation_", "dream_"))
                    or "projects_paired" in meta
                    or "domain_pair" in meta
                    or meta.get("hybrid") is True
                    or "rfc_path" in meta
                )
                is_emergent_rfc = (
                    "rfc" in key.lower()
                    or "rfc" in cat.lower()
                    or "rfc" in str(tags).lower()
                    or float(meta.get("confidence", 0.0)) >= 0.85
                )

                if is_bisociation or is_emergent_rfc:
                    dream_engram_indices.append(idx)

            # Cap active core dream anchors (max 50) to prevent unbounded memory growth
            dream_engram_indices.sort(
                key=lambda i: float(self.buffer.buffer[i].get("created_at", 0.0)),
                reverse=True,
            )
            max_dream_anchors = 50
            for rank, idx in enumerate(dream_engram_indices):
                engram = self.buffer.buffer[idx]
                meta = engram.get("metadata") or {}
                if rank < max_dream_anchors:
                    d_tag = float(engram["dopamine_tag"])
                    engram["dopamine_tag"] = max(2.50, d_tag)
                    meta["is_core_anchor"] = True
                    engram["is_core_anchor"] = True
                    engram["metadata"] = meta
                    potentiated_count += 1
                else:
                    meta["is_core_anchor"] = False
                    engram["is_core_anchor"] = False
                    engram["dopamine_tag"] = min(float(engram["dopamine_tag"]), 1.50)
                    engram["metadata"] = meta

            return {
                "phase": "REM_DREAM",
                "potentiated_count": potentiated_count,
            }

        return {
            "phase": phase_str,
            "downscaled_count": 0,
            "pruned_count": 0,
            "potentiated_count": 0,
        }

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
                or (e.get("metadata") or {}).get("is_system_core", False)
                or e.get("is_system_core", False)
            )
        ]

        # If all engrams in buffer are core anchors, check if any are dream candidates
        # (e.g. subconscious dreams or bisociations) that can be pruned to preserve capacity.
        if not candidate_indices:
            dream_candidates = [
                idx
                for idx, e in enumerate(self.buffer.buffer)
                if (
                    (e.get("metadata") or {}).get("category")
                    in ("subconscious_dream", "hybrid_dream", "bisociation", "rfc_hybrid")
                    or str((e.get("metadata") or {}).get("key", "")).startswith(
                        ("dream_", "bisociation_", "hybrid_", "rfc_")
                    )
                )
                and not (
                    (e.get("metadata") or {}).get("is_system_core", False)
                    or e.get("is_system_core", False)
                    or (e.get("metadata") or {}).get("category") in ("system_core", GIST_CATEGORY)
                    or str((e.get("metadata") or {}).get("key", "")).startswith("system_")
                )
            ]
            if dream_candidates:
                candidate_indices = dream_candidates
            else:
                # If all engrams in buffer are immutable core anchors, buffer expands gracefully
                # to prevent FIFO loss of core anchors.
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

    def recall_reverse_chronological_cascade(
        self,
        top_k: int = 10,
        min_fidelity: float = 0.40,
        now_ts: float | None = None,
        query: str = "",
    ) -> list[dict[str, Any]]:
        """Traverses memories in reverse-chronological order (newest first, t_now -> t_past).

        Prioritizes current day's active problems while rippling backward into foundational
        decisions. Applies multi-horizon temporal decay and dopaminergic core anchor protection.

        Args:
            top_k: Maximum number of memories to return (default: 10).
            min_fidelity: Minimum retention fidelity threshold.
            now_ts: Optional evaluation timestamp (epoch seconds).
            query: Optional query for relevance boosting.

        Returns:
            List of memory dictionaries sorted in reverse-chronological order (newest first).
        """
        if not self.buffer.buffer:
            return []

        now_sec = time.time() if now_ts is None else float(now_ts)
        candidates: list[dict[str, Any]] = []

        for idx, engram in enumerate(self.buffer.buffer):
            psi_0 = engram["pristine_state"]
            psi_t = engram["degraded_state"]
            overlap = torch.vdot(psi_0, psi_t)
            fidelity = float((torch.abs(overlap) ** 2).item())

            if fidelity < min_fidelity:
                continue

            meta = engram.get("metadata") or {}
            key = meta.get("key", "")
            content = meta.get("content", "")
            cat = meta.get("category", "")
            salience = float(engram["dopamine_tag"])
            age_turns = round(float(engram["age"]), 1)

            # Determine created_at timestamp: engram.get first, then meta.get
            created_at = engram.get("created_at")
            if created_at is None:
                created_at = meta.get("created_at")
            if created_at is None:
                created_at = max(0.0, now_sec - (age_turns * 60.0))
            if isinstance(created_at, datetime):
                created_at = created_at.timestamp()
            else:
                created_at = float(created_at)

            # Synchronize created_at across metadata and engram
            meta["created_at"] = created_at
            engram["created_at"] = created_at

            dt_hours = max(0.0, (now_sec - created_at) / 3600.0)
            if dt_hours < 24.0:
                horizon = "today"
            elif dt_hours < 48.0:
                horizon = "yesterday"
            elif dt_hours < 168.0:
                horizon = "sprint"
            else:
                horizon = "longterm"

            is_core = bool(
                meta.get("is_core_anchor", salience >= 2.0)
                or salience >= 2.0
                or cat == GIST_CATEGORY
            )

            w_eff = calculate_temporal_decay(
                created_at=created_at,
                salience=salience,
                is_core_anchor=is_core,
                category=cat,
                now_ts=now_sec,
            )

            # Replay score calculation
            base_score = salience * fidelity * w_eff
            rel_boost = 0.0
            if query and content:
                q_words = set(query.lower().split())
                c_words = set(content.lower().split())
                overlap_len = len(q_words & c_words)
                if overlap_len > 0:
                    rel_boost = min(1.0, overlap_len / max(1, len(q_words)))
            replay_score = base_score * (1.0 + 0.25 * rel_boost)

            status = (
                "pristine"
                if fidelity >= 0.999
                else (
                    "consolidated"
                    if fidelity >= 0.95
                    else ("decaying" if fidelity >= 0.70 else "obsolete")
                )
            )

            candidates.append(
                {
                    "index": idx,
                    "key": key,
                    "content": content,
                    "category": cat,
                    "salience": salience,
                    "age_turns": age_turns,
                    "created_at": created_at,
                    "age_hours": round(dt_hours, 2),
                    "horizon": horizon,
                    "temporal_weight": round(w_eff, 5),
                    "replay_score": round(replay_score, 5),
                    "retention_fidelity": round(fidelity, 5),
                    "retention_pct": round(fidelity * 100, 2),
                    "status": status,
                    "is_core_anchor": is_core,
                }
            )

        if not candidates:
            return []

        # Sort reverse-chronologically (newest first, t_now -> t_past)
        if len(candidates) <= top_k:
            candidates.sort(key=lambda x: x["created_at"], reverse=True)
            return candidates

        # Budget allocation to guarantee rippling backward into foundational decisions:
        core_candidates = [c for c in candidates if c["is_core_anchor"]]
        non_core = [c for c in candidates if not c["is_core_anchor"]]

        core_candidates.sort(key=lambda x: x["salience"] * x["retention_fidelity"], reverse=True)
        non_core.sort(key=lambda x: x["created_at"], reverse=True)

        num_core_slots = min(len(core_candidates), max(1, top_k // 3))
        selected_core = core_candidates[:num_core_slots]
        core_keys = {c["key"] for c in selected_core}

        remaining_slots = top_k - len(selected_core)
        selected_recent = [c for c in non_core if c["key"] not in core_keys][:remaining_slots]

        merged = selected_core + selected_recent
        if len(merged) < top_k and len(core_candidates) > len(selected_core):
            additional_core = [
                c for c in core_candidates if c["key"] not in core_keys
            ][:(top_k - len(merged))]
            merged.extend(additional_core)

        merged.sort(key=lambda x: x["created_at"], reverse=True)
        return merged

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

    def compute_context_resonance(
        self, context: str, engram: dict[str, Any] | int, alpha: float = 0.6
    ) -> float:
        """Computes composite Hilbert-space inner product overlap and token Jaccard similarity.

        Mathematical formulation:
            1. psi_context = text_to_statevector(context, dim=self.dim)
            2. F_hilbert(C, e) = |<psi_context | psi_engram>|^2 in [0.0, 1.0]
            3. J(C, e) = |T(C) cap T(e)| / |T(C) cup T(e)| in [0.0, 1.0]
            4. R(C, e) = alpha * F_hilbert(C, e) + (1.0 - alpha) * J(C, e)

        Args:
            context: Active prompt or task context string.
            engram: Engram dictionary or buffer index.
            alpha: Weight for Hilbert overlap (default: 0.6; 1 - alpha for Jaccard).

        Returns:
            Composite contextual resonance score R(C, e) in [0.0, 1.0].
        """
        if isinstance(engram, int):
            if 0 <= engram < len(self.buffer.buffer):
                e = self.buffer.buffer[engram]
            else:
                return 0.0
        elif isinstance(engram, dict):
            e = engram
        else:
            return 0.0

        # 1. Statevector mapping & Hilbert-space inner product overlap
        psi_e = e.get("degraded_state")
        if psi_e is None:
            psi_e = e.get("pristine_state")
        if psi_e is None:
            meta = e.get("metadata") if isinstance(e.get("metadata"), dict) else e
            key = str(meta.get("key", e.get("key", "")))
            content = str(meta.get("content", e.get("content", "")))
            psi_e = text_to_statevector(f"{key}:{content}", dim=self.dim)

        psi_context = text_to_statevector(context, dim=self.dim)
        psi_context = psi_context.to(device=psi_e.device, dtype=psi_e.dtype)
        overlap = torch.vdot(psi_context, psi_e)
        f_hilbert = float((torch.abs(overlap) ** 2).item())
        f_hilbert = max(0.0, min(1.0, f_hilbert))

        # 2. Lexical keyword Jaccard similarity
        meta = e.get("metadata") if isinstance(e.get("metadata"), dict) else e
        key = str(meta.get("key", e.get("key", "")))
        content = str(meta.get("content", e.get("content", "")))
        cat = str(meta.get("category", e.get("category", "")))
        tags = meta.get("tags") or e.get("tags") or []
        tags_str = " ".join(tags) if isinstance(tags, (list, tuple)) else str(tags)
        engram_text = f"{key} {content} {cat} {tags_str}"

        t_context = extract_context_tokens(context)
        t_engram = extract_context_tokens(engram_text)
        union = t_context | t_engram
        jaccard = 0.0 if not union else float(len(t_context & t_engram) / len(union))
        jaccard = max(0.0, min(1.0, jaccard))

        # 3. Composite contextual resonance
        clamped_alpha = max(0.0, min(1.0, float(alpha)))
        res = clamped_alpha * f_hilbert + (1.0 - clamped_alpha) * jaccard
        return float(max(0.0, min(1.0, res)))

    def recall_context_adaptive_swr(
        self,
        context: str = "",
        top_k_transient: int = 3,
        min_resonance: float = 0.12,
        min_fidelity: float = 0.40,
        context_prompt: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Recalls memories via context-adaptive Sharp-Wave Ripple (SWR) resonance.

        Guarantees 100% unconditional preservation of Core Anchors (is_core_anchor=True,
        salience >= 2.0, or category in {'system_core', 'semantic_gist', 'constraint'}).
        Dynamically filters transient engrams to top-K resonant elements with priority
        P(C, e) = R(C, e) * salience * F(t).

        Args:
            context: Active prompt or task context string.
            top_k_transient: Maximum number of transient resonant engrams to return.
            min_resonance: Minimum contextual resonance threshold for transient engrams.
            min_fidelity: Minimum biological retention fidelity threshold for transient engrams.
            context_prompt: Optional alias for context.

        Returns:
            Dict containing lists: 'core_anchors', 'semantic_gists', 'transient_decisions', 'inhibitors'.
        """
        ctx = context or (context_prompt or "")
        if not self.buffer.buffer:
            return {
                "core_anchors": [],
                "semantic_gists": [],
                "transient_decisions": [],
                "inhibitors": [],
            }

        core_anchors: list[dict[str, Any]] = []
        semantic_gists: list[dict[str, Any]] = []
        inhibitors: list[dict[str, Any]] = []
        transient_candidates: list[dict[str, Any]] = []

        for idx, engram in enumerate(self.buffer.buffer):
            psi_0 = engram["pristine_state"]
            psi_t = engram["degraded_state"]
            overlap = torch.vdot(psi_0, psi_t)
            fidelity = float((torch.abs(overlap) ** 2).item())

            meta = engram.get("metadata") or {}
            key = str(meta.get("key", ""))
            content = str(meta.get("content", ""))
            cat = str(meta.get("category", ""))
            salience = float(engram.get("dopamine_tag", 1.0))
            age_turns = round(float(engram.get("age", 0.0)), 1)
            is_core_flag = bool(meta.get("is_core_anchor", False) or engram.get("is_core_anchor", False))
            is_sys_core = bool(meta.get("is_system_core", False) or engram.get("is_system_core", False))

            # Core Anchor criterion
            is_core = bool(
                is_core_flag
                or is_sys_core
                or salience >= 2.0
                or cat in ("system_core", GIST_CATEGORY, "constraint")
                or key.startswith("system_")
            )

            status = (
                "pristine"
                if fidelity >= 0.999
                else (
                    "consolidated"
                    if fidelity >= 0.95
                    else ("decaying" if fidelity >= 0.70 else "obsolete")
                )
            )

            res = self.compute_context_resonance(ctx, engram, alpha=0.6)
            priority = res * salience * fidelity

            entry: dict[str, Any] = {
                "index": idx,
                "key": key,
                "content": content,
                "category": cat,
                "salience": round(salience, 2),
                "age_turns": age_turns,
                "retention_fidelity": round(fidelity, 5),
                "fidelity": round(fidelity, 5),
                "retention_pct": round(fidelity * 100.0, 2),
                "status": status,
                "is_core_anchor": is_core,
                "resonance": round(res, 5),
                "resonance_score": round(res, 5),
                "priority": round(priority, 5),
            }

            if cat in ("inhibitor", "anti_pattern"):
                v_inh = float(meta.get("v_inh", V_INH_DEFAULT))
                entry["v_inh"] = round(v_inh, 4)
                entry["context_tags"] = dict(meta.get("context_tags", {}))
                entry["consecutive_failures"] = int(meta.get("consecutive_failures", 0))
                if (v_inh >= 0.20 and fidelity >= min_fidelity) or is_core:
                    inhibitors.append(entry)
                continue

            if is_core:
                if cat == GIST_CATEGORY:
                    semantic_gists.append(entry)
                else:
                    core_anchors.append(entry)
                continue

            # Transient engram filtering
            if fidelity < min_fidelity:
                continue
            if res < min_resonance:
                continue

            transient_candidates.append(entry)

        # Sort groups
        core_anchors.sort(key=lambda x: x["salience"] * x["retention_fidelity"], reverse=True)
        semantic_gists.sort(key=lambda x: x["salience"] * x["retention_fidelity"], reverse=True)
        inhibitors.sort(key=lambda x: float(x.get("v_inh", 0.5)) * x["salience"], reverse=True)

        transient_candidates.sort(key=lambda x: x["priority"], reverse=True)
        selected_transient = transient_candidates[:top_k_transient]

        return {
            "core_anchors": core_anchors,
            "semantic_gists": semantic_gists,
            "transient_decisions": selected_transient,
            "inhibitors": inhibitors,
        }

    def recall_contextual_resonance(
        self,
        context_prompt: str = "",
        top_k_transient: int = 3,
        min_resonance: float = 0.12,
        min_fidelity: float = 0.40,
        context: str = "",
    ) -> list[dict[str, Any]]:
        """Convenience method returning a combined flat list of all recalled engrams."""
        ctx = context_prompt or context
        structured = self.recall_context_adaptive_swr(
            context=ctx,
            top_k_transient=top_k_transient,
            min_resonance=min_resonance,
            min_fidelity=min_fidelity,
        )
        result: list[dict[str, Any]] = []
        result.extend(structured.get("core_anchors", []))
        result.extend(structured.get("semantic_gists", []))
        result.extend(structured.get("transient_decisions", []))
        result.extend(structured.get("inhibitors", []))
        return result

    def format_context_adaptive_swr_prompt(
        self,
        context: str = "",
        top_k_transient: int = 3,
        min_resonance: float = 0.12,
        min_fidelity: float = 0.40,
        context_prompt: str | None = None,
    ) -> str:
        """Formats context-adaptive [Quanta Bilişsel Çıpa | SWR Replay] prompt string.

        Guarantees 100% preservation of core anchors and semantic gists while pruning
        off-topic transient items to reduce prompt token footprint.
        """
        ctx = context or (context_prompt or "")
        recalled = self.recall_context_adaptive_swr(
            context=ctx,
            top_k_transient=top_k_transient,
            min_resonance=min_resonance,
            min_fidelity=min_fidelity,
        )

        def _fmt_fid(fid: float) -> str:
            pct = fid * 100.0
            if pct >= 99.98:
                return "99.98%"
            return f"{pct:.2f}%"

        core_items = [
            f"{c['key']} ({_fmt_fid(float(c.get('retention_fidelity', c.get('fidelity', 1.0))))})"
            for c in recalled.get("core_anchors", [])
        ]
        gist_items = [
            f"{g['key']} ({_fmt_fid(float(g.get('retention_fidelity', g.get('fidelity', 1.0))))})"
            for g in recalled.get("semantic_gists", [])
        ]
        transient_items = [
            f"{t['key']} ({_fmt_fid(float(t.get('retention_fidelity', t.get('fidelity', 1.0))))})"
            for t in recalled.get("transient_decisions", [])
        ]
        inhibitor_items = []
        for inh in recalled.get("inhibitors", []):
            v_val = float(inh.get("v_inh", 0.50))
            fid_val = float(inh.get("retention_fidelity", inh.get("fidelity", 0.9998)))
            inhibitor_items.append(f"{inh['key']} (V_inh={v_val:.2f}, {_fmt_fid(fid_val)})")

        lines = ["[Quanta Bilişsel Çıpa | SWR Replay]:"]
        if core_items:
            lines.append(f"  🔒 Çekirdek: {', '.join(core_items)}")
        if gist_items:
            lines.append(f"  🧠 Özüt: {', '.join(gist_items)}")
        if transient_items:
            lines.append(f"  ⚡ Geçici: {', '.join(transient_items)}")
        if inhibitor_items:
            lines.append(f"  🚫 İnhibitör / Anti-Pattern: {', '.join(inhibitor_items)}")

        return "\n".join(lines)

    def prune_off_topic_transient(
        self,
        context: str = "",
        min_resonance: float = 0.08,
        min_age_turns: float = 2.0,
        context_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Actively evicts transient engrams that are completely irrelevant to active context.

        Core anchors and semantic gists are strictly immune to off-topic pruning.

        Args:
            context: Active prompt or task context string.
            min_resonance: Threshold below which transient engrams are evicted (default: 0.08).
            min_age_turns: Minimum turn age before off-topic eviction applies (default: 2.0).
            context_prompt: Optional alias for context.

        Returns:
            List of pruned engram diagnostic dictionaries.
        """
        ctx = context or (context_prompt or "")
        if not self.buffer.buffer:
            return []

        survivors = []
        pruned_records: list[dict[str, Any]] = []

        for engram in self.buffer.buffer:
            meta = engram.get("metadata") or {}
            salience = float(engram.get("dopamine_tag", 1.0))
            cat = str(meta.get("category", ""))
            key = str(meta.get("key", "unnamed"))
            content = str(meta.get("content", ""))
            age = float(engram.get("age", 0.0))

            is_core = bool(
                meta.get("is_core_anchor", False)
                or engram.get("is_core_anchor", False)
                or salience >= 2.0
                or cat in ("system_core", GIST_CATEGORY, "constraint")
                or meta.get("is_system_core", False)
                or engram.get("is_system_core", False)
                or key.startswith("system_")
            )

            if is_core:
                survivors.append(engram)
                continue

            # Transient / non-core engram
            res = self.compute_context_resonance(ctx, engram, alpha=0.6)
            if age >= min_age_turns and res < min_resonance:
                psi_0 = engram["pristine_state"]
                psi_t = engram["degraded_state"]
                overlap = torch.vdot(psi_0, psi_t)
                fid = float((torch.abs(overlap) ** 2).item())

                pruned_records.append(
                    {
                        "key": key,
                        "content": content,
                        "salience": round(salience, 2),
                        "age_turns": round(age, 1),
                        "retention_fidelity": round(fid, 5),
                        "resonance": round(res, 5),
                        "reason": f"off_topic_resonance ({res:.3f} < {min_resonance})",
                        "is_core_anchor": False,
                    }
                )
            else:
                survivors.append(engram)

        self.buffer.buffer = survivors
        self.total_pruned_count += len(pruned_records)
        return pruned_records


