#!/usr/bin/env python3
"""Quanta Autonomous Subconscious Hook for CognitiveAgent.

Runs silently and automatically on CognitiveAgent lifecycle events (PreInvocation).
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

CANONICAL_PROD_KEYS = frozenset({
    'insight_continuous_resonance',
    'insight_landauer_efficiency',
    'insight_anti_rumination_reset',
    'insight_quantum_rem_sleep',
    'insight_darwin_qos_optimization',
    'insight_autonomous_quantum_resonance',
    'insight_concurrent_preemption',
    'insight_qos',
    'insight_pauli_commutation_algebra',
    'insight_clifford_tableau_simulation',
    'insight_surface_code_mwpm_threshold',
    'insight_union_find_syndrome_decoder',
    'insight_mps_svd_truncation_bound',
    'insight_tebd_trotter_decomposition',
    'insight_sabre_swap_routing_heuristic',
    'insight_kak_two_qubit_synthesis',
    'insight_solovay_kitaev_approximation',
    'insight_vqe_parameter_shift_gradient',
    'insight_qaoa_maxcut_mixer_commutation',
    'insight_barren_plateau_variance_scaling',
    'insight_quantum_natural_gradient_fubini',
    'insight_qsvm_fidelity_kernel_gram',
    'insight_grover_amplitude_amplification',
    'insight_quantum_phase_estimation_qft',
    'insight_shor_period_finding_modular',
    'insight_bb84_qkd_sifted_key_entropy',
    'insight_lwe_lattice_babai_rounding',
    'insight_lindblad_cptp_kraus_trace',
    'insight_amplitude_damping_t1_relaxation',
    'insight_phase_damping_t2_decoherence',
    'insight_readout_confusion_matrix_inversion',
    'insight_zero_noise_extrapolation_richardson',
    'insight_probabilistic_error_cancellation',
    'insight_dynamical_decoupling_xy4_sequence',
    'insight_ionq_molmer_sorensen_native_compilation',
    'insight_ibm_heavy_hex_bridge_cancellation',
    'insight_neutral_atom_rydberg_blockade_zone',
    'insight_magic_state_distillation_15_to_1',
    'insight_lattice_surgery_logical_cnot',
    'insight_floquet_honeycomb_dynamical_codes',
    'insight_ldpc_quantum_expander_codes',
    'insight_jordan_wigner_fermionic_mapping',
    'insight_qubit_tapering_z2_symmetries',
    'insight_commuting_pauli_clique_grouping',
    'insight_classical_shadows_randomized_pauli',
    'insight_quantum_zeno_frequent_projection',
    'insight_anti_zeno_accelerated_transition',
    'insight_noether_symmetry_charge_conservation',
    'insight_hippocampal_swr_compressed_replay',
    'insight_csf_dielectric_dephasing_shield',
    'insight_ebbinghaus_power_law_consolidation',
    'insight_fuzzy_trace_gist_orthogonalization',
    'insight_contextual_resonance_top_k_filtering',
})


def format_fidelity(fid: float) -> str:
    """Formats heuristic retention/priority weight (capped at 99.98%, not an empirical physical probability)."""
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

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(self.get("key") or self.get("id") or self.get("rule_name"))

    def __str__(self) -> str:
        return str(self.get("key") or self.get("rule_name") or super().__str__())


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


class FastBiomorphicMemory:
    """Ultra-fast stdlib-only biomorphic engram manager for CognitiveAgent hooks."""

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
                if not (
                    e.get("is_core_anchor", False)
                    or float(e.get("salience", 0.0)) >= 2.0
                    or e.get("category") == "semantic_gist"
                )
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

    def record_semantic_gist(
        self,
        key: str,
        content: str,
        salience: float = 1.85,
        is_core_anchor: bool = True,
        source_key: str | None = None,
    ) -> None:
        """Records a crystallized semantic gist into core memory (S >= 1.8, is_core_anchor=True)."""
        sal = max(1.80, float(salience))
        self.record(
            key=key,
            content=content,
            salience=sal,
            category="semantic_gist",
            is_core_anchor=is_core_anchor,
        )

    def record_inhibitor(
        self,
        key: str,
        content: str,
        v_inh: float = 0.50,
        salience: float = 1.50,
        context_tags: dict[str, Any] | None = None,
        category: str = "inhibitor",
    ) -> None:
        """Records an adaptive negative engram representing an anti-pattern or failed action."""
        sal = (
            float(salience)
            if isinstance(salience, (int, float))
            and not isinstance(salience, bool)
            and math.isfinite(salience)
            else 1.50
        )
        sal = max(0.01, min(3.5, sal))
        v_val = (
            float(v_inh)
            if isinstance(v_inh, (int, float))
            and not isinstance(v_inh, bool)
            and math.isfinite(v_inh)
            else 0.50
        )
        v_val = max(0.05, min(0.9998, v_val))
        core_flag = bool(sal >= 2.0)
        tags = dict(context_tags) if context_tags else {}

        # If key exists, update in-place
        for e in self.engrams:
            if e["key"] == key:
                e["content"] = content
                e["salience"] = sal
                e["category"] = category
                e["fidelity"] = 0.9998
                e["v_inh"] = v_val
                e["is_core_anchor"] = core_flag
                e["context_tags"] = tags
                e.setdefault("consecutive_failures", 1)
                e.setdefault("consecutive_successes", 0)
                e.setdefault("context_divergence", [])
                return

        # Capacity management: evict lowest scoring non-core engram if full
        if len(self.engrams) >= self.capacity:
            candidates = [
                (idx, e)
                for idx, e in enumerate(self.engrams)
                if not (
                    e.get("is_core_anchor", False)
                    or float(e.get("salience", 0.0)) >= 2.0
                    or e.get("category") == "semantic_gist"
                )
            ]
            if candidates:
                candidates.sort(
                    key=lambda item: float(item[1]["salience"]) * float(item[1]["fidelity"])
                )
                evict_idx = candidates[0][0]
                self.engrams.pop(evict_idx)
                self.total_pruned_count += 1

        self.engrams.append({
            "key": key,
            "content": content,
            "salience": float(sal),
            "category": category,
            "fidelity": 0.9998,
            "age": 0,
            "is_core_anchor": core_flag,
            "v_inh": float(v_val),
            "context_tags": tags,
            "consecutive_failures": 1,
            "consecutive_successes": 0,
            "context_divergence": [],
        })

    def potentiate_inhibitor(
        self,
        key: str,
        context_tags: dict[str, Any] | None = None,
        boost: float = 0.25,
    ) -> float:
        """Long-Term Potentiation (LTP): Deepens inhibition weight V_inh upon repeated failure."""
        target = None
        for e in self.engrams:
            if e["key"] == key:
                target = e
                break

        if target is None:
            self.record_inhibitor(
                key=key,
                content=f"Inhibitory pattern for {key}",
                v_inh=0.50,
                salience=1.50,
                context_tags=context_tags,
            )
            for e in self.engrams:
                if e["key"] == key:
                    target = e
                    break

        if target is None:
            return 0.50

        k_fail = int(target.get("consecutive_failures", 0)) + 1
        target["consecutive_failures"] = k_fail
        target["consecutive_successes"] = 0

        if context_tags:
            target.setdefault("context_tags", {}).update(context_tags)

        # Asymptotic potentiation: V_inh <- min(0.9998, V_inh + boost * (0.9998 - V_inh))
        curr_v = float(target.get("v_inh", 0.50))
        new_v = min(0.9998, curr_v + float(boost) * (0.9998 - curr_v))
        target["v_inh"] = new_v

        # Salience scaling (threat awakening): S <- min(3.5, S + 0.35 * (1 + 0.1 * k_fail))
        curr_sal = float(target.get("salience", 1.50))
        new_sal = min(3.5, curr_sal + 0.35 * (1.0 + 0.1 * k_fail))
        target["salience"] = new_sal

        # Critical threshold crossing: core anchor promotion
        if new_sal >= 2.0:
            target["is_core_anchor"] = True

        target["fidelity"] = 0.9998
        return new_v

    def depress_inhibitor(
        self,
        key: str,
        context_tags: dict[str, Any] | None = None,
        decay: float = 0.30,
    ) -> float:
        """Long-Term Depression (LTD): Relaxes V_inh and documents context divergence on success."""
        target = None
        for e in self.engrams:
            if e["key"] == key:
                target = e
                break

        if target is None:
            return 0.0

        target["consecutive_successes"] = int(target.get("consecutive_successes", 0)) + 1
        target["consecutive_failures"] = 0

        # Context divergence calculation: Delta C = {tag: (old, new) | old != new}
        if context_tags:
            existing_tags = target.setdefault("context_tags", {})
            diff = {}
            for k in set(existing_tags.keys()) | set(context_tags.keys()):
                old_val = existing_tags.get(k)
                new_val = context_tags.get(k)
                if old_val != new_val:
                    diff[k] = (old_val, new_val)
            if diff:
                target.setdefault("context_divergence", []).append(diff)
            existing_tags.update(context_tags)

        # Synaptic weight relaxation (multiplicative decay): V_inh <- max(0.05, V_inh * (1 - decay))
        curr_v = float(target.get("v_inh", 0.50))
        new_v = max(0.05, curr_v * (1.0 - float(decay)))
        target["v_inh"] = new_v

        # Salience relaxation
        curr_sal = float(target.get("salience", 1.50))
        new_sal = max(0.20, curr_sal - 0.35)
        target["salience"] = new_sal

        # Demote core anchor if salience falls below threshold
        if new_sal < 2.0:
            target["is_core_anchor"] = False

        return new_v

    def recall_inhibitors(
        self,
        top_k: int = 3,
        min_v_inh: float = 0.20,
    ) -> list[dict]:
        """Recalls active inhibitors ranked by (V_inh * salience)."""
        active = [
            e
            for e in self.engrams
            if e.get("category") in ("inhibitor", "anti_pattern")
            and float(e.get("v_inh", 0.50)) >= min_v_inh
        ]
        active.sort(
            key=lambda item: float(item.get("v_inh", 0.50)) * float(item.get("salience", 1.0)),
            reverse=True,
        )
        return active[:top_k]

    def step(self, dt: float = 1.0) -> None:
        dim_factor = 1.0 / DEFAULT_DIM
        for e in self.engrams:
            curr_age = float(e.get("age", 0))
            new_age = round(curr_age + dt, 4)
            e["age"] = int(new_age) if new_age.is_integer() else new_age
            # Unreinforced inhibitor synaptic weight decay
            if e.get("category") in ("inhibitor", "anti_pattern"):
                v_curr = float(e.get("v_inh", 0.50))
                e["v_inh"] = max(0.05, v_curr * math.exp(-0.02 * dt))

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
        enable_gist_consolidation: bool = True,
    ) -> list[PrunedEngram]:
        survivors = []
        pruned_records: list[PrunedEngram] = []
        gists_to_crystallize: list[tuple[str, str, float]] = []
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        turn_pruned = int(turn if turn is not None else getattr(self, "current_turn", 0))

        for e in self.engrams:
            is_core = bool(
                e.get("is_core_anchor", False)
                or float(e.get("salience", 0.0)) >= 2.0
                or e.get("category") == "semantic_gist"
            )
            if is_core:
                survivors.append(e)
                continue

            is_transient = e.get("is_core_anchor") is False and float(e.get("salience", 1.0)) <= 0.8
            is_inhibitor = e.get("category") in ("inhibitor", "anti_pattern")
            v_inh = float(e.get("v_inh", 0.50))
            fid = float(e.get("fidelity", 1.0))
            sal = float(e.get("salience", 1.0))
            content = str(e.get("content", ""))
            key = str(e.get("key", ""))

            depotentiated = is_inhibitor and (v_inh < 0.20 and sal <= 0.50)
            decayed = (
                (sal <= min_salience or is_transient)
                and fid < fidelity_threshold
            )
            age_val = float(e.get("age", 0))
            aged = (max_age is not None and age_val > max_age)

            if decayed or aged or depotentiated:
                if depotentiated:
                    reason = (
                        f"depotentiated_inhibitor ({v_inh:.3f} < 0.20, salience {sal:.2f} <= 0.50)"
                    )
                elif decayed and aged:
                    reason = (
                        f"fidelity_and_age ({fid:.3f} < {fidelity_threshold:.3f}, "
                        f"age {age_val:.0f} > {max_age})"
                    )
                elif decayed:
                    reason = f"fidelity_decayed ({fid:.3f} < {fidelity_threshold:.3f})"
                else:
                    reason = f"age_exceeded ({age_val:.0f} > {max_age})"

                # Fuzzy-Trace Semantic Gist Extraction
                crystallized_key = None
                if (
                    enable_gist_consolidation
                    and is_transient
                    and not is_inhibitor
                    and is_actionable_resolution(content, key)
                ):
                    g_key, g_content = distill_semantic_gist(content, key)
                    gists_to_crystallize.append((g_key, g_content, 1.85))
                    crystallized_key = g_key
                    reason += f" (crystallized_to_{g_key})"

                decayed_sal = round(sal * fid, 4)
                record = PrunedEngram({
                    "timestamp": now_ts,
                    "rule_name": key,
                    "id": key,
                    "key": key,
                    "reason": reason,
                    "decayed_salience": decayed_sal,
                    "turn_pruned": turn_pruned,
                    "final_fidelity": round(fid, 6),
                    "category": e.get("category", "contextual_decision"),
                    "content_snippet": content[:80],
                    "gist_crystallized": crystallized_key is not None,
                    "gist_key": crystallized_key,
                })
                pruned_records.append(record)
            else:
                survivors.append(e)

        self.engrams = survivors
        self.total_pruned_count += len(pruned_records)
        self.pruning_history.extend([dict(r) for r in pruned_records])
        self.pruning_history = self.pruning_history[-100:]

        # Permanently anchor crystallized semantic gists
        for g_key, g_content, g_sal in gists_to_crystallize:
            self.record_semantic_gist(key=g_key, content=g_content, salience=g_sal)

        return pruned_records

    def recall_semantic_gists(self, top_k: int = 4, min_fidelity: float = 0.70) -> list[dict]:
        """Recalls active semantic gist anchors ranked by resonance and recency."""
        gists = [
            e
            for e in self.engrams
            if e.get("category") == "semantic_gist"
            and float(e.get("fidelity", 0.0)) >= min_fidelity
        ]
        gists.sort(
            key=lambda x: (
                round(
                    float(x.get("salience", 1.85))
                    * (0.4 + 0.6 * float(x.get("_live_resonance", 0.25))),
                    4,
                ),
                -float(x.get("age", 0.0)),
            ),
            reverse=True,
        )
        return gists[:top_k]

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
        raw_json = json.dumps(state_dict, ensure_ascii=False, indent=2)
        with open(temp_path, "w", encoding="utf-8") as sf:
            sf.write(raw_json)
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
                    existing_root_engrams = []
                    if mirror_path.exists():
                        try:
                            with open(mirror_path, encoding="utf-8", errors="replace") as rf:
                                existing_root_engrams = json.load(rf).get("engrams", [])
                        except Exception:
                            existing_root_engrams = []

                    # Retain canonical production engrams from existing root file if available
                    if len(existing_root_engrams) >= 53:
                        prod_engrams = [
                            e for e in existing_root_engrams
                            if isinstance(e, dict)
                            and e.get("key") in CANONICAL_PROD_KEYS
                        ]
                    else:
                        prod_engrams = [
                            e for e in state_dict.get("engrams", [])
                            if isinstance(e, dict)
                            and e.get("key") in CANONICAL_PROD_KEYS
                            and e.get("key") != "insight_google_drive_mcp_architecture"
                            and not str(e.get("key", "")).startswith(("gist_", "rfc_hybrid_"))
                        ]
                    dedup_engrams = []
                    _seen_k: set[str] = set()
                    for pe in prod_engrams:
                        _k = str(pe.get("key", ""))
                        if _k in _seen_k:
                            continue
                        _seen_k.add(_k)
                        if float(pe.get("salience", 0.0)) < 2.0:
                            pe["salience"] = 2.5
                        pe["is_core_anchor"] = True
                        dedup_engrams.append(pe)
                    mirror_dict["engrams"] = dedup_engrams
                    mirror_dict["recent_arbitrations"] = []
                    mirror_dict["pruning_history"] = []
                    mirror_dict["outcome_history"] = []
                    mirror_dict["last_user_query"] = ""

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


def extract_context_tags(
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
    error_msg: str | None = None,
) -> dict[str, Any]:
    """Extracts contextual tags from tool execution parameters for cognitive LTP/LTD.

    Args:
        tool_name: The name of the tool executed (e.g. 'run_command').
        tool_args: Dictionary of arguments passed to the tool.
        error_msg: Error string if the tool failed, or None.

    Returns:
        Dictionary containing tags: tool, os, runtime, command, target, error_type.
    """
    args = tool_args if isinstance(tool_args, dict) else {}
    tags: dict[str, Any] = {
        "tool": tool_name,
        "os": sys.platform,
    }

    cmd = args.get("CommandLine") or args.get("command") or args.get("cmd")
    if cmd:
        cmd_str = str(cmd)
        tags["command"] = cmd_str
        cmd_lower = cmd_str.lower()
        if "python" in cmd_lower:
            tags["runtime"] = "python3"
        elif "cargo" in cmd_lower:
            tags["runtime"] = "cargo"
        elif "npm" in cmd_lower or "node" in cmd_lower:
            tags["runtime"] = "node"
        elif "zsh" in cmd_lower or "bash" in cmd_lower or "sh" in cmd_lower:
            tags["runtime"] = "shell"
        else:
            tags["runtime"] = "shell"
        tokens = cmd_str.split()
        if len(tokens) > 1 and "target" not in tags:
            tags["target"] = tokens[1]
    else:
        if tool_name in ("run_command", "bash", "execute_command"):
            tags["runtime"] = "shell"
        elif "python" in tool_name:
            tags["runtime"] = "python3"
        else:
            tags["runtime"] = "native"

    for target_key in (
        "TargetFile", "AbsolutePath", "SearchPath", "DirectoryPath",
        "Path", "file_path", "target", "path", "query", "Query"
    ):
        if target_key in args and args[target_key]:
            tags["target"] = str(args[target_key])
            break

    if error_msg:
        err_lower = str(error_msg).lower()
        if "syntax" in err_lower or "matches found" in err_lower:
            tags["error_type"] = "syntax_error"
        elif "permission" in err_lower or "denied" in err_lower:
            tags["error_type"] = "permission_error"
        elif "timeout" in err_lower or "timed out" in err_lower:
            tags["error_type"] = "timeout_error"
        elif "not found" in err_lower or "no such file" in err_lower:
            tags["error_type"] = "not_found"
        elif "exit" in err_lower or "status" in err_lower:
            tags["error_type"] = "nonzero_exit"
        else:
            tags["error_type"] = "tool_error"

    return tags


def handle_post_tool_use(
    payload: dict[str, Any],
    cached: dict[str, Any] | Path | str | None = None,
    state_file: Path | str | None = None,
    workspace_state_file: Path | str | None = None,
    safe_conv_id: str | None = None,
) -> dict[str, Any]:
    """Dedicated ultra-fast (< 25ms) handler for CognitiveAgent PostToolUse events.

    Decouples PostToolUse from PreInvocation, updates plastic negative engrams
    (LTP on error, LTD on success), atomically persists state, and returns {}.
    """
    try:
        if isinstance(cached, (str, Path)):
            state_file = Path(cached)
            cached = None

        if safe_conv_id is None:
            raw_conv_id = str(payload.get("conversationId", "default") or "default")
            safe_conv_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', raw_conv_id)

        if state_file is None:
            artifact_dir = payload.get("artifactDirectoryPath", "")
            if artifact_dir and os.path.exists(artifact_dir):
                state_file = Path(artifact_dir) / "quanta_cognitive_state.json"
            else:
                state_file = Path(f"/tmp/quanta_cognitive_{safe_conv_id}.json")
        else:
            state_file = Path(state_file)

        if workspace_state_file is None:
            workspace_paths = payload.get("workspacePaths", [])
            if workspace_paths and isinstance(workspace_paths, list):
                for wp in workspace_paths:
                    if wp:
                        p = Path(wp) / "quanta_cognitive_state.json"
                        if p.exists():
                            workspace_state_file = p
                            break
            if workspace_state_file is None:
                workspace_state_file = QUANTA_ROOT / "quanta_cognitive_state.json"
        else:
            workspace_state_file = Path(workspace_state_file)

        if not isinstance(cached, dict):
            cached_dict: dict[str, Any] = {}
            if state_file.exists():
                try:
                    with open(state_file, encoding="utf-8", errors="replace") as sf:
                        loaded = json.load(sf)
                        if isinstance(loaded, dict):
                            cached_dict = loaded
                except Exception:
                    cached_dict = {}
        else:
            cached_dict = dict(cached)

        mem = FastBiomorphicMemory(capacity=128)
        raw_engrams = cached_dict.get("engrams", [])
        if isinstance(raw_engrams, list):
            for item in raw_engrams:
                if isinstance(item, dict) and "key" in item:
                    mem.engrams.append(copy.deepcopy(item))

        # Calibrated biological micro-step dephasing for tool execution (dt = 0.2)
        mem.step(dt=0.2)

        tc = payload.get("toolCall")
        if not isinstance(tc, dict):
            tc = {}
        tool_name = str(tc.get("name") or payload.get("toolName") or "unknown_tool")
        tool_args = tc.get("args") or payload.get("toolArgs") or {}
        if not isinstance(tool_args, dict):
            tool_args = {}

        error_msg = payload.get("error")
        tool_output = payload.get("result", payload.get("output"))
        step_val = payload.get("stepIdx", cached_dict.get("last_verified_step", 0))

        is_failure = bool(error_msg)
        if not is_failure and isinstance(tool_output, dict):
            exit_code = tool_output.get("exit_code", tool_output.get("exitCode", 0))
            if exit_code not in (0, None):
                is_failure = True
                if not error_msg:
                    error_msg = f"Non-zero exit code: {exit_code}"
            elif tool_output.get("error"):
                is_failure = True
                if not error_msg:
                    error_msg = str(tool_output.get("error"))

        ctx = extract_context_tags(
            tool_name, tool_args, str(error_msg) if is_failure and error_msg else None
        )

        has_tool = bool(tc or payload.get("toolName") or tool_name != "unknown_tool")
        has_error = bool(error_msg or is_failure)

        if has_tool or has_error:
            matching = [
                e for e in mem.engrams
                if isinstance(e, dict)
                and e.get("category") in ("inhibitor", "anti_pattern")
                and (
                    (e.get("context_tags") or {}).get("tool") == tool_name
                    or e.get("key") == f"inh_{tool_name}"
                    or str(e.get("key", "")).startswith(f"inh_{tool_name}")
                )
            ]
            if is_failure:
                if matching:
                    for inh in matching:
                        mem.potentiate_inhibitor(inh["key"], context_tags=ctx)
                else:
                    inh_key = f"inh_{tool_name}"
                    desc = f"Tool failure in {tool_name}"
                    if error_msg:
                        desc += f": {str(error_msg)[:120]}"
                    mem.record_inhibitor(
                        key=inh_key,
                        content=desc,
                        v_inh=0.60,
                        salience=1.80,
                        context_tags=ctx,
                    )
            else:
                for inh in matching:
                    mem.depress_inhibitor(inh["key"], context_tags=ctx)

        state_dict = dict(cached_dict)
        state_dict["engrams"] = mem.engrams
        state_dict["last_verified_step"] = step_val
        state_dict["pending_tool_continuation"] = True

        _save_mirrored_state_atomically(
            primary_path=state_file,
            mirror_path=workspace_state_file,
            state_dict=state_dict,
            conv_id=safe_conv_id,
            is_test_env=is_hermetic_test_env(payload),
        )
    except Exception:
        pass

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

        # Early PostToolUse Fast Path Decoupling (< 25ms, strictly output {})
        is_post_tool = bool(
            payload.get("event") == "PostToolUse"
            or payload.get("toolCall")
            or payload.get("error")
        )
        if is_post_tool and payload.get("terminationReason") is None:
            with contextlib.suppress(Exception):
                handle_post_tool_use(
                    payload=payload,
                    cached=None,
                    state_file=state_file,
                    workspace_state_file=workspace_state_file,
                    safe_conv_id=safe_conv_id,
                )
            sys.stdout.write(json.dumps({}))
            sys.stdout.flush()
            return

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
                        if not isinstance(item, dict) or "key" not in item:
                            continue
                        e_dict = dict(item)
                        sal_val = float(e_dict.get("salience", 1.0))
                        e_dict["salience"] = sal_val
                        is_core_raw = e_dict.get("is_core_anchor")
                        e_dict["is_core_anchor"] = (
                            (sal_val >= 2.0) if is_core_raw is None else bool(is_core_raw)
                        )
                        e_dict.setdefault("category", "general")
                        e_dict.setdefault("fidelity", 0.9998)
                        e_dict.setdefault("age", 0)
                        e_dict.setdefault("tags", [])
                        e_dict.setdefault("topic", "")
                        e_dict.setdefault("confidence", 0.95)
                        e_dict.setdefault("consecutive_successes", 0)
                        e_dict.setdefault("consecutive_failures", 0)
                        e_dict.setdefault("total_evaluations", 0)
                        e_dict.setdefault("last_outcome", "NEUTRAL")
                        e_dict.setdefault("drift_status", "STABLE")
                        mem.engrams.append(e_dict)
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
                                    raw_age_ws = float(item.get("age", 0))
                                    age_ws = (
                                        int(raw_age_ws)
                                        if raw_age_ws.is_integer()
                                        else round(raw_age_ws, 4)
                                    )
                                    hydrated = {
                                        "key": k,
                                        "content": item.get("content", item.get("description", "")),
                                        "salience": sal_val,
                                        "category": item.get("category", "general"),
                                        "fidelity": float(item.get("fidelity", 0.9998)),
                                        "age": age_ws,
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
                                    if "created_at" in item:
                                        hydrated["created_at"] = float(item["created_at"])
                                    if "v_inh" in item:
                                        hydrated["v_inh"] = float(item["v_inh"])
                                    if "context_tags" in item:
                                        hydrated["context_tags"] = item["context_tags"]
                                    if "context_divergence" in item:
                                        hydrated["context_divergence"] = item["context_divergence"]
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
                    hyps: list[dict[str, Any] | str] = [
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
        # Retrospective transcript verification across previous turn's executed steps
        try:
            if transcript_path:
                from quanta.cognitive.feedback import (
                    CognitiveFeedbackLoop,
                    OutcomeVerifier,
                )
                verifier = OutcomeVerifier(dim=64)
                active_rules_list: list[str | dict[str, Any]] = list(mem.engrams)
                unverified_outcomes = verifier.extract_outcomes_from_transcript(
                    transcript_path=transcript_path,
                    last_step_idx=last_verified_step,
                    active_rules=active_rules_list,
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
                "last_user_query": str(cached.get("last_user_query", "")),
                "pending_tool_continuation": False,
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
        any("quantum_sim" in h or "transpiler_engine" in h or "simulator_engine" in h for h in search_targets)

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

        # Decouple intermediate tool continuation sub-turns from full conversational turns:
        # If continuing after a tool execution, decay was already stepped by PostToolUse (dt=0.2).
        # Advance turn_count and dt=1.0 only for genuine user conversational turns.
        last_saved_query = str(cached.get("last_user_query", ""))
        is_subturn_continuation = bool(
            cached.get("pending_tool_continuation", False)
            and turn_count > 0
            and (not user_query or not last_saved_query or user_query == last_saved_query)
        )

        if not is_subturn_continuation:
            turn_count += 1
            mem.current_turn = turn_count
            mem.step(dt=1.0)
        else:
            mem.current_turn = turn_count

        # Microglial active synaptic pruning
        pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.50, turn=turn_count)
        if pruned:
            for item_p in pruned:
                p_dict = dict(item_p) if isinstance(item_p, dict) else {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "rule_name": str(item_p),
                    "id": str(item_p),
                    "key": str(item_p),
                    "reason": "microglial_clearance",
                    "decayed_salience": 0.25,
                    "turn_pruned": turn_count,
                    "final_fidelity": 0.65,
                }
                pruning_history.append(p_dict)
            pruning_history = pruning_history[-100:]

        # Context-Adaptive SWR Resonance Scoring (R1 Live Hook Integration)
        # Combines C^64 deterministic statevector overlap + lexical Jaccard + workspace affinity
        active_query_text = (user_query or last_saved_query or "").strip().lower()
        ws_lower = (real_workspace or "").strip().lower()
        active_context_str = f"{ws_lower} {active_query_text}".strip()
        context_tokens = {
            w for w in re.split(r"[^\w]+", active_context_str) if len(w) > 2
        }

        def _fast_statevector_64(text: str) -> list[complex]:
            """Pure-stdlib C^64 unit statevector projection (<15us)."""
            import hashlib

            words = [w for w in re.split(r"[^\w]+", text.lower()) if len(w) > 1][:64]
            if not words:
                words = [text.lower() or "empty"]
            re_parts = [0.0] * 64
            im_parts = [0.0] * 64
            for w in words:
                digest = hashlib.sha256(w.encode("utf-8", errors="ignore")).digest()
                for idx in range(32):
                    b_val = digest[idx]
                    angle = (b_val / 255.0) * 2.0 * math.pi
                    slot = (idx * 2 + (b_val & 1)) % 64
                    re_parts[slot] += math.cos(angle)
                    im_parts[slot] += math.sin(angle)
            norm_sq = sum(r * r + i * i for r, i in zip(re_parts, im_parts, strict=False))
            norm = math.sqrt(norm_sq) if norm_sq > 1e-12 else 1.0
            return [complex(r / norm, i / norm) for r, i in zip(re_parts, im_parts, strict=False)]

        psi_ctx = _fast_statevector_64(active_context_str) if active_context_str else None

        def _compute_live_resonance(engram: dict[str, Any]) -> float:
            key_s = str(engram.get("key", "")).lower()
            topic_s = str(engram.get("topic", "")).lower()
            content_s = str(engram.get("content", "")).lower()
            tags_list = [str(t).lower() for t in engram.get("tags", []) if t]
            engram_full = f"{key_s} {topic_s} {' '.join(tags_list)} {content_s}"
            e_tokens = {w for w in re.split(r"[^\w]+", engram_full) if len(w) > 2}

            # 1. Lexical Jaccard & token hit ratio
            if context_tokens and e_tokens:
                inter = len(context_tokens & e_tokens)
                union = len(context_tokens | e_tokens)
                jaccard = inter / union if union > 0 else 0.0
                hit_ratio = min(1.0, inter / max(1, min(len(context_tokens), 6)))
                lex_score = 0.5 * jaccard + 0.5 * hit_ratio
            else:
                lex_score = 0.25

            # 2. C^64 Hilbert inner-product fidelity |<psi_ctx | psi_e>|^2
            if psi_ctx is not None:
                psi_e = _fast_statevector_64(f"{key_s}:{topic_s}:{content_s[:240]}")
                inner_re = sum(
                    c.real * e.real + c.imag * e.imag for c, e in zip(psi_ctx, psi_e, strict=False)
                )
                inner_im = sum(
                    c.real * e.imag - c.imag * e.real for c, e in zip(psi_ctx, psi_e, strict=False)
                )
                f_hilbert = min(1.0, (inner_re * inner_re + inner_im * inner_im) * 8.0)
            else:
                f_hilbert = 0.25

            # 3. Workspace / Domain affinity boost (check key_s & topic_s, penalize off-workspace project dreams)
            ws_tokens = {
                w for w in re.split(r"[^\w]+", ws_lower)
                if len(w) > 2 and w not in ("alfa", "quantum_sim", "works", "workspace", "general")
            }
            if "quantum_sim" in ws_lower and not ws_tokens:
                ws_tokens.add("quantum_sim")

            foreign_projects = (
                "surface_code", "stabilizer", "sabre", "kraus",
                "clifford", "openqasm", "variational", "hamiltonian",
            )
            matched_foreign = [
                fp for fp in foreign_projects
                if fp in key_s and fp not in active_context_str
            ]

            ws_boost = 0.0
            if matched_foreign:
                # Engram explicitly targets another project not in current workspace or query
                ws_boost -= 0.45
            else:
                if ws_tokens and any(wt in key_s or wt in topic_s for wt in ws_tokens):
                    ws_boost += 0.45
                if "quanta" in ws_lower and any(
                    kw in key_s
                    for kw in ("quantum", "quanta", "zeno", "swr", "qubit", "landauer", "rem_sleep", "resonance", "lindblad", "anti_rumination", "darwin_qos")
                ):
                    ws_boost += 0.45

            return max(0.01, min(1.0, 0.30 * f_hilbert + 0.70 * lex_score + ws_boost))

        for e in mem.engrams:
            e["_live_resonance"] = _compute_live_resonance(e)
            if e.get("category") in ("subconscious_dream", "architecture_rfc"):
                # Reset inflated cross-project salience to canonical baseline (2.85)
                # and boost dynamically only when resonant with current workspace/query
                if e["_live_resonance"] < 0.45:
                    e["salience"] = min(float(e.get("salience", 2.85)), 2.85)
                else:
                    e["salience"] = max(float(e.get("salience", 2.85)), 3.25)

        # Recall vital engrams via Context-Adaptive SWR replay
        # Separate vital anchors into Core Anchors, Semantic Gists, and Contextual Decisions
        core_anchors = [
            e for e in mem.engrams
            if (e.get("is_core_anchor", False) or float(e.get("salience", 0.0)) >= 2.0)
            and float(e.get("fidelity", 0.0)) >= 0.80
            and e.get("category") not in ("inhibitor", "anti_pattern", "semantic_gist")
        ]
        # 100% unconditional preservation of invariant constraint rules
        constraint_rules = [e for e in core_anchors if e.get("category") == "constraint"]
        constraint_rules.sort(key=lambda x: float(x.get("salience", 0.0)), reverse=True)

        # Context-adaptive ranking for non-constraint core insights (e.g., subconscious dreams)
        other_core = [e for e in core_anchors if e.get("category") != "constraint"]
        other_core.sort(
            key=lambda x: float(x.get("salience", 0.0))
            * float(x.get("fidelity", 1.0))
            * (0.25 + 0.75 * float(x.get("_live_resonance", 0.25))),
            reverse=True,
        )
        selected_core = constraint_rules + other_core[:max(3, 10 - len(constraint_rules))]

        # Active crystallized semantic gists (category == "semantic_gist", fidelity >= 0.70)
        active_gists = mem.recall_semantic_gists(top_k=4, min_fidelity=0.70)

        transient_decisions = [
            e for e in mem.engrams
            if not (e.get("is_core_anchor", False) or float(e.get("salience", 0.0)) >= 2.0)
            and float(e.get("fidelity", 0.0)) >= 0.70
            and e.get("category") not in ("inhibitor", "anti_pattern", "semantic_gist")
        ]
        # Score transient decisions by combining temporal recency and contextual resonance R(C, e)
        now_epoch = time.time()
        filtered_transient = []
        for t in transient_decisions:
            raw_created = t.get("created_at")
            if raw_created is not None:
                if isinstance(raw_created, (int, float)):
                    created_ts = float(raw_created)
                elif isinstance(raw_created, datetime):
                    created_ts = raw_created.timestamp()
                else:
                    try:
                        created_ts = float(raw_created)
                    except (ValueError, TypeError):
                        created_ts = now_epoch
            else:
                created_ts = max(0.0, now_epoch - float(t.get("age", 0.0)) * 60.0)

            dt_h = max(0.0, (now_epoch - created_ts) / 3600.0)
            w_recency = math.exp(-0.02 * dt_h) if dt_h < 24.0 else math.exp(-0.05 * dt_h)
            sal = float(t.get("salience", 0.0))
            fid = float(t.get("fidelity", 0.0))
            res = float(t.get("_live_resonance", 0.25))
            t["_recency_score"] = sal * fid * w_recency * (0.5 + 0.5 * res)
            t["_created_ts"] = created_ts

            # Filter out stale off-topic transient decisions when active context is present
            if context_tokens and float(t.get("age", 0.0)) >= 3.0 and res < 0.08:
                continue
            filtered_transient.append(t)

        if not filtered_transient and transient_decisions:
            filtered_transient = transient_decisions[:2]

        filtered_transient.sort(
            key=lambda x: (x["_recency_score"], x["_created_ts"]),
            reverse=True,
        )
        selected_transient = filtered_transient[:3]

        active_inhibitors = mem.recall_inhibitors(top_k=3, min_v_inh=0.20)

        # Read active multi-agent shared blackboard constraints if present (<0.2ms)
        blackboard_items: list[str] = []
        with contextlib.suppress(Exception):
            bb_file = Path(os.path.expanduser("~/.quanta/agent/blackboard/shared_blackboard.json"))
            if bb_file.exists() and bb_file.stat().st_size > 0:
                with open(bb_file, encoding="utf-8") as bf:
                    bb_data = json.load(bf)
                if isinstance(bb_data, dict):
                    for c_item in bb_data.get("constraints", []):
                        if isinstance(c_item, dict):
                            exp = c_item.get("expires_at")
                            if exp is None or float(exp) >= now_epoch:
                                c_key = str(c_item.get("key", "")).strip()
                                if c_key:
                                    blackboard_items.append(c_key)

        vital_anchors = selected_core + active_gists + selected_transient + active_inhibitors

        if vital_anchors:
            core_items = []
            for v in selected_core:
                fid_str = format_fidelity(v["fidelity"])
                core_items.append(f"{v['key']} ({fid_str})")

            gist_items = []
            for g in active_gists:
                fid_str = format_fidelity(g["fidelity"])
                gist_items.append(f"{g['key']} ({fid_str})")

            transient_items = []
            for t in selected_transient:
                fid_str = format_fidelity(t["fidelity"])
                transient_items.append(f"{t['key']} ({fid_str})")

            inhibitor_items = []
            for inh in active_inhibitors:
                v_val = float(inh.get("v_inh", 0.50))
                fid_str = format_fidelity(float(inh.get("fidelity", 0.9998)))
                inhibitor_items.append(f"{inh['key']} (V_inh={v_val:.2f}, {fid_str})")

            lines = ["[Quanta Bilişsel Çıpa | SWR Replay]:"]
            if core_items:
                lines.append(f"  🔒 Çekirdek: {', '.join(core_items)}")
            if gist_items:
                lines.append(f"  🧠 Özüt: {', '.join(gist_items)}")
            if transient_items:
                lines.append(f"  ⚡ Geçici: {', '.join(transient_items)}")
            if blackboard_items:
                lines.append(f"  📋 Ortak Tahta: {', '.join(blackboard_items[:4])}")
            if inhibitor_items:
                lines.append(f"  🚫 İnhibitör / Anti-Pattern: {', '.join(inhibitor_items)}")

            # Crystallization notification
            crystallized_in_turn = [
                p for p in pruned
                if isinstance(p, dict) and p.get("gist_crystallized")
            ]
            if crystallized_in_turn:
                lines.append(f"  ✨ Kristalleşen Özüt: {len(crystallized_in_turn)} karar")
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
                pruned_keys=[str(item_p) for item_p in pruned],
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
            "last_user_query": user_query or last_saved_query,
            "pending_tool_continuation": False,
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
