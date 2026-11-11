"""
quanta.layer3.entity_resolution -- Quantum entity resolution (deduplication).

Uses monotonic similarity angle encoding ``theta_q = sim_q * (pi / 2) in [0, pi / 2]``
and variationally optimizes a correlation-clustering QAOA circuit + threshold layer
via ``scipy.optimize.minimize(..., method="COBYLA")`` without post-hoc threshold overrides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from quanta.simulator.factory import create_simulator

__all__ = ["resolve", "ResolutionResult", "compute_similarity"]


@dataclass
class Cluster:
    """A cluster of merged records."""

    record_ids: list[int]
    canonical: dict
    confidence: float


@dataclass
class ResolutionResult:
    """Result of entity resolution."""

    clusters: list[Cluster]
    num_records: int
    num_entities: int
    method: str
    quantum_blocks: int = 0
    total_qubits: int = 0
    comparison_accuracy: float | None = None

    def summary(self) -> str:
        lines = [
            "╔═══════════════════════════════════════════════════╗",
            "║  Quantum Entity Resolution                        ║",
            "╠═══════════════════════════════════════════════════╣",
            f"║  Records: {self.num_records:<5} → Entities: {self.num_entities:<5}"
            f"  ({self.num_records - self.num_entities} merged)  ║",
            f"║  Method: {self.method:<10}  Blocks: {self.quantum_blocks:<5}"
            f"  Qubits: {self.total_qubits:<4}  ║",
            "╠───────────────────────────────────────────────────╣",
        ]
        for i, c in enumerate(self.clusters):
            ids = ",".join(str(x) for x in c.record_ids)
            name = c.canonical.get("name", "?")[:20]
            conf = f"{c.confidence:.0%}"
            lines.append(
                f"║  Cluster {i+1}: [{ids:<12}] {name:<20} {conf:>4} ║"
            )
        if self.comparison_accuracy is not None:
            lines.append("╠───────────────────────────────────────────────────╣")
            lines.append(
                f"║  Accuracy vs ground truth: {self.comparison_accuracy:.0%}"
                f"{'':>23}║"
            )
        lines.append("╚═══════════════════════════════════════════════════╝")
        return "\n".join(lines)


def _normalize(s: str) -> str:
    """Normalize string for Turkish-aware comparison."""
    s = s.lower().strip()
    tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    return s.translate(tr_map)


def _levenshtein_sim(a: str, b: str) -> float:
    """Levenshtein similarity (1 - normalized distance)."""
    a, b = _normalize(a), _normalize(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    m, n = len(a), len(b)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return 1.0 - dp[n] / max(m, n)


def _field_similarity(a: dict, b: dict, field_name: str) -> float:
    """Similarity for a single record field."""
    va = str(a.get(field_name, "")).strip()
    vb = str(b.get(field_name, "")).strip()
    if not va or not vb:
        return 0.0

    if _normalize(va) == _normalize(vb):
        return 1.0

    if field_name in ("phone", "tc", "phone_number"):
        da = "".join(c for c in va if c.isdigit())
        db = "".join(c for c in vb if c.isdigit())
        if da and db:
            return 1.0 if da == db else 0.0

    if field_name == "email":
        return 1.0 if _normalize(va) == _normalize(vb) else 0.0

    return _levenshtein_sim(va, vb)


def compute_similarity(
    a: dict,
    b: dict,
    fields: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted similarity score in [0, 1] between two records."""
    if fields is None:
        fields = list(set(a.keys()) | set(b.keys()))

    if weights is None:
        weights = {
            "phone": 3.0,
            "tc": 3.0,
            "email": 2.5,
            "name": 2.0,
            "first_name": 1.5,
            "last_name": 2.0,
            "birth_date": 2.0,
            "city": 0.5,
            "address": 1.0,
        }

    total_weight = 0.0
    total_score = 0.0
    for f in fields:
        w = weights.get(f, 1.0)
        sim = _field_similarity(a, b, f)
        total_score += w * sim
        total_weight += w

    return total_score / total_weight if total_weight > 0 else 0.0


def _soundex_tr(s: str) -> str:
    """Turkish-aware Soundex encoding."""
    s = _normalize(s).upper()
    if not s:
        return ""

    code = s[0]
    mapping = {
        "B": "1",
        "F": "1",
        "P": "1",
        "V": "1",
        "C": "2",
        "G": "2",
        "J": "2",
        "K": "2",
        "Q": "2",
        "S": "2",
        "X": "2",
        "Z": "2",
        "D": "3",
        "T": "3",
        "L": "4",
        "M": "5",
        "N": "5",
        "R": "6",
    }

    prev = mapping.get(s[0], "0")
    for ch in s[1:]:
        digit = mapping.get(ch, "0")
        if digit != "0" and digit != prev:
            code += digit
        prev = digit
        if len(code) >= 4:
            break

    return (code + "000")[:4]


def _split_block_by_similarity(
    records: list[dict], block: list[int], max_size: int = 6
) -> list[list[int]]:
    """Splits a large block into sub-blocks keeping similar records together (max 6 -> <= 15 qubits)."""
    if len(block) <= max_size:
        return [block]

    n = len(block)
    sim_matrix: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            s = compute_similarity(records[block[i]], records[block[j]])
            sim_matrix[(i, j)] = s
            sim_matrix[(j, i)] = s

    sub_blocks: list[list[int]] = [[block[0]]]
    assigned = {0}

    for _ in range(1, n):
        best_record = -1
        best_sub = 0
        best_score = -1.0

        for ri in range(n):
            if ri in assigned:
                continue
            for si, sub in enumerate(sub_blocks):
                if len(sub) >= max_size:
                    continue
                avg_sim = sum(
                    sim_matrix.get((ri, block.index(m)), 0.0) for m in sub
                ) / len(sub)
                if avg_sim > best_score:
                    best_score = avg_sim
                    best_record = ri
                    best_sub = si

        if best_record < 0:
            break

        assigned.add(best_record)
        if best_score > 0.2 and len(sub_blocks[best_sub]) < max_size:
            sub_blocks[best_sub].append(block[best_record])
        else:
            sub_blocks.append([block[best_record]])

    return [sb for sb in sub_blocks if len(sb) > 0]


def _merge_overlapping_sets(sets: list[set[int]]) -> list[list[int]]:
    """Merges sets that share any element via Union-Find."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for s in sets:
        items = list(s)
        for i in range(1, len(items)):
            union(items[0], items[i])

    groups: dict[int, list[int]] = {}
    all_items: set[int] = set()
    for s in sets:
        all_items.update(s)

    for item in all_items:
        root = find(item)
        groups.setdefault(root, []).append(item)

    return [sorted(g) for g in groups.values() if len(g) > 1]


def _block_records(
    records: list[dict], block_keys: list[str] | None = None
) -> list[list[int]]:
    """3-layer blocking pipeline (exact keys + Turkish Soundex + sorted neighborhood)."""
    _ = block_keys
    blocks: dict[str, set[int]] = {}

    for i, rec in enumerate(records):
        phone = "".join(c for c in str(rec.get("phone", "")) if c.isdigit())
        if phone:
            blocks.setdefault(f"phone:{phone}", set()).add(i)

        email = _normalize(str(rec.get("email", "")).strip())
        if email:
            blocks.setdefault(f"email:{email}", set()).add(i)

        tc = str(rec.get("tc", "")).strip()
        if tc and len(tc) > 5:
            blocks.setdefault(f"tc:{tc}", set()).add(i)

        name = _normalize(str(rec.get("name", "")).strip())
        parts = name.split()
        if len(parts) >= 2:
            blocks.setdefault(f"last:{parts[-1]}", set()).add(i)
        elif len(parts) == 1 and "." in str(rec.get("name", "")):
            raw_parts = str(rec.get("name", "")).strip().split()
            if len(raw_parts) >= 2:
                blocks.setdefault(f"last:{_normalize(raw_parts[-1])}", set()).add(i)

        bdate = str(rec.get("birth_date", "")).strip()
        if bdate and parts:
            initial = parts[0][0] if parts[0] else ""
            blocks.setdefault(f"bdate:{initial}:{bdate}", set()).add(i)

    for i, rec in enumerate(records):
        name = str(rec.get("name", "")).strip()
        if not name:
            continue
        parts = name.split()
        full_sx = _soundex_tr(name.replace(".", "").replace(" ", ""))
        if full_sx:
            blocks.setdefault(f"sx_full:{full_sx}", set()).add(i)

        if len(parts) >= 2:
            last_sx = _soundex_tr(parts[-1])
            first_initial = _normalize(parts[0])[0] if parts[0] else ""
            if last_sx and first_initial:
                blocks.setdefault(f"sx_last:{first_initial}:{last_sx}", set()).add(i)

        if parts:
            first_word = parts[0].replace(".", "")
            if len(first_word) >= 2:
                blocks.setdefault(f"sx_first:{_soundex_tr(first_word)}", set()).add(i)

    indexed = []
    for i, rec in enumerate(records):
        name = str(rec.get("name", "")).strip()
        city = str(rec.get("city", "")).strip()
        sort_key = _soundex_tr(name) + _normalize(city)[:3]
        indexed.append((sort_key, i))

    indexed.sort(key=lambda x: x[0])
    window_size = 4
    for w in range(len(indexed)):
        for w2 in range(w + 1, min(w + window_size, len(indexed))):
            sk1, i1 = indexed[w]
            sk2, i2 = indexed[w2]
            if sk1[:3] == sk2[:3]:
                pair_key = f"window:{min(i1, i2)}:{max(i1, i2)}"
                blocks.setdefault(pair_key, set()).update({i1, i2})

    merged = _merge_overlapping_sets(list(blocks.values()))

    final_blocks: list[list[int]] = []
    for block in merged:
        if len(block) <= 6:
            final_blocks.append(block)
        else:
            final_blocks.extend(_split_block_by_similarity(records, block, max_size=6))

    all_assigned: set[int] = set()
    for block in final_blocks:
        all_assigned.update(block)
    for i in range(len(records)):
        if i not in all_assigned:
            final_blocks.append([i])

    return final_blocks


def _qaoa_optimize_block(
    records: list[dict],
    indices: list[int],
    threshold: float,
    fields: list[str] | None = None,
    seed: int | None = None,
) -> tuple[list[list[int]], int]:
    """Variational Correlation-Clustering QAOA within a candidate block.

    - Monotonic feature angle encoding: ``theta_q = clip(sim_q, 0, 1) * (pi / 2) in [0, pi/2]``
      so ``RY(2 * theta_q)|0>`` maps similarity monotonically onto merge amplitude ``sin(theta_q)``.
    - Variational cost + transitivity + threshold rotation layer optimized via COBYLA
      against the exact correlation-clustering energy (with transitivity triangle penalty).
    """
    n = len(indices)
    if n <= 1:
        return [[idx] for idx in indices], 0

    pairs: list[tuple[int, int]] = []
    pair_to_q: dict[tuple[int, int], int] = {}
    sims: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_similarity(records[indices[i]], records[indices[j]], fields)
            pair_to_q[(i, j)] = len(pairs)
            pairs.append((i, j))
            sims.append(float(np.clip(sim, 0.0, 1.0)))

    num_qubits = len(pairs)
    if num_qubits > 15:
        return _greedy_merge_block(records, indices, threshold, fields), 0

    sims_arr = np.asarray(sims, dtype=float)
    # Monotonic angle mapping in [0, pi / 2]
    mono_angles = sims_arr * (math.pi / 2.0)
    th_angle = float(np.clip(threshold, 0.0, 1.0)) * (math.pi / 2.0)

    triangles: list[tuple[int, int, int]] = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triangles.append(
                    (pair_to_q[(a, b)], pair_to_q[(b, c)], pair_to_q[(a, c)])
                )

    shared_edges: list[tuple[int, int]] = []
    for q1 in range(num_qubits):
        i1, j1 = pairs[q1]
        for q2 in range(q1 + 1, num_qubits):
            i2, j2 = pairs[q2]
            if i1 in (i2, j2) or j1 in (i2, j2):
                shared_edges.append((q1, q2))

    # Precompute correlation-clustering energy on all 2^num_qubits edge configurations:
    #   E(x) = -sum_q (sim_q - threshold) * x_q + 0.35 * (# open wedges where sum of triangle bits == 2)
    dim = 1 << num_qubits
    basis_idx = np.arange(dim, dtype=np.uint32)
    bits_mat = np.column_stack(
        [((basis_idx >> (num_qubits - 1 - q)) & 1) for q in range(num_qubits)]
    ).astype(float)
    edge_weights = sims_arr - threshold
    energies = -(bits_mat @ edge_weights)
    for qa, qb, qc in triangles:
        tri_sum = bits_mat[:, qa] + bits_mat[:, qb] + bits_mat[:, qc]
        energies += 0.35 * (tri_sum == 2.0).astype(float)

    def _run_variational_circuit(params: np.ndarray, run_seed: int | None = None):
        gamma_cost, gamma_trans, beta_th, beta_mix = (
            float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
        )
        sim = create_simulator(num_qubits, seed=run_seed)
        # Layer 0: Monotonic similarity angle loading in [0, pi/2] shifted by variational threshold layer
        for q in range(num_qubits):
            # Initial monotonic angle in [0, pi/2]: RY(2 * mono_angles[q]) sets P(|1>) = sin^2(sim * pi/2)
            sim.apply("RY", (q,), (2.0 * float(mono_angles[q]),))

        # Layer 1: Ising cost & transitivity interactions
        for q in range(num_qubits):
            # Z_q eigenvalue is +1 on |0>, -1 on |1>; x_q = (1 - Z_q)/2 -> h_q = +(sim_q - threshold)/2
            sim.apply("RZ", (q,), (gamma_cost * float(edge_weights[q]),))
        for q1, q2 in shared_edges:
            if abs(gamma_trans) > 1e-12:
                sim.apply("CX", (q1, q2))
                sim.apply("RZ", (q2,), (gamma_trans,))
                sim.apply("CX", (q1, q2))

        # Layer 2: Variational threshold calibration & mixer
        for q in range(num_qubits):
            sim.apply("RY", (q,), (beta_th * float(mono_angles[q] - th_angle),))
            if abs(beta_mix) > 1e-12:
                sim.apply("RX", (q,), (2.0 * beta_mix,))
        return sim

    def objective(params: np.ndarray) -> float:
        sim_eval = _run_variational_circuit(params)
        return float(np.dot(sim_eval.probabilities(), energies))

    x0 = np.array([0.5, 0.1, 1.5, 0.05], dtype=float)
    opt_res = scipy_minimize(
        objective,
        x0,
        method="COBYLA",
        options={"maxiter": 60, "rhobeg": 0.35},
    )

    sim_opt = _run_variational_circuit(opt_res.x, run_seed=seed)
    counts = sim_opt.sample(2048)
    best_bitstring = max(counts, key=lambda bs: counts[bs])

    # Decode clusters purely from the quantum QAOA bitstring (no classical 0.7*threshold override!)
    adjacency = [[False] * n for _ in range(n)]
    for q, (i, j) in enumerate(pairs):
        if best_bitstring[q] == "1":
            adjacency[i][j] = True
            adjacency[j][i] = True

    visited = [False] * n
    clusters: list[list[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        cluster: list[int] = []
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            cluster.append(indices[node])
            for neighbor in range(n):
                if adjacency[node][neighbor] and not visited[neighbor]:
                    stack.append(neighbor)
        clusters.append(sorted(cluster))

    return clusters, num_qubits


def _greedy_merge_block(
    records: list[dict],
    indices: list[int],
    threshold: float,
    fields: list[str] | None = None,
) -> list[list[int]]:
    """Classical greedy transitive closure baseline."""
    n = len(indices)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_similarity(records[indices[i]], records[indices[j]], fields)
            if sim >= threshold:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pi] = pj

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(indices[i])

    return [sorted(g) for g in groups.values()]


def _pick_canonical(records: list[dict], ids: list[int]) -> dict:
    """Picks the most complete record as canonical representative."""
    best = ids[0]
    best_fields = sum(1 for v in records[best].values() if str(v).strip())
    for idx in ids[1:]:
        filled = sum(1 for v in records[idx].values() if str(v).strip())
        if filled > best_fields:
            best = idx
            best_fields = filled
    return dict(records[best])


def resolve(
    records: list[dict],
    threshold: float = 0.65,
    fields: list[str] | None = None,
    method: str = "qaoa",
    seed: int | None = None,
) -> ResolutionResult:
    """Quantum entity resolution via blocking + variational correlation-clustering QAOA."""
    if not records:
        return ResolutionResult(
            clusters=[], num_records=0, num_entities=0, method=method
        )

    n = len(records)
    if fields is None:
        fields = list(records[0].keys())

    blocks = _block_records(records)
    all_clusters: list[list[int]] = []
    total_qubits = 0
    quantum_blocks = 0
    assigned: set[int] = set()

    for b_idx, block in enumerate(blocks):
        if len(block) <= 1:
            continue

        if method == "qaoa":
            b_seed = None if seed is None else int(seed) + b_idx * 31
            clusters, qubits = _qaoa_optimize_block(
                records, block, threshold, fields, b_seed
            )
            total_qubits += qubits
            if qubits > 0:
                quantum_blocks += 1
        else:
            clusters = _greedy_merge_block(records, block, threshold, fields)

        for cluster in clusters:
            all_clusters.append(cluster)
            assigned.update(cluster)

    for i in range(n):
        if i not in assigned:
            all_clusters.append([i])

    result_clusters: list[Cluster] = []
    for ids in all_clusters:
        canonical = _pick_canonical(records, ids)
        if len(ids) > 1:
            pair_sims = [
                compute_similarity(records[ids[a]], records[ids[b]], fields)
                for a in range(len(ids))
                for b in range(a + 1, len(ids))
            ]
            conf = float(np.mean(pair_sims))
        else:
            conf = 1.0

        result_clusters.append(
            Cluster(record_ids=sorted(ids), canonical=canonical, confidence=conf)
        )

    result_clusters.sort(key=lambda c: -len(c.record_ids))

    return ResolutionResult(
        clusters=result_clusters,
        num_records=n,
        num_entities=len(result_clusters),
        method=method,
        quantum_blocks=quantum_blocks,
        total_qubits=total_qubits,
    )
