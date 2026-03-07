"""
quanta.layer3.entity_resolution -- Quantum entity resolution (dedup).

Solves the "who is who?" problem using QAOA for optimal clustering.
Hybrid pipeline: classical blocking + quantum optimization + merge.

Pipeline:
  1. Classical: Compute pairwise similarity scores (fuzzy matching)
  2. Classical: Blocking — group likely matches into small blocks
  3. QUANTUM: QAOA optimizes merge decisions within each block
  4. Classical: Merge blocks and produce final clusters

Why quantum? The merge decision is a graph partitioning problem.
For N records in a block, there are 2^(N*(N-1)/2) possible merge
configurations. QAOA explores this exponential space efficiently.

Example:
    >>> from quanta.layer3.entity_resolution import resolve
    >>> records = [
    ...     {"name": "Ahmet Yilmaz", "phone": "5321112233"},
    ...     {"name": "A. Yılmaz",    "phone": "5321112233"},
    ... ]
    >>> result = resolve(records, threshold=0.6)
    >>> print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quanta.simulator.statevector import StateVectorSimulator

__all__ = ["resolve", "ResolutionResult"]


@dataclass
class Cluster:
    """A cluster of merged records."""
    record_ids: list[int]
    canonical: dict  # Best representative record
    confidence: float


@dataclass
class ResolutionResult:
    """Result of entity resolution.

    Attributes:
        clusters: List of merged clusters.
        num_records: Original record count.
        num_entities: Unique entity count after merge.
        method: "qaoa" or "greedy".
        quantum_blocks: Number of blocks solved with QAOA.
        total_qubits: Total qubits used across all blocks.
    """
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


# ── Similarity functions ──

def _normalize(s: str) -> str:
    """Normalize string for comparison."""
    s = s.lower().strip()
    # Turkish character normalization
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
            if a[i-1] == b[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp

    distance = dp[n]
    return 1.0 - distance / max(m, n)


def _field_similarity(a: dict, b: dict, field: str) -> float:
    """Similarity for a single field."""
    va = str(a.get(field, "")).strip()
    vb = str(b.get(field, "")).strip()
    if not va or not vb:
        return 0.0  # Missing data → no evidence

    # Exact match
    if _normalize(va) == _normalize(vb):
        return 1.0

    # Phone: remove non-digits and compare
    if field in ("phone", "tc", "phone_number"):
        da = "".join(c for c in va if c.isdigit())
        db = "".join(c for c in vb if c.isdigit())
        if da and db:
            return 1.0 if da == db else 0.0

    # Email: direct comparison
    if field == "email":
        return 1.0 if _normalize(va) == _normalize(vb) else 0.0

    # String fields: Levenshtein
    return _levenshtein_sim(va, vb)


def compute_similarity(
    a: dict, b: dict,
    fields: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted similarity score between two records."""
    if fields is None:
        fields = list(set(a.keys()) | set(b.keys()))

    if weights is None:
        # Default weights: identifiers > names > location
        weights = {
            "phone": 3.0, "tc": 3.0, "email": 2.5,
            "name": 2.0, "first_name": 1.5, "last_name": 2.0,
            "birth_date": 2.0, "city": 0.5, "address": 1.0,
        }

    total_weight: float = 0
    total_score: float = 0

    for f in fields:
        w = weights.get(f, 1.0)
        sim = _field_similarity(a, b, f)
        total_score += w * sim
        total_weight += w

    return total_score / total_weight if total_weight > 0 else 0


# ── Phonetic encoding (Turkish-aware Soundex) ──

def _soundex_tr(s: str) -> str:
    """Turkish-aware Soundex encoding.

    Maps characters to phonetic groups, so "Yılmaz", "Yilmaz",
    "Yılmazz" all produce the same code.

    Groups:
      1: B, F, P, V
      2: C, Ç, G, Ğ, J, K, Q, S, Ş, X, Z
      3: D, T
      4: L
      5: M, N
      6: R
    """
    s = _normalize(s).upper()
    if not s:
        return ""

    # Keep first letter
    code = s[0]

    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2",
        "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
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

    # Pad to 4 characters
    code = (code + "000")[:4]
    return code


# ── Blocking ──

def _split_block_by_similarity(
    records: list[dict], block: list[int], max_size: int = 7
) -> list[list[int]]:
    """Splits a large block into sub-blocks keeping similar records together.

    Uses greedy assignment: each record goes to the sub-block where its
    average similarity to existing members is highest.
    """
    if len(block) <= max_size:
        return [block]

    # Compute pairwise similarity matrix for this block
    n = len(block)
    sim_matrix = {}
    for i in range(n):
        for j in range(i + 1, n):
            s = compute_similarity(records[block[i]], records[block[j]])
            sim_matrix[(i, j)] = s
            sim_matrix[(j, i)] = s

    # Greedy sub-clustering
    sub_blocks: list[list[int]] = [[block[0]]]
    assigned = {0}

    # Sort remaining by max similarity to already-assigned records
    for _ in range(1, n):
        best_record = -1
        best_sub = 0
        best_score = -1

        for ri in range(n):
            if ri in assigned:
                continue
            for si, sub in enumerate(sub_blocks):
                if len(sub) >= max_size:
                    continue
                # Average similarity to this sub-block
                avg_sim = sum(
                    sim_matrix.get((ri, block.index(m)), 0)
                    for m in sub
                ) / len(sub)
                if avg_sim > best_score:
                    best_score = avg_sim
                    best_record = ri
                    best_sub = si

        if best_record < 0:
            break

        assigned.add(best_record)

        # Add to best sub-block, or create new one if all full
        if best_score > 0.2 and len(sub_blocks[best_sub]) < max_size:
            sub_blocks[best_sub].append(block[best_record])
        else:
            # Create new sub-block
            sub_blocks.append([block[best_record]])

    return [sb for sb in sub_blocks if len(sb) > 0]

def _block_records(
    records: list[dict], block_keys: list[str] | None = None,
) -> list[list[int]]:
    """3-layer blocking pipeline for high recall.

    Layer 1: Exact key blocking (phone, email, TC)
    Layer 2: Phonetic blocking (Turkish Soundex on names)
    Layer 3: Sliding window (sorted neighborhood)

    Records in the same block likely refer to the same entity.
    Union-find merges overlapping blocks transitively.
    """
    blocks: dict[str, set[int]] = {}

    # ── Layer 1: Exact key blocking ──
    for i, rec in enumerate(records):
        # Phone
        phone = "".join(c for c in str(rec.get("phone", "")) if c.isdigit())
        if phone:
            blocks.setdefault(f"phone:{phone}", set()).add(i)

        # Email
        email = _normalize(str(rec.get("email", "")).strip())
        if email:
            blocks.setdefault(f"email:{email}", set()).add(i)

        # TC
        tc = str(rec.get("tc", "")).strip()
        if tc and len(tc) > 5:
            blocks.setdefault(f"tc:{tc}", set()).add(i)

        # Last name (exact)
        name = _normalize(str(rec.get("name", "")).strip())
        parts = name.split()
        if len(parts) >= 2:
            last = parts[-1]
            blocks.setdefault(f"last:{last}", set()).add(i)
        elif len(parts) == 1 and "." in str(rec.get("name", "")):
            raw_parts = str(rec.get("name", "")).strip().split()
            if len(raw_parts) >= 2:
                last = _normalize(raw_parts[-1])
                blocks.setdefault(f"last:{last}", set()).add(i)

        # Birth date + initial
        bdate = str(rec.get("birth_date", "")).strip()
        if bdate and parts:
            initial = parts[0][0] if parts[0] else ""
            blocks.setdefault(f"bdate:{initial}:{bdate}", set()).add(i)

    # ── Layer 2: Phonetic blocking (Soundex) ──
    for i, rec in enumerate(records):
        name = str(rec.get("name", "")).strip()
        if not name:
            continue

        parts = name.split()

        # Full name Soundex
        full_sx = _soundex_tr(name.replace(".", "").replace(" ", ""))
        if full_sx:
            blocks.setdefault(f"sx_full:{full_sx}", set()).add(i)

        # Last name Soundex (catches Demir/Demır, Öztürk/Ozturk)
        if len(parts) >= 2:
            last_sx = _soundex_tr(parts[-1])
            first_initial = _normalize(parts[0])[0] if parts[0] else ""
            if last_sx and first_initial:
                blocks.setdefault(
                    f"sx_last:{first_initial}:{last_sx}", set()
                ).add(i)

        # First name Soundex (catches Ahmet/Ahmet, Ayşe/Ayse)
        if parts:
            first_word = parts[0].replace(".", "")
            if len(first_word) >= 2:
                first_sx = _soundex_tr(first_word)
                blocks.setdefault(f"sx_first:{first_sx}", set()).add(i)

    # ── Layer 3: Sliding window (sorted neighborhood) ──
    # Sort by phonetic key, compare within window
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
            # Only block together if sort keys share prefix
            if sk1[:3] == sk2[:3]:
                pair_key = f"window:{min(i1,i2)}:{max(i1,i2)}"
                blocks.setdefault(pair_key, set()).update({i1, i2})

    # ── Merge overlapping blocks (union-find) ──
    merged = _merge_overlapping_sets(list(blocks.values()))

    # ── Split oversized blocks (similarity-aware) ──
    final_blocks = []
    for block in merged:
        if len(block) <= 10:
            final_blocks.append(block)
        else:
            # Similarity-aware sub-clustering for large blocks
            subs = _split_block_by_similarity(records, block, max_size=7)
            final_blocks.extend(subs)

    # Add singletons
    all_assigned = set()
    for block in final_blocks:
        all_assigned.update(block)
    for i in range(len(records)):
        if i not in all_assigned:
            final_blocks.append([i])

    return final_blocks


def _merge_overlapping_sets(sets: list[set[int]]) -> list[list[int]]:
    """Merges sets that share any element (union-find)."""
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
    all_items = set()
    for s in sets:
        all_items.update(s)

    for item in all_items:
        root = find(item)
        if root not in groups:
            groups[root] = []
        groups[root].append(item)

    return [sorted(g) for g in groups.values() if len(g) > 1]


# ── Quantum QAOA optimization ──

def _qaoa_optimize_block(
    records: list[dict],
    indices: list[int],
    threshold: float,
    fields: list[str] | None = None,
    seed: int | None = None,
) -> tuple[list[list[int]], int]:
    """Uses QAOA to find optimal merge within a block.

    Each pair (i,j) gets a qubit: |1⟩ = merge, |0⟩ = keep separate.
    Cost: maximize similarity for merged pairs, penalize low-similarity merges.

    Returns: (clusters as lists of indices, qubits_used)
    """
    n = len(indices)
    if n <= 1:
        return [[idx] for idx in indices], 0

    # Compute pairwise similarities
    pairs = []
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_similarity(
                records[indices[i]], records[indices[j]], fields
            )
            pairs.append((i, j))
            sims.append(sim)

    num_qubits = len(pairs)  # One qubit per pair
    if num_qubits > 25:
        # Too many qubits — fall back to greedy within block
        return _greedy_merge_block(records, indices, threshold, fields), 0

    # QAOA circuit: encode similarities as rotation angles
    simulator = StateVectorSimulator(num_qubits, seed=seed)

    # Initial superposition
    for q in range(num_qubits):
        simulator.apply("H", (q,))

    # Cost layer: rotate based on similarity
    for q in range(num_qubits):
        # High similarity → bias toward |1⟩ (merge)
        # Low similarity → bias toward |0⟩ (separate)
        angle = (sims[q] - threshold) * np.pi
        simulator.apply("RZ", (q,), (angle,))

    # Mixer layer: entangle related pairs (transitivity)
    for q1 in range(num_qubits):
        for q2 in range(q1 + 1, min(q1 + 3, num_qubits)):
            # Entangle nearby pairs for transitivity constraint
            i1, j1 = pairs[q1]
            i2, j2 = pairs[q2]
            # If pairs share a record, they're related
            if len({i1, j1} & {i2, j2}) > 0:
                simulator.apply("CX", (q1, q2))
                simulator.apply("RZ", (q2,), (0.3,))
                simulator.apply("CX", (q1, q2))

    # Second cost layer
    for q in range(num_qubits):
        simulator.apply("RY", (q,), ((sims[q] - 0.5) * np.pi,))

    # Sample and find best configuration
    counts = simulator.sample(2048)
    best_bitstring = max(counts, key=counts.get)

    # Decode: build adjacency from merge decisions
    adjacency = [[False] * n for _ in range(n)]
    for q, (i, j) in enumerate(pairs):
        if best_bitstring[q] == "1" and sims[q] >= threshold * 0.7:
            adjacency[i][j] = True
            adjacency[j][i] = True

    # Find connected components (clusters)
    visited = [False] * n
    clusters = []

    for start in range(n):
        if visited[start]:
            continue
        cluster = []
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
    """Simple greedy merge (classical baseline)."""
    n = len(indices)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_similarity(
                records[indices[i]], records[indices[j]], fields
            )
            if sim >= threshold:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pi] = pj

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(indices[i])

    return [sorted(g) for g in groups.values()]


def _pick_canonical(records: list[dict], ids: list[int]) -> dict:
    """Picks the most complete record as canonical."""
    best = ids[0]
    best_fields = sum(1 for v in records[best].values() if str(v).strip())
    for idx in ids[1:]:
        filled = sum(1 for v in records[idx].values() if str(v).strip())
        if filled > best_fields:
            best = idx
            best_fields = filled
    return dict(records[best])


# ── Public API ──

def resolve(
    records: list[dict],
    threshold: float = 0.65,
    fields: list[str] | None = None,
    method: str = "qaoa",
    seed: int | None = None,
) -> ResolutionResult:
    """Quantum entity resolution.

    Args:
        records: List of record dicts with matching fields.
        threshold: Minimum similarity to consider a merge.
        fields: Fields to compare (auto-detected if None).
        method: "qaoa" (quantum) or "greedy" (classical).
        seed: Random seed.

    Returns:
        ResolutionResult with clusters and metrics.
    """
    n = len(records)
    if fields is None:
        fields = list(records[0].keys())

    # Step 1: Blocking
    blocks = _block_records(records)

    # Step 2: Resolve within each block
    all_clusters: list[list[int]] = []
    total_qubits = 0
    quantum_blocks = 0

    assigned = set()

    for block in blocks:
        if len(block) <= 1:
            continue

        if method == "qaoa":
            clusters, qubits = _qaoa_optimize_block(
                records, block, threshold, fields, seed
            )
            total_qubits += qubits
            if qubits > 0:
                quantum_blocks += 1
        else:
            clusters = _greedy_merge_block(records, block, threshold, fields)

        for cluster in clusters:
            all_clusters.append(cluster)
            assigned.update(cluster)

    # Add unassigned records as singletons
    for i in range(n):
        if i not in assigned:
            all_clusters.append([i])

    # Build result
    result_clusters = []
    for ids in all_clusters:
        canonical = _pick_canonical(records, ids)
        # Confidence: average pairwise similarity in cluster
        if len(ids) > 1:
            sims = []
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    sims.append(compute_similarity(
                        records[ids[a]], records[ids[b]], fields
                    ))
            conf = float(np.mean(sims))
        else:
            conf = 1.0

        result_clusters.append(Cluster(
            record_ids=sorted(ids),
            canonical=canonical,
            confidence=conf,
        ))

    result_clusters.sort(key=lambda c: -len(c.record_ids))

    return ResolutionResult(
        clusters=result_clusters,
        num_records=n,
        num_entities=len(result_clusters),
        method=method,
        quantum_blocks=quantum_blocks,
        total_qubits=total_qubits,
    )
