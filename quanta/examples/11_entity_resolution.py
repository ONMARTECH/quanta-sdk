"""
Example 11: Quantum Entity Resolution — OTA Customer Dedup

Real-world scenario: An online travel agency (OTA) has duplicate
customer records from multiple booking channels. Same person,
different spellings, missing fields.

Uses QAOA (quantum optimization) to find optimal merge decisions
that a greedy classical approach might miss.

Pipeline:
  1. Classical: Compute fuzzy similarity scores
  2. Classical: Blocking (group likely matches)
  3. QUANTUM: QAOA optimizes merge within blocks
  4. Classical: Merge and produce golden records

Running:
    python -m quanta.examples.11_entity_resolution
"""

from quanta.layer3.entity_resolution import resolve, compute_similarity


# ── 25 OTA customer records (8 columns each) ──
# Intentional duplicates, typos, missing data, Turkish chars

RECORDS = [
    # Cluster A: Ahmet Yılmaz (4 records)
    {"name": "Ahmet Yılmaz",  "phone": "532-111-2233", "email": "ahmet@gmail.com",
     "tc": "12345678901", "birth_date": "1985-03-15", "city": "İstanbul",
     "address": "Kadıköy Moda Cad. 12", "channel": "web"},

    {"name": "A. Yılmaz",     "phone": "532-111-2233", "email": "ahmet.yilmaz@mail.com",
     "tc": "", "birth_date": "1985-03-15", "city": "İstanbul",
     "address": "", "channel": "mobile"},

    {"name": "Ahmet Yilmaz",  "phone": "", "email": "ahmet@gmail.com",
     "tc": "12345678901", "birth_date": "", "city": "Ankara",
     "address": "Çankaya", "channel": "call_center"},

    {"name": "Ahmet Yılmazz", "phone": "532-111-2234", "email": "ahmet@gmail.com",
     "tc": "", "birth_date": "1985-03-15", "city": "İstanbul",
     "address": "Kadıköy", "channel": "web"},

    # Cluster B: Mehmet Demir (3 records)
    {"name": "Mehmet Demir",  "phone": "555-444-3322", "email": "mdemir@hotmail.com",
     "tc": "98765432109", "birth_date": "1990-07-22", "city": "İzmir",
     "address": "Alsancak Kordon", "channel": "web"},

    {"name": "M. Demir",      "phone": "555-444-3322", "email": "",
     "tc": "98765432109", "birth_date": "", "city": "İzmir",
     "address": "", "channel": "mobile"},

    {"name": "Mehmet Demır",  "phone": "", "email": "mdemir@hotmail.com",
     "tc": "", "birth_date": "1990-07-22", "city": "Izmir",
     "address": "Alsancak", "channel": "agency"},

    # Cluster C: Ayşe Kara (3 records)
    {"name": "Ayşe Kara",     "phone": "544-777-8899", "email": "ayse.kara@gmail.com",
     "tc": "11223344556", "birth_date": "1992-11-30", "city": "Bursa",
     "address": "Nilüfer Beşevler", "channel": "web"},

    {"name": "Ayse Kara",     "phone": "544-777-8899", "email": "",
     "tc": "", "birth_date": "1992-11-30", "city": "Bursa",
     "address": "", "channel": "mobile"},

    {"name": "A. Kara",       "phone": "", "email": "ayse.kara@gmail.com",
     "tc": "11223344556", "birth_date": "", "city": "Bursa",
     "address": "Nilüfer", "channel": "call_center"},

    # Cluster D: Fatih Öztürk (2 records)
    {"name": "Fatih Öztürk",  "phone": "533-222-4455", "email": "fatih.ozturk@yahoo.com",
     "tc": "55566677788", "birth_date": "1988-01-10", "city": "Antalya",
     "address": "Lara Cad. 45", "channel": "web"},

    {"name": "Fatih Ozturk",  "phone": "533-222-4455", "email": "",
     "tc": "", "birth_date": "1988-01-10", "city": "Antalya",
     "address": "", "channel": "agency"},

    # Cluster E: Zeynep Aksoy (2 records)
    {"name": "Zeynep Aksoy",  "phone": "542-333-5566", "email": "zaksoy@outlook.com",
     "tc": "99988877766", "birth_date": "1995-05-20", "city": "Ankara",
     "address": "Kızılay Atatürk Blv.", "channel": "web"},

    {"name": "Z. Aksoy",      "phone": "542-333-5566", "email": "zaksoy@outlook.com",
     "tc": "", "birth_date": "", "city": "Ankara",
     "address": "", "channel": "mobile"},

    # Singletons (unique records, no duplicates)
    {"name": "Can Yücel",     "phone": "536-999-1122", "email": "can.yucel@icloud.com",
     "tc": "44433322211", "birth_date": "1978-08-05", "city": "Muğla",
     "address": "Bodrum Yalı Mah.", "channel": "web"},

    {"name": "Elif Şahin",    "phone": "537-888-4433", "email": "elif.sahin@gmail.com",
     "tc": "33322211100", "birth_date": "1993-12-01", "city": "İstanbul",
     "address": "Beşiktaş Levent", "channel": "web"},

    {"name": "Burak Çelik",   "phone": "538-777-6655", "email": "bcelik@gmail.com",
     "tc": "22211100099", "birth_date": "1987-06-18", "city": "Konya",
     "address": "Selçuklu Meram", "channel": "agency"},

    {"name": "Deniz Aydın",   "phone": "539-666-7788", "email": "deniz.aydin@hotmail.com",
     "tc": "11100099988", "birth_date": "1991-09-25", "city": "Adana",
     "address": "Seyhan Merkez", "channel": "web"},

    {"name": "Gül Yıldırım",  "phone": "540-555-8899", "email": "gul.yildirim@yahoo.com",
     "tc": "00099988877", "birth_date": "1983-04-12", "city": "Trabzon",
     "address": "Ortahisar", "channel": "web"},

    {"name": "Hakan Arslan",  "phone": "541-444-9900", "email": "harslan@gmail.com",
     "tc": "88877766655", "birth_date": "1989-02-28", "city": "Eskişehir",
     "address": "Tepebaşı", "channel": "mobile"},

    # Tricky edge cases
    {"name": "Ali Yılmaz",    "phone": "535-000-1111", "email": "ali.yilmaz@gmail.com",
     "tc": "77766655544", "birth_date": "1994-10-03", "city": "İstanbul",
     "address": "Kadıköy", "channel": "web"},  # Different person, same surname + city as Ahmet!

    {"name": "Mehmet Kara",   "phone": "534-111-0000", "email": "m.kara@gmail.com",
     "tc": "66655544433", "birth_date": "1996-03-17", "city": "Bursa",
     "address": "Osmangazi", "channel": "web"},  # Same surname as Ayşe, different person!

    # Another Mehmet Demir edge case (different person, same name!)
    {"name": "Mehmet Demir",  "phone": "546-222-3344", "email": "mehmet.d@gmail.com",
     "tc": "55544433322", "birth_date": "1975-11-08", "city": "Samsun",
     "address": "Atakum", "channel": "web"},

    # Near-duplicate with very similar data
    {"name": "Fatih Öztürk",  "phone": "", "email": "fatih.ozturk@yahoo.com",
     "tc": "55566677788", "birth_date": "1988-01-10", "city": "Antalya",
     "address": "Lara", "channel": "web"},

    {"name": "Zeynep Aksoy",  "phone": "", "email": "",
     "tc": "99988877766", "birth_date": "1995-05-20", "city": "",
     "address": "", "channel": "call_center"},
]

# Ground truth: which records belong together
GROUND_TRUTH = {
    "Ahmet Yılmaz":  [0, 1, 2, 3],
    "Mehmet Demir":   [4, 5, 6],
    "Ayşe Kara":      [7, 8, 9],
    "Fatih Öztürk":   [10, 11, 23],
    "Zeynep Aksoy":   [12, 13, 24],
    "Can Yücel":      [14],
    "Elif Şahin":     [15],
    "Burak Çelik":    [16],
    "Deniz Aydın":    [17],
    "Gül Yıldırım":   [18],
    "Hakan Arslan":   [19],
    "Ali Yılmaz":     [20],
    "Mehmet Kara":    [21],
    "Mehmet Demir 2": [22],
}


def evaluate_accuracy(result, ground_truth):
    """Compare resolution result against ground truth."""
    gt_clusters = list(ground_truth.values())
    gt_sets = [frozenset(c) for c in gt_clusters]
    res_sets = [frozenset(c.record_ids) for c in result.clusters]

    correct = sum(1 for gs in gt_sets if gs in res_sets)
    return correct / len(gt_sets)


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Quantum Entity Resolution — OTA Customer Dedup     ║")
    print("╚══════════════════════════════════════════════════════╝")

    print(f"\n  Records: {len(RECORDS)}")
    print(f"  Columns: {len(RECORDS[0])} (name, phone, email, tc, birth, city, address, channel)")
    print(f"  True entities: {len(GROUND_TRUTH)}")

    fields = ["name", "phone", "email", "tc", "birth_date", "city", "address"]

    # ── QAOA (Quantum) ──
    print("\n  ▸ Running QAOA (quantum)...")
    qaoa_result = resolve(RECORDS, threshold=0.60, fields=fields, method="qaoa", seed=42)
    qaoa_acc = evaluate_accuracy(qaoa_result, GROUND_TRUTH)
    qaoa_result.comparison_accuracy = qaoa_acc
    print(qaoa_result.summary())

    # ── Greedy (Classical) ──
    print("\n  ▸ Running Greedy (classical)...")
    greedy_result = resolve(RECORDS, threshold=0.60, fields=fields, method="greedy", seed=42)
    greedy_acc = evaluate_accuracy(greedy_result, GROUND_TRUTH)
    greedy_result.comparison_accuracy = greedy_acc
    print(greedy_result.summary())

    # ── Comparison ──
    print("\n  ╔═══════════════════════════════════════╗")
    print("  ║  QAOA vs Greedy Comparison             ║")
    print("  ╠═══════════════════════════════════════╣")
    print(f"  ║  QAOA:   {qaoa_result.num_entities:>2} entities, "
          f"accuracy {qaoa_acc:.0%}  "
          f"({qaoa_result.total_qubits} qubits)  ║")
    print(f"  ║  Greedy: {greedy_result.num_entities:>2} entities, "
          f"accuracy {greedy_acc:.0%}"
          f"{'':>17}  ║")
    print(f"  ║  Ground: {len(GROUND_TRUTH):>2} entities"
          f"{'':>27}  ║")
    print("  ╚═══════════════════════════════════════╝")

    # ── Scaling analysis ──
    print("\n  Scaling projection (hybrid blocking + QAOA):")
    print("  ┌────────────┬────────────┬───────────┬──────────┐")
    print("  │  Records   │  Blocks    │  Qubits/b │  Method  │")
    print("  ├────────────┼────────────┼───────────┼──────────┤")
    print(f"  │  {len(RECORDS):>8}   │  {qaoa_result.quantum_blocks:>8}   │  "
          f"≤25       │  QAOA    │")
    print("  │      100   │     ~20    │  ≤15      │  QAOA    │")
    print("  │    1,000   │    ~200    │  ≤21      │  QAOA    │")
    print("  │   10,000   │   ~2000    │  ≤21      │  Hybrid  │")
    print("  │  100,000   │  ~20000    │  ≤21      │  Hybrid  │")
    print("  └────────────┴────────────┴───────────┴──────────┘")
    print("  * Block size 5-7 keeps qubits ≤21 (simulator limit: 27)")
    print("  * Real quantum hardware: blocks of 20+ feasible")


if __name__ == "__main__":
    main()
