"""Demonstration of Quanta SDK & Antigravity Cognitive Integration."""

import json

from quanta.cognitive import CognitiveMemoryManager, QuantumDecisionArbiter
from quanta.mcp_server import estimate_fault_tolerant_cost, quanta_reasoning_eval


def demo_cognitive_memory():
    print("=" * 70)
    print("1. BİYOMORFİK HAFIZA VE BAĞLAM TAKİBİ (NoisyHippocampalBuffer + CSF)")
    print("=" * 70)

    mem = CognitiveMemoryManager(capacity=10, enable_csf_shielding=True)

    print("→ Hafızaya 3 farklı kural ve ara karar kaydediliyor...")
    mem.record_decision(
        key="kritik_guvenlik_kurali",
        content="Üretim ortamında asla raw JWT ve API anahtarlarını loglama",
        salience=2.5,  # Çok yüksek dopamin koruması
        category="constraint",
    )
    mem.record_decision(
        key="mimari_tercih",
        content="Veritabanı erişiminde Prisma ORM yerine yerel query builder kullan",
        salience=1.2,
        category="decision",
    )
    mem.record_decision(
        key="gecici_not",
        content="Lokal test için port 3005 geçici olarak açıldı",
        salience=0.3,  # Düşük koruma, bilinçli sönümlenme
        category="temporary",
    )

    print("\n→ 25 sohbet adımı boyunca Lindblad faz difüzyonu simüle ediliyor...")
    for _ in range(25):
        mem.step(dt=1.0)

    recalled = mem.recall_vital_context(top_k=3)
    print("\n[SWR Replay ile Geri Çağrılan Hayati Bağlam]:")
    for r in recalled:
        pct = r["retention_fidelity"] * 100
        print(f" • [{r['key']}] (Salience: {r['salience']}, Sadakat: %{pct:.3f})")
        print(f"   İçerik: \"{r['content']}\"")

    status = mem.get_status_summary()
    mean_pct = status["mean_retention_pct"]
    print(f"\n→ CSF Kalkanı: {status['csf_shielded']}, Ortalama Sadakat: %{mean_pct:.2f}")

    print("\n→ [Bilinçli Sinaptik Budama (Conscious Synaptic Pruning) Çalıştırılıyor]...")
    pruned = mem.prune_obsolete(fidelity_threshold=0.70, min_salience=0.5)
    for p in pruned:
        fid = p["final_fidelity"] * 100
        print(
            f" ✂ [BUDANDI / UNUTULDU]: '{p['key']}' "
            f"(Son Sadakat: %{fid:.2f}, Neden: {p['reason']})"
        )

    remaining = mem.recall_vital_context(top_k=5)
    print(f"\n→ Budama Sonrası Aktif Hafıza ({len(remaining)} engram korundu):")
    for r in remaining:
        print(f" • [{r['key']}] (Sadakat: %{r['retention_pct']}, Durum: {r['status']})")


def demo_quantum_arbiter():
    print("\n" + "=" * 70)
    print("2. KUANTUM ZENO KARAR HAKEMİ (QuantumZenoAttention)")
    print("=" * 70)

    arb = QuantumDecisionArbiter(dim=16, num_heads=2)

    goal = "Yüksek eşzamanlılıkta (high-concurrency) minimum gecikmeli önbellek tasarımı"
    options = [
        "A: In-memory Redis kümesi ve asenkron write-behind yazma",
        "B: Her istekte doğrudan PostgreSQL üzerinde synchronous transaction",
        "C: Lokal sqlite dosyasına disk lock ile append-only loglama",
    ]

    print(f"Hedef: {goal}")
    print("Aday Seçenekler:")
    for opt in options:
        print(f" - {opt}")

    print("\n→ Kuantum Zeno Odak Kitleme (Zeno Pinning) ile değerlendiriliyor...")
    res = arb.arbitrate(goal, options, exploration_drive=0.15)

    print(f"\n[Önerilen Seçenek]: {res['recommended_option']}")
    print(f"[Güven Skoru]: %{res['confidence']*100:.1f}")
    print(f"[Zeno Kitleme Faktörü]: {res['zeno_pinning_factor']} ({res['regime']})")
    print("[Sıralama]:")
    for item in res["ranked_options"]:
        print(f"  * %{item['score']*100:.1f} - {item['option']}")


def demo_mcp_evaluation():
    print("\n" + "=" * 70)
    print("3. QUANTA MCP AKIL YÜRÜTME & FTQC MALİYET HESABI")
    print("=" * 70)

    code = (
        "@circuit(qubits=3)\n"
        "def circ(q):\n"
        "    H(q[0])\n"
        "    CX(q[0], q[1])\n"
        "    CX(q[1], q[2])\n"
        "    T(q[0])\n"
        "    CX(q[1], q[2])\n"
        "    return measure(q)\n"
    )

    print("→ Örnek devre üzerinde 'quanta_reasoning_eval' çalıştırılıyor...")
    eval_res = json.loads(quanta_reasoning_eval(circuit_code=code))
    print(f"Devre Derinliği: {eval_res['circuit_depth']}, Toplam Kapı: {eval_res['total_gates']}")
    print(f"Dolaşıklık Oranı: {eval_res['entangling_gate_ratio']}")
    print(f"İptal Edilebilir Kapı Sayısı: {eval_res['cancellable_gates']}")
    print(f"Önerilen Simülasyon Rejimi: {eval_res['recommended_simulation_regime']}")
    if eval_res.get("agent_feedback"):
        print(f"Ajanik Öneri: {eval_res['agent_feedback']}")

    print("\n→ Google Willow / FTQC için donanım maliyet tahmini alınıyor...")
    ftqc_cost = json.loads(
        estimate_fault_tolerant_cost(circuit_code=code, target_logical_error_rate=1e-10)
    )
    print(f"Gereken Kod Mesafesi (d): {ftqc_cost['surface_code_distance']}")
    print(f"Toplam Fiziksel Kübit İhtiyacı: {ftqc_cost['total_physical_qubits']}")
    print(f"T-Factory Ayak İzi: {ftqc_cost['physical_qubits_t_factory']} fiziksel kübit")
    print(f"Mimari Referans: {ftqc_cost['architecture']}")


if __name__ == "__main__":
    demo_cognitive_memory()
    demo_quantum_arbiter()
    demo_mcp_evaluation()
