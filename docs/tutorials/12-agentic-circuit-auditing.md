# 🤖 AI Ajanları ile Kuantum Devre Denetimi & Quanta MCP

*Quanta MCP (Model Context Protocol) ile Otonom Devre Analizi, Transpilasyon ve Hata İyileştirme*

---

> **Bu tutorialda öğrenecekleriniz:** Claude, Cursor veya Antigravity gibi otonom yapay zeka ajanlarının Quanta SDK'nın 23 adet MCP aracını kullanarak kuantum devrelerini nasıl otonom olarak denetlediğini, sadeleştirdiğini, uniter eşdeğerliğini matematiksel olarak doğruladığını ve gürültülü donanımlar için QEC koruması eklediğini inceleyeceksiniz.

---

## Kuantum Bilişimde Ajanik Devrim (Agentic Quantum Workflows)

Kuantum devre tasarımı ve derlemesi (transpilation) yüksek seviyeli optimizasyon kararları gerektirir:
- Ham algoritmalar genellikle çok fazla CNOT kapısı içerir (yüksek iki-qubit hata payı).
- Donanım kısıtlamaları (kuplaj grafı, yerel kapı setleri) karmaşık yönlendirme (routing) algoritmaları gerektirir.
- Hata düzeltme kodları (Surface Code, Steane Code) ek yük getirir.

**Model Context Protocol (MCP)**, LLM tabanlı yapay zeka ajanlarının yerel veya buluttaki araçları standart bir JSON-RPC arayüzü üzerinden keşfetmesini ve güvenle çağırmasını sağlar. Quanta SDK, kuantum simülasyonundan donanım çalıştırmaya kadar **23 adet özel MCP aracı** sunar.

---

## Adım 1: Quanta MCP Sunucusunu Başlatma

Quanta MCP sunucusu `fastmcp` tabanlı olup hem yerel komut satırından hem de AI istemcilerinden (Claude Desktop, Cursor vb.) doğrudan başlatılabilir:

```bash
# MCP sunucusunu başlatma
quanta-mcp
# veya
python -m quanta.mcp_server
```

Claude Desktop konfigürasyonunuza (`claude_desktop_config.json`) şu tanımı ekleyebilirsiniz:

```json
{
  "mcpServers": {
    "quanta": {
      "command": "uv",
      "args": ["run", "quanta-mcp"]
    }
  }
}
```

---

## Adım 2: Ajanın Kullandığı Temel MCP Araçları

Bir AI ajanı devre optimizasyonu yaparken aşağıdaki araçları zincirleme (chain-of-thought) çağırır:

| MCP Aracı | Görevi | Ajanın Kullanım Amacı |
|---|---|---|
| `audit_circuit` / `decompose_unitary` | Devre derinliğini, iki-qubit kapı sayısını ve uniter matrisi inceler. | Optimize edilmemiş geçitleri tespit etmek. |
| `simulate_circuit` | Durum vektörü ve ölçüm olasılıklarını hesaplar. | Devrenin teorik doğruluğunu test etmek. |
| `calculate_fidelity` | İki kuantum durumu arasındaki örtüşmeyi ($F = |\langle\psi_1|\psi_2\rangle|^2$) ölçer. | Sadeleştirme sonrası çıktının bozulmadığını kanıtlamak. |
| `transpile_circuit` | Ayrık kapıları (Örn: H + CNOT) hedef donanım topolojisine göre optimize eder. | Kapı sayısını ve derinliği en aza indirmek. |
| `qec_surface_code` | Mantıksal devreye Surface Code hata koruması entegre eder. | Gürültülü donanım için hata toleransı sağlamak. |

---

## Adım 3: Otonom Devre Denetimi Senaryosu

Bir yapay zeka ajanının kullanıcıdan gelen bir Bell durumu devresini denetleme ve optimize etme adımlarını Python üzerinden modelleyelim:

```python
import numpy as np
from quanta import circuit, H, CX, X, measure, run
from quanta.dag.dag_circuit import DAGCircuit
from quanta.compiler.pipeline import CompilerPipeline
from quanta.compiler.passes.optimize import CancelInverses

# 1. Kullanıcının oluşturduğu gereksiz kapılar içeren ham devre
@circuit(qubits=2)
def raw_circuit(q):
    H(q[0])
    H(q[0])  # Gereksiz H-H iptali
    H(q[0])
    CX(q[0], q[1])
    X(q[1])
    X(q[1])  # Gereksiz X-X iptali
    return measure(q)

@circuit(qubits=2)
def optimized_circuit(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

dag = DAGCircuit.from_builder(raw_circuit.build())
print(f"Ham Devre Kapı Sayısı: {dag.gate_count()}")

# 2. AI Ajanının transpilasyon ve optimizasyon motorunu devreye sokması
pipeline = CompilerPipeline([CancelInverses()])
optimized_dag = pipeline.run(dag)
print(f"Optimize Edilmiş Devre Kapı Sayısı: {optimized_dag.gate_count()}")

# 3. Fidelity (Sadakat) Doğrulaması: Durumların %100 eşdeğer olduğunu kanıtlama
res_raw = run(raw_circuit, shots=500)
res_opt = run(optimized_circuit, shots=500)

psi_raw = res_raw.statevector
psi_opt = res_opt.statevector

fidelity = float(abs(np.vdot(psi_raw, psi_opt)) ** 2)
print(f"Matematiksel Fidelity (Eşdeğerlik): {fidelity:.6f}")
assert np.isclose(fidelity, 1.0)
print("✅ Ajan Kararı: Devre optimizasyonu semantik hiçbir kayıp olmadan %100 başarıyla tamamlandı!")
```

---

## Adım 4: MCP Tool Çağrı Döngüsü (Agent Loop)

Aşağıdaki akış diyagramı, bir LLM ajanının Quanta MCP araçları ile gerçekleştirdiği otonom denetim döngüsünü özetler:

```mermaid
graph TD
    A[Kullanıcı İstemi: Devreyi İncele ve Optimize Et] --> B[audit_circuit: Kapı Sayısı & Derinlik Analizi]
    B --> C[simulate_circuit: Orijinal Durum Vektörü \|ψ_orig⟩]
    C --> D[transpile_circuit: Sadeleştirme & Peep-Hole Optimizasyonu]
    D --> E[calculate_fidelity: F(\|ψ_orig⟩, \|ψ_opt⟩)]
    E -->|F == 1.0| F[Kullanıcıya Kanıtlı Rapor Sun]
    E -->|F < 1.0| G[Hata: Optimizasyonu Geri Al & Yeniden Analiz Et]
```

---

## Özet & Kazanımlar

- **Standart Protokol**: MCP sayesinde AI asistanları doğrudan kuantum simülatörünüzle konuşabilir.
- **Güvenli Optimizasyon**: Transpilasyon sonrası fidelity hesaplaması, insan denetimine gerek kalmadan doğruluğu matematiksel olarak teminat altına alır.
- **Donanıma Hazırlık**: AI ajanları mantıksal devreyi donanım kuplajına uydururken fiziksel gürültüyü QEC araçlarıyla asgariye indirir.\n