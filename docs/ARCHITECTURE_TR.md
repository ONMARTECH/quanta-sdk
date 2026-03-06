# Quanta SDK — Mimari Dökümantasyonu

## Genel Bakış

Quanta, **3 katmanlı bağımsız mimari** ile tasarlanmıştır. Her katman bir üstüne bağımsız olarak kullanılabilir.

## Katman Diyagramı

```
┌─────────────────────────────────────────────────────┐
│              KATMAN 3: DEKLARATİF API               │
│  search()  │  optimize()  │  MultiAgentSystem       │
│  "Ne istiyorsun?" — Gate bilgisi GEREKMEZ            │
├─────────────────────────────────────────────────────┤
│              KATMAN 2: ALGORİTMİK DSL               │
│  @circuit  │  H/CX/RZ  │  measure()  │  run()      │
│  "Devreyi nasıl kuracağız?"                          │
├─────────────────────────────────────────────────────┤
│              KATMAN 1: FİZİKSEL MOTOR               │
│  DAG  │  Compiler  │  Simulator  │  QEC  │  Export  │
│  "Donanımda nasıl çalışacak?"                        │
└─────────────────────────────────────────────────────┘
```

## Bağımlılık Grafiği

```
layer3/ ──→ simulator/ ──→ core/
                │
runner.py ──→ dag/ ──→ core/
                │
compiler/ ──→ dag/ ──→ core/
                │
backends/ ──→ simulator/ ──→ core/
                │
export/ ──→ dag/ ──→ core/
                │
qec/ ──→ core/
```

**Kural**: Bağımlılık her zaman aşağı yönlüdür. Hiçbir alt katman, üst katmana bağımlı değildir.

## Modül Detayları

### core/ — Temel Yapı Taşları

| Dosya | Satır | Sorumluluk |
|-------|-------|------------|
| `types.py` | 164 | QubitRef, Instruction, QubitRegister, hata sınıfları |
| `gates.py` | 321 | 14 standart kapı + 3 parametrik (RX/RY/RZ) + broadcast |
| `circuit.py` | 174 | @circuit dekoratörü, CircuitBuilder, CircuitDefinition |
| `measure.py` | 66 | Esnek ölçüm (tam, kısmi, tekli) |
| `equivalence.py` | 153 | Unitär karşılaştırma, devre fidelity |

### dag/ — Yönlü Çizge Motoru

| Dosya | Satır | Sorumluluk |
|-------|-------|------------|
| `node.py` | 78 | InputNode, OpNode, OutputNode (immutable) |
| `dag_circuit.py` | 227 | Topolojik sort (Kahn), derinlik, paralel katmanlar |

### compiler/ — Optimizasyon Pipeline'ı

| Dosya | Satır | Sorumluluk |
|-------|-------|------------|
| `pipeline.py` | 137 | CompilerPass Protocol, zincirleme pipeline, istatistik |
| `passes/optimize.py` | 232 | CancelInverses (H·H=I), MergeRotations (RZ(a)+RZ(b)) |
| `passes/translate.py` | 180 | IBM/Google/Quantinuum gate set transpilasyonu |

### simulator/ — Simülasyon Motoru

| Dosya | Satır | Sorumluluk |
|-------|-------|------------|
| `statevector.py` | 233 | NumPy tam durum vektörü, Kronecker genişletme |
| `noise.py` | 236 | 4 gürültü kanalı: Depolarizing, BitFlip, PhaseFlip, AmplDamp |

### layer3/ — Deklaratif API

| Dosya | Satır | Sorumluluk |
|-------|-------|------------|
| `search.py` | 152 | Grover otomatik — hedef ver, bulsun |
| `optimize.py` | 194 | QAOA tabanlı — maliyet fonksiyonu ver, optimize etsin |
| `agent.py` | 265 | Kuantum karar modelleme — ajanlar, etkileşim, korelasyon |

### Yardımcı Modüller

| Dosya | Satır | Sorumluluk |
|-------|-------|------------|
| `runner.py` | 129 | 6 aşamalı orkestratör: build→DAG→compile→sim→sample→result |
| `result.py` | 89 | Ölçüm sonuçları, olasılıklar, özet |
| `visualize.py` | 114 | ASCII devre çizimi |
| `visualize_state.py` | 166 | Olasılık histogram, faz diyagramı |
| `export/qasm.py` | 162 | OpenQASM 3.0 export + parser |
| `qec/codes.py` | 226 | BitFlip [[3,1,1]], Steane [[7,1,3]] |
| `backends/base.py` | 68 | Backend soyut sınıfı |
| `backends/local.py` | 94 | Lokal NumPy simülatör backend |

## Veri Akışı

```
Kullanıcı Kodu          SDK İç Yapısı
     │                       │
 @circuit(qubits=N) ──→ CircuitDefinition
     │                       │
 H(q[0]), CX(...)    ──→ CircuitBuilder (lazy Instruction listesi)
     │                       │
 measure(q)          ──→ MeasureSpec
     │                       │
 run(circuit)        ──→ ┌─ DAGCircuit.from_builder()
                         ├─ CompilerPipeline.run(dag)
                         ├─ StateVectorSimulator.apply(ops)
                         ├─ simulator.sample(shots)
                         └─ Result(counts, probs, statevector)
```

## Tasarım Kararları

1. **Lazy Evaluation**: Kapılar anında uygulanmaz, Instruction olarak kaydedilir
2. **DAG Temsili**: Paralellik tespiti ve optimizasyon için zorunlu
3. **Protocol tabanlı**: CompilerPass bir Protocol — duck typing yeterli
4. **Immutable**: QubitRef, Instruction, node'lar frozen dataclass
5. **Thread-local Builder**: Birden fazla devre eşzamanlı oluşturulabilir
