# Quanta SDK — Mimari (v1.2.0-production)

## Genel Bakış

Quanta SDK, 2026 yılı kuantum bilişim standartlarında tasarlanmış, **5 Temel Bilimsel Paradigma** üzerine inşa edilmiş modüler ve bağımsız bir kuantum yazılım mimarisidir. Sistem, sıfır harici C++/LLVM derleme bağımlılığı ile saf Python/NumPy çekirdeğini Apple Silicon Metal GPU hızlandırması, sürekli Hilbert uzayı türevleri ve 2026 çift-kanallı hata toleransı (FTQC) ile birleştirir.

## Katman Mimarisi

```
+---------------------------------------------------------------------------------+
|                       KATMAN 4: AJAN & MCP ORKESTRASYONU                        |
|  23 Model Context Protocol (MCP) Aracı | Claude, Gemini, GPT Otonom Kuantum Ajanı|
|  "Kuantum iş akışlarını doğal dil ve otonom döngülerle yönetme"                 |
+---------------------------------------------------------------------------------+
|               KATMAN 3: DERİN ÖĞRENME & DEKLARATİF BİLİŞSEL API                 |
|  quanta.torch: QuantumLayer | Daleckii-Krein Autograd | Lie Cebiri Barren Analizi|
|  BiomorphicResonantBrain | SWR Replay | ContinuousResonator (Schrödinger Akışı)  |
|  search() | optimize() | vqe() | factor() | portfolio_optimize() | resolve()    |
|  "Ne çözülecek?" -- kapı sentezi gerekmeden doğrudan algoritmik çözüm           |
+---------------------------------------------------------------------------------+
|                       KATMAN 2: ALGORİTMİK DEVRE DSL                            |
|  @circuit | 31 Yerleşik Kapı (IBM Heron, Google Sycamore, IonQ yerel paritesi)  |
|  Parametrik Rotasyonlar (RX, RY, RZ, P, U) | measure() | sweep() | equivalence  |
|  "Devre nasıl kurgulanacak?"                                                    |
+---------------------------------------------------------------------------------+
|             KATMAN 1: 2026 ÇİFT-KANAL HATA TOLERANSI (FTQC MOTORU)              |
|  Track A: Edmonds Blossom MWPM | Willow Uyumlu 3D Uzay-Zaman Sendrom Döngüleri  |
|  Track B: Gross [[144, 12, 12]] qLDPC | Yerel Normalized Min-Sum BP-OSD-0       |
|  Non-Clifford: 15-to-1 Bravyi-Kitaev Sihirli Durum Damıtma | Örgü Cerrahisi     |
+---------------------------------------------------------------------------------+
|                 KATMAN 0: FİZİKSEL HESAPLAMA & DONANIM MOTORU                   |
|  DAG Devresi (Kahn) | Compiler Pipeline (CancelInverses, MergeRotations, Route) |
|  Metal/MLX Sıfır-Kopya GPU (52.09x) | SIMD Clifford (>3.13M g/s) | MPS (250q)   |
|  7-Kanal Kraus Lindblad Gürültü | Çoklu-Bulut (IBM REST, Google Cirq, IonQ)    |
|  "Donanım ve simülatörler üzerinde en yüksek başarımla nasıl yürütülecek?"      |
+---------------------------------------------------------------------------------+
```

## Bağımlılık Grafi

```
mcp_server.py ──┐
                ▼
      quanta.torch / layer3/ ───────► simulator/ ───────► core/
                 │                        │                 ▲
                 ▼                        ▼                 │
             qec/ (FTQC) ────────────► dag/ ────────────────┘
                 │                        ▲
                 ▼                        │
            compiler/ ────────────────────┘
                 │
                 ▼
         backends/ & export/
```

**Kural**: Bağımlılıklar daima aşağı ve çekirdeğe doğru akar. Çekirdek (`core/`) harici hiçbir katmana bağımlı değildir.

## Modul Detaylari

### core/ -- Temel Yapi Taslari

| Dosya | Sorumluluk |
|-------|------------|
| `types.py` | QubitRef, Instruction, QubitRegister |
| `gates.py` | 31 kapi + broadcast (IBM Heron paritesi) |
| `circuit.py` | @circuit dekoratoru, CircuitBuilder |
| `measure.py` | Esnek olcum (tam, kismi) |
| `equivalence.py` | Uniter karsilastirma, sadakat |
| `custom_gate.py` | Kullanici tanimli uniter kapilar |

### dag/ -- Yonlu Dongusuz Graf

| Dosya | Sorumluluk |
|-------|------------|
| `node.py` | InputNode, OpNode, OutputNode |
| `dag_circuit.py` | Topolojik siralama (Kahn), derinlik, paralel katmanlar |

### compiler/ -- Optimizasyon Hatti

| Dosya | Sorumluluk |
|-------|------------|
| `pipeline.py` | CompilerPass Protokolu, zincirleme, istatistikler |
| `passes/optimize.py` | CancelInverses (H.H=I), MergeRotations |
| `passes/translate.py` | IBM/Google/Quantinuum kapi seti cevirisi |
| `passes/routing.py` | Topoloji bazli SWAP ekleme (linear/ring/grid) |

### simulator/ -- Simulasyon Motorlari

| Dosya | Sorumluluk |
|-------|------------|
| `base.py` | `SimulatorBackend` ABC — tüm simülatörler için soyut arayüz |
| `statevector.py` | Yoğun tensor contraction, 27 qubite kadar (tam doğru) |
| `sparse.py` | Sözlük tabanlı seyrek statevector, 50 qubite kadar, O(k) bellek |
| `mps.py` | Matris Çarpım Durumu (SVD), 200+ qubit, O(n·χ²) bellek |
| `factory.py` | `create_simulator()` — en iyi arka ucu otomatik seçer |
| `router.py` | Devre-duyarlı yönlendirme (Clifford algılama, qubit sayısı) |
| `density_matrix.py` | Karisik durumlar + Kraus gurultu, 13 qubite kadar |
| `pauli_frame.py` | Aaronson-Gottesman stabilizer tablosu, 50-qubit GHZ <5s |
| `noise.py` | 7 gurultu kanali: Depolarizing, BitFlip, PhaseFlip, AmplitudeDamping, T2Relaxation, Crosstalk, ReadoutError |
| `accelerated.py` | JAX-GPU / CuPy otomatik algilama, NumPy fallback |

### layer3/ -- Deklaratif API

| Dosya | Sorumluluk |
|-------|------------|
| `search.py` | Otomatik Grover aramasi |
| `optimize.py` | QAOA optimizasyonu |
| `agent.py` | Coklu ajan karar modelleme |
| `vqe.py` | Variasyonel Kuantum Ozdeger Cozucu |
| `shor.py` | Tam sayi carpanlara ayirma (periyot bulma + QFT) |
| `qsvm.py` | Kuantum cekirdek SVM siniflandirma |
| `finance.py` | Portfoy optimizasyonu (Markowitz + QAOA) |
| `hamiltonian.py` | Trotter zaman evrimi, molekuler Hamiltonianlar |
| `entity_resolution.py` | QAOA tabanli musteri tekillestime |
| `monte_carlo.py` | Kuantum Monte Carlo, genlik kestirimi, opsiyon fiyatlama |
| `clustering.py` | Kuantum swap-test uzakliklari + k-means kumeleme |
| `qml.py` | Kuantum ML: variasyonel siniflandirici, kuantum cekirdek, ozellik haritalari |

### export/ -- QASM Giris/Cikis

| Dosya | Sorumluluk |
|-------|------------|
| `qasm.py` | OpenQASM 3.0 cikti |
| `qasm_import.py` | QASM 2.0/3.0 girdi -> DAG |

### qec/ -- 2026 Çift-Kanal Hata Düzeltme (FTQC)

| Dosya | Sorumluluk |
|-------|------------|
| `codes.py` | BitFlip [[3,1,3]], PhaseFlip [[3,1,3]], Steane [[7,1,3]] |
| `surface_code.py` | Döndürülmüş Surface Code [[d^2,1,d]], 3D uzay-zaman sendrom döngüleri |
| `color_code.py` | 2D Üçgensel Color Code, transversal Clifford, restriction dekoderi |
| `decoder.py` | Edmonds Blossom MWPM (tam ağırlıklı mükemmel eşleme) ve Union-Find |
| `qldpc.py` | Gross [[144, 12, 12]] Bivariate Bicycle kodu, Normalized Min-Sum BP-OSD-0 |
| `distillation.py` | 15-to-1 Bravyi-Kitaev sihirli durum damıtma fabrikası, örgü cerrahisi |

### quanta.torch / cognitive/ -- Derin Öğrenme & Biyomorfik Kuantum Motoru

| Modül / Dosya | Sorumluluk |
|---------------|------------|
| `quanta.torch.QuantumLayer` | PyTorch nn.Module katmanı, analitik parameter-shift autograd VJP |
| `quanta.torch.ContinuousResonator` | Sürekli Schrödinger akışı, Daleckii-Krein analitik Fréchet türevleri |
| `quanta.torch.BiomorphicResonantBrain` | Çift hemisferli, 4-nöromodülatörlü (DA, ACh, 5-HT, NE) kuantum beyni |
| `quanta.torch.lie_algebra` | Dinamik Lie cebiri dim(g) boyutu ve analitik barren plateau teşhis motoru |
| `quanta.cognitive` | SWR hafıza pekiştirme, REM uyku konsolidasyonu, CSF faz kalkanı |

### benchmark/ -- Kalite & Kıyaslama Ölçümü

| Dosya | Sorumluluk |
|-------|------------|
| `qasmbench.py` | 10 standart + 3 büyük QASMBench devresi |
| `benchpress_adapter.py` | SDK arası karşılaştırma API'si (Nation et al.) |
| `run_paper_benchmarks.py` | Hakemli yayın için ampirik mikrosaniye kıyaslamaları |

### Destek Modülleri

| Dosya | Sorumluluk |
|-------|------------|
| `runner.py` | 6 aşamalı orkestratör: build > DAG > compile > sim > noise > sample > result |
| `result.py` | Ölçüm sonuçları, olasılıklar, Dirac notasyonu, durum vektörü |
| `visualize.py` | ASCII ve SVG devre diyagramları |
| `visualize_state.py` | Olasılık histogramı, Bloch küresi, faz diyagramı |
| `mcp_server.py` | MCP sunucusu — Otonom AI ajanları için **23 kuantum aracı** (SSE + stdio) |

## Veri Akışı

```
Kullanıcı Kodu / AI Ajanı (MCP)
          │
          ▼
   @circuit / layer3 / quanta.torch
          │
          ▼
   DAGCircuit (Kahn topolojik sıralama)
          │
          ▼
   CompilerPipeline (CancelInverses, MergeRotations, Routing)
          │
          ▼
   QEC Koruma Katmanı (Edmonds Blossom MWPM / Gross qLDPC BP-OSD)
          │
          ▼
   Yürütme Arka Ucu (Metal/MLX, StateVector, Clifford SIMD, MPS, veya IBM/Google/IonQ Donanımı)
          │
          ▼
   Result (Ölçüm sayımları, durum vektörü, analitik gradyanlar, hata sendromları)
```

## Tasarım Kararları

1. **İlk-İlkelerden Bağımsızlık**: Ağır C++/LLVM derleme zincirleri olmaksızın, saf Python ve NumPy ile edge cihazlardan HPC kümelere kadar tam taşınabilirlik.
2. **Apple Silicon Sıfır-Kopya**: Metal Performance Shaders / MLX ile birleşik bellekte CPU-GPU kopyalama gecikmesini sıfırlayan doğrudan GPU tensör hızlandırması.
3. **Analitik Hilbert Gradyanları**: Padé sapmalarını ortadan kaldıran Daleckii-Krein Fréchet türevleri ve Lie cebiri barren plateau garantisi.
4. **2026 FTQC Çift-Kanal**: Hem 2D yüzey kodlarında Edmonds Blossom MWPM hem de yüksek dereceli Gross qLDPC BP-OSD ile 12 kat donanım tasarrufu.
5. **AI-Native MCP Mimarisi**: 23 adet yerleşik MCP aracı ile Claude, GPT ve Gemini ajanlarının kuantum optimizasyonu ve denetimini doğrudan yürütebilmesi.
6. **Yanlışlanabilir Bilimsel Titizlik (Falsifiable Empiricism)**: 2.076 adet regresyonsuz test ile üniterlik ve CPTP iz korunum garantisi.

---

## Mimari Künyesi & Yazarlık

- **Baş Mimar**: Abdullah Enes SARI (ORCID: [0000-0002-8827-0587](https://orcid.org/0000-0002-8827-0587))
- **Kurum**: ONMARTECH Kuantum Bilişim İnisiyatifi (`info@onmartech.com`)
- **Yazılım DOI**: [10.5281/zenodo.22952779](https://doi.org/10.5281/zenodo.22952779)

