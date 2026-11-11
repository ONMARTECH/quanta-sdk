# Quanta SDK — Mimari (v1.2.0-production)

## Genel Bakış

Quanta SDK, 2026 yılı kuantum bilişim standartlarında tasarlanmış, **5 Temel Bilimsel Paradigma** üzerine inşa edilmiş modüler ve bağımsız bir kuantum yazılım mimarisidir. Çekirdek devre, derleyici ve statevector motorları yalnızca NumPy gerektirir (SciPy ve NetworkX Katman-3 çözücülerini ve QEC dekoderlerini destekler); sistem bu çekirdeği Apple Silicon Metal GPU hızlandırması, sürekli Hilbert uzayı türevleri ve 2026 çift-kanallı hata toleransı (FTQC) ile birleştirir.

## Katman Mimarisi

```
+---------------------------------------------------------------------------------+
|                       KATMAN 4: AJAN & MCP ORKESTRASYONU                        |
|  25 Model Context Protocol (MCP) Aracı | Claude, Gemini, GPT Otonom Kuantum Ajanı|
|  "Kuantum iş akışlarını doğal dil ve otonom döngülerle yönetme"                 |
+---------------------------------------------------------------------------------+
|               KATMAN 3: DERİN ÖĞRENME & DEKLARATİF BİLİŞSEL API                 |
|  quanta.torch: QuantumLayer | Daleckii-Krein Autograd | Lie Cebiri Barren Analizi|
|  BiomorphicResonantBrain | SWR Replay | ContinuousResonantLayer (Schrödinger)   |
|  search() | optimize() | vqe() | factor() | portfolio_optimize() | resolve()    |
|  "Ne çözülecek?" -- kapı sentezi gerekmeden doğrudan algoritmik çözüm           |
+---------------------------------------------------------------------------------+
|                       KATMAN 2: ALGORİTMİK DEVRE DSL                            |
|  @circuit | 34 Yerleşik Kapı (IBM Heron, Google Sycamore, IonQ yerel paritesi)  |
|  Parametrik Rotasyonlar (RX, RY, RZ, CRX, CRY, CRZ, P, U) | measure() | sweep() |
|  "Devre nasıl kurgulanacak?"                                                    |
+---------------------------------------------------------------------------------+
|             KATMAN 1: 2026 ÇİFT-KANAL HATA TOLERANSI (FTQC MOTORU)              |
|  Track A: Blossom MWPM & Union-Find | ColorCode A2+ILP | Willow 3D Sendrom      |
|  Track B: Gross [[144, 12, 12]] qLDPC | Yerel Normalized Min-Sum BP-OSD         |
|  Non-Clifford: 15-to-1 Bravyi-Kitaev Sihirli Durum Damıtma | Örgü Cerrahisi     |
+---------------------------------------------------------------------------------+
|                 KATMAN 0: FİZİKSEL HESAPLAMA & DONANIM MOTORU                   |
|  DAG Devresi (Kahn) | Sentez (ZYZ, Heron, Cartan/KAK SU(4)) | 127/156Q Rota     |
|  Metal/MLX Sıfır-Kopya GPU | SIMD Clifford ((2n+1)x(2n+1)) | MPS (200+Q)        |
|  GKSL Lindblad & 7-Kanal Kraus Gürültü | Çoklu-Bulut (IBM REST, Cirq, IonQ)    |
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
| `gates.py` | 34 yerleşik kapı (`CRX`, `CRY`, `CRZ` + tam `iSWAP`, `ECR`, `CSWAP`, `RCCX`, `RC3X`) + broadcast |
| `circuit.py` | @circuit dekoratoru, CircuitBuilder |
| `measure.py` | Esnek olcum (tam, kismi) |
| `equivalence.py` | Uniter karsilastirma, sadakat |
| `custom_gate.py` | Kullanici tanimli uniter kapilar |

### dag/ -- Yonlu Dongusuz Graf

| Dosya | Sorumluluk |
|-------|------------|
| `node.py` | InputNode, OpNode, OutputNode |
| `dag_circuit.py` | Topolojik siralama (Kahn), derinlik, paralel katmanlar |

### compiler/ -- Optimizasyon & Üniter Sentez Hattı

| Dosya | Sorumluluk |
|-------|------------|
| `pipeline.py` | CompilerPass Protokolu, zincirleme, istatistikler |
| `synthesis.py` | Evrensel 1Q ZYZ Euler ayrıştırması, IBM Heron `RZ-SX-RZ-SX-RZ` ve 2Q Cartan/KAK $SU(4)$ sentezi |
| `passes/optimize.py` | CancelInverses (H.H=I), MergeRotations |
| `passes/translate.py` | IBM/Google/Quantinuum kapi seti cevirisi |
| `passes/routing.py` | Topoloji bazli SWAP ekleme (`linear`, `ring`, `grid`, 127Q/156Q IBM Heavy-Hex) ve dinamik SWAP yerleşim takibi |

### simulator/ -- Simulasyon Motorlari

| Dosya | Sorumluluk |
|-------|------------|
| `base.py` | `SimulatorBackend` ABC — tüm simülatörler için soyut arayüz |
| `statevector.py` | Yoğun tensor contraction, 27 qubite kadar (tam doğru) |
| `mlx.py` | Apple Silicon Metal GPU sıfır-kopya Birleşik Bellek tensör motoru (30 qubite kadar) |
| `sparse.py` | Sözlük tabanlı seyrek statevector, 50 qubite kadar, O(k) bellek |
| `mps.py` | Matris Çarpım Durumu (SVD), 200+ qubit, mixed-canonical Born örnekleme ve komşu-olmayan SWAP yönlendirme |
| `factory.py` | `create_simulator()` — en iyi arka ucu otomatik seçer |
| `router.py` | Devre-duyarlı yönlendirme (Clifford algılama, qubit sayısı) |
| `density_matrix.py` | 13 qubite kadar karışık durumlar (`np.tensordot` CPTP Kraus kanalları + tam `p/15` 2Q depolarizasyon) |
| `lindblad.py` | Açık sistem Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) ana denklemi çözücüsü |
| `pauli_frame.py` | Faz vektörlü tam $(2n+1)\times(2n+1)$ Aaronson-Gottesman stabilizer/destabilizer tablosu |
| `noise.py` | Kapı-bazlı stokastik yörüngeler + `ReadoutError` (Depolarizing, BitFlip, PhaseFlip, AmplitudeDamping, T2Relaxation, Crosstalk, ReadoutError) |
| `accelerated.py` | JAX-GPU / CuPy otomatik algilama, NumPy fallback |

### layer3/ -- Deklaratif API

| Dosya | Sorumluluk |
|-------|------------|
| `search.py` | Otomatik Grover aramasi |
| `optimize.py` | QAOA optimizasyonu |
| `agent.py` | Coklu ajan karar modelleme |
| `vqe.py` | Variasyonel Kuantum Ozdeger Cozucu |
| `shor.py` | Griffiths-Niu / Beauregard yarı-klasik QPE tam sayı çarpanlara ayırma (`shor.factor`) |
| `qsvm.py` | Cortes-Vapnik Dual QP kuantum çekirdek SVM sınıflandırma |
| `finance.py` | Portfoy optimizasyonu (Markowitz + QAOA) |
| `hamiltonian.py` | Trotter zaman evrimi, molekuler Hamiltonianlar |
| `entity_resolution.py` | QAOA tabanli musteri tekillestime |
| `monte_carlo.py` | Grinko İteratif Kuantum Genlik Kestirimi (IQAE) + Ortak Rastgele Sayılar (CRN) Black-Scholes Greeks |
| `clustering.py` | Buhrman Swap Test kuantum uzaklıkları + k-means kümeleme |
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
| `surface_code.py` | Döndürülmüş Surface Code [[d^2,1,d]], tam $\mathbb{F}_2$ homolojisi ve 3D uzay-zaman sendrom döngüleri |
| `color_code.py` | 2D Üçgensel Color Code, transversal Clifford ve yalnızca-sendrom Kubica-Delfosse $A_2$ örgüsü + ILP dekoderi |
| `decoder.py` | Edmonds Blossom MWPM (tam ağırlıklı mükemmel eşleme) ve gerçek Delfosse-Nickerson `UnionFindDecoder` |
| `qldpc.py` | Gross [[144, 12, 12]] Bivariate Bicycle kodu, Normalized Min-Sum BP-OSD |
| `distillation.py` | 15-to-1 Bravyi-Kitaev sihirli durum damıtma fabrikası, örgü cerrahisi |

### quanta.torch / cognitive/ -- Derin Öğrenme & Biyomorfik Esinli Motor

| Modül / Dosya | Sorumluluk |
|---------------|------------|
| `quanta.torch.QuantumLayer` | PyTorch `nn.Module` katmanı, analitik parameter-shift autograd VJP |
| `quanta.torch.ContinuousResonantLayer` | Sürekli Schrödinger akışı, Daleckii-Krein analitik Fréchet türevleri |
| `quanta.torch.BiomorphicResonantBrain` | Kuantum ve nörobiyoloji esinli diferansiyellenebilir çift hemisferli PyTorch mimarisi |
| `quanta.lie_algebra` | Dinamik Lie cebiri $\dim(\mathfrak{g})$ boyutu ve analitik barren plateau teşhis motoru |
| `quanta.cognitive` | Klasik/PyTorch biyomorfik sezgiseller: SWR bellek pekiştirme, `QuantumZenoAttention`, `QuantumDecisionArbiter` |

### benchmark/ -- Kalite & Kıyaslama Ölçümü

| Dosya | Sorumluluk |
|-------|------------|
| `qasmbench.py` | 10 standart + 3 büyük QASMBench devresi |
| `benchpress_adapter.py` | SDK arası karşılaştırma API'si (Nation et al.) |
| `run_paper_benchmarks.py` | Hakemli yayın için ampirik mikrosaniye kıyaslamaları |

### Destek Modülleri

| Dosya | Sorumluluk |
|-------|------------|
| `gradients.py` | Mari-Bromley-Killoran ayar-değişmez QFIM, 4-terimli Anselmetti 2Q parameter-shift ve $O(G)$ `adjoint_gradient` |
| `runner.py` | 6 aşamalı orkestratör: build > DAG > compile > sim > noise > sample > result |
| `result.py` | Ölçüm sonuçları, olasılıklar, Dirac notasyonu, durum vektörü |
| `visualize.py` | ASCII ve SVG devre diyagramları |
| `visualize_state.py` | Olasılık histogramı, Bloch küresi, faz diyagramı |
| `mcp_server.py` | MCP sunucusu — Otonom AI ajanları için **25 kuantum aracı** (SSE + stdio) |

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

1. **İlk-İlkelerden Bağımsızlık**: Çekirdek devre/derleyici/statevector motorları ağır C++/LLVM derleme zincirleri olmaksızın saf Python ve NumPy ile çalışır (SciPy ve NetworkX Katman-3 çözücülerini ve QEC dekoderlerini destekler).
2. **Apple Silicon Sıfır-Kopya**: Metal Performance Shaders / MLX ile birleşik bellekte CPU-GPU kopyalama gecikmesini sıfırlayan doğrudan GPU tensör hızlandırması (kapı derinliğine ve qubit sayısına göre 58x–404x'e varan hızlanma).
3. **Analitik & Adjoint Hilbert Gradyanları**: Daleckii-Krein Fréchet türevleri ($9.99\times 10^{-16}$), $O(G)$ `adjoint_gradient`, 4-terimli Anselmetti 2Q parameter-shift ve Mari-Bromley-Killoran ayar-değişmez QFIM.
4. **2026 FTQC Çift-Kanal**: Tam $\mathbb{F}_2$ homolojili yüzey kodlarında Edmonds Blossom MWPM ve Delfosse-Nickerson Union-Find, Kubica-Delfosse $A_2$+ILP Color Code ve yüksek dereceli Gross $[[144, 12, 12]]$ qLDPC BP-OSD ile 12 kat donanım tasarrufu.
5. **AI-Native MCP Mimarisi**: **25 adet yerleşik MCP aracı** ile Claude, GPT ve Gemini ajanlarının kuantum optimizasyonu ve denetimini doğrudan yürütebilmesi.
6. **Yanlışlanabilir Bilimsel Titizlik (Falsifiable Empiricism)**: **2.826+ otomatize test** ile üniterlik ve CPTP iz korunum garantisi.

---

## Mimari Künyesi & Yazarlık

- **Baş Mimar**: Abdullah Enes SARI (ORCID: [0000-0002-8827-0587](https://orcid.org/0000-0002-8827-0587))
- **Kurum**: ONMARTECH Kuantum Bilişim İnisiyatifi (`info@onmartech.com`)
- **GitHub Release**: [GitHub Release v1.2.0](https://github.com/ONMARTECH/quanta-sdk/releases/tag/v1.2.0)

