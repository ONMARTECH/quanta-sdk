# Quanta SDK — Ozellikler

## Kapı Seti (34 Kapı)

| Kapi | Qubit | Aciklama |
|------|-------|----------|
| H | 1 | Hadamard — superpozisyon olusturur |
| X | 1 | Pauli-X — bit cevirme (NOT) |
| Y | 1 | Pauli-Y — bit + faz cevirme |
| Z | 1 | Pauli-Z — faz cevirme |
| S | 1 | S kapisi — π/2 faz |
| T | 1 | T kapisi — π/4 faz |
| CX | 2 | CNOT — kontrollü NOT |
| CZ | 2 | Kontrollü-Z — kontrollü faz |
| CY | 2 | Kontrollü-Y |
| SWAP | 2 | Qubit degisimi |
| CCX | 3 | Toffoli — cift kontrollü NOT |
| RX(θ) | 1 | X-ekseni rotasyonu |
| RY(θ) | 1 | Y-ekseni rotasyonu |
| RZ(θ) | 1 | Z-ekseni rotasyonu |
| CRX(θ) | 2 | Kontrollü X-ekseni rotasyonu |
| CRY(θ) | 2 | Kontrollü Y-ekseni rotasyonu |
| CRZ(θ) | 2 | Kontrollü Z-ekseni rotasyonu |
| P(θ) | 1 | Faz kapisi |
| U(θ,φ,λ) | 1 | Genel tek-qubit uniter |
| I | 1 | Birim kapi |
| SDG | 1 | S-dagger (−π/2 faz) |
| TDG | 1 | T-dagger (−π/4 faz) |
| SX | 1 | X karekoku |
| SXdg | 1 | SX-dagger |
| RXX(θ) | 2 | XX rotasyonu (2-qubit) |
| RZZ(θ) | 2 | ZZ rotasyonu (2-qubit) |
| RCCX | 3 | Goreli-fazli CCX |
| RC3X | 4 | Goreli-fazli C3X |
| ECR | 2 | Ekolu capraz-rezonans (IBM Heron yerli) |
| iSWAP | 2 | Imajiner SWAP (Google Sycamore yerli) |
| CSWAP | 3 | Kontrollu-SWAP (Fredkin) |
| CH | 2 | Kontrollu-Hadamard |
| CP(θ) | 2 | Kontrollu-Faz |
| MS(θ) | 2 | Mølmer-Sørensen (IonQ tuzsuzlu-iyon yerli) |

### Ozel Kapilar

```python
from quanta import custom_gate
import numpy as np

custom_gate("SqrtX", np.array([[0.5+0.5j, 0.5-0.5j],
                                [0.5-0.5j, 0.5+0.5j]]))
```

### Broadcast Destegi

```python
H(q)        # Tum qubitlere H uygula
H(q[0])     # Sadece q[0]'a uygula
CX(q[0], q[1])  # Iki-qubit kapisi
```

## Derleyici Optimizasyonları & Üniter Sentez

| Geçiş / Modül | Ne Yapar | Örnek |
|---------------|----------|-------|
| CancelInverses | Ters kapıları iptal eder | H·H → (boş), X·X → (boş) |
| MergeRotations | Rotasyonları birleştirir | RZ(π/4)·RZ(π/4) → RZ(π/2) |
| TranslateToTarget | Hedef donanım kapı setine çevirir | SWAP → 3×CX |
| `synthesis.py` | Evrensel 1Q ZYZ, IBM Heron `RZ-SX-RZ-SX-RZ` ve 2Q Cartan/KAK $SU(4)$ sentezi | Keyfi $4\times 4$ üniter → $\le 3$ CNOT |

### Qubit Yönlendirme

Dinamik SWAP yerleşim takipli topoloji bazlı yönlendirme:

| Topoloji | Kullanım |
|----------|----------|
| Linear | İyon tuzağı, süperiletken zincirler |
| Ring | Dairesel bağlantı |
| Grid | 2D süperiletken (Google Sycamore/Willow) |
| Heavy-Hex (127Q / 156Q) | IBM Eagle / Heron r2 & r3 süperiletken işlemciler |

## Desteklenen Donanım Kapı Setleri

| Donanım | Kapı Seti |
|---------|-----------|
| IBM Heron | {CX / ECR, RZ, SX, X} |
| Google Sycamore | {CZ, RZ, RX, RY} |
| Quantinuum H-Series | {CX, RZ, RY, RX} |

## Simülatörler

| Simülatör | Maks Qubit | Özellikler |
|-----------|-----------|------------|
| `StateVectorSimulator` | 27 | Yoğun tensör kasılması, $O(2^n)$ |
| `MLXSimulator` | 30 | Apple Silicon Metal GPU sıfır-kopya Birleşik Bellek |
| `SparseSimulator` | 50 | Sözlük tabanlı seyrek durum vektörü, $O(k)$ aktif baz durumu |
| `MPSSimulator` | 200+ | Mixed-canonical SVD Matris Çarpım Durumu, Born örnekleme & komşu-olmayan SWAP yönlendirme |
| `PauliFrame` | 50+ | Faz vektörlü tam $(2n+1)\times(2n+1)$ Aaronson-Gottesman stabilizer/destabilizer tablosu |
| `DensityMatrixSimulator` | 13 | `np.tensordot` CPTP Kraus kanalları + tam `p/15` 2Q depolarizasyon ile karışık durumlar |
| `LindbladMasterEquation` | 12 | Açık sistem Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) ana denklemi çözücüsü |
| `AcceleratedSimulator` | 27 | JAX-GPU / CuPy otomatik algılama (NumPy fallback) |

### Gurultu Entegrasyonu

Gurultu, calistirma hattinin birinci sinif vatandasidir:

```python
from quanta import run
from quanta.simulator.noise import NoiseModel, Depolarizing

result = run(bell, shots=1024, noise=NoiseModel().add(Depolarizing(0.01)))
```

## Gurultu Modelleri

| Kanal | Aciklama | Parametre | Donanim Ref |
|-------|----------|-----------|-------------|
| Depolarizing | Rastgele Pauli hatasi | p ∈ [0,1] | — |
| BitFlip | |0⟩↔|1⟩ cevirme | p ∈ [0,1] | — |
| PhaseFlip | Faz hatasi (Z) | p ∈ [0,1] | — |
| AmplitudeDamping | Enerji kaybi (T1) | γ ∈ [0,1] | IBM: 100-300μs |
| T2Relaxation | Saf defazlama (T2) | γ ∈ [0,1] | IBM: 100-200μs |
| Crosstalk | Komsu qubit ZZ etkilesimi | p ∈ [0,1] | ~%0.1-1 / kapi |
| ReadoutError | Olcum bit-cevirme | p01, p10 | IBM: %0.5-2 |

## 2026 Çift-Kanal Hata Toleransı Motoru (Dual-Track FTQC)

Quanta SDK, 2026 yılı hata toleranslı kuantum hesaplama (FTQC) hedeflerine yönelik olarak çift-kanallı (Dual-Track) bir QEC mimarisi sunar:

### Track A: 2D Topolojik Yüzey Kodları, Edmonds Blossom MWPM & Union-Find
- **Döndürülmüş Yüzey Kodları ($[[d^2, 1, d]]$)**: $d \in \{3, 5, 7\}$ mesafelerinde tam $\mathbb{F}_2$ homolojili stabilizer sendrom çıkarımı.
- **Edmonds Blossom MWPM & Delfosse-Nickerson `UnionFindDecoder`**: Tam graf ağırlıklı mükemmel eşleme ve doğrusal-benzeri zamanlı Union-Find küme büyütme/soyma kod çözümü.
- **Google Willow Uyumlu 3D Uzay-Zaman Döngüleri**: Ölçüm ve kapı hatalarını zaman ekseninde tespit eden 3D sendrom grafı.
- **Color Codes ($[[n, 1, d]]$)**: Transversal Clifford kapı seti (`H`, `S`, `CNOT`) ve yalnızca-sendrom Kubica-Delfosse $A_2$ örgüsü + tamsayılı doğrusal programlama (ILP) dekoderi.
- **Standart Kodlar**: Steane $[[7,1,3]]$, 3-qubit Bit-Flip ve Phase-Flip kodları.

### Track B: Yüksek Dereceli qLDPC Kodları & Yerel BP-OSD
- **Kanonik Gross $[[144, 12, 12]]$ Bivariate Bicycle Kodu**: 2D yüzey kodlarına kıyasla aynı mantıksal koruma için **$12\times$ daha az fiziksel qubit**.
- **Yerel Normalized Min-Sum BP-OSD Dekoderi**: İnanç yayılımı (Belief Propagation) ve sıralı istatistik kod çözme (`BPOSDDecoder`).
- **Non-Clifford Evrensellik**:
  - **15-to-1 Bravyi-Kitaev Sihirli Durum Damıtma**: Gürültülü $|T\rangle$ durumlarından saflaştırılmış mantıksal $|T\rangle_L$ üretimi ($\epsilon_{\text{out}} \le 35 p^3$).
  - **Örgü Cerrahisi (Lattice Surgery)**: Mantıksal qubitler arası etkileşim ve CNOT birleştirme/ayırma protokolleri.

| Kod / Mimari | Notasyon | Qubit Tasarrufu | Dekoder |
|--------------|----------|-----------------|---------|
| Gross Bivariate Bicycle | $[[144, 12, 12]]$ | **12× tasarruf** (12 mantıksal qubit) | Normalized Min-Sum BP-OSD |
| Rotated Surface Code | $[[d^2, 1, d]]$ | Referans 2D ($\mathbb{F}_2$ homolojisi) | Edmonds Blossom MWPM & `UnionFindDecoder` |
| Color Code | $[[n, 1, d]]$ | Transversal Clifford | Kubica-Delfosse $A_2$ Örgüsü + ILP |
| Steane Code | $[[7, 1, 3]]$ | Analitik benchmark | Lookup / Syndrome |
| 15-to-1 BK Distillation | $|T\rangle$ Factory | $\epsilon_{\text{out}} \le 35 p^3$ | Parite Projeksiyonu |

## Algoritmalar (Katman 3)

| Algoritma | Fonksiyon | Açıklama |
|-----------|----------|----------|
| Grover | `search()` | Yapısız aramada karesel hızlanma |
| QAOA | `optimize()` | Kombinatorik optimizasyon |
| VQE | `vqe()` | Moleküler enerji için varyasyonel özvektör |
| Shor | `factor()` | Griffiths-Niu / Beauregard yarı-klasik QPE tam sayı çarpanlara ayırma |
| QSVM | `qsvm_classify()` | Cortes-Vapnik Dual QP kuantum çekirdek SVM sınıflandırma |
| Portföy | `portfolio_optimize()` | Markowitz + QAOA finansal portföy optimizasyonu |
| Hamiltonian | `evolve()` | Trotter zaman evrimi |
| Tekilleştirme | `resolve()` | QAOA tabanlı müşteri tekilleştirme |
| Çoklu-Ajan | `MultiAgentSystem` | Kuantum karar modelleme |
| Monte Carlo & Greeks | `quantum_monte_carlo()`, `compute_greeks()` | Grinko IQAE opsiyon fiyatlama + Ortak Rastgele Sayılar (CRN) Black-Scholes Greeks |
| Kümeleme | `quantum_cluster()` | Buhrman Swap Test kuantum uzaklık + k-means |
| QML Sınıflandırıcı | `QuantumClassifier` | Varyasyonel kuantum sınıflandırma |

## Analitik Gradyanlar (`quanta.gradients`)

- **Parameter-Shift Kuralı (`parameter_shift`)**: 1-qubit Pauli üreteçleri için 2-terimli, 2-qubit dolaşıklık kapıları (`CRX`, `CRY`, `CRZ`, `CP`, `RXX`, `RZZ`, `MS`) için **4-terimli Anselmetti shift** kuralı.
- **Adjoint Türev (`adjoint_gradient`)**: Parametre sayısından bağımsız tam **$O(G)$ ters durum vektörü süpürmesi**.
- **Kuantum Doğal Gradyan (`natural_gradient`)**: Fubini-Study önkoşullu optimizasyon için **Mari-Bromley-Killoran ayar-değişmez Kuantum Fisher Bilgi Matrisi (QFIM)**.

## QASM Destegi

| Yon | Versiyon | Aciklama |
|-----|---------|----------|
| Cikti | QASM 3.0 | Devre → OpenQASM dizesi |
| Girdi | QASM 2.0/3.0 | OpenQASM dizesi → DAG |

## Benchmark Altyapisi

| Arac | Aciklama |
|------|----------|
| QASMBench | 10 standart + 3 buyuk (20-24 qubit) devre |
| Benchpress Adapter | SDK arasi karsilastirma API'si |
| Turnusol Testi | 8 testlik kalite testi |

## Parametre Taramasi

```python
from quanta import sweep

results = sweep(my_circuit, params={"theta": [0, 0.5, 1.0, 1.5]})
for r in results:
    print(r.summary())
```

## Gorsellestirme

- Olasilik histogrami: `print(result)`
- Dirac notasyonu: `result.dirac_notation()`
- Durum vektoru gosterimi: `show_statevector(sv, n)`

## MCP Server (AI Entegrasyonu — 25 Araç)

Quanta SDK, `quanta/mcp_server.py` içinde yerel ve uzak **25 MCP (Model Context Protocol)** aracı içerir. Gemini, Claude ve GPT ajanları tam yetkiyle kuantum devresi tasarlayabilir, gürültü profili çıkarabilir, donanım maliyeti hesaplayabilir ve kuantum algoritmalarını çalıştırabilir.

| Araç Grubu | Tam `@mcp.tool` İsimleri |
|------------|--------------------------|
| Devre, Derleme & Simülasyon (8) | `run_circuit`, `mps_circuit_simulate`, `create_bell_state`, `list_gates`, `draw_circuit`, `explain_result`, `optimize_circuit`, `transpile_for_target` |
| Algoritmalar, Optimizasyon & QML (5) | `grover_search`, `shor_factor`, `qaoa_optimize`, `cluster_data`, `qml_classify` |
| Kantitatif Finans (3) | `monte_carlo_price`, `option_greeks`, `portfolio_optimize` |
| Gürültü, QEC & Hata Toleransı (5) | `simulate_noise`, `surface_code_simulate`, `compare_decoders`, `qec_diagnose`, `estimate_fault_tolerant_cost` |
| Bulut Donanım & Bilişsel Değerlendirme (4) | `run_on_ibm`, `ibm_backends`, `ibm_job_result`, `quanta_reasoning_eval` |

---

## PyTorch & Biyomorfik Esinli Kuantum Motoru (v1.2.0-production — `quanta.torch`)

### 1. Differentiable `QuantumLayer`
- PyTorch `nn.Module` tam uyumluluğu.
- Analitik **Parameter-Shift Kuralı** (4-terimli Anselmetti 2Q shift dahil) ve $O(G)$ `adjoint_gradient` ile kesin VJP ve autograd gradyanları.
- Donanım verimli ansatz şablonları: `hardware_efficient`, `strong_entangling`, `reuploading`, `real_amplitudes`.
- CPU, CUDA ve Apple Silicon MPS (Metal) desteği.

### 2. Sürekli Kuantum Rezonansı (`ContinuousResonantLayer`)
- Hamiltonyen evrimi: $U(t) = e^{-i H(x, \theta) t}$.
- **Daleckii-Krein Fréchet matris üssü gradyanları**: Padé sapmalarını ($>1.3\times 10^{-6}$) sıfırlayan, $9.99\times 10^{-16}$ makine hassasiyetinde analitik türev.
- **Dinamik Lie Cebiri & Barren Plateau Teşhisi**: $\mathfrak{g} = \langle i H_k \rangle_{\text{Lie}}$ boyutu üzerinden ansatz ekspresifliği ve gradyan sönümleme risklerinin önceden tespiti.
- **Ehrenfest teorem zaman türevleri** ve Lindblad faz difüzyon süperoperatörleri.

### 3. Biyomorfik Kuantum Rezonans Beyni (`BiomorphicResonantBrain`)
> **Bilimsel Sınır Notu**: `quanta.torch.brain` ve `quanta.cognitive` altındaki tüm modüller **kuantum ve nörobiyoloji esinli diferansiyellenebilir PyTorch / klasik sezgisel mimarilerdir**; biyolojik dokuda fiziksel kuantum donanımı iddiası taşımazlar.

- **Çift Hemisfer Mimarisi**: Sol hemisfer (yapısal/mantıksal) ve sağ hemisfer (çağrışımsal/örüntü) diferansiyellenebilir Hamiltonyen etkileşimi.
- **4 Nöromodülatör Dinamiği**:
  - *Dopamin ($DA$)*: Kazanç ve motivasyon modülasyonu.
  - *Asetilkolin ($ACh$)*: Dikkat ve plastisite faktörü.
  - *Serotonin ($5\text{-}HT$)*: Sabır ve risk toleransı.
  - *Noradrenalin ($NE$)*: Uyarılma ve yenilik odaklı keşif tepkisi.
- **Serebral Oksijenasyon ($sO_2$)**: Metabolik ve enerji kısıt optimizasyonu.
- **`QuantumREMSleep`**: Felaket unutmasını (catastrophic forgetting) azaltan çoklu zaman ölçekli Hamiltonyen faz-dağılımı pekiştirmesi.
- **`NoisyHippocampalBuffer`**: Lindblad difüzyonlu ve SWR (Sharp-Wave Ripple) bellek replay tamponu.
- **`QuantumZenoAttention` & `QuantumDecisionArbiter`**: Kuantum Zeno esinli sık-projeksiyon dikkat kapılaması (`quanta.torch.brain`) ve klasik sezgisel karar hakemi (`quanta.cognitive`).
- **`CSFShieldedEnvironment` & `CSFShieldedResonantLayer`**: Dielektrik perdeleme ve viskozite analojileriyle parametrelenen diferansiyellenebilir PyTorch açık-sistem Lindblad faz sönümleme katmanı.

---

## Apple Silicon Metal / MLX Hızlandırma (v1.2.0)

- **Sıfır-Kopya (Zero-Copy) Birleşik Bellek**: M-serisi çiplerde CPU-GPU bellek kopyalama yükünü ortadan kaldıran Metal/MLX motoru ile kapı derinliğine ve qubit sayısına göre **58x–404x'e varan hızlanma**.
- **SIMD Vektörize Clifford Motoru**: Tam $(2n+1)\times(2n+1)$ Aaronson-Gottesman ikili tablosu ile **>3.13 milyon kapı/saniye** stabilizatör yürütme performansı.
- **MPS (Matrix Product States)**: Mixed-canonical Born örnekleme ve komşu-olmayan SWAP yönlendirmeli 200+ qubit tensör ağı simülatörü (250-qubit GHZ durumunda 3.42 ms).
- **Büyük Ölçekli Durum Vektörü**: M-serisi birleşik bellekte 30 qubite kadar durum vektörü tensör kasılmaları.

---

## Çoklu Bulut Kuantum Donanımı

- **IonQ Cloud**: REST API v0.3 protokolü, 29-qubit simülatör/donanım emülasyonu ve telemetrisi.
- **IBM Quantum**: 127Q/156Q Eagle ve Heron işlemciler (`ibm_torino`, `ibm_fez`), yerel Heavy-Hex ISA transpilasyonu ve IAM token entegrasyonu.
- **Google Cirq**: Sycamore yerli kapı seti simülasyonu ve Google Colab GPU uyumluluğu.

---

## Doğrulanabilirlik & Test Kapsamı

- **2.826+ Otomatize Test**: Regresyonsuz, sıfır mock, yanlışlanabilir test altyapısı.
- **Üniterlik Garantisi**: Tüm 34 kapıda ve Cartan/KAK sentezlenen üniterlerde makine hassasiyetinde $\|U^\dagger U - I\| < 10^{-14}$.
- **CPTP İz Korunumu**: $|\text{Tr}(\rho) - 1.0| < 10^{-12}$.

---

## Dağıtım

| Hedef | Yöntem | Kullanım |
|-------|--------|----------|
| Yerel | `pip install quanta-sdk` | Geliştirme, araştırma |
| PyTorch / AI | `pip install "quanta-sdk[torch]"` | Derin öğrenme, hibrit QNN |
| Apple Metal | `pip install "quanta-sdk[metal]"` | Apple Silicon GPU hızlandırma |
| Claude / Gemini / GPT | MCP Sunucu Entegrasyonu (`fastmcp`) | 25 araçlı otonom AI ajanları |
| Cloud Run / Docker | Dockerfile.mcp + SSE | Sürekli aktif uzak kuantum servisi |
