# Quanta SDK — Ozellikler

## Kapi Seti (31 Kapi)

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

## Derleyici Optimizasyonlari

| Gecis | Ne Yapar | Ornek |
|-------|----------|-------|
| CancelInverses | Ters kapilari iptal eder | H·H → (bos), X·X → (bos) |
| MergeRotations | Rotasyonlari birlestirir | RZ(π/4)·RZ(π/4) → RZ(π/2) |
| TranslateToTarget | Hedef donanim kapi setine cevirir | SWAP → 3×CX |

### Qubit Yonlendirme

Topoloji bazli SWAP ekleme:

| Topoloji | Kullanim |
|----------|----------|
| Linear | Iyon tuzagi, superconducting zincirler |
| Ring | Dairesel baglanti |
| Grid | 2D superconducting (IBM, Google) |

## Desteklenen Donanim Kapi Setleri

| Donanim | Kapi Seti |
|---------|-----------|
| IBM Heron | {CX, RZ, SX, X} |
| Google Sycamore | {CZ, RZ, RX, RY} |
| Quantinuum H-Series | {CX, RZ, RY, RX} |

## Simulatorler

| Simulator | Maks Qubit | Ozellikler |
|-----------|-----------|------------|
| Statevector | 27 | Tensor contraction, O(2^n) |
| Pauli Frame | 50 | Stabilizer tablosu (Aaronson-Gottesman), O(n) |
| Density Matrix | 13 | Karisik durumlar, Kraus kanallari |
| Accelerated | 27 | JAX-GPU / CuPy otomatik algilama |

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

### Track A: 2D Topolojik Yüzey Kodları & Edmonds Blossom MWPM
- **Döndürülmüş Yüzey Kodları ($[[d^2, 1, d]]$)**: $d \in \{3, 5, 7\}$ mesafelerinde stabilizer sendrom çıkarımı.
- **Edmonds Blossom MWPM Dekoderi**: Açgözlü (greedy) eşleme sezgisellerinin aksine, kuramsal eşik ($p_{\text{th}} \approx 1\%$) sağlayan tam graf ağırlıklı mükemmel eşleme.
- **Google Willow Uyumlu 3D Uzay-Zaman Döngüleri**: Ölçüm ve kapı hatalarını zaman ekseninde tespit eden 3D sendrom grafı.
- **Color Codes ($[[n, 1, d]]$)**: Transversal Clifford kapı seti ve restriction dekoderi.
- **Standart Kodlar**: Steane $[[7,1,3]]$, 3-qubit Bit-Flip ve Phase-Flip kodları.

### Track B: Yüksek Dereceli qLDPC Kodları & Yerel BP-OSD
- **Kanonik Gross $[[144, 12, 12]]$ Bivariate Bicycle Kodu**: 2D yüzey kodlarına kıyasla aynı mantıksal koruma için **$12\times$ daha az fiziksel qubit**.
- **Yerel Normalized Min-Sum BP-OSD-0 Dekoderi**: İnanç yayılımı (Belief Propagation) ve sıralı istatistik kod çözme (OSD-0) kombinasyonu ile milisaniyelik ($1.54\text{ ms}$) yüksek hızlı sendrom çözümü.
- **Non-Clifford Evrensellik**:
  - **15-to-1 Bravyi-Kitaev Sihirli Durum Damıtma**: Gürültülü $|T\rangle$ durumlarından saflaştırılmış mantıksal $|T\rangle_L$ üretimi ($\epsilon_{\text{out}} \le 35 p^3$).
  - **Örgü Cerrahisi (Lattice Surgery)**: Mantıksal qubitler arası etkileşim ve CNOT birleştirme/ayırma protokolleri.

| Kod / Mimari | Notasyon | Qubit Tasarrufu | Dekoder |
|--------------|----------|-----------------|---------|
| Gross Bivariate Bicycle | $[[144, 12, 12]]$ | **12× tasarruf** (12 mantıksal qubit) | Normalized Min-Sum BP-OSD-0 |
| Rotated Surface Code | $[[d^2, 1, d]]$ | Referans 2D | Edmonds Blossom MWPM |
| Color Code | $[[n, 1, d]]$ | Transversal Clifford | Restriction Decoder |
| Steane Code | $[[7, 1, 3]]$ | Analitik benchmark | Lookup / Syndrome |
| 15-to-1 BK Distillation | $|T\rangle$ Factory | $\epsilon_{\text{out}} \le 35 p^3$ | Parite Projeksiyonu |

## Algoritmalar (Katman 3)

| Algoritma | Fonksiyon | Aciklama |
|-----------|----------|----------|
| Grover | `search()` | Yapisiz aramada karesel hizlanma |
| QAOA | `optimize()` | Kombinatorik optimizasyon |
| VQE | `vqe()` | Molekuler enerji icin variasyonel ozvektor |
| Shor | `factor()` | Periyot bulma ile tam sayi carpanlara ayirma |
| QSVM | `qsvm_classify()` | Kuantum cekirdek SVM siniflandirma |
| Portfoy | `portfolio_optimize()` | Finansal portfoy optimizasyonu |
| Hamiltonian | `evolve()` | Trotter zaman evrimi |
| Tekillestime | `resolve()` | QAOA tabanli musteri tekillestime |
| Coklu-Ajan | `MultiAgentSystem` | Kuantum karar modelleme |
| Monte Carlo | `monte_carlo_price()` | Opsiyon fiyatlama icin genlik kestirimi |
| Kumeleme | `cluster_data()` | Swap-test kuantum uzaklik + k-means |
| QML Siniflandirici | `QuantumClassifier` | Variasyonel kuantum siniflandirma |

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

## MCP Server (AI Entegrasyonu — 23 Araç)

Quanta SDK, yerel ve uzak **23 MCP (Model Context Protocol)** aracı içerir. Gemini, Claude ve GPT ajanları tam yetkiyle kuantum devresi tasarlayabilir, gürültü profili çıkarabilir, donanım maliyeti hesaplayabilir ve kuantum algoritmalarını çalıştırabilir.

| Araç Grubu | Yetenekler |
|------------|------------|
| Temel Devre | `run_circuit`, `create_bell_state`, `list_gates`, `inspect_circuit` |
| Algoritmalar | `grover_search`, `shor_factor`, `vqe_ground_state`, `quantum_clustering` |
| Gürültü & Hata | `simulate_noise` (7 kanal), `estimate_fault_tolerant_cost` (Google Willow / IBM Starling) |
| Optimizasyon & Çözüm | `qubo_solve`, `portfolio_optimization`, `entity_resolution_match` |

---

## PyTorch & Biyomorfik Kuantum Motoru (v1.2.0-production — `quanta.torch`)

### 1. Differentiable `QuantumLayer`
- PyTorch `nn.Module` tam uyumluluğu.
- Analitik **Parameter-Shift Kuralı** ile kesin VJP ve autograd gradyanları.
- Donanım verimli ansatz şablonları: `hardware_efficient`, `strong_entangling`, `reuploading`, `real_amplitudes`.
- CPU, CUDA ve Apple Silicon MPS (Metal) desteği.

### 2. Sürekli Kuantum Rezonansı (`ContinuousResonator`)
- Hamiltonyen evrimi: $U(t) = e^{-i H(x, \theta) t}$.
- **Daleckii-Krein Fréchet matris üssü gradyanları**: Padé sapmalarını ($>1.3\times 10^{-6}$) sıfırlayan, $9.99\times 10^{-16}$ makine hassasiyetinde analitik türev.
- **Dinamik Lie Cebiri & Barren Plateau Teşhisi**: $\mathfrak{g} = \langle i H_k \rangle_{\text{Lie}}$ boyutu üzerinden ansatz ekspresifliği ve gradyan sönümleme risklerinin önceden tespiti.
- **Ehrenfest teorem zaman türevleri** ve Lindblad faz difüzyon süperoperatörleri.

### 3. Biyomorfik Kuantum Rezonans Beyni (`BiomorphicResonantBrain`)
- **Çift Hemisfer Mimarisi**: Sol hemisfer (analitik/mantıksal) ve sağ hemisfer (sezgisel/örrüntü) rezonansı.
- **4 Nöromodülatör Dinamiği**:
  - *Dopamin ($DA$)*: Kazanç ve motivasyon modülasyonu.
  - *Asetilkolin ($ACh$)*: Dikkat ve plastisite faktörü.
  - *Serotonin ($5\text{-}HT$)*: Sabır ve risk toleransı.
  - *Noradrenalin ($NE$)*: Uyarılma ve acil durum tepkisi.
- **Serebral Oksijenasyon ($sO_2$)**: Metabolik ve enerji kısıt optimizasyonu.
- **REM Uykusu Konsolidasyonu**: Felaket unutmasını (catastrophic forgetting) önleyen kuantum bellek pekiştirmesi.
- **NoisyHippocampalBuffer**: Lindblad difüzyonlu ve SWR (Sharp-Wave Ripple) bellek replay tamponu.
- **DialecticalSynthesizer**: Kuantum durumlarında Tez-Antitez çatışmasını sentezleyen diyalektik motor.
- **CSFBiophysicalShield**: Çevresel gürültüye karşı koruyucu biyofiziksel kuantum faz zırhı.

---

## Apple Silicon Metal / MLX Hızlandırma (v1.2.0)

- **Sıfır-Kopya (Zero-Copy) Birleşik Bellek**: M-serisi çiplerde CPU-GPU bellek kopyalama yükünü ortadan kaldıran Metal/MLX motoru ile **52.09× tepe hızlanma**.
- **SIMD Vektörize Clifford Motoru**: Aaronson-Gottesman ikili tablosu ile **>3.13 milyon kapı/saniye** stabilizatör yürütme performansı.
- **MPS (Matrix Product States)**: 250-qubit GHZ durumunu yalnızca 3.42 milisaniyede hazırlayan düşük dolaşıklıklı tensör ağı simülatörü.
- **Büyük Ölçekli Durum Vektörü**: M-serisi birleşik bellekte 27+ qubitlik durum vektörü tensör kasılmaları.

---

## Çoklu Bulut Kuantum Donanımı

- **IonQ Cloud**: REST API v0.3 protokolü, 29-qubit donanım emülasyonu ve telemetrisi.
- **IBM Quantum**: 156 qubit Heron r3 işlemciler (`ibm_torino`, `ibm_fez`), ISA transpilasyonu ve IAM token entegrasyonu.
- **Google Cirq**: Sycamore yerli kapı seti simülasyonu ve Google Colab GPU uyumluluğu.

---

## Doğrulanabilirlik & Test Kapsamı

- **2.076 Adet Otomatize Test**: Regresyonsuz, sıfır mock, falsifiable test altyapısı.
- **Üniterlik Garantisi**: Makine hassasiyetinde $\|U^\dagger U - I\| < 10^{-14}$.
- **CPTP İz Korunumu**: $|\text{Tr}(\rho) - 1.0| < 10^{-12}$.

---

## Dağıtım

| Hedef | Yöntem | Kullanım |
|-------|--------|----------|
| Yerel | `pip install quanta-sdk` | Geliştirme, araştırma |
| PyTorch / AI | `pip install "quanta-sdk[torch]"` | Derin öğrenme, hibrit QNN |
| Apple Metal | `pip install "quanta-sdk[metal]"` | Apple Silicon GPU hızlandırma |
| Claude / Gemini / GPT | MCP Sunucu Entegrasyonu (`fastmcp`) | 23 araçlı otonom AI ajanları |
| Cloud Run / Docker | Dockerfile.mcp + SSE | Sürekli aktif uzak kuantum servisi |
