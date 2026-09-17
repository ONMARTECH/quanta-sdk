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

## Hata Duzeltme Kodlari

| Kod | Notasyon | Duzeltilen Hatalar |
|-----|----------|-------------------|
| BitFlip | [[3,1,3]] | 1 bit-flip |
| PhaseFlip | [[3,1,3]] | 1 faz-flip |
| Steane | [[7,1,3]] | 1 keyfi tek-qubit hatasi |
| Surface Code | [[d²,1,d]] | ⌊(d-1)/2⌋ hata, stabilizer sendromu |
| Color Code | [[n,1,d]] | Transversal Clifford kapilari, restriction decoder |

### QEC Kod Cozuculer

| Cozucu | Karmasiklik | Aciklama |
|--------|-----------|----------|
| MWPM | O(n³) | Gozucu minimum agirlik mukemmel esleme |
| Union-Find | O(n·α(n)) | Yaklasik dogrusal kume tabanli kod cozme |

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

## PyTorch & Biyomorfik Kuantum Motoru (v1.1.0 — `quanta.torch`)

### 1. Differentiable `QuantumLayer`
- PyTorch `nn.Module` tam uyumluluğu.
- Analitik **Parameter-Shift Kuralı** ile kesin VJP ve autograd gradyanları.
- Donanım verimli ansatz şablonları: `hardware_efficient`, `strong_entangling`, `reuploading`, `real_amplitudes`.
- CPU, CUDA ve Apple Silicon MPS (Metal) desteği.

### 2. Sürekli Kuantum Rezonansı (`ContinuousResonator`)
- Hamiltonyen evrimi: $U(t) = e^{-i H(x, \theta) t}$.
- **Daleckii-Krein Fréchet matris üssü gradyanları** ve **Ehrenfest teorem zaman türevleri**.
- Lindblad faz difüzyon süperoperatörleri.

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

## Apple Silicon Metal / MLX Hızlandırma (v1.0.0)

- **404x Hızlanma**: 26 qubitlik devrelerde M-serisi çiplerde CPU'ya kıyasla 404 kat daha hızlı simülasyon.
- **Unified Memory**: M5 Pro 48 GB donanımda 30+ qubitlik durum vektörü tensör kasılmaları.
- **Otomatik Yönlendirme**: macOS ARM64 tespit edildiğinde en yüksek öncelikli hızlandırıcı olarak devreye girer.

---

## Çoklu Bulut Kuantum Donanımı

- **IonQ Cloud**: REST API v0.3 protokolü, 29-qubit donanım emülasyonu ve telemetrisi.
- **IBM Quantum**: 156 qubit Heron r3 işlemciler (`ibm_torino`, `ibm_fez`), ISA transpilasyonu ve IAM token entegrasyonu.
- **Google Cirq**: Sycamore yerli kapı seti simülasyonu ve Google Colab GPU uyumluluğu.

---

## Dağıtım

| Hedef | Yöntem | Kullanım |
|-------|--------|----------|
| Yerel | `pip install quanta-sdk` | Geliştirme, araştırma |
| PyTorch / AI | `pip install "quanta-sdk[torch]"` | Derin öğrenme, hibrit QNN |
| Apple Metal | `pip install "quanta-sdk[metal]"` | Apple Silicon GPU hızlandırma |
| Claude / Gemini | MCP Sunucu Entegrasyonu | Otonom AI ajanları |
| Cloud Run / Docker | Dockerfile.mcp + SSE | Sürekli aktif uzak kuantum servisi |
