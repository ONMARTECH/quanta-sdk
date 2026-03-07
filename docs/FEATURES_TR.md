# Quanta SDK — Ozellikler

## Kapi Seti (17 Kapi)

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
| U3(θ,φ,λ) | 1 | Genel tek-qubit uniter |
| ISWAP | 2 | Imajiner SWAP |
| SX | 1 | X karekoku |

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
| Density Matrix | 13 | Karisik durumlar, Kraus kanallari |
| Accelerated | 27 | JAX-GPU / CuPy otomatik algilama |

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
| BitFlip | [[3,1,1]] | 1 bit-flip |
| PhaseFlip | [[3,1,1]] | 1 faz-flip |
| Steane | [[7,1,3]] | 1 keyfi tek-qubit hatasi |
| Surface Code | [[d²,1,d]] | ⌊(d-1)/2⌋ hata, esik ~%1 |

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

## MCP Server (AI Entegrasyonu)

Quanta SDK, MCP (Model Context Protocol) sunucusu olarak calisabilir.
Claude gibi AI asistanlar dogrudan kuantum simulasyonu yapabilir.

| Arac | Aciklama |
|------|----------|
| `run_circuit` | Serbest kuantum devresi calistirma |
| `create_bell_state` | Hizli dolanıklık gosterimi |
| `grover_search` | Grover arama algoritmasi |
| `shor_factor` | Shor tam sayi carpanlarina ayirma |
| `simulate_noise` | Gurultulu simulasyon (7 kanal) |
| `list_gates` | Kapi referansi |
| `explain_result` | Olcum sonuclarini yorumlama |

```bash
# Yerel (Claude Desktop)
fastmcp install quanta/mcp_server.py --name "Quanta Quantum SDK"

# Uzak (Cloud Run)
python -m quanta.mcp_server --transport sse --port 8080
```

## Dagitim

| Hedef | Yontem | Kullanim |
|-------|--------|----------|
| Yerel | `pip install quanta-sdk` | Gelistirme, arastirma |
| Claude Desktop | `fastmcp install` | AI destekli simulasyon |
| Cloud Run | Dockerfile.mcp + CI/CD | Surekli aktif MCP sunucu |
| Lambda/Functions | Hafif paket | Sunucusuz hesaplama |
| CI/CD Pipeline | `pip install quanta-sdk` | Otomatik KH testi |

**Hafiflik avantaji**: Saf Python + NumPy. Agir bagimliliksiz.
Sunucusuz (Lambda, Cloud Functions), edge computing ve
CI/CD pipeline icine gomulme icin ideal.
