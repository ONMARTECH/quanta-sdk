# Quanta SDK — Özellikler

## Kapı Seti (14 + 3 Parametrik)

| Kapı | Qubit | Açıklama | Matris Boyutu |
|------|-------|----------|---------------|
| H | 1 | Hadamard — süperpozisyon oluşturur | 2×2 |
| X | 1 | Pauli-X — bit çevirme (NOT) | 2×2 |
| Y | 1 | Pauli-Y — bit + faz çevirme | 2×2 |
| Z | 1 | Pauli-Z — faz çevirme | 2×2 |
| S | 1 | S kapısı — π/2 faz | 2×2 |
| T | 1 | T kapısı — π/4 faz | 2×2 |
| CX | 2 | CNOT — kontrollü NOT | 4×4 |
| CZ | 2 | Kontrollü-Z — kontrollü faz | 4×4 |
| CY | 2 | Kontrollü-Y | 4×4 |
| SWAP | 2 | Qubit değiş-tokuş | 4×4 |
| CCX | 3 | Toffoli — çift kontrollü NOT | 8×8 |
| RX(θ) | 1 | X ekseni rotasyonu | 2×2 |
| RY(θ) | 1 | Y ekseni rotasyonu | 2×2 |
| RZ(θ) | 1 | Z ekseni rotasyonu | 2×2 |

## Broadcast Desteği

```python
H(q)        # Tüm qubit'lere H uygula
H(q[0])     # Sadece q[0]'a
CX(q[0], q[1])  # İki qubit arası
```

## Compiler Optimizasyonları

| Pass | Ne Yapar | Örnek |
|------|----------|-------|
| CancelInverses | Ters kapıları iptal eder | H·H → (boş), X·X → (boş) |
| MergeRotations | Rotasyonları birleştirir | RZ(π/4)·RZ(π/4) → RZ(π/2) |
| TranslateToTarget | Hedef donanım kapı setine çevirir | SWAP → 3×CX |

## Desteklenen Donanım Gate Setleri

| Donanım | Gate Set |
|---------|----------|
| IBM Heron | {CX, RZ, SX, X} |
| Google Sycamore | {CZ, RZ, RX, RY} |
| Quantinuum H-Series | {CX, RZ, RY, RX} |

## Gürültü Modelleri

| Kanal | Açıklama | Parametre |
|-------|----------|-----------|
| Depolarizing | Rastgele Pauli hatası | p ∈ [0,1] |
| BitFlip | |0⟩↔|1⟩ çevirme | p ∈ [0,1] |
| PhaseFlip | Faz hatası (Z) | p ∈ [0,1] |
| AmplitudeDamping | Enerji kaybı (T1) | γ ∈ [0,1] |

## Hata Düzeltme Kodları

| Kod | Notasyon | Düzeltilebilir Hata |
|-----|----------|---------------------|
| BitFlip | [[3,1,1]] | 1 bit-flip |
| PhaseFlip | [[3,1,1]] | 1 faz-flip |
| Steane | [[7,1,3]] | 1 rastgele tek-qubit hatası |

## Katman 3 — Deklaratif API

### search(num_bits, target, shots)
- Grover algoritmasını otomatik uygular
- Optimal iterasyon sayısını hesaplar: π/4·√(N/M)
- Hedef: int veya lambda fonksiyonu
- Başarı: %96+ (4 qubit'te)

### optimize(num_bits, cost, minimize, layers)
- QAOA algoritmasını otomatik uygular
- Grid search ile parametre optimizasyonu
- Minimize veya maximize
- 50 rastgele deneme ile en iyi parametreleri bulur

### MultiAgentSystem
- Ajan = qubit (süperpozisyonda)
- Etkileşim = dolanıklık (strength: 0-1)
- Karar = ölçüm (collapse)
- Çıktı: marjinal olasılıklar, korelasyonlar

## Görselleştirme

- **ASCII devre çizimi**: `draw(circuit)`
- **Olasılık histogramı**: `show_probabilities(result)`
- **Durum vektörü tablosu**: `show_statevector(sv, n)`
- **Faz diyagramı**: `show_phases(sv, n)`

## OpenQASM 3.0 Desteği

```python
from quanta.export.qasm import to_qasm
print(to_qasm(bell_state))
# OPENQASM 3.0;
# include "stdgates.inc";
# qubit[2] q;
# h q[0];
# cx q[0], q[1];
```
