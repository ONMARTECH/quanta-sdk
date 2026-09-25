# Quanta SDK -- Karşılaştırma (v1.2.0-production)

## Quanta vs Diğer Kuantum Kütüphaneleri

### Kapsamlı Mimari & Yetenek Karşılaştırması

| Boyut / Özellik | **Quanta SDK** | Qiskit | Cirq | PennyLane | Stim | PyMatching | QuTiP |
|---|---|---|---|---|---|---|---|
| **Birincil Felsefe** | Local-First & Bilişsel | Kurumsal / Bulut | Donanım (Google) | Hibrit QML | Hızlı Stabilizer | MWPM Dekoder | Açık Kuantum |
| **Harici Bağımlılık** | **0** (Saf Python/NumPy) | 20+ paket | 10+ paket | 10+ paket | 1 (C++ derleme) | 1 (C++ derleme) | 5+ (SciPy/Cython) |
| **Donanım Hızlandırma** | **Metal / MLX Zero-Copy** | C++/Rust (Aer) | C++ (qsim) | JAX/Torch C++ | SIMD C++ (AVX2) | SIMD C++ | C/OpenMP |
| **Kuantum Kapı Seti** | **31 Yerleşik Kapı** | 50+ | 60+ | 30+ | Yalnızca Clifford | N/A | Operatör Bazlı |
| **Türev & Gradyan** | **Daleckii-Krein + Parameter-Shift** | Sonlu Farklar | Manuel | Parameter-Shift / AD | N/A | N/A | N/A |
| **Autograd / PyTorch Entegrasyonu**| **Tam (`quanta.torch` nn.Module)** | Kısmi / Eklenti | Eklenti | **Dahili (Torch/JAX)** | N/A | N/A | N/A |
| **Lie Cebiri Barren Analizi** | **Dahili ($\dim(\mathfrak{g})$)** | N/A | N/A | N/A | N/A | N/A | N/A |
| **Topolojik QEC (Track A)** | **Edmonds Blossom MWPM (Willow 3D)** | Ayrık repo | N/A | N/A | Sadece Algılama | **Sadece Eşleme** | N/A |
| **Yüksek Dereceli qLDPC (Track B)** | **Gross $[[144, 12, 12]]$ + BP-OSD-0** | N/A | N/A | N/A | N/A | N/A | N/A |
| **Sihirli Durum Damıtma** | **15-to-1 Bravyi-Kitaev + Örgü Cerrahi** | N/A | N/A | N/A | N/A | N/A | N/A |
| **AI Ajan Entegrasyonu** | **Dahili (23 MCP Aracı)** | N/A | N/A | N/A | N/A | N/A | N/A |
| **Otomatize Test Sayısı** | **2.076 Test (%100 Başarı)** | 5000+ | 3000+ | 2500+ | 800+ | 200+ | 1200+ |

### Kod Karsilastirmasi: Bell Durumu

**Quanta (5 satir)**
```python
@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)
```

**Qiskit (10 satir)**
```python
from qiskit import QuantumCircuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
from qiskit_aer import AerSimulator
simulator = AerSimulator()
result = simulator.run(qc, shots=1024).result()
counts = result.get_counts()
```

### VQE Karsilastirmasi

**Quanta (3 satir)**
```python
from quanta.layer3.vqe import vqe
result = vqe(2, hamiltonian=[("ZZ", 1.0), ("XI", 0.5)], layers=3)
print(result.energy)
```

**Qiskit (20+ satir)**
```python
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.algorithms.minimum_eigensolvers import VQE
# Mapper, ansatz, optimizer, VQE kurulumu, calistirma...
```

## Quanta'nin Farkliliklari

### 1. 3 Katmanli Soyutlama
- **Katman 3**: Kapi bilmeden kuantum kullanimi
- **Katman 2**: Standart devre programlama
- **Katman 1**: Donanim optimizasyonu

### 2. Gercek Dunya Kullanim Alanlari
- Musteri tekillestime (entity resolution)
- Portfoy optimizasyonu (finans)
- Molekuler simulasyon (H2, LiH, HeH+)

### 3. Tek Bagimlilik
Sadece NumPy. 200MB kurulum yok, Java yok, Rust toolchain yok.

## Sayısal Karşılaştırma & Özet Metrikler

| Metrik | Quanta SDK | Qiskit | PennyLane |
|--------|------------|--------|-----------|
| Bell State kodu | **5 satır** | 10 satır | 6 satır |
| Grover araması | **1 satır (L3)** | 30+ satır | 25+ satır |
| `pip install` boyutu | **~5 MB** | ~200 MB | ~150 MB |
| Harici Bağımlılıklar | **0** (Saf NumPy) | 20+ paket | 10+ paket |
| Test Sayısı | **2.076 adet** (%100 Başarı) | 5000+ | 2500+ |
| Maks Qubit (Simülasyon) | **200+ (MPS) / 27+ (Metal)** | 32 (Aer) | 26 (Default) |
| Clifford Kapı Hızı | **>3.13M kapı/saniye** | ~500k | N/A |
| FTQC qLDPC Bellek Tasarrufu | **12× Qubit Tasarrufu** | Yok | Yok |

## Diferansiyel Kuantum Hesaplama & Autograd

PennyLane'in temel avantajı diferansiyellenebilir kuantum programlamadır. Quanta SDK `v1.2.0-production` sürümünde bu alanı doğrudan aşan analitik ve ters-mod yetenekler sunar:

| Özellik | Quanta SDK (`quanta.torch`) | PennyLane |
|---------|-----------------------------|-----------|
| **Parameter-shift kuralı** | `parameter_shift()` & `QuantumLayer` | `qml.gradients.param_shift` |
| **Sonlu farklar** | `finite_diff()` | `qml.gradients.finite_diff` |
| **Doğal gradyan (QFIM)** | `natural_gradient()` (Fubini-Study) | `qml.QNGOptimizer` |
| **Geri Yayılım (Autograd)** | **Tam Dahili (`torch.autograd` / VJP)** | Dahili (Torch/JAX/TF) |
| **Daleckii-Krein Fréchet Türevi**| **Var ($9.99\times 10^{-16}$ kapalı form)** | Yok (Padé / finite diff) |
| **Lie Cebiri Barren Analizi**| **Dahili ($\dim(\mathfrak{g})$ analitik tarama)** | Yok |
| **Çerçeve entegrasyonu** | **Saf NumPy + PyTorch (`nn.Module`)** | JAX, PyTorch, TensorFlow |
| **Biyomorfik Bellek Rezonansı** | **Dahili (`BiomorphicResonantBrain`)** | Yok |

### Quanta'nın Ayırıcı Üstünlükleri
- **Daleckii-Krein Spektral Doğruluğu**: Matris üstellerinin türevinde Padé hatalarını ($>10^{-6}$) ortadan kaldıran makine hassasiyetinde türev.
- **Sıfır Bağımlılıklı Çekirdek**: İsteğe bağlı PyTorch olmadan da saf NumPy üzerinde parameter-shift ve analitik gradyanlar.
- **2026 Çift-Kanal FTQC**: Hem Edmonds Blossom MWPM yüzey kodları hem de Gross $[[144, 12, 12]]$ qLDPC kod çözümü.
- **MCP Ajan Entegrasyonu**: AI asistanlarının kuantum devrelerini uzaktan inşa edip gradyanlarını optimize edebileceği 23 araç.

---

## Yazar & Mimarlık Künyesi

- **Baş Mimar**: Abdullah Enes SARI (ORCID: [0000-0002-8827-0587](https://orcid.org/0000-0002-8827-0587))
- **Kurum**: ONMARTECH Kuantum Bilişim İnisiyatifi (`info@onmartech.com`)
- **Yazılım DOI**: [10.5281/zenodo.22952779](https://doi.org/10.5281/zenodo.22952779)

