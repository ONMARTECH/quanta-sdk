# Quanta SDK

Python için yapay zeka odaklı, modüler ve yüksek başarımlı kuantum hesaplama SDK'si. **v1.2.0-production** — [PyPI](https://pypi.org/project/quanta-sdk/) · [Dokümantasyon](https://quanta.onmartech.com/) · [GitHub Release](https://github.com/ONMARTECH/quanta-sdk/releases/tag/v1.2.0)

## Genel Bakış

Quanta SDK, 2026 yılı kuantum bilişim standartlarında tasarlanmış, yapay zeka ajanları (MCP), derin öğrenme araştırmacıları ve hataya dayanıklı kuantum hesaplama (FTQC) iş yükleri için geliştirilmiş bağımsız bir kuantum yazılım mimarisidir. Sistem 5 temel bilimsel paradigma üzerine kuruludur:

1. **İlk-İlkeler Çekirdeği (Local-First Core)**: Çekirdek devre/derleyici/statevector yalnızca NumPy gerektirir (SciPy ve NetworkX Katman-3 çözücülerini ve QEC dekoderlerini destekler); ağır C++/LLVM derleme zincirleri veya CUDA kısıtları olmadan **34 yerleşik kuantum kapısı**, evrensel **1Q ZYZ / Heron / 2Q Cartan KAK $SU(4)$ üniter sentezi** ve **127Q/156Q IBM Heavy-Hex yönlendirmesi** sunar (IBM Heron, Google Sycamore ve IonQ uyumlu).
2. **Apple Silicon Metal / MLX Sıfır-Kopya GPU Hızlandırma**: Birleşik bellek (Unified Memory) mimarisinde CPU-GPU veri kopyalama maliyetlerini sıfırlayan Metal/MLX tensör motoru (kapı derinliğine ve qubit sayısına bağlı olarak Apple Silicon Metal GPU üzerinde 58x–404x'e varan hızlanma) ve SIMD vektörize >3.13M kapı/sn Clifford simülatörü.
3. **Sürekli Hilbert Gradyanları & Daleckii-Krein Autograd (`quanta.torch`)**: Padé sapmalarını sıfırlayan $10^{-15}$ hassasiyetinde kapalı formlu Fréchet matris türevi, dinamik Lie cebiri ($\dim(\mathfrak{g})$) tabanlı barren plateau analizi ve kuantum ile nörobiyoloji esinli diferansiyellenebilir sürekli-zamanlı Hamiltonyen PyTorch mimarileri (`ContinuousResonantLayer`, `BiomorphicResonantBrain`).
4. **2026 Çift-Kanal Hata Toleransı Motoru (Dual-Track FTQC)**:
   - **Track A (2D Topolojik)**: Edmonds Blossom MWPM ve Delfosse-Nickerson Union-Find kod çözme, Kubica-Delfosse $A_2$ örgüsü + ILP Color Code dekoderi ve Google Willow uyumlu 3D uzay-zaman sendrom döngüleri.
   - **Track B (Yüksek Dereceli qLDPC)**: 2D yüzey kodlarına kıyasla $12\times$ qubit tasarrufu sağlayan kanonik Gross $[[144, 12, 12]]$ Bivariate Bicycle kodu ve yerel Normalized Min-Sum BP-OSD dekoderi.
   - **Non-Clifford Evrensellik**: 15-to-1 Bravyi-Kitaev sihirli durum damıtma (magic state distillation) fabrikası ($\epsilon_{\text{out}} \le 35 p^3$) ve örgü cerrahisi (lattice surgery).
5. **Ajan Odaklı MCP Katmanı**: Otonom AI ajanlarının (Claude, Gemini, GPT) kuantum hesaplamalarını doğrudan yönetebilmesi için **25 Model Context Protocol (MCP) aracı**.
6. **Doğrulanabilir Bilimsel Titizlik**: **2.826+ otomatize test** ile üniterlik ($\|U^\dagger U - I\| < 10^{-14}$) ve CPTP iz korunum garantisi.

---

## Hızlı Başlangıç

### 1. Klasik Kuantum Devresi

```python
from quanta import circuit, H, CX, measure, run

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = run(bell, shots=1024)
print(result.summary())
```

### 2. PyTorch Diferansiyellenebilir `QuantumLayer`

```python
import torch
import torch.nn as nn
from quanta.torch import QuantumLayer

# Standart PyTorch modeline kuantum katmanı entegrasyonu
model = nn.Sequential(
    nn.Linear(4, 4),
    QuantumLayer(num_qubits=4, circuit_fn="hardware_efficient", num_layers=2),
    nn.Linear(4, 2)
)

x = torch.randn(8, 4, requires_grad=True)
out = model(x)
loss = out.sum()
loss.backward()  # Analitik parameter-shift gradyanları otomatik hesaplanır
```

### 3. Biyomorfik Kuantum Rezonans Beyni (`BiomorphicResonantBrain`)

`BiomorphicResonantBrain`, **kuantum ve nörobiyoloji esinli diferansiyellenebilir sürekli-zamanlı Hamiltonyen PyTorch mimarisidir** (biyolojik dokuda fiziksel kuantum üstünlüğü iddiası değil; açık kuantum sistem dinamiği ve nöromodülasyondan esinlenen klasik derin öğrenme temsil modelidir):

```python
import torch
from quanta.torch.brain import BiomorphicResonantBrain, QuantumREMSleep

# Çift hemisferli diferansiyellenebilir Hamiltonyen PyTorch mimarisi
brain = BiomorphicResonantBrain(in_features=16, num_left_qubits=2, num_right_qubits=2)

x = torch.randn(2, 16)
output = brain(x)

# Çoklu zaman ölçekli Hamiltonyen faz-dağılımı tekrarı (sürekli öğrenme için)
rem = QuantumREMSleep(brain_module=brain, sleep_cycles=3)
rem.register_memory_state(output["state"])
stats = rem.sleep()
print("Bellek Korunum Tahmini:", stats["retention_estimate"])
```

---

## Donanım Hızlandırması (Apple Silicon Metal)

Apple Silicon Metal GPU üzerinde kapı derinliğine ve qubit sayısına bağlı olarak 58x–404x'e varan hızlanma:

```python
from quanta import run

result = run(large_circuit, backend="mlx")
```

---

## Kurulum

```bash
# PyPI üzerinden kararlı sürüm
pip install quanta-sdk

# PyTorch ve Derin Öğrenme desteğiyle birlikte
pip install "quanta-sdk[torch]"

# Apple Silicon Metal (MLX) desteğiyle birlikte
pip install "quanta-sdk[metal]"

# Geliştirici modu ve testler
git clone https://github.com/ONMARTECH/quanta-sdk.git
cd quanta-sdk
pip install -e ".[dev]"
pytest
```

---

## Dokümantasyon
 
Detaylı Türkçe ve İngilizce kılavuzlar için dokümantasyon merkezini ziyaret edebilirsiniz:

- [Resmi Dokümantasyon Portalı](https://quanta.onmartech.com/)
- [Quanta Mimari Makalesi (Whitepaper)](papers/quanta_framework_paper.md)
- [Mimari Detayları](ARCHITECTURE_TR.md)
- [Kapsamlı Özellikler ve Kapı Seti](FEATURES_TR.md)
- [Diğer SDK'lar ile Karşılaştırma](COMPARISON_TR.md)
- [Kurulum Kılavuzu](INSTALL_TR.md)
- [Akademik Teori ve Monograflar](theory/quantum_brain_frontiers.md)

---

## Geliştirici & Yazar Künyesi

- **Baş Mimar & Yazar**: Abdullah Enes SARI (ORCID: [0000-0002-8827-0587](https://orcid.org/0000-0002-8827-0587))
- **Kurum**: ONMARTECH Kuantum Teknolojileri İnisiyatifi (`info@onmartech.com`)
- **Web**: [onmartech.com](https://onmartech.com) · [quanta.onmartech.com](https://quanta.onmartech.com)
- **GitHub Release**: [GitHub Release v1.2.0](https://github.com/ONMARTECH/quanta-sdk/releases/tag/v1.2.0)

## Lisans

Apache License 2.0

