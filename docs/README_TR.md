# Quanta SDK

Python için yapay zeka odaklı, modüler ve yüksek başarımlı kuantum hesaplama SDK'si. **v1.1.0** — [PyPI](https://pypi.org/project/quanta-sdk/)

## Genel Bakış

Quanta, yapay zeka ajanları (MCP), derin öğrenme araştırmacıları ve üretim iş yükleri için tasarlanmış çok katmanlı modern bir kuantum çalışma ortamı sunar:

- **Derin Öğrenme Katmanı (`quanta.torch`)**: PyTorch `nn.Module` tabanlı diferansiyellenebilir `QuantumLayer` (analitik parameter-shift gradyanları) ve sürekli zamanlı `ContinuousResonator` ile biyomorfik kuantum beyni.
- **Deklaratif Katman (Katman 3)**: `search()`, `optimize()`, `vqe()`, `factor()`, `resolve()` — kapı seviyesi detaylara girmeden doğrudan algoritmik çözüm.
- **Devre DSL Katmanı (Katman 2)**: `@circuit`, 31 yerleşik kapı (H, CX, RZ, MS, ECR vb.), parametrik rotasyonlar ve ölçüm yönetimi.
- **Fiziksel & Donanım Hızlandırıcı Katmanı (Katman 1)**:
  - **Apple Silicon Metal/MLX**: M-serisi çiplerde birleşik bellek (Unified Memory) tensör kasılmalarıyla **404 kat hızlanma**.
  - **NVIDIA cuStateVec**: Büyük ölçekli GPU durum vektörü simülasyonu.
  - **Çoklu Bulut Kuantum Donanımı**: Gerçek donanım üzerinde IonQ Cloud REST API v0.3, IBM Quantum Heron r3 (156 qubit) ve Google Cirq Sycamore entegrasyonu.
- **Ajan Odaklı MCP Katmanı**: AI ajanlarının (Claude, Gemini, GPT) kuantum hesaplamalarını doğrudan yönetebilmesi için **23 Model Context Protocol (MCP) aracı**.

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
    QuantumLayer(n_qubits=4, ansatz="hardware_efficient", n_layers=2),
    nn.Linear(4, 2)
)

x = torch.randn(8, 4, requires_grad=True)
out = model(x)
loss = out.sum()
loss.backward()  # Analitik parameter-shift gradyanları otomatik hesaplanır
```

### 3. Biyomorfik Kuantum Rezonans Beyni (`BiomorphicResonantBrain`)

```python
from quanta.torch import BiomorphicResonantBrain

# Çift hemisferli, 4 nöromodülatörlü (DA, ACh, 5-HT, NE) kuantum beyni
brain = BiomorphicResonantBrain(n_qubits=4)

x = torch.randn(1, 4)
output = brain(x)

# REM uykusu bellek konsolidasyonu (katastrofik unutmayı önler)
stats = brain.consolidate_rem_sleep(replay_cycles=3)
print("Konsolidasyon Tamamlandı:", stats["retained_fidelity"])
```

---

## Donanım Hızlandırması (Apple Silicon Metal)

Apple Silicon (M1/M2/M3/M4/M5) işlemcilerde devasa hız:

```python
from quanta import run

# 26 qubitlik devrede Metal Performance Shaders ile 404 kat hızlanma:
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

Detaylı Türkçe ve İngilizce kılavuzlar için `docs/` dizinini inceleyebilirsiniz:

- [Mimari Detayları](ARCHITECTURE_TR.md)
- [Kapsamlı Özellikler ve Kapı Seti](FEATURES_TR.md)
- [Diğer SDK'lar ile Karşılaştırma](COMPARISON_TR.md)
- [Kurulum Kılavuzu](INSTALL_TR.md)
- [Akademik Teori ve Kanıtlar](theory/quantum_brain_frontiers.md)

---

## Geliştirici & İletişim

**Abdullah Enes SARI** — ONMARTECH  
E-posta: info@onmartech.com  
Web: [onmartech.com](https://onmartech.com)

## Lisans

Apache License 2.0
