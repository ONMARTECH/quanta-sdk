# Quanta SDK — Kurulum Kılavuzu / Installation Guide

## Temel Kurulum / Basic Installation

### 1. PyPI (Tavsiye edilen / Recommended)

```bash
# Temel sürüm / Base install (NumPy core)
pip install quanta-sdk

# Seçimli paket ekstraları / Optional extras
pip install "quanta-sdk[torch]"   # PyTorch & quanta.torch autograd
pip install "quanta-sdk[metal]"   # Apple Silicon Metal / MLX GPU hızlandırma
pip install "quanta-sdk[mcp]"     # 25 araçlı FastMCP sunucusu / AI Agent MCP server
pip install "quanta-sdk[qml]"     # scikit-learn QML yardımcıları
pip install "quanta-sdk[docs]"    # MkDocs Material dokümantasyon derleme
```

### 2. Kaynaktan Kurulum / From Source

```bash
# Projeyi klonla / Clone the project
git clone https://github.com/ONMARTECH/quanta-sdk.git
cd quanta-sdk

# Geliştirici modunda kur / Install in development mode
pip install -e ".[dev]"

# Testleri çalıştır / Run tests
pytest
```

### 3. Bağımlılıklar & Ekstralar / Dependencies & Extras

| Paket / Ekstra | Amaç / Purpose | Zorunlu / Required |
|----------------|----------------|-------------------|
| `numpy>=1.24` | Çekirdek devre, derleyici ve statevector motoru / Core engine | ✅ Evet / Yes |
| `scipy`, `networkx` | Katman-3 çözücüler ve QEC dekoderleri (MWPM, ColorCode ILP) / Layer-3 & QEC | ⚙️ İlgili modüller için / For L3 & QEC |
| `[torch]` (`torch>=2.0`) | `QuantumLayer`, Daleckii-Krein autograd ve `quanta.torch` modülleri | ➕ İsteğe bağlı / Optional |
| `[metal]` (`mlx>=0.20`, `mlx-metal>=0.20`) | Apple Silicon Metal GPU sıfır-kopya simülasyon (`MLXSimulator`) | ➕ İsteğe bağlı / Optional |
| `[mcp]` (`fastmcp>=3.0`) | 25 araçlı Model Context Protocol (MCP) sunucusu | ➕ İsteğe bağlı / Optional |
| `[qml]` (`scikit-learn>=1.0`) | QML veri ön-işleme ve değerlendirme araçları | ➕ İsteğe bağlı / Optional |
| `[docs]` (`mkdocs-material`, `mkdocstrings`) | Dokümantasyon sitesi üretimi | 🔧 Geliştirici / Dev only |
| `[dev]` (`pytest`, `ruff`, `mypy`, `hypothesis`) | Test, statik tip kontrolü ve kod kalitesi (2,826+ test) | 🔧 Geliştirici / Dev only |

---

## Google Quantum Engine Kurulumu / Google Quantum Engine Setup

Google'ın kuantum bilgisayarlarını kullanmak için:
To use Google's quantum computers:

### Adım 1: Google Cloud Hesabı / Step 1: Google Cloud Account

```bash
# Google Cloud SDK kur / Install Google Cloud SDK
brew install google-cloud-sdk        # macOS
# veya / or: https://cloud.google.com/sdk/docs/install

# Giriş yap / Login
gcloud auth login
gcloud auth application-default login
```

### Adım 2: Proje Oluştur / Step 2: Create Project

```bash
# Yeni proje oluştur / Create new project
gcloud projects create quanta-quantum --name="Quanta Quantum"

# Projeyi seç / Select project
gcloud config set project quanta-quantum

# Quantum Engine API'yi etkinleştir / Enable Quantum Engine API
gcloud services enable quantum.googleapis.com

# Faturalandırmayı etkinleştir / Enable billing
# Google Cloud Console → Billing → Link project
```

### Adım 3: Cirq Kur / Step 3: Install Cirq

```bash
# Cirq + Google backend
pip install cirq-google

# Doğrula / Verify
python -c "import cirq; import cirq_google; print('✅ Cirq kuruldu / installed')"
```

### Adım 4: Quanta ile Kullan / Step 4: Use with Quanta

```python
from quanta import circuit, H, CX, measure, run
from quanta.backends.google import GoogleBackend

# ── Lokal test (Google hesabı gerekmez) ──
# ── Local test (no Google account needed) ──
backend = GoogleBackend(simulate_locally=True)

@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)

result = backend.execute(bell, shots=1024)
print(result.summary())

# ── Gerçek donanım (Google Cloud gerekli) ──
# ── Real hardware (Google Cloud required) ──
backend = GoogleBackend(
    project_id="quanta-quantum",
    processor_id="rainbow",  # veya / or: "weber"
)
result = backend.execute(bell, shots=1024)
```

### Mevcut Google İşlemciler / Available Google Processors

| İşlemci / Processor | Qubit | Durumu / Status | Not / Note |
|---------------------|-------|----------------|------------|
| `rainbow` | 23 | ✅ Aktif / Active | Erişim gerekli / Access required |
| `weber` | 53 | ✅ Aktif / Active | Erişim gerekli / Access required |
| Willow | 105 | 🔬 Sınırlı / Limited | 2024+ |

> [!IMPORTANT]
> Google Quantum Engine erişimi için başvuru gerekir:
> Google Quantum Engine access requires application:
> https://quantumai.google/quantum-computing-service

---

## Mimari: QASM Köprüsü / Architecture: QASM Bridge

```
┌─────────────┐     ┌──────────┐     ┌──────────┐     ┌──────────────┐
│  Quanta     │────→│  QASM    │────→│  Cirq    │────→│  Google      │
│  @circuit   │     │  3.0     │     │  Circuit │     │  Quantum     │
│  H, CX...  │     │  export  │     │  import  │     │  Engine      │
└─────────────┘     └──────────┘     └──────────┘     └──────────────┘
    Bağımsız           Ortak dil        Transport       Donanım
    Independent        Common lang      Transport       Hardware
```

**Neden bu yöntem? / Why this approach?**

1. **Bağımsızlık / Independence**: Quanta, Cirq'a bağımlı DEĞİL. Cirq sadece Google backend kullanılırsa lazy import edilir.
2. **Genişletilebilirlik / Extensibility**: Aynı QASM çıktısı ile IBM Quantum, Amazon Braket'e de gönderilebilir.
3. **Test edilebilirlik / Testability**: `simulate_locally=True` ile Cirq'ın lokal simülatörü kullanılır — donanım gerekmez.
4. **Standart uyumluluk / Standards compliance**: QASM 3.0 endüstri standardıdır.

---

## IBM Quantum Kurulumu / IBM Quantum Setup (Zero-Qiskit Native REST)

Quanta SDK, harici Qiskit kurulumuna ihtiyaç duymadan doğrudan IBM Quantum Cloud REST API v1 ve IAM kimlik doğrulaması ile çalışan yerleşik **`IBMRestBackend`** sürücüsünü önerir (156-qubit Heron r2/r3 işlemciler: `ibm_torino`, `ibm_fez`):

```bash
export IBM_API_KEY="YOUR_IBM_CLOUD_API_KEY"
export IBM_INSTANCE_CRN="crn:v1:bluemix:public:quantum-computing:..."
```

```python
import os
from quanta.backends.ibm_rest import IBMRestBackend

# Sıfır-Qiskit doğrudan IBM Heron REST bağlantısı (Önerilen / Recommended)
backend = IBMRestBackend(
    api_key=os.environ["IBM_API_KEY"],
    instance_crn=os.environ.get("IBM_INSTANCE_CRN", ""),
    backend_name="ibm_torino",  # veya / or: "ibm_fez" (156Q Heron)
)

# Mevcut aktif işlemcileri listele / List active Heron processors
# print(backend.list_backends())
# result = backend.execute(bell, shots=1024)
```

> **Not**: İsteğe bağlı eski Qiskit uyumluluk köprüsü için `from quanta.backends.ibm import IBMBackend` (`simulate_locally=True`) de kullanılabilir; ancak üretim iş yüklerinde sıfır bağımlılıklı `IBMRestBackend` ve aktif Heron işlemcileri (`ibm_torino`, `ibm_fez`) tercih edilmelidir (eski `ibm_kyoto` emekliye ayrılmıştır).

---

## Sorun Giderme / Troubleshooting

| Sorun / Issue | Çözüm / Solution |
|---------------|-------------------|
| `ModuleNotFoundError: cirq` | `pip install cirq-google` |
| `gcloud auth` hatası | `gcloud auth application-default login` |
| `Quota exceeded` | Google Cloud Console → Quotas |
| `Processor not found` | Erişim başvurusu yapın / Apply for access |
| Lokal test yapamıyorum | `GoogleBackend(simulate_locally=True)` |
