# Quanta SDK — Kurulum Kılavuzu / Installation Guide

## Temel Kurulum / Basic Installation

### 1. Quanta SDK

```bash
# Projeyi klonla / Clone the project
git clone <repo-url>
cd quanta

# Geliştirici modunda kur / Install in development mode
pip install -e ".[dev]"

# Testleri çalıştır / Run tests
pytest
```

### 2. Bağımlılıklar / Dependencies

| Paket / Package | Amaç / Purpose | Zorunlu / Required |
|-----------------|----------------|-------------------|
| `numpy` | Simülatör motoru / Simulator engine | ✅ Evet / Yes |
| `pytest` | Test çerçevesi / Test framework | 🔧 Geliştirici / Dev only |
| `ruff` | Linting + format | 🔧 Geliştirici / Dev only |
| `hypothesis` | Property-based test | 🔧 Geliştirici / Dev only |

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

## IBM Quantum Kurulumu (gelecek) / IBM Quantum Setup (future)

```bash
# Henüz implemente edilmedi, ama yol haritasında
# Not yet implemented, but on the roadmap
pip install qiskit-ibm-runtime
```

---

## Sorun Giderme / Troubleshooting

| Sorun / Issue | Çözüm / Solution |
|---------------|-------------------|
| `ModuleNotFoundError: cirq` | `pip install cirq-google` |
| `gcloud auth` hatası | `gcloud auth application-default login` |
| `Quota exceeded` | Google Cloud Console → Quotas |
| `Processor not found` | Erişim başvurusu yapın / Apply for access |
| Lokal test yapamıyorum | `GoogleBackend(simulate_locally=True)` |
