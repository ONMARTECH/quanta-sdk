# Quanta SDK -- Karsilastirma

## Quanta vs Mevcut SDK'lar

### Ozellik Karsilastirmasi

| Ozellik | Quanta | Qiskit | Cirq | PennyLane |
|---------|--------|--------|------|-----------|
| **Dil** | Python | Python | Python | Python |
| **Ogrenme Egrisi** | Kolay | Zor | Orta | Orta |
| **Deklaratif API** | Evet (Katman 3) | Hayir | Hayir | Hayir |
| **Kapisiz Kullanim** | Evet | Hayir | Hayir | Hayir |
| **Broadcast** | `H(q)` | Manuel | Manuel | Kismi |
| **@circuit Dekoratoru** | Evet | Hayir | Hayir | `@qml.qnode` |
| **DAG Temsili** | Dahili | Dahili | Moments | Yok |
| **Derleyici** | 3-gecis + yonlendirme | PassManager | Optimizer | Sinirli |
| **Gurultu Modeli** | 7 kanal | Kapsamli | Kapsamli | Plugin |
| **QEC Kodlari** | 6 kod (surface + color) | Dis | Dis | Yok |
| **QASM I/O** | 2.0 + 3.0 | 2.0/3.0 | 2.0 | Yok |
| **Coklu-Ajan** | Evet | Hayir | Hayir | Hayir |
| **VQE** | Dahili | qiskit-nature | cirq-core | Dahili |
| **Shor** | Dahili | Dis | Yok | Yok |
| **Tekillestime** | Dahili (QAOA) | Yok | Yok | Yok |
| **Bagimlilik** | 1 (numpy) | 20+ | 10+ | 10+ |
| **MCP Sunucusu** | Dahili (7 arac) | Yok | Yok | Yok |
| **Gradyanlar** | Parameter-shift + Natural | Manuel | Manuel | **Dahili (autograd)** |

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

## Sayisal Karsilastirma

| Metrik | Quanta | Qiskit |
|--------|--------|--------|
| Bell State kodu | 5 satir | 10 satir |
| Grover aramasi | 1 satir (L3) | 30+ satir |
| `pip install` boyutu | ~1 MB | ~200 MB |
| Bagimliliklar | 1 (numpy) | 20+ |
| Testler | 457 | 5000+ |
| Maks qubit (sim) | 27 | 32 |

## Diferansiyel Kuantum Hesaplama

PennyLane'in temel avantaji autograd ile diferansiyel programlamadir.
Quanta artik karsilastirmali gradyan destegi sunar:

| Ozellik | Quanta | PennyLane |
|---------|--------|-----------|
| **Parameter-shift kurali** | `parameter_shift()` | `qml.gradients.param_shift` |
| **Sonlu farklar** | `finite_diff()` | `qml.gradients.finite_diff` |
| **Dogal gradyan** | `natural_gradient()` (QFIM) | `qml.QNGOptimizer` |
| **Beklenen deger** | `expectation()` | `qml.expval()` |
| **Geri yayilim** | Henuz yok | **Evet (JAX/Torch/TF)** |
| **Cerceve entegrasyonu** | NumPy-yerel | JAX, PyTorch, TensorFlow |

### Quanta'nin Avantaji
- **Sifir bagimlilik**: Gradyanlar sadece NumPy ile calisir
- **Acik kontrol**: Yontem bazinda secim, cihaz bazinda degil
- **QFIM dahili**: Fubini-Study metrigi ile dogal gradyan
- **MCP entegrasyonu**: AI asistanlar uzaktan gradyan hesaplayabilir

### PennyLane'in Avantaji
- **Autograd geri yayilim**: Devreler uzerinden gercek ters-mod AD
- **Cerceve koprusu**: Yerlesik JAX/PyTorch/TensorFlow destegi
- **Buyuk ekosistem**: Daha fazla optimizer, daha fazla cihaz
