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
| **Gurultu Modeli** | 4 kanal | Kapsamli | Kapsamli | Plugin |
| **QEC Kodlari** | 4 kod (surface dahil) | Dis | Dis | Yok |
| **QASM I/O** | 2.0 + 3.0 | 2.0/3.0 | 2.0 | Yok |
| **Coklu-Ajan** | Evet | Hayir | Hayir | Hayir |
| **VQE** | Dahili | qiskit-nature | cirq-core | Dahili |
| **Shor** | Dahili | Dis | Yok | Yok |
| **Tekillestime** | Dahili (QAOA) | Yok | Yok | Yok |
| **Bagimlilik** | 1 (numpy) | 20+ | 10+ | 10+ |

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
| Testler | 150+ | 5000+ |
| Maks qubit (sim) | 27 | 32 |
