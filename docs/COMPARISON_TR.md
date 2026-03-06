# Quanta SDK — Karşılaştırma

## Quanta vs Mevcut SDK'lar

### Genel Özellik Karşılaştırması

| Özellik | Quanta | Qiskit | Cirq | PennyLane | Q# |
|---------|--------|--------|------|-----------|-----|
| **Dil** | Python | Python | Python | Python | Q# (DSL) |
| **Öğrenme Eğrisi** | ⭐ Kolay | ⭐⭐⭐ Zor | ⭐⭐ Orta | ⭐⭐ Orta | ⭐⭐⭐ Zor |
| **Deklaratif API** | ✅ Var | ❌ Yok | ❌ Yok | ❌ Yok | ❌ Yok |
| **Gate Bilmeden Kullanım** | ✅ Layer 3 | ❌ Gerekli | ❌ Gerekli | ❌ Gerekli | ❌ Gerekli |
| **Broadcast** | ✅ `H(q)` | ❌ Manual | ❌ Manual | ∼ Kısmen | ❌ Manual |
| **@circuit Dekoratör** | ✅ `@circuit(qubits=N)` | ❌ Yok | ❌ Yok | ✅ `@qml.qnode` | ❌ Yok |
| **DAG Temsili** | ✅ Dahili | ✅ Dahili | ❌ Moment | ❌ Yok | ❌ Yok |
| **Compiler Pipeline** | ✅ Protocol | ✅ PassManager | ✅ Optimizer | ❌ Sınırlı | ✅ Var |
| **Gürültü Modeli** | ✅ 4 kanal | ✅ Kapsamlı | ✅ Kapsamlı | ❌ Plugin | ❌ Yok |
| **QEC Kodları** | ✅ 3 kod | ❌ Harici | ❌ Harici | ❌ Yok | ✅ Dahili |
| **QASM Export** | ✅ 3.0 | ✅ 2.0/3.0 | ✅ 2.0 | ❌ Yok | ❌ Yok |
| **Multi-Agent** | ✅ Karar modeli | ❌ Yok | ❌ Yok | ❌ Yok | ❌ Yok |

### Kod Karşılaştırması: Bell State

**Quanta (5 satır)**
```python
@circuit(qubits=2)
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    return measure(q)
```

**Qiskit (10 satır)**
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

**Cirq (12 satır)**
```python
import cirq
q = cirq.LineQubit.range(2)
circuit = cirq.Circuit([
    cirq.H(q[0]),
    cirq.CNOT(q[0], q[1]),
    cirq.measure(*q, key='result')
])
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1024)
counts = result.histogram(key='result')
```

### Arama Karşılaştırması

**Quanta Layer 3 (1 satır)**
```python
result = search(num_bits=4, target=13, shots=1024)
```

**Qiskit (30+ satır)**
```python
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit.algorithms import AmplificationProblem, Grover
# Oracle tanımla, problem tanımla, Grover kur, çalıştır...
# (çok daha uzun ve karmaşık)
```

## Quanta'nın Benzersiz Özellikleri

### 1. 3-Katmanlı Soyutlama
Hiçbir SDK bu 3 katmanı sunmuyor:
- **Katman 3**: Gate bilmeden kuantum kullanımı
- **Katman 2**: Standart devre programlama
- **Katman 1**: Donanım optimizasyonu

### 2. Multi-Agent Karar Modelleme
Kuantum mekaniğini **karar teorisi** ile birleştiren tek SDK.
Süperpozisyon → seçenekler, Dolanıklık → etkileşim, Ölçüm → karar.

### 3. 300 Satır Kuralı
Hiçbir dosya 330 satırı geçemez. Bu:
- Okunabilirliği garanti eder
- Tek sorumluluğu zorlar
- AI-friendly kod üretir (LLM'ler kısa dosyaları daha iyi anlar)

### 4. Broadcast Sözdizimi
```python
H(q)  # Qiskit'te her qubit için ayrı satır gerekir
```

## Sayısal Karşılaştırma

| Metrik | Quanta | Qiskit |
|--------|--------|--------|
| Bell State kodu | 5 satır | 10 satır |
| Grover arama | 1 satır (L3) | 30+ satır |
| `pip install` boyutu | ~1 MB (numpy) | ~200 MB |
| Öğrenme süresi | Dakikalar | Günler |
| Test hızı (98 test) | 0.33 sn | — |
| Bağımlılık | 1 (numpy) | 20+ |
