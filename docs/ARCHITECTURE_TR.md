# Quanta SDK — Mimari

## Genel Bakis

Quanta, **3 katmanli bagimsiz mimari** ile tasarlanmistir. Her katman bagimsiz olarak kullanilabilir.

## Katman Diyagrami

```
+---------------------------------------------------------+
|              KATMAN 3: DEKLARATIF API                   |
|  search() | optimize() | vqe() | factor() | qsvm()     |
|  portfolio_optimize() | resolve() | MultiAgentSystem    |
|  "Ne istiyorsunuz?" -- kapi bilgisi gereksiz            |
+---------------------------------------------------------+
|              KATMAN 2: ALGORITMIK DSL                   |
|  @circuit | H/CX/RZ | measure() | run() | sweep()      |
|  custom_gate() | 17 yerlesik kapi                       |
|  "Devreyi nasil kuracagiz?"                             |
+---------------------------------------------------------+
|              KATMAN 1: FIZIKSEL MOTOR                   |
|  DAG | Derleyici | Yonlendirme | Simulator | QEC | QASM|
|  "Donanim uzerinde nasil calisacak?"                    |
+---------------------------------------------------------+
```

## Bagimlilik Grafi

```
layer3/ -------> simulator/ -------> core/
                      |
runner.py -------> dag/ -------> core/
                      |
compiler/ -------> dag/ -------> core/
                      |
backends/ -------> simulator/ -------> core/
                      |
export/ -------> dag/ -------> core/
                      |
benchmark/ -------> export/ + simulator/ + compiler/
                      |
qec/ -------> core/
```

**Kural**: Bagimliliklar daima asagi akar. Alt katman ust katmana bagli degildir.

## Modul Detaylari

### core/ -- Temel Yapi Taslari

| Dosya | Sorumluluk |
|-------|------------|
| `types.py` | QubitRef, Instruction, QubitRegister |
| `gates.py` | 25 kapi + broadcast (IBM Heron paritesi) |
| `circuit.py` | @circuit dekoratoru, CircuitBuilder |
| `measure.py` | Esnek olcum (tam, kismi) |
| `equivalence.py` | Uniter karsilastirma, sadakat |
| `custom_gate.py` | Kullanici tanimli uniter kapilar |

### dag/ -- Yonlu Dongusuz Graf

| Dosya | Sorumluluk |
|-------|------------|
| `node.py` | InputNode, OpNode, OutputNode |
| `dag_circuit.py` | Topolojik siralama (Kahn), derinlik, paralel katmanlar |

### compiler/ -- Optimizasyon Hatti

| Dosya | Sorumluluk |
|-------|------------|
| `pipeline.py` | CompilerPass Protokolu, zincirleme, istatistikler |
| `passes/optimize.py` | CancelInverses (H.H=I), MergeRotations |
| `passes/translate.py` | IBM/Google/Quantinuum kapi seti cevirisi |
| `passes/routing.py` | Topoloji bazli SWAP ekleme (linear/ring/grid) |

### simulator/ -- Simulasyon Motorlari

| Dosya | Sorumluluk |
|-------|------------|
| `statevector.py` | Tensor contraction, 27 qubite kadar, `apply_phase()` + `apply_noise()` public API |
| `density_matrix.py` | Karisik durumlar + Kraus gurultu, 13 qubite kadar |
| `pauli_frame.py` | Aaronson-Gottesman stabilizer tablosu, 50-qubit GHZ <5s |
| `noise.py` | 7 gurultu kanali: Depolarizing, BitFlip, PhaseFlip, AmplitudeDamping, T2Relaxation, Crosstalk, ReadoutError |
| `accelerated.py` | JAX-GPU / CuPy otomatik algilama, NumPy fallback |

### layer3/ -- Deklaratif API

| Dosya | Sorumluluk |
|-------|------------|
| `search.py` | Otomatik Grover aramasi |
| `optimize.py` | QAOA optimizasyonu |
| `agent.py` | Coklu ajan karar modelleme |
| `vqe.py` | Variasyonel Kuantum Ozdeger Cozucu |
| `shor.py` | Tam sayi carpanlara ayirma (periyot bulma + QFT) |
| `qsvm.py` | Kuantum cekirdek SVM siniflandirma |
| `finance.py` | Portfoy optimizasyonu (Markowitz + QAOA) |
| `hamiltonian.py` | Trotter zaman evrimi, molekuler Hamiltonianlar |
| `entity_resolution.py` | QAOA tabanli musteri tekillestime |

### export/ -- QASM Giris/Cikis

| Dosya | Sorumluluk |
|-------|------------|
| `qasm.py` | OpenQASM 3.0 cikti |
| `qasm_import.py` | QASM 2.0/3.0 girdi -> DAG |

### qec/ -- Hata Duzeltme

| Dosya | Sorumluluk |
|-------|------------|
| `codes.py` | BitFlip [[3,1,3]], PhaseFlip [[3,1,3]], Steane [[7,1,3]] |
| `surface_code.py` | Surface code [[d^2,1,d]], stabilizer-tabanli sendrom cikarimi |
| `color_code.py` | Color code, ucgensel kafes, restriction decoder |
| `decoder.py` | MWPM + Union-Find kod cozuculer |

### benchmark/ -- Kalite Olcumu

| Dosya | Sorumluluk |
|-------|------------|
| `qasmbench.py` | 10 standart + 3 buyuk QASMBench devresi |
| `benchpress_adapter.py` | SDK arasi karsilastirma API'si (Nation et al.) |

### Destek Modulleri

| Dosya | Sorumluluk |
|-------|------------|
| `runner.py` | 6 asamali orkestrator: build > DAG > compile > sim > noise > sample > result |
| `result.py` | Olcum sonuclari, olasiliklar, Dirac notasyonu |
| `visualize.py` | ASCII devre diyagrami |
| `visualize_state.py` | Olasilik histogrami, faz diyagrami |
| `mcp_server.py` | MCP sunucusu — AI destekli kuantum hesaplama icin 14 arac (SSE + stdio) |

## Veri Akisi

```
Kullanici Kodu          SDK Ic Yapisi
    |                       |
@circuit(qubits=N) ---> CircuitDefinition
    |                       |
H(q[0]), CX(...)    ---> CircuitBuilder (tembel Instruction listesi)
    |                       |
measure(q)          ---> MeasureSpec
    |                       |
run(circuit)        ---> +- DAGCircuit.from_builder()
                         +- CompilerPipeline.run(dag)
                         +- StateVectorSimulator.apply(ops)
                         +- simulator.sample(shots)
                         +- Result(counts, probs, statevector)
```

## Tasarim Kararlari

1. **Tembel Degerlendirme**: Kapilar aninda uygulanmaz, Instruction olarak kaydedilir
2. **DAG Temsili**: Paralellik tespiti ve optimizasyon icin gerekli
3. **Protokol tabanli**: CompilerPass bir Protocol -- duck typing yeterli
4. **Degismez**: QubitRef, Instruction, dugumler frozen dataclass
5. **Thread-local Builder**: Birden fazla devre esanlamli kurulabilir
6. **Hibrit yaklasim**: Gercek dunya problemleri icin klasik bloklama + kuantum optimizasyon
7. **Hafiflik**: Saf Python + NumPy — sunucusuz (Lambda, Cloud Functions), edge computing ve CI/CD entegrasyonu icin ideal
8. **AI-yerlesik**: MCP sunucusu AI asistanlarin dogrudan kuantum hesaplama yapmasini saglar
9. **Kapsulleme**: Tum simulator durum erisimi public API uzerinden (`state`, `apply_phase`, `apply_noise`) — dis `_state` erisimi yok
10. **Gurultu-oncelikli**: Gurultu kanallari `run()` hattinda entegre, sonradan eklenmemis
