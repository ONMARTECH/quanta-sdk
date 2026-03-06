# Quanta SDK

Python icin temiz ve moduler kuantum hesaplama SDK'si.

## Genel Bakis

Quanta, kuantum hesaplama icin 3 katmanli mimari sunar:

- **Katman 3** (Deklaratif): `search()`, `optimize()`, `vqe()`, `factor()`, `resolve()` -- kapi bilmeden kuantum
- **Katman 2** (Devre): `@circuit`, H, CX, RZ, `measure()`, `run()` -- standart devre programlama
- **Katman 1** (Fiziksel): DAG, derleyici, yonlendirme, simulator, QEC, QASM -- donanim optimizasyonu

## Hizli Baslangic

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

## Ornekler

11 demo: Bell, GHZ, isinlanma, Deutsch-Jozsa, Grover, VQE, portfoy optimizasyonu, QKD, Shor, QSVM ve tekillestime.

```bash
python -m quanta.examples.01_bell_state
python -m quanta.examples.11_entity_resolution
```

## Kurulum

```bash
git clone https://github.com/ONMARTECH/quanta-sdk.git
cd quanta-sdk
pip install -e ".[dev]"
pytest
```

## Dokumantasyon

Detayli dokumantasyon icin `docs/` dizinine bakin:

- [Mimari](ARCHITECTURE_TR.md)
- [Ozellikler](FEATURES_TR.md)
- [Karsilastirma](COMPARISON_TR.md)
- [Kurulum](INSTALL_TR.md)

## Gelistirici

Abdullah Enes SARI -- ONMARTECH

info@onmartech.com

## Lisans

Apache License 2.0
