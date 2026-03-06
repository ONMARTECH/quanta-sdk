# Quanta SDK

Temiz, modüler ve kuantum-doğal bir kuantum hesaplama SDK'sı.

## Vizyon

Quanta, mevcut kuantum SDK'larının (Qiskit, Cirq, PennyLane) karmaşıklığını
ortadan kaldırmak için tasarlanmıştır. Klasik bilgisayar mantığını kuantuma
uyarlamak yerine, kuantum ilkelerini doğal olarak kucaklar.

## 3 Katmanlı Mimari

```
Katman 3 — Deklaratif     "Ne istiyorsun?"
├── search()               Kuantum arama (Grover otomatik)
├── optimize()             Kombinasyonel optimizasyon (QAOA)
└── MultiAgentSystem       Çok ajanlı karar modelleme

Katman 2 — Algoritmik     "Devreyi nasıl kuracağız?"
├── @circuit + kapılar     H, CX, RZ, CCX...
├── measure()              Esnek ölçüm
└── run()                  Tek komutla çalıştırma

Katman 1 — Fiziksel       "Donanımda nasıl çalışacak?"
├── DAG motoru             Topolojik sıralama, paralellik
├── Compiler               Optimizasyon + transpilasyon
├── Simülatör              Durum vektörü + gürültü modeli
├── QEC                    Hata düzeltme kodları
└── Export                 OpenQASM 3.0 çıktı
```

## Hızlı Başlangıç

### Katman 2: Gate tabanlı programlama

```python
from quanta import circuit, H, CX, measure, run

@circuit(qubits=2)
def bell(q):
    H(q[0])           # Süperpozisyon
    CX(q[0], q[1])    # Dolanıklık
    return measure(q)

result = run(bell, shots=1024)
print(result.summary())
```

### Katman 3: Gate bilgisi gerekmez!

```python
from quanta.layer3.search import search

# 16 elemanlık uzayda 13'ü bul — kuantum otomatik
result = search(num_bits=4, target=13, shots=1024)
print(f"Bulunan: {result.most_frequent}")  # → 1101
```

### Multi-Agent Karar Modelleme

```python
from quanta.layer3.agent import Agent, MultiAgentSystem

system = MultiAgentSystem([
    Agent("müşteri", ["al", "vazgeç"]),
    Agent("rakip", ["indirim", "koru"]),
])
system.interact("müşteri", "rakip", strength=0.7)
result = system.simulate(shots=1024)
print(result.summary())
```

## Kurulum

```bash
pip install -e ".[dev]"
```

## Test

```bash
pytest                    # 98 test, 0.33 saniye
pytest --tb=short -v      # Detaylı çıktı
```

## Proje Yapısı

```
quanta/
├── core/           Temel tipler, kapılar, devre dekoratörü
├── dag/            DAG motoru (topolojik sort, katmanlar)
├── compiler/       Optimizasyon + transpilasyon pipeline'ı
├── simulator/      Durum vektörü simülatörü + gürültü
├── backends/       Backend soyutlama (lokal, bulut)
├── layer3/         Deklaratif API (search, optimize, agent)
├── export/         OpenQASM 3.0 çıktı
├── qec/            Kuantum hata düzeltme kodları
├── examples/       Örnek algoritmalar
└── docs/           Dökümantasyon (TR + EN)
```

## Kod Standartları

- **Max 330 satır/dosya** (300 + %10 tolerans)
- **~%30 yorum/dökümantasyon oranı**
- **Modüler**: Her dosya tek sorumluluk
- **Tip güvenli**: Tam type hint desteği
- **İmmutable**: Frozen dataclass'lar
- **Test edilmiş**: 98 test

## Lisans

MIT
