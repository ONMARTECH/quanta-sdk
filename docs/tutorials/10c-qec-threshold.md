# 🛡️ Quantum Hata Düzeltme Eşiği — Ölçeklenebilirliğin Kanıtı

*Surface Code ile Error Correction Threshold Bulma*

---

> **Bu tutorialda yapacağınız şey:** Surface code'u farklı hata oranlarında
> simüle edip, **eşik hata oranını (threshold)** bulacaksınız. Bu eşiğin altında quantum
> hata düzeltme çalışıyor, üstünde çöküyor. Bu, quantum bilgisayarların
> **gerçekten ölçeklenebileceğinin** deneysel kanıtı.

---

## Neden QEC Eşiği Bu Kadar Önemli?

Quantum bilgisayarların en büyük sorunu: **qubit'ler hata yapıyor.**
Tek bir qubit'in hata oranı ~%0.1–1. 1000 qubit'lik bir hesaplama
yaparsanız, neredeyse kesin hata alırsınız.

Çözüm: **Quantum Error Correction (QEC).**

Ama QEC'in çalışması fiziksel gürültünün **eşik hata oranının ($p_{\text{th}}$)** altında olmasına bağlıdır. Surface code için eşik değeri, gürültü modelinin ayrıntı düzeyine göre üç standart rejimde incelenir:

| Gürültü Modeli | Açıklama | Surface Code Eşiği ($p_{\text{th}}$) | Quanta Metodu |
|---|---|---|---|
| **Code-Capacity** | Yalnızca veri qubitlerinde bağımsız hata, kusursuz sendrom ölçümü | **~%10.3** | `simulate_error_correction()` |
| **3D Phenomenological** | Veri hatası + gürültülü çok döngülü sendrom ölçümü ($\Delta s_t$) | **~%2.9** | `simulate_dynamic()` |
| **Circuit-Level Depolarizing** | Sendrom çıkarma CNOT kapılarının kendisi de gürültülü | **~%0.7 – %1.1** | Donanım / tam devre seviyesi |

Bu tutorialda `simulate_error_correction()` ile **code-capacity (~%10.3)** eşiğini doğrudan gözlemleyeceğiz (3D uzay-zaman modeli için bkz. [Tutorial 11 — Willow Dynamic QEC](11-willow-dynamic-qec.md)). Eşiğin altında:
- Code distance ($d$) artırınca logical hata oranı **üstel olarak düşer**
- Yani daha fazla qubit = daha az hata

Bu eşiğin üstünde:
- QEC hata düzeltmekten daha fazla mantıksal hata üretir
- Daha fazla qubit = daha fazla hata

---

## Adım 1: Surface Code Temelleri

Surface code, veri qubit'lerini $d \times d = d^2$ boyutlu 2D bir ızgaraya yerleştirir ve 1 logical qubit kodlar. $d$ arttıkça düzeltilebilir hata ağırlığı $t = \lfloor(d-1)/2\rfloor$ artar:

```python
from quanta.qec.surface_code import SurfaceCode

# Farklı code distance'larda surface code
for d in [3, 5, 7]:
    code = SurfaceCode(distance=d)
    t = (d - 1) // 2  # Correctable errors
    print(f"Distance d={d}:")
    print(f"  Fiziksel qubit:  {code.n_physical}")
    print(f"  Logical qubit:   {code.n_logical}")
    print(f"  Düzeltilebilir:  {t} hata")
    print(f"  X stabilizer:    {code.n_syndrome_x}")
    print(f"  Z stabilizer:    {code.n_syndrome_z}")
    print()

print("💡 d=3: 9 veri qubiti ile 1 hatayı düzelt")
print("   d=5: 25 veri qubiti ile 2 hatayı düzelt")
print("   d=7: 49 veri qubiti ile 3 hatayı düzelt")
```

---

## Adım 2: Hata Düzeltme Simülasyonu

Farklı fiziksel hata oranlarında surface code'un ne kadar başarılı çalıştığını test edelim:

```python
from quanta.qec.surface_code import SurfaceCode

code = SurfaceCode(distance=3)

print("Surface Code d=3 — Code-Capacity Hata Oranı Taraması\n")
print(f"{'Fiziksel':>10} │ {'Logical':>10} │ {'Bastırma':>10} │ Durum")
print("─" * 56)

error_rates = [0.001, 0.005, 0.01, 0.03, 0.05, 0.08, 0.12, 0.15]

for p in error_rates:
    result = code.simulate_error_correction(error_rate=p, rounds=2000, seed=42)

    if result.logical_error_rate > 0:
        suppression = p / result.logical_error_rate
        sup_str = f"{suppression:.1f}x"
    else:
        sup_str = "∞"

    status = "✅ QEC bastırıyor" if result.logical_error_rate < p else "❌ Eşik yakını/üstü"
    if result.logical_error_rate == 0:
        status = "✅ Sıfır hata!"

    print(f"  {p:>8.4f} │ {result.logical_error_rate:>10.4f} │ {sup_str:>10} │ {status}")

print(f"\nCode-capacity eşik tahmini: {result.threshold_estimate:.1%}")
```

---

## Adım 3: Code Distance Karşılaştırma — Eşik Kanıtı

Asıl kanıt burada: **code-capacity eşiğinin (~%10.3) altında $d$ artınca logical hata düşer (`📉 iyileşiyor`), üstünde ise $d$ artınca logical hata artar (`📈 kötüleşiyor`).** Bu kesişim (crossover) eşiği deneysel olarak kanıtlar:

```python
from quanta.qec.surface_code import SurfaceCode

distances = [3, 5, 7]
error_rates = [0.01, 0.03, 0.05, 0.08, 0.12, 0.15]

print("Code Distance vs Hata Oranı Matrisi (Code-Capacity)\n")
header = f"{'p_phys':>8}"
for d in distances:
    header += f" │ {'d='+str(d):>10}"
header += " │ Trend"
print(header)
print("─" * 65)

for p in error_rates:
    row = f"  {p:>6.3f}"
    logicals = []

    for d in distances:
        code = SurfaceCode(distance=d)
        r = code.simulate_error_correction(error_rate=p, rounds=2000, seed=42)
        logicals.append(r.logical_error_rate)
        row += f" │ {r.logical_error_rate:>10.4f}"

    # Trend: d artınca logical düşüyor mu?
    if logicals[-1] < logicals[0]:
        trend = "📉 iyileşiyor"
    elif logicals[-1] > logicals[0]:
        trend = "📈 kötüleşiyor"
    else:
        trend = "➡️  sabit"

    row += f" │ {trend}"
    print(row)

print("\n🔑 Eşiğin altı (p < ~%10.3 code-capacity): d artınca logical error DÜŞER")
print("   Eşiğin üstü (p > ~%10.3 code-capacity): d artınca logical error ARTAR")
print("   (Devre seviyesi CNOT gürültüsünde aynı geçiş ~%1.1 civarında gerçekleşir.)")
```

---

## Adım 4: Hata Bastırma Grafiği

Eşiğin altında her distance artışı üstel iyileşme sağlar:

```python
from quanta.qec.surface_code import SurfaceCode
import math

# Eşiğin belirgin altında bir hata oranı: p = 0.02
p = 0.02
print(f"Hata Bastırma Analizi (p = {p:.3f})\n")
print(f"{'Distance':>10} │ {'Qubit':>6} │ {'Logical':>10} │ {'Bastırma':>10} │ Verimlilik")
print("─" * 65)

for d in [3, 5, 7]:
    code = SurfaceCode(distance=d)
    r = code.simulate_error_correction(error_rate=p, rounds=5000, seed=42)

    n_qubits = code.n_physical
    if r.logical_error_rate > 0:
        suppression = p / r.logical_error_rate
        sup_str = f"{suppression:>9.1f}x"
        efficiency = f"1:{n_qubits}"
    else:
        suppression = float("inf")
        sup_str = f"{'∞':>10}"
        efficiency = f"1:{n_qubits} (0 hata)"

    print(f"  d={d:>5} │ {n_qubits:>6} │ {r.logical_error_rate:>10.5f} │ {sup_str} │ {efficiency}")

print("\n💡 d=3→5→7: her adımda üstel iyileşme")
print("   Google Willow (2024): tam devre seviyesinde d=3→5→7 üstel bastırmayı deneysel gösterdi")
```

---

## Bu Neden Devrimsel?

2024 sonunda Google Quantum AI, **Willow çipinde** devre seviyesi (~%0.1–%0.3 kapı hatası, < ~%1.1 devre eşiği) altında tam olarak bu $d=3 \to 5 \to 7$ üstel bastırma eğrisini gösterdi:

| Rejim | Eşik ($p_{\text{th}}$) | $p < p_{\text{th}}$ Davranışı |
|---|---|---|
| Code-Capacity (`simulate_error_correction`) | ~%10.3 | $d=3 \to 5 \to 7$ üstel bastırma (`📉 iyileşiyor`) |
| 3D Phenomenological (`simulate_dynamic`) | ~%2.9 | Ölçüm gürültüsü altında uzay-zaman bastırma |
| Tam Devre Seviyesi (Google Willow donanımı) | ~%0.7 – %1.1 | $\Lambda \approx 2.14\times$ her distance artışında hata yarılanması |

---

## Sonuç

1. Surface code'un distance–qubit ilişkisini ($d^2$ veri qubiti) öğrendiniz
2. Code-capacity (~%10.3), 3D fenomenolojik (~%2.9) ve devre seviyesi (~%1.1) eşik farklarını incelediniz
3. Adım 3'te eşik altı (`📉 iyileşiyor`) ve eşik üstü (`📈 kötüleşiyor`) geçişini bizzat doğruladınız
4. Distance artışının eşik altında **üstel hata bastırma** sağladığını kanıtladınız
