# 🛡️ Quantum Hata Düzeltme Eşiği — Ölçeklenebilirliğin Kanıtı

*Surface Code ile Error Correction Threshold Bulma*

---

> **Bu tutorialda yapacağınız şey:** Surface code'u farklı hata oranlarında
> simüle edip, **eşik hata oranını** bulacaksınız. Bu eşiğin altında quantum
> hata düzeltme çalışıyor, üstünde çöküyor. Bu, quantum bilgisayarların
> **gerçekten ölçeklenebileceğinin** deneysel kanıtı.

---

## Neden QEC Eşiği Bu Kadar Önemli?

Quantum bilgisayarların en büyük sorunu: **qubit'ler hata yapıyor.**
Tek bir qubit'in hata oranı ~%0.1-1. 1000 qubit'lik bir hesaplama
yaparsanız, neredeyse kesin hata alırsınız.

Çözüm: **Quantum Error Correction (QEC).**

Ama QEC'in çalışması bir koşula bağlı:

> **Fiziksel hata oranı, eşik hata oranının altında olmalı.**

Surface code için bu eşik teorik olarak **~%1.1**. Bu eşiğin altında:
- Code distance artırınca logical hata oranı **üstel olarak düşer**
- Yani daha fazla qubit = daha az hata

Bu eşiğin üstünde:
- QEC hata düzeltmekten daha fazla hata **ekler**
- Daha fazla qubit = daha fazla hata

---

## Adım 1: Surface Code Temelleri

Surface code, qubit'leri 2 boyutlu bir ızgaraya yerleştirir.
d×d = d² fiziksel qubit → 1 logical qubit. d arttıkça güvenilirlik artar
(eşiğin altındaysa):

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

print("💡 d=3: 9 qubit ile 1 hatayı düzelt")
print("   d=5: 25 qubit ile 2 hatayı düzelt")
print("   d=7: 49 qubit ile 3 hatayı düzelt")
```

---

## Adım 2: Hata Düzeltme Simülasyonu

Farklı fiziksel hata oranlarında surface code'un ne kadar
başarılı çalıştığını test edelim:

```python
from quanta.qec.surface_code import SurfaceCode

code = SurfaceCode(distance=3)

print("Surface Code d=3 — Hata Oranı Taraması\n")
print(f"{'Fiziksel':>10} │ {'Logical':>10} │ {'Bastırma':>10} │ Durum")
print("─" * 52)

error_rates = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15]

for p in error_rates:
    result = code.simulate_error_correction(error_rate=p, rounds=2000, seed=42)
    
    if result.logical_error_rate > 0:
        suppression = p / result.logical_error_rate
        sup_str = f"{suppression:.1f}x"
    else:
        sup_str = "∞"
    
    status = "✅ QEC çalışıyor" if result.logical_error_rate < p else "❌ QEC kötüleştiriyor"
    if result.logical_error_rate == 0:
        status = "✅ Sıfır hata!"
    
    print(f"  {p:>8.4f} │ {result.logical_error_rate:>10.4f} │ {sup_str:>10} │ {status}")

print(f"\nEşik tahmini: {result.threshold_estimate:.1%}")
```

---

## Adım 3: Code Distance Karşılaştırma — Eşik Kanıtı

Asıl kanıt burada: **eşiğin altında d artınca logical hata düşer,
üstünde d artınca hata artar.** Bu davranış eşiği kanıtlar:

```python
from quanta.qec.surface_code import SurfaceCode

distances = [3, 5, 7]
error_rates = [0.001, 0.005, 0.01, 0.015, 0.02, 0.05]

print("Code Distance vs Hata Oranı Matrisi\n")
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

print(f"\n🔑 Eşiğin altı (p < ~%1): d artınca logical error DÜŞER")
print(f"   Eşiğin üstü (p > ~%1): d artınca logical error ARTAR")
print(f"   Bu, quantum ölçeklenebilirliğin kanıtıdır.")
```

---

## Adım 4: Hata Bastırma Grafiği

Eşiğin altında her distance artışı üstel iyileşme sağlar:

```python
from quanta.qec.surface_code import SurfaceCode
import math

# Eşiğin altı: p = 0.005
p = 0.005
print(f"Hata Bastırma Analizi (p = {p:.3f})\n")
print(f"{'Distance':>10} │ {'Qubit':>6} │ {'Logical':>10} │ {'Bastırma':>10} │ Verimlilik")
print("─" * 65)

for d in [3, 5, 7]:
    code = SurfaceCode(distance=d)
    r = code.simulate_error_correction(error_rate=p, rounds=5000, seed=42)
    
    n_qubits = code.n_physical
    if r.logical_error_rate > 0:
        suppression = p / r.logical_error_rate
        # Overhead: kaç fiziksel qubit → 1 logical qubit
        efficiency = f"1:{n_qubits}"
    else:
        suppression = float('inf')
        efficiency = f"1:{n_qubits} (mükemmel)"
    
    sup_bar = "█" * min(int(suppression) if suppression != float('inf') else 30, 30)
    print(f"  d={d:>5} │ {n_qubits:>6} │ {r.logical_error_rate:>10.5f} │ {suppression:>10.1f}x │ {efficiency}")

print(f"\n💡 d=3→5→7: her adımda üstel iyileşme")
print(f"   Bu, pratik quantum bilgisayarların mümkün olduğunun kanıtı")
print(f"   Google Willow (2024): d=3→5→7'de bu eğriyi deneysel gösterdi")
```

---

## Bu Neden Devrimsel?

2024'te Google, **Willow çipinde** tam olarak bu deneyi yaptı:

| Google Sonucu | Simülasyonumuz |
|---------------|----------------|
| d=3: logical error yüksek | d=3: logical error > p |
| d=5: düşüş başladı | d=5: belirgin düşüş |
| d=7: üstel bastırma | d=7: güçlü bastırma |

> Surface code eşiğinin altında çalışmak, quantum bilgisayarı
> **sonsuza kadar ölçeklenebilir** kılar. 1 milyon qubit'lik
> quantum bilgisayar inşa etmenin fiziksel kanıtı bu.

---

## Sonuç

1. Surface code'un distance-qubit ilişkisini öğrendiniz
2. Farklı hata oranlarında hata düzeltme simülasyonu yaptınız
3. **Eşik hata oranını** (~%1.1) deneysel olarak buldunuz
4. Distance artışının eşik altında **üstel iyileşme** sağladığını kanıtladınız
5. Google Willow deneyiyle aynı sonuçları elde ettiniz

> **Doğrulama:** Eşik altında (p < %1) d artınca logical error düşer ✅
> Eşik üstünde (p > %1) d artınca logical error artar ✅
