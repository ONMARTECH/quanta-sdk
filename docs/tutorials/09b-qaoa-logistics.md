# 🚚 8 Teslimat, 1 Kamyon: Quantum ile Rota Optimizasyonu

*QAOA ile Lojistik ve Kombinatoryal Optimizasyon*

---

> **TL;DR:** Bir kargo şirketinin 8 teslimat noktası var. Hangi noktaları
> bir turdaki kamyona atasın ki toplam maliyet minimum olsun? Klasik bilgisayar
> 2⁸ = 256 kombinasyonu tek tek deniyor. Quantum bilgisayar?
> **Hepsini aynı anda değerlendiriyor.**

---

## Problem: Neden Bu Kadar Zor?

Kombinatoryal optimizasyon problemi her yerde:
- **Lojistik:** Hangi depolardan hangi müşterilere gönderim yapılsın?
- **Finans:** Hangi 5 hisseyi portföye koyalım?
- **Telekom:** Baz istasyonu frekans ataması
- **Üretim:** Makine-iş sıralaması

Hepsinin ortak noktası: seçenek sayısı **üstel** artıyor.
8 nokta → 256 seçenek. 20 nokta → 1 milyon. 50 nokta → 10¹⁵.

**QAOA (Quantum Approximate Optimization Algorithm)** bu problemi
quantum süperpozisyonla çözer.

---

## Adım 1: En Basit Problem — Bit Seçimi

3 bit'ten en düşük maliyetli kombinasyonu bulalım.
Her bit "aktif" veya "pasif" bir kararı temsil ediyor:

```python
from quanta.layer3.optimize import optimize

# Maliyet fonksiyonu: her bitin bir fiyatı var
# Bit 0 = 3₺, Bit 1 = 5₺, Bit 2 = 7₺
# Ceza: en az 1 bit aktif olmalı (yoksa +100₺)
def delivery_cost(x):
    bit0 = (x >> 0) & 1  # Kadıköy deposu
    bit1 = (x >> 1) & 1  # Şişli deposu
    bit2 = (x >> 2) & 1  # Beşiktaş deposu
    
    cost = 3 * bit0 + 5 * bit1 + 7 * bit2
    
    # En az 1 depo açık olmalı
    if bit0 + bit1 + bit2 == 0:
        cost += 100
    
    return cost

result = optimize(num_bits=3, cost=delivery_cost, minimize=True, seed=42)

print("Quantum sonucu:")
print(f"  En iyi: |{result.best_bitstring}⟩ → {result.best_cost}₺")
print(f"\nTüm çözümler (olasılıklara göre):")
for bits, cost, prob in result.all_solutions[:5]:
    bar = "█" * int(prob * 40)
    aktif = [["Kadıköy","Şişli","Beşiktaş"][i] for i in range(3) if (int(bits,2)>>i)&1]
    print(f"  |{bits}⟩ {','.join(aktif) or '-':>20} → {cost:>3}₺  P={prob:.2f} {bar}")
```

Quantum bilgisayar en ucuz seçeneği buldu: sadece Kadıköy deposu (3₺).

---

## Adım 2: Depo Atama Problemi — 5 Şehir, Maliyet Matrisi

Bir lojistik firma 5 şehirde depo açmayı düşünüyor. Her deponun açılış
maliyeti ve müşterilere uzaklık maliyeti farklı. **Tam olarak 2 depo** açılmalı:

```python
from quanta.layer3.optimize import optimize

# 5 şehir: [İstanbul, Ankara, İzmir, Bursa, Antalya]
# Her deponun açılış maliyeti (bin ₺)
opening_cost = [50, 30, 40, 25, 35]

# Müşterilere ortalama nakliye maliyeti (bin ₺)
shipping_cost = [10, 20, 15, 12, 25]

def warehouse_cost(x):
    selected = [(x >> i) & 1 for i in range(5)]
    n_open = sum(selected)
    
    # Tam 2 depo açılmalı (kısıt: ceza)
    penalty = 50 * (n_open - 2) ** 2
    
    # Toplam maliyet: açılış + nakliye
    total = sum(
        (opening_cost[i] + shipping_cost[i]) * selected[i]
        for i in range(5)
    )
    
    return total + penalty

result = optimize(num_bits=5, cost=warehouse_cost, minimize=True, layers=2, seed=42)

cities = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"]
print("Depo Atama Sonuçları:\n")
print(f"{'Kombinasyon':>12} │ {'Şehirler':>25} │ {'Maliyet':>8}")
print("─" * 55)

for bits, cost, prob in result.all_solutions[:6]:
    if prob < 0.01:
        continue
    selected = [cities[i] for i in range(5) if (int(bits,2)>>i)&1]
    print(f"  |{bits}⟩      │ {', '.join(selected):>25} │ {cost:>6.0f}₺")

best_cities = [cities[i] for i in range(5) if (int(result.best_bitstring,2)>>i)&1]
print(f"\n✅ Optimal: {' + '.join(best_cities)} → {result.best_cost:.0f}₺")
```

---

## Adım 3: Max-Cut — Ağ Bölümleme

Telekom şirketinin 4 baz istasyonu var. Birbirine yakın istasyonlar
aynı frekansı **kullanamaz** (girişim olur). Ağı ikiye böl ki
kenar sayısı (çakışma riski) minimum olsun:

```python
from quanta.layer3.optimize import optimize

# 4 istasyon, aralarındaki bağlantılar (kenarlar)
# (0,1), (0,2), (1,2), (1,3), (2,3)
edges = [(0,1), (0,2), (1,2), (1,3), (2,3)]

def maxcut_cost(x):
    """Kesilmiş kenar sayısı (maximize etmek istiyoruz)."""
    cut = 0
    for i, j in edges:
        bi = (x >> i) & 1
        bj = (x >> j) & 1
        if bi != bj:  # Farklı grupta → kenar kesildi
            cut += 1
    return cut

result = optimize(
    num_bits=4,
    cost=maxcut_cost,
    minimize=False,  # Maximize: mümkün olduğunca çok kenar kes
    layers=2,
    seed=42,
)

print("Max-Cut (Ağ Bölümleme):\n")
print(f"En iyi kesim: |{result.best_bitstring}⟩ → {int(result.best_cost)} kenar\n")

# Grupları göster
best = int(result.best_bitstring, 2)
group_a = [f"İstasyon-{i}" for i in range(4) if (best >> i) & 1]
group_b = [f"İstasyon-{i}" for i in range(4) if not (best >> i) & 1]
print(f"  Grup A (Frekans 1): {', '.join(group_a)}")
print(f"  Grup B (Frekans 2): {', '.join(group_b)}")
```

Bu tam olarak **Google, IBM ve diğer quantum şirketlerinin** benchmark
olarak kullandığı problem. QAOA bu problemde klasik algoritmalarla
yarışıyor — ve qubit sayısı arttıkça avantaj büyüyor.

---

## Adım 4: Portföy Seçimi — 6 Hisseden 3'ünü Seç

Finans dünyasından bir varyasyon. 6 hisse var, tam olarak 3 tanesini
portföye almak istiyorsun. Risk minimize, getiri maximize:

```python
from quanta.layer3.optimize import optimize

# 6 hisse: [THYAO, ASELS, BIMAS, SASA, KOZAA, TUPRS]
expected_return = [0.12, 0.15, 0.08, 0.20, 0.18, 0.10]
risk = [0.25, 0.30, 0.10, 0.40, 0.35, 0.15]

def portfolio_score(x):
    selected = [(x >> i) & 1 for i in range(6)]
    n = sum(selected)
    
    # Tam 3 hisse seçilmeli
    penalty = 20 * (n - 3) ** 2
    
    # Skor: getiri - 0.5*risk (Sharpe-benzeri)
    total_return = sum(expected_return[i] * selected[i] for i in range(6))
    total_risk = sum(risk[i] * selected[i] for i in range(6))
    
    # Minimize edeceğiz, o yüzden negatif skor
    score = -(total_return - 0.5 * total_risk) + penalty
    return score

result = optimize(num_bits=6, cost=portfolio_score, minimize=True, layers=2, seed=42)

stocks = ["THYAO", "ASELS", "BIMAS", "SASA", "KOZAA", "TUPRS"]
print("Portföy Optimizasyonu:\n")

for bits, cost, prob in result.all_solutions[:5]:
    if prob < 0.01:
        continue
    selected = [stocks[i] for i in range(6) if (int(bits,2)>>i)&1]
    n = len(selected)
    marker = "👈 optimal" if bits == result.best_bitstring else ""
    print(f"  |{bits}⟩  {'+'.join(selected):>20}  ({n} hisse)  {marker}")

best_stocks = [stocks[i] for i in range(6) if (int(result.best_bitstring,2)>>i)&1]
print(f"\n✅ Quantum portföy: {', '.join(best_stocks)}")
```

---

## Neden QAOA Önemli?

| | Klasik (Brute Force) | QAOA |
|---|---|---|
| 10 değişken | 1,024 deneme | ~10 iterasyon |
| 20 değişken | 1,048,576 deneme | ~20 iterasyon |
| 50 değişken | 10¹⁵ deneme | ~50 iterasyon |
| Ölçekleme | O(2ⁿ) | O(poly(n)) |

> **Anahtar insight:** QAOA tüm kombinasyonları **aynı anda** değerlendiriyor
> (quantum süperpozisyon) ve varasyonel parametrelerle en iyisini öne çıkarıyor.

---

## Sonuç

Bu tutorialda:

1. **Bit seçimi** ile temel QAOA kullanımını gördünüz
2. **Depo atama** problemiyle kısıtlı optimizasyon yaptınız
3. **Max-Cut** ile ağ bölümleme (telekomda frekans ataması) çözdünüz
4. **Portföy seçimi** ile finans uygulaması gördünüz

Hepsi aynı `optimize()` fonksiyonuyla — sadece cost fonksiyonu değişiyor.

### Sonraki Adımlar

- [Shor ile RSA Kırma](09a-shor-cryptography.md) — Gerçek hayat problemi #1

