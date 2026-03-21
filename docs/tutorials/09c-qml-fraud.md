# 🕵️ Sahte İşlem Tespiti: Quantum Machine Learning ile Fraud Detection

*Variational Quantum Classifier ile Kredi Kartı Dolandırıcılığını Yakalamak*

---

> **TL;DR:** Her yıl dünyada **32 milyar dolarlık** kredi kartı dolandırıcılığı
> gerçekleşiyor. Klasik ML modelleri bunu yakalamakta zorlanıyor çünkü
> sahte işlemler tüm işlemlerin %0.1'i — iğneyi samanlıkta aramak gibi.
> Quantum ML, verileri **üstel boyutlu** bir uzaya eşleyerek bunu daha iyi yapabilir.

---

## Problem: Neden Fraud Detection Bu Kadar Zor?

1. **Dengesiz veri:** 10.000 işlemin sadece 10'u sahte
2. **Değişen kalıplar:** Dolandırıcılar sürekli taktik değiştiriyor
3. **Gerçek zamanlı karar:** Her işlem 100ms'de onaylanmalı
4. **Yüksek boyutlu özellikler:** Konum, saat, tutar, sıklık, cihaz...

Quantum ML, özellikle **küçük ama yüksek boyutlu** veri setlerinde
klasik ML'den avantajlı. Tam olarak fraud detection'ın profili.

---

## Adım 1: Sentetik İşlem Verisi Oluşturma

Gerçekçi kredi kartı işlem verisi üretelim — iki sınıf:
normal ve sahte.

```python
import numpy as np

rng = np.random.default_rng(42)

# Normal işlemler: düşük tutar, gündüz, düşük sıklık
n_normal = 40
normal = np.column_stack([
    rng.normal(0.25, 0.1, n_normal),   # tutar (normalize)
    rng.normal(0.5, 0.15, n_normal),   # saat (gündüz)
    rng.normal(0.2, 0.1, n_normal),    # sıklık (düşük)
])

# Sahte işlemler: yüksek tutar, gece, yüksek sıklık
n_fraud = 20
fraud = np.column_stack([
    rng.normal(0.8, 0.1, n_fraud),     # tutar (yüksek)
    rng.normal(0.1, 0.1, n_fraud),     # saat (gece)
    rng.normal(0.8, 0.1, n_fraud),     # sıklık (yüksek)
])

# Birleştir ve [0,1] aralığına kırp
X = np.clip(np.vstack([normal, fraud]), 0, 1)
y = np.array([0]*n_normal + [1]*n_fraud)

# Karıştır
idx = rng.permutation(len(X))
X, y = X[idx], y[idx]

# Train/Test ayır (ilk 40 train, son 20 test)
X_train, y_train = X[:40], y[:40]
X_test, y_test = X[40:], y[40:]

print(f"Eğitim seti: {len(X_train)} işlem ({sum(y_train==1)} sahte)")
print(f"Test seti:   {len(X_test)} işlem ({sum(y_test==1)} sahte)")
print(f"Özellikler:  tutar, saat, sıklık")
```

---

## Adım 2: Quantum Classifier ile Eğitim

Quanta'nın `QuantumClassifier`'ı sklearn gibi kullanılıyor:
`fit()`, `predict()`, `score()`. Ama arkada quantum devreleri çalışıyor.

```python
import numpy as np
from quanta.layer3.qml import QuantumClassifier

rng = np.random.default_rng(42)

# Veri üret (aynı seed ile)
n_normal, n_fraud = 40, 20
normal = np.column_stack([
    rng.normal(0.25, 0.1, n_normal),
    rng.normal(0.5, 0.15, n_normal),
    rng.normal(0.2, 0.1, n_normal),
])
fraud = np.column_stack([
    rng.normal(0.8, 0.1, n_fraud),
    rng.normal(0.1, 0.1, n_fraud),
    rng.normal(0.8, 0.1, n_fraud),
])
X = np.clip(np.vstack([normal, fraud]), 0, 1)
y = np.array([0]*n_normal + [1]*n_fraud)
idx = rng.permutation(len(X))
X, y = X[idx], y[idx]
X_train, y_train = X[:40], y[:40]
X_test, y_test = X[40:], y[40:]

# 3 qubit = 3 özellik (tutar, saat, sıklık)
clf = QuantumClassifier(
    n_qubits=3,
    n_layers=2,
    feature_map="angle",
    learning_rate=0.3,
    seed=42,
)

# Eğit
result = clf.fit(X_train, y_train, epochs=15)

print(f"Quantum Classifier Sonuçları:")
print(f"  Qubit sayısı:    {result.n_qubits}")
print(f"  Parametre sayısı: {result.n_params}")
print(f"  Eğitim doğruluğu: {result.accuracy:.0%}")
print(f"  İlk loss:  {result.loss_history[0]:.3f}")
print(f"  Son loss:  {result.loss_history[-1]:.3f}")

# Test
test_acc = clf.score(X_test, y_test)
print(f"  Test doğruluğu:  {test_acc:.0%}")
```

3 qubit, 12 eğitilebilir parametre, 15 epoch — ve fraud'ları
yakalayabilen bir quantum classifier'ınız var!

---

## Adım 3: Quantum Kernel — Farklı Bir Yaklaşım

Variational classifier yerine **Quantum Kernel** kullanabiliriz.
Bu, verileri quantum state space'e eşleyip benzerlik hesaplar:

```python
import numpy as np
from quanta.layer3.qml import QuantumKernel

rng = np.random.default_rng(42)

# Küçük bir veri seti (kernel matrix hesabı O(n²))
n_normal, n_fraud = 8, 4
normal = np.column_stack([
    rng.normal(0.25, 0.1, n_normal),
    rng.normal(0.5, 0.15, n_normal),
    rng.normal(0.2, 0.1, n_normal),
])
fraud = np.column_stack([
    rng.normal(0.8, 0.1, n_fraud),
    rng.normal(0.1, 0.1, n_fraud),
    rng.normal(0.8, 0.1, n_fraud),
])
X = np.clip(np.vstack([normal, fraud]), 0, 1)
y = np.array([0]*n_normal + [1]*n_fraud)

# Quantum kernel
kernel = QuantumKernel(n_qubits=3, feature_map="zz")
K = kernel.matrix(X)

print("Quantum Kernel Matrisi (12×12):\n")
print("Normal vs Normal (üst-sol blok — yüksek benzerlik):")
print(f"  Ortalama: {K[:n_normal, :n_normal].mean():.3f}")
print(f"\nFraud vs Fraud (alt-sağ blok — yüksek benzerlik):")
print(f"  Ortalama: {K[n_normal:, n_normal:].mean():.3f}")
print(f"\nNormal vs Fraud (çapraz — düşük benzerlik):")
print(f"  Ortalama: {K[:n_normal, n_normal:].mean():.3f}")
print(f"\n💡 Quantum kernel, normal ve sahte işlemleri ayırt edebiliyor!")
```

Normal-normal benzerliği yüksek, fraud-fraud benzerliği yüksek,
ama normal-fraud benzerliği düşük → **iyi bir sınıflandırma sinyali**.

---

## Adım 4: Olasılık Tahmini — "Bu İşlem Ne Kadar Şüpheli?"

Gerçek dünyada sadece "sahte/değil" demek yetmez.
**Şüphe skoru** vermek lazım — banka buna göre önlem alır:

```python
import numpy as np
from quanta.layer3.qml import QuantumClassifier

rng = np.random.default_rng(42)

# Eğitim verisi oluştur
n_normal, n_fraud = 30, 15
normal = np.column_stack([
    rng.normal(0.25, 0.1, n_normal),
    rng.normal(0.5, 0.15, n_normal),
    rng.normal(0.2, 0.1, n_normal),
])
fraud = np.column_stack([
    rng.normal(0.8, 0.1, n_fraud),
    rng.normal(0.1, 0.1, n_fraud),
    rng.normal(0.8, 0.1, n_fraud),
])
X_train = np.clip(np.vstack([normal, fraud]), 0, 1)
y_train = np.array([0]*n_normal + [1]*n_fraud)

# Eğit
clf = QuantumClassifier(n_qubits=3, n_layers=2, feature_map="angle", seed=42)
clf.fit(X_train, y_train, epochs=10)

# Yeni işlemler değerlendir
new_transactions = np.array([
    [0.2, 0.6, 0.1],   # Normal: düşük tutar, gündüz, nadiren
    [0.9, 0.05, 0.9],  # Şüpheli: yüksek tutar, gece yarısı, çok sık
    [0.5, 0.3, 0.5],   # Belirsiz: orta tutar, akşam, orta sıklık
    [0.1, 0.7, 0.15],  # Normal: çok düşük tutar, öğlen, nadir
])

labels = ["Marketten alışveriş", "Gece yüksek transfer", 
          "Online alışveriş", "Kahveci ödemesi"]

probas = clf.predict_proba(new_transactions)

print("İşlem Risk Analizi:\n")
print(f"{'İşlem':<22} │ {'Tutar':>5} │ {'Saat':>5} │ {'Sıklık':>6} │ {'Risk':>6} │ Karar")
print("─" * 75)

for i, (label, tx) in enumerate(zip(labels, new_transactions)):
    risk = probas[i][1]  # P(fraud)
    if risk > 0.7:
        decision = "🚫 BLOKE ET"
    elif risk > 0.4:
        decision = "⚠️  SMS Doğrula"
    else:
        decision = "✅ Onayla"
    print(f"{label:<22} │ {tx[0]:>5.2f} │ {tx[1]:>5.2f} │ {tx[2]:>6.2f} │ {risk:>5.1%} │ {decision}")
```

---

## Quantum ML vs Klasik ML: Nerede Avantajlı?

| Durum | Klasik ML | Quantum ML |
|-------|-----------|------------|
| Büyük veri (>1M) | ✅ Daha hızlı | ❌ Qubit limiti |
| Küçük veri + yüksek boyut | ❌ Overfitting riski | ✅ Üstel feature space |
| Gerçek zamanlı | ✅ ms seviyesi | ⚠️ Henüz yavaş |
| Karmaşık kalıplar | ✅ İyi ama sınırlı | ✅ Quantum kernel avantajı |

> **Bugünkü quantum ML** küçük veri setlerinde quantum kernel ile avantaj sağlıyor.
> Google'ın 2024 çalışması: quantum kernel, bazı yapısal veri setlerinde
> klasik kernel'lerden üstün performans gösterdi.

---

## Sonuç

Bu tutorialda:

1. **Sentetik fraud verisi** oluşturdunuz (tutar, saat, sıklık)
2. **QuantumClassifier** ile variational quantum sınıflandırma yaptınız
3. **QuantumKernel** ile quantum benzerlik matrisi hesapladınız
4. **Risk skoru** ile gerçek dünya karar sistemi simüle ettiniz

Aynı mimari farklı alanlara uygulanabilir: tıbbi teşhis, müşteri kaybı
tahmini, ürün önerisi, siber saldırı tespiti...

### Sonraki Adımlar

- [Shor ile RSA Kırma](09a-shor-cryptography.md) — Gerçek hayat problemi #1
- [QAOA ile Lojistik](09b-qaoa-logistics.md) — Gerçek hayat problemi #2
