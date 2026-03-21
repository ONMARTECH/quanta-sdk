# 🔔 Bell Eşitsizliği — Quantum'un Klasikten Farkını Matematiksel Olarak Kanıtla

*CHSH Deneyi: Hiçbir Klasik Bilgisayarın Taklit Edemeyeceği Bir Sonuç*

---

> **Bu tutorialda yapacağınız şey:** 1964'te John Bell'in ortaya koyduğu
> eşitsizliği quantum devrelerle test edeceksiniz. Eğer S > 2 çıkarsa,
> bu sonucu **hiçbir klasik algoritma, hiçbir gizli değişken teorisi üretemez.**
> Quantum mekaniğinin en temel kanıtı.

---

## Arka Plan: Einstein Haklı mıydı?

Einstein, quantum dolanıklığın "ürkütücü uzaktan etki" (spooky action at
a distance) olduğunu söyleyip itiraz etti. Dedi ki: *"Partiküller ayrılmadan
önce sonucu zaten biliyorlar — gizli değişkenler var."*

1964'te John Bell bir eşitsizlik türetti:

> **Eğer gizli değişkenler varsa, S ≤ 2 olmalı.**  
> **Quantum mekaniği ise S = 2√2 ≈ 2.828 öngörüyor.**

2022 Nobel Fizik Ödülü, bu deneyi gerçekleştirenalere verildi (Aspect,
Clauser, Zeilinger). Şimdi aynı deneyi **Quanta ile** yapacağız.

---

## Adım 1: Bell State Oluştur ve Dolanıklığı Gözlemle

İlk adım: iki qubit'i dolanık hale getir. Bu, quantum bilgisayarların
yapabildiği ama klasik bilgisayarların **yapamadığı** şey:

```python
from quanta.simulator.statevector import StateVectorSimulator

# |Φ+⟩ Bell state oluştur
sim = StateVectorSimulator(2)
sim.apply("H", (0,))      # Qubit 0'ı süperpozisyona al
sim.apply("CX", (0, 1))   # CNOT ile dolanıklık kur

# Statevector'ı incele
state = sim.state
print("Bell State |Φ+⟩:")
print(f"  |00⟩ genliği: {state[0]:.4f}")
print(f"  |01⟩ genliği: {state[1]:.4f}")
print(f"  |10⟩ genliği: {state[2]:.4f}")
print(f"  |11⟩ genliği: {state[3]:.4f}")
print(f"\n  Yorumlama: Ölçersen ya |00⟩ ya da |11⟩ çıkar")
print(f"  Her ikisi de %50 olasılıklı — AMA her zaman aynı!")
print(f"  Bu korelasyon ürkütücü: biri 0 verirse diğeri kesinlikle 0.")

# Dolanıklık kanıtı: örnekleme
counts = sim.sample(1000)
print(f"\n  1000 ölçüm: {counts}")
same = sum(v for k, v in counts.items() if k[0] == k[1])
print(f"  Aynı sonuç oranı: {same/1000:.1%} (klasikte max %75)")
```

!!! note "Simülasyon Notu"
    Bu devre gerçek quantum donanımında (IBM, Google) bire bir aynı
    şekilde çalışır. Simülatörümüz aynı quantum gate'lerini kullanıyor —
    fark sadece statevector'ın bellekte tutulması.

---

## Adım 2: CHSH Deneyi — Farklı Açılarda Ölçüm

CHSH deneyinde Alice ve Bob farklı açılarda ölçüm yapar.
4 farklı açı kombinasyonu, 4 korelasyon değeri:

```python
import numpy as np
from quanta.simulator.statevector import StateVectorSimulator

def measure_correlation(alice_angle, bob_angle):
    """Bell state'i farklı bazlarda ölçüp korelasyon hesapla.
    
    Devre: H → CX → RY(alice) ⊗ RY(bob) → ölçüm
    Quantum mekaniği: E(a,b) = cos(2(a-b))
    """
    sim = StateVectorSimulator(2)
    
    # Bell state |Φ+⟩
    sim.apply("H", (0,))
    sim.apply("CX", (0, 1))
    
    # Ölçüm bazını döndür (RY ile Z-bazından θ-bazına)
    sim.apply("RY", (0,), (-2 * alice_angle,))
    sim.apply("RY", (1,), (-2 * bob_angle,))
    
    # Olasılıkları hesapla
    probs = sim.probabilities()
    # E = P(aynı sonuç) - P(farklı sonuç)
    E = float((probs[0] + probs[3]) - (probs[1] + probs[2]))
    return E

# CHSH optimal açıları (maximum Bell violation)
# Alice: 0 ve π/4    Bob: π/8 ve 3π/8
a0, a1 = 0, np.pi/4
b0, b1 = np.pi/8, 3*np.pi/8

E00 = measure_correlation(a0, b0)
E01 = measure_correlation(a0, b1)
E10 = measure_correlation(a1, b0)
E11 = measure_correlation(a1, b1)

print("CHSH Korelasyonları:")
print(f"  E(a₀=0,    b₀=π/8)  = {E00:+.4f}  (teori: {np.cos(2*(a0-b0)):+.4f})")
print(f"  E(a₀=0,    b₁=3π/8) = {E01:+.4f}  (teori: {np.cos(2*(a0-b1)):+.4f})")
print(f"  E(a₁=π/4,  b₀=π/8)  = {E10:+.4f}  (teori: {np.cos(2*(a1-b0)):+.4f})")
print(f"  E(a₁=π/4,  b₁=3π/8) = {E11:+.4f}  (teori: {np.cos(2*(a1-b1)):+.4f})")

# CHSH S parametresi
S = E00 - E01 + E10 + E11

print(f"\n  S = E₀₀ - E₀₁ + E₁₀ + E₁₁")
print(f"  S = {E00:+.4f} - ({E01:+.4f}) + {E10:+.4f} + {E11:+.4f}")
print(f"  S = {S:.4f}")
```

---

## Adım 3: Bell Eşitsizliğini Test Et

Şimdi kritik an: S > 2 mi?

```python
import numpy as np
from quanta.simulator.statevector import StateVectorSimulator

def measure_correlation(a, b):
    sim = StateVectorSimulator(2)
    sim.apply("H", (0,))
    sim.apply("CX", (0, 1))
    sim.apply("RY", (0,), (-2 * a,))
    sim.apply("RY", (1,), (-2 * b,))
    probs = sim.probabilities()
    return float((probs[0] + probs[3]) - (probs[1] + probs[2]))

a0, a1, b0, b1 = 0, np.pi/4, np.pi/8, 3*np.pi/8
S = measure_correlation(a0,b0) - measure_correlation(a0,b1) + measure_correlation(a1,b0) + measure_correlation(a1,b1)

bell_limit = 2.0
quantum_prediction = 2 * np.sqrt(2)

print("╔══════════════════════════════════════════════╗")
print("║       BELL EŞİTSİZLİĞİ TEST SONUCU         ║")
print("╠══════════════════════════════════════════════╣")
print(f"║  Ölçülen S değeri:    {S:.4f}                 ║")
print(f"║  Bell sınırı (klasik): {bell_limit:.4f}                 ║")
print(f"║  Quantum öngörüsü:    {quantum_prediction:.4f}                 ║")
print("╠══════════════════════════════════════════════╣")

if abs(S) > bell_limit:
    violation = abs(S) - bell_limit
    print(f"║  ✅ BELL EŞİTSİZLİĞİ İHLAL EDİLDİ!         ║")
    print(f"║  İhlal miktarı: {violation:.4f}                    ║")
    print(f"║                                              ║")
    print(f"║  Bu sonuç kanıtlar ki:                       ║")
    print(f"║  → Gizli değişken teorileri YANLIŞ            ║")
    print(f"║  → Quantum dolanıklık GERÇEK                  ║")
    print(f"║  → Hiçbir klasik sistem bunu ÜRETEMEZ         ║")
else:
    print(f"║  ❌ Bell eşitsizliği ihlal edilmedi            ║")
print("╚══════════════════════════════════════════════╝")
```

---

## Adım 4: İstatistiksel Güvenilirlik — Sonlu Örneklemle Test

Gerçek laboratuvarda sonsuz ölçüm yapılamaz. Sonlu shot sayısıyla
Bell violation hâlâ istatistiksel olarak anlamlı mı?

```python
import numpy as np
from quanta.simulator.statevector import StateVectorSimulator

def chsh_with_shots(shots, seed=42):
    """CHSH deneyini sonlu örneklemle yap."""
    a0, a1 = 0, np.pi/4
    b0, b1 = np.pi/8, 3*np.pi/8
    
    correlations = []
    for a, b in [(a0,b0), (a0,b1), (a1,b0), (a1,b1)]:
        sim = StateVectorSimulator(2, seed=seed)
        sim.apply("H", (0,))
        sim.apply("CX", (0, 1))
        sim.apply("RY", (0,), (-2*a,))
        sim.apply("RY", (1,), (-2*b,))
        
        counts = sim.sample(shots)
        same = sum(v for k,v in counts.items() if k[0]==k[1])
        diff = shots - same
        E = (same - diff) / shots
        correlations.append(E)
    
    S = correlations[0] - correlations[1] + correlations[2] + correlations[3]
    return S

print("Shot sayısına göre CHSH S değeri:\n")
print(f"{'Shots':>10} │ {'S değeri':>10} │ {'|S|>2?':>8} │ Güven")
print("─" * 50)

for shots in [100, 500, 1000, 5000, 10000, 50000]:
    # Birden fazla deneme yap
    S_values = [chsh_with_shots(shots, seed=s) for s in range(10)]
    S_mean = np.mean(S_values)
    S_std = np.std(S_values)
    violation = "✅" if abs(S_mean) > 2 else "❌"
    sigma = (abs(S_mean) - 2) / max(S_std, 1e-10)
    
    print(f"  {shots:>8} │ {S_mean:>+10.4f} │ {violation:>8} │ {sigma:.1f}σ")

print(f"\n💡 Ne kadar çok ölçüm → o kadar kesin Bell violation")
print(f"   Gerçek laboratuvarlarda >5σ güven seviyesi aranır")
```

---

## Quantum vs Klasik: Neden S > 2 İmkansız?

Bell'in kanıtı (1964):

> Eğer ölçüm sonuçları **önceden belirlenmiş** (deterministic) ise,
> yani Alice ve Bob'un partikülleri ayrılmadan önce "ne vereceğini
> biliyorsa", o zaman herhangi bir korelasyon fonksiyonu için
> **|S| ≤ 2** olmak zorundadır.

Quantum mekaniği bunu ihlal ediyor çünkü:

- Sonuçlar ölçümden **önce** belirlenmiş değil
- Dolanık partiküller **tek bir quantum durumu** paylaşıyor
- Ölçüm, durumu **anlık olarak** çökertiyor

> **2022 Nobel Ödülü** bu deneyi yapan Aspect, Clauser ve Zeilinger'a verildi.
> Siz de az önce aynı deneyi Quanta SDK ile yaptınız.

---

## Sonuç

1. Bell state |Φ+⟩ oluşturup dolanıklığı gözlemlediniz
2. CHSH deneyini 4 farklı açıda çalıştırdınız
3. **S = 2.828 > 2** → Bell eşitsizliği ihlal edildi
4. Bu, **hiçbir klasik bilgisayarın taklit edemeyeceği** bir sonuç
5. Sonlu shot ile istatistiksel güvenilirliği doğruladınız

> **Doğrulama:** S = 2√2 = 2.8284... (quantum teorisi ile tam uyum)
