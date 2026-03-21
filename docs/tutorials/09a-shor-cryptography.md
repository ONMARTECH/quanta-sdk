# 🔓 Neden Şifreleriniz 10 Yıl İçinde Güvensiz?

*Shor's Algorithm ile RSA Kriptografisini Kırmak*

---

> **TL;DR:** Bugün bankacılık, e-ticaret ve devlet sırları RSA şifrelemesiyle korunuyor.
> RSA'nın güvenliği tek bir varsayıma dayanıyor: *büyük sayıları çarpanlarına
> ayırmak klasik bilgisayarlarla pratik olarak imkansız.* Quantum bilgisayarlar
> bu varsayımı yıkıyor. Bu tutorialda, bunu bizzat göreceksiniz.

---

## Hikaye: RSA Nasıl Çalışır?

Birisi size şifreli mesaj göndermek istediğinde:

1. İki **büyük asal sayı** seçersiniz: `p = 61`, `q = 53`
2. Bunları çarparsınız: `N = p × q = 3233`
3. `N` değerini herkesle paylaşırsınız (**public key**)
4. `p` ve `q` değerlerini saklarsınız (**private key**)

Güvenlik şuradan gelir: `N = 3233` biliniyor, ama `3233 = ? × ?` sorusunu cevaplamak
klasik bilgisayarla çok zor. RSA-2048 için bu sayı **617 haneli**.

**Peki ya quantum bilgisayar olsa?**

---

## Adım 1: RSA Sayılarını Kıralım

Shor's algorithm ile RSA'nın temelini oluşturan "çarpanlara ayırma" problemini çözelim.
Küçük sayılarla başlayıp büyüyelim:

```python
from quanta.layer3.shor import factor

# Bir dizi RSA-benzeri sayı kır
test_cases = [15, 21, 35, 77, 143, 221, 323, 899]

print(f"{'N':>6} │ {'Çarpanlar':>12} │ Doğrulama")
print("─" * 42)

for N in test_cases:
    result = factor(N, seed=42)
    p, q = result.factors
    check = "✅" if p * q == N else "❌"
    print(f"{N:>6} │ {p:>4} × {q:<4}   │ {check} {p}×{q}={p*q}")
```

Her biri doğru kırıldı. Simülatörde küçük sayılar için Shor'un optimizasyonları
devreye giriyor — tıpkı gerçek bir quantum bilgisayarın klasik ön-işleme
yapması gibi. Önemli olan: **quantum bilgisayar büyüdüğünde aynı algoritma
RSA-2048'i de kıracak.**

---

## Adım 2: Perde Arkası — Quantum Fourier Transform

Shor's algorithm'ın gizli silahı **QFT (Quantum Fourier Transform)**.
Bir sayının periyodunu (tekrar döngüsünü) bulmak için kullanılır.

Quanta'nın Shor modülü gerçek QFT gate'leri kullanıyor — simülatörde
çalışıyor ama devre yapısı tamamen gerçek:

```python
from quanta.layer3.shor import _build_qft_dag

# 4-qubit QFT devresi oluştur
qft = _build_qft_dag(4)

# Devredeki gate'leri incele
gate_counts = {}
for op in qft.op_nodes():
    gate_counts[op.gate_name] = gate_counts.get(op.gate_name, 0) + 1

print("QFT-4 Devresi:")
for gate, count in sorted(gate_counts.items()):
    print(f"  {gate}: {count} adet")
print(f"  Toplam: {sum(gate_counts.values())} gate")
print(f"\n💡 QFT-n = O(n²) gate — klasik FFT'den üstel hızlı!")
```

4 qubit'lik QFT bile birden fazla gate katmanı içeriyor.
RSA-2048 kırmak için ~4000 qubit'lik QFT gerekir.

---

## Adım 3: Tam Prime Decomposition

Gerçek dünyada büyük sayılar birden fazla asalın çarpımı olabilir.
`factor_recursive` bunu tamamen çözer:

```python
from quanta.layer3.shor import factor_recursive

# Gerçek dünya sayıları
numbers = [
    (9999,  "kredi kartı son 4 hane"),
    (2025,  "bu yılın sayısı"),
    (1001,  "bin bir gece masalı"),
    (30030, "ilk 7 asalın çarpımı"),
]

for N, label in numbers:
    primes = factor_recursive(N, seed=42)
    decomp = " × ".join(str(p) for p in primes)
    print(f"{N:>6} ({label})")
    print(f"       = {decomp}")
    print()
```

---

## Adım 4: RSA Kırılma Zaman Çizelgesi

Peki gerçek dünyada nerede duruyoruz? Quantum bilgisayarlar ne zaman
RSA'yı kıracak kadar güçlü olacak?

```python
import math

rsa_sizes = [
    ("RSA-512",   512,  "1999'da kırıldı (klasik, 7 ay)"),
    ("RSA-768",   768,  "2009'da kırıldı (klasik, 2 yıl)"),
    ("RSA-1024", 1024,  "Güvensiz sayılıyor"),
    ("RSA-2048", 2048,  "Bugün standart — bankalar bunu kullanıyor"),
    ("RSA-4096", 4096,  "Yüksek güvenlik — devlet sırları"),
]

print("RSA Kırılma Haritası\n")
print(f"{'Standart':<12} │ {'Bit':>6} │ {'Shor Qubit':>12} │ Durum")
print("─" * 72)

for name, bits, status in rsa_sizes:
    qubits_needed = 2 * bits  # Shor ~2n logical qubits
    print(f"{name:<12} │ {bits:>6} │ {qubits_needed:>12} │ {status}")

print()
print("Quantum bilgisayar gelişimi:")
print("  2022  IBM Eagle       →    127 qubit")
print("  2023  IBM Condor      →  1,121 qubit")
print("  2025  IBM Kookaburra  →  4,158 qubit (hedef)")
print("  2030? Hata düzeltmeli → ~4,000 logical qubit")
print()
print("⚠️  RSA-2048 → 4,096 logical qubit + QEC overhead gerekli")
print("📅 Tahmini kırılma: 2030–2035")
```

---

## Adım 5: "Harvest Now, Decrypt Later" Tehdidi

Bugünün en büyük güvenlik tehdidi: devletler ve hackerlar **bugün şifreli
veriyi kaydedip, quantum bilgisayar hazır olunca kırabilir**.

Yani 2025'te gönderdiğiniz şifreli mesaj, 2033'te okunabilir.

```python
print("=" * 55)
print("  POST-QUANTUM GÜVENLİK KONTROL LİSTESİ")
print("=" * 55)

checks = [
    ("AES-256 (simetrik)",   "✅ Güvenli", "Grover 128-bit'e düşürür, yeterli"),
    ("RSA-2048",             "⚠️  Risk",   "Shor ile kırılacak (2030-35)"),
    ("ECDSA P-256",          "⚠️  Risk",   "Shor ile kırılacak"),
    ("SHA-256 / SHA-3",      "✅ Güvenli", "Quantum etkisi minimal"),
    ("ML-KEM (Kyber)",       "✅ Güvenli", "NIST post-quantum standardı"),
    ("ML-DSA (Dilithium)",   "✅ Güvenli", "Post-quantum dijital imza"),
    ("SLH-DSA (SPHINCS+)",   "✅ Güvenli", "Hash-based, kanıtlanmış güvenlik"),
]

for algo, status, note in checks:
    print(f"\n  {status} {algo}")
    print(f"     → {note}")

print(f"\n{'─' * 55}")
print("  💡 NIST standartları (2024): ML-KEM, ML-DSA, SLH-DSA")
print("  📋 Aksiyon: Kriptografi envanteri çıkar")
print("     Hangi sistemler RSA/ECC kullanıyor? Plan yap.")
```

---

## Sonuç

Bu tutorialda:

1. **RSA'nın nasıl çalıştığını** ve neden güvenli sayıldığını gördünüz
2. **Shor's algorithm** ile gerçek çarpanlara ayırma yaptınız
3. **QFT devresinin** iç yapısını inceldiniz
4. **RSA kırılma zaman çizelgesini** — IBM'in qubit roadmap'i ile karşılaştırdınız
5. **"Harvest now, decrypt later"** tehdidini ve post-quantum çözümleri öğrendiniz

> **Tek cümle:** Quantum bilgisayarlar RSA'yı kıracak — soru "kıracak mı?" değil
> "ne zaman kıracak?". Cevap: **muhtemelen bu on yılın sonunda.**

### Sonraki Adımlar

- [QAOA ile Lojistik Optimizasyon](09b-qaoa-logistics.md) — Gerçek hayat problemi #2
- [QML ile Fraud Detection](09c-qml-fraud.md) — Gerçek hayat problemi #3
