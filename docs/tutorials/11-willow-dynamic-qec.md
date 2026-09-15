# 🌲 Google Willow Dinamik Surface Code & Çok Döngülü QEC

*Google Willow Mimarisi, Çok Döngülü Sendrom Çıkarımı ve Zamansal Hata Tespiti (Temporal Defect Detection)*

---

> **Bu tutorialda öğrenecekleriniz:** Google Willow işlemcisinin temsil ettiği kuantum hata düzeltme atılımını, çok döngülü dinamik stabilizatör çıkarımını, zamansal defekt tespitini ($\Delta s_t = s_t \oplus s_{t-1}$) ve fiziksel hata eşiğinin altındaki üssel hata baskılama katsayısını ($\Lambda \approx 2.14$) Quanta SDK ile simüle edeceksiniz.

---

## Neden Statik QEC Yetersiz, Dinamik QEC Gerekli?

Geleneksel kuantum hata düzeltme (QEC) modellerinde genellikle tek bir anlık sendrom ölçümü yapılır. Ancak gerçek kuantum işlemcilerde:
1. **Sendrom ölçümünün kendisi de gürültülüdür** (readout noise $p_{\\text{meas}}$).
2. Tek bir hatalı ölçüm, algoritmanın var olmayan bir hatayı düzeltmeye çalışarak veri qubitlerine fazladan hata enjekte etmesine yol açabilir.
3. Çözüm: Stabilizatörler zaman boyunca art arda $T$ döngü boyunca ölçülür ve 2 boyutlu uzaysal ızgara, **3 boyutlu bir uzay-zaman (space-time) grafiğine** dönüştürülür.

Google Willow (Nature 2024/2025) mimarisinde, code distance $d=3, 5, 7$ için her rauntta ölçülen sendromlar bir önceki rauntla XOR’lanarak **zamansal defektler (temporal defects)** tespit edilir:

$$\Delta s_t = s_t \oplus s_{t-1}$$

Eğer fiziksel hata oranı eşiğin altındaysa ($p_{\\text{phys}} < p_{\\text{th}}$), logical hata oranı code distance ile üssel olarak düşer:

$$P_L(d) \propto \Lambda^{-(d+1)/2}$$

Burada $\Lambda > 1$ olduğu sürece, qubit sayısı arttıkça hata oranı azalır!

---

## Adım 1: Dinamik Surface Code Simülasyonu

Quanta SDK, `SurfaceCode.simulate_dynamic()` metodu ile Google Willow çok döngülü hata modelini doğrudan simüle eder:

```python
from quanta.qec.surface_code import SurfaceCode

# Distance d=3 Willow Surface Code (17 fiziksel qubit, 9 data, 8 stabilizer)
code_d3 = SurfaceCode(distance=3)

# 10 döngü boyunca dinamik sendrom çıkarımı
# p_phys: Kapı/Pauli hata olasılığı
# p_meas: Sendrom ölçüm gürültüsü
res_d3 = code_d3.simulate_dynamic(
    rounds=10,
    p_phys=0.001,
    p_meas=0.002,
    seed=42,
)

print("=== Google Willow Dynamic QEC (d=3) ===")
print(f"Toplam döngü:         {res_d3.rounds}")
print(f"Sendrom geçmişi:     {len(res_d3.syndrome_history)} raunt")
print(f"Zamansal defektler:   {len(res_d3.defects)} tespit edildi")
print(f"Logical hata oranı:   {res_d3.logical_error_rate:.4e}")
print(f"Eşik altı durumu:     {res_d3.below_threshold}")
print(f"Lambda faktörü:       {res_d3.lambda_factor:.2f}")
```

---

## Adım 2: Zamansal Defekt Tespiti (Temporal Defects)

Uzay-zaman hata düzeltmesinde asıl bilgi kaynağı sendromun kendisi değil, **sendromun zaman içindeki değişimidir**. 

Bir stabilizatör ölçümü 1 geldiğinde bu bir hatayı gösterir; ancak bir sonraki döngüde de 1 gelmeye devam ediyorsa yeni bir hata oluşmamıştır ($\Delta s_t = 1 \oplus 1 = 0$). Sadece hata başladığı ve bittiği zaman $\Delta s_t = 1$ defekti tetiklenir:

```python
# Sendrom geçmişindeki ardışık rauntları inceleme
for t, round_data in enumerate(res_d3.syndrome_history[:4]):
    x_syn = round_data.get("x_syndromes", [])
    z_syn = round_data.get("z_syndromes", [])
    print(f"Döngü t={t}: X-Sendrom={x_syn}, Z-Sendrom={z_syn}")

print(f"\nTespit edilen toplam zamansal defekt sayısı: {len(res_d3.defects)}")
for defect in res_d3.defects[:3]:
    print(f"  Defekt: Zaman={defect.time}, Stabilizer={defect.stabilizer_id}, Tip={defect.basis}")
```

---

## Adım 3: Willow Ölçeklenme Analizi ($d=3$ vs $d=5$ vs $d=7$)

Google Willow atılımının temel göstergesi, code distance büyüdükçe hata oranının azalmasıdır. Farklı mesafeleri karşılaştıralım:

```python
distances = [3, 5, 7]
results = {}

for d in distances:
    code = SurfaceCode(distance=d)
    res = code.simulate_dynamic(
        rounds=12,
        p_phys=0.001,
        p_meas=0.001,
        seed=100,
    )
    results[d] = res
    print(f"Distance d={d} (Fiziksel Qubit: {code.n_physical}):")
    print(f"  Logical Hata Oranı: {res.logical_error_rate:.6f}")
    print(f"  Lambda (Baskılama):  {res.lambda_factor:.2f}")
    print(f"  Eşik Altında mı?   {res.below_threshold}")
    print()

# Lambda analizi:
lambda_val = results[3].lambda_factor
print(f"💡 Ortalama Willow Lambda Faktörü: {lambda_val:.2f}")
if lambda_val > 1.0:
    print("✅ EŞİK ALTI REJİM (Below Threshold): Distance arttıkça sistem güvenilirleşiyor!")
```

---

## Adım 4: Ölçüm Gürültüsü (Readout Noise) Toleransı

Ölçüm gürültüsünün ($p_{\\text{meas}}$) etkisini test ederek çok döngülü yapının önemini doğrulayalım:

```python
# Yüksek ölçüm gürültüsü altında bile defekt korelasyonu
noisy_res = code_d3.simulate_dynamic(
    rounds=20,
    p_phys=0.0005,
    p_meas=0.01,  # %1 ölçüm gürültüsü
    seed=42,
)

print(f"Yüksek ölçüm gürültüsü altında toplam defekt: {len(noisy_res.defects)}")
print(f"Logical hata oranı: {noisy_res.logical_error_rate:.4e}")
```

---

## Özet & Çıkarımlar

1. **Çok Döngülü Çıkarım**: Donanımdaki ölçüm hatalarını tolere etmek için stabilizatörler $T \ge d$ döngü boyunca ölçülür.
2. **Zamansal Defektler**: $\Delta s_t = s_t \oplus s_{t-1}$ formülü ile uzay-zamansal sendrom eşleştirmesi yapılır.
3. **Willow Faktörü ($\Lambda \approx 2.14$)**: Eşik altındaki donanımlarda code distance her 2 adım arttığında hata oranı yarıdan fazlaya iner.\n