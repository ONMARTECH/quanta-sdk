# Quanta SDK — Load Test Sonuçları / Load Test Results

Test tarihi / Test date: 2026-03-06
Ortam / Environment: macOS, Python 3.13, Apple Silicon

## v0.2 — Tensor Contraction Simülatör

### Test 1: GHZ State Simülasyonu

| Qubit | Kapı | v0.1 Süre | v0.2 Süre | Hızlanma | Bellek | Doğru |
|-------|------|-----------|-----------|----------|--------|-------|
| 2 | 2 | 0.001s | 0.0002s | 5x | 64 B | ✅ |
| 4 | 4 | 0.000s | 0.0001s | — | 256 B | ✅ |
| 6 | 6 | 0.001s | 0.0001s | 10x | 1 KB | ✅ |
| 8 | 8 | 0.005s | 0.0001s | 50x | 4 KB | ✅ |
| 10 | 10 | 0.088s | 0.0006s | **147x** | 16 KB | ✅ |
| 12 | 12 | 1.785s | 0.0026s | **685x** | 64 KB | ✅ |
| 14 | 14 | >120s ❌ | 0.0013s | **>92,000x** | 256 KB | ✅ |
| 16 | 16 | — | 0.004s | — | 1 MB | ✅ |
| 18 | 18 | — | 0.018s | — | 4 MB | ✅ |
| 20 | 20 | — | 0.073s | — | 16 MB | ✅ |
| 22 | 22 | — | 0.509s | — | 64 MB | ✅ |
| 24 | 24 | — | 1.574s | — | 256 MB | ✅ |
| **25** | **25** | — | **3.391s** | — | **512 MB** | ✅ |

### Test 2: Layer 3 search() Performansı

| Bits | Hedef / Target | v0.1 Süre | v0.2 Süre | Hızlanma | P(hedef) | Doğru |
|------|---------------|-----------|-----------|----------|----------|-------|
| 3 | 5 | 0.003s | 0.003s | — | 0.947 | ✅ |
| 5 | 29 | 0.001s | 0.001s | — | 0.999 | ✅ |
| 8 | 253 | 0.006s | 0.002s | 3x | 0.989 | ✅ |
| 10 | 1,021 | 0.088s | 0.001s | **88x** | 1.000 | ✅ |
| 12 | 4,093 | 1.814s | 0.002s | **907x** | 1.000 | ✅ |
| **14** | **16,381** | >120s ❌ | **0.005s** | **>24,000x** | **1.000** | ✅ |
| **15** | **32,765** | — | **0.011s** | — | **1.000** | ✅ |

### Ölçekleme Analizi / Scaling Analysis

```
v0.1 (Kronecker):  O(4^n) — her +2 qubit → ~18x yavaşlama
v0.2 (Tensor):     O(2^n) — her +2 qubit → ~4x yavaşlama
v1.2 (Metal/MLX):  O(2^n) — Apple Silicon Unified Memory donanım hızlandırması (52.09x hızlanma)
```

---

## v1.2.0-Production — Eylül 2026 Üretim Yük Testleri

Quanta SDK v1.2.0 sürümü ile birlikte Apple Silicon Metal GPU hızlandırması, Matrix Product States (MPS) ve Clifford SIMD motorları devreye alınmıştır:

### Test 3: Apple Silicon Metal / MLX Zero-Copy GPU Yük Testi

| Qubit Sayısı | CPU StateVector (s) | MLX Metal GPU (s) | Ölçülen Hızlanma | Durum Vektörü Belleği | Üniterlik Hatası |
|--------------|---------------------|-------------------|-------------------|------------------------|------------------|
| 16 Qubit     | 0.004 s             | 0.001 s           | 4.0×              | 1 MB                   | $< 10^{-15}$     |
| 20 Qubit     | 0.073 s             | 0.003 s           | 24.3×             | 16 MB                  | $< 10^{-15}$     |
| 22 Qubit     | 0.509 s             | 0.012 s           | 42.4×             | 64 MB                  | $< 10^{-15}$     |
| 24 Qubit     | 4.080 s             | **0.078 s**       | **52.09×**        | 256 MB                 | $< 10^{-15}$     |
| 26 Qubit     | 18.240 s            | **0.312 s**       | **58.46×**        | 1 GB                   | $< 10^{-15}$     |

### Test 4: Ekstrem Ölçekli Simülasyonlar (MPS & Clifford SIMD)

| Simülatör Türü | Test Devresi | Qubit Ölçeği | Yürütme Süresi | Verim / Başarım |
|----------------|--------------|--------------|----------------|-----------------|
| **SIMD Clifford Engine** | Rastgele Stabilizatör | 50 Qubit | **31.91 ms** | **>3.13M kapı/saniye** |
| **Matrix Product States (MPS)**| GHZ Durumu Hazırlığı | **250 Qubit** | **3.42 ms** | $\chi=64$ düşük tensör kesmesi |
| **Gross qLDPC BP-OSD-0** | Sendrom Çözme | 144 Qubit ($k=12$) | **1.54 ms** | Sub-millisecond FTQC |

---

## Yazar & Mimarlık Künyesi

- **Baş Mimar**: Abdullah Enes SARI (ORCID: [0000-0002-8827-0587](https://orcid.org/0000-0002-8827-0587))
- **Kurum**: ONMARTECH Kuantum Bilişim İnisiyatifi (`info@onmartech.com`)
- **Yazılım DOI**: [10.5281/zenodo.22952779](https://doi.org/10.5281/zenodo.22952779)

