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

v0.1: 12 qubit = 1.8s,   14 qubit = timeout
v0.2: 12 qubit = 0.003s, 25 qubit = 3.4s
```

> [!TIP]
> Tensor contraction yöntemi, durum vektörünü `[2, 2, ..., 2]`
> tensör olarak tutar ve `np.tensordot` ile kapıyı sadece ilgili
> eksenlere uygular. Bu, tam 2^n × 2^n matris oluşturmayı ortadan kaldırır.
