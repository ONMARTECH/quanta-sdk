# Quanta SDK — Yapılacaklar & Durum

> Son güncelleme: 2026-09-15 | Mevcut versiyon: v0.9.3 hazırlığı | Donanım: Apple Silicon M5 Pro (48 GB RAM)

---

## ✅ Tamamlanan (2026 Eylül Güncellemesi)

### 🔴 Version Drift & Metadata Çözüldü
- [x] `quanta/backends/ibm_rest.py` — USER_AGENT `0.9.2` yapıldı.
- [x] `quanta/mcp_server.py` — 23 MCP aracı, `quanta://info` kaynağı ve promptlar güncellendi.
- [x] `.well-known/mcp/server-card.json` — 23 tool eksiksiz tanımlandı.
- [x] `README.md` — 16 → 23 MCP tools tablosu, rozeti ve mimari açıklamaları güncellendi.
- [x] `docs/` — 14 tutorial ve migrasyon rehberindeki tüm `v0.8.1` referansları `v0.9.2`'ye çekildi.
- [x] `cookbook/index.md` oluşturuldu ve mkdocs kırık linki giderildi.

### 🔴 Donanım & Bellek Optimizasyonu (Apple Silicon M5 Pro 48 GB)
- [x] `quanta/config.py` — dinamik RAM algılama (`get_system_memory_gb()`, `get_max_dense_qubits()`) eklendi.
- [x] `quanta/simulator/statevector.py` — 48 GB RAM için `MAX_QUBITS` 27'den 30 kübite çıkarıldı.
- [x] `tests/test_tier1_coverage.py` — `MAX_QUBITS` testi donanım RAM limitine dinamik duyarlı hale getirildi.

### 🔴 Tip Güvenliği & Kod Kalitesi (Mypy & Ruff)
- [x] 32 Mypy hatası giderildi (Core & Simulator paketinde sıfır hata).
- [x] `quanta/core/gates.py` — `GATE_REGISTRY` tipi `Gate | ParametricGate | MultiParametricGate` olarak düzeltildi, eksik dönüş tipleri tamamlandı.
- [x] `quanta/core/custom_gate.py` — Base `Gate` sınıfı mimarisiyle tam uyumlu hale getirildi (`_build_matrix()`) ve %100 coverage sağlandı.
- [x] `quanta/simulator/custatevec.py` — ruff uyarıları ve workspace pointer güvenliği çözüldü.
- [x] `quanta/backends/__init__.py` — `Backend`, `LocalSimulator`, `LocalBackend`, `IBMBackend`, `IBMRestBackend`, `GoogleBackend`, `IonQBackend` export edildi.
- [x] `quanta/backends/base.py` — Hatalı `IBMQuantumBackend` ve `GoogleQuantumBackend` sınıf isimleri düzeltildi.
- [x] `quanta/export/qasm.py` — Hem `CircuitDefinition` hem de `DAGCircuit` nesnelerini OpenQASM 3.0'a derleyecek şekilde genişletildi.
- [x] Tüm projede `ruff check quanta tests` 0 hata ile geçiyor.

### 🔴 Yeni Nesil Agentic MCP Araçları (Gemini 3.8 / Claude 3.7 / GPT-5)
- [x] `estimate_fault_tolerant_cost`: Devrenin T-gate sayısı, mantıksal kübit ve fiziksel kübit ihtiyacını (Willow / Surface code) hesaplama.
- [x] `quanta_reasoning_eval`: Akıl yürütme modellerinin tasarladığı devrelerin kuantum üstünlük potansiyeli, dolanıklık oranı ve derinlik analizini yapma.
- [x] `transpile_for_target`: Devreleri IBM Heron, Google Willow veya IonQ donanım native kapı setlerine derleme.
- [x] `tests/test_mcp_server.py`: 18 test ile tüm araçlar doğrulandı.

### 🔴 Test Kapsamı Artırımı (Coverage: %80.37 → %88.17)
- [x] `tests/test_mcp_server.py` — 18 test eklendi, MCP coverage %0'dan %65'e çıktı.
- [x] `tests/test_density_matrix.py` — 7 test eklendi, DensityMatrix coverage %0'dan %96'ya çıktı.
- [x] `tests/test_simulator_router.py` — 5 test eklendi, Router coverage %18'den %94'e çıktı.
- [x] `tests/test_visualize.py` — 4 test eklendi, Visualize coverage %0'dan %90'a çıktı.
- [x] `tests/test_custom_gate.py` — 6 test eklendi, CustomGate coverage %37'den %100'e çıktı.
- [x] Toplam test sayısı: **860 passed, 1 skipped (0 failed)**.
- [x] Dokümantasyon (`mkdocs build`) hatasız derleniyor (1.47 saniye).

---

## 🚀 Sırada Ne Var? (v1.0 GA Yol Haritası)

### 📌 Faz 3: QEC & Willow Dinamik Yüzey Kodları
- [ ] Google Willow mimarisine uygun dinamik yüzey kodları (Dynamic Surface Code) sendrom çözücüsü.
- [ ] CUDA-Q Logical & QUOPS benzeri mantıksal kübit benchmark entegrasyonu.

### 📌 Faz 4: QML Modül Geliştirmeleri
- [ ] `quanta/qml/` paketine yeni ansatz presetleri ve hibrit kuantum-klasik optimizasyon eklentileri.

### 📌 Faz 5: v1.0 GA Lansmanı (Ekim/Kasım 2026)
- [ ] PyPI v1.0.0 paketi ve duyuru.

