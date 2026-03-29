# Quanta SDK — Roadmap

> Son güncelleme: 29 Mart 2026 · Mevcut: **v0.9.0** · Hedef: **v1.0 (Temmuz 2026)**

---

## Mevcut Durum Özeti

```
Version:     0.9.0            Tests:       669 (89% coverage)
Gates:       31               Files:       76
Tutorials:   14 + 14 notebook MCP Tools:   16
Backends:    4 (local + IBM + IonQ + Google)
Compiler:    6-pass (cancel + merge + translate + route + decompose + validate)
QML:         Classifier + QSVM + 3 feature maps
QEC:         6 codes (bit/phase/steane/shor/surface/color)
```

---

## 4 Geliştirme Hattı

| Hat | Kapsam | Faz |
|-----|--------|-----|
| 🔴 **Çekirdek & Derleyici** | @circuit, DAG, compiler, QASM | Faz 1–2 |
| 🟡 **QML & Algoritmalar** | Classifier, QSVM, Monte Carlo, Entity Res. | Faz 2–3 |
| 🔵 **Backend & İnfra** | Multi-backend, noise, QEC, config | Faz 2–3 |
| 🟢 **Docs & MCP** | Tutorials, notebooks, CI, roadmap | Tüm fazlar |

---

## Faz 1 — Çekirdek Stabilizasyon (Nisan 2026 · v0.10.x)

### Task 1 — API Ergonomisi & Tip Güvenliği
**Durum: 🟡 Kısmen mevcut** — Type hint'ler var ama tutarsız.

| Alt görev | Detay |
|-----------|-------|
| 1.1 | `@circuit`, `measure()`, gate çağrıları için tam `-> ReturnType` annotation |
| 1.2 | `CircuitDefinition.build()` parametrik circuit'lerde IDE autocomplete desteği |
| 1.3 | Yanlış gate/parametre kullanımında net `CircuitError`, `GateError` mesajları |
| 1.4 | `py.typed` marker dosyası + mypy strict pass |

**Dosyalar:** `quanta/core/circuit.py`, `quanta/core/gates.py`, `quanta/core/types.py`

---

### Task 2 — Gate Set & Meta-veri
**Durum: ✅ Büyük ölçüde tamamlanmış** — 31 gate, `GATE_REGISTRY` mevcut.

| Alt görev | Detay |
|-----------|-------|
| 2.1 | Her gate'e `.inverse` property (ters gate otomatik hesaplama) |
| 2.2 | Her gate'e `.controlled(num_ctrl)` metodu (otomatik kontrollü versiyon) |
| 2.3 | Gate metadata: `decomposition_hint`, `native_on` backend bilgisi |
| 2.4 | Yeni gate eklemek tek satırla: `register_gate("CCX", matrix=..., qubits=3)` |

**Dosyalar:** `quanta/core/gates.py`, `quanta/core/custom_gate.py`

---

### Task 3 — DAG IR İyileştirme
**Durum: ✅ Temel mevcut** — `DAGCircuit`, `DAGNode`, topological sort var.

| Alt görev | Detay |
|-----------|-------|
| 3.1 | Node'lara `layer_index` (zaman katmanı) property |
| 3.2 | Pattern-matching: komşu H-H iptali, seri CNOT, Rz-Rz birleşim |
| 3.3 | `dag.substitute_node()` ve `dag.remove_node()` API |
| 3.4 | DAG görselleştirme: `dag.to_dot()` → Graphviz export |

**Dosyalar:** `quanta/dag/dag_circuit.py`, `quanta/dag/node.py`

---

### Task 4 — Compiler Pass Netleştirme
**Durum: ✅ Temel mevcut** — `optimize.py`, `routing.py`, `translate.py` var.

| Alt görev | Detay |
|-----------|-------|
| 4.1 | Her pass için `CompilerPass` base class + `run(dag) -> dag` interface |
| 4.2 | `GateCancellationPass` — H-H, CX-CX, Rz(0) iptali |
| 4.3 | `GateMergingPass` — ardışık Rz birleştirme |
| 4.4 | Birim test: küçük devreler için gate sayısı/derinlik "önce/sonra" karşılaştırma |

**Dosyalar:** `quanta/compiler/passes/*.py`, `quanta/compiler/pipeline.py`

---

### Task 5 — Topoloji-Aware Routing
**Durum: ✅ Temel mevcut** — line/ring/grid topoloji, RouteToTopology pass.

| Alt görev | Detay |
|-----------|-------|
| 5.1 | `Topology.line(5)`, `Topology.grid(4,4)`, `Topology.from_backend("ibm_fez")` API |
| 5.2 | SWAP sayısı profili: line vs ring vs grid benchmark |
| 5.3 | Weighted shortest-path heuristic (edge cost'a göre en kısa SWAP yolu) |
| 5.4 | `Topology.custom(edges=[(0,1), (1,2), ...])` kullanıcı tanımlı topoloji |

**Dosyalar:** `quanta/compiler/passes/routing.py`

---

### Task 6 — QASM 3.0 Export & Import
**Durum: ✅ Export + Import mevcut** — `qasm.py` + `qasm_import.py`.

| Alt görev | Detay |
|-----------|-------|
| 6.1 | Qiskit QASM 3 importer uyumluluk regression testleri |
| 6.2 | `if/else` ve `for` loop QASM 3 subset desteği |
| 6.3 | Round-trip test: export → import → export → aynı QASM çıktısı |
| 6.4 | QASM 2.0 legacy import desteği |

**Dosyalar:** `quanta/export/qasm.py`, `quanta/export/qasm_import.py`

---

## Faz 2 — QML & Backend (Mayıs 2026 · v0.11.x)

### Task 7 — QML API Tasarımı
**Durum: ✅ Temel mevcut** — `QuantumClassifier` ile `.fit()/.predict()/.score()` var.

| Alt görev | Detay |
|-----------|-------|
| 7.1 | `from quanta.qml import Classifier, QSVM, FeatureMap` üst-seviye import |
| 7.2 | Pandas DataFrame giriş desteği + otomatik label sütun algılama |
| 7.3 | Optimizer seçimi: `optimizer="adam"`, `"cobyla"`, `"spsa"` parameter |
| 7.4 | scikit-learn `Pipeline` uyumluluğu: `.get_params()`, `.set_params()` |

**Dosyalar:** `quanta/layer3/qml.py` → `quanta/qml/` modüle taşıma

---

### Task 8 — Feature Map & Ansatz Kütüphanesi
**Durum: 🟡 3 feature map var, ansatz preset eksik**

| Alt görev | Detay |
|-----------|-------|
| 8.1 | İsimlendirilmiş preset'ler: `"zz_feature_map"`, `"amplitude_encoding"`, `"angle_encoding"` |
| 8.2 | Ansatz'lar: `"hardware_efficient"`, `"reuploading"`, `"strongly_entangling"` |
| 8.3 | `FeatureMap.list_available()` ve `Ansatz.list_available()` discovery API |
| 8.4 | Kullanıcı: `Classifier(feature_map="zz", ansatz="hardware_efficient", layers=3)` |

**Dosyalar:** `quanta/qml/feature_maps.py`, `quanta/qml/ansatz.py` (yeni)

---

### Task 9 — QSVM & Classifier Stabilizasyon
**Durum: 🟡 Çalışıyor ama benchmark dataset eksik**

| Alt görev | Detay |
|-----------|-------|
| 9.1 | `make_moons`, `make_circles` benchmark dataset testleri |
| 9.2 | Accuracy + runtime tablo: layer=1,2,3 × lr=0.01,0.1 matris |
| 9.3 | Default hiperparametre tuning: en iyi default'ları belirle |
| 9.4 | Cross-validation: `clf.cv_score(X, y, folds=5)` metodu |

**Dosyalar:** `quanta/layer3/qml.py`, `tests/test_qml.py`

---

### Task 10 — Quantum Monte Carlo & Finance
**Durum: ✅ Temel mevcut** — `monte_carlo.py` + `finance.py`, 16 test.

| Alt görev | Detay |
|-----------|-------|
| 10.1 | `monte_carlo_price(spot, strike, vol, r, T)` basitleştirilmiş API |
| 10.2 | `portfolio_optimize(returns, cov, risk_limit, n_qubits)` API |
| 10.3 | Klasik MC vs Quantum AE karşılaştırma notebook |
| 10.4 | Greeks (delta, gamma, vega) hesaplama desteği |

**Dosyalar:** `quanta/layer3/finance.py`, `quanta/layer3/monte_carlo.py`

---

### Task 11 — Entity Resolution & Use Cases
**Durum: ✅ Temel mevcut** — `entity_resolution.py` + OTA Türkçe örneği.

| Alt görev | Detay |
|-----------|-------|
| 11.1 | `entity_resolution.run(df, config)` genel API |
| 11.2 | Modüler yapı: `blocking.py`, `similarity.py`, `qaoa_solver.py` |
| 11.3 | Farklı dataset desteği: İngilizce isim, ürün, adres |
| 11.4 | Performans raporu: precision/recall/F1 metrikleri |

**Dosyalar:** `quanta/layer3/entity_resolution.py`

---

### Task 12 — Backend Soyutlama
**Durum: ✅ Büyük ölçüde mevcut** — `Backend` ABC + 4 implementasyon.

| Alt görev | Detay |
|-----------|-------|
| 12.1 | `run(circuit, shots, noise_model, **kwargs)` standardize interface |
| 12.2 | `Backend.capabilities()` → qubit sayısı, native gate'ler, bağlantı durumu |
| 12.3 | `Backend.from_name("ibm_fez")` factory metodu |
| 12.4 | Backend health check: `.is_available()`, `.queue_depth()` |

**Dosyalar:** `quanta/backends/base.py`, `quanta/backends/*.py`

---

### Task 13 — Konfigürasyon & Credential Yönetimi
**Durum: 🟡 Env-var tabanlı, CLI yok**

| Alt görev | Detay |
|-----------|-------|
| 13.1 | `quanta configure` CLI komutu: interaktif API key girişi |
| 13.2 | `~/.quanta/config.toml` dosya desteği |
| 13.3 | Credential encryption at rest |
| 13.4 | `quanta backends list` → mevcut ve erişilebilir backend'leri listele |

**Dosyalar:** `quanta/cli/configure.py` (yeni), `quanta/config.py` (yeni)

---

### Task 14 — Noise Model API Sadeleştirme
**Durum: ✅ 7 channel mevcut** — `NoiseModel.add()` chain API var.

| Alt görev | Detay |
|-----------|-------|
| 14.1 | Builder pattern: `NoiseModel.depolarizing(0.01).with_readout(0.02).build()` |
| 14.2 | Preset noise profilleri: `NoiseModel.ibm_heron()`, `.ionq_aria()` |
| 14.3 | Per-gate noise: `model.add(Depolarizing(0.01), gates=["CX"])` |
| 14.4 | Noise summary: `model.describe()` → tablo formatında kanal listesi |

**Dosyalar:** `quanta/simulator/noise.py`

---

### Task 15 — Error Correction Modülleri
**Durum: ✅ Büyük ölçüde mevcut** — 6 kod, 2 decoder, 97% coverage.

| Alt görev | Detay |
|-----------|-------|
| 15.1 | Her kod için `encode(logical_state)` → `physical_circuit` generator |
| 15.2 | `decode(syndrome)` → `correction` fonksiyonu |
| 15.3 | Noise altında logical error rate benchmark script |
| 15.4 | Threshold analizi: physical error rate vs logical error rate grafiği |

**Dosyalar:** `quanta/qec/*.py`

---

## Faz 3 — Stabilizasyon & Ekosistem (Haziran 2026 · v0.12.x)

### Task 16 — Resmi Dokümantasyon & Tutorial Serisi
**Durum: ✅ 14 tutorial + MkDocs live**

| Alt görev | Detay |
|-----------|-------|
| 16.1 | Mimari diyagramı → MkDocs docs'a taşı (Mermaid) |
| 16.2 | "QML Classifier Quickstart" tutorial |
| 16.3 | "Quantum Monte Carlo ile Opsiyon Fiyatlama" tutorial |
| 16.4 | "Entity Resolution: Gerçek Dünya Örneği" tutorial |

**Dosyalar:** `docs/tutorials/`, `mkdocs.yml`

---

### Task 17 — Örnek Repo / Notebooks
**Durum: ✅ 14 notebook + Colab badges**

| Alt görev | Detay |
|-----------|-------|
| 17.1 | Her Layer 3 algoritması için 1 notebook (finance, entity_res, QEC) |
| 17.2 | `examples/` dizini: tek dosyalık çalıştırılabilir script'ler |
| 17.3 | `requirements.txt` ile bağımsız kurulum |
| 17.4 | Binder badge desteği (Colab alternatifi) |

**Dosyalar:** `notebooks/`, `examples/`

---

### Task 18 — Test Coverage & CI
**Durum: ✅ 669 test, 89% coverage, 4 CI workflow**

| Alt görev | Detay |
|-----------|-------|
| 18.1 | Minimum coverage eşiği: `--cov-fail-under=85` |
| 18.2 | Coverage boşlukları: `mcp_server` (39%), `accelerated` (61%), `pauli_frame` (83%) |
| 18.3 | Integration test: IBM backend mock testi |
| 18.4 | Performance regression CI: benchmark sonuçlarını commit'e bağla |

**Dosyalar:** `.github/workflows/tests.yml`, `pyproject.toml`

---

### Task 19 — MCP / AI Tool Entegrasyonları
**Durum: ✅ 16 tool mevcut, coverage 39%**

| Alt görev | Detay |
|-----------|-------|
| 19.1 | Her tool'un input/output JSON schema dokümantasyonu |
| 19.2 | Doğal dil → tool çağrısı örnek akışları |
| 19.3 | MCP test coverage'ı %39 → %70+ |
| 19.4 | Rate limiting ve error handling iyileştirme |

**Dosyalar:** `quanta/mcp_server.py`

---

### Task 20 — ROADMAP Dokümanı
**Durum: ✅ Bu dosya**

---

## Faz 4 — v1.0 GA (Temmuz 2026)

| Deliverable | Açıklama |
|-------------|----------|
| v1.0 stable API | Breaking change freeze, deprecation policy |
| arXiv paper | Quanta SDK teknik rapor, QMC + MCP entegrasyonu |
| Interactive docs | MkDocs + WASM simulator + canlı kod örnekleri |
| ETS domain pack | OTA müşteri dedup, erken rezervasyon QML modeli |
| Blog + HN launch | onmartech.com duyurusu, Hacker News submission |

---

## Versiyon Haritası

```
v0.9.0  ✅  Primitives, @quantum, async, benchmarks (Mart 2026)
v0.10.0     Core & Compiler revamp (Nisan 2026)
v0.11.0     QML stable API + Backend GA (Mayıs 2026)
v0.12.0     Docs & ecosystem polish (Haziran 2026)
v1.0.0      Production-ready GA (Temmuz 2026)
```

---

## Öncelik Matrisi

| Öncelik | Task | Etki | Efor |
|---------|------|------|------|
| 🔴 Kritik | 1, 4, 7, 18 | Yüksek | Orta |
| 🟠 Yüksek | 2, 3, 8, 12, 14 | Yüksek | Orta |
| 🟡 Orta | 5, 6, 9, 10, 13, 15 | Orta | Orta |
| 🟢 Normal | 11, 16, 17, 19, 20 | Orta | Düşük |

---

<sub>Quanta SDK · ONMARTECH · Apache 2.0</sub>
