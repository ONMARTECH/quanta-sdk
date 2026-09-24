# Original User Request

## 2026-09-24T06:08:44Z

Use a very large team of agents. Ekibi akademisyenler, teorik fizikçiler, uygulamalı/deneysel fizikçiler, matematikçiler ve kuantum bilişim uzmanlarından oluşan çok yönlü bir akademik hakem ve mühendislik teftiş kurulu olarak yapılandır.

Quanta SDK ve framework mimarisini baştan sona denetle; 2026 Eylül ayı kuantum bilişim literatürü, endüstriyel standartları ve kuramsal atılımları doğrultusunda eksiklikleri, kuramsal zayıflıkları ve sıçrama yapılabilecek gelişim alanlarını tespit et, her bulguyu kesin matematiksel ve deneysel testlerle doğrula.

Working directory: /Users/aes/Antigravity Projects/Alfa/quanta
Integrity mode: development

## Core Architecture Principles (Zero-Dependency & First-Principles)
- **Standalone Native-First Core**: Quanta'nın çekirdek mimarisi harici ağır framework'lere (Qiskit, Pennylane, Cirq) bağımlı olmadan saf Python/NumPy ve Apple Silicon Metal/MLX ile çalışmaya devam etmelidir.
- **Differential Verification (Golden Standard Testing)**: Algoritmik ve kuramsal doğruluk testlerinde, harici açık kaynak referanslar (örn. PyMatching, Stim, SymPy, SciPy) opsiyonel altın standart oracles olarak kullanılabilir; ancak üretim çalışma zamanı (runtime) bağımsızlığını korumalıdır.
- **Zero Mock / Falsifiable Empiricism**: Asla varsayımsal, uydurma ("mış gibi") veya sentetik verilerle geçiştirilmiş test kabul edilmez. Her kuramsal iddia, sınır koşulu ve fiziksel korunum yasası çalıştırılabilir, tekrarlanabilir kod ve property-based testlerle ispatlanmalıdır.

## Requirements

### R1. Kuramsal Fizik & Matematiksel Titizlik Denetimi (Theoretical Physics & Math Foundations)
1. **Hilbert Uzayı & Üniterlik Korunumu**:
   - Tüm gate operasyonları, durum vektörü (statevector) evrimleri ve zaman bağımlı Hamiltonian simülasyonları için üniterlik koşulunun ($U^\dagger U = I$) makine hassasiyetinde ($\epsilon < 10^{-12}$) korunduğunun analitik ve numerik teyidi.
   - Açık kuantum sistemleri (`density_matrix`, Lindblad master equation, Kraus operatörleri $\sum K_i^\dagger K_i = I$) için Tam Pozitiflik ve İz Korunumu (CPTP) doğrulaması.
2. **Sürekli Rezonans & Hilbert Gradyanları (`quanta.torch`)**:
   - Daleckii-Krein Schur ayrışımı tabanlı matris üstel türevlerinin (matrix exponential autograd) parameter-shift ve sonlu farklar (finite difference) ile gradyan uyumu ve enerji korunum sınırlarının ispatı.
   - Dinamik Lie Cebirleri ($\mathfrak{g} = \langle i H_k \rangle_{\text{Lie}}$) ve Barren Plateau teorisi: Ansatz derinliği, qubit sayısı ve Lie cebir boyutu arasındaki ilişki üzerinden gradyan sönümleme risklerinin matematiksel haritalanması.

### R2. Gerçek Zamanlı Hata Düzeltme (QEC) & 2026 FTQC Standartları
1. **Yüzey Kodları & Dekoder Başarımı**:
   - Mevcut açgözlü (greedy) eşleme algoritmasının kuramsal eşik (threshold) sınırları ile gerçek MWPM (Edmonds Blossom / PyMatching referansı) ve Union-Find çözümleyicileri arasındaki başarı/hata farklarının sayısal analizi.
   - Willow mimarisi ile uyumlu uzay-zaman (spacetime) sendrom çıkarma döngüleri, ölçüm hatası toleransı ve $\Lambda$ (hata bastırma çarpanı) ölçeklenme testleri.
2. **2026 FTQC Sıçrama Alanları (qLDPC & Transversal Gates)**:
   - 2D yüzey kodlarının ötesine geçiş: Bivariate Bicycle / qLDPC kodları (Gross code [[144, 12, 12]]) için seyreltik parite kontrol matrisleri ve BP-OSD (Belief Propagation with Ordered Statistics Decoding) gereksinim analizi.
   - Non-Clifford kapı sentezi: T-kapısı / CCZ için sihirli durum damıtma (magic state distillation) ve örgü cerrahisi (lattice surgery) eksikliklerinin kuramsal tasarımı.

### R3. Donanım Hızlandırma & Yürütme Motoru (Apple Silicon Metal/MLX & Simülatörler)
1. **MPS & Tensör Ağları Sınırları**:
   - Matrix Product States (`quanta.simulator.mps`) simülatörünün bağ boyutu ($\chi$ bond dimension) ölçeklenmesi, Schmidt katsayıları kesme hatası (truncation error) ve 200+ qubit dolaşıklık sınırlarının analizi.
   - Apple Silicon M-serisi birleşik bellek (Unified Memory) bant genişliği optimizasyonu; Metal Performance Shaders / MLX matris tensörleme darboğazlarının teşhisi.
2. **Clifford & Dinamik Devreler**:
   - Stabilizer / Clifford simülatörünün (Aaronson-Gottesman tabanlı Pauli frame) milyon gate/saniye ölçeğindeki performans profili ve OpenQASM 3.0 dinamik mid-circuit ölçüm / feedforward yürütme kabiliyeti.

### R4. Kapsamlı Boşluk Analizi Raporu & Uygulanabilir Yol Haritası (September 2026)
1. Kod tabanında "eksik", "kısmi/sezgisel (heuristic)", "tam/üretim düzeyinde" ve "akademik atılım fırsatı" olan tüm bileşenlerin kategorik dökümü.
2. 2026 Eylül ayı güncel kuantum ekosistemi (Google Willow, IBM Heron/Condor, Harvard/QuEra nötr atomlar, AWS Braket) karşısındaki rekabetçi ve bilimsel konumlandırma matrisi.
3. Tespit edilen her teorik eksik veya gelişim alanı için matematiksel temellendirme, kod mimarisi önerisi ve doğrulama stratejisi.

## Acceptance Criteria

### Kuramsal ve Fiziksel Doğruluk
- [ ] Üniterlik ($U^\dagger U = I$), CPTP süperoperatör iz korunumu ($\text{Tr}(\rho)=1$) ve Daleckii-Krein matris türevleri analitik ve numerik testlerle $\epsilon < 10^{-10}$ toleransında doğrulanmış olmalıdır.
- [ ] Barren plateau ve dinamik Lie cebiri analizleri matematiksel teoremlerle desteklenmeli, ansatz gradyan varyansının qubit sayısına göre davranışı test edilmelidir.

### QEC & Dekoder Kıyaslama
- [ ] Mevcut `MWPMDecoder` greedy eşleme performansı ile gerçek minimum ağırlıklı mükemmel eşleme arasındaki eşik ve mantıksal hata oranı farkı nicel testlerle ortaya konmalıdır.
- [ ] Spacetime sendrom döngülerinde fiziksel hata olasılığı $p < p_{\text{th}}$ durumunda mantıksal hata oranının mesafe $d$ ile üssel bastırıldığı ($P_L \propto (p/p_{\text{th}})^{(d+1)/2}$) kanıtlanmalıdır.

### Simülatör & Hesaplama Performansı
- [ ] MPS ve Statevector simülatörlerinin dolaşıklık entropisi, Schmidt ayrışımı ve bağ boyutu ($\chi$) kesme doğruluğu analitik Bell / GHZ / W durumları üzerinde test edilmelidir.
- [ ] MLX ve PyTorch backend'lerinin bellek tüketimi (RSS) ve tensör tensörleme süreleri ölçülmüş, sentetik olmayan gerçekçi devrelerle doğrulanmalıdır.

### Test Bütünlüğü ve Falsifiability
- [ ] Hiçbir iddia testi olmadan raporlanamaz: Her eksiklik veya gelişim alanı için `tests/` dizininde çalıştırılabilir bir test, kıyaslama (benchmark) veya property-based test (Hypothesis) yazılmış olmalıdır.
- [ ] Tüm yeni testler `/Users/aes/Antigravity Projects/Alfa/quanta/.venv/bin/python -m pytest` altında %100 başarıyla geçmeli, mevcut 1600+ testte hiçbir regresyon oluşmamalıdır.


## 2026-09-24T20:14:23Z

Implement Plastic Biomorphic Cognitive Immunity (LTP/LTD-governed Negative Engrams) and Multi-Branch Decision Tree Deliberation (DAG Rollouts) in Quanta SDK with seamless Antigravity hook integration.

Working directory: /Users/aes/Antigravity Projects/Alfa/quanta
Integrity mode: development

## Requirements

### R1. Plastic Negative Engrams & Cognitive Immunity (LTP/LTD)
Implement an adaptive inhibitory memory layer in `FastBiomorphicMemory` and `CognitiveMemoryManager`:
- Negative engrams (`category="inhibitor"` or `"anti_pattern"`) must feature dynamic synaptic weights ($V_{\text{inh}}$) and context tags (`runtime`, `library`, `os`, etc.).
- **Long-Term Potentiation (LTP)**: Repeated failure in the same context progressively deepens the inhibition strength and increases salience.
- **Long-Term Depression (LTD) & Context Switching**: If an operation succeeds under a different or updated context, the inhibitor relaxes its weight and documents the contextual divergence.
- SWR Replay serialization in `quanta_subconscious_hook.py` must include a designated `🚫 İnhibitör / Anti-Pattern` line alongside `🔒 Çekirdek` and `⚡ Geçici`.

### R2. Non-Binary Multi-Branch Decision Manifold (DAG Rollouts)
Extend Quanta's cognitive deliberation to support multi-option branching ($A, B, C, D, E \to \text{sub-branches}$):
- Model decision trajectories as a directed acyclic graph (DAG) or multi-branch tensor superposition.
- Provide biomorphic evaluation that cascades through downstream consequences (e.g. secondary trade-offs, maintenance costs, latency).
- Prefrontal Zeno Arbiter prunes dominated sub-branches while Generative Dreamer (DMN) maintains exploratory lateral paths.

### R3. Antigravity Hook & Feedback Pipeline Integration
Wire up bidirectional telemetry between Antigravity and Quanta:
- Add a fail-safe `PostToolUse` handler to `scripts/hooks/quanta_subconscious_hook.py` (and update `~/.gemini/config/hooks.json`).
- When a tool emits an error (`error` in payload or non-zero exit), automatically register or potentiate the corresponding negative engram.
- When subsequent tools succeed in related tasks, depress or clear temporary inhibitory blocks.

### R4. Architectural Invariants, Code Quality & Test Suite
Maintain 100% adherence to Quanta's framework standards:
- All code, comments, and docstrings in English (Google-style docstrings, 100-character line limit).
- Type hints on all public interfaces, zero Ruff lint errors, and passing `mypy quanta/ --ignore-missing-imports`.
- Comprehensive unit and E2E integration test suite in `tests/test_cognitive_plasticity.py` or new dedicated test modules verifying LTP/LTD dynamics, hook execution (< 25ms latency), and zero regression across the existing test suite.

## Acceptance Criteria

### Biomorphic Memory & Plasticity
- [ ] `FastBiomorphicMemory` and `CognitiveMemoryManager` support recording, reinforcing (LTP), and depressing (LTD) negative engrams.
- [ ] Negative engrams decay or relax when validated against positive context switches.
- [ ] Hook output produces formatted `🚫 İnhibitör:` lines formatted identically to the biological SWR standard.

### Hook Pipeline & Latency
- [ ] `PostToolUse` handler executes fail-safe with `< 25ms` latency, never interrupting tool workflows.
- [ ] Non-zero exit status or tool error triggers plastic negative engram updates.

### Quality & Test Suite
- [ ] `ruff check quanta/` and `mypy quanta/ --ignore-missing-imports` pass with zero errors.
- [ ] All new tests pass via `/Users/aes/Antigravity Projects/Alfa/quanta/.venv/bin/pytest`.
- [ ] Existing cognitive test suites (`test_cognitive_*.py`, `test_subconscious*.py`) suffer zero regressions.

## 2026-09-24T21:14:19Z

Use a very large team of agents. Ekibi kıdemli kuantum yazılım mimarları, teorik kuantum fizikçileri, uygulamalı matematikçiler, kuantum hata düzeltme (QEC) araştırmacıları ve akademik kıyaslama (benchmarking) mühendislerinden oluşan çok yönlü bir akademik yazım ve hakem heyeti olarak yapılandır.

Quanta SDK için uluslararası hakemli dergi ve preprint standartlarında (**arXiv `quant-ph`**, **Zenodo**, ve **Journal of Open Source Software - JOSS**) kapsamlı bir **Akademik Yazılım Mimarisi Makalesi (Software Architecture Paper)** ve eksiksiz yayın paketi üret. Makale, Quanta'nın özgün felsefesini, metodolojisini, diğer framework'lerden (Qiskit, Cirq, PennyLane, Stim, PyMatching, QuTiP) yapısal farklarını ve ampirik başarılarını sıfır halüsinasyon ve kesin matematiksel/deneysel temellerle ortaya koymalıdır.

Working directory: /Users/aes/Antigravity Projects/Alfa/quanta  
Integrity mode: development  

---

## Authorship & Identity Metadata
- **Lead Author & Principal Architect**: Abdullah Enes SARI (ORCID: [0000-0002-8827-0587](https://orcid.org/0000-0002-8827-0587))
- **Affiliation**: ONMARTECH Quantum Computing Initiative, Istanbul, Turkey (`info@onmartech.com`)
- **Co-Author & Peer Inspection Board**: Quanta Quantum Research Group & Antigravity Agentic AI Board
- **Target Venues**:
  1. **arXiv.org**: Category `quant-ph` (Quantum Physics) / `cs.MS` (Mathematical Software)
  2. **Journal of Open Source Software (JOSS)**: Open-source research software track
  3. **Zenodo / CERN**: Permanent citable software release DOI (`v1.2.0`)
  4. **quanta.onmartech.com**: Web whitepaper publication

---

## Core Scientific Principles & Architectural Methodology

Quanta SDK'nın akademik özgünlüğü, piyasadaki mevcut araçların kopyası olmamasından ve 5 temel metodolojik ayırıcı sütun üzerine kurulmasından kaynaklanır:

1. **Paradigm 1: Local-First, Zero-Dependency First-Principles Core**
   - Ağır C++/LLVM derleme zincirlerine, CUDA kısıtlarına veya şişkin bağımlılıklara ihtiyaç duymadan, saf Python/NumPy ile taşınabilir, şeffaf ve ilk ilkelerden (first-principles) çalışan kuantum yürütme motoru.
2. **Paradigm 2: Apple Silicon Metal / MLX Zero-Copy GPU Acceleration**
   - Birleşik bellek (Unified Memory) mimarisinde CPU-GPU veri kopyalama maliyetlerini sıfırlayan (zero-copy), 24 qubite kadar $48\times$ hızlanan Metal Performance Shaders / MLX tensör motoru ve SIMD vektörize $>1.1 \times 10^6$ kapı/sn Clifford ikili tablo simülatörü.
3. **Paradigm 3: Continuous Hilbert Gradients & Daleckii-Krein Autograd (`quanta.torch`)**
   - Padé yaklaşımlarının norm sapmalarını (`> 1.3e-6`) sıfırlayarak `complex128` hassasiyetinde kapalı formlu Fréchet türevi ($\frac{d}{d\theta} e^X$) sağlayan Daleckii-Krein spektral teorem entegrasyonu ve dinamik Lie cebiri ($\dim(\mathfrak{g})$) tabanlı analitik barren plateau teşhis motoru.
4. **Paradigm 4: 2026 FTQC Dual-Track Fault-Tolerance Engine**
   - **Track A (2D Topolojik)**: Açgözlü eşleme açıklarını kapatan Edmonds Blossom MWPM dekoderi ve Google Willow uyumlu 3D uzay-zaman sendrom döngüleri.
   - **Track B (Yüksek Boyutlu qLDPC)**: 2D yüzey kodlarına göre $12\times$ bellek tasarrufu sağlayan kanonik Gross $[[144, 12, 12]]$ Bivariate Bicycle kodu ve yerel BP-OSD (Normalized Min-Sum + OSD-0) dekoderi.
   - **Non-Clifford Evrensellik**: 15-to-1 Bravyi-Kitaev sihirli durum damıtma fabrikası ($\epsilon_{\text{out}} \le 35 p^3$) ve örgü cerrahisi (lattice surgery).
5. **Paradigm 5: Falsifiable Empiricism & Certified Rigor**
   - 1.907 adet çalışan, regresyonsuz test ile üniterlik ($\|U^\dagger U - I\| < 10^{-14}$), CPTP iz korunumu ($|\text{Tr}(\rho) - 1.0| < 10^{-12}$) ve kuantum faz doğruluğunun matematiksel garantisi.

---

## Requirements

### R1. Academic Software Paper (RevTeX 4-2 LaTeX & Markdown Editions)
1. **RevTeX 4-2 Manuscript (`docs/arxiv/quanta_framework/main.tex`)**:
   - Standart APS/IEEE formatında, çift sütun, eksiksiz formülasyonlar, mimari akış diyagramları, tablolar ve denklem türetimleri içeren tam teşekküllü akademik makale.
   - Yazar künyesi: `Abdullah Enes SARI \orcidlink{0000-0002-8827-0587}` ve `ONMARTECH`.
2. **Web Whitepaper Edition (`docs/papers/quanta_framework_paper.md`)**:
   - `quanta.onmartech.com` üzerinde yayınlanmak üzere MkDocs Material ve MathJax uyumlu, zengin LaTeX denklemleri içeren edisyon.

### R2. Metodolojik Karşılaştırma Matrisi & Ampirik Kıyaslama (Benchmarking)
1. **Sistematik Karşılaştırma Tablosu**:
   - Quanta SDK vs. Qiskit, Cirq, PennyLane, Stim, PyMatching, QuTiP, Julia QuantumClifford:
     - Çekirdek bağımlılık yapısı (Zero-dependency vs. C++/LLVM).
     - Bellek mimarisi (Apple Silicon Zero-Copy vs. PCIe Host-Device transfer).
     - Sürekli türev yaklaşımı (Daleckii-Krein Fréchet vs. Parameter-shift vs. Finite Difference).
     - QEC dekoder kabiliyeti (Edmonds Blossom MWPM + qLDPC BP-OSD + Willow 3D).
     - Macroscopic MPS ölçeklenmesi ve Schmidt kesme norm garantisi.
2. **Çalıştırılabilir Kıyaslama Betiği (`benchmarks/run_paper_benchmarks.py`)**:
   - Makalede iddia edilen tüm sayısal metrikleri (Clifford gate/sn, MLX hızlanma çarpanı, MPS 250-qubit GHZ süresi, Daleckii-Krein gradyan hatası, BP-OSD vs. MWPM eşleşme ağırlığı) doğrudan yerel makinede koşturup çıktı üreten tekrarlanabilir Python scripti.

### R3. Doğrulanmış Akademik Kaynakça (Sıfır Halüsinasyon Mandatı)
1. **Sıfır Halüsinasyon İlkesi**:
   - Makalede atıf yapılan istisnasız HER makale, kuantum fiziği, bilgisayar bilimi veya matematik literatüründe fiilen mevcut, hakemli veya resmi arXiv preprinti olmalıdır.
   - Her BibTeX girdisi (`references.bib`) resmi DOI (`10.xxxx/...`) ve doğrulanmış arXiv ID içermelidir. Uydurma başlık, uydurma yazar veya kırık link kesinlikle kabul edilmez.
2. **Kapsamlı Literatür Temellendirmesi**:
   - QEC & Yüzey Kodları (Google Quantum AI Nature 2021/2023, Fowler et al. PRA 2012, Horsman et al. NJP 2012).
   - Eşleme Algoritmaları (Edmonds CJM 1965, Kolmogorov MPC 2009, Higgott ACM TQC 2022).
   - qLDPC Kodları & BP-OSD (Bravyi et al. Nature 2024, Panteleev-Kalachev Quantum 2021, Roffe et al. PRResearch 2020).
   - Sihirli Durum Damıtma (Bravyi-Kitaev PRA 2005).
   - Matris Türevleri & Lie Cebirleri (Daleckii-Krein AMS 1974, Mathias SIAM 1996, McClean Nature Comm 2018, Fontana et al. Nature Comm 2024).
   - Simülasyon Temelleri (Lindblad CMP 1976, Gorini et al. JMP 1976, Schollwöck Ann. Phys. 2011, Vidal PRL 2003, Aaronson-Gottesman PRA 2004).
   - Kuantum Yazılım Referansları (Qiskit 2019, PennyLane 2018, Stim 2021, QuTiP 2012).

### R4. JOSS & Zenodo Gönderim Paketi
1. **JOSS Paper (`docs/papers/joss/paper.md` ve `paper.bib`)**:
   - Journal of Open Source Software standartlarına uygun `paper.md` (Summary, Statement of Need, State of the Field, Mathematics, Acknowledgements).
2. **Zenodo Release Üstverisi (`.zenodo.json`)**:
   - Zenodo'nun GitHub release sırasında otomatik olarak algılayacağı JSON üstveri dosyası (yazar adı, ORCID, lisans, anahtar kelimeler).

---

## Acceptance Criteria

### Kuramsal ve Metodolojik Doğruluk
- [ ] Makale metninde Quanta SDK'nın 5 özgün metodolojik sütunu matematiksel kesinlikle tanımlanmış ve türetilmiş olmalıdır.
- [ ] Diğer framework'lerle olan mimari farklar tarafsız, teknik ve ampirik bir tablo ile ortaya konmalıdır.

### Kaynakça ve Doğrulama
- [ ] Kaynakçadaki tüm makaleler resmi DOI ve arXiv kayıtlarıyla %100 eşleşmeli; otomatik `curl`/API doğrulamasıyla sıfır 404/hata vermelidir.
- [ ] Yazarın adı istisnasız her yerde `Abdullah Enes SARI` ve `ORCID: 0000-0002-8827-0587` olarak yer almalıdır.

### Tekrarlanabilirlik ve Kod Bütünlüğü
- [ ] `benchmarks/run_paper_benchmarks.py` scripti hatasız çalışmalı ve makaledeki sayısal metrikleri doğrulamalıdır.
- [ ] Mevcut 1.907 testte hiçbir regresyon olmamalı, test suite %100 yeşil kalmalıdır.
- [ ] `mkdocs build --strict` komutu 0 uyarı ile derlenmeli ve web edisyonu siteye entegre edilmelidir.
