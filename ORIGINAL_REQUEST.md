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
