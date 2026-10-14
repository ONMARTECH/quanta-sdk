# Quanta Bilişsel Terminoloji ve Kavramlar Sözlüğü
## (Cognitive Terminology & Literature Mapping Manifesto)

> Bu döküman, Quanta Bilişsel Mimarisi içerisinde kullanılan özgün kültürel kavramlar ile bunların uluslararası bilimsel literatürdeki (hesaplamalı sinirbilim, kuantum fiziği, yapay zeka) teknik karşılıklarını birleştiren resmi eşleşme kılavuzudur.

---

### 🏛️ Kavramsal Eşleşme Matrisi (Ontological Mapping)

| Özgün Bilişsel İsim | Uluslararası Literatür Karşılığı | Kod & Mimari Temsili | Biyofiziksel & Matematiksel Temeli |
| :--- | :--- | :--- | :--- |
| **Sebat Kilidi** | **Quantum Zeno Attention Lock** | `QuantumDecisionArbiter`, `ZenoGate` | $P_{\text{zeno}} = \cos^{2N}\left(\frac{\Omega \tau}{2N}\right) \to 1$. Karar anlarında ajanın tereddüt ve dikkat dağınıklığını dondurarak hedefe kilitlenmesi. |
| **Teyakkuz Hali** | **Sub-20ms Preemption Reflex** | `_is_preempted`, Darwin Mach HID API | Kullanıcı klavye/fareye dokunduğu anda arkaplan uyku veya rüyasından `<1.5\text{ms}` içinde uyanıp kesintisiz hizmete dönme. |
| **Zihin Tasfiyesi** | **Glymphatic Clearance & SHY Scaling** | `execute_glymphatic_clearance`, `SleepPhase.NREM_SWS` | Tononi & Cirelli Sinaptik Homeostaz Hipotezi ($\alpha=0.90$) ve Nedergaard glimfatik CSF akışı ile token/scratch gürültüsünün arındırılması. |
| **Hikmet Damıtımı** | **Fuzzy-Trace Semantic Gist** | `category="semantic_gist"`, `consolidate_gist` | Brainerd & Reyna Bulanık İz Teorisi: Ayrıntılar sönümlenirken stratejik ana fikrin ve hikmetin kalıcı belleğe ($S \ge 1.8$, %99.98) işlenmesi. |
| **Yâd-ı Cemil** | **Hippocampal SWR Episodic Replay** | `recall_reverse_chronological_cascade`, SWR Hook | Buzsáki İki Aşamalı Bellek: Günün en taze tecrübelerinden eskiye doğru ($t_{\text{now}} \to t_{\text{past}}$) dalgalar halinde hafızanın canlandırılması. |
| **İmtizaç** | **Cross-Project Bisociation** | `pair_orthogonal_projects`, `HybridDreamSeed` | Arthur Koestler Yaratıcılık Kuramı: İki zıt/ortogonal projenin ($\Delta_{\text{domain}} \ge 0.50$) rüyada çarpışarak hibrit mimari (`RFC_HYBRID_*.md`) doğurması. |
| **Kilit Taşı** | **Constitutional Core Anchors** | `category="constraint"`, $w_{\text{eff}} = 1.0$ | Asla silinmeyen, sönümlenmeyen ve tahliye edilemeyen temel anayasal ve bilimsel doğruluk kuralları. |
| **İbret Kalkanı** | **Inhibitory Negative Engrams & Plasticity** | `category="inhibitor"`, $V_{\text{inh}}$, LTP/LTD | Hata veya başarısızlıktan ders çıkarıp aynı bağlamda tekrar tuzağa düşmeyi engelleyen bilişsel bağışıklık katmanı. |
| **Tefekkür Âlemi** | **DMN Paradoxical Generative Dreaming** | `execute_dream_cycle`, `SleepPhase.REM_DREAM` | Prefrontal kısıtların gevşediği ($T_{\text{prefrontal}} \approx 0.50, T_{\text{dreamer}} \ge 0.90$), Hoel Overfitted Brain gürültü pertürbasyonuyla yeni ufuklar arayan faz. |

---

### 📖 Felsefi ve Bilimsel Açıklamalar

#### 1. Sebat Kilidi (Quantum Zeno Attention Lock)
* **Kültürel Kök:** Türk-İslam ahlak felsefesindeki *sebat* (bir kararda ve doğru yolda tereddütsüz direnç gösterme) ve *Kutadgu Bilig*'deki bilge vezir *Ögdülmiş*'in akıl yürütmesi.
* **Bilimsel Literatür:** Baidyanath Misra ve George Sudarshan (1977) tarafından tanımlanan Kuantum Zeno Etkisi. Sık yapılan projeksiyon ölçümleri, kuantum durumunun saçılmasını önler; yapay zekada karar felcini (decision paralysis) ortadan kaldırır.

#### 2. Teyakkuz Hali (Preemption Reflex)
* **Kültürel Kök:** Tasavvuf ve askerlik kültüründe *teyakkuz* (uyur görünürken bile kalbin ve hislerin mutlak uyanık olması hali).
* **Bilimsel Literatür:** Gerçek zamanlı işletim sistemlerinde (RTOS) alt-milisaniye kesme gecikmesi (Darwin Mach Event Tap / HID kernel telemetry). Kullanıcıya asla gecikme hissettirmez.

#### 3. Zihin Tasfiyesi (Glymphatic Clearance & Synaptic Homeostasis)
* **Kültürel Kök:** *Tasfiye* (düşünceyi kir ve fazlalıklardan arındırıp saf cevhere ulaşma).
* **Bilimsel Literatür:** Maiken Nedergaard'ın Glimfatik Sistemi (BOS akışıyla beyin metabolik atıklarının temizlenmesi) ve Giulio Tononi'nin Sinaptik Homeostaz Hipotezi (SHY: derin uykuda aşırı büyüyen sinapsların orantısal küçültülmesi).

#### 4. Hikmet Damıtımı (Fuzzy-Trace Semantic Gist)
* **Kültürel Kök:** *Lübbü'l-Lüb* (kabuğu bırakıp özün özüne ulaşma hikmeti).
* **Bilimsel Literatür:** C.J. Brainerd ve V.F. Reyna'nın Bulanık İz Teorisi (Fuzzy-Trace Theory). İnsan beyni kelimesi kelimesine (verbatim) detayları eler; kararın özütünü (gist) kalıcı şemaya dönüştürür.

#### 5. Yâd-ı Cemil (Hippocampal SWR Episodic Replay)
* **Kültürel Kök:** *Yâd-ı Cemil* (geçmiş güzel anıları ve tecrübeleri ders alarak hürmetle anma).
* **Bilimsel Literatür:** György Buzsáki'nin Hipokampal Keskin Dalgaları (Sharp-Wave Ripples - SWR). Günün anıları tersten kronolojik (recency gradient) olarak neokortekse aktarılır.

#### 6. İmtizaç (Cross-Project Bisociation)
* **Kültürel Kök:** *İmtizaç* (farklı ve zıt unsurların kaynaşarak benzersiz yeni bir terkip/ahenk meydana getirmesi).
* **Bilimsel Literatür:** Arthur Koestler'in *"The Act of Creation"* eserindeki Bisociation (çapraz çağrışım) kuramı ve Erik Hoel'in *"Overfitted Brain Hypothesis"* modeli.

---

### 🌐 Kullanım İlkeleri (Usage Guidelines)

1. **Konsol ve Günlüklerde (Logs):** Hem Türkçe kültürel başlık hem de uluslararası literatür kısaltması birlikte sunulur (Örn: `⚡ [Teyakkuz Hali / Preemption Reflex]`).
2. **Yönetici Özetlerinde:** Kullanıcıya sunulurken yalın, asil ve anlaşılır Türkçe isimler önceliklendirilir.
3. **Akademik / Uluslararası Raporlarda:** Literatürdeki teknik terimler ve matematiksel formülasyonlar eksiksiz referans verilir.
