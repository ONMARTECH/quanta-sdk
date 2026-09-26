# Quanta SDK — Proje Kuralları ve Bilimsel Dürüstlük Protokolü

Bu dosya, Quanta SDK geliştirme ve araştırma süreçlerinde uyulması ZORUNLU kuralları ve ilkeleri belirler.

---

## 🚫 1. Sıfır Toleranslı Anti-Metric Hacking ve Anti-Overfitting Kuralı

Bir yapay zeka ajanının veya mühendislik ekibinin en büyük yanılgısı, bir başarı metriğini tutturmak için problemi yapay olarak basitleştirmesi veya kendi kendini kandırmasıdır (Goodhart Yasası).

### Kesinlikle Yasak Olan Davranışlar:
1. **Benchmark Leakage / Arkadan Sonuç Öğretme:** 
   Çözülmesi hedeflenen problemi klasik bir sezgisel algoritmayla (örn. Babai Nearest Plane, greedy search vb.) önceden çözüp, kuantum veya bilişsel modele sanki o bulmuş gibi aktarmak ve "Kuantum/Bilişsel model %100 çözdü" demek KESİNLİKLE YASAKTIR.
2. **Sentetik Testleri Kolaylaştırma (Cherry-Picking):**
   Hedef vektörleri veya test girdilerini, sistemin zaten kolayca bulabileceği aşırı dar / yakın parametre aralıklarına sıkıştırmak yasaktır.
3. **Sahte Kısırlık Platosu (Barren Plateau) Zaferleri:**
   Problemin değişkenlerinin %90'ını klasik yöntemle sabitleyip, kuantuma 2 kübit bırakarak "Barren plateau sorununu aştık" iddiasında bulunmak bilimsel sahteciliktir. 2 kübitlik devrede zaten barren plateau fiziksel olarak var olamaz.

---

## 🔬 2. Zorunlu Bilimsel Doğrulama Standartları

Herhangi bir yeni algoritma veya hibrit çözücü geliştirildiğinde şu 3 zorunlu test yapılmadan ZAFER İLAN EDİLEMEZ:

1. **Ablasyon Analizi (Ablation Study):**
   Hibrit sistemin her bir parçası tek tek test edilmelidir:
   - Sadece Klasik Yöntem tek başına ne yapıyor?
   - Sadece Kuantum tek başına ne yapıyor?
   - Hibrit sistem, parçaların toplamından GERÇEKTEN daha üstün bir sonuç üretiyor mu?
   Eğer klasik yöntem tek başına aynı sonucu alıyorsa, kuantumun veya bilişsel katmanın bir katkısı olmadığı açıkça belgelenmelidir.
2. **Adversarial / Gerçek Zorlukta Testler (Hard Instances):**
   Benchmark setleri, klasik yöntemlerin (örn. LLL/Babai) matematiksel olarak tıkandığı veya hata yaptığı zorlu parametre bölgelerini (örn. $\epsilon > \lambda_1 / 2$ veya gerçek LWE gürültü seviyelerini) MUTLAKA içermelidir.
3. **Metafor vs Gerçeklik Ayrımı:**
   "Zeno Etkisi", "SWR Replay", "CSF Kalkanı" gibi kavramların birer **sezgisel (heuristic) optimizasyon ve bellek tekniği** olduğu dürüstçe belirtilmeli; bunlar doğa kanunu veya kesin matematiksel ispat gibi sunulmamalıdır.

---

## ⚙️ 3. Quanta Çekirdek Kuralları

- **Python Ortamı:** `/Users/aes/Antigravity Projects/Alfa/quanta/.venv/bin/python` doğrudan kullanılmalıdır.
- **Sunum Dili:** Yönetici özeti sade, net, dürüst ve abartısız olmalıdır.
