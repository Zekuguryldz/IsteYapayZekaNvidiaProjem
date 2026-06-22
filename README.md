# NVIDIA ile Yapay Zekaya Giriş — Online Staj Deneyimi Modülü

Bu repo, **NVIDIA ile Yapay Zekaya Giriş Online Staj Deneyimi Modülü** (İş'te Yapay Zeka Programı — İSTKA Destekli · Nextern.ai · NVIDIA DLI) kapsamında geliştirilen makine öğrenmesi sınıflandırma projesidir. Çalışma Google Colab ortamında (**NVIDIA Tesla T4 GPU**) iki farklı veri seti üzerinde yürütülmüştür:

- **Iris** veri seti — çok sınıflı sınıflandırma
- **Breast Cancer (Meme Kanseri)** veri seti — ikili sınıflandırma

**Hazırlayan:** Zeki Uğur Yıldız
**İletişim:** zek.uguryildiz@gmail.com

---

## İçerik

- [Proje Hakkında](#proje-hakkında)
- [Kullanılan Ortam ve Araçlar](#kullanılan-ortam-ve-araçlar)
- [1. Iris Veri Seti Sınıflandırması](#1-iris-veri-seti-sınıflandırması)
- [2. Breast Cancer Veri Seti Sınıflandırması](#2-breast-cancer-veri-seti-sınıflandırması)
- [Genel Çıkarımlar](#genel-çıkarımlar)
- [Dosyalar](#dosyalar)

---

## Proje Hakkında

Projenin amacı, iki ayrı veri seti üzerinde farklı sınıflandırma algoritmalarını eğitmek, performanslarını çeşitli metriklerle (Doğruluk, Kesinlik, Duyarlılık, F1-Skoru) karşılaştırmak ve her problem için en güvenilir modeli belirlemektir. Çalışma; aşırı öğrenme (overfitting) davranışını ve metrik seçiminin problem bağlamına göre nasıl değiştiğini de incelemektedir.

## Kullanılan Ortam ve Araçlar

- **Google Colab** + **NVIDIA Tesla T4 GPU** (`nvidia-smi` ile doğrulandı)
- **Python**
- **Pandas** — veri analizi (`df.head()`, `df.shape`, `df.columns`, `df.describe()`, `df.info()`, `isnull()`)
- **Matplotlib / Seaborn** — görselleştirme (pairplot, countplot, boxplot, korelasyon haritası)
- **Scikit-learn** — modeller ve metrikler (`train_test_split`, `accuracy_score`, `recall_score`, `precision_score`, `f1_score`)

---

## 1. Iris Veri Seti Sınıflandırması

Klasik **Iris** veri seti (150 örnek, 4 özellik, 3 tür: *setosa*, *versicolor*, *virginica*) üzerinde çok sınıflı sınıflandırma yapılmıştır. Veri hem UCI kaynağından (`bezdekIris.data`) hem de `sklearn.datasets.load_iris` ile yüklenmiş, eksik değer kontrolü ve `pairplot` ile keşifsel analiz yapılmıştır.

- **Train/Test ayrımı:** `test_size=0.2`, `random_state=42` → 120 eğitim / 30 test örneği

### Sonuçlar

| Model | Test Doğruluğu (Accuracy) |
|-------|:-------------------------:|
| Lojistik Regresyon | **%100** |
| Karar Ağacı | **%100** |
| Random Forest | **%100** |
| SVM | **%100** |

Dört modelin de test setinde %100 doğruluk üretmesi dikkat çekicidir. Iris, sınıfları (özellikle *setosa*) oldukça iyi ayrışan küçük bir veri setidir; ancak tüm modellerin kusursuz skor vermesi, pratikte her zaman olağan bir durum değildir ve **hedef/veri sızıntısı (data leakage)** ya da veri setinin yapısı gibi konuların incelenmesini gerektirir. Bu gözlem, yalnızca accuracy'ye bakmanın yeterli olmadığını gösterir ve bir sonraki çalışmadaki (Breast Cancer) daha derin metrik analizinin motivasyonunu oluşturur.

---

## 2. Breast Cancer Veri Seti Sınıflandırması

**Breast Cancer** veri seti (569 örnek, 30 özellik; sınıflar: *malignant/kötü huylu* = 212, *benign/iyi huylu* = 357) üzerinde ikili sınıflandırma yapılmıştır. Sınıf dağılımı, kutu grafikleri ve korelasyon haritası ile keşifsel analiz gerçekleştirilmiştir.

- **Train/Test ayrımı:** `test_size=0.2`, `random_state=42` → 455 eğitim / 114 test örneği
- **Modeller:** Lojistik Regresyon · Karar Ağacı · Random Forest

### Sonuçlar (Model Karşılaştırma Tablosu)

Metrikler, kötü huylu sınıfın (kritik sınıf) tespiti üzerinden hesaplanmıştır:

| Model | Accuracy | Recall | Precision | F1-Score |
|-------|:--------:|:------:|:---------:|:--------:|
| Lojistik Regresyon | 0.9561 | 0.9070 | 0.9750 | 0.9398 |
| Karar Ağacı | 0.9474 | 0.9302 | 0.9302 | 0.9302 |
| **Random Forest** | **0.9649** | 0.9302 | **0.9756** | **0.9524** |

**En güvenilir model: Random Forest.** Hem en yüksek genel doğruluğu (0.9649) hem de en yüksek F1-Skorunu (0.9524) elde ederek, hatalı pozitifleri ve hatalı negatifleri dengeli biçimde minimize etmiştir.

### Overfitting (Aşırı Öğrenme) Analizi

E�itim ve test doğrulukları karşılaştırıldığında:

- **Lojistik Regresyon:** Eğitim 0.9582 / Test 0.9561 — aradaki fark yalnızca **0.0021**. Aşırı öğrenmeden uzak, genellenebilirliği yüksek bir model.
- **Karar Ağacı:** Eğitim setinde ~%100 doğruluğa ulaşırken testte en sert düşüşü yaşar — klasik **overfitting** davranışı.
- **Random Forest:** Eğitimde yüksek skora ulaşsa da topluluk öğrenmesi sayesinde testte sert düşüş yaşamaz; varyansı kontrol altında tutar.

### Bu Çalışmadan Çıkarım: Tıbbi Bağlamda Metrik Seçimi

Meme kanseri probleminde yalnızca **accuracy** yeterli değildir. Kötü huylu bir tümörün iyi huylu olarak tahmin edilmesi (False Negative) hastanın hayatını riske atar. Bu nedenle **Precision** ve özellikle **Recall** kritik öneme sahiptir; **F1-Skoru** bu ikisi arasındaki dengeyi sağlayarak modelin gerçek güvenilirliğini ortaya koyar.

---

## Genel Çıkarımlar

- Aynı algoritmalar farklı veri setlerinde farklı davranır; model seçimi her zaman probleme ve veriye bağlıdır.
- **Accuracy tek başına yanıltıcı olabilir** — Iris'te tüm modeller %100 verirken, asıl ayrıştırıcı analiz Breast Cancer'da recall/precision/F1 ile yapılabilmiştir.
- Karar Ağacı eğitim verisini ezberlemeye (overfitting) eğilimliyken, topluluk yöntemi olan **Random Forest** bu riski azaltarak en sağlam (robust) sonucu vermiştir.

## Dosyalar

| Dosya | Açıklama |
|-------|----------|
| [`NVIDIA_Zeki_Ugur_Yildiz.ipynb`](./NVIDIA_Zeki_Ugur_Yildiz.ipynb) | Projenin tüm kodunu içeren Google Colab notebook'u |
| [`Nvidia_Final_Proje_Sunum_Sablonu.pptx`](./Nvidia_Final_Proje_Sunum_Sablonu.pptx) | Proje sunumu |
| [`Iris_Siniflandirma_Raporu.docx`](./Iris_Siniflandirma_Raporu.docx) | Iris sınıflandırma raporu |
