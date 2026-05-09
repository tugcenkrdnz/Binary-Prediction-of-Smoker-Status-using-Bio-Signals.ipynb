# Kaggle S3E24 - 🚭 Sigara Kullanımı Tahmin Sistemi (Binary Classification)

Bu proje, biyometrik veriler ve kan tahlili sonuçlarını kullanarak bir bireyin sigara içip içmediğini tahmin etmeyi amaçlar. Kaggle'ın **Playground Series S3E24** yarışması kapsamında geliştirilmiştir.

## 🎯 Projenin Amacı

Sağlık taramalarında (check-up) elde edilen rutin laboratuvar sonuçlarını (Hemoglobin, Gtp, Kolesterol vb.) analiz ederek, sigara kullanımının vücut kimyası üzerindeki izlerini makine öğrenmesi ile tespit etmek.

## 📊 Veri Seti ve Analiz (EDA)

Veri seti sentetik olup, gerçek sağlık taraması verileri üzerine inşa edilmiştir.

* **Kritik Özellikler:** `hemoglobin`, `Gtp` (karaciğer enzimi) ve `height(cm)` sigara kullanımı ile en yüksek korelasyona sahip değişkenler olarak belirlenmiştir.
* **Özellik Mühendisliği:** Uç değerleri (outliers) normalize etmek için `Log Transformation` uygulanmış; ayrıca `BMI`, `AST/ALT oranı` gibi tıbbi metrikler yeni özellik olarak eklenmiştir.

## 🛠️ Kullanılan Teknolojiler

* **Dil:** Python 3.x
* **Kütüphaneler:** Pandas, NumPy, Scikit-learn
* **Model:** LightGBM (Gradient Boosting Machine)
* **Arayüz:** Streamlit (Web Uygulaması)
* **Metrik:** ROC-AUC

## 🚀 Model Performansı

Model, 5-Fold Stratified Cross-Validation yöntemiyle eğitilmiştir.

* **Cross-Validation AUC:** 0.8673
* **Public Leaderboard AUC:** 0.8669

## 💻 Kurulum ve Çalıştırma

1. Projeyi bilgisayarınıza indirin:
```bash
git clone https://github.com/tugcenkrdnz/Binary-Prediction-of-Smoker-Status-using-Bio-Signals.ipynb.git
cd sigara-tahmin-projesi

```


2. Gerekli kütüphaneleri kurun:
```bash
pip install -r requirements.txt

```


3. Modeli ve web arayüzünü çalıştırın:
```bash
streamlit run app.py

```



## 📈 Sonuç

Bu model, sigara kullanımının biyobelirteçler üzerindeki etkisini yüksek bir isabet oranıyla saptayabilmektedir. Özellikle karaciğer enzimleri ve kandaki oksijen taşıma kapasitesindeki (hemoglobin) değişimler, modelin karar verme sürecindeki en güçlü sinyalleri oluşturmaktadır.

---

### Dosya Yapısı:

* `app.py`: Streamlit arayüz kodu.
* `lgbm_smoking_model.pkl`: Eğitilmiş ve kaydedilmiş model.
* `requirements.txt`: Gerekli kütüphaneler listesi.
* `notebook.ipynb`: Veri analizi ve eğitim süreci (isteğe bağlı eklenebilir).
