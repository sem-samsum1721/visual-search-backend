# Yapay Zeka Destekli Görsel Arama API (FastAPI Prototipi) 🧠

Bu proje, bir mobil uygulamaya (veya herhangi bir istemciye) görsel ürün arama yeteneği kazandıran, Python, FastAPI ve TensorFlow ile geliştirilmiş bir REST API servisidir. Bir resim yüklemesi aldığında, bu resmi analiz eder ve bir veritabanındaki ürünlerle karşılaştırarak en benzer olanları bir benzerlik skoruyla birlikte döndürür.

Bu proje, bir Flutter mobil uygulamasının backend'i olarak hizmet vermesi amacıyla **hızlı prototipleme** yaklaşımıyla geliştirilmiştir.

---

### ✨ Öne Çıkan Özellikler

*   **Hızlı ve Modern API:** Yüksek performanslı web framework'ü olan **FastAPI** kullanılarak geliştirilmiştir.
*   **Yapay Zeka Entegrasyonu:** Görüntü analizi için, önceden eğitilmiş bir derin öğrenme modeli olan **TensorFlow/Keras (MobileNetV2)** kullanılmıştır.
*   **Görsel Benzerlik Hesaplama:** Yüklenen bir resim ile veritabanındaki resimlerin özellik vektörleri (feature vectors) arasındaki benzerlik, `scikit-learn` kütüphanesi kullanılarak **kosinüs benzerliği (cosine similarity)** ile hesaplanır.
*   **Asenkron Yapı:** FastAPI'nin doğal asenkron desteği sayesinde, aynı anda birden fazla isteği verimli bir şekilde yönetebilir.

---

### 🛠️ Kullanılan Teknolojiler

*   **Dil:** Python 3.11+
*   **Framework:** FastAPI
*   **AI / Makine Öğrenmesi:**
    *   TensorFlow / Keras
    *   Scikit-learn
    *   NumPy
*   **Web Sunucusu:** Uvicorn

---

### ⚙️ Kurulum ve Çalıştırma

1.  **Projeyi Klonla:**
    ```bash
    git clone https://github.com/sem-samsum1721/visual-search-backend.git
    ```
2.  **Klasöre Git:**
    ```bash
    cd visual-search-backend
    ```
3.  **Sanal Ortam Oluştur ve Aktif Et:**
    ```bash
    python -m venv .venv
    # Windows için:
    .\.venv\Scripts\Activate.ps1
    ```
4.  **Gerekli Kütüphaneleri Yükle:**
    *   Projenin ana dizininde bir `requirements.txt` dosyası oluşturun ve içine `fastapi`, `uvicorn`, `tensorflow`, `scikit-learn`, `numpy`, `Pillow` gibi paketleri ekleyin.
    *   ```bash
      pip install -r requirements.txt
      ```
5.  **Veritabanını Oluştur:**
    *   `urun_veritabani/images/` klasörüne kendi ürün resimlerinizi ekleyin.
    *   ```bash
      python veritabani_olustur.py
      ```
6.  **Sunucuyu Başlat:**
    ```bash
    uvicorn main_api:app --reload
    ```
7.  **Test Et:** Tarayıcınızda `http://127.0.0.1:8000/docs` adresine giderek interaktif API dokümantasyonunu kullanabilirsiniz.

---

### 🔗 Frontend Projesi

Bu API servisi, aşağıdaki Flutter mobil uygulamasına hizmet vermek için geliştirilmiştir:
[visual-search-flutter-app](https://github.com/sem-samsum1721/visual-search-flutter-app)
