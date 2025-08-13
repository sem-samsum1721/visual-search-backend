import json
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

#  modeli ve gerekli araçları import ediyoruz
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model

# benzerlik hesaplama
from sklearn.metrics.pairwise import cosine_similarity

# --- UYGULAMA BAŞLANGICI ---

print("FastAPI uygulaması başlatılıyor...")

#  FastAPI uygulamasını oluşturma
app = FastAPI(title="Akıllı Görsel Ürün Arama API")

#eğitilmiş modeli yükle 
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
print("Yapay zeka modeli belleğe yüklendi.")

# JSON veritabanını yükleme
DATABASE_PATH = 'urun_veritabani/urun_veritabani.json'
with open(DATABASE_PATH, 'r', encoding='utf-8') as f:
    database = json.load(f)
print(f"Ürün veritabanı '{DATABASE_PATH}' dosyasından yüklendi. {len(database)} ürün bulundu.")




def preprocess_image(img_bytes: bytes):
    """Gelen resmi modele uygun formata getirir."""
    img = Image.open(io.BytesIO(img_bytes))
    # Resmi RGB'ye çevir (eğer 4 kanallı PNG ise)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# --- API ENDPOINT (Mobil Uygulamanın İletişim Kuracağı Adres) ---

@app.post("/gorsel-ile-ara/")
async def find_similar_products(image_file: UploadFile = File(...)):
    """
    Yüklenen bir görsele en çok benzeyen ürünleri veritabanından bulur.
    """
    try:
        
        contents = await image_file.read()
        
        #  ön işleme
        processed_image = preprocess_image(contents)
        
        #  özellik vektörünü model ile çıkar
        query_vector = base_model.predict(processed_image).flatten()
        
        results = []
        
        # Veritabanındaki her bir ürünle karşılaştır
        for product in database:
            # Vektörler arasında kosinüs benzerliğini hesapla
            # Vektörleri 2D array haline getirmemiz gerekiyor
            db_vector = np.array(product['vektor'])
            similarity = cosine_similarity(query_vector.reshape(1, -1), db_vector.reshape(1, -1))[0][0]
            
            results.append({
                'urun_adi': product['urun_adi'],
                'benzerlik_skoru': float(similarity)
            })
            
        # Sonuçları en yüksek benzerlik skoruna göre sırala
        sorted_results = sorted(results, key=lambda x: x['benzerlik_skoru'], reverse=True)
        
        # En çok benzeyen ilk 5 ürünü döndür
        return JSONResponse(
            status_code=200,
            content={"results": sorted_results[:5]}
        )

    except Exception as e:
        # Bir hata olursa, hatayı JSON olarak döndür
        return JSONResponse(
            status_code=500,
            content={"error": f"Bir hata oluştu: {str(e)}"}
        )

@app.get("/")
def read_root():
    return {"message": "Akıllı Görsel Arama API'sine hoş geldiniz! Test için /docs adresine gidin."}
