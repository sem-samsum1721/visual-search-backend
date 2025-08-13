import os
import json
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# 1. Önceden eğitilmiş MobileNetV2 modelini yükle
# 'include_top=False' demek, modelin son katmanını (1000 sınıfı tahmin eden) atıp,
# bunun yerine bir önceki katmandaki zengin özellik vektörünü almamızı sağlar.
# pooling='avg' ise özellik haritasını tek bir vektöre indirger.
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Resimlerin bulunduğu klasörün yolu
# 'os.path.join' kullanmak, farklı işletim sistemlerinde (Windows/Mac/Linux)
# yol birleştirme işleminin hatasız yapılmasını sağlar.
IMAGE_FOLDER = os.path.join('urun_veritabani', 'images')

# Oluşturulacak JSON veritabanı dosyasının yolu ve adı
JSON_OUTPUT_FILE = os.path.join('urun_veritabani', 'urun_veritabani.json')

database = []

print("Veritabanı oluşturma işlemi başladı...")
print(f"Resimler '{IMAGE_FOLDER}' klasöründen okunacak.")

# Images klasöründeki her bir dosyayı işle
# 'os.listdir' ile klasördeki tüm dosya ve klasör isimlerini alıyoruz.
for i, filename in enumerate(os.listdir(IMAGE_FOLDER)):
    # Sadece resim dosyalarını (png, jpg, jpeg) işleme al
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # Resmin tam yolunu oluştur
            img_path = os.path.join(IMAGE_FOLDER, filename)

            # Resmi yükle ve modele uygun boyuta (224x224) getir
            img = image.load_img(img_path, target_size=(224, 224))
            
            # Resmi bir numpy dizisine (matematiksel bir matrise) çevir
            x = image.img_to_array(img)
            
            # Modelin bekledeği formata (batch) genişlet. Yani tek bir resim bile olsa
            # onu bir "grup" gibi göster. Boyut (224, 224, 3) iken (1, 224, 224, 3) olur.
            x = np.expand_dims(x, axis=0)
            
            # MobileNetV2'nin gerektirdiği ön işlemeyi yap (renk kanallarını vs. ayarlar)
            x = preprocess_input(x)
            
            # Model ile resmin özellik vektörünü çıkar. 'verbose=0' ekranda ilerleme çubuğu göstermesini engeller.
            feature_vector = base_model.predict(x, verbose=0).flatten()
            
            # Topladığımız bilgileri bir Python sözlüğüne (dictionary) kaydet
            product_data = {
                'id': i,
                'urun_adi': filename,
                'resim_yolu': img_path,
                # Numpy dizisini JSON'a kaydetmek için standart Python listesine çeviriyoruz.
                'vektor': feature_vector.tolist() 
            }
            
            database.append(product_data)
            print(f"{i+1}. ürün işlendi: {filename}")

        except Exception as e:
            print(f"Hata: {filename} dosyası işlenemedi. Hata detayı: {e}")

# Döngü bittikten sonra, toplanan tüm veriyi JSON dosyasına güzel bir formatta yaz
# 'w' -> write (yazma) modu, 'encoding='utf-8'' türkçe karakter desteği sağlar.
# 'indent=4' json dosyasının okunabilir, girintili olmasını sağlar.
with open(JSON_OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(database, f, ensure_ascii=False, indent=4)

print(f"\nİşlem tamamlandı! Veritabanı '{JSON_OUTPUT_FILE}' dosyasına kaydedildi.")