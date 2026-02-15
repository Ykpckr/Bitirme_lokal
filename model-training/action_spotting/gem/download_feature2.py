from SoccerNet.Downloader import SoccerNetDownloader as SNdl
import os
import time

# Senin ayarladığın klasör yolu
LOCAL_DIRECTORY = r"C:/FoMAC_Dataset/action_spotting"

if not os.path.exists(LOCAL_DIRECTORY):
    os.makedirs(LOCAL_DIRECTORY)

mySNdl = SNdl(LocalDirectory=LOCAL_DIRECTORY)

# İndirilmesi gereken dosyaların listesi
dosyalar = [
    "Labels-v2.json",
    "1_ResNET_TF2_PCA512.npy", 
    "2_ResNET_TF2_PCA512.npy",
    "1_baidu_soccer_embeddings.npy", 
    "2_baidu_soccer_embeddings.npy"
]

# Hangi veri setleri?
splits = ["train", "valid", "test", "challenge"]

print(f"🔄 Eksik dosyalar taranıyor ve tamamlanıyor: {LOCAL_DIRECTORY}\n")

for split in splits:
    for dosya in dosyalar:
        basarili = False
        deneme_sayisi = 0
        max_deneme = 5  # Hata verirse en fazla 5 kere daha denesin

        while not basarili and deneme_sayisi < max_deneme:
            try:
                # SoccerNet kütüphanesi dosya varsa zaten indirmeden geçer.
                # Biz burada sadece hata olursa yakalayıp tekrar deniyoruz.
                print(f"⏳ Kontrol ediliyor: [{split}] -> {dosya} (Deneme: {deneme_sayisi + 1})")
                
                mySNdl.downloadGames(files=[dosya], split=[split], verbose=False)
                
                print(f"✅ Tamam: {split}/{dosya}")
                basarili = True  # Döngüden çık
                
            except Exception as e:
                deneme_sayisi += 1
                print(f"⚠️ HATA: {e}")
                print(f"zzz... {deneme_sayisi * 3} saniye bekleniyor...")
                time.sleep(deneme_sayisi * 3) # Her hatada bekleme süresini artır

        if not basarili:
            print(f"❌ BAŞARISIZ: {split}/{dosya} 5 denemede indirilemedi. Sonra tekrar dene.")

print("\n🏁 Tarama bitti. İnmeyen kaldıysa internetini kontrol edip tekrar çalıştırabilirsin.")