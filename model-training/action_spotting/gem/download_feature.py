from SoccerNet.Downloader import SoccerNetDownloader as SNdl
import os

# Senin ana klasörün
LOCAL_DIRECTORY = r"C:/FoMAC_Dataset/action_spotting"

if not os.path.exists(LOCAL_DIRECTORY):
    os.makedirs(LOCAL_DIRECTORY)

# İndiriciyi senin klasörüne ayarla
mySNdl = SNdl(LocalDirectory=LOCAL_DIRECTORY)

print(f"📥 İndirme konumu: {LOCAL_DIRECTORY}")

# 1. Etiketleri İndir (Gol, Faul vb. nerede?)
mySNdl.downloadGames(
    files=["Labels-v2.json"], 
    split=["train","valid","test","challenge"] # Challenge da ekledim eksik kalmasın
)

# 2. ResNet Özelliklerini İndir (Modelin Gözü)
# Not: TF2 versiyonu genelde daha günceldir, onu indirmen iyi olmuş.
mySNdl.downloadGames(
    files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], 
    split=["train","valid","test","challenge"]
)

# 3. Baidu Özelliklerini İndir (Alternatif Model Gözü)
# Eğer diskte yer sorunun varsa bunu indirmeyebilirsin, ResNet yeterlidir.
# Ama denemek istersen kalsın.
mySNdl.downloadGames(
    files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"], 
    split=["train","valid","test","challenge"]
)

print("✅ Tüm dosyalar hiyerarşik yapı korunarak indirildi.")