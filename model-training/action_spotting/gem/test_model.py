# test_model.py
import torch
import numpy as np
import os
import config as cfg
from model import RMSNet

# --- AYARLAR ---
# DİKKAT: Yeni eğitim başlattığın için buradaki ismi güncellemelisin!
# Örneğin: "arra7_gaussian_ep1.pth" (Eğitimden çıkan ilk dosyayı dene)
MODEL_PATH = "arra7_gaussian_ep1.pth"  

# Test edilecek maçın yolu
TEST_FEATURE_FILE = r"C:\FoMAC_Dataset\action_spotting\europe_uefa-champions-league\2015-2016\2015-09-15 - 21-45 Galatasaray 0 - 2 Atl. Madrid\1_baidu_soccer_embeddings.npy"

# Eşik değeri (Bunu duruma göre 0.10 veya 0.20 yapabilirsin)
DEBUG_THRESHOLD = 0.05 

def load_model(path):
    print(f"Model yükleniyor: {path}")
    model = RMSNet().to(cfg.DEVICE)
    model.load_state_dict(torch.load(path, map_location=cfg.DEVICE))
    model.eval()
    return model

def test_single_match(model, feature_path):
    if not os.path.exists(feature_path):
        print(f"HATA: Dosya bulunamadı -> {feature_path}")
        return

    print(f"Maç analiz ediliyor... ({os.path.basename(os.path.dirname(feature_path))})")
    
    # Özellikleri yükle (Hata almamak için .copy() kullanıyoruz)
    features = np.load(feature_path, mmap_mode='r')
    total_frames = features.shape[0]
    win_size = cfg.WINDOW_FRAME
    step = win_size // 2 
    
    print(f"Toplam Kare: {total_frames} | Tahmini Süre: {int(total_frames / cfg.FRAMERATE / 60)} dakika")
    print("-" * 80)
    print(f"{'ZAMAN':<8} {'EN GÜÇLÜ TAHMİN':<25} {'ALTERNATİF 1':<20} {'ALTERNATİF 2'}")
    print("-" * 80)
    
    found_any = False

    for i in range(0, total_frames - win_size, step):
        # Pencereyi kopyala
        window = features[i : i + win_size].copy()
        
        # Tensor Hazırla: (Batch, Feat, Time) -> (1, 8576, 40)
        inp = torch.from_numpy(window).float().to(cfg.DEVICE)
        inp = inp.transpose(0, 1).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(inp)
            probs = torch.sigmoid(logits)
        
        # --- ZAMANSAL MAX POOLING ---
        # Pencere içindeki 40 karenin en yüksek skorunu al
        probs_max_time, _ = torch.max(probs, dim=2) # Shape: (1, 17)
        
        # --- TOP-3 TAHMİN ---
        top_vals, top_inds = torch.topk(probs_max_time, k=3, dim=1)
        
        score1 = top_vals[0][0].item()
        
        # Eğer en güçlü tahmin eşiği geçiyorsa yazdır
        if score1 > DEBUG_THRESHOLD:
            found_any = True
            center_frame = i + win_size // 2
            seconds = center_frame / cfg.FRAMERATE
            minutes = int(seconds // 60)
            sec = int(seconds % 60)
            
            # İsimleri bul
            names = []
            scores = []
            for k in range(3):
                sc = top_vals[0][k].item()
                idx = top_inds[0][k].item()
                # Sözlükten ismi çek
                name = [key for key, val in cfg.EVENT_DICTIONARY.items() if val == idx][0]
                names.append(name)
                scores.append(sc)
            
            # Satırı oluştur
            row = f"{minutes}:{sec:02d}    "
            row += f"{names[0]} (%{int(scores[0]*100)})".ljust(25)
            row += f"{names[1]} (%{int(scores[1]*100)})".ljust(20)
            row += f"{names[2]} (%{int(scores[2]*100)})"
            
            print(row)

    print("-" * 80)
    if not found_any:
        print("⚠️ Hiçbir olay tespit edilemedi. (Eşik değeri çok yüksek olabilir veya model henüz öğrenmedi)")
    else:
        print("✅ Analiz Tamamlandı.")

if __name__ == "__main__":
    # Eğer model dosyası henüz oluşmadıysa uyarı verelim
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        test_single_match(model, TEST_FEATURE_FILE)
    else:
        print(f"HATA: Model dosyası bulunamadı -> {MODEL_PATH}")
        print("Eğitimin 1. Epoch'unun bitmesini bekle veya dosya adını kontrol et.")