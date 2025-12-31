import cv2
import numpy as np
import torch
from ultralytics import YOLO
import sys
from pathlib import Path

# --- ÖNCEKİ DOSYADAN SINIFLARI ÇAĞIRIYORUZ ---
# Dosyanın adı 'team_utils.py' ise:
try:
    from team_clasifier import AutoLabEmbedder, AutomaticTeamClusterer
except ImportError:
    print("❌ HATA: 'team_utils.py' dosyası bulunamadı!")
    print("   Lütfen önceki çalışan kodunu 'team_utils.py' adıyla aynı klasöre kaydet.")
    sys.exit()

# --- KÜTÜPHANE KONTROLÜ ---
try:
    from boxmot import ByteTrack
except ImportError:
    print("❌ HATA: 'boxmot' eksik.")
    sys.exit()

# --- AYARLAR ---
YOLO_MODEL_PATH = 'best.pt'
VIDEO_PATH = "1_720p.mkv"
CALIBRATION_FRAMES = 1300

# !!! ReID MODELİNİN YOLU !!!
REID_MODEL_PATH = r"C:/FoMAC_Logs/resnet50_triplet_384/best_model.pt"

def run_system():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Çalışma Ortamı: {device}")
    
    # 1. Modelleri Hazırla
    detector = YOLO(YOLO_MODEL_PATH)
    
    # --- ReID Modeli Kontrolü ---
    reid_path = Path(REID_MODEL_PATH)
    if reid_path.exists():
        print(f"🧠 ReID Modeli Bulundu: {reid_path.name}")
        reid_weights = reid_path
    else:
        print("⚠️ ReID Modeli Bulunamadı! Takip 'Motion Only' modunda çalışacak.")
        reid_weights = None

    # --- DİĞER DOSYADAN GELEN SINIFLARI BAŞLAT ---
    embedder = AutoLabEmbedder()
    clusterer = AutomaticTeamClusterer()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # ==========================================
    # AŞAMA 1: OTOMATİK VERİ TOPLAMA (Kalibrasyon)
    # ==========================================
    print("\n🕵️ AŞAMA 1: Takım Renkleri Öğreniliyor...")
    
    frame_idx = 0
    samples = 0
    
    while samples < CALIBRATION_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        # Hızlanmak için kare atla
        if frame_idx % 3 != 0: continue
        
        results = detector(frame, verbose=False)
        if len(results[0].boxes) == 0: continue
        
        dets = results[0].boxes.data.cpu().numpy()
        
        for det in dets:
            # Sadece İnsan (0) ve Yüksek Güven (>0.70)
            if int(det[5]) == 0 and det[4] > 0.70:
                x1, y1, x2, y2 = det[:4].astype(int)
                
                if (y2-y1) < 50: continue # Küçükleri at
                
                # Sınır kontrolü
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                crop = frame[y1:y2, x1:x2]
                
                # Diğer dosyadaki fonksiyonu kullanıyoruz:
                feat = embedder.get_features(crop)
                
                if feat is not None:
                    clusterer.collect(feat)
                    samples += 1
                    sys.stdout.write(f"\r⏳ Toplanan Örnek: {samples}/{CALIBRATION_FRAMES}")
                    sys.stdout.flush()

    print("\n\n🧠 Yapay Zeka Karar Veriyor...")
    if not clusterer.train():
        print("❌ Veri yetersiz, video çok kısa veya kimse yok.")
        return

    # ==========================================
    # AŞAMA 2: PROFESYONEL TAKİP (ReID + Lab Cluster)
    # ==========================================
    print("\n🎬 AŞAMA 2: ReID Destekli Takip Başlıyor!")
    
    # Videoyu başa sar
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Tracker'ı başlat (ReID ile)
    # track_buffer=120: Oyuncu kadrajdan çıksa bile 4 saniye boyunca kimliğini hatırlar.
    tracker = ByteTrack(
        reid_weights=reid_weights, 
        device=device, 
        half=True, 
        frame_rate=30,
        track_buffer=120
    )
    
    color_map = {0: (255, 100, 0), 1: (0, 0, 255)} # Görsel renkler (Mavi - Kırmızı)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Tespit
        results = detector(frame, verbose=False)
        dets = results[0].boxes.data.cpu().numpy()
        if len(dets) == 0: dets = np.empty((0, 6))
        
        # Takip (ReID burada çalışır)
        tracks = tracker.update(dets, frame)
        
        for track in tracks:
            bbox = track[:4].astype(int)
            track_id = int(track[4])
            cls_id = int(track[6])
            
            color = (200, 200, 200) # Bilinmiyor
            label = ""
            
            if cls_id == 0: # Oyuncu
                x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])
                crop = frame[y1:y2, x1:x2]
                
                # Diğer dosyadaki sınıfları kullanarak takım tahmini
                feat = embedder.get_features(crop)
                team_id = clusterer.predict(feat)
                
                if team_id != -1:
                    color = color_map[team_id]
                    # ID'yi ReID sağlıyor, Takımı Lab Cluster sağlıyor
                    label = f"ID:{track_id} T{team_id}"
                else:
                    label = f"ID:{track_id}"
            
            elif cls_id == 3: # Top
                color = (0, 255, 0)
                label = "Top"
                
            # Çizim
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            if label:
                cv2.putText(frame, label, (bbox[0], bbox[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Göster
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('Main ReID Tracker', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()