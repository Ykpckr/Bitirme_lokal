import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import sys

# --- GEREKLİ KÜTÜPHANELER ---
try:
    from boxmot import ByteTrack
except ImportError:
    print("❌ HATA: 'boxmot' kütüphanesi bulunamadı.")
    sys.exit()

# --- AYARLAR ---
YOLO_MODEL_PATH = 'best.pt' 
VIDEO_PATH = "2_720p.mkv" 

# ❌ ReID Model Yolu ARTIK YOK.

def run_tracker():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Kullanılan Cihaz: {device}")

    print("YOLO Modeli Yükleniyor...")
    detector = YOLO(YOLO_MODEL_PATH)
    
    print("⚡ Saf Hareket Takibi (Motion Only) Başlatılıyor...")
    
    try:
        # ReID ağırlıklarını vermiyoruz (None)
        tracker = ByteTrack(
            reid_weights=None,  # <--- BURASI ARTIK BOŞ
            device=device,
            half=True,
            
            # --- HAREKET TABANLI AYARLAR ---
            det_thresh=0.25,   # Tespit eşiği
            track_buffer=60,   # Hafıza (2 saniye yeterli, ReID yokken çok uzun tutmak ID karıştırabilir)
            match_thresh=0.80, # IoU Eşiği (Kutular ne kadar üst üste biniyor?)
            frame_rate=30
        )
        print("✅ ByteTrack (Motion Mode) Hazır!")
    except Exception as e:
        print(f"\n❌ TRACKER HATASI: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Hata: Video açılamadı!")
        return

    print("\n✅ TAKİP BAŞLADI! (ReID Kapalı)")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        # 1. Tespit (YOLO)
        results = detector(frame, verbose=False)
        dets = results[0].boxes.data.cpu().numpy()

        if len(dets) == 0:
            dets = np.empty((0, 6))

        # 2. Takip (Sadece Hareket)
        # ReID olmadığı için tracker sadece kutuların kayma hızına bakacak
        tracks = tracker.update(dets, frame)

        # 3. Çizim
        for track in tracks:
            bbox = track[:4].astype(int)
            track_id = int(track[4])
            cls_id = int(track[6])

            color = (0, 255, 0)
            label_text = f"ID:{track_id}"

            if cls_id == 0: # Oyuncu
                color = (255, 140, 0) # Turuncu
            elif cls_id == 3: # Top
                color = (0, 0, 255) # Kırmızı
                label_text = "Top"

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (bbox[0], bbox[1]-20), (bbox[0]+w, bbox[1]), color, -1)
            cv2.putText(frame, label_text, (bbox[0], bbox[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Bilgi
        cv2.putText(frame, f"Mode: MOTION ONLY | Frame: {frame_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('FoMAC Motion Tracker', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracker()