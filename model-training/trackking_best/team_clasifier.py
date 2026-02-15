import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from ultralytics import YOLO
import sys

# --- KÜTÜPHANE KONTROLÜ ---
try:
    from boxmot import ByteTrack
except ImportError:
    print("❌ HATA: 'boxmot' kütüphanesi eksik. 'pip install boxmot' ile kurabilirsin.")
    sys.exit()

# --- AYARLAR ---
YOLO_MODEL_PATH = 'best.pt'   # Senin modelin
VIDEO_PATH = "1_720p.mkv"     # Video dosyan
CALIBRATION_FRAMES = 1300     # Analiz için kaç kareye bakılsın? (Otomatik öğrenme süresi)

class AutoLabEmbedder:
    """
    Bu sınıf oyuncunun görüntüsünden IŞIĞI SİLER, sadece RENK ÖZÜNÜ (Lab) alır.
    Gölgedeki kırmızıyı ve güneşteki kırmızıyı aynı sayıya çevirir.
    """
    def get_features(self, image):
        if image.size == 0: return None
        
        # 1. TORSO CROP (Akıllı Kesim)
        # Kafayı (%15) ve Şortu/Bacakları (%40) atıyoruz. 
        # Sadece göğüs kısmındaki forma rengine odaklanıyoruz.
        h, w, _ = image.shape
        y1 = int(h * 0.15)
        y2 = int(h * 0.60)
        x1 = int(w * 0.20)
        x2 = int(w * 0.80)
        
        if (y2-y1) < 5 or (x2-x1) < 5: return None # Çok küçükse at
        crop = image[y1:y2, x1:x2]
        
        # 2. YEŞİL (ÇİM) TEMİZLİĞİ
        # HSV uzayında yeşilleri bulup maskeliyoruz.
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        not_green = cv2.bitwise_not(mask) # Yeşil olmayan alanlar
        
        # 3. LAB DÖNÜŞÜMÜ (Gölge Savar)
        lab_image = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
        
        # Sadece yeşil olmayan (oyuncu) piksellerini al
        valid_pixels = lab_image[not_green > 0]
        
        if len(valid_pixels) < 10: return None # Yeterli piksel yoksa at
        
        # 4. ORTALAMA AL (L Kanalını At)
        # Lab = [Lightness, a, b]
        # Biz sadece [a, b] kanallarını alıyoruz. Işık (L) umrumuzda değil.
        mean_lab = np.mean(valid_pixels, axis=0)
        color_vector = mean_lab[1:3] # [a, b]
        
        return color_vector

class AutomaticTeamClusterer:
    """
    Bu sınıf toplanan verileri K-Means ile otomatik gruplar.
    İnteraktif seçime gerek kalmaz.
    """
    def __init__(self):
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        self.data_bank = []
        self.trained = False
        
    def collect(self, vector):
        if vector is not None:
            self.data_bank.append(vector)
            
    def train(self):
        if len(self.data_bank) < 50:
            print("⚠️ Yetersiz veri toplandı!")
            return False
            
        print(f"📊 {len(self.data_bank)} oyuncu verisi üzerinden OTOMATİK öğrenme yapılıyor...")
        self.kmeans.fit(self.data_bank)
        self.trained = True
        
        # Merkezleri yazdır (Bilgi amaçlı)
        c1 = self.kmeans.cluster_centers_[0]
        c2 = self.kmeans.cluster_centers_[1]
        print(f"✅ Takımlar Kilitlendi! (Lab Merkezleri: {c1} vs {c2})")
        
        self.data_bank = [] # Hafızayı temizle
        return True
        
    def predict(self, vector):
        if not self.trained or vector is None: return -1
        return self.kmeans.predict([vector])[0]

def run_auto_tracker():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Çalışma Ortamı: {device}")
    
    # 1. Modeller
    detector = YOLO(YOLO_MODEL_PATH)
    tracker = ByteTrack(reid_weights=None, device=device, half=True, frame_rate=30)
    
    embedder = AutoLabEmbedder()
    clusterer = AutomaticTeamClusterer()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # --- AŞAMA 1: OTOMATİK VERİ TOPLAMA (Ekrana görüntü basmadan hızlıca) ---
    print("\n🕵️ AŞAMA 1: Video taranıyor ve takım renkleri öğreniliyor...")
    print("    (Bu işlem birkaç saniye sürecek, lütfen bekleyin)\n")
    
    frame_idx = 0
    samples = 0
    
    while samples < CALIBRATION_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        # Her 3. kareye bak (Hızlanmak için)
        if frame_idx % 3 != 0: continue
        
        results = detector(frame, verbose=False)
        if len(results[0].boxes) == 0: continue
        
        dets = results[0].boxes.data.cpu().numpy()
        
        for det in dets:
            # Sadece İnsan (Class 0) ve Yüksek Güvenilirlik (>0.70)
            if int(det[5]) == 0 and det[4] > 0.70:
                x1, y1, x2, y2 = det[:4].astype(int)
                
                # Çok küçük veya çok kenardaki kutuları alma
                if (y2-y1) < 50: continue 
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                crop = frame[y1:y2, x1:x2]
                
                # Özellik çıkar ve havuza at
                feat = embedder.get_features(crop)
                if feat is not None:
                    clusterer.collect(feat)
                    samples += 1
                    sys.stdout.write(f"\r⏳ Toplanan Örnek: {samples}/{CALIBRATION_FRAMES}")
                    sys.stdout.flush()

    # --- EĞİTİM ---
    print("\n\n🧠 Yapay Zeka Karar Veriyor...")
    if not clusterer.train():
        print("❌ Eğitim başarısız oldu (Yetersiz veri). Video kalitesini kontrol et.")
        return

    # --- AŞAMA 2: TAKİP VE ÇİZİM ---
    print("\n🎬 AŞAMA 2: Takip Başlıyor! (Gölge Korumalı)")
    
    # Videoyu başa sar
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # Tracker'ı sıfırla (Temiz başlangıç)
    tracker = ByteTrack(reid_weights=None, device=device, half=True, frame_rate=30)
    
    # Renk Haritası (Takım 0 ve Takım 1 için görsel renkler)
    # K-Means hangisine 0 hangisine 1 dediğini bilemeyiz ama iki zıt renk yeterli.
    color_map = {0: (255, 100, 0), 1: (0, 0, 255)} # Mavi vs Kırmızı
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Tespit
        results = detector(frame, verbose=False)
        dets = results[0].boxes.data.cpu().numpy()
        if len(dets) == 0: dets = np.empty((0, 6))
        
        # Takip
        tracks = tracker.update(dets, frame)
        
        for track in tracks:
            bbox = track[:4].astype(int)
            track_id = int(track[4])
            cls_id = int(track[6])
            
            color = (200, 200, 200) # Gri (Bilinmiyor/Hakem)
            label = ""
            
            if cls_id == 0: # Oyuncu
                x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])
                crop = frame[y1:y2, x1:x2]
                
                # Tahmin Et (Lab Vektörü ile)
                feat = embedder.get_features(crop)
                team_id = clusterer.predict(feat)
                
                if team_id != -1:
                    color = color_map[team_id]
                    label = f"T{team_id}"
            
            elif cls_id == 3: # Top (Eğer eğitilmişse)
                color = (0, 255, 0)
                label = "Top"
                
            # Çizim
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            if label:
                cv2.putText(frame, label, (bbox[0], bbox[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Göster
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('Auto Lab Tracker (Shadow Proof)', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_auto_tracker()