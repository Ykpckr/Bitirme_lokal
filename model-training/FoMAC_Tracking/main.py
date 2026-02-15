import cv2
import time
import logging
from pathlib import Path
from tqdm import tqdm

# Kendi modüllerimizi çağırıyoruz
from src.utils.config_utils import load_config
from src.utils.visualizer import Visualizer
from src.detection.detector import YOLODetector
from src.reid.reid_extractor import ReIDExtractor
from src.tracker.tracker import Tracker
from src.spotter.event_spotter import EventSpotter

# Loglama Ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MAIN")

def main():
    # 1. Ayarları Yükle
    logger.info("Sistem başlatılıyor...")
    cfg = load_config("configs/config.yaml")
    
    # Çıktı klasörünü oluştur
    out_dir = Path(cfg.video['output_folder'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Modülleri Başlat
    detector = YOLODetector(cfg)
    reid = ReIDExtractor(cfg)
    tracker = Tracker(cfg)
    spotter = EventSpotter(cfg)
    visualizer = Visualizer()

    # 3. Video Kaynağını Aç
    video_path = cfg.video['input_path']
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Video açılamadı: {video_path}")
        return

    # Video Bilgileri
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Video Kaydedici (Eğer aktifse)
    writer = None
    if cfg.video['save_video']:
        save_path = out_dir / "tracked_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
        logger.info(f"Video kaydı yapılacak: {save_path}")

    # --- ANA DÖNGÜ ---
    frame_idx = 0
    start_time = time.time()
    
    # Tqdm ile ilerleme çubuğu
    pbar = tqdm(total=total_frames, desc="İşleniyor", unit="frame")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # A. DETECTION (YOLO)
            # Çıktı: [x1, y1, x2, y2, score, cls]
            detections = detector.detect(frame)
            
            # Tracker için sadece oyuncuları (Class 0) filtrele
            # Top (Class 1) ayrı işlenecek
            player_dets = [d for d in detections if int(d[5]) == 0]
            player_boxes = [d[:4] for d in player_dets]
            
            # B. ReID (Özellik Çıkarma)
            # Sadece oyuncu kutularından embedding çıkar
            if len(player_boxes) > 0:
                embeddings = reid.extract(frame, player_boxes)
            else:
                embeddings = []
            
            # C. TRACKING (DeepSORT)
            # Player detections + Embeddings -> Tracker Update
            tracks = tracker.update(player_dets, embeddings)
            
            # D. SPOTTER (Olay Analizi)
            # Hem detection (top için) hem tracks (oyuncu için) gönderilir
            spot_info = spotter.update(frame_idx, detections, tracks)
            
            # E. GÖRSELLEŞTİRME
            # 1. Topu çiz (Raw detection'dan)
            visualizer.draw_ball(frame, detections)
            # 2. Oyuncuları çiz (Tracker'dan)
            visualizer.draw_tracks(frame, tracks)
            # 3. İstatistikleri çiz
            current_fps = frame_idx / (time.time() - start_time)
            visualizer.draw_stats(frame, spot_info, frame_idx, current_fps)
            
            # F. KAYIT VE GÖSTERİM
            if writer:
                writer.write(frame)
                
            if cfg.video['show_video']:
                # Görüntüyü biraz küçült (Ekranı kaplamaması için)
                display_frame = cv2.resize(frame, (1280, 720))
                cv2.imshow("FoMAC Tracking System", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            pbar.update(1)

    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından durduruldu.")
    except Exception as e:
        logger.exception("Beklenmedik bir hata oluştu.")
    finally:
        # Temizlik
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        pbar.close()
        
        # Sonuçları Kaydet
        spotter.save_events(out_dir)
        logger.info("İşlem tamamlandı.")

if __name__ == "__main__":
    main()