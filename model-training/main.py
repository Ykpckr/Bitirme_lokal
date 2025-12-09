import sys
import logging
import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Klasör yapısına uygun importlar
from utils.visualization_utils import (
    get_video_info, draw_detections, add_text_overlay, ensure_dir
)
from utils.config_utils import create_cli_parser
from trackers.deepsort_tracker import DeepSortTracker
from spotters.event_spotter import EventSpotter

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

def safe_tracks_to_dicts(tracks):
    """DeepSORT çıktısını görselleştirici formatına çevirir."""
    out = []
    for t in tracks:
        # t formatı: (track_id, [x1, y1, x2, y2])
        if isinstance(t, (tuple, list)) and len(t) >= 2:
            out.append({'track_id': int(t[0]), 'bbox': [float(x) for x in t[1]]})
    return out

def main():
    parser = create_cli_parser()
    parser.add_argument('--model', required=True, help='YOLO model path (örn: yolov8n.pt)')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--save_video', action='store_true', help='Save processed video')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()

    # Yollar
    model_path = Path(args.model)
    video_path = Path(args.video)
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    # Modeli Yükle
    logger.info(f"YOLO modeli yükleniyor: {model_path}")
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        logger.error(f"Model yüklenemedi: {e}")
        return

    # Video Bilgisi
    vid_info = get_video_info(video_path)
    if vid_info['fps'] is None:
        logger.error("Video açılamadı veya bilgileri okunamadı.")
        return

    fps = vid_info['fps']
    frame_w, frame_h = vid_info['width'], vid_info['height']
    logger.info(f"Video: {video_path.name} | FPS: {fps:.2f} | Boyut: {frame_w}x{frame_h}")

    # Tracker Başlat (ReID modeli olmadan geometry-only modunda çalışır)
    tracker = DeepSortTracker(
        max_age=200,        # Bir obje kaybolduktan sonra kaç kare hafızada tutulsun
        max_cosine=0.8,    # Benzerlik eşiği
        max_spatial_dist=250
    )

    # Spotter Başlat
    # window: Hareketli ortalama penceresi (FPS'e göre dinamik)
    spotter = EventSpotter(
        window=int(fps/2), 
        shot_threshold=15.0,     # Şut algılama eşiği (px/frame hızı)
        accel_threshold=100.0,   # İvme eşiği
        fps=fps
    )

    # Video Yakalama
    cap = cv2.VideoCapture(str(video_path))
    
    save_writer = None
    if args.save_video:
        out_vid_path = output_dir / f"processed_{video_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        save_writer = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (frame_w, frame_h))
        logger.info(f"Kayıt başlatıldı: {out_vid_path}")

    frame_idx = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # 1. Tespit (YOLO)
            results = model(frame, conf=0.45, verbose=False, iou=0.5)[0]
            
            detections = [] # [x1, y1, x2, y2, cls, conf]
            if results.boxes:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                clss = results.boxes.cls.cpu().numpy()
                for box, conf, cls in zip(boxes, confs, clss):
                    # x1, y1, x2, y2, cls, conf
                    detections.append([*box, int(cls), float(conf)])

            # 2. Takip (DeepSORT)
            # Tracker sadece [x1,y1,x2,y2] bekler
            tracker_inputs = [d[:4] for d in detections]
            
            # Tracker güncelle (features=None olduğu için geometrik takip yapar)
            tracks = tracker.update(tracker_inputs, features=None, frame_id=frame_idx)
            
            # Format dönüştürme
            final_tracks = []
            for t in tracker.tracks:
                # 1. Onaylanmış mı? (Confirmed)
                # 2. En az 5 karedir hayatına devam ediyor mu? (hits >= 5)
                # 3. Son 1 karedir görüldü mü? (time_since_update <= 1 -> Ghosting engeller)
                if t.is_confirmed and t.hits >= 5 and t.time_since_update <= 1:
                    final_tracks.append({'track_id': t.track_id, 'bbox': t.to_xyxy()})
            
            tracked_objects = [(t['track_id'], t['bbox']) for t in final_tracks] # Spotter için
            tracks_dict_list = final_tracks # Çizim için

            # 3. Olay Algılama (Spotter)
            spot_info = spotter.update(frame_idx, detections, tracked_objects)

            # 4. Görselleştirme
            annotated_frame = draw_detections(frame.copy(), detections, tracks_dict_list, spot_info)
            
            # Bilgi Ekranı
            info_text = [
                f"Frame: {frame_idx}/{vid_info['frames']}",
                f"FPS: {frame_idx / (time.time() - start_time):.1f}",
                f"Tracks: {len(tracks)}",
                f"Events: {len(spotter.events)}"
            ]
            add_text_overlay(annotated_frame, info_text)

            # Göster ve Kaydet
            cv2.imshow("Tracking & Spotting", annotated_frame)
            if save_writer:
                save_writer.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Kullanıcı tarafından durduruldu.")
                break

    except KeyboardInterrupt:
        logger.info("Durduruluyor...")
    except Exception as e:
        logger.exception(f"Hata oluştu: {e}")
    finally:
        cap.release()
        if save_writer:
            save_writer.release()
        cv2.destroyAllWindows()
        
        # Olayları kaydet
        event_file = output_dir / f"{video_path.stem}_events.json"
        spotter.save_events(event_file)

if __name__ == "__main__":
    main()