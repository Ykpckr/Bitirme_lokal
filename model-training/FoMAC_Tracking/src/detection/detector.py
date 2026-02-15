import sys
import numpy as np
import torch
from ultralytics import YOLO
import logging

# Loglama ayarı
logger = logging.getLogger(__name__)

class YOLODetector:
    """
    YOLO modelini yöneten ve tracker için uygun çıktı üreten sınıf.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (Config): config_utils.py ile yüklenen ayar objesi
        """
        self.cfg = cfg
        self.model_path = cfg.detection.get('model_path')
        self.conf_thres = cfg.detection.get('conf_thres', 0.45)
        self.iou_thres = cfg.detection.get('iou_thres', 0.5)
        self.classes = cfg.detection.get('classes', [0, 1]) # Default: 0=Player, 1=Ball

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Detector başlatılıyor... Cihaz: {self.device}")
        
        try:
            self.model = YOLO(self.model_path)
            # Modeli GPU'ya taşı (varsa)
            self.model.to(self.device)
            logger.info(f"Model başarıyla yüklendi: {self.model_path}")
        except Exception as e:
            logger.error(f"Model yüklenirken hata oluştu: {e}")
            sys.exit(1)

    def detect(self, frame):
        """
        Gelen kare üzerinde nesne tespiti yapar.

        Args:
            frame (np.ndarray): OpenCV formatında görüntü (BGR)

        Returns:
            np.ndarray: Tespit edilen nesneler.
                        Format: [[x1, y1, x2, y2, score, class_id], ...]
        """
        # verbose=False ile konsol kirliliğini engelliyoruz
        results = self.model(
            frame, 
            conf=self.conf_thres, 
            iou=self.iou_thres, 
            classes=self.classes,
            verbose=False
        )[0]

        detections = []

        if results.boxes:
            # CPU'ya al ve numpy'a çevir
            boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            scores = results.boxes.conf.cpu().numpy() # [score]
            class_ids = results.boxes.cls.cpu().numpy() # [class_id]

            # Hepsini tek bir matriste birleştir: [N, 6]
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                score = scores[i]
                cls_id = int(class_ids[i])
                
                detections.append([x1, y1, x2, y2, score, cls_id])

        return np.array(detections)

if __name__ == "__main__":
    # --- TEST BLOĞU ---
    # Bu dosyayı doğrudan çalıştırırsan modülü test edebilirsin.
    # Örn: python src/detection/detector.py
    
    import cv2
    from src.utils.config_utils import load_config
    
    # Config yükle
    try:
        cfg = load_config('configs/config.yaml')
        detector = YOLODetector(cfg)
        
        # Test için boş siyah bir resim oluştur veya gerçek resim yükle
        print("Test görüntüsü oluşturuluyor...")
        fake_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Tespit yap
        dets = detector.detect(fake_frame)
        print(f"Tespit sonucu (Boş resim olduğu için boş dönmeli): {dets}")
        print("Detector modülü başarıyla çalışıyor.")
        
    except Exception as e:
        print(f"Test sırasında hata: {e}")