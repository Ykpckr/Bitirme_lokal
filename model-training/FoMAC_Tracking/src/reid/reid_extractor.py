import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2
import logging
import sys

# Torchreid kütüphanesini içe aktarmayı dene
try:
    import torchreid
except ImportError:
    print("HATA: 'torchreid' kütüphanesi bulunamadı.")
    print("Lütfen şu komutla kurun: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git")
    sys.exit(1)

logger = logging.getLogger(__name__)

class ReIDExtractor:
    def __init__(self, cfg):
        """
        Re-Identification modelini başlatır.
        
        Args:
            cfg (Config): config_utils.py ile gelen ayarlar
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = cfg.reid.get('model_name', 'osnet_x1_0')
        self.input_size = cfg.reid.get('input_size', [256, 128]) # [H, W]
        
        logger.info(f"ReID Modeli yükleniyor: {self.model_name} (Cihaz: {self.device})")
        
        # 1. Modeli oluştur (Torchreid üzerinden)
        # num_classes önemli değil çünkü sadece feature extraction (özellik çıkarma) yapacağız
        self.model = torchreid.models.build_model(
            name=self.model_name,
            num_classes=1000, 
            loss='softmax',
            pretrained=True # ImageNet üzerinde eğitilmiş ağırlıkları indir
        )
        
        self.model.to(self.device)
        self.model.eval() # Eğitim modunu kapat
        
        # 2. Ön işleme (Preprocessing) transformları
        # ReID modelleri genelde normalize edilmiş, belirli boyutta resim ister
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, frame, boxes):
        """
        Verilen kareden, belirtilen kutuların (oyuncuların) özellik vektörlerini çıkarır.

        Args:
            frame (np.ndarray): Orijinal BGR görüntü (OpenCV formatı)
            boxes (list or np.ndarray): [x1, y1, x2, y2] formatında kutular

        Returns:
            np.ndarray: [N, 512] boyutunda özellik vektörleri (Embeddings)
        """
        if len(boxes) == 0:
            return np.empty((0, 512))

        # 1. Görüntüden oyuncuları kırp (Crop) ve Tensor'a çevir
        crops = []
        h, w, _ = frame.shape
        
        # Görüntüyü RGB'ye çevir (OpenCV BGR kullanır, Torch RGB ister)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Sınır kontrolleri (Resim dışına taşmayı engelle)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Eğer kutu çok küçükse veya hatalıysa atla
            if x2 <= x1 or y2 <= y1:
                # Boş bir siyah resim ekle (Hata vermemesi için)
                crop = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
            else:
                crop = frame_rgb[y1:y2, x1:x2]
            
            # Transform uygula
            crop_tensor = self.transform(crop)
            crops.append(crop_tensor)

        if not crops:
            return np.empty((0, 512))

        # 2. Batch oluştur ve GPU'ya at
        batch = torch.stack(crops).to(self.device)

        # 3. Modelden geçir (Inference)
        with torch.no_grad():
            features = self.model(batch)
        
        # 4. Normalizasyon ve CPU'ya dönüş
        # ReID için Cosine Distance kullanılacağından vektörleri normalize etmek kritiktir
        features = F.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy()

if __name__ == "__main__":
    # --- TEST BLOĞU ---
    # python src/reid/reid_extractor.py
    
    from src.utils.config_utils import load_config
    
    print("ReID Modülü test ediliyor...")
    
    # 1. Config yükle
    cfg = load_config('configs/config.yaml')
    
    # 2. Modeli başlat
    extractor = ReIDExtractor(cfg)
    
    # 3. Sahte veri ile dene
    fake_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    fake_boxes = [[100, 100, 200, 300], [400, 400, 450, 550]] # İki rastgele kutu
    
    features = extractor.extract(fake_frame, fake_boxes)
    
    print(f"Girdi Kutuları: {len(fake_boxes)}")
    print(f"Çıktı Vektör Boyutu: {features.shape}") # (2, 512) olmalı
    
    if features.shape == (2, 512):
        print("✅ ReID Modülü Başarıyla Çalıştı!")
    else:
        print("❌ HATA: Çıktı boyutu yanlış.")