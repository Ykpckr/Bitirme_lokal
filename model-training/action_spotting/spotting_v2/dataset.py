import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import config as cfg

class SoccerNetDataset(Dataset):
    def __init__(self, split="train", augment=True):
        """
        Args:
            split (str): "train", "valid", "test" veya "challenge".
            augment (bool): Masking gibi veri artırma tekniklerini uygula.
        """
        self.split = split
        self.augment = augment
        self.samples = [] # (feature_path, frame_index, label_id, offset)
        
        print(f"📂 Dataset hazırlanıyor: {split.upper()} (Feat: {cfg.FEATURE_TYPE}, FPS: {cfg.FPS})")
        self._load_metadata()
        
    def _load_metadata(self):
        """
        Dosya sistemini tarar ve eğitim örneklerini (samples) oluşturur.
        Labels-v2.json dosyasındaki her olayı bir örnek olarak ekler.
        Ayrıca dengeli eğitim için rastgele Background örnekleri de ekler.
        """
        # Dataset kök klasörü altındaki ligleri gez (Spain, England, vs.)
        search_dir = cfg.DATASET_DIR
        
        # SoccerNet klasör yapısı: Lig / Sezon / Maç / ...
        # Tüm Labels-v2.json dosyalarını bul
        # (Bu işlem biraz sürebilir, o yüzden optimize edilebilir ama şimdilik güvenli yol)
        # Pratiklik için: Split dosyalarını kullanmak daha iyidir ama biz klasör tarayacağız.
        
        count_events = 0
        count_videos = 0
        
        for root, dirs, files in os.walk(search_dir):
            if "Labels-v2.json" in files:
                # İlgili split mi? (Basitçe klasör adına veya listeye bakılabilir)
                # Şimdilik tüm datayı tarayıp train/valid diye ayırmak yerine
                # Klasör yapısına göre filtreleme yapılabilir. 
                # Standart SoccerNet splitleri ayrı dosyalarda verilir ama biz manuel split yapalım:
                # Örn: %80 Train, %10 Valid, %10 Test (Hash based veya random)
                
                # Basit Split Mantığı: Maç isminin hash'ine göre
                # Bu sayede her çalıştırışta aynı maçlar aynı sete düşer.
                match_name = os.path.basename(root)
                match_hash = hash(match_name) % 100
                
                is_train = match_hash < 80     # %80
                is_valid = 80 <= match_hash < 90 # %10
                # is_test = match_hash >= 90
                
                # Split kontrolü
                if self.split == "train" and not is_train: continue
                if self.split == "valid" and not is_valid: continue
                if self.split == "test" and match_hash < 90: continue # Sadece test set
                
                # Feature dosyasını bul (1. ve 2. devre)
                for half in [1, 2]:
                    feat_file = f"{half}{cfg.FEATURE_SUFFIX}"
                    feat_path = os.path.join(root, feat_file)
                    
                    if not os.path.exists(feat_path):
                        continue
                        
                    # Özellik dosyasının boyutunu (frame sayısını) öğrenmek için mmap ile aç
                    try:
                        # Mmap modunda sadece shape okur, RAM'e yüklemez
                        feats_mmap = np.load(feat_path, mmap_mode='r')
                        total_frames = feats_mmap.shape[0]
                    except:
                        continue
                        
                    # Etiketleri yükle
                    label_path = os.path.join(root, "Labels-v2.json")
                    with open(label_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Olayları işle
                    for ann in data['annotations']:
                        # 1. Devre kontrolü (gameTime: "1 - 05:00")
                        game_half = int(ann['gameTime'].split(' - ')[0])
                        if game_half != half:
                            continue
                            
                        # 2. Sınıf kontrolü
                        label_name = ann['label']
                        if label_name not in cfg.EVENT_DICTIONARY:
                            continue
                        
                        label_id = cfg.EVENT_DICTIONARY[label_name]
                        
                        # 3. Zaman/Frame hesaplama (KRİTİK)
                        # position değeri her zaman milisaniyedir (SoccerNet-v2).
                        position_ms = int(ann['position'])
                        
                        # Frame Index = (ms / 1000) * FPS
                        center_frame = int((position_ms / 1000.0) * cfg.FPS)
                        
                        # 4. Temporal Jittering (Sadece TRAIN modunda)
                        # Olayın tam üstünü değil, biraz sağını solunu da alalım ki
                        # model "ne kadar uzaktayım" (offset) bilgisini öğrensin.
                        shift = 0
                        if self.split == "train" and self.augment:
                            max_shift = cfg.FPS * 2 # +/- 2 saniye kayabilir
                            shift = random.randint(-max_shift, max_shift)
                        
                        # Yeni merkez (Modelin baktığı pencerenin ortası)
                        anchor_frame = center_frame + shift
                        
                        # Offset Hesaplama (Hedeflenen - Şu anki)
                        # DÜZELTME (SOTA): Offset normalize edilmeli ki model loss patlamasın.
                        # [-0.5, 0.5] aralığına sıkıştır.
                        offset_val = -shift / cfg.WINDOW_SIZE_FRAMES
                        
                        # 5. Sınır kontrolü (Pencere dışına taşmamalı)
                        half_window = cfg.WINDOW_SIZE_FRAMES // 2
                        if anchor_frame < half_window or anchor_frame >= total_frames - half_window:
                            continue
                            
                        # Örneği Ekle: (Dosya, Merkez Frame, Sınıf, Offset)
                        self.samples.append({
                            'path': feat_path,
                            'frame': anchor_frame, # ARTIK KAYDIRILMIŞ FRAME
                            'label': label_id,
                            'offset': offset_val,  # ARTIK 0 DEĞİL!
                            'is_background': False
                        })
                        count_events += 1
                    
                    
                    # Background Sampling (Veri Dengeleme)
                    # Safe Background: Rastgele seçilen frame bir olaya denk geliyor mu?
                    # Eğer geliyorsa, o background'ı çöpe at.
                    num_bg = 10 
                    attempts = 0
                    added_bg = 0
                    
                    # O videodaki olayların frame listesi
                    event_frames = [s['frame'] for s in self.samples if s['path'] == feat_path and not s['is_background']]
                    safe_margin = cfg.FPS * 2 # Olayın +/- 2 saniye yakınına background koyma
                    
                    while added_bg < num_bg and attempts < num_bg * 5:
                         attempts += 1
                         rand_frame = random.randint(half_window, total_frames - half_window - 1)
                         
                         is_safe = True
                         for ev_f in event_frames:
                             if abs(rand_frame - ev_f) < safe_margin:
                                 is_safe = False
                                 break
                        
                         if is_safe:
                             self.samples.append({
                                'path': feat_path,
                                'frame': rand_frame,
                                'label': cfg.BACKGROUND_CLASS,
                                'offset': 0.0,
                                'is_background': True
                            })
                             added_bg += 1
                    
                    count_videos += 1
        
        print(f"   ✅ Yüklendi: {len(self.samples)} örnek ({count_events} olay) - {count_videos} video parçasından.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. Feature Yükle (Sadece pencereyi oku)
        # Mmap mode 'r' ile dosya açılır, sadece slice okunur. Hızlıdır.
        feat_mmap = np.load(sample['path'], mmap_mode='r')
        
        center = sample['frame']
        half_window = cfg.WINDOW_SIZE_FRAMES // 2
        
        # Window sınırları
        start = center - half_window
        end = center + half_window
        
        # Özellikleri kesip al (Numpy -> Tensor)
        window_feats = feat_mmap[start:end].copy() # Copy önemli (mmap'ten memory'e al)
        window_feats = torch.from_numpy(window_feats).float()
        
        # 2. RMS-Net Masking Strategy (Sadece eğitimde ve olaylarda)
        # Olayın öncesini (%50 ihtimalle) sıfırla.
        if self.augment and self.split == 'train' and not sample['is_background']:
            if random.random() < cfg.MASK_PROB:
                # Yarısına kadar olan kısmı maskele (Olay öncesi)
                # window_feats[0:half_window] = 0
                # Veya rastgele bir noise/background ile değiştir.
                # Sıfırlamak en temiz yöntemdir (Missing data simülasyonu)
                window_feats[:half_window] = 0.0
        
        # 3. Etiket ve Offset
        label = sample['label']
        offset = sample['offset'] # Şimdilik 0.0
        
        # Dönüş: (Input, Class Label, Regression Offset)
        # Class label long olmalı (CrossEntropy için), ama BCE için float one-hot yapılacak (Loss içinde).
        return window_feats, torch.tensor(label).long(), torch.tensor(offset).float()

# Test Bloğu
if __name__ == "__main__":
    ds = SoccerNetDataset(split="train")
    if len(ds) > 0:
        f, l, o = ds[0]
        print(f"Örnek Çıktı Shape: {f.shape}") # Beklenen: (20, 8576)
        print(f"Sample Label: {l}, Offset: {o}")
