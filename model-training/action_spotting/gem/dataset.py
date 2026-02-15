# src/dataset.py
import os
import json
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import config as cfg

class SoccerNetDataset(Dataset):
    def __init__(self, split="train"):
        self.root_dir = cfg.ROOT_DIR
        self.split = split
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        print(f"[{self.split.upper()}] Veri (Pozitif + Negatif) hazırlanıyor...")
        
        for root, dirs, files in os.walk(self.root_dir):
            if "Labels-v2.json" not in files: continue
            
            # Baidu özellik dosyası kontrolü
            if not ("1_baidu_soccer_embeddings.npy" in files or "2_baidu_soccer_embeddings.npy" in files):
                continue

            try:
                with open(os.path.join(root, "Labels-v2.json"), 'r') as f:
                    labels = json.load(f)
            except: continue

            # Olayları topla
            events = []
            for ann in labels["annotations"]:
                if ann["label"] not in cfg.EVENT_DICTIONARY: continue
                game_time = ann["gameTime"]
                
                try:
                    half = int(game_time.split(' - ')[0])
                    time_parts = game_time.split(' - ')[1].split(':')
                    seconds = int(time_parts[0]) * 60 + int(time_parts[1])
                    frame = int(seconds * cfg.FRAMERATE)
                    events.append({'half': half, 'frame': frame, 'label': cfg.EVENT_DICTIONARY[ann["label"]]})
                except:
                    continue

            # STRATEJİ: Her olay için 1 Pozitif + 1 Negatif örnek
            for evt in events:
                # --- 1. POZİTİF ÖRNEK (OLAY) ---
                # Data Augmentation: Olayı pencerenin içinde hafifçe kaydır (-5 ile +5 kare)
                shift = random.randint(-5, 5) 
                center = evt['frame'] + shift
                
                samples.append({
                    "path": root,
                    "half": evt['half'],
                    "window_center": center, 
                    "event_frame": evt['frame'],
                    "label": evt['label'],      # utils.py 'label' bekliyor
                    "is_background": False      # utils.py bunu bekliyor
                })

                # --- 2. NEGATİF ÖRNEK (BACKGROUND) ---
                # Olaydan rastgele bir uzaklıkta (örn: 10-50 sn uzağında) boş bir an seç
                # Bu sayede model "Olay Yok" durumunu da öğrenir.
                bg_shift = random.choice([-100, -80, -60, 60, 80, 100]) # Frame cinsinden uzaklık
                bg_center = evt['frame'] + bg_shift
                
                samples.append({
                    "path": root,
                    "half": evt['half'],
                    "window_center": bg_center,
                    "event_frame": -1,          # Olay yok
                    "label": -1,                # Etiket yok
                    "is_background": True       # Bu bir background
                })
        
        print(f"Toplam eğitilecek pencere sayısı: {len(samples)}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. Özellikleri Yükle
        feat_path = os.path.join(item['path'], f"{item['half']}_baidu_soccer_embeddings.npy")
        try:
            full_features = np.load(feat_path, mmap_mode='r')
        except:
            return torch.zeros((cfg.FEATURE_DIM, cfg.WINDOW_FRAME)), torch.zeros((cfg.NUM_CLASSES, cfg.WINDOW_FRAME))
        
        # 2. GÜVENLİ PENCERE KESME
        total_frames = full_features.shape[0]
        feature_dim = full_features.shape[1]
        half_win = cfg.WINDOW_FRAME // 2
        
        start_idx = item['window_center'] - half_win
        end_idx = item['window_center'] + half_win
        
        buffer = np.zeros((cfg.WINDOW_FRAME, feature_dim), dtype=np.float32)
        
        valid_start = max(0, start_idx)
        valid_end = min(total_frames, end_idx)
        
        buf_start = valid_start - start_idx
        buf_end = buf_start + (valid_end - valid_start)
        
        if valid_end > valid_start:
            buffer[buf_start:buf_end] = full_features[valid_start:valid_end].copy()
            
        features = torch.from_numpy(buffer).float().transpose(0, 1)

        # 3. GAUSSIAN SOFT LABEL OLUŞTURMA
        target = torch.zeros((cfg.NUM_CLASSES, cfg.WINDOW_FRAME), dtype=torch.float32)
        
        # Eğer bu bir Background örneği ise hedef tamamen 0 kalır (Sessizlik).
        # Eğer Olay örneği ise Gaussian tepeciği oluşturulur.
        if not item['is_background']:
            relative_event_pos = item['event_frame'] - start_idx
            
            if 0 <= relative_event_pos < cfg.WINDOW_FRAME:
                x = np.arange(0, cfg.WINDOW_FRAME)
                gaussian = np.exp(-0.5 * ((x - relative_event_pos) / cfg.LABEL_SIGMA) ** 2)
                target[item['label'], :] = torch.from_numpy(gaussian).float()

        return features, target