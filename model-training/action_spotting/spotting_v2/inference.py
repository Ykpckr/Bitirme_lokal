import torch
import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import config as cfg
# Model Seçimi
if cfg.MODEL_TYPE == "transformer":
    from model import ActionTransformer as CurrentModel
elif cfg.MODEL_TYPE == "cnn":
    from model import CNNActionSpotter as CurrentModel
else:
    raise ValueError(f"Unknown MODEL_TYPE: {cfg.MODEL_TYPE}")

def load_model(checkpoint_path):
    print(f"📥 Model yükleniyor: {checkpoint_path}")
    model = CurrentModel().to(cfg.DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

def predict(model, feature_path, threshold=0.5):
    # Features Yükle
    print(f"📂 Özellikler okunuyor: {feature_path}")
    features = np.load(feature_path, mmap_mode='r')
    total_frames = features.shape[0]
    
    predictions = []
    
    # Sliding Window
    # Adım boyutu (Stride) pencerenin yarısı olabilir veya 1 frame
    # 1 Frame en hassas olanıdır ama yavaştır.
    # Biz test için stride=FPS (1 saniye) kullanalım, production'da 1 frame.
    stride = cfg.FPS 
    window_half = cfg.WINDOW_SIZE_FRAMES // 2
    
    print("running inference...")
    for frame_idx in tqdm(range(window_half, total_frames - window_half, stride)):
        # Pencereyi kes
        start = frame_idx - window_half
        end = frame_idx + window_half
        
        # (Window, Feat) -> (1, Window, Feat)
        chunk = features[start:end].copy()
        tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(cfg.DEVICE)
        
        with torch.no_grad():
            cls_logits, reg_offset = model(tensor)
            
        # Softmax yerine Sigmoid (Multi-label)
        # Focal Loss ile eğittik, yani Binary Classification mantığı var.
        probs = torch.sigmoid(cls_logits).squeeze(0) # (18,)
        
        # En yüksek skorlu sınıf
        score, cls_id = torch.max(probs[:-1], dim=0) # Background'ı hariç tut (son sınıf)
        
        if score.item() > threshold:
            # Zaman Düzeltmesi (Regression)
            # Offset tahmini Normalize edilmiş [-0.5, +0.5] aralığında
            # Frame'e çevirmek için Window Size ile çarpmalıyız.
            predicted_shift_frames = reg_offset.item() * cfg.WINDOW_SIZE_FRAMES
            event_frame = frame_idx + predicted_shift_frames
            event_time = event_frame / cfg.FPS
            
            predictions.append({
                "time": event_time,
                "label": cfg.ID_TO_EVENT[cls_id.item()],
                "score": score.item(),
                "frame": frame_idx
            })
            
    return predictions

def nms(predictions, window_sec=10):
    """Basit NMS: Yakın olanlardan skoru düşük olanı sil"""
    if not predictions: return []
    
    # Skora göre sırala
    predictions.sort(key=lambda x: x['score'], reverse=True)
    
    keep = []
    while predictions:
        current = predictions.pop(0)
        keep.append(current)
        
        # Sadece AYNI SINIFTAN olan ve yakın olanları sil
        # Farklı sınıftan (örn: Gol ve Kart) ise silme.
        predictions = [
            p for p in predictions 
            if not (p['label'] == current['label'] and abs(p['time'] - current['time']) < window_sec)
        ]
        
    # Zamana göre sırala
    keep.sort(key=lambda x: x['time'])
    return keep

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    
    model = load_model(args.checkpoint)
    preds = predict(model, args.features)
    final_preds = nms(preds)
    
    print(f"\n🎯 TESPİT EDİLEN OLAYLAR ({len(final_preds)}):")
    print("-" * 50)
    for p in final_preds:
        m, s = divmod(int(p['time']), 60)
        print(f"⏰ {m:02d}:{s:02d} - {p['label']:<20} ({p['score']:.2f})")
    print("-" * 50)
