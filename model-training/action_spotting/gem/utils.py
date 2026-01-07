# src/utils.py dosyasının tamamını bununla değiştir:
import torch
import numpy as np
import config as cfg

def calculate_weights_sqrt(dataset):
    print("Sınıf ağırlıkları hesaplanıyor (SQRT)...")
    targets = []
    for item in dataset.samples:
        if not item['is_background']:
            targets.append(item['label'])
    
    class_counts = np.bincount(targets, minlength=cfg.NUM_CLASSES)
    total_samples = len(dataset)
    pos_weights = []
    
    for cls_id in range(cfg.NUM_CLASSES):
        pos_count = max(1, class_counts[cls_id])
        neg_count = total_samples - pos_count
        raw_ratio = neg_count / pos_count
        weight = np.sqrt(raw_ratio)
        weight = max(1.0, min(weight, 100.0))
        pos_weights.append(weight)
        
    return torch.tensor(pos_weights, dtype=torch.float32).to(cfg.DEVICE)

def calculate_metrics(pred_logits, target_cls):
    """
    Accuracy ve F1 Score hesaplar.
    DÜZELTME: Threshold tekrar 0.5 yapıldı.
    """
    with torch.no_grad():
        probs = torch.sigmoid(pred_logits)
        
        # >>>> GERİ DÖNÜŞ: STANDART 0.5 EŞİĞİ <<<<
        preds = (probs > 0.5).float()
        
        # --- Accuracy ---
        is_event = target_cls.sum(dim=1) > 0
        correct_event = (probs.argmax(dim=1) == target_cls.argmax(dim=1)) & is_event
        # Background'u doğru bilmesi için tahminin 0.5'ten küçük olması lazım
        correct_bg = (probs.max(dim=1)[0] < 0.5) & (~is_event)
        
        acc = (correct_event.sum() + correct_bg.sum()).float() / pred_logits.size(0)

        # --- F1 Score ---
        tp = (preds * target_cls).sum(dim=0)
        fp = (preds * (1 - target_cls)).sum(dim=0)
        fn = ((1 - preds) * target_cls).sum(dim=0)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return acc.item(), f1.mean().item()