import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import numpy as np
import sys
import time

# Dosya yolları
sys.path.append(str(Path(__file__).parent))

import config as cfg
from dataset import SoccerNetDataset
from loss import ActionSpottingLoss

# Model Seçimi
if cfg.MODEL_TYPE == "transformer":
    from model import ActionTransformer as CurrentModel
elif cfg.MODEL_TYPE == "cnn":
    from model import CNNActionSpotter as CurrentModel
else:
    raise ValueError(f"Unknown MODEL_TYPE: {cfg.MODEL_TYPE}")

# -----------------------------------------------------------------------------
# METRİK HESAPLAMA (mAP)
# -----------------------------------------------------------------------------
def compute_mAP(y_true, y_scores):
    """
    Her sınıf için Average Precision (AP) hesaplar ve ortalamasını (mAP) alır.
    y_true: (N, NumClasses) - One-Hot veya Multi-label Ground Truth
    y_scores: (N, NumClasses) - Modelin Sigmoid çıktıları (0-1 arası)
    """
    ap_list = []
    num_classes = y_true.shape[1]
    
    # Her sınıf için ayrı ayrı hesapla
    for i in range(num_classes):
        # Sadece bu sınıfın doğruları ve skorları
        y_t = y_true[:, i]
        y_s = y_scores[:, i]
        
        # Eğer o sınıftan hiç örnek yoksa pas geç (0.0)
        if y_t.sum() == 0:
            ap_list.append(0.0)
            continue
            
        # Skorlara göre büyükten küçüğe sırala
        sorted_indices = np.argsort(-y_s)
        sorted_scores = y_s[sorted_indices]
        sorted_truth = y_t[sorted_indices]
        
        # Cumulative True Positives & False Positives
        tp = sorted_truth
        fp = 1 - tp
        
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        
        # Precision & Recall
        precision = tp_cum / (tp_cum + fp_cum + 1e-8)
        recall = tp_cum / (tp_cum[-1] + 1e-8)
        
        # Average Precision (Riemann Sum / Area Under Curve)
        # Basitleştirilmiş: P * deltaR
        # Veya pratik yöntem: Sadece TP noktalarındaki Precision'ların ortalaması
        ap = np.sum(precision * tp) / (np.sum(tp) + 1e-8)
        ap_list.append(ap)
        
    return np.mean(ap_list) * 100  # Yüzdelik dilim

# -----------------------------------------------------------------------------
# EĞİTİM DÖNGÜSÜ
# -----------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval() # Değerlendirme modu (Dropout kapalı, BatchNorm sabit)
    
    total_loss = 0
    total_cls = 0
    total_reg = 0
    correct = 0
    total_samples = 0
    
    # mAP Hesaplaması için tüm tahminleri saklayacağız (Sadece Validasyonda)
    all_targets = []
    all_scores = []
    
    loop = tqdm(loader, desc="Train" if is_train else "Valid", leave=False)
    
    for i, (feats, labels, offsets) in enumerate(loop):
        feats = feats.to(device)
        labels = labels.to(device)
        offsets = offsets.to(device)
        
        # Forward Pass
        # Validation'da gradient hesaplama (Hız ve Bellek tasarrufu)
        with torch.set_grad_enabled(is_train):
            cls_logits, reg_preds = model(feats)
            loss_dict = criterion(cls_logits, reg_preds, labels, offsets)
            loss = loss_dict['total']
            
            # Backward Pass (Sadece Train)
            if is_train:
                loss = loss / cfg.ACCUMULATION_STEPS
                loss.backward()
                
                if (i + 1) % cfg.ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
        
        # Metrikler
        batch_size = labels.size(0)
        total_loss += loss.item() * (cfg.ACCUMULATION_STEPS if is_train else 1)
        total_cls += loss_dict['cls'].item()
        total_reg += loss_dict['reg'].item()
        
        # Accuracy
        preds = torch.argmax(cls_logits, dim=1)
        correct += (preds == labels).sum().item()
        total_samples += batch_size
        
        # mAP Verisi Topla (Background sınıfını dahil etmiyoruz)
        if not is_train:
            # Logits -> Sigmoid (Olasılık)
            probs = torch.sigmoid(cls_logits).detach().cpu().numpy()
            
            # Labels -> One Hot (Target)
            # Labels (B,) -> OneHot (B, 18)
            targets_np = np.zeros((batch_size, cfg.NUM_CLASSES + 1))
            targets_np[np.arange(batch_size), labels.cpu().numpy()] = 1
            
            # Background sınıfını (son sütun) mAP hesabından çıkar
            # Çünkü background'ı tahmin etmek değil, golü tahmin etmek istiyoruz.
            all_scores.append(probs[:, :-1])
            all_targets.append(targets_np[:, :-1])
        
        loop.set_postfix(loss=loss.item())
        
    avg_loss = total_loss / len(loader)
    acc = 100 * correct / (total_samples + 1e-6)
    
    # mAP Hesapla (Sadece Validasyon sonunda)
    map_score = 0.0
    if not is_train and len(all_scores) > 0:
        all_scores = np.concatenate(all_scores, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        map_score = compute_mAP(all_targets, all_scores)
    
    return {
        'loss': avg_loss,
        'cls': total_cls / len(loader),
        'reg': total_reg / len(loader),
        'acc': acc,
        'map': map_score
    }

# -----------------------------------------------------------------------------
# ANA FONKSİYON
# -----------------------------------------------------------------------------
def main():
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Çalıştırma ismi (Tarih saatli)
    run_name = f"v3_{cfg.MODEL_TYPE}_{datetime.now().strftime('%m%d_%H%M')}"
    print(f"🚀 Eğitim Başlıyor: {run_name}")
    print(f"⚙️  Model: {cfg.MODEL_TYPE} | Feat: {cfg.FEATURE_TYPE} | Cluster: {cfg.NETVLAD_CLUSTERS}")
    
    # 1. Datasetleri Yükle
    # Train: Jitter (Kaydırma) var, Masking var.
    train_ds = SoccerNetDataset(split="train", augment=True)
    # Valid: Jitter yok (Tam merkezden bakar), Masking yok. Gerçek performans ölçümü.
    valid_ds = SoccerNetDataset(split="valid", augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, 
                              num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, 
                              num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
    
    # 2. Model, Optimizer, Loss
    model = CurrentModel().to(cfg.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=cfg.LEARNING_RATE * 10,
        steps_per_epoch=len(train_loader) // cfg.ACCUMULATION_STEPS,
        epochs=cfg.EPOCHS
    )
    
    criterion = ActionSpottingLoss(gamma=2.0).to(cfg.DEVICE)
    
    best_map = 0.0
    
    print(f"{'Epoch':^5} | {'Train Loss':^10} | {'Valid Loss':^10} | {'Train Acc':^9} | {'Valid Acc':^9} | {'Valid mAP':^9} | {'Time':^8}")
    print("-" * 75)
    
    start_time = time.time()
    
    for epoch in range(cfg.EPOCHS):
        epoch_start = time.time()
        
        # --- TRAIN ---
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE, is_train=True)
        
        # --- VALID ---
        valid_metrics = run_epoch(model, valid_loader, criterion, optimizer, cfg.DEVICE, is_train=False)
        
        # Loglama
        duration = time.time() - epoch_start
        print(f"{epoch+1:02d}/{cfg.EPOCHS} | "
              f"{train_metrics['loss']:.4f}     | "
              f"{valid_metrics['loss']:.4f}     | "
              f"{train_metrics['acc']:.2f}%     | "
              f"{valid_metrics['acc']:.2f}%     | "
              f"{valid_metrics['map']:.2f}%     | " # <-- İŞTE BURASI
              f"{duration:.0f}s")
        
        # En iyi modeli kaydet (mAP'e göre)
        if valid_metrics['map'] > best_map:
            best_map = valid_metrics['map']
            torch.save(model.state_dict(), cfg.CHECKPOINT_DIR / f"{run_name}_best_map.pth")
            print(f"   🏆 New Best mAP: {best_map:.2f}% Saved!")
            
        # Periyodik kaydet
        save_path = cfg.CHECKPOINT_DIR / f"{run_name}_last.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': valid_metrics['loss'],
            'map': valid_metrics['map']
        }, save_path)
        
    print("-" * 75)
    print(f"🏁 Eğitim Tamamlandı. Toplam Süre: {(time.time()-start_time)/60:.1f} dk")
    print(f"💾 En iyi model: checkpoints/{run_name}_best_map.pth")

if __name__ == "__main__":
    main()
