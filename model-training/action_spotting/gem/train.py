# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import config as cfg
from dataset import SoccerNetDataset
from model import RMSNet
from loss import SpottingLoss
# Eksik olan import eklendi:
from utils import calculate_weights_sqrt 

def train():
    print(f"Arra7 SOTA Eğitim (Weighted Gaussian) başlatılıyor... GPU: {cfg.DEVICE}")

    dataset = SoccerNetDataset(split="train")
    if len(dataset) == 0:
        print("HATA: Veri seti boş!")
        return

    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, pin_memory=True)
    
    model = RMSNet().to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    # --- KRİTİK DÜZELTME BURADA ---
    # 1. Veri setini tarayıp ağırlıkları hesapla (Gol'e yüksek, Taç'a düşük puan)
    pos_weight = calculate_weights_sqrt(dataset)
    pos_weight = pos_weight.view(-1, 1)
    
    # 2. Bu ağırlıkları Loss fonksiyonuna ver
    criterion = SpottingLoss(pos_weight=pos_weight)
    
    scaler = GradScaler()

    print("Eğitim Başlıyor...")

    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        
        for features, target_gaussian in pbar:
            features = features.to(cfg.DEVICE)
            target_gaussian = target_gaussian.to(cfg.DEVICE)

            with autocast():
                logits = model(features)
                loss = criterion(logits, target_gaussian)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{total_loss / (pbar.n + 1):.6f}"})

        # Modeli kaydet
        torch.save(model.state_dict(), f"arra7_gaussian_ep{epoch+1}.pth")
        print(f"✅ Epoch {epoch+1} Bitti. Avg Loss: {total_loss/len(loader):.6f}")

if __name__ == "__main__":
    train()