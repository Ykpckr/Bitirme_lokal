import torch
from torch.utils.data import DataLoader

from dataset import SoccerNetDataset
from model import ActionSpottingTransformerNet
from loss import ActionSpottingLoss

# =====================
# AYARLAR
# =====================
DATASET_PATH = "C:/FoMAC_Dataset/action_spotting"
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# =====================
# DATASET
# =====================
dataset = SoccerNetDataset(DATASET_PATH)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

# =====================
# MODEL
# =====================
model = ActionSpottingTransformerNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

criterion = ActionSpottingLoss()  # SADECE CLASSIFICATION

# =====================
# TRAIN LOOP
# =====================
model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for x, cls_gt in loader:
        x = x.to(DEVICE)           # (B, T, D)
        cls_gt = cls_gt.to(DEVICE) # (B, T)

        optimizer.zero_grad()

        cls_pred = model(x)        # (B, num_classes)

        # sadece orta frame supervise edilir (SoccerNet standardı)
        center = cls_gt.shape[1] // 2
        loss = criterion(cls_pred, cls_gt[:, center])

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {epoch_loss:.4f}")

# =====================
# MODEL KAYDET
# =====================
torch.save(model.state_dict(), "action_spotting_cls_only.pth")
print("✅ Eğitim tamamlandı (classification only)")
