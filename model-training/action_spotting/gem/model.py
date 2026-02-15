# src/model.py
import torch
import torch.nn as nn
import config as cfg

class RMSNet(nn.Module):
    def __init__(self):
        super(RMSNet, self).__init__()
        
        hidden_dim = 512
        
        self.input_proj = nn.Sequential(
            nn.Conv1d(cfg.FEATURE_DIM, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim), # Düşük Batch için GroupNorm şart
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.backbone = nn.Sequential(
            self._res_block(hidden_dim),
            self._res_block(hidden_dim),
            self._res_block(hidden_dim)
        )

        # ÇIKTI KATMANI DEĞİŞTİ:
        # Artık "Pooling" yapıp zamanı yok etmiyoruz.
        # Her zaman adımı (frame) için bir tahmin üretiyoruz.
        # Çıktı: (Batch, Num_Classes, Window_Frame)
        self.head = nn.Conv1d(hidden_dim, cfg.NUM_CLASSES, kernel_size=1)

    def _res_block(self, dim):
        return nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.backbone(x)
        logits = self.head(x) # (Batch, 17, 40)
        return logits