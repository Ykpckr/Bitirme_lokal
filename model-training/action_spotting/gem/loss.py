# src/loss.py
import torch
import torch.nn as nn
import config as cfg

class SpottingLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(SpottingLoss, self).__init__()
        # Ağırlıkları BCE Loss'a veriyoruz.
        # Bu sayede Gol kaçırmak, Taç kaçırmaktan 10 kat daha fazla ceza puanı yazar.
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, pred_logits, target_gaussian):
        loss = self.bce(pred_logits, target_gaussian)
        return loss