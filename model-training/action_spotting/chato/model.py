import torch
import torch.nn as nn

class ActionSpottingTransformerNet(nn.Module):
    def __init__(self, input_dim=8576, num_classes=18):
        """
        num_classes = 17 action + 1 background
        """
        super().__init__()

        self.proj = nn.Linear(input_dim, 512)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.cls_head = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        x: (B, T, D)
        return: (B, num_classes)
        """
        x = self.proj(x)            # (B, T, 512)
        x = self.encoder(x)         # (B, T, 512)

        center = x.shape[1] // 2
        x_center = x[:, center]     # (B, 512)

        cls_logits = self.cls_head(x_center)
        return cls_logits
