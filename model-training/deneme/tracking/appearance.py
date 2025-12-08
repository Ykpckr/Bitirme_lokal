# tracking/appearance.py
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2

class ReIDModel(nn.Module):
    def __init__(self, output_dim=256, pretrained=True, device='cpu'):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # remove final fc
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.fc = nn.Linear(in_features, output_dim)
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.fc(self.backbone(x))

def preprocess_crop(img_crop):
    # img_crop: BGR numpy
    img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128,256))
    img = img.astype('float32') / 255.0
    img = (img - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
    img = np.transpose(img, (2,0,1))
    return img

class AppearanceEncoder:
    def __init__(self, device='cpu', output_dim=256):
        self.device = torch.device(device if device else 'cpu')
        self.model = ReIDModel(output_dim=output_dim, pretrained=True, device=self.device)
        self.model.eval()

    def encode(self, crops):
        """
        crops: list of BGR np arrays
        returns: numpy array of embeddings (N, D)
        """
        import torch
        if not crops:
            return np.zeros((0, self.model.fc.out_features))
        tensors = [preprocess_crop(c) for c in crops]
        batch = torch.tensor(tensors, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            feats = self.model(batch)
            feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy()
