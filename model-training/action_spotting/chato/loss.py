import torch.nn as nn

class ActionSpottingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, cls_pred, cls_gt):
        """
        cls_pred: (B, num_classes)
        cls_gt:   (B,)
        """
        return self.ce(cls_pred, cls_gt)
