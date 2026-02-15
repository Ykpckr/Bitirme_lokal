import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg

class ActionSpottingLoss(nn.Module):
    """
    V2 Kapsamlı Loss Fonksiyonu (CALF Inspired)
    
    Özellikler:
    1. Focal Loss: Sınıf dengesizliği için (CrossEntropy yerine).
       "Hep aynı şeyi tahmin etme" sorununu çözer.
    2. Ignore Region: Olayın çok yakınındaki (ama tam üstü olmayan) kareleri yok sayar.
    3. Regression Loss: Zamanlamayı keskinleştirir.
    """
    def __init__(self, weight=None, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
        # Regresyon kaybı (MSE)
        self.reg_criterion = nn.MSELoss(reduction='none')

    def focal_loss(self, logits, targets):
        """
        Focal Loss Implementasyonu.
        Easy examples (Zaten bildiği) için kaybı düşürür, Hard examples'a odaklanır.
        """
        # Sigmoid uygula (Multi-label mantığı, her sınıf bağımsız)
        probs = torch.sigmoid(logits)
        
        # BCE Loss (Reduction yok)
        # targets one-hot formatında olmalı
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal Term: (1 - p_t)^gamma
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        # Alpha Term (Balance)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = loss * alpha_t
            
        return loss.mean()

    def forward(self, cls_logits, reg_preds, labels, offsets):
        """
        Args:
            cls_logits: (B, NumClasses+1)
            reg_preds: (B, 1)
            labels: (B,) Class Indices
            offsets: (B, 1) Target offsets
        """
        batch_size = cls_logits.size(0)
        num_classes = cls_logits.size(1) # 17 + 1 = 18
        device = cls_logits.device
        
        # 1. Target Hazırlama (One-Hot)
        # Focal Loss/BCE için one-hot vektörlere ihtiyacımız var.
        targets = torch.zeros(batch_size, num_classes, device=device)
        targets.scatter_(1, labels.view(-1, 1), 1.0)
        
        # 2. CALF Stratejisi: Ignore Region
        # Eğer offset değeri çok büyükse (örneğin olaya 1 saniye uzaklıktaysak)
        # Bu sample'ı "Background" olarak eğitmek kafa karıştırabilir.
        # Ama bizim Dataset yapımızda zaten "Center Frame" alıyoruz.
        # Yani örneklerimiz ya tam olay anı ya da rastgele background.
        # O yüzden "Ignore" maskesine şimdilik gerek yok (Dataset temiz).
        # Ancak yine de Focal Loss ile "Background" sınıfının baskınlığını kıracağız.
        
        cls_loss = self.focal_loss(cls_logits, targets)
        
        # 3. Regression Loss
        # Sadece gerçek olaylarda (Background olmayan) hesaplanır.
        # Background ID = 17
        bg_mask = (labels != cfg.BACKGROUND_CLASS)
        
        reg_loss = torch.tensor(0.0, device=device)
        if bg_mask.sum() > 0:
            # Sadece olay olanların regresyon hatasını al
            pred_offsets = reg_preds[bg_mask]       # (N_events, 1)
            target_offsets = offsets[bg_mask].view(-1, 1) # (N_events, 1) - HATA DÜZELTİLDİ
            
            reg_l = self.reg_criterion(pred_offsets, target_offsets)
            reg_loss = reg_l.mean()
            
        # 4. Total Loss
        total_loss = cls_loss + cfg.LAMBDA_REG * reg_loss
        
        return {
            'total': total_loss,
            'cls': cls_loss,
            'reg': reg_loss
        }
