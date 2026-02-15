import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg

class NetVLAD(nn.Module):
    """
    NetVLAD Katmanı: Zamansal özellikleri (Time Dimension) tek bir vektörde özetler.
    Video (T, D) -> Vektör (K*D)
    """
    def __init__(self, feature_size, num_clusters=64, add_batch_norm=True):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size
        self.num_clusters = num_clusters
        
        # Soft Assignment için ağırlıklar
        self.softmax = nn.Softmax(dim=1)
        
        # Trainable Cluster Centers ve Weights
        self.cluster_weights = nn.Parameter(torch.randn(num_clusters, feature_size) * 1 / feature_size**.5)
        self.cluster_weights2 = nn.Parameter(torch.randn(num_clusters, feature_size) * 1 / feature_size**.5)
        
        self.bn = nn.BatchNorm1d(num_clusters * feature_size) if add_batch_norm else None

    def forward(self, x):
        # x: (Batch, Time, Dim) -> (Batch, Dim, Time) işlem kolaylığı için
        x = x.transpose(1, 2)
        B, D, T = x.shape
        
        # Soft Assignment (karelerin hangi kümeye ait olduğu)
        # (Clusters, Dim) x (Batch, Dim, Time) -> (Batch, Clusters, Time)
        activation = torch.matmul(self.cluster_weights, x)
        
        # Bias ekleme: (1, K) -> (1, K, 1) broadcast için
        bias = self.cluster_weights2.sum(1, keepdim=True).transpose(0, 1)
        activation += bias.unsqueeze(2)
        
        activation = self.softmax(activation) # (B, Clusters, T)

        # Core VLAD calculation
        # Kaç kare bu kümeye ait?
        a_sum = activation.sum(-1, keepdim=True) # (B, K, 1)
        
        # Ağırlıklı küme merkezi (Weighted Cluster Center)
        a = a_sum * self.cluster_weights2 # (B, K, D)
        
        # Vlad Residuals: (x - c_k)
        # Önce permütasyonlar:
        activation = torch.transpose(activation, 2, 1) # (B, T, K)
        x = x.transpose(1, 2) # (B, T, D)
        
        # (B, K, D)
        vlad = torch.matmul(activation.transpose(1, 2), x) 
        vlad = vlad.transpose(2, 1) - a.transpose(1, 2)
        vlad = F.normalize(vlad, p=2, dim=1) # Intra-normalization
        
        # Flatten: (B, K*D)
        vlad = vlad.reshape(B, -1) 
        
        # L2-Normalize
        vlad = F.normalize(vlad, p=2, dim=1)
        
        if self.bn:
            vlad = self.bn(vlad)
            
        return vlad

class CNNActionSpotter(nn.Module):
    def __init__(self):
        super(CNNActionSpotter, self).__init__()
        
        # 1. Giriş Projeksiyonu
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.FEATURE_DIM, cfg.PROJECTION_DIM),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
        # 2. Backbone: 1D CNN (Zaman ekseninde yerel desenler)
        self.conv_1 = nn.Conv1d(cfg.PROJECTION_DIM, cfg.PROJECTION_DIM, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # BatchNorm eklemek eğitimi hızlandırır
        self.bn_1 = nn.BatchNorm1d(cfg.PROJECTION_DIM) 

        # 3. NetVLAD++ (Geçmiş ve Gelecek için Ayrı Havuzlama)
        # Cluster sayısı Config'den alınmalı (Consistency)
        self.k_clusters = cfg.NETVLAD_CLUSTERS 
        self.netvlad_past = NetVLAD(feature_size=cfg.PROJECTION_DIM, num_clusters=self.k_clusters)
        self.netvlad_future = NetVLAD(feature_size=cfg.PROJECTION_DIM, num_clusters=self.k_clusters)
        
        # NetVLAD Çıktı Boyutu: 2 (Past+Future) * K * Dim
        vlad_dim = 2 * self.k_clusters * cfg.PROJECTION_DIM
        
        # 4. Heads (Classification & Regression)
        self.cls_head = nn.Sequential(
            nn.Linear(vlad_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, cfg.NUM_CLASSES + 1)
        )
        
        self.reg_head = nn.Sequential(
            nn.Linear(vlad_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # x: (Batch, Window, Feat) -> (B, T, D)
        
        # 1. Projection
        x = self.input_proj(x)
        
        # 2. CNN Layers
        x = x.permute(0, 2, 1) # (B, D, T)
        x = self.relu(self.bn_1(self.conv_1(x)))
        x = x.permute(0, 2, 1) # (B, T, D) tekrar
        
        # 3. NetVLAD++ Split (Geçmiş ve Geleceği Ayır)
        T = x.size(1)
        middle = T // 2
        
        # Geçmiş: [0, middle)
        x_past = x[:, :middle, :]
        # Gelecek: [middle, T)
        x_future = x[:, middle:, :]
        
        # VLAD Pooling
        vlad_past = self.netvlad_past(x_past)     # (B, K*D)
        vlad_future = self.netvlad_future(x_future) # (B, K*D)
        
        # Birleştir (Concatenate)
        combined = torch.cat([vlad_past, vlad_future], dim=1)
        
        # 4. Heads
        cls_logits = self.cls_head(combined)
        reg_offset = self.reg_head(combined)
        
        return cls_logits, reg_offset

if __name__ == "__main__":
    # Test
    model = CNNActionSpotter()
    dummy_input = torch.randn(4, 20, 8576)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    c, r = model(dummy_input)
    print(f"Logits: {c.shape}, Offset: {r.shape}")
