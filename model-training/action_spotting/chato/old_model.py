import torch
import torch.nn as nn
import torch.nn.functional as F

class NetVLAD(nn.Module):
    def __init__(self, cluster_size, feature_size):
        super().__init__()
        self.cluster_size = cluster_size
        self.feature_size = feature_size

        self.centroids = nn.Parameter(torch.rand(cluster_size, feature_size))
        self.assignment = nn.Linear(feature_size, cluster_size, bias=True)

    def forward(self, x):
        # x: (B, T, D)
        soft_assign = F.softmax(self.assignment(x), dim=-1)
        x_exp = x.unsqueeze(2)
        c_exp = self.centroids.unsqueeze(0).unsqueeze(0)
        residual = x_exp - c_exp
        vlad = (soft_assign.unsqueeze(-1) * residual).sum(dim=1)
        vlad = F.normalize(vlad, p=2, dim=-1)
        return vlad.view(x.size(0), -1)

class ActionSpottingNet(nn.Module):
    def __init__(self, input_dim=512, num_classes=17):
        super().__init__()

        self.reducer = nn.Linear(input_dim, 512)

        self.vlad_past = NetVLAD(32, 512)
        self.vlad_future = NetVLAD(32, 512)

        vlad_dim = 32 * 512 * 2

        self.cls_head = nn.Linear(vlad_dim, num_classes + 1)
        self.reg_head = nn.Sequential(
            nn.Linear(vlad_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, D)
        x = self.reducer(x)
        mid = x.shape[1] // 2

        v_p = self.vlad_past(x[:, :mid])
        v_f = self.vlad_future(x[:, mid:])

        feat = torch.cat([v_p, v_f], dim=-1)
        return self.cls_head(feat), self.reg_head(feat).squeeze(-1)
