import torch
import torch.nn as nn
from torch.nn import functional as F

from .backbone import BayesianCNNEncoder


class BayesianProtoNet(nn.Module):
    def __init__(self, scale=15.0, encoder_config=None):
        super().__init__()
        self.backbone = BayesianCNNEncoder(config=encoder_config)
        self.scale = scale

    def forward(self, support_x, support_y, query_x, n_way):
        _, mu_s, logvar_s = self.backbone(support_x)
        _, mu_q, logvar_q = self.backbone(query_x)

        z_support = F.normalize(mu_s, p=2, dim=1)
        z_query = F.normalize(mu_q, p=2, dim=1)

        std_s = torch.exp(0.5 * logvar_s)
        prototypes = []

        for c in torch.unique(support_y):
            class_feat = z_support[support_y == c]
            class_std = std_s[support_y == c]

            weight = 1.0 / (class_std.mean(dim=1) + 1e-6)
            weight = weight / weight.sum()

            proto = (class_feat * weight.unsqueeze(1)).sum(dim=0)
            prototypes.append(proto)

        prototypes = F.normalize(torch.stack(prototypes), p=2, dim=1)

        cosine_sim = torch.matmul(z_query, prototypes.t())
        std_q = torch.exp(0.5 * logvar_q)
        uncertainty_q = std_q.mean(dim=1, keepdim=True)
        logits = cosine_sim * self.scale / (1.0 + uncertainty_q)

        return logits, mu_s, logvar_s, mu_q, logvar_q
