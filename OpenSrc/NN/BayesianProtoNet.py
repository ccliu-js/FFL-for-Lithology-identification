import torch
import torch.nn as nn
from torch.nn import functional as F
from .backbone import BayesianCNNEncoder



class BayesianProtoNet(nn.Module):
    def __init__(self, scale=15.0):
        super().__init__()
        self.backbone = BayesianCNNEncoder()
        self.scale = scale  # 不可学习

    def forward(self, support_x, support_y, query_x, n_way):
        # ===== 1. 提取特征（包含不确定性）=====
        z_support, mu_s, logvar_s = self.backbone(support_x)
        z_query, mu_q, logvar_q = self.backbone(query_x)

        



        # 👉 用 μ 作为 embedding（更稳定）
        z_support = mu_s
        z_query = mu_q

        # ===== 2. L2 normalize =====
        z_support = F.normalize(z_support, p=2, dim=1)
        z_query = F.normalize(z_query, p=2, dim=1)

        # ===== 3. 计算 support 不确定性 =====
        std_s = torch.exp(0.5 * logvar_s)  # [Ns, D]

        # ===== 4. 构建 prototype（带权重🔥）=====
        prototypes = []
        classes = torch.unique(support_y)

        for c in classes:
            class_feat = z_support[support_y == c]      # [K, D]
            class_std = std_s[support_y == c]           # [K, D]

            # 👉 核心：不确定性越小 → 权重越大
            weight = 1.0 / (class_std.mean(dim=1) + 1e-6)  # [K]
            weight = weight / weight.sum()

            proto = (class_feat * weight.unsqueeze(1)).sum(dim=0)
            prototypes.append(proto)

        prototypes = torch.stack(prototypes)  # [n_way, D]

        # 再 normalize
        prototypes = F.normalize(prototypes, p=2, dim=1)

        # ===== 5. cosine similarity =====
        cosine_sim = torch.matmul(z_query, prototypes.t())  # [Nq, n_way]

        # ===== 6. query 不确定性抑制🔥 =====
        std_q = torch.exp(0.5 * logvar_q)
        uncertainty_q = std_q.mean(dim=1, keepdim=True)  # [Nq, 1]

        logits = cosine_sim * self.scale / (1.0 + uncertainty_q)

        # ===== 7. 返回（用于 loss）=====
        return logits, mu_s, logvar_s, mu_q, logvar_q






