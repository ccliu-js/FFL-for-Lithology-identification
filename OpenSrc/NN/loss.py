import torch.nn.functional as F
import torch.nn as nn
import torch

import torch
import torch.nn.functional as F


class BNNLoss():
    def __init__(self, kl_weight=1e-6, latent_kl_weight=1e-5):
        self.kl_weight = kl_weight                # 权重KL（blitz）
        self.latent_kl_weight = latent_kl_weight  # 特征KL（你新增的）

    def kl_loss_latent(self, mu, logvar):
        """
        latent KL: 控制 embedding 分布
        """
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def KL_and_CE_loss(
        self,
        model,
        query_x,
        query_y,
        support_x,
        support_y,
        n_way,
        num_samples=5
    ):
        all_logits = []

        mu_s_list, logvar_s_list = [], []
        mu_q_list, logvar_q_list = [], []

        # ===== 1. Monte Carlo Sampling =====
        for _ in range(num_samples):
            logits, mu_s, logvar_s, mu_q, logvar_q = model(
                support_x, support_y, query_x, n_way
            )

            all_logits.append(logits)

            mu_s_list.append(mu_s)
            logvar_s_list.append(logvar_s)
            mu_q_list.append(mu_q)
            logvar_q_list.append(logvar_q)

        # ===== 2. 平均 logits（分类用）=====
        all_logits = torch.stack(all_logits)   # [S, Nq, n_way]
        avg_logits = all_logits.mean(dim=0)

        loss_ce = F.cross_entropy(avg_logits, query_y)

        # ===== 3. 权重 KL（blitz）=====
        loss_kl_weight = model.backbone.nn_kl_divergence()

        # ===== 4. latent KL（你新增的🔥）=====
        mu_s = torch.stack(mu_s_list).mean(0)
        logvar_s = torch.stack(logvar_s_list).mean(0)

        mu_q = torch.stack(mu_q_list).mean(0)
        logvar_q = torch.stack(logvar_q_list).mean(0)

        loss_kl_latent = (
            self.kl_loss_latent(mu_s, logvar_s) +
            self.kl_loss_latent(mu_q, logvar_q)
        )

        # ===== 5. 总 loss =====
        total_loss = (
            loss_ce
            + self.kl_weight * loss_kl_weight
            + self.latent_kl_weight * loss_kl_latent
        )

        return total_loss, loss_ce, loss_kl_weight, loss_kl_latent