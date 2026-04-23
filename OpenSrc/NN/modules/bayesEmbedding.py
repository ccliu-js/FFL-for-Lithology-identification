import torch
import torch.nn as nn
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from torch.nn import functional as F


class BayesianEmbedding(nn.Module):
    def __init__(self, in_dim=288, embed_dim=128):
        super().__init__()

        self.bayes_fc1 = BayesianLinear(in_dim, 128)
     

        # 输出 μ 和 logσ
        self.bayes_mu = BayesianLinear(128, embed_dim)
        self.bayes_logvar = BayesianLinear(128, embed_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bayes_fc1(x))


        mu = self.bayes_mu(x)
        logvar = self.bayes_logvar(x)


        # reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std


        return z, mu, logvar