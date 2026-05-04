import torch
import torch.nn as nn
from blitz.modules import BayesianLinear


class BayesianEmbedding(nn.Module):
    def __init__(self, in_dim=288, latent_dim=128, embed_dim=128):
        super().__init__()

        self.bayes_fc1 = BayesianLinear(in_dim, latent_dim)
        self.bayes_mu = BayesianLinear(latent_dim, embed_dim)
        self.bayes_logvar = BayesianLinear(latent_dim, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bayes_fc1(x))

        mu = self.bayes_mu(x)
        logvar = self.bayes_logvar(x)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar
