import torch
import torch.nn as nn


class ChannelAttention1D(nn.Module):
    def __init__(self, channels, reduction=8, min_hidden=4):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        hidden = max(channels // reduction, min_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _ = x.shape

        avg = self.avg_pool(x).view(batch_size, channels)
        max_ = self.max_pool(x).view(batch_size, channels)

        attn = self.mlp(avg) + self.mlp(max_)
        attn = self.sigmoid(attn).view(batch_size, channels, 1)

        return x * attn


class TemporalAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg, max_], dim=1)
        attn = self.sigmoid(self.conv(attn))

        return x * attn


class LiteSEFusion(nn.Module):
    def __init__(self, time_dim=128, freq_dim=64, hidden=8, beta_init=0.1):
        super().__init__()
        self.scale = nn.Sequential(
            nn.Linear(freq_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, time_dim),
            nn.Sigmoid(),
        )
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x_time, x_freq):
        scale = self.scale(x_freq.squeeze(1)).unsqueeze(-1)
        freq_bias = x_freq.mean(dim=-1, keepdim=True)
        return x_time * (1.0 + scale) + self.beta * freq_bias


class FusionModule(nn.Module):
    def __init__(
        self,
        channel_attn=128,
        fusion_dim=128,
        freq_dim=64,
        channel_reduction=8,
        min_channel_hidden=4,
        temporal_kernel_size=7,
        hidden_dim=8,
        beta_init=0.1,
    ):
        super().__init__()
        self.time_ChannelAttention1D = ChannelAttention1D(
            channel_attn,
            reduction=channel_reduction,
            min_hidden=min_channel_hidden,
        )
        self.time_TemporalAttention = TemporalAttention(kernel_size=temporal_kernel_size)
        self.fusion = LiteSEFusion(
            time_dim=fusion_dim,
            freq_dim=freq_dim,
            hidden=hidden_dim,
            beta_init=beta_init,
        )

    def forward(self, x_time, x_freq):
        x_time = self.time_ChannelAttention1D(x_time)
        x_time = self.time_TemporalAttention(x_time)
        return self.fusion(x_time, x_freq)
