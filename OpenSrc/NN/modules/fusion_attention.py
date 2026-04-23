import torch
import torch.nn as nn
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from torch.nn import functional as F

class LightAttention1D(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.channel_att = ChannelAttention1D(channels)
        self.dwconv = MultiScaleDWConv1D(channels)
        self.spatial_att = SpatialAttention1D()

    def forward(self, x):
        # x: [B, C, L]

        x = self.channel_att(x)   # 通道注意力
        x = self.dwconv(x)        # 多尺度建模
        x = self.spatial_att(x)   # 时间注意力

        return x
    


class ChannelAttention1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = channels // reduction

        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(hidden, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, L]

        avg = torch.mean(x, dim=2, keepdim=True)
        max_ = torch.max(x, dim=2, keepdim=True)[0]

        attn = self.mlp(avg) + self.mlp(max_)
        attn = self.sigmoid(attn)

        return x * attn
    


class MultiScaleDWConv1D(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.branch1 = nn.Conv1d(channels, channels, 5, padding=2, groups=channels)
        self.branch2 = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.branch3 = nn.Conv1d(channels, channels, 11, padding=5, groups=channels)
        self.branch4 = nn.Conv1d(channels, channels, 21, padding=10, groups=channels)

        self.pointwise = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        out = b1 + b2 + b3 + b4
        out = self.pointwise(out)

        return out
    


class SpatialAttention1D(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv1d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, L]

        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True)[0]

        attn = torch.cat([avg, max_], dim=1)
        attn = self.sigmoid(self.conv(attn))

        return x * attn