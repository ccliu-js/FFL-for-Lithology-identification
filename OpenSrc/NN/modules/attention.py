import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, L]
        B, C, L = x.shape

        avg = self.avg_pool(x).view(B, C)
        max_ = self.max_pool(x).view(B, C)

        avg_out = self.mlp(avg)
        max_out = self.mlp(max_)

        attn = avg_out + max_out
        attn = self.sigmoid(attn).view(B, C, 1)

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
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, L]
        avg = torch.mean(x, dim=1, keepdim=True)        # [B, 1, L]
        max_, _ = torch.max(x, dim=1, keepdim=True)     # [B, 1, L]

        x_cat = torch.cat([avg, max_], dim=1)           # [B, 2, L]
        attn = self.conv(x_cat)                         # [B, 1, L]
        attn = self.sigmoid(attn)

        return x * attn


# class LiteSEFusion(nn.Module):
#     def __init__(self, time_dim=128, freq_dim=64, hidden=8):
#         super().__init__()
#         self.scale = nn.Sequential(
#             nn.Linear(freq_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, time_dim),
#             nn.Sigmoid()
#         )
#         self.beta = nn.Parameter(torch.tensor(0.1))

#     def forward(self, x_time, x_freq):
#         # x_time: [B, 128, 128]
#         # x_freq: [B, 1, 64]

#         s = self.scale(x_freq.squeeze(1)).unsqueeze(1)   # [B, 1, 128]

#         # 一个很轻量的频域残差注入方式：
#         # 先把频域压成一个标量，再广播，不增加额外大参数
#         f = x_freq.mean(dim=-1, keepdim=True)            # [B, 1, 1]
#         f = f.expand(-1, x_time.size(1), x_time.size(2)) # [B, 128, 128]

#         out = x_time * (1.0 + s) + self.beta * f
#         return out

import torch
import torch.nn as nn

class LiteSEFusion(nn.Module):
    def __init__(self, time_dim=128, freq_dim=64, hidden=8):
        super().__init__()
        self.scale = nn.Sequential(
            nn.Linear(freq_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, time_dim),
            nn.Sigmoid()
        )
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_time, x_freq):
        # x_time: [B, 128, 512] (根据你前文的日志，长度是512)
        # x_freq: [B, 1, 64]

        # ✅ 修复点 1：将 unsqueeze(1) 改为 unsqueeze(-1)
        # 将 [B, 128] 扩展为 [B, 128, 1]，让 128 对齐到通道维度
        s = self.scale(x_freq.squeeze(1)).unsqueeze(-1)   

        # 一个很轻量的频域残差注入方式：
        # 先把频域压成一个标量，再广播，不增加额外大参数
        f = x_freq.mean(dim=-1, keepdim=True)            # [B, 1, 1]
        
        # ✅ 优化点 2：可以直接利用 PyTorch 的广播机制，不需要显式 expand
        # f = f.expand(-1, x_time.size(1), x_time.size(2)) 这一行可以安全删掉
        # [B, 128, 512] * [B, 128, 1] + [B, 1, 1] 可以自动广播对齐

        out = x_time * (1.0 + s) + self.beta * f
        return out


class FusionModule(nn.Module):
    def __init__(self, channel_attn=128, fusion_dim=128):
        super().__init__()
        self.time_ChannelAttention1D = ChannelAttention1D(channel_attn)
        self.time_TemporalAttention = TemporalAttention()
        self.fusion = LiteSEFusion(time_dim=128, freq_dim=64, hidden=8)


    def forward(self, x_time, x_freq):
        """
        x_time: [B, 128, 512]
        x_freq: [B, 1, 128]
        return: [B, fusion_dim]
        """

        # 1. 时域注意力增强
        x_time = self.time_ChannelAttention1D(x_time)   # [B, 128, 512]
        x_time = self.time_TemporalAttention(x_time)    # [B, 128, 512]

        # print("After attention, time features shape:", x_time.shape)
        # print("Freq features shape (before fusion):", x_freq.shape)

        # 2. 融合
        x = self.fusion(x_time, x_freq)           # [B, 128, 512]


        #查看参数量
        total_params = sum(p.numel() for p in self.parameters())
        # print(f"Total parameters in FusionModule: {total_params}")



        return x
    
if __name__ == "__main__":
    fusion_module = FusionModule(channel_attn=128, fusion_dim=128)
    x_time = torch.randn(8, 128, 512)  # Example time features
    x_freq = torch.randn(8, 1, 64)     # Example freq features
    fused = fusion_module(x_time, x_freq)
    print("Fused features shape:", fused.shape)