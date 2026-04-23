import torch
import torch.nn as nn
from torch.nn import Module
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from torch.nn import functional as F




import os
import pickle
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


"""TimeDomain: 用于提取时域特征的卷积神经网络模块。
输入: [B, 1, 128] (单通道的长度为128的时域信号)
如果 is_gap=True，则输出: [B, 1, 128] (提取的128维时域特征向量，保持单通道以便后续融合)
如果 is_gap=False，则输出: [B, 128*nf*4] (提取的时域特征向量，未进行全局池化，保持原始维度以便后续融合)
融合) 
"""


class InceptionModule(Module):
    def __init__(self, ni, nf, ks=15, bottleneck=True):
        super().__init__()
        ks = [ks // (2**i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        bottleneck = bottleneck if ni > 1 else False
        self.bottleneck = nn.Conv1d(ni, nf, kernel_size=1, bias=False) if bottleneck else nn.Identity()
        self.convs = nn.ModuleList([
            nn.Conv1d(nf if bottleneck else ni, nf, kernel_size=k, padding=k//2, bias=False)
            for k in ks
        ])
        self.maxconvpool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(ni, nf, kernel_size=1, bias=False)
        )
        self.bn = nn.BatchNorm1d(nf * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = torch.cat([l(x) for l in self.convs] + [self.maxconvpool(input_tensor)], dim=1)
        return self.act(self.bn(x))


class InceptionBlock(Module):
    def __init__(self, ni, nf=32, residual=True, depth=3):
        super().__init__()
        self.residual, self.depth = residual, depth
        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        for d in range(depth):
            self.inception.append(InceptionModule(ni if d == 0 else nf * 4, nf))
            if self.residual and d % 3 == 2: 
                n_in, n_out = ni if d == 2 else nf * 4, nf * 4
                self.shortcut.append(
                    nn.BatchNorm1d(n_in) if n_in == n_out else
                    nn.Sequential(
                        nn.Conv1d(n_in, n_out, kernel_size=1, bias=False),
                        nn.BatchNorm1d(n_out)
                    )
                )
        self.act = nn.ReLU()
        
    def forward(self, x):
        res = x
        for d in range(self.depth):
            x = self.inception[d](x)
            if self.residual and d % 3 == 2:
                res = x = self.act(x + self.shortcut[d // 3](res))
        return x




class TimeDomain(Module):
    def __init__(self, c_in, nf=32,is_gap=False):
        super().__init__()
        self.backbone = InceptionBlock(c_in, nf,depth=2)
        self.is_gap = is_gap


        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )


    def forward(self, x):
        if x.dim() == 2:  # 如果输入是 [B, L]，则添加通道维度
            x = x.unsqueeze(1)  # 变为 [B, 1, L]
        x = self.backbone(x)
        if self.is_gap:
            x = self.gap(x)
            x=x.unsqueeze(1)  # Add a dimension to make it (batch_size, 1, features)
        
        #查看模块的参数量
        # total_params = sum(p.numel() for p in self.parameters())
        # print(f"Total parameters in TimeDomain: {total_params}")

        return x
    

if __name__ == "__main__":
    model = TimeDomain(c_in=1, nf=32)
    print(model)
    x = torch.randn(8, 1, 128)  # Example input: batch of 8, 1 channel, length 128
    out = model(x)
    print(out.shape)  # Should be [8, 1, 128]