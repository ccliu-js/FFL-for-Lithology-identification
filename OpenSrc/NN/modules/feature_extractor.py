import torch
import torch.nn as nn
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from torch.nn import functional as F




class Frequency_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(16, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )



    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return x






class TimeCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # ===== Level 1（高分辨率）=====
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.GroupNorm(8, 32),
            nn.ReLU()
        )

        # ===== Level 2 =====
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # ===== Level 3 =====
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=2),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        # x: [B, 1, 4000]
        x=x.unsqueeze(1)  # [B, 1, 4000]

        f1 = self.layer1(x)        # [B, 32, 4000]
        f2 = self.layer2(f1)       # [B, 64, 2000]
        f3 = self.layer3(f2)       # [B, 128, 1000]

        time=[f1,f2,f3]

        return time
    



    

