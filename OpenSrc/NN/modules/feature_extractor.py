import torch.nn as nn


class Frequency_CNN(nn.Module):
    def __init__(
        self,
        input_channels=1,
        channels=(16, 32, 64),
        norm_groups=(4, 8, 8),
    ):
        super().__init__()
        c1, c2, c3 = channels
        g1, g2, g3 = norm_groups

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, c1, 3, padding=1),
            nn.GroupNorm(g1, c1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.GroupNorm(g2, c2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.GroupNorm(g3, c3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_channels = c3

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)


class TimeCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=2),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f1, f2, f3]
