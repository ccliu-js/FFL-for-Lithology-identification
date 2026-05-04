import torch
import torch.nn as nn
from torch.nn import Module


class InceptionModule(Module):
    def __init__(self, ni, nf, ks=15, bottleneck=True):
        super().__init__()
        ks = [ks // (2**i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]
        bottleneck = bottleneck if ni > 1 else False

        self.bottleneck = (
            nn.Conv1d(ni, nf, kernel_size=1, bias=False)
            if bottleneck
            else nn.Identity()
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    nf if bottleneck else ni,
                    nf,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                )
                for k in ks
            ]
        )
        self.maxconvpool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(ni, nf, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(nf * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = torch.cat([layer(x) for layer in self.convs] + [self.maxconvpool(input_tensor)], dim=1)
        return self.act(self.bn(x))


class InceptionBlock(Module):
    def __init__(self, ni, nf=32, residual=True, depth=3, ks=15, bottleneck=True):
        super().__init__()
        self.residual = residual
        self.depth = depth
        self.inception = nn.ModuleList()
        self.shortcut = nn.ModuleList()

        for d in range(depth):
            self.inception.append(
                InceptionModule(
                    ni if d == 0 else nf * 4,
                    nf,
                    ks=ks,
                    bottleneck=bottleneck,
                )
            )
            if self.residual and d % 3 == 2:
                n_in, n_out = ni if d == 2 else nf * 4, nf * 4
                self.shortcut.append(
                    nn.BatchNorm1d(n_in)
                    if n_in == n_out
                    else nn.Sequential(
                        nn.Conv1d(n_in, n_out, kernel_size=1, bias=False),
                        nn.BatchNorm1d(n_out),
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
    def __init__(
        self,
        c_in,
        nf=32,
        is_gap=False,
        depth=3,
        kernel_size=15,
        residual=True,
        bottleneck=True,
    ):
        super().__init__()
        self.backbone = InceptionBlock(
            c_in,
            nf,
            depth=depth,
            residual=residual,
            ks=kernel_size,
            bottleneck=bottleneck,
        )
        self.is_gap = is_gap
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.backbone(x)
        if self.is_gap:
            x = self.gap(x)
            x = x.unsqueeze(1)
        return x
