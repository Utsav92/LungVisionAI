"""Custom 3D ResNet-style classifier/regressor for nodule crops (Model C). Random init."""
from __future__ import annotations
import torch, torch.nn as nn
from .blocks3d import ConvNormAct, ResBlock3d


class Nodule3DResNet(nn.Module):
    def __init__(self, in_channels=1, base=16, num_outputs=1, norm="group", act="silu", dropout=0.2):
        super().__init__()
        self.stem = ConvNormAct(in_channels, base, 3, 1, norm, act)
        self.r1 = ResBlock3d(base, base, 1, norm, act)
        self.r2 = ResBlock3d(base, 2 * base, 2, norm, act)
        self.r3 = ResBlock3d(2 * base, 4 * base, 2, norm, act)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * base, num_outputs)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.r3(self.r2(self.r1(self.stem(x))))
        return self.fc(self.drop(self.pool(x).flatten(1)))
