"""Building blocks for the from-scratch 3D networks (no pretrained weights anywhere)."""
from __future__ import annotations
import torch
import torch.nn as nn


def make_norm(kind: str, channels: int) -> nn.Module:
    if kind == "group":
        g = 8
        while channels % g:
            g //= 2
        return nn.GroupNorm(g, channels)
    if kind == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    raise ValueError(f"unknown norm {kind}")


def make_act(kind: str) -> nn.Module:
    return nn.SiLU(inplace=True) if kind == "silu" else nn.ReLU(inplace=True)


class ConvNormAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, stride=1, norm="group", act="silu"):
        super().__init__(
            nn.Conv3d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
            make_norm(norm, cout), make_act(act))


class ResBlock3d(nn.Module):
    """Two 3x3x3 conv-norm-act layers with an identity/1x1 projection shortcut."""
    def __init__(self, cin, cout, stride=1, norm="group", act="silu"):
        super().__init__()
        self.c1 = ConvNormAct(cin, cout, 3, stride, norm, act)
        self.c2 = nn.Sequential(nn.Conv3d(cout, cout, 3, padding=1, bias=False), make_norm(norm, cout))
        self.short = (nn.Identity() if cin == cout and stride == 1 else
                      nn.Sequential(nn.Conv3d(cin, cout, 1, stride=stride, bias=False), make_norm(norm, cout)))
        self.act = make_act(act)

    def forward(self, x):
        return self.act(self.c2(self.c1(x)) + self.short(x))


class Down(nn.Module):
    """Strided-conv downsampling stage (halves each spatial dim) followed by a residual block."""
    def __init__(self, cin, cout, norm="group", act="silu"):
        super().__init__()
        self.body = nn.Sequential(ConvNormAct(cin, cout, 3, 2, norm, act), ResBlock3d(cout, cout, 1, norm, act))

    def forward(self, x):
        return self.body(x)


class Up(nn.Module):
    """Trilinear upsample (MPS-supported; avoids transposed-conv checkerboard) + skip concat + res block."""
    def __init__(self, cin, cskip, cout, norm="group", act="silu"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.reduce = ConvNormAct(cin, cout, 1, 1, norm, act)
        self.body = ResBlock3d(cout + cskip, cout, 1, norm, act)

    def forward(self, x, skip):
        x = self.reduce(self.up(x))
        if x.shape[2:] != skip.shape[2:]:  # odd sizes
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.body(torch.cat([x, skip], dim=1))
