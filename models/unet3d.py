"""Custom 3D residual U-Net, random init. Stages: stem -> enc1 -> enc2 -> enc3 -> bottleneck -> dec3..dec0."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from .blocks3d import ConvNormAct, ResBlock3d, Down, Up


@dataclass
class UNet3DConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 16
    norm: str = "group"
    act: str = "silu"
    deep_supervision: bool = False


class Custom3DUNet(nn.Module):
    def __init__(self, cfg: UNet3DConfig | None = None, **kw):
        super().__init__()
        self.cfg = cfg or UNet3DConfig(**kw)
        c, n, a = self.cfg.base_channels, self.cfg.norm, self.cfg.act
        self.stem = nn.Sequential(ConvNormAct(self.cfg.in_channels, c, 3, 1, n, a), ResBlock3d(c, c, 1, n, a))
        self.enc1 = Down(c, 2 * c, n, a)
        self.enc2 = Down(2 * c, 4 * c, n, a)
        self.enc3 = Down(4 * c, 8 * c, n, a)
        self.bottleneck = nn.Sequential(Down(8 * c, 16 * c, n, a), ResBlock3d(16 * c, 16 * c, 1, n, a))
        self.dec3 = Up(16 * c, 8 * c, 8 * c, n, a)
        self.dec2 = Up(8 * c, 4 * c, 4 * c, n, a)
        self.dec1 = Up(4 * c, 2 * c, 2 * c, n, a)
        self.dec0 = Up(2 * c, c, c, n, a)
        self.head = nn.Conv3d(c, self.cfg.out_channels, 1)  # 1x1x1 conv -> logits
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # bias prior for rare foreground (p~0.01) stabilises early BCE
        nn.init.constant_(self.head.bias, -4.6)

    def forward(self, x):
        """Returns LOGITS. Apply sigmoid for probabilities (predict_proba)."""
        s0 = self.stem(x); s1 = self.enc1(s0); s2 = self.enc2(s1); s3 = self.enc3(s2)
        b = self.bottleneck(s3)
        d = self.dec3(b, s3); d = self.dec2(d, s2); d = self.dec1(d, s1); d = self.dec0(d, s0)
        return self.head(d)

    @torch.no_grad()
    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))

    def summary(self, patch=(64, 64, 64), dtype_bytes=4):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        shapes, hooks = [], []
        for name, m in self.named_children():
            hooks.append(m.register_forward_hook(lambda mod, i, o, nm=name: shapes.append((nm, tuple(o.shape)))))
        was = self.training; self.eval()
        with torch.no_grad():
            self(torch.zeros(1, self.cfg.in_channels, *patch, device=next(self.parameters()).device))
        self.train(was)
        for h in hooks: h.remove()
        act_elems = sum(torch.Size(s[1]).numel() for s in shapes)
        return {"total_params": total, "trainable_params": train,
                "input_shape": (1, self.cfg.in_channels, *patch),
                "feature_maps": shapes,
                "approx_weights_MB": round(total * dtype_bytes / 2**20, 2),
                # coarse: top-level stage outputs only; real training peak is several x higher
                "approx_stage_activation_MB_fwd": round(act_elems * dtype_bytes / 2**20, 1),
                "config": asdict(self.cfg)}
