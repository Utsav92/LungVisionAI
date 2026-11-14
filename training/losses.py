from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


def soft_dice_loss(logits, target, eps=1.0):
    """Batch-global soft Dice (stable for patches with empty targets thanks to eps smoothing)."""
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    return 1 - (2 * inter + eps) / (p.sum() + target.sum() + eps)


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, pos_weight: float | None = None):
        super().__init__()
        self.dw, self.bw = dice_weight, bce_weight
        self.pos_weight = pos_weight

    def forward(self, logits, target):
        pw = None if self.pos_weight is None else torch.tensor(self.pos_weight, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw) if self.bw else logits.new_zeros(())
        dice = soft_dice_loss(logits, target) if self.dw else logits.new_zeros(())
        return self.dw * dice + self.bw * bce


def build_loss(cfg: dict):
    return DiceBCELoss(cfg.get("dice_weight", 0.5), cfg.get("bce_weight", 0.5), cfg.get("pos_weight"))
