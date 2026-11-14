from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi


def overlap_metrics(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum()); tn = int(p.size - tp - fp - fn)
    dice = 1.0 if (p.sum() + g.sum()) == 0 else 2 * tp / (p.sum() + g.sum())
    iou = 1.0 if (p | g).sum() == 0 else tp / (p | g).sum()
    return {"dice": dice, "iou": iou, "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"), "fpr": fp / (fp + tn) if fp + tn else float("nan"),
            "tp": tp, "fp": fp, "fn": fn}


def _surface(m):
    return m & ~ndi.binary_erosion(m, border_value=0)


def surface_distances(pred, gt, spacing):
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() or not g.any(): return None
    sp, sg = _surface(p), _surface(g)
    d_to_g = ndi.distance_transform_edt(~sg, sampling=spacing); d_to_p = ndi.distance_transform_edt(~sp, sampling=spacing)
    return d_to_g[sp], d_to_p[sg]


def hd95(pred, gt, spacing=(1, 1, 1)):
    s = surface_distances(pred, gt, spacing)
    return float("nan") if s is None else float(np.percentile(np.concatenate(s), 95))


def surface_dice(pred, gt, spacing=(1, 1, 1), tol_mm=1.0):
    s = surface_distances(pred, gt, spacing)
    if s is None: return float("nan")
    a, b = s; return float(((a <= tol_mm).sum() + (b <= tol_mm).sum()) / (len(a) + len(b)))


def all_metrics(pred, gt, spacing=(1, 1, 1)):
    m = overlap_metrics(pred, gt); m["hd95_mm"] = hd95(pred, gt, spacing); m["surface_dice_1mm"] = surface_dice(pred, gt, spacing); return m
