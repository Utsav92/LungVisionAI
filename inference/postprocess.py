"""Helpers to bring predictions back to the original CT grid."""
from __future__ import annotations
import numpy as np, SimpleITK as sitk


def probability_to_original_grid(prob_zyx, meta, threshold=None):
    """Resample a (z,y,x) volume on the resampled LPS grid back to the original-scan LPS grid (linear for prob, NN for mask)."""
    r, o = meta["resampled"], meta["original"]
    img = sitk.GetImageFromArray(prob_zyx.astype(np.float32))
    img.SetSpacing(r["spacing"]); img.SetOrigin(r["origin"]); img.SetDirection(r["direction"])
    ref = sitk.Image(o["size"], sitk.sitkFloat32); ref.SetSpacing(o["spacing"]); ref.SetOrigin(o["origin"]); ref.SetDirection(o["direction"])
    out = sitk.Resample(img, ref, sitk.Transform(), sitk.sitkLinear, 0.0)
    a = sitk.GetArrayFromImage(out)
    return (a >= threshold).astype(np.uint8) if threshold is not None else a
