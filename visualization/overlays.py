from __future__ import annotations
import numpy as np


def window(arr, level=-600, width=1500):
    lo, hi = level - width / 2, level + width / 2
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1)


def get_slice(vol, plane, idx):
    """vol is (z,y,x). axial: z; coronal: y; sagittal: x. Returns 2D array (rows, cols)."""
    return {"axial": lambda: vol[idx], "coronal": lambda: vol[:, idx, :], "sagittal": lambda: vol[:, :, idx]}[plane]()


def plane_aspect(plane, spacing_zyx):
    """pixel aspect (row_mm/col_mm) so physical proportions are preserved."""
    z, y, x = spacing_zyx
    return {"axial": y / x, "coronal": z / x, "sagittal": z / y}[plane]


def axis_len(plane, shape):
    return {"axial": shape[0], "coronal": shape[1], "sagittal": shape[2]}[plane]
