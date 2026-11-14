"""Synthetic 3D CT-like volumes with known spheres (HU background -800 lung, nodules +50)."""
import numpy as np


def make_volume(shape=(64, 64, 64), n_spheres=2, radius=(4, 8), seed=0):
    rng = np.random.default_rng(seed)
    vol = rng.normal(-800, 20, shape).astype(np.float32)
    mask = np.zeros(shape, np.uint8)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    centers = []
    for _ in range(n_spheres):
        r = rng.integers(*radius)
        c = [int(rng.integers(r + 2, s - r - 2)) for s in shape]
        d = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= r * r
        vol[d] = rng.normal(50, 20, d.sum()); mask[d] = 1; centers.append((c, r))
    return vol, mask, centers
