"""Model B: probability volume -> nodule candidates (threshold, CCL, volume, bbox, centroid, mm diameter, confidence, dedup)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from scipy import ndimage as ndi
from lvdata.preprocess_ct import voxel_to_physical, voxel_to_original_index


@dataclass
class Candidate:
    centroid_zyx: tuple
    bbox_zyx: tuple          # ((z0,z1),(y0,y1),(x0,x1)) in resampled grid, end-exclusive
    volume_mm3: float
    diameter_mm: float       # equivalent-sphere diameter
    max_extent_mm: float     # longest bbox side (upper bound on max diameter)
    confidence: float        # mean probability inside component
    peak_prob: float
    centroid_physical_lps: tuple | None = None
    centroid_original_xyz: tuple | None = None

    def to_dict(self): return asdict(self)


def eq_diameter(volume_mm3):
    return float(2 * (3 * volume_mm3 / (4 * np.pi)) ** (1 / 3))


def detect_nodules(prob, spacing_zyx=(1.0, 1.0, 1.0), threshold=0.5, min_diameter_mm=3.0, dedup_dist_mm=4.0, meta=None):
    vox_mm3 = float(np.prod(spacing_zyx))
    lab, n = ndi.label(prob >= threshold, structure=np.ones((3, 3, 3)))
    cands = []
    if n:
        idx = np.arange(1, n + 1)
        vols = ndi.sum(np.ones_like(prob), lab, idx); means = ndi.mean(prob, lab, idx); maxs = ndi.maximum(prob, lab, idx)
        coms = ndi.center_of_mass(np.ones_like(prob), lab, idx); objs = ndi.find_objects(lab)
        for i in range(n):
            v = float(vols[i]) * vox_mm3; d = eq_diameter(v)
            if d < min_diameter_mm: continue
            bb = tuple((int(s.start), int(s.stop)) for s in objs[i])
            ext = max((b[1] - b[0]) * sp for b, sp in zip(bb, spacing_zyx))
            c = Candidate(tuple(float(x) for x in coms[i]), bb, v, d, float(ext), float(means[i]), float(maxs[i]))
            if meta is not None:
                c.centroid_physical_lps = tuple(float(x) for x in voxel_to_physical(c.centroid_zyx, meta))
                c.centroid_original_xyz = tuple(float(x) for x in voxel_to_original_index(c.centroid_zyx, meta))
            cands.append(c)
    return dedup(cands, spacing_zyx, dedup_dist_mm)


def dedup(cands, spacing_zyx, dist_mm):
    """Greedy NMS: keep the highest-confidence candidate, drop others whose centroid lies within max(dist_mm, radius)."""
    out = []
    for c in sorted(cands, key=lambda c: -c.confidence):
        p = np.array(c.centroid_zyx) * np.array(spacing_zyx)
        if all(np.linalg.norm(p - np.array(k.centroid_zyx) * np.array(spacing_zyx)) > max(dist_mm, k.diameter_mm / 2) for k in out):
            out.append(c)
    return out
