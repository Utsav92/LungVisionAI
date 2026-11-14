"""Patch index (disk-backed, JSON) + positive/negative sampling and augmentation.

Positives: patch intersects a GT nodule component (centre jittered so nodules are not always centred).
Negatives: (a) random in-body points far from any GT voxel, (b) 'hard' points on lung vessels/walls (HU>-500 inside lung).
Unannotated != nodule-free (annotation policy: <3 readers or <50% consensus => background), so negatives are not
claimed to be healthy; they are simply 'not in default GT'.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def build_scan_index(case_id: str, neg_stride=8, margin=16):
    img = np.load(PROC / "volumes" / f"{case_id}.img.npy", mmap_mode="r")
    mp = PROC / "volumes" / f"{case_id}.mask.npy"
    mask = np.load(mp, mmap_mode="r") if mp.exists() else None
    sub = np.asarray(img[::neg_stride, ::neg_stride, ::neg_stride])
    body = sub > -500
    body = ndi.binary_opening(body, iterations=1)
    lung_air = (sub < -600) & (sub > -990)
    # lung = air regions not connected to the volume border (roughly)
    lab, n = ndi.label(lung_air)
    border = set(np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                           lab[:, :, 0].ravel(), lab[:, :, -1].ravel()])))
    keep = np.isin(lab, [l for l in range(1, n + 1) if l not in border])
    lung = ndi.binary_closing(keep, iterations=3)
    lung_dil = ndi.binary_dilation(lung, iterations=2)
    hard = lung_dil & (sub > -500)
    easy = lung
    pos = []
    if mask is not None and mask.any():
        lab3, k = ndi.label(np.asarray(mask))
        for i, sl in enumerate(ndi.find_objects(lab3), 1):
            vox = np.argwhere(lab3[sl] == i) + np.array([s.start for s in sl])
            pos.append({"centroid": vox.mean(0).round(1).tolist(), "bbox": [[int(s.start), int(s.stop)] for s in sl], "voxels": int(len(vox))})
    gt_near = np.zeros_like(lung)
    if mask is not None and mask.any():
        gt_near = ndi.binary_dilation(np.asarray(mask[::neg_stride, ::neg_stride, ::neg_stride]) > 0, iterations=margin // neg_stride + 2)
    def pts(m):
        c = np.argwhere(m & ~gt_near) * neg_stride + neg_stride // 2
        return c.astype(np.int32)
    easy_c, hard_c = pts(easy), pts(hard)
    return {"case_id": case_id, "shape": list(img.shape), "positives": pos, "n_easy": int(len(easy_c)), "n_hard": int(len(hard_c))}, easy_c, hard_c


def save_scan_index(case_id):
    d = PROC / "index"; d.mkdir(parents=True, exist_ok=True)
    meta, easy, hard = build_scan_index(case_id)
    np.save(d / f"{case_id}.easy.npy", easy); np.save(d / f"{case_id}.hard.npy", hard)
    (d / f"{case_id}.json").write_text(json.dumps(meta)); return meta


def load_scan_index(case_id):
    d = PROC / "index"
    return json.loads((d / f"{case_id}.json").read_text()), np.load(d / f"{case_id}.easy.npy"), np.load(d / f"{case_id}.hard.npy")


def crop(vol, center, size, pad_value):
    """Crop a cube of `size` around `center` (z,y,x ints) with constant padding."""
    out = np.full(size, pad_value, vol.dtype)
    lo = [int(c) - s // 2 for c, s in zip(center, size)]
    src_lo = [max(l, 0) for l in lo]; src_hi = [min(l + s, d) for l, s, d in zip(lo, size, vol.shape)]
    dst_lo = [sl - l for sl, l in zip(src_lo, lo)]; dst_hi = [dst_lo[i] + src_hi[i] - src_lo[i] for i in range(3)]
    if all(h > l for l, h in zip(src_lo, src_hi)):
        out[tuple(slice(a, b) for a, b in zip(dst_lo, dst_hi))] = vol[tuple(slice(a, b) for a, b in zip(src_lo, src_hi))]
    return out


def random_affine(img, msk, rng, out_size, rot_deg=12.0, scale=(0.9, 1.1)):
    """Small rotation + isotropic scale applied identically to image and mask (linear / nearest)."""
    ang = np.deg2rad(rng.uniform(-rot_deg, rot_deg, 3)); s = rng.uniform(*scale)
    cz, cy, cx = np.cos(ang); sz, sy, sx = np.sin(ang)
    Rz = np.array([[1, 0, 0], [0, cz, -sz], [0, sz, cz]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[cx, -sx, 0], [sx, cx, 0], [0, 0, 1]])
    M = (Rz @ Ry @ Rx) / s
    c_in = (np.array(img.shape) - 1) / 2; c_out = (np.array(out_size) - 1) / 2
    off = c_in - M @ c_out
    i2 = ndi.affine_transform(img, M, offset=off, output_shape=out_size, order=1, mode="constant", cval=float(img.min()))
    m2 = ndi.affine_transform(msk.astype(np.float32), M, offset=off, output_shape=out_size, order=0, mode="constant", cval=0)
    return i2, (m2 > 0.5).astype(np.float32), {"rot_deg": np.rad2deg(ang).tolist(), "scale": float(s)}


def augment(img, msk, rng, out_size, p_affine=0.5):
    params = {}
    if rng.random() < p_affine:
        img, msk, params["affine"] = random_affine(img, msk, rng, out_size)
    else:
        sl = tuple(slice((a - b) // 2, (a - b) // 2 + b) for a, b in zip(img.shape, out_size)); img, msk = img[sl], msk[sl]
    flips = [ax for ax in range(3) if rng.random() < 0.5]  # L-R / A-P / S-I mirrors: anatomically tolerable for nodule shape
    if flips: img, msk = np.flip(img, flips).copy(), np.flip(msk, flips).copy(); params["flips"] = flips
    if rng.random() < 0.8:
        a, b = rng.uniform(0.95, 1.05), rng.uniform(-0.03, 0.03); img = img * a + b; params["intensity"] = [float(a), float(b)]
    if rng.random() < 0.5:
        sig = rng.uniform(0, 0.02); img = img + rng.normal(0, sig, img.shape).astype(np.float32); params["noise_sigma"] = float(sig)
    return img.astype(np.float32), msk.astype(np.float32), params
