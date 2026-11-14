"""CT preprocessing: load NIfTI (HU) -> LPS orientation -> resample to isotropic spacing -> store int16 HU + uint8 mask.

Intensity windowing/normalisation is applied at load time (configurable, stored in checkpoints), NOT baked in.
Everything needed to map results back to the original scan grid is saved in <case>.meta.json.
"""
from __future__ import annotations
import json, argparse
from pathlib import Path
import numpy as np, SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "lidc"
PROC = ROOT / "data" / "processed"

DEFAULT_PRE = {"hu_min": -1000.0, "hu_max": 400.0, "spacing": [1.0, 1.0, 1.0], "orientation": "LPS", "range": [0.0, 1.0]}


def normalize_hu(arr, hu_min=-1000.0, hu_max=400.0, **_):
    """Clip to [hu_min, hu_max], scale to [0,1]. Fixed (scan-independent) window keeps HU semantics."""
    a = np.clip(arr.astype(np.float32), hu_min, hu_max)
    return (a - hu_min) / (hu_max - hu_min)


def read_oriented(path, orient="LPS"):
    img = sitk.ReadImage(str(path))
    return sitk.DICOMOrient(img, orient)


def resample(img, spacing, is_mask, default=-1024):
    spacing = [float(s) for s in spacing]
    size = [int(round(osz * osp / nsp)) for osz, osp, nsp in zip(img.GetSize(), img.GetSpacing(), spacing)]
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(spacing); f.SetSize(size); f.SetOutputOrigin(img.GetOrigin()); f.SetOutputDirection(img.GetDirection())
    f.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    f.SetDefaultPixelValue(0 if is_mask else default)
    return f.Execute(img)


def geom(img):
    return {"size": list(img.GetSize()), "spacing": list(img.GetSpacing()), "origin": list(img.GetOrigin()),
            "direction": list(img.GetDirection())}


def preprocess_case(case_id, spacing=(1.0, 1.0, 1.0), force=False, with_mask=True):
    out = PROC / "volumes"; out.mkdir(parents=True, exist_ok=True)
    if (out / f"{case_id}.meta.json").exists() and not force:
        return json.loads((out / f"{case_id}.meta.json").read_text())
    img = read_oriented(RAW / "images" / f"{case_id}.nii.gz")
    meta = {"case_id": case_id, "orientation": "LPS", "original": geom(img), "target_spacing": list(spacing)}
    r = resample(img, spacing, False)
    arr = np.clip(sitk.GetArrayFromImage(r), -1024, 3071).astype(np.int16)  # (z,y,x)
    np.save(out / f"{case_id}.img.npy", arr)
    meta["resampled"] = geom(r); meta["array_axes"] = "z,y,x (SimpleITK array order)"
    if with_mask and (RAW / "masks" / f"{case_id}.nii.gz").exists():
        m = read_oriented(RAW / "masks" / f"{case_id}.nii.gz")
        assert m.GetSize() == img.GetSize(), f"image/mask size mismatch {case_id}"
        assert np.allclose(m.GetOrigin(), img.GetOrigin(), atol=1e-3) and np.allclose(m.GetSpacing(), img.GetSpacing(), atol=1e-4)
        mr = sitk.GetArrayFromImage(resample(m, spacing, True)).astype(np.uint8)
        np.save(out / f"{case_id}.mask.npy", mr)
        meta["mask_voxels"] = int(mr.sum())
    (out / f"{case_id}.meta.json").write_text(json.dumps(meta))
    return meta


def voxel_to_original_index(zyx, meta):
    """Map resampled-grid index (z,y,x) -> original-grid continuous index (x,y,z), via physical coords."""
    r, o = meta["resampled"], meta["original"]
    def phys(g, idx_xyz):
        d = np.array(g["direction"]).reshape(3, 3); return np.array(g["origin"]) + d @ (np.array(idx_xyz) * np.array(g["spacing"]))
    z, y, x = zyx
    p = phys(r, (x, y, z)); d = np.array(o["direction"]).reshape(3, 3)
    return tuple((np.linalg.inv(d) @ (p - np.array(o["origin"]))) / np.array(o["spacing"]))


def voxel_to_physical(zyx, meta):
    r = meta["resampled"]; d = np.array(r["direction"]).reshape(3, 3); z, y, x = zyx
    return tuple(np.array(r["origin"]) + d @ (np.array([x, y, z]) * np.array(r["spacing"])))


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", nargs="*"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cases = a.cases or pd.read_csv(ROOT / "data/manifests/downloaded_cases.csv").case_id.tolist()
    for c in cases:
        m = preprocess_case(c, force=a.force); print(c, m["resampled"]["size"], "mask vox", m.get("mask_voxels"), flush=True)


if __name__ == "__main__":
    main()
