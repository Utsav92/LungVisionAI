"""Dataset audit -> reports/dataset_audit.{json,csv,html}. Metadata checks run on all scans; file-level checks
(corruption, alignment, intensity, orientation, empty masks) run on scans present locally."""
from __future__ import annotations
import json, html, argparse
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "lidc"; REP = ROOT / "reports"


def audit_file(r):
    row = {"case_id": r.case_id, "patient_id": r.patient_id, "issues": []}
    ip, mp = RAW / r.scan_path, RAW / r.mask_path
    row["img_exists"], row["mask_exists"] = ip.exists(), mp.exists()
    if not ip.exists(): row["issues"].append("missing_image")
    if not mp.exists(): row["issues"].append("missing_mask")
    if not (ip.exists() and mp.exists()): return row
    try:
        im, mk = nib.load(ip), nib.load(mp); a = np.asarray(im.dataobj); m = np.asarray(mk.dataobj)
    except Exception as e:
        row["issues"].append(f"corrupt:{type(e).__name__}"); return row
    row["shape"] = list(a.shape); row["zooms"] = [round(float(z), 4) for z in im.header.get_zooms()[:3]]
    row["orientation"] = "".join(nib.aff2axcodes(im.affine)); row["mask_orientation"] = "".join(nib.aff2axcodes(mk.affine))
    if a.shape != m.shape: row["issues"].append("shape_mismatch")
    if not np.allclose(im.affine, mk.affine, atol=1e-3): row["issues"].append("affine_mismatch")
    if a.ndim != 3 or min(a.shape) < 16: row["issues"].append("invalid_dims")
    if not all(0.2 < z < 10 for z in row["zooms"]): row["issues"].append("invalid_spacing")
    exp = [r.spacing_x, r.spacing_y, r.spacing_z]
    if not np.allclose(sorted(row["zooms"]), sorted(exp), atol=1e-2): row["issues"].append("spacing_differs_from_csv")
    row["hu_min"], row["hu_max"], row["hu_p1"], row["hu_p99"] = int(a.min()), int(a.max()), float(np.percentile(a[::4, ::4, ::2], 1)), float(np.percentile(a[::4, ::4, ::2], 99))
    row["has_fov_padding"] = bool(row["hu_min"] <= -2000)  # scanner out-of-FOV padding (-2048/-3024): expected; clipped in preprocessing
    if row["hu_min"] < -3100 or row["hu_max"] > 32000 or row["hu_p99"] > 1000: row["issues"].append("unusual_intensity")
    row["mask_values"] = sorted(int(v) for v in np.unique(m)); row["mask_voxels"] = int(m.sum())
    if not set(row["mask_values"]) <= {0, 1}: row["issues"].append("non_binary_mask")
    if row["mask_voxels"] == 0: row["empty_mask_note"] = "empty default-GT mask (annotation policy: nodules need >=3 reads); NOT excluded"
    if row["mask_voxels"] != r.default_gt_voxels: row["issues"].append("mask_voxels_differs_from_scans_csv")
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-files", type=int, default=10**9); a = ap.parse_args()
    df = pd.read_csv(ROOT / "data/manifests/scan_manifest.csv"); REP.mkdir(exist_ok=True)
    nod = pd.read_csv(RAW / "nodules.csv")
    meta = {
        "n_scans": len(df), "n_patients": int(df.patient_id.nunique()),
        "duplicate_series_uid": int(df.series_uid.duplicated().sum()), "duplicate_case_id": int(df.case_id.duplicated().sum()),
        "patients_with_multiple_scans": sorted(df[df.patient_id.duplicated(keep=False)].patient_id.unique().tolist()),
        "scans_without_default_gt_nodule": int((df.nodule_count == 0).sum()),
        "scans_without_any_contour(has_seg=False)": int((~df.has_seg).sum()),
        "irregular_z_spacing_scans": int((df.z_gap_max - df.z_gap_min > 0.01).sum()),
        "slice_thickness_distribution": df.spacing_z.round(2).value_counts().head(8).to_dict(),
        "manufacturers": df.manufacturer.value_counts().to_dict(),
        "annotation_policy": "default mask = nodules with >=3 annotations; voxel kept if >=50% of that nodule's annotations include it. "
                             "1-2-annotation nodules are background in masks/ (NOT 'nodule-free'). Nodules >=3mm only.",
        "annotation_disagreement": {"nodules_total": len(nod), "nodules_in_default_gt": int(nod.in_default_gt.sum()),
                                    "by_n_annotations": nod.n_annotations.value_counts().sort_index().to_dict(),
                                    "median_dice_like(intersection/union)": float((nod.intersection_voxels / nod.union_voxels).median())},
        "conflicting_label_definitions": "malignancy_score is a radiologist 1-5 suspicion rating, not pathology.",
        "luna16_overlap": {"in_luna16": int(df.in_luna16.sum()), "note": "LUNA16 is a strict subset of LIDC-IDRI; NOT independent validation."},
    }
    local = df[[ (RAW / p).exists() for p in df.scan_path ]].head(a.max_files)
    rows = [audit_file(r) for r in local.itertuples()]
    meta["file_level_scans_checked"] = len(rows)
    meta["issue_counts"] = pd.Series([i.split(":")[0] for r in rows for i in r["issues"]]).value_counts().to_dict()
    meta["orientation_codes"] = pd.Series([r.get("orientation") for r in rows]).value_counts().to_dict()
    meta["empty_default_masks_among_checked"] = int(sum(r.get("mask_voxels", 1) == 0 for r in rows))
    (REP / "dataset_audit.json").write_text(json.dumps({"summary": meta, "files": rows}, indent=2, default=str))
    pd.DataFrame([{**r, "issues": ";".join(r["issues"])} for r in rows]).to_csv(REP / "dataset_audit.csv", index=False)
    h = "<html><body><h1>LIDC-IDRI dataset audit</h1><pre>" + html.escape(json.dumps(meta, indent=2, default=str)) + "</pre>" \
        + pd.DataFrame([{**r, "issues": ";".join(r["issues"])} for r in rows]).to_html(index=False) + "</body></html>"
    (REP / "dataset_audit.html").write_text(h); print(json.dumps(meta, indent=2, default=str)[:2500])


if __name__ == "__main__":
    main()
