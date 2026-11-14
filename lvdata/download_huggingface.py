"""Inspect MedOtter/LIDC-IDRI, build the scan manifest, and resumably download a chosen subset.

Never downloads everything blindly: prints total size and refuses if free disk would drop below --reserve-gb.
Usage:
  python -m lvdata.download_huggingface --inspect
  python -m lvdata.download_huggingface --n-scans 60 --only-with-nodules
"""
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "MedOtter/LIDC-IDRI"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "lidc"
MAN = ROOT / "data" / "manifests"
LICENSE = "CC BY 3.0 (TCIA LIDC-IDRI; QIICR DICOM-LIDC-IDRI-Nodules). Attribution required; see README."


def fetch_meta():
    RAW.mkdir(parents=True, exist_ok=True)
    return {f: hf_hub_download(REPO, f, repo_type="dataset", local_dir=RAW) for f in ["scans.csv", "nodules.csv", "README.md"]}


def file_sizes(prefix):
    api = HfApi()
    return {e.path: e.size for e in api.list_repo_tree(REPO, path_in_repo=prefix, repo_type="dataset") if hasattr(e, "size")}


def build_manifest():
    paths = fetch_meta()
    scans = pd.read_csv(paths["scans.csv"])
    scans["scan_path"] = "images/" + scans.case_id + ".nii.gz"
    scans["mask_path"] = "masks/" + scans.case_id + ".nii.gz"
    scans["voxel_spacing"] = scans[["spacing_x", "spacing_y", "spacing_z"]].apply(lambda r: [float(v) for v in r], axis=1)
    scans["image_shape"] = scans[["rows", "cols", "n_slices"]].apply(lambda r: [int(v) for v in r], axis=1)
    scans["nodule_count"] = scans.n_nodules_default       # nodules inside the default GT mask
    scans["annotation_count"] = scans.n_annotations
    scans["source_dataset"] = "LIDC-IDRI (MedOtter/LIDC-IDRI mirror)"
    scans["source_version"] = HfApi().dataset_info(REPO).sha
    scans["license"] = LICENSE
    return scans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--n-scans", type=int, default=0)
    ap.add_argument("--only-with-nodules", action="store_true", help="prefer scans that have default-GT nodules")
    ap.add_argument("--reserve-gb", type=float, default=40.0)
    ap.add_argument("--cases", nargs="*", help="explicit case_ids (overrides selection)")
    ap.add_argument("--split-manifest", default=str(MAN / "split_manifest.csv"))
    a = ap.parse_args()

    scans = build_manifest()
    img_sizes, msk_sizes = file_sizes("images"), file_sizes("masks")
    scans["image_bytes"] = scans.scan_path.map(img_sizes)
    scans["mask_bytes"] = scans.mask_path.map(msk_sizes)
    MAN.mkdir(parents=True, exist_ok=True)
    out = MAN / "scan_manifest.csv"
    scans.to_csv(out, index=False)
    tot = scans.image_bytes.sum() + scans.mask_bytes.sum()
    free = shutil.disk_usage(ROOT).free
    print(f"scans={len(scans)} patients={scans.patient_id.nunique()} total download={tot/1e9:.1f} GB free disk={free/1e9:.1f} GB")
    print(f"scans with default-GT nodules: {(scans.nodule_count>0).sum()}; manifest -> {out}")
    if a.inspect or not (a.n_scans or a.cases):
        return

    sel = scans
    have_split = Path(a.split_manifest).exists()
    if have_split:
        sel = sel.merge(pd.read_csv(a.split_manifest)[["case_id", "split"]], on="case_id")
    if a.cases:
        sel = sel[sel.case_id.isin(a.cases)]
    else:
        if a.only_with_nodules:
            sel = sel[sel.nodule_count > 0]
        if have_split:  # keep train/val/test material in the pilot
            n = a.n_scans; ntr, nva = int(n * .6), int(n * .2)
            per = {"train": ntr, "val": nva, "test": n - ntr - nva}
            sel = pd.concat([g.sort_values("case_id").head(per[s]) for s, g in sel.groupby("split") if s in per])
        else:
            sel = sel.sort_values("case_id").head(a.n_scans)
    need = sel.image_bytes.sum() + sel.mask_bytes.sum()
    if free - need < a.reserve_gb * 1e9:
        sys.exit(f"REFUSING: need {need/1e9:.1f} GB, free {free/1e9:.1f} GB, reserve {a.reserve_gb} GB")
    print(f"downloading {len(sel)} scans ({need/1e9:.2f} GB) ...")
    for i, r in enumerate(sel.itertuples(), 1):
        for rel in (r.scan_path, r.mask_path):  # hf_hub_download resumes partial files and checks size/etag
            hf_hub_download(REPO, rel, repo_type="dataset", local_dir=RAW)
        print(f"  [{i}/{len(sel)}] {r.case_id}", flush=True)
    sel[["case_id"]].to_csv(MAN / "downloaded_cases.csv", index=False)


if __name__ == "__main__":
    main()
