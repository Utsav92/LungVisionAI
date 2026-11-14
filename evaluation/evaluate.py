"""Full-volume evaluation on a split: segmentation metrics, detection FROC, subgroup tables.
python -m evaluation.evaluate --ckpt experiments/pilot_unet16/best.pt --split test --out reports/eval_test"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy import ndimage as ndi
from lvdata.preprocess_ct import PROC
from inference.segmentation import load_model, segment_volume
from inference.nodule_detection import detect_nodules, eq_diameter
from evaluation.segmentation_metrics import all_metrics
from evaluation.froc import match_scan, froc_curve, sens_at_fp, bootstrap_froc, FP_RATES
from training.trainer import pick_device

ROOT = Path(__file__).resolve().parents[1]


def ref_nodules(mask, spacing=(1, 1, 1)):
    lab, n = ndi.label(mask > 0, structure=np.ones((3, 3, 3))); out = []
    for i in range(1, n + 1):
        v = (lab == i).sum() * np.prod(spacing)
        out.append({"centroid_zyx": tuple(float(x) for x in ndi.center_of_mass(lab == i)), "diameter_mm": eq_diameter(v), "volume_mm3": float(v)})
    return out


def binned(val, bins):
    for lo, hi in bins:
        if lo <= val < hi: return f"{lo}-{hi}" if hi < 1000 else f">={lo}"
    return "other"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True); ap.add_argument("--max-scans", type=int, default=10**9)
    ap.add_argument("--inference-config", default=str(ROOT / "configs/inference.yaml")); ap.add_argument("--no-bootstrap", action="store_true")
    a = ap.parse_args(); ic = yaml.safe_load(open(a.inference_config)); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = pick_device("mps"); model, ck = load_model(a.ckpt, dev.type)
    pc = pd.read_csv(ROOT / "data/manifests/processed_cases.csv"); pc = pc[pc.split == a.split].head(a.max_scans)
    sc = pd.read_csv(ROOT / "data/manifests/scan_manifest.csv").set_index("case_id")
    nod = pd.read_csv(ROOT / "data/raw/lidc/nodules.csv")
    per_scan, groups, seg_rows, cand_rows, ref_rows = [], [], [], [], []
    for r in pc.itertuples():
        t = time.time(); vol = np.load(PROC / "volumes" / f"{r.case_id}.img.npy"); gt = np.load(PROC / "volumes" / f"{r.case_id}.mask.npy")
        meta = json.loads((PROC / "volumes" / f"{r.case_id}.meta.json").read_text())
        prob, st = segment_volume(model, ck, vol, dev.type, ic["overlap"], ic["batch_size"])
        np.save(out / f"{r.case_id}.prob.npy", prob.astype(np.float16))
        pred = prob >= ic["detect_threshold"]
        m = all_metrics(pred, gt, (1, 1, 1)); m.update(case_id=r.case_id, patient_id=r.patient_id, gt_voxels=int(gt.sum()), pred_voxels=int(pred.sum()),
                                                       spacing_z=float(sc.loc[r.case_id, "spacing_z"]), scan_seconds=time.time() - t, **{"windows_run": st["windows_run"]})
        seg_rows.append(m)
        cands = detect_nodules(prob, (1, 1, 1), ic["detect_threshold"], ic["min_diameter_mm"], ic["dedup_dist_mm"], meta)
        cd = [{"centroid_zyx": c.centroid_zyx, "confidence": c.confidence} for c in cands]; refs = ref_nodules(gt)
        matches = match_scan(cd, refs); per_scan.append((matches, len(refs))); groups.append(r.patient_id)
        hit = {h for _, h in matches if h >= 0}
        for j, rf in enumerate(refs):
            ref_rows.append({"case_id": r.case_id, "patient_id": r.patient_id, "ref_idx": j, "diameter_mm": rf["diameter_mm"], "detected": j in hit,
                             "size_bin": binned(rf["diameter_mm"], ic["size_bins_mm"]), "spacing_z": float(sc.loc[r.case_id, "spacing_z"])})
        for c, (conf, h) in zip(cands, matches): cand_rows.append({"case_id": r.case_id, **c.to_dict(), "matched_ref": h})
        print(f"{r.case_id}: dice {m['dice']:.3f} refs {len(refs)} cands {len(cands)} hits {len(hit)} ({time.time()-t:.0f}s)", flush=True)
    seg, refd, cdf = pd.DataFrame(seg_rows), pd.DataFrame(ref_rows), pd.DataFrame(cand_rows)
    seg.to_csv(out / "per_scan_segmentation.csv", index=False); refd.to_csv(out / "per_reference_nodule.csv", index=False); cdf.to_csv(out / "candidates.csv", index=False)
    thr, sens, fps = froc_curve(per_scan); at = sens_at_fp(sens, fps)
    ci = {} if a.no_bootstrap else bootstrap_froc(per_scan, groups)
    pos = seg[seg.gt_voxels > 0]
    res = {"split": a.split, "n_scans": len(seg), "n_patients": len(set(groups)), "n_reference_nodules": len(refd), "checkpoint": a.ckpt, "checkpoint_split_id": ck.get("split_id"),
           "segmentation_full_volume": {"mean_dice_all_scans": float(seg.dice.mean()), "mean_dice_scans_with_gt": float(pos.dice.mean()) if len(pos) else None,
                                         "mean_iou_scans_with_gt": float(pos.iou.mean()) if len(pos) else None, "mean_hd95_mm": float(pos.hd95_mm.mean(skipna=True)) if len(pos) else None,
                                         "mean_surface_dice_1mm": float(pos.surface_dice_1mm.mean(skipna=True)) if len(pos) else None,
                                         "micro_dice": float(2 * seg.tp.sum() / max(2 * seg.tp.sum() + seg.fp.sum() + seg.fn.sum(), 1))},
           "detection": {"lesion_sensitivity_at_threshold": float(refd.detected.mean()) if len(refd) else None, "fp_per_scan_at_threshold": float((cdf.matched_ref < 0).sum() / max(len(seg), 1)) if len(cdf) else 0.0,
                         "sensitivity_at_fp_per_scan": {str(k): v for k, v in at.items()}, "sensitivity_ci95_patient_bootstrap": {str(k): v for k, v in ci.items()},
                         "froc_mean_sensitivity_7pt_NOT_official_CPM": float(np.mean(list(at.values())))},
           "subgroups": {"by_size_bin": refd.groupby("size_bin").detected.agg(["mean", "count"]).reset_index().to_dict("records"),
                         "by_slice_thickness": refd.assign(th=refd.spacing_z.map(lambda v: binned(v, ic["slice_thickness_bins_mm"]))).groupby("th").detected.agg(["mean", "count"]).reset_index().to_dict("records")}}
    (out / "metrics.json").write_text(json.dumps(res, indent=2)); np.savez(out / "froc_curve.npz", thr=thr, sens=sens, fps=fps)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
