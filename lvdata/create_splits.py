"""Deterministic patient-level 70/15/15 split over the FULL manifest. Refuses to overwrite an existing split.

Group = patient_id. Scans sharing a SeriesInstanceUID necessarily share a patient, so they cannot cross partitions
(verified explicitly). Stratified over groups by (has default-GT nodules) so each partition has nodule-bearing patients.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data" / "manifests"


def make_split(df: pd.DataFrame, seed=42, fr=(0.70, 0.15, 0.15)) -> pd.DataFrame:
    pat = df.groupby("patient_id").agg(has_nod=("nodule_count", lambda s: int((s > 0).any()))).reset_index()
    rng = np.random.default_rng(seed)
    assign = {}
    # stratified patient-group split: shuffle patients within each stratum, then cut by fraction
    for _, g in pat.groupby("has_nod"):
        ids = g.patient_id.sort_values().to_numpy(); rng.shuffle(ids)
        n = len(ids); a = int(round(n * fr[0])); b = a + int(round(n * fr[1]))
        for i, p in enumerate(ids): assign[p] = "train" if i < a else ("val" if i < b else "test")
    out = df.copy(); out["split"] = out.patient_id.map(assign)
    return out


def verify(df):
    P = {s: set(df[df.split == s].patient_id) for s in ["train", "val", "test"]}
    S = {s: set(df[df.split == s].series_uid) for s in P}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not P[a] & P[b], f"patient leakage {a}/{b}"
        assert not S[a] & S[b], f"series leakage {a}/{b}"
    assert df.split.notna().all() and set(df.split) <= set(P)
    return True


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = MAN / "split_manifest.csv"
    if out.exists():
        raise SystemExit(f"{out} exists. The split is persistent and must never be regenerated.")
    df = pd.read_csv(MAN / "scan_manifest.csv")
    sp = make_split(df, a.seed); verify(sp)
    sp[["case_id", "patient_id", "series_uid", "split"]].to_csv(out, index=False)
    sid = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    summ = {"split_id": sid, "seed": a.seed, "algorithm": "patient-grouped, stratified by has_default_GT_nodule, 70/15/15",
            "counts": {s: {"patients": int(sp[sp.split == s].patient_id.nunique()), "scans": int((sp.split == s).sum()),
                           "scans_with_nodules": int(((sp.split == s) & (sp.nodule_count > 0)).sum())} for s in ["train", "val", "test"]}}
    (MAN / "split_id.json").write_text(json.dumps(summ, indent=2)); print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
