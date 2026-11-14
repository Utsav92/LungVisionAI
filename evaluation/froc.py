"""Free-response ROC. Matching: candidate centroid within the reference nodule radius (LUNA16-style hit criterion).
NOTE: this is OUR implementation on LIDC default-GT components, not the official LUNA16 evaluation script; its
'mean sensitivity over 7 FP rates' is therefore reported as such and NOT called CPM."""
from __future__ import annotations
import numpy as np

FP_RATES = (0.125, 0.25, 0.5, 1, 2, 4, 8)


def match_scan(cands, refs, spacing=(1, 1, 1), min_radius_mm=3.0):
    """cands: list of dict(centroid_zyx, confidence). refs: list of dict(centroid_zyx, diameter_mm).
    Returns per-candidate (conf, hit_ref_index or -1). A reference can be hit by multiple candidates; extras count as FP
    only if they don't hit any *new* ref? -> LUNA16 counts duplicates on an already-hit ref as ignored (not FP)."""
    sp = np.array(spacing); out = []
    for c in cands:
        cp = np.array(c["centroid_zyx"]) * sp; hit = -1
        for j, r in enumerate(refs):
            if np.linalg.norm(cp - np.array(r["centroid_zyx"]) * sp) <= max(r["diameter_mm"] / 2, min_radius_mm): hit = j; break
        out.append((c["confidence"], hit))
    return out


def froc_curve(per_scan):
    """per_scan: list of (matches list, n_refs). Returns thresholds, sensitivity, fp_per_scan arrays."""
    n_scans = len(per_scan); n_refs = sum(n for _, n in per_scan)
    allc = [(conf, si, hit) for si, (m, _) in enumerate(per_scan) for conf, hit in m]
    allc.sort(key=lambda t: -t[0]); thr = sorted({t[0] for t in allc}, reverse=True)
    sens, fps = [], []
    for t in thr:
        hits, fp = set(), 0
        for conf, si, hit in allc:
            if conf < t: break
            if hit >= 0: hits.add((si, hit))
            else: fp += 1
        sens.append(len(hits) / max(n_refs, 1)); fps.append(fp / max(n_scans, 1))
    return np.array(thr), np.array(sens), np.array(fps)


def sens_at_fp(sens, fps, rates=FP_RATES):
    out = {}
    for r in rates:
        ok = np.where(fps <= r)[0]; out[r] = float(sens[ok].max()) if len(ok) else 0.0
    return out


def bootstrap_froc(per_scan, groups, n=200, seed=0):
    """Patient-grouped bootstrap CI of sens@FP rates."""
    rng = np.random.default_rng(seed); ug = sorted(set(groups)); idx = {g: [i for i, x in enumerate(groups) if x == g] for g in ug}
    res = {r: [] for r in FP_RATES}
    for _ in range(n):
        pick = rng.choice(len(ug), len(ug)); sel = [per_scan[i] for g in pick for i in idx[ug[g]]]
        _, s, f = froc_curve(sel); d = sens_at_fp(s, f)
        for r in FP_RATES: res[r].append(d[r])
    return {r: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for r, v in res.items()}
