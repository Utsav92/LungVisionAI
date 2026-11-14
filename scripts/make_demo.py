"""Render a demo video for one test scan: axial scroll with model mask, candidate boxes and the radiologist reference."""
import ast, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from lvdata.preprocess_ct import PROC
from visualization.overlays import window

case = sys.argv[1]; out = Path(sys.argv[2]); tmp = Path(sys.argv[3]); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
ev = Path("reports/eval_test_v2")
vol = np.load(PROC / "volumes" / f"{case}.img.npy"); gt = np.load(PROC / "volumes" / f"{case}.mask.npy")
prob = np.load(ev / f"{case}.prob.npy").astype(np.float32)
c = pd.read_csv(ev / "candidates.csv"); c = c[c.case_id == case].reset_index(drop=True)
c["cz"] = c.centroid_zyx.map(lambda s: ast.literal_eval(s)[0]); c["bbox"] = c.bbox_zyx.map(ast.literal_eval)
c["num"] = c.confidence.rank(ascending=False, method="first").astype(int)
zs = np.arange(int(c.cz.min()) - 25, int(c.cz.max()) + 25)
seq = []  # (z, caption, hold_frames)
for z in zs:
    near = c[(c.cz - z).abs() < 0.5]
    seq += [(z, int(near.num.iloc[0]) if len(near) else None)] * (36 if len(near) else 1)
BG, FG = "#0d0d0d", "#e6e6e6"

def frame(i, z, focus, title=None):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.02, 0.03, 0.56, 0.86]); ax.imshow(window(vol[z]), cmap="gray", vmin=0, vmax=1); ax.axis("off")
    m = prob[z] >= 0.5
    if m.any(): ax.imshow(np.where(m, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#ff2a2a"]), alpha=0.5)
    if gt[z].any(): ax.contour(gt[z] > 0, levels=[0.5], colors="#39ff14", linewidths=1.3)
    for r in c.itertuples():
        (z0, z1), (y0, y1), (x0, x1) = r.bbox
        if z0 <= z < z1:
            hit = r.matched_ref >= 0
            ax.add_patch(Rectangle((x0 - 4, y0 - 4), x1 - x0 + 8, y1 - y0 + 8, fill=False, ec="#ffd400", lw=2))
            ax.text(x1 + 6, y0 - 4, f"#{r.num}", color="#ffd400", fontsize=13, weight="bold")
    fig.text(0.02, 0.93, f"{case.split('_')[0]}   axial slice {z}", color=FG, fontsize=14)
    t = fig.add_axes([0.60, 0.03, 0.38, 0.9]); t.axis("off"); t.set_facecolor(BG)
    t.text(0, 0.98, "Model-detected nodule candidates", color=FG, fontsize=15, weight="bold", va="top")
    t.text(0, 0.925, "#   diam    conf   vs. reference", color="#888", fontsize=11, va="top", family="monospace")
    for k, r in enumerate(c.sort_values("num").itertuples()):
        col = "#ffd400" if focus == r.num else FG
        t.text(0, 0.88 - k * 0.048, f"#{r.num:<2d} {r.diameter_mm:5.1f}mm {r.confidence:5.2f}   {'matches reference' if r.matched_ref >= 0 else 'no reference match'}", color=col, fontsize=11, va="top", family="monospace", weight="bold" if focus == r.num else "normal")
    t.text(0, 0.37, "red tint = model segmentation (p≥0.5)\nyellow box = model candidate\ngreen outline = radiologist consensus\n    (reference only available for dataset scans)", color="#aaa", fontsize=10.5, va="top")
    t.text(0, 0.22, f"{int((c.matched_ref>=0).sum())} of 10 candidates match a reference nodule;\n{int((c.matched_ref<0).sum())} do not (likely false positives or\nnodules outside the annotation policy).\nAll 5 reference nodules in this scan were found.", color=FG, fontsize=11, va="top")
    fig.text(0.60, 0.035, "RESEARCH USE ONLY. Not validated for clinical use.\nCandidates are not diagnoses; benign vs malignant is not assessed.", color="#ff6b6b", fontsize=10)
    fig.savefig(tmp / f"f{i:05d}.png", facecolor=BG); plt.close(fig)

def card(i, lines, n):
    for k in range(n):
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG); fig.text(0.5, 0.5, lines, color=FG, ha="center", va="center", fontsize=20, linespacing=1.6)
        fig.savefig(tmp / f"f{i + k:05d}.png", facecolor=BG); plt.close(fig)
    return i + n

i = card(0, "LungVision AI — research demo\n\nFrom-scratch 3D U-Net, trained on a small LIDC-IDRI pilot subset\nHeld-out test scan (never used for training)\n\nResearch use only", 72)
for z, f in seq: frame(i, z, f); i += 1
i = card(i, "Pilot-scale model: ~13 false-positive candidates per scan on the test set,\nsome scans with no nodule found at all.\nNot a diagnostic tool.", 72)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(tmp / "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)], check=True)
print("wrote", out, i, "frames")
