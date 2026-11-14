"""Build reports/lungvision_report.html (self-contained, base64 overlay images) from eval folders."""
import base64, io, json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lvdata.preprocess_ct import PROC
from visualization.overlays import window

ROOT = Path(__file__).resolve().parents[1]
runs = {"pilot 1 (15 train scans, 1 val)": ("reports/eval_test", "experiments/pilot_unet16"),
        "pilot 2 (20 train, 8 val)": ("reports/eval_test_v2", "experiments/pilot_unet16_v2")}
runs = {k: v for k, v in runs.items() if (ROOT / v[0] / "metrics.json").exists()}
M = {k: json.loads((ROOT / v[0] / "metrics.json").read_text()) for k, v in runs.items()}
def b64(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=110, bbox_inches="tight", facecolor="#111"); plt.close(fig); return base64.b64encode(b.getvalue()).decode()
def f(x, n=3): return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"

def curve_img():
    fig, ax = plt.subplots(figsize=(5.2, 3.6), facecolor="#111"); ax.set_facecolor("#111")
    for k, v in runs.items():
        z = np.load(ROOT / v[0] / "froc_curve.npz"); ax.plot(np.maximum(z["fps"], 0.01), z["sens"], label=k)
    ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_xlabel("false positives / scan", color="#ccc"); ax.set_ylabel("sensitivity", color="#ccc")
    ax.tick_params(colors="#ccc"); ax.legend(fontsize=8); ax.grid(alpha=.2); return b64(fig)

def loss_img():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), facecolor="#111")
    for k, v in runs.items():
        h = pd.DataFrame([json.loads(l) for l in (ROOT / v[1] / "metrics.jsonl").read_text().splitlines()])
        ax[0].plot(h.epoch, h.val_loss, label=k); ax[1].plot(h.epoch, h.val_patch_dice, label=k)
    for a, t in zip(ax, ["validation loss", "validation patch Dice (patch-level, not full scan)"]):
        a.set_facecolor("#111"); a.set_title(t, color="#ccc", fontsize=9); a.tick_params(colors="#ccc"); a.grid(alpha=.2)
    ax[0].legend(fontsize=7); return b64(fig)

def overlays(folder, n=6):
    seg = pd.read_csv(ROOT / folder / "per_scan_segmentation.csv").sort_values("dice"); out = []
    pick = pd.concat([seg.head(2), seg.iloc[len(seg)//2-1:len(seg)//2+1], seg.tail(2)]).drop_duplicates("case_id")
    for r in pick.itertuples():
        img = np.load(PROC / "volumes" / f"{r.case_id}.img.npy", mmap_mode="r"); gt = np.load(PROC / "volumes" / f"{r.case_id}.mask.npy")
        prob = np.load(ROOT / folder / f"{r.case_id}.prob.npy").astype(np.float32)
        z = int(np.argmax(gt.sum((1, 2)))) if gt.any() else img.shape[0] // 2
        fig, ax = plt.subplots(1, 2, figsize=(7, 3.6), facecolor="#111")
        for a, ttl in zip(ax, ["reference (green)", "model (red, p≥0.5)"]):
            a.imshow(window(np.asarray(img[z])), cmap="gray"); a.axis("off"); a.set_title(ttl, color="#ccc", fontsize=8)
        ax[0].contour(gt[z] > 0, levels=[.5], colors="lime", linewidths=1); ax[1].contour(prob[z] >= .5, levels=[.5], colors="red", linewidths=1)
        fig.suptitle(f"{r.case_id.split('_')[0]}  full-scan Dice {r.dice:.2f}  (axial z={z})", color="#ccc", fontsize=9)
        out.append(b64(fig))
    return out

rows = ""
for k, m in M.items():
    s, d = m["segmentation_full_volume"], m["detection"]; at = d["sensitivity_at_fp_per_scan"]
    rows += f"<tr><td>{k}</td><td>{f(s['mean_dice_all_scans'],2)}</td><td>{f(s['micro_dice'],2)}</td><td>{f(s['mean_surface_dice_1mm'],2)}</td><td>{f(d['lesion_sensitivity_at_threshold'],2)}</td><td>{f(d['fp_per_scan_at_threshold'],1)}</td>" + "".join(f"<td>{f(at[str(r)],2)}</td>" for r in ["0.125", "0.25", "0.5", "1", "2", "4", "8"]) + "</tr>"
last = list(M)[-1]; sub = M[last]["subgroups"]
subrows = "".join(f"<tr><td>{x['size_bin']} mm</td><td>{x['count']}</td><td>{x['mean']:.2f}</td></tr>" for x in sub["by_size_bin"])
subrows += "".join(f"<tr><td>slice {x['th']} mm</td><td>{x['count']}</td><td>{x['mean']:.2f}</td></tr>" for x in sub["by_slice_thickness"])
imgs = "".join(f'<img src="data:image/png;base64,{b}">' for b in overlays(runs[last][0])) if M else ""
n = M[last]
html = f"""<!doctype html><meta charset=utf-8><title>LungVision AI report</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>body{{font:15px/1.5 system-ui;background:#111;color:#ddd;max-width:980px;margin:auto;padding:16px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #333;padding:4px 6px;text-align:right}}td:first-child,th:first-child{{text-align:left}}
.warn{{background:#5a1a1a;border:1px solid #a33;padding:10px;border-radius:6px}}img{{max-width:100%;margin:6px 0}}h2{{border-bottom:1px solid #333;padding-bottom:4px;margin-top:28px}}code{{background:#222;padding:1px 4px}}</style>
<h1>LungVision AI — pilot report</h1>
<p class=warn><b>Research use only.</b> Not validated or authorized for clinical diagnosis, screening, treatment decisions, or patient management. Pilot-scale results; not representative of a fully trained model.</p>
<h2>Setup</h2><p>Custom 3D residual U-Net (11.3 M parameters), trained from random initialisation on LIDC-IDRI (CC BY 3.0) at 1 mm isotropic, 64³ patches, Dice+BCE loss, Apple M5 Pro / MPS. Test set: <b>{n['n_scans']} scans / {n['n_patients']} patients / {n['n_reference_nodules']} reference nodules</b> from the patient-level held-out test split (split id <code>{n.get('checkpoint_split_id')}</code>). This is a small subset of the 153-scan test split — wide uncertainty.</p>
<h2>Test results (full-volume)</h2>
<table><tr><th>run</th><th>mean Dice/scan</th><th>pooled Dice</th><th>surf Dice 1mm</th><th>lesion sens @0.5</th><th>FP/scan @0.5</th><th>sens@0.125</th><th>@0.25</th><th>@0.5</th><th>@1</th><th>@2</th><th>@4</th><th>@8 FP/scan</th></tr>{rows}</table>
<p>Sensitivity columns are our FROC (centroid within reference radius). Not the official LUNA16 evaluation; no CPM is claimed. Reference = default ground truth (nodules with ≥3 radiologist annotations); unannotated ≠ nodule-free, so some "false positives" may be real nodules outside that policy.</p>
<h2>FROC and training curves</h2><img src="data:image/png;base64,{curve_img()}"><img src="data:image/png;base64,{loss_img()}">
<h2>Detection by subgroup ({last})</h2><table><tr><th>group</th><th>n nodules</th><th>sensitivity @0.5</th></tr>{subrows}</table>
<h2>Example overlays ({last}) — worst, median and best scans by Dice</h2>{imgs}
<h2>Known failure modes</h2><ul><li>Many false-positive candidates per scan (vessels, pleural/ chest-wall structures) — no false-positive-reduction stage yet.</li><li>Some scans have Dice 0 / missed nodules (see per_reference_nodule.csv).</li><li>Mean HD95 is dominated by distant false positives.</li><li>Single retrospective collection; no external validation. LUNA16 ⊂ LIDC-IDRI, so not independent.</li><li>Malignancy is <b>not</b> predicted: a confirmed-malignancy model is blocked (see reports/label_provenance.json).</li></ul>
<h2>Try a new scan</h2><p>Run <code>.venv/bin/uvicorn app.api:app --host 127.0.0.1 --port 8000</code> then open <a href="http://127.0.0.1:8000/ui" style="color:#8cf">http://127.0.0.1:8000/ui</a>, or <code>.venv/bin/streamlit run app/streamlit_app.py</code>. Scans stay on this Mac.</p>"""
(ROOT / "reports/lungvision_report.html").write_text(html); print("wrote reports/lungvision_report.html", len(html) // 1024, "KB")
