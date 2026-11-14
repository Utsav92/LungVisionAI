"""LungVision AI Research Studio. Launch: streamlit run app/streamlit_app.py  (binds to localhost by default)."""
from __future__ import annotations
import json, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd, psutil, streamlit as st, SimpleITK as sitk, plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = "Research use only. This model is not validated or authorized for clinical diagnosis, screening, treatment decisions, or patient management."
st.set_page_config(page_title="LungVision AI Research Studio", layout="wide")
st.error(DISCLAIMER, icon="⚠️")   # persistent on every view
st.title("LungVision AI Research Studio")
view = st.sidebar.radio("View", ["Dashboard", "CT Scan Viewer", "Nodule Segmentation", "Model Training", "Model Evaluation", "Research Limitations"])
EXP = st.sidebar.text_input("Experiment", "pilot_unet16")


def jl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()] if Path(p).exists() else []


def load_upload(up):
    d = tempfile.mkdtemp(prefix="lvui_"); p = Path(d) / ("s.nii.gz" if up.name.endswith(".gz") else "s.nii"); p.write_bytes(up.getvalue())
    img = sitk.ReadImage(str(p)); return sitk.DICOMOrient(img, "LPS"), p


def plane_slider(vol, key):
    plane = st.radio("Plane", ["axial", "coronal", "sagittal"], horizontal=True, key=key + "p")
    n = {"axial": vol.shape[0], "coronal": vol.shape[1], "sagittal": vol.shape[2]}[plane]
    return plane, st.slider("Slice", 0, n - 1, n // 2, key=key + "s")


if view == "Dashboard":
    man = ROOT / "data/manifests"; pc = pd.read_csv(man / "processed_cases.csv") if (man / "processed_cases.csv").exists() else pd.DataFrame()
    aud = ROOT / "reports/dataset_audit.json"; hist = jl(ROOT / "experiments" / EXP / "metrics.jsonl")
    ev = ROOT / "reports/eval_test/metrics.json"
    c = st.columns(4); c[0].metric("Scans processed (local)", len(pc)); c[1].metric("Patients", pc.patient_id.nunique() if len(pc) else 0)
    c[2].metric("Epochs trained", len(hist)); c[3].metric("Dataset audit", "available" if aud.exists() else "not run")
    if (man / "split_id.json").exists(): st.json(json.loads((man / "split_id.json").read_text()))
    ck = ROOT / "experiments" / EXP / "best.pt"; st.write("Model version:", f"{EXP}/best.pt" if ck.exists() else "no trained checkpoint yet")
    if ev.exists():
        m = json.loads(ev.read_text()); st.subheader("Latest test evaluation"); st.json(m["segmentation_full_volume"]); st.json(m["detection"]["sensitivity_at_fp_per_scan"])

elif view == "CT Scan Viewer":
    from visualization.ct_viewer import slice_figure
    up = st.file_uploader("De-identified NIfTI CT (.nii/.nii.gz). Processed locally; never uploaded elsewhere.", type=["nii", "gz"])
    if up:
        img, _ = load_upload(up); vol = sitk.GetArrayFromImage(img); sp = img.GetSpacing()[::-1]
        st.json({"size_xyz": img.GetSize(), "spacing_xyz_mm": img.GetSpacing(), "origin": img.GetOrigin(), "direction": img.GetDirection(), "HU range": [float(vol.min()), float(vol.max())]})
        if vol.min() > -200: st.warning("Intensity range does not look like HU chest CT.")
        lvl, wid = st.slider("Window level", -1000, 400, -600), st.slider("Window width", 100, 2500, 1500)
        cols = st.columns(3)
        for col, pl in zip(cols, ["axial", "coronal", "sagittal"]):
            n = {"axial": vol.shape[0], "coronal": vol.shape[1], "sagittal": vol.shape[2]}[pl]; i = col.slider(pl, 0, n - 1, n // 2, key=pl)
            col.plotly_chart(slice_figure(vol, pl, i, sp, lvl, wid), use_container_width=True)

elif view == "Nodule Segmentation":
    ck = ROOT / "experiments" / EXP / "best.pt"
    if not ck.exists(): st.warning("No trained checkpoint yet. Train with `python train.py --config configs/train_mac.yaml`.")
    up = st.file_uploader("De-identified NIfTI CT", type=["nii", "gz"], key="seg")
    if up and ck.exists() and st.button("Run segmentation"):
        import torch
        from lvdata.preprocess_ct import resample
        from inference.segmentation import load_model, segment_volume
        from inference.nodule_detection import detect_nodules
        from visualization.ct_viewer import slice_figure
        img, _ = load_upload(up); r = resample(img, (1, 1, 1), False); vol = np.clip(sitk.GetArrayFromImage(r), -1024, 3071).astype(np.int16)
        dev = "mps" if torch.backends.mps.is_available() else "cpu"; model, c = load_model(ck, dev); bar = st.progress(0.0)
        prob, stats = segment_volume(model, c, vol, dev, progress=lambda i, n: bar.progress(i / max(n, 1)))
        meta = {"resampled": {"size": list(r.GetSize()), "spacing": list(r.GetSpacing()), "origin": list(r.GetOrigin()), "direction": list(r.GetDirection())},
                "original": {"size": list(img.GetSize()), "spacing": list(img.GetSpacing()), "origin": list(img.GetOrigin()), "direction": list(img.GetDirection())}}
        cands = detect_nodules(prob, (1, 1, 1), 0.5, 3.0, 4.0, meta); st.session_state.update(vol=vol, prob=prob, cands=[c.to_dict() for c in cands])
    if "prob" in st.session_state:
        from visualization.ct_viewer import slice_figure
        vol, prob, cands = st.session_state["vol"], st.session_state["prob"], st.session_state["cands"]
        st.subheader(f"{len(cands)} model-detected nodule candidate(s) — research output, not a diagnosis")
        st.dataframe(pd.DataFrame([{"centroid_zyx(1mm)": tuple(round(x) for x in c["centroid_zyx"]), "eq_diameter_mm": round(c["diameter_mm"], 1), "max_extent_mm": round(c["max_extent_mm"], 1),
                                    "volume_mm3": round(c["volume_mm3"]), "model_confidence": round(c["confidence"], 3), "original_xyz": tuple(round(x, 1) for x in c["centroid_original_xyz"] or [])} for c in cands]))
        pl, i = plane_slider(vol, "seg"); thr = st.slider("Probability threshold", 0.1, 0.9, 0.5)
        st.plotly_chart(slice_figure(vol, pl, i, (1, 1, 1), pred=prob >= thr, candidates=cands), use_container_width=True)

elif view == "Model Training":
    h = jl(ROOT / "experiments" / EXP / "metrics.jsonl")
    if not h: st.info("No training log yet.")
    else:
        d = pd.DataFrame(h); st.line_chart(d.set_index("epoch")[["train_loss", "val_loss"]]); st.line_chart(d.set_index("epoch")[["val_patch_dice"]])
        st.line_chart(d.set_index("epoch")[["lr"]]); st.line_chart(d.set_index("epoch")[["mem_used_gb", "swap_used_gb"]]); st.metric("Current epoch", int(d.epoch.max()))
    vm = psutil.virtual_memory(); st.metric("System memory used (now)", f"{vm.used/1e9:.1f} / {vm.total/1e9:.0f} GB")

elif view == "Model Evaluation":
    name = st.text_input("Evaluation folder", "eval_test"); f = ROOT / "reports" / name / "metrics.json"
    if not f.exists(): st.info("Run: python -m evaluation.evaluate --ckpt experiments/<exp>/best.pt --split test --out reports/eval_test")
    else:
        m = json.loads(f.read_text()); st.json({k: m[k] for k in ["split", "n_scans", "n_patients", "n_reference_nodules"]}); st.json(m["segmentation_full_volume"]); st.json(m["detection"])
        z = np.load(ROOT / "reports" / name / "froc_curve.npz"); fig = go.Figure(go.Scatter(x=z["fps"], y=z["sens"], mode="lines")); fig.update_xaxes(type="log", title="FP / scan"); fig.update_yaxes(title="Sensitivity", range=[0, 1]); st.plotly_chart(fig)
        st.subheader("Subgroups"); st.json(m["subgroups"]); st.write("Per-reference-nodule results (false negatives = detected False):")
        st.dataframe(pd.read_csv(ROOT / "reports" / name / "per_reference_nodule.csv")); st.write("Candidates (matched_ref = -1 are false positives):"); st.dataframe(pd.read_csv(ROOT / "reports" / name / "candidates.csv"))

else:
    st.markdown("""### Research limitations
- Trained from scratch on a **pilot subset** of LIDC-IDRI; metrics are not representative of a fully trained model.
- LIDC-IDRI is a single retrospective collection; no external validation. LUNA16 is a subset of it, not independent.
- Ground truth = consensus of ≥3 radiologist outlines (≥50% voxel agreement). 1–2-reader nodules and nodules <3 mm are background; "false positives" may include real unannotated nodules.
- Radiologist malignancy scores are subjective suspicion ratings, **not** pathology. No malignancy classifier is provided.
- Never interpret outputs as 'cancer detected' or 'cancer-free'.""")
