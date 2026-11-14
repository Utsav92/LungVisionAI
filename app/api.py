"""Local-only research API. Run: uvicorn app.api:app --host 127.0.0.1 --port 8000
Research use only. No scan leaves this machine; uploads are held in a temp dir and deleted after each request."""
from __future__ import annotations
import json, os, tempfile, time, asyncio
from pathlib import Path
import numpy as np, SimpleITK as sitk
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from lvdata.preprocess_ct import resample, voxel_to_physical
from inference.segmentation import load_model, segment_volume
from inference.nodule_detection import detect_nodules

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = int(os.environ.get("LV_MAX_UPLOAD_MB", 600)) * 2**20
TIMEOUT_S = int(os.environ.get("LV_TIMEOUT_S", 900))
CKPT = Path(os.environ.get("LV_CKPT", ROOT / "experiments/pilot_unet16/best.pt"))
DISCLAIMER = "Research use only. Not validated or authorized for clinical diagnosis, screening, treatment decisions, or patient management."
app = FastAPI(title="LungVision AI (research)", version="0.1.0"); _M = {}


def model():
    if "m" not in _M:
        if not CKPT.exists(): raise HTTPException(503, f"no trained checkpoint at {CKPT}")
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"; m, ck = load_model(CKPT, dev); _M.update(m=m, ck=ck, dev=dev)
    return _M["m"], _M["ck"], _M["dev"]


async def _read(file: UploadFile):
    name = (file.filename or "").lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")): raise HTTPException(415, "only .nii / .nii.gz supported")
    d = tempfile.mkdtemp(prefix="lv_"); p = Path(d) / ("scan.nii.gz" if name.endswith(".gz") else "scan.nii"); n = 0
    with open(p, "wb") as f:
        while chunk := await file.read(1 << 20):
            n += len(chunk)
            if n > MAX_BYTES: f.close(); _cleanup(p); raise HTTPException(413, "file too large")
            f.write(chunk)
    return p


def _cleanup(p):
    import shutil; shutil.rmtree(p.parent, ignore_errors=True)


def _load_validate(p):
    try: img = sitk.ReadImage(str(p))
    except Exception: raise HTTPException(422, "unreadable NIfTI")
    if img.GetDimension() != 3 or min(img.GetSize()) < 64: raise HTTPException(422, "not a 3D CT volume (need 3D, >=64 voxels per axis)")
    sp = img.GetSpacing()
    if not all(0.2 < s < 10 for s in sp): raise HTTPException(422, f"implausible voxel spacing {sp}")
    a = sitk.GetArrayViewFromImage(img)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 500 or lo > -200: raise HTTPException(422, f"intensity range [{lo:.0f},{hi:.0f}] is not Hounsfield-unit chest CT")
    return sitk.DICOMOrient(img, "LPS"), {"size": list(img.GetSize()), "spacing": list(sp), "hu_range": [lo, hi]}


@app.get("/health")
def health(): return {"status": "ok", "research_only": True, "disclaimer": DISCLAIMER}


@app.get("/model/info")
def model_info():
    if not CKPT.exists(): return {"checkpoint": None, "disclaimer": DISCLAIMER}
    from training.checkpoint import load_checkpoint
    s = load_checkpoint(CKPT)
    return {"checkpoint": CKPT.name, "epoch": s["epoch"], "best_val_patch_dice": s.get("best_metric"), "arch": s["arch"], "preprocessing": s["preprocessing"],
            "split_id": s.get("split_id"), "pretrained": False, "disclaimer": DISCLAIMER}


@app.post("/scan/validate")
async def scan_validate(file: UploadFile = File(...)):
    p = await _read(file)
    try: _, info = _load_validate(p); return {"valid": True, **info, "disclaimer": DISCLAIMER}
    finally: _cleanup(p)


@app.post("/scan/preprocess")
async def scan_preprocess(file: UploadFile = File(...)):
    p = await _read(file)
    try:
        img, info = _load_validate(p); r = resample(img, (1, 1, 1), False); return {**info, "resampled_size": list(r.GetSize()), "target_spacing": [1, 1, 1], "disclaimer": DISCLAIMER}
    finally: _cleanup(p)


def _run(p, with_det):
    img, info = _load_validate(p); r = resample(img, (1, 1, 1), False); vol = np.clip(sitk.GetArrayFromImage(r), -1024, 3071).astype(np.int16)
    m, ck, dev = model(); t = time.time(); prob, st = segment_volume(m, ck, vol, dev)
    meta = {"resampled": {"size": list(r.GetSize()), "spacing": list(r.GetSpacing()), "origin": list(r.GetOrigin()), "direction": list(r.GetDirection())},
            "original": {"size": list(img.GetSize()), "spacing": list(img.GetSpacing()), "origin": list(img.GetOrigin()), "direction": list(img.GetDirection())}}
    out = {**info, "segmented_voxels_mm3": float((prob >= .5).sum()), "inference_seconds": round(time.time() - t, 1), "windows": st, "disclaimer": DISCLAIMER}
    if with_det:
        out["candidates"] = [c.to_dict() for c in detect_nodules(prob, (1, 1, 1), 0.5, 3.0, 4.0, meta)]
        from visualization.preview import axial_stack, candidate_crops
        cl = out["candidates"]; out["slices"] = [{"z": z, "img": im} for z, im in axial_stack(vol, prob, cl)]
        out["crops"] = [candidate_crops(vol, prob, c, i + 1) for i, c in enumerate(cl[:12])]
        out["n_slices"] = int(vol.shape[0])
        out["note"] = "Model-detected nodule candidates (suspected pulmonary nodules). Not a diagnosis; not malignancy assessments."
    return out


async def _guarded(file, with_det):
    p = await _read(file)
    try: return await asyncio.wait_for(asyncio.to_thread(_run, p, with_det), TIMEOUT_S)
    except asyncio.TimeoutError: raise HTTPException(504, "processing timeout")
    finally: _cleanup(p)


@app.post("/scan/segment")
async def scan_segment(file: UploadFile = File(...)): return await _guarded(file, False)


@app.post("/scan/detect")
async def scan_detect(file: UploadFile = File(...)): return await _guarded(file, True)


@app.get("/training/status")
def training_status(experiment: str = "pilot_unet16"):
    f = ROOT / "experiments" / experiment / "metrics.jsonl"
    if not f.exists(): return {"experiment": experiment, "epochs_done": 0}
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return {"experiment": experiment, "epochs_done": len(rows), "last": rows[-1]}


@app.get("/evaluation/metrics")
def evaluation_metrics(name: str = "eval_test"):
    f = ROOT / "reports" / name / "metrics.json"
    if not f.exists(): raise HTTPException(404, "no evaluation yet; run python -m evaluation.evaluate")
    return json.loads(f.read_text())
# NOTE: /scan/malignancy-research intentionally NOT implemented: no validated research classifier exists (see reports/label_provenance.json).


UI = """<!doctype html><meta charset=utf-8><title>LungVision AI - try a scan</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>body{font:15px/1.5 system-ui;background:#111;color:#ddd;max-width:1000px;margin:auto;padding:16px}.w{background:#5a1a1a;border:1px solid #a33;padding:10px;border-radius:6px}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #333;padding:4px 8px;text-align:right}tr.c{cursor:pointer}tr.c:hover{background:#222}button{padding:8px 14px}
#v{display:flex;gap:16px;flex-wrap:wrap}#main{background:#000}img{max-width:100%}.crops{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}.crops div{font-size:12px;text-align:center}.crops img{width:150px}.leg span{margin-right:14px}</style>
<h1>LungVision AI - try a scan</h1>
<p class=w><b>Research use only.</b> Not validated or authorized for clinical diagnosis, screening, treatment decisions, or patient management. Use de-identified NIfTI only. Processing is local; nothing is uploaded elsewhere.</p>
<input type=file id=f accept=".nii,.gz"> <button onclick=run()>Detect nodule candidates</button> <span id=s></span><div id=o></div>
<script>let S=[];
function show(z){let k=0,b=1e9;S.forEach((s,i)=>{const d=Math.abs(s.z-z);if(d<b){b=d;k=i}});document.getElementById('sl').value=k;draw(k)}
function draw(k){document.getElementById('main').src='data:image/jpeg;base64,'+S[k].img;document.getElementById('zl').textContent='axial slice '+S[k].z}
async function run(){const f=document.getElementById('f').files[0];if(!f)return;const s=document.getElementById('s'),o=document.getElementById('o');o.innerHTML='';s.textContent='running (about 30-60 s)...';
const fd=new FormData();fd.append('file',f);const r=await fetch('/scan/detect',{method:'POST',body:fd});const j=await r.json();if(!r.ok){s.textContent='error: '+(j.detail||r.status);return}
s.textContent='done in '+j.inference_seconds+' s';const C=j.candidates.map((c,i)=>({...c,i:i+1,crop:j.crops[i]})).sort((a,b)=>b.confidence-a.confidence);S=j.slices;
let h='<h2>'+C.length+' model-detected nodule candidate(s)</h2><p>'+(C.length?'Yellow boxes mark candidates, red tint is the model segmentation. These are <b>candidates, not diagnoses</b>; the model cannot tell benign from malignant, and many candidates are false positives.':'No candidates found. This does not mean the scan is free of nodules.')+'</p>';
h+='<div id=v><div><img id=main><br><input type=range id=sl min=0 max='+(S.length-1)+' value=0 style="width:420px" oninput="draw(+this.value)"> <span id=zl></span></div><div style="flex:1;min-width:340px"><table><tr><th>#</th><th>diam mm</th><th>conf</th><th>z,y,x</th></tr>';
C.forEach(c=>{h+='<tr class=c onclick="show('+Math.round(c.centroid_zyx[0])+')"><td>'+c.i+'</td><td>'+c.diameter_mm.toFixed(1)+'</td><td>'+c.confidence.toFixed(2)+'</td><td>'+c.centroid_zyx.map(x=>Math.round(x)).join(',')+'</td></tr>'});
h+='</table><p style="font-size:12px">Click a row to jump to that slice. Coordinates are on the 1 mm resampled grid.</p></div></div><h2>Candidate close-ups (axial / coronal / sagittal)</h2>';
C.slice(0,12).forEach(c=>{h+='<b>#'+c.i+'</b> d='+c.diameter_mm.toFixed(1)+' mm, conf '+c.confidence.toFixed(2)+'<div class=crops><div><img src="data:image/jpeg;base64,'+c.crop.axial+'">axial</div><div><img src="data:image/jpeg;base64,'+c.crop.coronal+'">coronal</div><div><img src="data:image/jpeg;base64,'+c.crop.sagittal+'">sagittal</div></div>'});
o.innerHTML=h;show(C.length?Math.round(C[0].centroid_zyx[0]):S[Math.floor(S.length/2)].z)}</script>"""


@app.get("/ui", response_class=HTMLResponse)
def ui(): return UI
