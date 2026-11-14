import io, numpy as np, SimpleITK as sitk
from fastapi.testclient import TestClient
from app.api import app
from tests.synthetic import make_volume

c = TestClient(app)


def _nii(arr, spacing=(1, 1, 1), tmp=None):
    img = sitk.GetImageFromArray(arr); img.SetSpacing(spacing)
    p = tmp / "x.nii.gz"; sitk.WriteImage(img, str(p)); return p.read_bytes()


def test_health_and_disclaimer():
    r = c.get("/health").json(); assert r["status"] == "ok" and "Research use only" in r["disclaimer"]


def test_validate_accepts_ct_like(tmp_path):
    v, _, _ = make_volume((80, 80, 80), 1, (5, 8), 0); v = np.where(v < -900, -1000, v).astype(np.int16); v[:5] = 600
    r = c.post("/scan/validate", files={"file": ("a.nii.gz", _nii(v, tmp=tmp_path))}); assert r.status_code == 200 and r.json()["valid"]


def test_rejects_non_ct_and_bad_ext(tmp_path):
    assert c.post("/scan/validate", files={"file": ("a.png", b"x")}).status_code == 415
    flat = (np.random.rand(80, 80, 80) * 255).astype(np.int16)           # 0..255 image, not HU
    assert c.post("/scan/validate", files={"file": ("a.nii.gz", _nii(flat, tmp=tmp_path))}).status_code == 422
    assert c.post("/scan/validate", files={"file": ("a.nii.gz", b"garbage")}).status_code == 422


def test_no_malignancy_endpoint():
    assert c.post("/scan/malignancy-research").status_code == 404
