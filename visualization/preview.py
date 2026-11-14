"""Server-side preview rendering for the local UI: axial slices with mask tint + candidate boxes, and 3-plane crops."""
from __future__ import annotations
import base64, io
import numpy as np
from PIL import Image, ImageDraw
from .overlays import window


def _jpeg(img: Image.Image, q=80):
    b = io.BytesIO(); img.save(b, "JPEG", quality=q); return base64.b64encode(b.getvalue()).decode()


def _compose(gray, mask, boxes, size, flip_v=False, labels=None):
    """gray/mask 2D (rows, cols) at 1 mm; boxes [(r0,r1,c0,c1,label)] in the same index space."""
    rgb = np.stack([gray] * 3, -1)
    if mask is not None and mask.any():
        rgb[mask] = 0.55 * rgb[mask] + 0.45 * np.array([1.0, 0.1, 0.1])
    if flip_v:
        rgb = rgb[::-1]; boxes = [(gray.shape[0] - r1, gray.shape[0] - r0, c0, c1, l) for r0, r1, c0, c1, l in boxes]
    im = Image.fromarray((rgb * 255).astype(np.uint8)); s = size / max(im.size)
    im = im.resize((max(int(im.width * s), 1), max(int(im.height * s), 1)), Image.BILINEAR); d = ImageDraw.Draw(im)
    for r0, r1, c0, c1, l in boxes:
        pad = 3
        d.rectangle([c0 * s - pad, r0 * s - pad, c1 * s + pad, r1 * s + pad], outline=(255, 220, 0), width=2)
        d.text((c1 * s + pad + 2, r0 * s - pad), str(l), fill=(255, 220, 0))
    return im


def axial_stack(vol, prob, cands, thr=0.5, size=420, stride=2):
    """Returns list of (z, jpeg_b64). Candidate boxes drawn on slices they span."""
    out = []
    for z in range(0, vol.shape[0], stride):
        boxes = [(c["bbox_zyx"][1][0], c["bbox_zyx"][1][1], c["bbox_zyx"][2][0], c["bbox_zyx"][2][1], i + 1)
                 for i, c in enumerate(cands) if c["bbox_zyx"][0][0] <= z < c["bbox_zyx"][0][1]]
        out.append((z, _jpeg(_compose(window(vol[z]), prob[z] >= thr, boxes, size))))
    return out


def candidate_crops(vol, prob, c, idx, half=32, size=210, thr=0.5):
    cz, cy, cx = [int(round(v)) for v in c["centroid_zyx"]]; (z0, z1), (y0, y1), (x0, x1) = c["bbox_zyx"]
    def sl(center, n): lo = max(min(center - half, n - 2 * half), 0); return lo, min(lo + 2 * half, n)
    zl, zh = sl(cz, vol.shape[0]); yl, yh = sl(cy, vol.shape[1]); xl, xh = sl(cx, vol.shape[2])
    ax = _compose(window(vol[cz, yl:yh, xl:xh]), prob[cz, yl:yh, xl:xh] >= thr, [(y0 - yl, y1 - yl, x0 - xl, x1 - xl, idx)], size)
    co = _compose(window(vol[zl:zh, cy, xl:xh]), prob[zl:zh, cy, xl:xh] >= thr, [(z0 - zl, z1 - zl, x0 - xl, x1 - xl, idx)], size, flip_v=True)
    sa = _compose(window(vol[zl:zh, yl:yh, cx]), prob[zl:zh, yl:yh, cx] >= thr, [(z0 - zl, z1 - zl, y0 - yl, y1 - yl, idx)], size, flip_v=True)
    return {"axial": _jpeg(ax), "coronal": _jpeg(co), "sagittal": _jpeg(sa), "z": cz}
