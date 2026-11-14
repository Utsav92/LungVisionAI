"""Full-volume sliding-window inference with Gaussian importance weighting. Skips air-only windows."""
from __future__ import annotations
import numpy as np, torch


def _gauss_window(size, sigma_scale=0.125):
    ax = [np.exp(-0.5 * ((np.arange(s) - (s - 1) / 2) / (s * sigma_scale * 2)) ** 2) for s in size]
    w = ax[0][:, None, None] * ax[1][None, :, None] * ax[2][None, None, :]
    return (w / w.max()).astype(np.float32) + 1e-3


def _starts(n, p, stride):
    if n <= p: return [0]
    s = list(range(0, n - p + 1, stride))
    if s[-1] != n - p: s.append(n - p)
    return s


@torch.no_grad()
def sliding_window_predict(model, vol_hu, normalize, patch=(64, 64, 64), overlap=0.5, batch_size=4, device="mps",
                           skip_below_hu=-700, progress=None):
    """vol_hu: (z,y,x) int16/float HU. Returns float32 probability volume, same shape.
    Windows whose max HU < skip_below_hu (pure air outside body) are skipped (prob 0) and counted in stats."""
    model.eval(); dev = torch.device(device)
    shape = vol_hu.shape; pad = [max(p - s, 0) for p, s in zip(patch, shape)]
    if any(pad): vol_hu = np.pad(np.asarray(vol_hu), [(0, q) for q in pad], constant_values=-1024)
    S = vol_hu.shape; stride = [max(int(p * (1 - overlap)), 1) for p in patch]
    acc = np.zeros(S, np.float32); wsum = np.zeros(S, np.float32); w = _gauss_window(patch)
    coords = [(z, y, x) for z in _starts(S[0], patch[0], stride[0]) for y in _starts(S[1], patch[1], stride[1])
              for x in _starts(S[2], patch[2], stride[2])]
    todo = [c for c in coords if vol_hu[c[0]:c[0]+patch[0], c[1]:c[1]+patch[1], c[2]:c[2]+patch[2]].max() >= skip_below_hu]
    for i in range(0, len(todo), batch_size):
        cs = todo[i:i + batch_size]
        x = np.stack([normalize(np.asarray(vol_hu[z:z+patch[0], y:y+patch[1], xx:xx+patch[2]], dtype=np.float32)) for z, y, xx in cs])
        p = torch.sigmoid(model(torch.from_numpy(x)[:, None].to(dev))).float().cpu().numpy()[:, 0]
        for (z, y, xx), pp in zip(cs, p):
            sl = (slice(z, z+patch[0]), slice(y, y+patch[1]), slice(xx, xx+patch[2]))
            acc[sl] += pp * w; wsum[sl] += w
        if progress: progress(min(i + batch_size, len(todo)), len(todo))
    prob = np.where(wsum > 0, acc / np.maximum(wsum, 1e-6), 0).astype(np.float32)
    prob = prob[:shape[0], :shape[1], :shape[2]]
    prob.flags.writeable = True
    return prob, {"windows_total": len(coords), "windows_run": len(todo), "patch": list(patch), "overlap": overlap}
