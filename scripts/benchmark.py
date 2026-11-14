"""Apple-Silicon benchmark: CPU vs MPS, patch sizes, base channels, batch size; peak memory; loader throughput."""
import csv, json, time, psutil, torch, numpy as np, pandas as pd
from models.unet3d import Custom3DUNet
from training.losses import DiceBCELoss
from lvdata.dataset_loader import PatchDataset

pc = pd.read_csv("data/manifests/processed_cases.csv"); cases = pc[pc.split == "train"].case_id.tolist()[:4] or pc.case_id.tolist()[:4]
rows, mem = [], {}
def run(dev, patch, ch, bs, steps=4):
    torch.manual_seed(0); m = Custom3DUNet(base_channels=ch).to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-4); lf = DiceBCELoss()
    x = torch.rand(bs, 1, *patch, device=dev); y = (torch.rand(bs, 1, *patch, device=dev) > .98).float()
    def step():
        opt.zero_grad(); l = lf(m(x), y); l.backward(); opt.step()
    step(); (torch.mps.synchronize() if dev == "mps" else None)
    if dev == "mps": torch.mps.empty_cache(); base = torch.mps.driver_allocated_memory()
    t = time.time()
    for _ in range(steps): step()
    if dev == "mps": torch.mps.synchronize()
    dt = (time.time() - t) / steps; peak = torch.mps.driver_allocated_memory() / 2**30 if dev == "mps" else float("nan")
    return {"device": dev, "patch": "x".join(map(str, patch)), "base_channels": ch, "batch": bs, "step_s": round(dt, 3),
            "patches_per_s": round(bs / dt, 2), "voxels_per_s": round(bs * np.prod(patch) / dt), "mps_driver_alloc_GB": round(peak, 2),
            "sys_mem_used_GB": round(psutil.virtual_memory().used / 2**30, 1), "swap_used_GB": round(psutil.swap_memory().used / 2**30, 2)}
cfgs = [("cpu", (32,)*3, 16, 1), ("mps", (32,)*3, 16, 1), ("cpu", (64,)*3, 16, 1), ("mps", (64,)*3, 16, 1), ("mps", (64,)*3, 16, 2),
        ("mps", (64,)*3, 8, 1), ("mps", (64,)*3, 24, 1), ("mps", (64,)*3, 32, 1), ("mps", (96,)*3, 16, 1)]
for c in cfgs:
    try: r = run(*c, steps=2 if c[0] == "cpu" else 5)
    except Exception as e: r = {"device": c[0], "patch": "x".join(map(str, c[1])), "base_channels": c[2], "batch": c[3], "error": str(e)[:100]}
    print(r, flush=True); rows.append(r)
pd.DataFrame(rows).to_csv("reports/apple_silicon_benchmark.csv", index=False)
ds = PatchDataset(cases, (64,)*3, 40, train=True, seed=0); t = time.time(); [ds[i] for i in range(40)]; dt = time.time() - t
mem = {"loader_patches_per_s_train_aug": round(40 / dt, 2), "unified_memory_total_GB": round(psutil.virtual_memory().total / 2**30, 1),
       "recommended_working_set_GB_mps": round(torch.mps.recommended_max_memory() / 2**30, 2) if hasattr(torch.mps, "recommended_max_memory") else None,
       "note": "tokens/s is N/A for this imaging model; patches/s and voxels/s reported instead"}
json.dump({"loader": mem, "runs": rows}, open("reports/memory_profile.json", "w"), indent=2); print(mem)
