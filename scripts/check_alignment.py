"""Real-data sanity: inside GT nodules HU should be soft-tissue-like (>> lung); report per scan."""
import numpy as np, pandas as pd
from pathlib import Path
from lvdata.preprocess_ct import PROC
cs = pd.read_csv("data/manifests/processed_cases.csv").case_id
rows = []
for c in cs:
    i = np.load(PROC/"volumes"/f"{c}.img.npy", mmap_mode="r"); m = np.load(PROC/"volumes"/f"{c}.mask.npy", mmap_mode="r")
    assert i.shape == m.shape
    inside = np.asarray(i)[np.asarray(m) > 0]
    # shifted-mask control: if alignment is right, true mask HU >> mask shifted by 20 voxels
    sh = np.asarray(i)[np.roll(np.asarray(m), 20, axis=1) > 0]
    rows.append((c, inside.mean(), np.median(inside), sh.mean()))
d = pd.DataFrame(rows, columns=["case", "mean_HU_in_mask", "median_HU_in_mask", "mean_HU_shifted_control"])
print(d.round(0).to_string(index=False)); print("aligned (mask HU > control HU) in", (d.mean_HU_in_mask > d.mean_HU_shifted_control).sum(), "/", len(d))
