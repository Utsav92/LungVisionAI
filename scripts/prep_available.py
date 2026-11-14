"""Preprocess + index every downloaded case that has both image and mask on disk."""
import sys, pandas as pd
from pathlib import Path
from lvdata.preprocess_ct import preprocess_case, RAW
from lvdata.patch_sampler import save_scan_index
sp = pd.read_csv("data/manifests/split_manifest.csv")
done = []
for c in sp.case_id:
    if (RAW/"images"/f"{c}.nii.gz").exists() and (RAW/"masks"/f"{c}.nii.gz").exists():
        m = preprocess_case(c); i = save_scan_index(c); done.append(c)
        print(c, m["resampled"]["size"], "gtvox", m.get("mask_voxels"), "pos", len(i["positives"]), "easy", i["n_easy"], "hard", i["n_hard"], flush=True)
pd.DataFrame({"case_id": done}).merge(sp, on="case_id").to_csv("data/manifests/processed_cases.csv", index=False)
print(pd.read_csv("data/manifests/processed_cases.csv").split.value_counts())
