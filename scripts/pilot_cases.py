"""Print case_ids for the pilot: n_val val scans + enough train scans to reach n_train (already-downloaded ones counted)."""
import os, sys, pandas as pd
n_train, n_val = int(sys.argv[1]), int(sys.argv[2])
m = pd.read_csv("data/manifests/scan_manifest.csv").merge(pd.read_csv("data/manifests/split_manifest.csv")[["case_id", "split"]], on="case_id")
m = m[m.nodule_count > 0].sort_values("case_id"); have = {f[:-7] for f in os.listdir("data/raw/lidc/images")}
va = m[m.split == "val"].case_id.tolist()[:n_val]; tr = m[m.split == "train"].case_id.tolist()
tr_have = [c for c in tr if c in have]; tr_new = [c for c in tr if c not in have][:max(0, n_train - len(tr_have))]
print(" ".join(va + tr_new))
