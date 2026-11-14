"""python train.py --config configs/train_mac.yaml [--resume] [--max-minutes N] [--set key=value ...]"""
import argparse, copy, hashlib, json, time
from pathlib import Path
import pandas as pd, yaml, torch
from models.unet3d import Custom3DUNet, UNet3DConfig
from lvdata.dataset_loader import PatchDataset
from training.trainer import Trainer

ROOT = Path(__file__).parent


def load_cfg(path):
    c = yaml.safe_load(open(path)); out = {}
    for b in c.pop("base", []): out.update(load_cfg(ROOT / b) if (ROOT / b).exists() else {})
    out.update(c); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float); ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(); cfg = load_cfg(a.config)
    for kv in a.set:
        k, v = kv.split("="); cfg[k] = yaml.safe_load(v)
    name = cfg.get("experiment", "exp"); out = ROOT / "experiments" / name; out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    pc = pd.read_csv(ROOT / "data/manifests/processed_cases.csv")
    split_bytes = (ROOT / "data/manifests/split_manifest.csv").read_bytes(); split_id = hashlib.sha256(split_bytes).hexdigest()[:16]
    tr, va = pc[pc.split == "train"].case_id.tolist(), pc[pc.split == "val"].case_id.tolist()
    assert tr, "no processed training scans"; print(f"train scans {len(tr)}  val scans {len(va)}  split_id {split_id}")
    patch = tuple(cfg["patch_size"]); pre = cfg["preprocessing"]
    tds = PatchDataset(tr, patch, cfg["steps_per_epoch"], cfg["pos_fraction"], cfg["hard_neg_fraction"], True, cfg["seed"], pre)
    vds = PatchDataset(va, patch, cfg["val_steps"], cfg["pos_fraction"], cfg["hard_neg_fraction"], False, cfg["seed"] + 1, pre) if va else None
    arch = UNet3DConfig(base_channels=cfg["base_channels"], norm=cfg.get("norm", "group"), act=cfg.get("act", "silu"))
    model = Custom3DUNet(arch); s = model.summary(patch)
    print(f"params {s['total_params']:,} (trainable {s['trainable_params']:,}) input {s['input_shape']} weights {s['approx_weights_MB']} MB")
    for n, sh in s["feature_maps"]: print(f"  {n:11s} {sh}")
    extra = {"arch": s["config"], "preprocessing": pre, "patch": list(patch), "split_id": split_id, "train_cfg": cfg, "seed": cfg["seed"]}
    tr_ = Trainer(model, cfg, tds, vds, out, extra)
    if a.resume and (out / "last.pt").exists(): print("resuming from epoch", tr_.resume(out / "last.pt")["epoch"] + 1)
    tr_.fit(a.max_minutes)


if __name__ == "__main__":
    main()
