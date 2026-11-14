"""Reproducible training engine: AdamW, grad accumulation+clipping, cosine LR, early stopping, resume."""
from __future__ import annotations
import json, random, time, hashlib
from pathlib import Path
import numpy as np, psutil, torch
from torch.utils.data import DataLoader
from .checkpoint import save_checkpoint, load_checkpoint, set_rng_state
from .losses import build_loss
from .optimizer import build_optimizer
from .scheduler import build_scheduler


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def pick_device(pref="mps"):
    if pref == "mps" and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def sync(dev):
    if dev.type == "mps": torch.mps.synchronize()


@torch.no_grad()
def dice_from_logits(logits, target, thr=0.5, eps=1.0):
    p = (torch.sigmoid(logits) > thr).float()
    return float((2 * (p * target).sum() + eps) / (p.sum() + target.sum() + eps))


class Trainer:
    def __init__(self, model, cfg: dict, train_ds, val_ds, out_dir, extra_ckpt: dict | None = None):
        self.cfg, self.out = cfg, Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.dev = pick_device(cfg.get("device", "mps"))
        seed_everything(cfg.get("seed", 1337))
        self.model = model.to(self.dev)
        self.loss_fn = build_loss(cfg.get("loss", {}))
        self.opt = build_optimizer(self.model, cfg)
        self.epochs = cfg.get("epochs", 50)
        self.sched = build_scheduler(self.opt, cfg, self.epochs)
        nw = cfg.get("num_workers", 0)
        self.train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 1), shuffle=False, num_workers=nw,
                                   persistent_workers=nw > 0)
        self.val_dl = DataLoader(val_ds, batch_size=cfg.get("batch_size", 1), shuffle=False, num_workers=0) if val_ds else None
        self.train_ds, self.accum = train_ds, cfg.get("gradient_accumulation", 1)
        self.clip = cfg.get("grad_clip", 1.0)
        self.extra = extra_ckpt or {}
        self.start_epoch, self.best, self.bad, self.history = 0, -1.0, 0, []
        self.log_path = self.out / "metrics.jsonl"

    def resume(self, path):
        s = load_checkpoint(path, self.model, self.opt, self.sched, map_location=self.dev)
        self.start_epoch, self.best, self.history = s["epoch"] + 1, s.get("best_metric") or -1.0, s.get("history", [])
        set_rng_state(s["rng"])
        return s

    def _train_epoch(self, epoch):
        self.model.train()
        if hasattr(self.train_ds, "set_epoch"): self.train_ds.set_epoch(epoch)
        tot, n, t0, nvox = 0.0, 0, time.time(), 0
        self.opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(self.train_dl):
            x, y = x.to(self.dev), y.to(self.dev)
            loss = self.loss_fn(self.model(x), y)
            (loss / self.accum).backward()
            if (i + 1) % self.accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.opt.step(); self.opt.zero_grad(set_to_none=True)
            tot += float(loss.detach()); n += 1; nvox += x.numel()
        sync(self.dev)
        dt = time.time() - t0
        return tot / max(n, 1), {"patches_per_s": n / dt, "voxels_per_s": nvox / dt, "epoch_s": dt}

    @torch.no_grad()
    def _validate(self):
        self.model.eval(); tot, d, n = 0.0, 0.0, 0
        for x, y in self.val_dl:
            x, y = x.to(self.dev), y.to(self.dev); lg = self.model(x)
            tot += float(self.loss_fn(lg, y)); d += dice_from_logits(lg, y); n += 1
        return tot / max(n, 1), d / max(n, 1)

    def fit(self, max_minutes: float | None = None):
        t_start = time.time()
        for ep in range(self.start_epoch, self.epochs):
            tl, perf = self._train_epoch(ep)
            vl, vd = self._validate() if self.val_dl else (float("nan"), float("nan"))
            self.sched.step()
            rec = {"epoch": ep, "train_loss": tl, "val_loss": vl, "val_patch_dice": vd,
                   "lr": self.opt.param_groups[0]["lr"], "mem_used_gb": psutil.virtual_memory().used / 1e9,
                   "swap_used_gb": psutil.swap_memory().used / 1e9, **perf, "time": time.time()}
            self.history.append(rec)
            with open(self.log_path, "a") as f: f.write(json.dumps(rec) + "\n")
            print(f"[ep {ep}] train {tl:.4f} val {vl:.4f} dice {vd:.4f} lr {rec['lr']:.2e} {perf['patches_per_s']:.2f} patch/s", flush=True)
            score = vd if self.val_dl else -tl
            improved = score > self.best
            if improved: self.best, self.bad = score, 0
            else: self.bad += 1
            ck = dict(self.extra)
            if (ep + 1) % self.cfg.get("checkpoint_interval", 1) == 0:
                save_checkpoint(self.out / "last.pt", self.model, self.opt, self.sched, ep, ck, self.best, self.history)
            if improved:
                save_checkpoint(self.out / "best.pt", self.model, self.opt, self.sched, ep, ck, self.best, self.history)
            if self.bad >= self.cfg.get("early_stopping_patience", 10):
                print("early stopping"); break
            if max_minutes and (time.time() - t_start) / 60 > max_minutes:
                print("time budget reached; resume with --resume"); break
        return self.history
