"""Checkpoints: weights + optimizer + scheduler + epoch + RNG + preprocessing + arch config + split id."""
from __future__ import annotations
import random, numpy as np, torch
from pathlib import Path
from safetensors.torch import save_file


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def set_rng_state(s):
    random.setstate(s["python"]); np.random.set_state(s["numpy"]); torch.set_rng_state(s["torch"])


def save_checkpoint(path, model, optimizer, scheduler, epoch, extra: dict, best_metric=None, history=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict() if scheduler else None, "epoch": epoch,
             "rng": rng_state(), "best_metric": best_metric, "history": history or [], **extra}
    tmp = path.with_suffix(".tmp"); torch.save(state, tmp); tmp.replace(path)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(path.with_suffix(".safetensors")))


def load_checkpoint(path, model=None, optimizer=None, scheduler=None, map_location="cpu"):
    # weights_only=False: file holds RNG state + config dicts; only load checkpoints you created.
    s = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None: model.load_state_dict(s["model"])
    if optimizer is not None and s.get("optimizer"): optimizer.load_state_dict(s["optimizer"])
    if scheduler is not None and s.get("scheduler"): scheduler.load_state_dict(s["scheduler"])
    return s
