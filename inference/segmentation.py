"""Load a trained checkpoint and segment an (already LPS+resampled) HU volume."""
from __future__ import annotations
import numpy as np, torch
from models.unet3d import Custom3DUNet, UNet3DConfig
from training.checkpoint import load_checkpoint
from lvdata.preprocess_ct import normalize_hu
from .sliding_window import sliding_window_predict


def load_model(ckpt_path, device="mps"):
    s = load_checkpoint(ckpt_path, map_location="cpu")
    model = Custom3DUNet(UNet3DConfig(**s["arch"])); model.load_state_dict(s["model"]); model.to(device).eval()
    return model, s


def segment_volume(model, ckpt, vol_hu, device="mps", overlap=0.5, batch_size=4, progress=None):
    pre = ckpt["preprocessing"]; patch = tuple(ckpt.get("patch", (64, 64, 64)))
    return sliding_window_predict(model, vol_hu, lambda a: normalize_hu(a, **pre), patch, overlap, batch_size, device, progress=progress)
