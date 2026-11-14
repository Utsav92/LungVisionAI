"""Disk-backed (memory-mapped) patch dataset. Only the requested crop is read from disk; no full-CT RAM cache."""
from __future__ import annotations
import numpy as np, torch
from torch.utils.data import Dataset
from pathlib import Path
from .preprocess_ct import PROC, normalize_hu, DEFAULT_PRE
from .patch_sampler import load_scan_index, crop, augment


class PatchDataset(Dataset):
    """Deterministic given (seed, epoch, idx). pos_fraction of samples are nodule-centred (if scan has nodules)."""
    def __init__(self, cases, patch=(64, 64, 64), length=200, pos_fraction=0.5, hard_neg_fraction=0.5,
                 train=True, seed=0, pre=None, jitter=24, max_open=8):
        self.cases = list(cases); self.patch = tuple(patch); self.length = length
        self.pf, self.hf, self.train, self.seed, self.pre, self.jit = pos_fraction, hard_neg_fraction, train, seed, pre or DEFAULT_PRE, jitter
        self.epoch = 0; self.idx = {c: load_scan_index(c) for c in self.cases}
        self.pos_cases = [c for c in self.cases if self.idx[c][0]["positives"]]
        self.neg_ok = [c for c in self.cases if len(self.idx[c][1]) + len(self.idx[c][2]) > 0]
        self._mm = {}; self.max_open = max_open
        assert self.cases, "empty case list"

    def set_epoch(self, e): self.epoch = e
    def __len__(self): return self.length

    def _open(self, c):
        if c not in self._mm:
            if len(self._mm) >= self.max_open: self._mm.pop(next(iter(self._mm)))   # bounded cache of memmaps
            v = PROC / "volumes"; self._mm[c] = (np.load(v / f"{c}.img.npy", mmap_mode="r"), np.load(v / f"{c}.mask.npy", mmap_mode="r"))
        return self._mm[c]

    def sample_center(self, rng):
        want_pos = self.pos_cases and rng.random() < self.pf
        if want_pos:
            c = self.pos_cases[rng.integers(len(self.pos_cases))]; nod = self.idx[c][0]["positives"]
            n = nod[rng.integers(len(nod))]; ctr = np.array(n["centroid"]) + (rng.integers(-self.jit, self.jit + 1, 3) if self.train else 0)
            return c, ctr.round().astype(int), True
        pool = self.neg_ok if self.neg_ok else self.cases
        c = pool[rng.integers(len(pool))]; _, easy, hard = self.idx[c]
        use_hard = len(hard) and (rng.random() < self.hf or not len(easy)); arr = hard if use_hard else easy
        if not len(arr): arr = np.array([[s // 2 for s in self.idx[c][0]["shape"]]])
        return c, arr[rng.integers(len(arr))], False

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch if self.train else 0, i])
        c, ctr, pos = self.sample_center(rng); img, msk = self._open(c)
        big = tuple(int(np.ceil(s * 1.35)) for s in self.patch) if self.train else self.patch
        a = crop(img, ctr, big, -1024).astype(np.float32); b = crop(msk, ctr, big, 0).astype(np.float32)
        a = normalize_hu(a, **self.pre)
        if self.train: a, b, _ = augment(a, b, rng, self.patch)
        return torch.from_numpy(np.ascontiguousarray(a))[None], torch.from_numpy(np.ascontiguousarray(b))[None]
