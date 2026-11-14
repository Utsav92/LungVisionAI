import torch
def build_scheduler(opt, cfg, epochs):
    kind = cfg.get("scheduler", "cosine")
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=cfg.get("min_lr", 1e-6))
    if kind == "constant":
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0)
    raise ValueError(kind)
