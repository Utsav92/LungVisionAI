import torch
def build_optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg.get("learning_rate", 1e-4), weight_decay=cfg.get("weight_decay", 1e-5))
