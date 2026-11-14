import numpy as np, torch, pytest
from models.unet3d import Custom3DUNet
from models.resnet3d import Nodule3DResNet
from training.losses import soft_dice_loss, DiceBCELoss
from training.checkpoint import save_checkpoint, load_checkpoint
from training.trainer import Trainer
from tests.synthetic import make_volume

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def test_forward_backward_shapes():
    m = Custom3DUNet(base_channels=8)
    x = torch.randn(1, 1, 32, 32, 32)
    y = m(x); assert y.shape == x.shape
    y.mean().backward()
    assert all(p.grad is not None for p in m.parameters())


def test_odd_sizes_forward():
    m = Custom3DUNet(base_channels=8).eval()
    assert m(torch.randn(1, 1, 48, 40, 56)).shape == (1, 1, 48, 40, 56)


def test_from_scratch_init_differs_by_seed():
    torch.manual_seed(0); a = Custom3DUNet(base_channels=8)
    torch.manual_seed(1); b = Custom3DUNet(base_channels=8)
    assert not torch.equal(a.stem[0][0].weight, b.stem[0][0].weight)


def test_dice_loss_values():
    t = torch.zeros(1, 1, 8, 8, 8); t[..., :4, :, :] = 1
    perfect = (t * 40 - 20)  # logits
    assert soft_dice_loss(perfect, t).item() < 0.01
    assert soft_dice_loss(-perfect, t).item() > 0.9
    # empty target + empty prediction must not blow up
    e = torch.zeros_like(t); assert soft_dice_loss(e - 20, e).item() < 0.01


def test_resnet_forward():
    assert Nodule3DResNet(base=8)(torch.randn(2, 1, 32, 32, 32)).shape == (2, 1)


def test_checkpoint_roundtrip(tmp_path):
    m = Custom3DUNet(base_channels=8); opt = torch.optim.AdamW(m.parameters())
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10)
    save_checkpoint(tmp_path / "c.pt", m, opt, sch, 3, {"preprocessing": {"hu": [-1000, 400]}, "split_id": "abc"})
    m2 = Custom3DUNet(base_channels=8); opt2 = torch.optim.AdamW(m2.parameters())
    s = load_checkpoint(tmp_path / "c.pt", m2, opt2)
    assert s["epoch"] == 3 and s["split_id"] == "abc" and s["preprocessing"]["hu"] == [-1000, 400]
    for a, b in zip(m.parameters(), m2.parameters()): assert torch.equal(a, b)
    assert (tmp_path / "c.safetensors").exists()


class TinyDS(torch.utils.data.Dataset):
    def __init__(self, n=2):
        self.items = []
        for s in range(n):
            v, m, _ = make_volume((32, 32, 32), 1, (4, 7), seed=s)
            v = (np.clip(v, -1000, 400) + 1000) / 1400
            self.items.append((torch.from_numpy(v)[None], torch.from_numpy(m.astype(np.float32))[None]))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def test_overfit_tiny_synthetic(tmp_path):
    """Milestone 7: the pipeline must be able to overfit 2 synthetic volumes (val dice > 0.8)."""
    ds = TinyDS(2)
    cfg = {"device": DEV.type, "epochs": 60, "learning_rate": 3e-3, "scheduler": "constant", "batch_size": 1,
           "gradient_accumulation": 1, "early_stopping_patience": 100, "seed": 0}
    tr = Trainer(Custom3DUNet(base_channels=8), cfg, ds, ds, tmp_path)
    hist = tr.fit()
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert hist[-1]["val_patch_dice"] > 0.8, hist[-1]
