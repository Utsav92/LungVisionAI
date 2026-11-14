import numpy as np, torch
from inference.nodule_detection import detect_nodules, eq_diameter
from inference.sliding_window import sliding_window_predict
from evaluation.segmentation_metrics import overlap_metrics, hd95, surface_dice
from evaluation.froc import match_scan, froc_curve, sens_at_fp
from lvdata.preprocess_ct import normalize_hu, voxel_to_physical, voxel_to_original_index
from lvdata.patch_sampler import crop, augment
from tests.synthetic import make_volume


def test_connected_components_and_diameter():
    v, m, ctr = make_volume((64, 64, 64), 2, (5, 8), seed=3)
    cands = detect_nodules(m.astype(np.float32), threshold=0.5)
    assert len(cands) == 2
    for c, (cc, r) in zip(sorted(cands, key=lambda c: c.centroid_zyx), sorted(ctr, key=lambda t: t[0])):
        assert np.allclose(c.centroid_zyx, cc, atol=0.6)
        assert abs(c.diameter_mm - 2 * r) < 2.0      # discrete sphere ~ analytic
    assert abs(eq_diameter(4 / 3 * np.pi * 125) - 10) < 1e-6


def test_anisotropic_spacing_diameter():
    m = np.zeros((20, 40, 40), np.float32); m[8:12, 10:30, 10:30] = 1   # box
    c = detect_nodules(m, spacing_zyx=(2.5, 1, 1), min_diameter_mm=1)[0]
    assert abs(c.volume_mm3 - 4 * 20 * 20 * 2.5) < 1e-6


def test_dedup_removes_nearby():
    m = np.zeros((40, 40, 40), np.float32); m[10:16, 10:16, 10:16] = 0.9; m[10:16, 10:16, 18:24] = 0.6
    assert len(detect_nodules(m, dedup_dist_mm=10.0)) == 1


def test_metrics():
    a = np.zeros((10, 10, 10), bool); a[2:6, 2:6, 2:6] = 1
    assert overlap_metrics(a, a)["dice"] == 1.0 and hd95(a, a) == 0.0 and surface_dice(a, a) == 1.0
    b = np.roll(a, 2, 0); assert 0 < overlap_metrics(a, b)["dice"] < 1 and hd95(a, b) > 0
    assert overlap_metrics(np.zeros_like(a), np.zeros_like(a))["dice"] == 1.0


def test_froc():
    refs = [{"centroid_zyx": (10, 10, 10), "diameter_mm": 8}]
    c1 = [{"centroid_zyx": (11, 10, 10), "confidence": .9}, {"centroid_zyx": (30, 30, 30), "confidence": .5}]
    c2 = [{"centroid_zyx": (5, 5, 5), "confidence": .8}]
    ps = [(match_scan(c1, refs), 1), (match_scan(c2, []), 0)]
    thr, s, f = froc_curve(ps)
    assert s.max() == 1.0 and f.max() == 1.0       # 2 FPs over 2 scans
    assert sens_at_fp(s, f)[0.5] == 1.0            # top-confidence hit before any FP... 0.5 FP/scan reached at t=.9: 0 FP
    assert match_scan(c1, refs)[0][1] == 0 and match_scan(c1, refs)[1][1] == -1


def test_coordinate_roundtrip():
    meta = {"original": {"size": [100, 100, 50], "spacing": [0.7, 0.7, 2.5], "origin": [-100., -120., 30.], "direction": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
            "resampled": {"size": [70, 70, 125], "spacing": [1., 1., 1.], "origin": [-100., -120., 30.], "direction": [1, 0, 0, 0, 1, 0, 0, 0, 1]}}
    x, y, z = voxel_to_original_index((50, 20, 10), meta)    # resampled (z=50,y=20,x=10)
    assert np.allclose([x, y, z], [10 / 0.7, 20 / 0.7, 50 / 2.5])
    assert np.allclose(voxel_to_physical((50, 20, 10), meta), [-90, -100, 80])


def test_normalize_and_crop_and_augment():
    a = np.array([-2000, -1000, -300, 400, 3000.]); n = normalize_hu(a)
    assert n.min() == 0 and n.max() == 1 and abs(n[2] - 700 / 1400) < 1e-6
    v = np.ones((10, 10, 10), np.float32); c = crop(v, (0, 0, 0), (8, 8, 8), -5)     # off-edge crop pads
    assert c.shape == (8, 8, 8) and c.min() == -5 and c.max() == 1
    vol, m, _ = make_volume((48, 48, 48), 1, (6, 8), 1); vol = normalize_hu(vol)
    rng = np.random.default_rng(0)
    for _ in range(5):
        i2, m2, p = augment(vol, m.astype(np.float32), rng, (32, 32, 32))
        assert i2.shape == m2.shape == (32, 32, 32) and set(np.unique(m2)) <= {0., 1.}


def test_mask_nearest_resample_preserves_binary_and_alignment():
    import SimpleITK as sitk
    from lvdata.preprocess_ct import resample
    vol, m, ctr = make_volume((40, 60, 60), 1, (5, 8), 5)
    im = sitk.GetImageFromArray(m); im.SetSpacing((0.7, 0.7, 2.5)); ms = resample(im, (1, 1, 1), True)
    a = sitk.GetArrayFromImage(ms); assert set(np.unique(a)) <= {0, 1}
    com_phys = np.array(np.argwhere(a).mean(0)) * 1.0; com_orig = np.array(np.argwhere(m).mean(0)) * np.array([2.5, .7, .7])
    assert np.linalg.norm(com_phys - com_orig) < 1.5


def test_sliding_window_matches_dense_on_synthetic():
    class Id(torch.nn.Module):          # "model" that returns logits from the input => deterministic
        def forward(self, x): return (x - 0.5) * 20
    v, m, _ = make_volume((70, 80, 90), 1, (6, 9), 2)
    prob, st = sliding_window_predict(Id(), v, lambda a: normalize_hu(a), (32, 32, 32), 0.5, 4, "cpu", skip_below_hu=-5000)
    ref = torch.sigmoid(torch.from_numpy((normalize_hu(v) - .5) * 20)).numpy()
    assert prob.shape == v.shape and np.abs(prob - ref).max() < 1e-4
    prob2, st2 = sliding_window_predict(Id(), v, lambda a: normalize_hu(a), (32, 32, 32), 0.5, 4, "cpu", skip_below_hu=-700)
    assert st2["windows_run"] <= st2["windows_total"]
