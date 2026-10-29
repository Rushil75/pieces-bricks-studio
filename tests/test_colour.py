"""engine.colour must match scikit-image, which the engine used before (dev-only dependency)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import colour  # noqa: E402

skc = pytest.importorskip("skimage.color")
skm = pytest.importorskip("skimage.metrics")

rng = np.random.default_rng(0)
RGB = np.concatenate([rng.random((5000, 3)), np.eye(3), np.zeros((1, 3)), np.ones((1, 3)),
                      [[0.5, 0.5, 0.5], [0.01, 0.01, 0.01]]])


def test_rgb2lab_matches_skimage():
    np.testing.assert_allclose(colour.rgb2lab(RGB), skc.rgb2lab(RGB), atol=1e-9)


@pytest.mark.filterwarnings("ignore:Conversion from CIE-LAB")  # skimage warns on clipped Z
def test_lab2rgb_matches_skimage():
    lab = skc.rgb2lab(RGB)
    np.testing.assert_allclose(colour.lab2rgb(lab), skc.lab2rgb(lab), atol=1e-9)
    # out-of-gamut LAB values, as produced by k-means centres
    odd = np.array([[50, 90, -90], [5, -40, 60], [98, 20, 20], [0, 0, 0], [100, 0, 0]], float)
    np.testing.assert_allclose(colour.lab2rgb(odd), skc.lab2rgb(odd), atol=1e-9)


@pytest.mark.parametrize("kL", [1.0, 0.85])
def test_ciede2000_matches_skimage(kL):
    lab = skc.rgb2lab(RGB)
    other = np.roll(lab, 7, axis=0)
    np.testing.assert_allclose(colour.deltaE_ciede2000(lab, other, kL=kL),
                               skc.deltaE_ciede2000(lab, other, kL=kL), atol=1e-9)
    # identical and achromatic colours (C = 0 branches)
    grey = np.array([[50, 0, 0], [60, 0, 0], [50, 0, 0]], float)
    np.testing.assert_allclose(colour.deltaE_ciede2000(grey, grey[::-1]),
                               skc.deltaE_ciede2000(grey, grey[::-1]), atol=1e-9)


def test_ssim_matches_skimage():
    a = rng.random((48, 48)) * 100
    b = np.clip(a + rng.normal(0, 8, a.shape), 0, 100)
    assert colour.ssim(a, b, 100) == pytest.approx(
        skm.structural_similarity(a, b, data_range=100, win_size=7), abs=1e-9)
