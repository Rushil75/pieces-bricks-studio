"""Colour maths in plain NumPy: sRGB <-> CIELAB (D65), CIEDE2000 and SSIM.

Same formulas and constants as scikit-image (rgb2lab, lab2rgb, deltaE_ciede2000,
structural_similarity), so results match it to floating-point precision, without pulling
SciPy + scikit-image (~170 MB) into the serverless bundle. tests/test_colour.py checks this.
"""
import cv2
import numpy as np

# sRGB (D65) <-> XYZ, as in skimage.color
XYZ_FROM_RGB = np.array([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]])
RGB_FROM_XYZ = np.linalg.inv(XYZ_FROM_RGB)
WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


def rgb2lab(rgb):
    """sRGB floats in [0, 1], shape (..., 3) -> CIELAB D65."""
    c = np.asarray(rgb, dtype=np.float64)
    lin = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = (lin @ XYZ_FROM_RGB.T) / WHITE_D65
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab2rgb(lab):
    """CIELAB D65, shape (..., 3) -> sRGB floats clipped to [0, 1]."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = lab[..., 1] / 500.0 + fy
    fz = np.maximum(fy - lab[..., 2] / 200.0, 0.0)
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f > 0.2068966, f ** 3, (f - 16.0 / 116.0) / 7.787) * WHITE_D65
    lin = xyz @ RGB_FROM_XYZ.T
    rgb = np.where(lin > 0.0031308, 1.055 * np.power(np.maximum(lin, 0), 1 / 2.4) - 0.055, lin * 12.92)
    return np.clip(rgb, 0.0, 1.0)


def _polar_2pi(x, y):
    r = np.hypot(x, y)
    t = np.arctan2(y, x)
    return r, t + (t < 0) * 2 * np.pi


def deltaE_ciede2000(lab1, lab2, kL=1.0, kC=1.0, kH=1.0):
    """CIEDE2000 colour difference between LAB arrays of shape (..., 3)."""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    cbar = 0.5 * (np.hypot(a1, b1) + np.hypot(a2, b2))
    c7 = cbar ** 7
    G = 0.5 * (1 - np.sqrt(c7 / (c7 + 25.0 ** 7)))
    scale = 1 + G
    C1, h1 = _polar_2pi(a1 * scale, b1)
    C2, h2 = _polar_2pi(a2 * scale, b2)

    Lbar = 0.5 * (L1 + L2)
    tmp = (Lbar - 50) ** 2
    SL = 1 + 0.015 * tmp / np.sqrt(20 + tmp)
    L_term = (L2 - L1) / (kL * SL)

    Cbar = 0.5 * (C1 + C2)
    SC = 1 + 0.045 * Cbar
    C_term = (C2 - C1) / (kC * SC)

    h_diff = h2 - h1
    h_sum = h1 + h2
    CC = C1 * C2
    dH = h_diff.copy()
    dH[h_diff > np.pi] -= 2 * np.pi
    dH[h_diff < -np.pi] += 2 * np.pi
    dH[CC == 0.0] = 0.0
    dH_term = 2 * np.sqrt(CC) * np.sin(dH / 2)

    Hbar = h_sum.copy()
    mask = np.logical_and(CC != 0.0, np.abs(h_diff) > np.pi)
    Hbar[mask & (h_sum < 2 * np.pi)] += 2 * np.pi
    Hbar[mask & (h_sum >= 2 * np.pi)] -= 2 * np.pi
    Hbar[CC == 0.0] *= 2
    Hbar *= 0.5

    T = (1 - 0.17 * np.cos(Hbar - np.deg2rad(30)) + 0.24 * np.cos(2 * Hbar)
         + 0.32 * np.cos(3 * Hbar + np.deg2rad(6)) - 0.20 * np.cos(4 * Hbar - np.deg2rad(63)))
    SH = 1 + 0.015 * Cbar * T
    H_term = dH_term / (kH * SH)

    c7 = Cbar ** 7
    Rc = 2 * np.sqrt(c7 / (c7 + 25.0 ** 7))
    dtheta = np.deg2rad(30) * np.exp(-(((np.rad2deg(Hbar) - 275) / 25) ** 2))
    R_term = -np.sin(2 * dtheta) * Rc * C_term * H_term

    dE2 = L_term ** 2 + C_term ** 2 + H_term ** 2 + R_term
    return np.sqrt(np.maximum(dE2, 0))


def ssim(x, y, data_range, win_size=7):
    """Mean structural similarity of two 2-D images (uniform window, sample covariance),
    as skimage.metrics.structural_similarity with its defaults."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def mean(img):
        return cv2.blur(img, (win_size, win_size), borderType=cv2.BORDER_REFLECT)

    ux, uy = mean(x), mean(y)
    uxx, uyy, uxy = mean(x * x), mean(y * y), mean(x * y)
    cov_norm = win_size ** 2 / (win_size ** 2 - 1)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
    pad = (win_size - 1) // 2
    return float(s[pad:-pad, pad:-pad].mean())
