"""Palettes: the client's fixed 40 colours, or an adaptive palette built per photo."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .colour import deltaE_ciede2000, lab2rgb, rgb2lab

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def hex_to_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*(int(v) for v in rgb))


def rgb_to_lab(rgb_u8):
    """sRGB uint8 (..., 3) -> CIELAB D65. sRGB is linearised first."""
    return rgb2lab(np.asarray(rgb_u8, dtype=np.float64) / 255.0)


def lab_to_rgb_u8(lab):
    return np.clip(np.round(lab2rgb(lab) * 255.0), 0, 255).astype(np.uint8)


def ciede2000_matrix(lab_points, lab_palette, kL=1.0, chunk=200_000):
    """Distance from each of M points to each of P palette colours -> (M, P).
    Computed in chunks of ~`chunk` pairs so large grids and palettes stay light on memory."""
    m, p = len(lab_points), len(lab_palette)
    out = np.empty((m, p))
    step = max(1, chunk // max(p, 1))
    for i in range(0, m, step):
        pts = lab_points[i:i + step]
        a = np.broadcast_to(pts[:, None, :], (len(pts), p, 3)).reshape(-1, 3)
        b = np.broadcast_to(lab_palette[None, :, :], (len(pts), p, 3)).reshape(-1, 3)
        out[i:i + step] = deltaE_ciede2000(a, b, kL=kL).reshape(len(pts), p)
    return out


@dataclass
class Palette:
    name: str
    version: str
    entries: list  # [{"code", "name", "hex"}]

    def __post_init__(self):
        self.codes = [e["code"] for e in self.entries]
        self.rgb = np.array([hex_to_rgb(e["hex"]) for e in self.entries], dtype=np.uint8)
        self.lab = rgb_to_lab(self.rgb)

    def to_dict(self):
        return {"name": self.name, "version": self.version, "colors": self.entries}

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["version"], d["colors"])


def load_client_palette(path=None):
    path = Path(path) if path else CONFIG_DIR / "palette.json"
    data = json.loads(path.read_text())
    return Palette("client", data["version"], data["colors"])


def _lloyd(lab, w, centers, kL, iters):
    """Weighted k-means: CIEDE2000 assignment, weighted LAB means. Empty clusters are dropped."""
    for _ in range(iters):
        assign = ciede2000_matrix(lab, centers, kL).argmin(axis=1)
        new = []
        for k in range(len(centers)):
            sel = assign == k
            if sel.any():
                new.append((lab[sel] * w[sel, None]).sum(0) / w[sel].sum())
        new = np.array(new)
        done = new.shape == centers.shape and np.abs(new - centers).max() < 0.05
        centers = new
        if done:
            break
    return centers


def _init_quantize(cells_rgb, w, n, method):
    """PIL quantizer seed (median cut / max coverage), repeating face cells so they get a fair share."""
    reps = np.clip(np.round(w), 1, None).astype(int)
    seed_img = Image.fromarray(np.repeat(cells_rgb, reps, axis=0).reshape(1, -1, 3))
    q = seed_img.quantize(colors=n, method=method, dither=Image.Dither.NONE)
    pal = np.array(q.getpalette()[: 3 * n], dtype=np.uint8).reshape(-1, 3)
    return rgb_to_lab(pal[np.unique(np.asarray(q).reshape(-1))])


def _init_farthest(lab, w, n, kL):
    """Deterministic k-means++-style seed: start at the heaviest cell, then keep adding the cell
    with the largest weighted distance to the colours chosen so far."""
    centers = [lab[int(np.argmax(w))]]
    err = deltaE_ciede2000(lab, np.broadcast_to(centers[0], lab.shape), kL=kL)
    for _ in range(n - 1):
        i = int(np.argmax(err * w))
        centers.append(lab[i])
        err = np.minimum(err, deltaE_ciede2000(lab, np.broadcast_to(lab[i], lab.shape), kL=kL))
    return np.array(centers)


INITS = ("mediancut", "maxcoverage", "farthest")


def build_adaptive_palette(cells_rgb, weights, n_colors, kL=1.0, iters=25, reference=None, reserve=0.2,
                           max_fit=9216, inits=("mediancut",), polish=0):
    """Pick the n_colors that best represent the grid cells (weighted k-means in LAB).

    Deterministic: each seed in `inits` (median cut, max coverage, farthest point) is refined
    with Lloyd iterations using CIEDE2000 assignment, and the palette with the lowest weighted
    error wins. `weights` lets face cells pull more colours towards skin tones. A `reserve`
    share of the budget is then spent greedily on the worst-matched cells (weighted), so small
    but important details such as eyes and lips keep their colour; `polish` runs a few more
    Lloyd steps over the whole palette afterwards.
    """
    cells_rgb = cells_rgb.reshape(-1, 3).astype(np.uint8)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(cells_rgb) > max_fit:
        # Big boards: fit on an evenly spread, deterministic sample of cells (keeps it fast).
        idx = np.linspace(0, len(cells_rgb) - 1, max_fit).round().astype(int)
        cells_rgb, w = cells_rgb[idx], w[idx]
    n_extra = int(round(n_colors * reserve))
    n_main = n_colors - n_extra
    lab = rgb_to_lab(cells_rgb)

    best, best_cost = None, np.inf
    for init in inits:
        if init == "farthest":
            seed = _init_farthest(lab, w, n_main, kL)
        else:
            method = Image.Quantize.MAXCOVERAGE if init == "maxcoverage" else Image.Quantize.MEDIANCUT
            seed = _init_quantize(cells_rgb, w, n_main, method)
        centers = _lloyd(lab, w, seed, kL, iters)

        err = ciede2000_matrix(lab, centers, kL).min(axis=1)
        for _ in range(n_extra):
            i = int(np.argmax(err * w))
            if err[i] < 1.0:
                break
            centers = np.vstack([centers, lab[i]])
            err = np.minimum(err, deltaE_ciede2000(lab, np.broadcast_to(lab[i], lab.shape), kL=kL))
        if polish:
            centers = _lloyd(lab, w, centers, kL, polish)
            err = ciede2000_matrix(lab, centers, kL).min(axis=1)

        cost = float((err * w).sum())
        if cost < best_cost:
            best, best_cost = centers, cost
    centers = best

    # Snap to real sRGB colours, drop duplicates, order dark -> light.
    rgb = np.unique(lab_to_rgb_u8(centers), axis=0)
    rgb = rgb[np.argsort(rgb_to_lab(rgb)[:, 0], kind="stable")]

    entries = []
    for i, c in enumerate(rgb, 1):
        name = "Custom"
        if reference is not None:
            j = int(ciede2000_matrix(rgb_to_lab(c[None]), reference.lab).argmin())
            name = f"Custom (near {reference.codes[j]} {reference.entries[j]['name']})"
        code = f"A{i:02d}" if len(rgb) <= 99 else f"A{i:03d}"
        entries.append({"code": code, "name": name, "hex": rgb_to_hex(c)})
    version = "adaptive-" + hashlib.sha1("".join(e["hex"] for e in entries).encode()).hexdigest()[:10]
    return Palette("adaptive", version, entries)
