"""Experiments on the saint photo at 96x96 against the Brick.me screenshot (run from project root)."""
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from benchmark import extract_grid, score, table  # noqa: E402
from engine import pipeline  # noqa: E402
from engine.palette import load_client_palette, select_subset  # noqa: E402
from engine.quantize import match, optimise_view  # noqa: E402

N = 96
PHOTO = "input/saint.jpg"
orig = np.asarray(Image.open(PHOTO).convert("RGB"))
ref = cv2.resize(orig, (N, N), interpolation=cv2.INTER_AREA)
bm = extract_grid("input/ref_brickme_saint.png", (717, 105, 1487, 876), N, N)
valid = np.ones((N, N), bool)
for r in range(N):
    for c in range(N):
        x, y = 717 + (c + 0.5) * 770 / N, 105 + (r + 0.5) * 771 / N
        valid[r, c] = not (y > 840 and 980 < x < 1225)

client = load_client_palette()
NEUTRAL = {"contrast": 0, "brightness": 0, "detail": 1.0}
CURRENT = {"contrast": 15, "brightness": 5, "detail": 2.0}


def cells_for(over):
    s = pipeline.default_settings({"file": "saint.jpg", "crop": [0, 0, 554], **over})
    _, _, pre, cells = pipeline.convert(PHOTO, N, False, s, return_cells=True)
    fm = cv2.resize(pre["face_mask"], (N, N), interpolation=cv2.INTER_AREA)
    return cells, fm, s


def run(name, over, n_colours, opt=None, kL=1.0):
    cells, fm, s = cells_for(over)
    face = fm > 0.5
    weights = 1.0 + (s["face_weight"] - 1.0) * np.clip(fm, 0, 1)
    pal = select_subset(cells, weights, client, n_colours, kL) if n_colours < 58 else client
    t = time.time()
    idx = match(cells, pal, kL)
    if opt is not None:
        opt = dict(opt)
        if "fid_face" in opt:
            ff, fb = opt.pop("fid_face"), opt.pop("fid_bg")
            opt["fidelity"] = fb + (ff - fb) * np.clip(fm * 1.5, 0, 1)
        idx = optimise_view(cells, pal, idx, kL=kL, **opt)
    grid = pal.rgb[idx]
    r = score(grid, ref, face, valid)
    r["time"] = time.time() - t
    return name, r, grid


if __name__ == "__main__":

    face = cells_for(NEUTRAL)[1] > 0.5
    rows = [("Brick.me (screenshot)", score(bm, ref, face, valid), bm)]
    NB = {**NEUTRAL, "bg_filter": "none"}
    H = {"fid_face": 5.0, "fid_bg": 0.3}
    for name, over, n, opt, kL in [
        ("nearest kL .6, bg smooth", NEUTRAL, 20, None, 0.6),
        ("nearest kL .6, bg original", NB, 20, None, 0.6),
        ("hybrid kL .6, bg smooth", NEUTRAL, 20, H, 0.6),
        ("hybrid kL .6, bg original", NB, 20, H, 0.6),
        ("hybrid sL .5 fbg .5, bg original", NB, 20, {**H, "fid_bg": 0.5, "sigma_l": 0.5}, 0.6),
        ("hybrid fface 2, bg original", NB, 20, {**H, "fid_face": 2.0}, 0.6),
    ]:
        rows.append(run(name, over, n, opt, kL))
    table([(n, r) for n, r, _ in rows])
    for n, r, _ in rows:
        if "time" in r:
            print(f"{n:34s} optimise time {r['time']:.2f}s")
    np.save("output/exp_grids.npy", np.stack([g for _, _, g in rows]))
