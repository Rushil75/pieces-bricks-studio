"""run(job, size, dither) -> result. Deterministic: same photo + settings = same matrix."""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from . import export, metrics, render
from .palette import build_adaptive_palette, load_client_palette
from .preprocess import preprocess
from .quantize import dither as fs_dither
from .quantize import adjust, downscale, downscale_mask, face_detail, match

ROOT = Path(__file__).resolve().parent.parent
PREPROCESS_KEYS = (
    "crop", "crop_mode", "face_scale", "shift_down", "group_pad", "face_score", "upscale_min",
    "denoise_h", "clahe_clip", "clahe_tiles", "sharpen", "face_sharpen", "sharpen_sigma",
    "saturation", "bg_filter", "bg_spatial", "bg_color", "bg_passes", "file", "work_max",
)


def load_config(path=None):
    return yaml.safe_load(Path(path or ROOT / "config" / "photos.yaml").read_text())


def job_settings(job, overrides=None, config=None):
    cfg = config or load_config()
    s = dict(cfg["defaults"])
    s.update({k: v for k, v in cfg["jobs"][job].items() if k != "sizes"})
    s.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return s


@lru_cache(maxsize=int(os.environ.get("PREPROCESS_CACHE", "4")))  # each entry ~40-100 MB
def _preprocess_cached(path, frozen):
    return preprocess(path, dict(frozen))


def _freeze(s):
    def v(x):
        return tuple(x) if isinstance(x, list) else x
    return tuple((k, v(s.get(k))) for k in PREPROCESS_KEYS)


def default_settings(overrides=None, config=None):
    s = dict((config or load_config())["defaults"])
    s.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return s


def convert(path, size, dither, s, return_cells=False):
    """Core conversion for any photo: returns (grid codes, full palette, preprocess result),
    plus the target colour of every brick when return_cells (used to re-match edited palettes)."""
    # Keep at least ~10 source pixels per brick on big boards.
    s = {**s, "work_max": max(int(s.get("work_max", 1600)), size * 10)}
    pre = _preprocess_cached(str(path), _freeze(s))
    fmask = downscale_mask(pre["face_mask"], size)
    cells = face_detail(downscale(pre["image"], size), fmask, s["grid_face_detail"], s["grid_detail_sigma"])
    cells = adjust(cells, s)

    client = load_client_palette()
    if s["palette"] == "client":
        palette = client
    else:
        weights = 1.0 + (s["face_weight"] - 1.0) * np.clip(fmask, 0, 1)
        palette = build_adaptive_palette(cells, weights, int(s["colors"]), s["kL"], reference=client,
                                         reserve=s.get("palette_reserve", 0.2), iters=s.get("palette_iters", 25),
                                         inits=tuple(s.get("palette_inits", ["mediancut"])), polish=s.get("palette_polish", 0))

    if dither:
        # dither may be True (configured strength) or a 0-1 strength from the page's slider
        strength = s["dither_strength"] if dither is True else float(dither)
        face = s["dither_face_strength"] * strength / max(s["dither_strength"], 1e-6)
        idx = fs_dither(cells, palette, fmask, strength, face, s["dither_clamp"], s["kL"])
    else:
        idx = match(cells, palette, s["kL"])

    grid = [[palette.codes[i] for i in row] for row in idx.tolist()]
    if return_cells:
        return grid, palette, pre, cells
    return grid, palette, pre


def run(job, size, dither=False, overrides=None, out_root=None, config=None):
    s = job_settings(job, overrides, config)
    grid, palette, pre = convert(ROOT / "input" / s["file"], size, dither, s)
    variant = "dither" if dither else "nodither"
    settings = {**s, "crop_resolved": pre["crop_box"], "faces_in_crop": pre["faces"], "dither": dither}
    matrix = export.build_matrix(grid, palette, settings, {"job": job, "variant": variant})
    matrix["similarity"] = metrics.similarity(matrix, pre["crop"], pre["face_mask"])

    out = Path(out_root or ROOT / "output") / f"{job}_{size}_{variant}"
    out.mkdir(parents=True, exist_ok=True)
    export.write_matrix(matrix, out / "matrix.json")
    Image.fromarray(pre["crop"]).save(out / "crop.png")
    write_outputs(out)
    return {"dir": out, "matrix": matrix, "crop": pre["crop"]}


def write_outputs(out):
    """preview, flat, grid and counts are rendered only from the saved matrix.json."""
    out = Path(out)
    matrix = export.read_matrix(out / "matrix.json")
    preview, flat, grid = render.cell_sizes(matrix["size"])
    render.render_preview(matrix, preview).save(out / "preview.png")
    render.render_flat(matrix, flat).save(out / "flat.png")
    render.render_grid(matrix, grid).save(out / "grid.png")
    export.write_counts(matrix, out / "counts.csv")


def write_compare(job, size, out_root=None):
    root = Path(out_root or ROOT / "output")
    a, b = root / f"{job}_{size}_nodither", root / f"{job}_{size}_dither"
    ma, mb = export.read_matrix(a / "matrix.json"), export.read_matrix(b / "matrix.json")
    crop = Image.open(a / "crop.png").convert("RGB")
    cell = render.cell_sizes(size)[0]
    panels = [("No dithering", render.render_preview(ma, cell)), ("Dithering", render.render_preview(mb, cell))]
    pal = ma["palette"]["name"]
    caption = (f"{job} | {size} x {size} = {size * size:,} bricks | palette: {pal} | "
               f"colours used: {len(ma['palette']['colors'])} (no dither), {len(mb['palette']['colors'])} (dither)")
    path = root / f"{job}_{size}_compare.png"
    render.render_compare(crop, panels, caption).save(path)
    return path
