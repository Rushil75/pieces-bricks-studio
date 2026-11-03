"""run(job, size, dither) -> result. Deterministic: same photo + settings = same matrix.

A board size is either N (an N x N square) or (width, height) in bricks."""
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from . import export, metrics, render, studio
from .palette import build_adaptive_palette, load_client_palette, rgb_to_lab, select_subset
from .preprocess import preprocess
from .quantize import dither as fs_dither
from .quantize import adjust, downscale, downscale_mask, face_detail, match, optimise_view

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


def dims(size):
    """Board size -> (width, height) in bricks."""
    if isinstance(size, (int, np.integer)):
        return int(size), int(size)
    w, h = size
    return int(w), int(h)


def size_label(size):
    w, h = dims(size)
    return str(w) if w == h else f"{w}x{h}"


@lru_cache(maxsize=int(os.environ.get("PREPROCESS_CACHE", "4")))  # each entry ~40-100 MB
def _preprocess_cached(path, frozen, aspect=1.0):
    return preprocess(path, dict(frozen), aspect)


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
    if s.get("engine", "classic") == "studio":
        # dither: False/0 = the studio_dither setting, True = at least 0.5, or a 0-1 strength
        strength = s.get("studio_dither", 0.0) if not dither else (max(0.5, s.get("studio_dither", 0.0)) if dither is True else float(dither))
        grid, palette, pre, cells = studio.convert(path, size, s, strength)
        return (grid, palette, pre, cells) if return_cells else (grid, palette, pre)
    w, h = dims(size)
    # Over-saturation guard: on with the Vivid filter (boosted colour would otherwise pick garish
    # oranges for pastels); off by default because it also nudges ordinary skin duller.
    s = {**s, "oversat": s.get("oversat", 0.5 if s.get("filter") == "vivid" else 0.0)}
    # Keep at least ~10 source pixels per brick on big boards.
    s = {**s, "work_max": max(int(s.get("work_max", 1600)), max(w, h) * 10)}
    pre = _preprocess_cached(str(path), _freeze(s), round(w / h, 6))
    fmask = downscale_mask(pre["face_mask"], (w, h))
    cells = face_detail(downscale(pre["image"], (w, h)), fmask, s["grid_face_detail"], s["grid_detail_sigma"])
    cells = adjust(cells, s)

    client = load_client_palette()
    if s["palette"] == "client":
        palette = client
        n = int(s.get("client_colors") or 0)
        if 0 < n < len(client.entries):
            weights = 1.0 + (s["face_weight"] - 1.0) * np.clip(fmask, 0, 1)
            palette = select_subset(cells, weights, client, n, s["kL"], kC=s.get("kC", 1.0),
                                    oversat=s.get("oversat", 0.0))
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
        idx = match(cells, palette, s["kL"], s.get("kC", 1.0), s.get("oversat", 0.0))
        blend = s.get("blend", "background")
        if blend != "off":
            # Viewing-distance optimisation: neighbouring bricks mix as the eye sees them, for
            # smooth gradients and a natural colour spread - but only where the photo itself has
            # texture or gradients. Flat areas keep one clean colour (blending them only adds grain)
            # and faces stay clean unless blend == "all".
            # Faces are never blended (1e9 = a face brick never trades its own best colour for a mix:
            # the client colours lack muted skin tones, so mixing would add grey bricks to skin).
            clean, mixed = s.get("blend_face_fidelity", 1e9), s.get("blend_fidelity", 0.3)
            L = rgb_to_lab(cells)[..., 0]
            mu = cv2.GaussianBlur(L, (0, 0), 1.0)
            texture = np.sqrt(np.maximum(cv2.GaussianBlur(L * L, (0, 0), 1.0) - mu * mu, 0))
            lo, hi = s.get("blend_texture", [1.5, 4.0])
            t = np.clip((texture - lo) / (hi - lo), 0, 1)
            if blend != "all":
                face = cv2.dilate(np.clip(fmask * 2, 0, 1).astype(np.float32), np.ones((3, 3), np.uint8))
                t = t * (1 - face)
            flat = s.get("blend_flat_fidelity", 5.0)  # flat areas: one clean colour
            fid = flat + (mixed - flat) * t
            if blend != "all":
                fid = np.where(face > 0.25, clean, fid)
            idx = optimise_view(cells, palette, idx, s.get("blend_sigma_l", 0.35), s.get("blend_sigma_ab", 1.4),
                                fid, s["kL"], max_extra=s.get("blend_max_extra", 6.0), kC=s.get("kC", 1.0),
                                oversat=s.get("oversat", 0.0))

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

    out = Path(out_root or ROOT / "output") / f"{job}_{size_label(size)}_{variant}"
    out.mkdir(parents=True, exist_ok=True)
    export.write_matrix(matrix, out / "matrix.json")
    Image.fromarray(pre["crop"]).save(out / "crop.png")
    write_outputs(out)
    return {"dir": out, "matrix": matrix, "crop": pre["crop"]}


def write_outputs(out):
    """preview, flat, grid and counts are rendered only from the saved matrix.json."""
    out = Path(out)
    matrix = export.read_matrix(out / "matrix.json")
    preview, flat, grid = render.cell_sizes(max(export.dims(matrix)))
    render.render_preview(matrix, preview).save(out / "preview.png")
    render.render_flat(matrix, flat).save(out / "flat.png")
    render.render_grid(matrix, grid).save(out / "grid.png")
    export.write_counts(matrix, out / "counts.csv")


def write_compare(job, size, out_root=None):
    root = Path(out_root or ROOT / "output")
    label = size_label(size)
    a, b = root / f"{job}_{label}_nodither", root / f"{job}_{label}_dither"
    ma, mb = export.read_matrix(a / "matrix.json"), export.read_matrix(b / "matrix.json")
    crop = Image.open(a / "crop.png").convert("RGB")
    w, h = dims(size)
    cell = render.cell_sizes(max(w, h))[0]
    panels = [("No dithering", render.render_preview(ma, cell)), ("Dithering", render.render_preview(mb, cell))]
    pal = ma["palette"]["name"]
    caption = (f"{job} | {w} x {h} = {w * h:,} bricks | palette: {pal} | "
               f"colours used: {len(ma['palette']['colors'])} (no dither), {len(mb['palette']['colors'])} (dither)")
    path = root / f"{job}_{label}_compare.png"
    render.render_compare(crop, panels, caption).save(path)
    return path
