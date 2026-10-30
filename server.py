"""Demo API + static page.

  uvicorn server:app --port 8000      (from pieces_bricks_demo/)

The page calls the same engine as run_demo.py, so web results match the offline set exactly.

Environment (all optional):
  ALLOWED_ORIGINS   comma-separated origins allowed to call the API (e.g. the Netlify site); default *
  DEMO_PASSWORD     if set, every /api call needs header X-Demo-Password with this value
"""
import hashlib
import hmac
import tempfile
import io
import json
import os
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from engine import export, metrics, pipeline, render
from engine.palette import load_client_palette
from engine.preprocess import detect_faces, load_image, plan_crop

ROOT = Path(__file__).resolve().parent
# Serverless (Vercel) has a read-only app folder: keep uploads in the temp dir there.
UPLOADS = Path(tempfile.gettempdir()) / "pb_uploads" if os.environ.get("VERCEL") else ROOT / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
WEB = ROOT / "web" / "dist"
VIEW_MAX = 1600
MAX_UPLOAD = 25 * 1024 * 1024

SAMPLES = {
    "p1": ("Girl with teddy", "p1"),
    "p2": ("Studio family", "p2a"),
    "p3": ("Birthday family", "p3a"),
    "p4": ("Toddler on bench", "p4"),
    "p5": ("Man, colour background", "p5"),
}

PASSWORD = os.environ.get("DEMO_PASSWORD", "")
# Paths that work without the password: health check, and photo previews (ids are unguessable
# hashes, and <img> tags cannot send headers).
OPEN_PATHS = ("/api/health",)

app = FastAPI(title="PIECES & BRICKS demo")


@app.middleware("http")
async def require_password(request: Request, call_next):
    path = request.url.path
    if (PASSWORD and request.method != "OPTIONS" and path.startswith("/api/")
            and path not in OPEN_PATHS and not path.endswith("/view")):
        given = request.headers.get("x-demo-password", "")
        if not hmac.compare_digest(given.encode(), PASSWORD.encode()):
            return JSONResponse({"detail": "password required"}, status_code=401)
    return await call_next(request)


# Added after the password check so CORS headers are also set on its 401 responses.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Demo-Password"],
)


def _sample_file(job):
    return ROOT / "input" / pipeline.load_config()["jobs"][job]["file"]


@app.get("/api/health")
def health():
    return {"ok": True, "password": bool(PASSWORD)}


def _photo_path(pid):
    if not pid.isalnum():
        raise HTTPException(400, "bad id")
    hits = [p for p in UPLOADS.glob(pid + ".*") if p.suffix != ".json"]
    if not hits:
        raise HTTPException(404, "photo not found, upload it again")
    return hits[0]


def _save(src_bytes, suffix):
    """Store by content hash, so the same photo reuses the engine's preprocessing cache."""
    path = UPLOADS / f"{hashlib.sha1(src_bytes).hexdigest()[:16]}{suffix}"
    if not path.exists():
        path.write_bytes(src_bytes)
    return path


def _register(src_bytes, suffix, job=None):
    path = _save(src_bytes, suffix)
    pid = path.stem
    try:
        rgb = load_image(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "not an image we can read")
    h, w = rgb.shape[:2]
    view = UPLOADS / f"{pid}_view.jpg"
    if not view.exists():
        im = Image.fromarray(rgb)
        im.thumbnail((VIEW_MAX, VIEW_MAX), Image.LANCZOS)
        im.save(view, quality=92)

    s = pipeline.job_settings(job) if job else pipeline.default_settings()
    faces = detect_faces(rgb, s["face_score"])
    auto = plan_crop(rgb.shape, faces, s)
    meta = {"job": job}
    (UPLOADS / f"{pid}.json").write_text(json.dumps(meta))
    return {
        "id": pid,
        "width": w,
        "height": h,
        "view_url": f"/api/photo/{pid}/view",
        "faces": [[round(v) for v in f[:4]] for f in faces],
        "auto_crop": list(auto),
    }


async def _read_upload(file):
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "photo is larger than 25 MB")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"):
        suffix = ".img"
    return data, suffix


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    return _register(*(await _read_upload(file)))


@app.get("/api/samples")
def samples():
    # Client photos are not in the public repo; only offer the ones present on this server.
    return [{"key": k, "title": t, "job": job} for k, (t, job) in SAMPLES.items() if _sample_file(job).exists()]


@app.post("/api/samples/{key}")
def load_sample(key: str):
    if key not in SAMPLES or not _sample_file(SAMPLES[key][1]).exists():
        raise HTTPException(404)
    job = SAMPLES[key][1]
    src = _sample_file(job)
    out = _register(src.read_bytes(), src.suffix.lower(), job)
    out["title"] = SAMPLES[key][0]
    out["job"] = job
    return out


@app.get("/api/samples/{key}/file")
def sample_file(key: str):
    if key not in SAMPLES or not _sample_file(SAMPLES[key][1]).exists():
        raise HTTPException(404)
    return FileResponse(_sample_file(SAMPLES[key][1]))


@app.get("/api/photo/{pid}/view")
def photo_view(pid: str):
    _photo_path(pid)
    return FileResponse(UPLOADS / f"{pid}_view.jpg", media_type="image/jpeg")


@app.get("/api/palette")
def client_palette():
    return load_client_palette().to_dict()


class ConvertIn(BaseModel):
    job: str | None = None
    crop: list[int] = Field(min_length=3, max_length=3)
    size: int = Field(48, ge=8, le=256)
    dither: bool = False
    palette: str = Field("adaptive", pattern="^(adaptive|client)$")
    colors: int = Field(60, ge=4, le=256)
    face_detail: float = Field(0.15, ge=0, le=2)
    background: str = Field("bilateral", pattern="^(bilateral|meanshift|none)$")


@app.post("/api/convert")
async def convert(file: UploadFile = File(...), options: str = Form(...)):
    """Stateless: the photo comes with every request, so any server instance can answer
    (serverless hosts do not share disk between requests)."""
    body = ConvertIn.model_validate_json(options)
    path = _save(*(await _read_upload(file)))
    job = body.job if body.job in pipeline.load_config()["jobs"] else None
    base = pipeline.job_settings(job) if job else pipeline.default_settings()
    s = {**base, "file": path.name, "crop": body.crop, "palette": body.palette, "colors": body.colors,
         "grid_face_detail": body.face_detail, "bg_filter": body.background}
    grid, palette, pre = pipeline.convert(path, body.size, body.dither, s)
    settings = {**s, "crop_resolved": pre["crop_box"], "faces_in_crop": pre["faces"], "dither": body.dither}
    matrix = export.build_matrix(grid, palette, settings, {"variant": "dither" if body.dither else "nodither"})
    matrix["similarity"] = metrics.similarity(matrix, pre["crop"], pre["face_mask"])
    # The page may repaint cells with any colour of this palette, or the client's 40.
    matrix["palette_full"] = palette.entries
    return matrix


class ExportIn(BaseModel):
    matrix: dict


def _clean_matrix(m):
    """Validate an edited matrix and keep only the palette colours it actually uses."""
    grid = m.get("grid")
    n = m.get("size")
    if not isinstance(grid, list) or len(grid) != n or any(not isinstance(r, list) or len(r) != n for r in grid):
        raise HTTPException(400, "grid must be size x size")
    known = {e["code"]: e for e in (m.get("palette_full") or []) + load_client_palette().entries + m["palette"]["colors"]}
    used = {c for r in grid for c in r}
    missing = used - known.keys()
    if missing:
        raise HTTPException(400, f"unknown colour codes: {sorted(missing)[:5]}")
    out = {k: v for k, v in m.items() if k != "palette_full"}
    out["palette"] = {**m["palette"], "colors": [known[c] for c in sorted(used)]}
    out["edited"] = bool(m.get("edited"))
    return out


@app.post("/api/export")
def export_pack(body: ExportIn):
    m = _clean_matrix(body.matrix)
    n = m["size"]
    preview, flat, grid = render.cell_sizes(n)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("matrix.json", json.dumps(m, indent=1))
        for name, im in (("preview.png", render.render_preview(m, preview)),
                         ("flat.png", render.render_flat(m, flat)),
                         ("grid.png", render.render_grid(m, grid))):
            b = io.BytesIO()
            im.save(b, "PNG")
            z.writestr(name, b.getvalue())
        z.writestr("counts.csv", export.counts_csv(m))
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="mosaic_{n}x{n}.zip"'})


if WEB.exists():
    app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")
