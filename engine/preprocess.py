"""Steps 1-6: orientation, face detection, square crop, upscale, enhance, background."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
YUNET = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
FSRCNN = MODELS_DIR / "FSRCNN_x4.pb"


def load_image(path):
    """Step 1: read, apply EXIF rotation, RGB uint8."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.array(img)


def detect_faces(rgb, score=0.8, min_face=0.02, max_side=1280):
    """Step 2: YuNet face boxes [(x, y, w, h, score)] in full-image pixels, largest first."""
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    det = cv2.FaceDetectorYN.create(str(YUNET), "", (bgr.shape[1], bgr.shape[0]), score, 0.3, 50)
    _, faces = det.detect(bgr)
    out = []
    if faces is not None:
        for f in faces:
            x, y, fw, fh = (f[:4] / scale).tolist()
            if fw >= min_face * min(h, w):
                out.append((x, y, fw, fh, float(f[-1])))
    return sorted(out, key=lambda b: -b[2] * b[3])


def _clamp_square(cx, cy, side, w, h):
    side = int(round(min(side, w, h)))
    x = int(round(min(max(cx - side / 2, 0), w - side)))
    y = int(round(min(max(cy - side / 2, 0), h - side)))
    return x, y, side


def plan_crop(shape, faces, s):
    """Step 3: square crop (x, y, size). Manual crop wins; else frame the detected faces."""
    h, w = shape[:2]
    if s.get("crop") not in (None, "auto"):
        x, y, side = s["crop"]
        return _clamp_square(x + side / 2, y + side / 2, side, w, h)
    if not faces:
        return _clamp_square(w / 2, h / 2, min(w, h), w, h)
    if len(faces) == 1 or s.get("crop_mode") == "single":
        x, y, fw, fh, _ = faces[0]
        side = max(fw, fh) * s["face_scale"]
        return _clamp_square(x + fw / 2, y + fh / 2 + side * s["shift_down"], side, w, h)
    x0 = min(f[0] for f in faces)
    y0 = min(f[1] for f in faces)
    x1 = max(f[0] + f[2] for f in faces)
    y1 = max(f[1] + f[3] for f in faces)
    fsize = np.median([max(f[2], f[3]) for f in faces])
    side = max(x1 - x0, y1 - y0) + fsize * s["group_pad"]
    return _clamp_square((x0 + x1) / 2, (y0 + y1) / 2 + side * s["shift_down"], side, w, h)


def upscale(rgb, min_width, max_width=1600):
    """Step 4: FSRCNN x4 super-resolution when the crop is small (fallback: Lanczos).
    The result is capped at max_width: plenty of detail per brick, far less memory."""
    if rgb.shape[1] > max_width:
        f = max_width / rgb.shape[1]
        return cv2.resize(rgb, (max_width, max_width), interpolation=cv2.INTER_AREA), f
    if rgb.shape[1] >= min_width:
        return rgb, 1.0
    if rgb.shape[1] * 4 > max_width * 1.25:
        # Modest upscale needed: Lanczos is enough and avoids FSRCNN's large feature maps.
        f = min_width / rgb.shape[1]
        return cv2.resize(rgb, (min_width, min_width), interpolation=cv2.INTER_LANCZOS4), f
    out, factor = _super_res(rgb)
    if out.shape[1] > max_width:
        out = cv2.resize(out, (max_width, max_width), interpolation=cv2.INTER_AREA)
        factor = max_width / rgb.shape[1]
    return out, factor


def _super_res(rgb):
    factor = 4.0
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(FSRCNN))
        sr.setModel("fsrcnn", 4)
        bgr = sr.upsample(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        out = cv2.resize(rgb, None, fx=factor, fy=factor, interpolation=cv2.INTER_LANCZOS4)
    return out, factor


def face_mask(shape, faces, grow=1.25):
    """Soft elliptical mask over each face (1 = face)."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.float32)
    for x, y, fw, fh, _ in faces:
        c = (int(x + fw / 2), int(y + fh / 2))
        ax = (int(fw / 2 * grow), int(fh / 2 * grow * 1.15))
        cv2.ellipse(m, c, ax, 0, 0, 360, 1.0, -1)
    k = max(3, int(min(h, w) * 0.02) | 1)
    return cv2.GaussianBlur(m, (k, k), 0)


def person_mask(rgb, faces, work_side=512):
    """Person mask via GrabCut seeded from the face boxes (MediaPipe-free, deterministic)."""
    h, w = rgb.shape[:2]
    if not faces:
        return np.ones((h, w), np.float32)
    sc = work_side / max(h, w)
    small = cv2.resize(rgb, (max(1, round(w * sc)), max(1, round(h * sc))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    gc = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    for x, y, fw, fh, _ in faces:
        x, y, fw, fh = x * sc, y * sc, fw * sc, fh * sc
        # Head, hair and a body column below the face are probably the person.
        x0, x1 = int(max(0, x - fw * 1.3)), int(min(sw, x + fw * 2.3))
        y0 = int(max(0, y - fh * 0.6))
        gc[y0:, x0:x1] = cv2.GC_PR_FGD
    for x, y, fw, fh, _ in faces:
        x, y, fw, fh = x * sc, y * sc, fw * sc, fh * sc
        cv2.ellipse(gc, (int(x + fw / 2), int(y + fh / 2)), (int(fw * 0.4), int(fh * 0.45)), 0, 0, 360, cv2.GC_FGD, -1)
    cv2.setRNGSeed(0)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        pass
    m = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.float32)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    k = max(3, int(min(h, w) * 0.03) | 1)
    return cv2.GaussianBlur(m, (k, k), 0)


def enhance(rgb, fmask, s):
    """Step 5: denoise, CLAHE on L only, unsharp (stronger on faces), saturation."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if s["denoise_h"] > 0:
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, s["denoise_h"], s["denoise_h"], 7, 21)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    if s["clahe_clip"] > 0:
        clahe = cv2.createCLAHE(clipLimit=s["clahe_clip"], tileGridSize=(s["clahe_tiles"],) * 2)
        lab[..., 0] = clahe.apply(lab[..., 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)

    sigma = max(1.0, img.shape[1] / 600.0) * s["sharpen_sigma"]
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    amount = s["sharpen"] + (s["face_sharpen"] - s["sharpen"]) * fmask
    img = np.clip(img + (img - blur) * amount[..., None], 0, 255).astype(np.uint8)

    if s["saturation"] != 1.0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * s["saturation"], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def simplify_background(rgb, pmask, s):
    """Step 6: flatten everything outside the person mask."""
    mode = s["bg_filter"]
    if mode == "none":
        return rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    sc = 800 / max(bgr.shape[:2])  # filter at a fixed working size so strength is size-independent
    small = cv2.resize(bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    if mode == "meanshift":
        flat = cv2.pyrMeanShiftFiltering(small, s["bg_spatial"], s["bg_color"], maxLevel=1)
    else:
        flat = small
        for _ in range(s["bg_passes"]):
            flat = cv2.bilateralFilter(flat, 9, s["bg_color"], s["bg_spatial"])
    flat = cv2.resize(flat, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    m = pmask[..., None]
    out = bgr.astype(np.float32) * m + flat.astype(np.float32) * (1 - m)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2RGB)


def preprocess(path, s):
    """Run steps 1-6. Returns the untouched crop, the processed crop and its masks."""
    rgb = load_image(path)
    faces = detect_faces(rgb, s["face_score"])
    x, y, side = plan_crop(rgb.shape, faces, s)
    crop = rgb[y:y + side, x:x + side].copy()

    crop_faces = []
    for fx, fy, fw, fh, sc in faces:
        cx, cy = fx + fw / 2 - x, fy + fh / 2 - y
        if 0 <= cx < side and 0 <= cy < side:
            crop_faces.append((fx - x, fy - y, fw, fh, sc))

    work, factor = upscale(crop, s["upscale_min"], s.get("work_max", 1600))
    work_faces = [(a * factor, b * factor, c * factor, d * factor, e) for a, b, c, d, e in crop_faces]
    fmask = face_mask(work.shape, work_faces)
    pmask = np.maximum(person_mask(work, work_faces), fmask) if s["bg_filter"] != "none" else np.ones(work.shape[:2], np.float32)

    work = enhance(work, fmask, s)
    work = simplify_background(work, pmask, s)
    return {
        "crop": crop,
        "crop_box": [int(x), int(y), int(side)],
        "faces": [[round(v, 1) for v in f[:4]] for f in crop_faces],
        "image": work,
        "face_mask": fmask,
        "person_mask": pmask,
    }
