import { useEffect, useRef, useState } from "react";
import type { Crop, Photo } from "../api";

interface Props {
  photo: Photo;
  image: HTMLImageElement | null;
  crop: Crop;
  cols: number; // board width in bricks
  rows: number; // board height in bricks
  onChange: (c: Crop) => void;
}

const MIN_SIDE = 48;

/** Crop box [x, y, w, h] with the board's shape (aspect = w / h), kept inside the photo. */
export function clampCrop(c: Crop, w: number, h: number, aspect: number): Crop {
  let cw = Math.max(c[2], MIN_SIDE);
  let ch = cw / aspect;
  const k = Math.min(1, w / cw, h / ch);
  cw *= k;
  ch *= k;
  const x = Math.min(Math.max(c[0], 0), w - cw);
  const y = Math.min(Math.max(c[1], 0), h - ch);
  return [Math.round(x), Math.round(y), Math.round(cw), Math.round(ch)];
}

/** Re-shape a crop for a new board shape: same centre, and its short side stays the same size,
 * so the face keeps the same number of bricks across the board's short side. */
export function reshapeCrop(c: Crop, w: number, h: number, aspect: number): Crop {
  const side = Math.min(c[2], c[3]);
  const [cw, ch] = aspect >= 1 ? [side * aspect, side] : [side, side / aspect];
  return clampCrop([c[0] + c[2] / 2 - cw / 2, c[1] + c[3] / 2 - ch / 2, cw, ch], w, h, aspect);
}

/** The largest board-shaped crop that fits the photo, centred. */
export function wholeCrop(w: number, h: number, aspect: number): Crop {
  const [cw, ch] = w / h > aspect ? [h * aspect, h] : [w, w / aspect];
  return clampCrop([(w - cw) / 2, (h - ch) / 2, cw, ch], w, h, aspect);
}

export default function CropEditor({ photo, image, crop, cols, rows, onChange }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ px: number; py: number; crop: Crop } | null>(null);
  const [showGrid, setShowGrid] = useState(true);
  const aspect = cols / rows;
  const maxW = wholeCrop(photo.width, photo.height, aspect)[2];
  const minW = Math.max(MIN_SIDE, MIN_SIDE * aspect);

  useEffect(() => {
    const cv = ref.current;
    if (!cv || !image) return;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.round(cv.clientWidth * dpr);
    const H = Math.round(W / aspect);
    cv.width = W;
    cv.height = H;
    const ctx = cv.getContext("2d")!;
    const vs = image.naturalWidth / photo.width; // view image is a scaled copy
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(image, crop[0] * vs, crop[1] * vs, crop[2] * vs, crop[3] * vs, 0, 0, W, H);

    if (showGrid) {
      ctx.strokeStyle = "rgba(255,255,255,0.22)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 1; i < cols; i++) {
        const p = Math.round((i * W) / cols) + 0.5;
        ctx.moveTo(p, 0);
        ctx.lineTo(p, H);
      }
      for (let i = 1; i < rows; i++) {
        const p = Math.round((i * H) / rows) + 0.5;
        ctx.moveTo(0, p);
        ctx.lineTo(W, p);
      }
      ctx.stroke();
    }

    const k = W / crop[2];
    ctx.font = `${Math.round(12 * dpr)}px system-ui, sans-serif`;
    for (const [fx, fy, fw, fh] of photo.faces) {
      const x = (fx - crop[0]) * k;
      const y = (fy - crop[1]) * k;
      if (x + fw * k < 0 || y + fh * k < 0 || x > W || y > H) continue;
      ctx.strokeStyle = "rgba(255,214,0,0.9)";
      ctx.lineWidth = 2 * dpr;
      ctx.strokeRect(x, y, fw * k, fh * k);
      const bricks = Math.round((fw / crop[2]) * cols);
      const label = `face ≈ ${bricks} bricks`;
      const tw = ctx.measureText(label).width + 10 * dpr;
      ctx.fillStyle = "rgba(0,0,0,0.65)";
      ctx.fillRect(x, y - 20 * dpr, tw, 18 * dpr);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x + 5 * dpr, y - 6 * dpr);
    }
  }, [image, crop, cols, rows, aspect, photo, showGrid]);

  const toImage = (cv: HTMLCanvasElement, dx: number) => (dx / cv.clientWidth) * crop[2];

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { px: e.clientX, py: e.clientY, crop };
  };
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = drag.current;
    if (!d) return;
    const cv = e.currentTarget;
    const dx = toImage(cv, e.clientX - d.px);
    const dy = toImage(cv, e.clientY - d.py);
    onChange(clampCrop([d.crop[0] - dx, d.crop[1] - dy, d.crop[2], d.crop[3]], photo.width, photo.height, aspect));
  };
  const onPointerUp = () => (drag.current = null);

  const zoomAround = (factor: number, fx = 0.5, fy = 0.5) => {
    const cw = Math.min(Math.max(crop[2] * factor, minW), maxW);
    const ch = cw / aspect;
    const ax = crop[0] + crop[2] * fx;
    const ay = crop[1] + crop[3] * fy;
    onChange(clampCrop([ax - cw * fx, ay - ch * fy, cw, ch], photo.width, photo.height, aspect));
  };

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = cv.getBoundingClientRect();
      zoomAround(Math.exp(e.deltaY * 0.0015), (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
    };
    cv.addEventListener("wheel", wheel, { passive: false });
    return () => cv.removeEventListener("wheel", wheel);
  });

  // Slider: 0 = whole photo, 100 = tightest crop.
  const span = Math.log(maxW / minW) || 1;
  const zoomPct = Math.round((1 - Math.log(crop[2] / minW) / span) * 100);
  const setZoomPct = (p: number) => zoomAround((minW * Math.exp(span * (1 - p / 100))) / crop[2]);

  return (
    <div className="crop">
      <canvas
        ref={ref}
        className="crop-canvas"
        style={{ aspectRatio: `${cols} / ${rows}` }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        aria-label="Crop: drag to move, scroll to zoom"
      />
      <div className="crop-controls">
        <label className="slider">
          <span>Zoom</span>
          <input type="range" min={0} max={100} value={zoomPct} onChange={(e) => setZoomPct(+e.target.value)} />
        </label>
        <div className="btn-row">
          <button className="btn ghost sm" onClick={() => onChange(reshapeCrop(photo.auto_crop, photo.width, photo.height, aspect))}>
            Auto (faces)
          </button>
          <button className="btn ghost sm" onClick={() => onChange(wholeCrop(photo.width, photo.height, aspect))}>
            Whole photo
          </button>
          <label className="check">
            <input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} /> Brick grid
          </label>
        </div>
        <p className="hint">
          Drag to move, scroll to zoom. Faces with 12+ bricks across read clearly ({cols}×{rows} board).
        </p>
      </div>
    </div>
  );
}
