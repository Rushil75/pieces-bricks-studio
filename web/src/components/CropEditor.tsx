import { useEffect, useRef, useState } from "react";
import type { Crop, Photo } from "../api";

interface Props {
  photo: Photo;
  image: HTMLImageElement | null;
  crop: Crop;
  gridSize: number;
  onChange: (c: Crop) => void;
}

const MIN_SIDE = 48;

export function clampCrop(c: Crop, w: number, h: number): Crop {
  const side = Math.round(Math.min(Math.max(c[2], Math.min(MIN_SIDE, w, h)), w, h));
  const x = Math.round(Math.min(Math.max(c[0], 0), w - side));
  const y = Math.round(Math.min(Math.max(c[1], 0), h - side));
  return [x, y, side];
}

export default function CropEditor({ photo, image, crop, gridSize, onChange }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ px: number; py: number; crop: Crop } | null>(null);
  const [showGrid, setShowGrid] = useState(true);
  const maxSide = Math.min(photo.width, photo.height);

  useEffect(() => {
    const cv = ref.current;
    if (!cv || !image) return;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.round(cv.clientWidth * dpr);
    cv.width = cv.height = W;
    const ctx = cv.getContext("2d")!;
    const vs = image.naturalWidth / photo.width; // view image is a scaled copy
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(image, crop[0] * vs, crop[1] * vs, crop[2] * vs, crop[2] * vs, 0, 0, W, W);

    if (showGrid) {
      ctx.strokeStyle = "rgba(255,255,255,0.22)";
      ctx.lineWidth = 1;
      const step = W / gridSize;
      ctx.beginPath();
      for (let i = 1; i < gridSize; i++) {
        const p = Math.round(i * step) + 0.5;
        ctx.moveTo(p, 0);
        ctx.lineTo(p, W);
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
      if (x + fw * k < 0 || y + fh * k < 0 || x > W || y > W) continue;
      ctx.strokeStyle = "rgba(255,214,0,0.9)";
      ctx.lineWidth = 2 * dpr;
      ctx.strokeRect(x, y, fw * k, fh * k);
      const bricks = Math.round((fw / crop[2]) * gridSize);
      const label = `face ≈ ${bricks} bricks`;
      const tw = ctx.measureText(label).width + 10 * dpr;
      ctx.fillStyle = "rgba(0,0,0,0.65)";
      ctx.fillRect(x, y - 20 * dpr, tw, 18 * dpr);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x + 5 * dpr, y - 6 * dpr);
    }
  }, [image, crop, gridSize, photo, showGrid]);

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
    onChange(clampCrop([d.crop[0] - dx, d.crop[1] - dy, d.crop[2]], photo.width, photo.height));
  };
  const onPointerUp = () => (drag.current = null);

  const zoomAround = (factor: number, fx = 0.5, fy = 0.5) => {
    const side = Math.min(Math.max(crop[2] * factor, MIN_SIDE), maxSide);
    const ax = crop[0] + crop[2] * fx;
    const ay = crop[1] + crop[2] * fy;
    onChange(clampCrop([ax - side * fx, ay - side * fy, side], photo.width, photo.height));
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
  const zoomPct = Math.round((1 - Math.log(crop[2] / MIN_SIDE) / Math.log(maxSide / MIN_SIDE)) * 100);
  const setZoomPct = (p: number) => {
    const side = MIN_SIDE * Math.pow(maxSide / MIN_SIDE, 1 - p / 100);
    zoomAround(side / crop[2]);
  };

  return (
    <div className="crop">
      <canvas
        ref={ref}
        className="crop-canvas"
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
          <button className="btn ghost sm" onClick={() => onChange(clampCrop(photo.auto_crop, photo.width, photo.height))}>
            Auto (faces)
          </button>
          <button
            className="btn ghost sm"
            onClick={() =>
              onChange(clampCrop([(photo.width - maxSide) / 2, (photo.height - maxSide) / 2, maxSide], photo.width, photo.height))
            }
          >
            Whole photo
          </button>
          <label className="check">
            <input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} /> Brick grid
          </label>
        </div>
        <p className="hint">
          Drag to move, scroll to zoom. Faces with 12+ bricks across read clearly at {gridSize}×{gridSize}.
        </p>
      </div>
    </div>
  );
}
