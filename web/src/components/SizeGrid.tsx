import { useEffect, useMemo, useRef } from "react";
import type { Matrix } from "../api";
import { brickTile } from "../bricks";

export interface SizeResult {
  plates: [number, number];
  matrix: Matrix | null; // null while it is being generated
}

const THUMB = 350; // long side of each card's mosaic, CSS px
const PLATE_LINES = 16; // baseplate seams in the large view and the PNG

/** Draw a mosaic's bricks onto ctx at (x0, y0), `cell` px per brick, optional baseplate seams. */
function drawBricks(ctx: CanvasRenderingContext2D, matrix: Matrix, hexOf: (c: string) => string, cell: number, x0 = 0, y0 = 0, plates = false) {
  const rows = matrix.grid.length;
  const cols = matrix.grid[0].length;
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const hex = hexOf(matrix.grid[r][c]);
      if (cell >= 4) ctx.drawImage(brickTile(hex, cell), x0 + c * cell, y0 + r * cell);
      else {
        ctx.fillStyle = hex;
        ctx.fillRect(x0 + c * cell, y0 + r * cell, cell, cell);
      }
    }
  if (plates) {
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.lineWidth = Math.max(1, cell / 6);
    ctx.beginPath();
    for (let k = PLATE_LINES; k < cols; k += PLATE_LINES) {
      ctx.moveTo(x0 + k * cell, y0);
      ctx.lineTo(x0 + k * cell, y0 + rows * cell);
    }
    for (let k = PLATE_LINES; k < rows; k += PLATE_LINES) {
      ctx.moveTo(x0, y0 + k * cell);
      ctx.lineTo(x0 + cols * cell, y0 + k * cell);
    }
    ctx.stroke();
  }
}

function Thumb({ matrix, hexOf, size = THUMB, plates = false }: { matrix: Matrix; hexOf: (c: string) => string; size?: number; plates?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const rows = matrix.grid.length;
  const cols = matrix.grid[0].length;
  const longest = Math.max(cols, rows);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    const cell = Math.max(2, Math.ceil((size * dpr) / longest));
    cv.width = cols * cell;
    cv.height = rows * cell;
    drawBricks(cv.getContext("2d")!, matrix, hexOf, cell, 0, 0, plates);
  }, [matrix, hexOf, cols, rows, longest, size, plates]);
  return <canvas ref={ref} style={{ width: (size * cols) / longest, height: (size * rows) / longest }} />;
}

export function colourCounts(matrix: Matrix): [string, number][] {
  const m = new Map<string, number>();
  for (const row of matrix.grid) for (const c of row) m.set(c, (m.get(c) ?? 0) + 1);
  return [...m.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}

/** PNG of the mosaic with a legend underneath: swatch, code, name and count of every colour. */
export async function posterPng(matrix: Matrix, hexOf: (c: string) => string, nameOf: (c: string) => string, title: string): Promise<Blob> {
  const rows = matrix.grid.length;
  const cols = matrix.grid[0].length;
  const cell = Math.max(6, Math.min(20, Math.floor(2400 / Math.max(cols, rows))));
  const pad = 40;
  const W = Math.max(cols * cell + pad * 2, 900);
  const counts = colourCounts(matrix);
  const colW = 300;
  const perRow = Math.max(1, Math.floor((W - pad * 2) / colW));
  const lineH = 34;
  const legendH = 70 + Math.ceil(counts.length / perRow) * lineH + 20;
  const cv = document.createElement("canvas");
  cv.width = W;
  cv.height = pad + 50 + rows * cell + legendH + pad;
  const ctx = cv.getContext("2d")!;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = "#16181d";
  ctx.font = "600 26px system-ui, sans-serif";
  ctx.fillText(title, pad, pad + 24);
  const mx = (W - cols * cell) / 2;
  drawBricks(ctx, matrix, hexOf, cell, mx, pad + 50, true);
  let y = pad + 50 + rows * cell + 50;
  ctx.font = "600 20px system-ui, sans-serif";
  ctx.fillText(`Colours and brick counts (${counts.length} colours, ${(rows * cols).toLocaleString()} bricks)`, pad, y);
  y += 22;
  ctx.font = "17px system-ui, sans-serif";
  counts.forEach(([code, n], i) => {
    const x = pad + (i % perRow) * colW;
    const yy = y + Math.floor(i / perRow) * lineH;
    ctx.fillStyle = hexOf(code);
    ctx.fillRect(x, yy + 6, 22, 22);
    ctx.strokeStyle = "rgba(0,0,0,.25)";
    ctx.strokeRect(x + 0.5, yy + 6.5, 21, 21);
    ctx.fillStyle = "#16181d";
    ctx.fillText(`${code} ${nameOf(code)}: ${n.toLocaleString()}`, x + 32, yy + 23);
  });
  return new Promise((ok, no) => cv.toBlob((b) => (b ? ok(b) : no(new Error("could not make the PNG"))), "image/png"));
}

interface BigProps {
  result: SizeResult;
  title: string;
  hexOf: (c: string) => string;
  nameOf: (c: string) => string;
  onClose: () => void;
  onEdit: () => void;
  onDownload: () => void;
}

/** Large view of one size with its colour legend (as the reference demo's pop-up). */
export function BigView({ result, title, hexOf, nameOf, onClose, onEdit, onDownload }: BigProps) {
  const m = result.matrix!;
  const counts = useMemo(() => colourCounts(m), [m]);
  const size = Math.min(window.innerHeight * 0.72, window.innerWidth * 0.9);
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  return (
    <div className="big-view" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="big-title">{title}</div>
      <Thumb matrix={m} hexOf={hexOf} size={size} plates />
      <div className="big-legend">
        {counts.map(([code, n]) => (
          <span key={code}>
            <i style={{ background: hexOf(code) }} />
            {code} {nameOf(code)}: {n.toLocaleString()}
          </span>
        ))}
      </div>
      <div className="big-actions">
        <button className="btn primary" onClick={onDownload}>Download PNG</button>
        <button className="btn" onClick={onEdit}>Edit this size</button>
        <button className="btn" onClick={onClose}>Close</button>
      </div>
    </div>
  );
}

interface Props {
  results: SizeResult[];
  current: [number, number];
  plate: number;
  cm: (studs: number) => string;
  hexOf: (c: string) => string;
  onOpen: (r: SizeResult) => void;
}

/** Every board size of the current shape at once, as in the reference demo. Click to edit one. */
export default function SizeGrid({ results, current, plate, cm, hexOf, onOpen }: Props) {
  return (
    <div className="size-grid">
      {results.map((r) => {
        const [pw, ph] = r.plates;
        const w = pw * plate;
        const h = ph * plate;
        const on = current[0] === w && current[1] === h;
        return (
          <button key={`${pw}x${ph}`} className={`size-result ${on ? "on" : ""}`} disabled={!r.matrix} onClick={() => r.matrix && onOpen(r)}>
            <div className="thumb" style={{ width: THUMB, height: THUMB }}>
              {r.matrix ? <Thumb matrix={r.matrix} hexOf={hexOf} /> : <div className="spinner" />}
            </div>
            <b>
              {cm(w)} × {cm(h)} cm
            </b>
            <small>
              {pw}×{ph} baseplates, {w}×{h} studs ({(w * h).toLocaleString()} bricks)
            </small>
          </button>
        );
      })}
    </div>
  );
}
