// Optional "printed tiles" mode: print-ready files for UV-printing the photo onto 1x1 tiles.
// Built in the browser from the full-resolution photo, so no server size limits apply.
import type { Crop } from "./api";

export type PrintLayout = "seamless" | "jig";

export interface PrintOptions {
  cols: number; // tiles across (board width in studs)
  rows: number; // tiles down (board height in studs)
  tileMm: number; // tile pitch, 8 mm for standard 1x1 pieces
  dpi: number;
  layout: PrintLayout;
  gapMm: number; // jig only: space between tiles in the jig
  bleedMm: number; // jig only: image printed past each tile edge
}

export const PRINT_DEFAULTS: Omit<PrintOptions, "cols" | "rows"> = {
  tileMm: 8,
  dpi: 300,
  layout: "seamless",
  gapMm: 2,
  bleedMm: 0.3,
};

const MAX_SIDE = 16000; // browsers refuse larger canvases

// Whole pixels per millimetre, so every size prints exactly: an 8 mm tile is a whole number of
// pixels, a 12.8 cm baseplate too, and the PNG stores the exact resolution (px per metre).
const mmToPx = (mm: number, pxPerMm: number) => mm * pxPerMm;

/** Sheet size in px and mm, and the resolution actually used (lowered if the sheet would be too big). */
export function sheetInfo(o: PrintOptions) {
  const span = (n: number) => (o.layout === "seamless" ? n * o.tileMm : n * o.tileMm + (n + 1) * o.gapMm);
  const widthMm = span(o.cols);
  const heightMm = span(o.rows);
  const longMm = Math.max(widthMm, heightMm);
  let pxPerMm = Math.max(1, Math.round(o.dpi / 25.4)); // 300 dpi -> 12 px/mm (304.8 dpi)
  if (longMm * pxPerMm > MAX_SIDE) pxPerMm = Math.max(1, Math.floor(MAX_SIDE / longMm));
  const dpi = Math.round(pxPerMm * 25.4 * 10) / 10;
  return { widthMm, heightMm, dpi, pxPerMm, widthPx: Math.round(widthMm * pxPerMm), heightPx: Math.round(heightMm * pxPerMm) };
}

export async function renderPrintSheet(
  image: HTMLImageElement,
  photoWidth: number,
  crop: Crop,
  o: PrintOptions,
): Promise<Blob> {
  const { pxPerMm, widthPx, heightPx } = sheetInfo(o);
  const vs = image.naturalWidth / photoWidth;
  const [cx, cy, cw, ch] = [crop[0] * vs, crop[1] * vs, crop[2] * vs, crop[3] * vs];
  const cv = document.createElement("canvas");
  cv.width = widthPx;
  cv.height = heightPx;
  const ctx = cv.getContext("2d")!;
  ctx.imageSmoothingQuality = "high";
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, widthPx, heightPx);

  if (o.layout === "seamless") {
    ctx.drawImage(image, cx, cy, cw, ch, 0, 0, widthPx, heightPx);
  } else {
    const tile = mmToPx(o.tileMm, pxPerMm);
    const gap = mmToPx(o.gapMm, pxPerMm);
    const bleed = Math.min(mmToPx(o.bleedMm, pxPerMm), gap / 2);
    const src = cw / o.cols; // source pixels per tile (crop has the board's shape)
    const srcBleed = (bleed / tile) * src;
    for (let r = 0; r < o.rows; r++) {
      for (let c = 0; c < o.cols; c++) {
        const x = gap + c * (tile + gap);
        const y = gap + r * (tile + gap);
        // Clamp the bleed at the photo's outer edge so the sheet never samples outside the crop.
        const l = c === 0 ? 0 : srcBleed, t = r === 0 ? 0 : srcBleed;
        const rr = c === o.cols - 1 ? 0 : srcBleed, b = r === o.rows - 1 ? 0 : srcBleed;
        const kx = tile / src;
        ctx.drawImage(
          image,
          cx + c * src - l, cy + r * src - t, src + l + rr, src + t + b,
          x - l * kx, y - t * kx, tile + (l + rr) * kx, tile + (t + b) * kx,
        );
      }
    }
  }
  const png = await new Promise<Blob>((ok, fail) => cv.toBlob((b) => (b ? ok(b) : fail(new Error("print sheet too large"))), "image/png"));
  return withResolution(png, pxPerMm);
}

// ---- PNG physical size (pHYs chunk), so print software uses the right size ----

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (const b of bytes) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

async function withResolution(png: Blob, pxPerMm: number): Promise<Blob> {
  const buf = new Uint8Array(await png.arrayBuffer());
  const ppm = pxPerMm * 1000; // exact
  const chunk = new Uint8Array(21);
  const v = new DataView(chunk.buffer);
  v.setUint32(0, 9);
  chunk.set([0x70, 0x48, 0x59, 0x73], 4); // "pHYs"
  v.setUint32(8, ppm);
  v.setUint32(12, ppm);
  chunk[16] = 1; // unit: metre
  v.setUint32(17, crc32(chunk.subarray(4, 17)));
  const ihdrEnd = 8 + 25; // signature + IHDR chunk
  return new Blob([buf.subarray(0, ihdrEnd), chunk, buf.subarray(ihdrEnd)], { type: "image/png" });
}
