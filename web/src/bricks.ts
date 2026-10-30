// Canvas brick tiles. Same look as engine/render.py so the page matches preview.png.


function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as [number, number, number];
}

function mix(c: [number, number, number], t: number, a: number): string {
  return `rgb(${c.map((v) => Math.round(v * (1 - a) + t * a)).join(",")})`;
}

export function luminance(hex: string): number {
  const [r, g, b] = hexToRgb(hex);
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

const tileCache = new Map<number, HTMLCanvasElement>();

/** Transparent edges of a flat 1x1 tile (UV printing needs smooth, stud-less tops), laid over
 * the photo to preview the printed-tiles option. */
export function printedTileOverlay(cell: number): HTMLCanvasElement {
  const hit = tileCache.get(cell);
  if (hit) return hit;
  const ss = 4;
  const s = cell * ss;
  const big = document.createElement("canvas");
  big.width = big.height = s;
  const d = big.getContext("2d")!;
  const k = Math.min(1, Math.max(0.35, (cell - 4) / 16));
  const g = Math.max(1, Math.round(s * 0.035));
  const bev = Math.max(1, Math.round(s * 0.04));
  d.fillStyle = "rgba(20,18,22,0.8)"; // gap between tiles
  d.fillRect(0, 0, s, s);
  d.clearRect(g, g, s - 2 * g, s - 2 * g);
  d.fillStyle = `rgba(255,255,255,${0.18 * k})`; // glossy top/left edge
  d.fillRect(g, g, s - 2 * g, bev);
  d.fillRect(g, g, bev, s - 2 * g);
  d.fillStyle = `rgba(0,0,0,${0.16 * k})`; // bottom/right edge
  d.fillRect(g, s - g - bev, s - 2 * g, bev);
  d.fillRect(s - g - bev, g, bev, s - 2 * g);
  const tile = document.createElement("canvas");
  tile.width = tile.height = cell;
  const t = tile.getContext("2d")!;
  t.imageSmoothingQuality = "high";
  t.drawImage(big, 0, 0, cell, cell);
  tileCache.set(cell, tile);
  return tile;
}

const cache = new Map<string, HTMLCanvasElement>();

export function brickTile(hex: string, cell: number): HTMLCanvasElement {
  const key = `${hex}@${cell}`;
  const hit = cache.get(key);
  if (hit) return hit;

  const ss = 4; // supersample, then draw scaled down for smooth studs
  const s = cell * ss;
  const big = document.createElement("canvas");
  big.width = big.height = s;
  const d = big.getContext("2d")!;
  const rgb = hexToRgb(hex);
  // Bricks touch like real ones: the seam is a darker shade of the brick's own colour (no black
  // grid), and the stud's shadow and highlight give the 3D look. Same rules as engine/render.py.
  const k = Math.min(1, Math.max(0.35, (cell - 4) / 16));
  const g = Math.max(1, Math.round(s * 0.025));
  const bev = Math.max(1, Math.round(s * 0.04));

  d.fillStyle = mix(rgb, 0, 0.35);
  d.fillRect(0, 0, s, s);
  d.fillStyle = mix(rgb, 0, 0.12 * k);
  d.fillRect(g, g, s - 2 * g, s - 2 * g);
  d.fillStyle = mix(rgb, 255, 0.12 * k);
  d.fillRect(g, g, s - 2 * g - bev, s - 2 * g - bev);
  d.fillStyle = hex;
  d.fillRect(g + bev, g + bev, s - 2 * g - 2 * bev, s - 2 * g - 2 * bev);

  const c = s / 2;
  const r = s * 0.3;
  const off = s * 0.045 * k;
  const circle = (x: number, y: number, rad: number, fill: string) => {
    d.beginPath();
    d.arc(x, y, rad, 0, Math.PI * 2);
    d.fillStyle = fill;
    d.fill();
  };
  circle(c + off, c + off, r, mix(rgb, 0, 0.3 * k)); // stud shadow
  circle(c, c, r, mix(rgb, 0, 0.1 * k)); // stud side
  circle(c - off * 0.4, c - off * 0.4, r * 0.86, mix(rgb, 255, 0.06 * k)); // stud top
  d.beginPath();
  d.arc(c, c, r * 0.86, (190 * Math.PI) / 180, (265 * Math.PI) / 180);
  d.strokeStyle = mix(rgb, 255, 0.5 * k); // highlight
  d.lineWidth = Math.max(1, s * 0.05);
  d.stroke();

  const tile = document.createElement("canvas");
  tile.width = tile.height = cell;
  const t = tile.getContext("2d")!;
  t.imageSmoothingQuality = "high";
  t.drawImage(big, 0, 0, cell, cell);
  if (cache.size > 4000) cache.clear();
  cache.set(key, tile);
  return tile;
}
