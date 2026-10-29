// Canvas brick tiles. Same look as engine/render.py so the page matches preview.png.

const GAP = "rgb(24,22,26)";

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
  // Gap, bevel and shading scale with the brick, so small bricks (96x96 on screen)
  // keep their real brightness instead of being dominated by dark edges.
  const k = Math.min(1, Math.max(0.35, (cell - 4) / 16));
  const g = Math.max(1, Math.round(s * 0.045));
  const bev = Math.max(1, Math.round(s * 0.045));

  d.fillStyle = cell >= 16 ? GAP : mix(rgb, 0, 0.5);
  d.fillRect(0, 0, s, s);
  d.fillStyle = mix(rgb, 0, 0.16 * k);
  d.fillRect(g, g, s - 2 * g, s - 2 * g);
  d.fillStyle = mix(rgb, 255, 0.14 * k);
  d.fillRect(g, g, s - 2 * g - bev, s - 2 * g - bev);
  d.fillStyle = hex;
  d.fillRect(g + bev, g + bev, s - 2 * g - 2 * bev, s - 2 * g - 2 * bev);

  const c = s / 2;
  const r = s * 0.31;
  const off = s * 0.04 * k;
  const circle = (x: number, y: number, rad: number, fill: string) => {
    d.beginPath();
    d.arc(x, y, rad, 0, Math.PI * 2);
    d.fillStyle = fill;
    d.fill();
  };
  circle(c + off, c + off, r, mix(rgb, 0, 0.22 * k));
  circle(c, c, r, mix(rgb, 255, 0.08 * k));
  circle(c, c, r * 0.84, hex);
  d.beginPath();
  d.arc(c, c, r * 0.9, (195 * Math.PI) / 180, (255 * Math.PI) / 180);
  d.strokeStyle = mix(rgb, 255, 0.45 * k);
  d.lineWidth = Math.max(1, s * 0.064);
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
