// sRGB -> CIELAB (D65) and CIEDE2000, the same formulas as engine/colour.py, so colours removed
// in the page are re-matched exactly the way the engine would match them.

export type Lab = [number, number, number];

const M = [
  [0.412453, 0.35758, 0.180423],
  [0.212671, 0.71516, 0.072169],
  [0.019334, 0.119193, 0.950227],
];
const WHITE = [0.95047, 1.0, 1.08883];

export function hexToLab(hex: string): Lab {
  const h = hex.replace("#", "");
  const c = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
  const lin = c.map((v) => (v > 0.04045 ? ((v + 0.055) / 1.055) ** 2.4 : v / 12.92));
  const f = M.map((row, i) => {
    const t = (row[0] * lin[0] + row[1] * lin[1] + row[2] * lin[2]) / WHITE[i];
    return t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
  });
  return [116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])];
}

const deg = Math.PI / 180;

export function ciede2000(p: Lab, q: Lab, kL = 1): number {
  const [L1, a1, b1] = p;
  const [L2, a2, b2] = q;
  const cbar = 0.5 * (Math.hypot(a1, b1) + Math.hypot(a2, b2));
  const c7 = cbar ** 7;
  const G = 0.5 * (1 - Math.sqrt(c7 / (c7 + 25 ** 7)));
  const polar = (a: number, b: number): [number, number] => {
    const t = Math.atan2(b, a);
    return [Math.hypot(a, b), t < 0 ? t + 2 * Math.PI : t];
  };
  const [C1, h1] = polar(a1 * (1 + G), b1);
  const [C2, h2] = polar(a2 * (1 + G), b2);

  const Lbar = 0.5 * (L1 + L2);
  const tmp = (Lbar - 50) ** 2;
  const SL = 1 + (0.015 * tmp) / Math.sqrt(20 + tmp);
  const Lt = (L2 - L1) / (kL * SL);
  const Cbar = 0.5 * (C1 + C2);
  const Ct = (C2 - C1) / (1 + 0.045 * Cbar);

  const CC = C1 * C2;
  const hd = h2 - h1;
  let dH = hd > Math.PI ? hd - 2 * Math.PI : hd < -Math.PI ? hd + 2 * Math.PI : hd;
  if (CC === 0) dH = 0;
  const dHt = 2 * Math.sqrt(CC) * Math.sin(dH / 2);
  let Hbar = h1 + h2;
  if (CC === 0) Hbar *= 2;
  else if (Math.abs(hd) > Math.PI) Hbar += Hbar < 2 * Math.PI ? 2 * Math.PI : -2 * Math.PI;
  Hbar *= 0.5;
  const T =
    1 - 0.17 * Math.cos(Hbar - 30 * deg) + 0.24 * Math.cos(2 * Hbar) + 0.32 * Math.cos(3 * Hbar + 6 * deg) - 0.2 * Math.cos(4 * Hbar - 63 * deg);
  const Ht = dHt / (1 + 0.015 * Cbar * T);
  const c7b = Cbar ** 7;
  const Rc = 2 * Math.sqrt(c7b / (c7b + 25 ** 7));
  const dTheta = 30 * deg * Math.exp(-(((Hbar / deg - 275) / 25) ** 2));
  const Rt = -Math.sin(2 * dTheta) * Rc * Ct * Ht;
  return Math.sqrt(Math.max(Lt * Lt + Ct * Ct + Ht * Ht + Rt, 0));
}

/** Code of the nearest allowed colour to `target` (hex without #). */
export function nearest(target: string, allowed: { code: string; lab: Lab }[], kL: number): string {
  const t = hexToLab(target);
  let best = allowed[0].code;
  let bd = Infinity;
  for (const c of allowed) {
    const d = ciede2000(t, c.lab, kL);
    if (d < bd) {
      bd = d;
      best = c.code;
    }
  }
  return best;
}
