// Number formatting for the page, and the {{path|format}} filler used by vite.config.ts.
// One implementation: src/format.ts re-exports these for the browser code.
//
// Rounding is done on integers (Math.round(x * 10**k + 1e-9)), never with toFixed on the
// raw float, so 0.4875 reads 48.8% as in the README.

const MINUS = "−";

/** Round x to k decimals and print it with exactly k decimals. */
function fixed(x, k) {
  if (typeof x !== "number" || !Number.isFinite(x)) throw new Error(`not a finite number: ${x}`);
  const n = Math.round(x * 10 ** k + 1e-9);
  const sign = n < 0 ? MINUS : "";
  const abs = Math.abs(n);
  if (k === 0) return sign + String(abs);
  const s = String(abs).padStart(k + 1, "0");
  return `${sign}${s.slice(0, -k)}.${s.slice(-k)}`;
}

export const int = (x) => fixed(x, 0);
export const dp1 = (x) => fixed(x, 1);
export const dp2 = (x) => fixed(x, 2);
export const dp3 = (x) => fixed(x, 3);
export const dp4 = (x) => fixed(x, 4);
export const pct0 = (x) => `${fixed(x * 100, 0)}%`;
export const pct1 = (x) => `${fixed(x * 100, 1)}%`;
export const signed3 = (x) => {
  const s = fixed(x, 3);
  return s.startsWith(MINUS) || /^0\.0+$/.test(s) ? s : `+${s}`;
};

/** A positive number below 1 written out in full to 3 significant figures: 1.347e-17 -> 0.0000000000000000135. */
export const sig3 = (x) => {
  if (typeof x !== "number" || !(x > 0 && x < 1)) throw new Error(`sig3 needs 0 < x < 1, got ${x}`);
  const e = Math.floor(Math.log10(x));
  let digits = Math.round(x * 10 ** (2 - e) + 1e-9);
  let zeros = -e - 1;
  if (digits === 1000) {
    digits = 100;
    zeros -= 1;
  }
  return `0.${"0".repeat(zeros)}${digits}`;
};

const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export const formats = {
  int,
  dp1,
  dp2,
  dp3,
  dp4,
  pct0,
  pct1,
  signed3,
  sig3,
  /** Length of an array. */
  len: (x) => {
    if (!Array.isArray(x)) throw new Error("len needs an array");
    return String(x.length);
  },
  /** A string value, HTML-escaped. */
  text: (x) => escapeHtml(x),
};

/**
 * Resolve a path like `split.meanGap`, `benchmark.0.mean` or
 * `singleFeature[model=ALL 16 FEATURES].personAuc` against data.
 */
export function resolve(data, path) {
  const parts = path.match(/[^.[\]]+|\[[^\]]+\]/g) ?? [];
  let cur = data;
  for (const part of parts) {
    if (cur == null) break;
    if (part.startsWith("[")) {
      const [key, value] = part.slice(1, -1).split("=");
      if (!Array.isArray(cur)) throw new Error(`${path}: ${part} applied to a non-array`);
      cur = cur.find((item) => String(item[key]) === value);
    } else {
      cur = cur[part];
    }
  }
  if (cur === undefined) throw new Error(`no value at ${path}`);
  return cur;
}

const TOKEN = /\{\{\s*([^|}]+?)\s*\|\s*(\w+)\s*\}\}/g;

/** Replace every {{path|format}} token in html with the formatted value from data. Throws on any bad token. */
export function fillTokens(html, data) {
  const out = html.replace(TOKEN, (_, path, format) => {
    const fn = formats[format];
    if (!fn) throw new Error(`unknown format "${format}" in {{${path}|${format}}}`);
    try {
      return fn(resolve(data, path));
    } catch (err) {
      throw new Error(`{{${path}|${format}}}: ${err.message}`);
    }
  });
  if (out.includes("{{")) throw new Error(`unfilled token near: ${out.slice(out.indexOf("{{"), out.indexOf("{{") + 60)}`);
  return out;
}
