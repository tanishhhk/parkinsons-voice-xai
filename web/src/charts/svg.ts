// Small SVG helpers shared by the charts. No dependencies.
const NS = "http://www.w3.org/2000/svg";

export type Attrs = Record<string, string | number>;

export function svgEl<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Attrs = {},
  parent?: Element,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  parent?.appendChild(node);
  return node;
}

export function svgText(parent: Element, text: string, attrs: Attrs = {}): SVGTextElement {
  const t = svgEl("text", attrs, parent);
  t.textContent = text;
  return t;
}

/** Adds a native tooltip (and accessible name) to an SVG element. */
export function svgTitle(parent: Element, text: string): void {
  const t = svgEl("title", {});
  t.textContent = text;
  parent.prepend(t);
}

/** Linear scale: maps a value in `domain` to a position in `range`. */
export function linear(domain: [number, number], range: [number, number]): (v: number) => number {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0;
  return (v) => (span === 0 ? r0 : r0 + ((v - d0) / span) * (r1 - r0));
}

/** Evenly spaced tick values from `start` to `end` inclusive, rounded to dodge float noise. */
export function ticks(start: number, end: number, step: number): number[] {
  const out: number[] = [];
  const n = Math.round((end - start) / step);
  for (let i = 0; i <= n; i++) out.push(Math.round((start + i * step) * 1e6) / 1e6);
  return out;
}
