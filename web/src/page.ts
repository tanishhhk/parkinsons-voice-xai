// Static page furniture: the hero trace, the cohort grid and the section navigation state.

const SVG = "http://www.w3.org/2000/svg";

/** Small seeded generator so the illustration is the same on every visit. */
function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/**
 * Illustration of a held vowel: quasi-periodic glottal cycles that start steady, then gain
 * cycle-to-cycle variation in period (jitter) and height (shimmer). Not drawn from the dataset.
 */
export function drawTrace(svg: SVGSVGElement, animate = true): void {
  // One user unit per CSS pixel, so cycles keep the same spacing on a phone and a wide screen.
  const W = Math.round(svg.clientWidth) || 1600;
  svg.setAttribute("viewBox", `0 0 ${W} 200`);
  const H = 200;
  const mid = H / 2;
  const rand = seeded(174);
  const base = 30;
  const points: string[] = [];
  let x = 0;
  while (x < W) {
    const u = smooth(0.3 * W, 0.92 * W, x);
    const period = base * (1 + u * 0.55 * (rand() - 0.5) + u * 0.22 * Math.sin(x / 120));
    const amp = 62 * (1 + u * 0.95 * (rand() - 0.5));
    const steps = 12;
    for (let s = 0; s < steps; s++) {
      const p = s / steps;
      const w = Math.sin(2 * Math.PI * p) + 0.45 * Math.sin(4 * Math.PI * p + 0.9) + 0.2 * Math.sin(6 * Math.PI * p + 1.7);
      const y = mid - (amp / 1.45) * w + u * 5 * (rand() - 0.5);
      points.push(`${(x + p * period).toFixed(1)},${Math.min(H - 2, Math.max(2, y)).toFixed(1)}`);
    }
    x += period;
  }
  const axis = document.createElementNS(SVG, "line");
  axis.setAttribute("class", "axis");
  axis.setAttribute("x1", "0");
  axis.setAttribute("x2", String(W));
  axis.setAttribute("y1", String(mid));
  axis.setAttribute("y2", String(mid));
  const wave = document.createElementNS(SVG, "polyline");
  wave.setAttribute("class", animate ? "wave" : "wave drawn");
  wave.setAttribute("pathLength", "1");
  wave.setAttribute("points", points.join(" "));
  svg.replaceChildren(axis, wave);
}

/** One mark per person: filled squares for Parkinson's, rings for healthy. */
export function drawCohort(el: HTMLElement, pd: number, healthy: number): void {
  const marks: HTMLElement[] = [];
  for (let i = 0; i < pd + healthy; i++) {
    const m = document.createElement("i");
    m.className = i < pd ? "pd" : "hc";
    marks.push(m);
  }
  el.setAttribute("role", "img");
  el.setAttribute("aria-label", `${pd} people with Parkinson's and ${healthy} healthy people`);
  el.replaceChildren(...marks);
}

/** Mark the nav link of the section currently in view. */
export function trackSections(nav: HTMLElement): void {
  const links = new Map<string, HTMLAnchorElement>();
  nav.querySelectorAll<HTMLAnchorElement>("a[href^='#']").forEach((a) => links.set(a.hash.slice(1), a));
  const observer = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        links.forEach((a) => a.removeAttribute("aria-current"));
        const link = links.get(e.target.id);
        if (link) {
          link.setAttribute("aria-current", "true");
          const list = link.closest("ol");
          if (list) list.scrollLeft = link.offsetLeft - (list.clientWidth - link.offsetWidth) / 2;
        }
      }
    },
    { rootMargin: "-40% 0px -55% 0px" },
  );
  links.forEach((_, id) => {
    const section = document.getElementById(id);
    if (section) observer.observe(section);
  });
}
