import { dp2, dp3, int } from "../format";
import { linear, svgEl, svgText, svgTitle, ticks } from "./svg";

export interface BiomarkerRow {
  feature: string;
  plainName: string;
  meaning: string;
  direction: string;
  d: number;
  auc: number;
}

/** Cohen's conventional threshold for a "large" effect (a definition, not a result). */
const LARGE_EFFECT = 0.8;

const ROW_H = 46;
const TOP = 24;
const PAD_R = 150;
const AXIS_H = 44;

/** Bars show |Cohen's d|, sorted by it. The direction and single-measurement AUC are stated in text. */
export function renderBiomarkers(el: HTMLElement, data: BiomarkerRow[]): void {
  let width = 0;
  const rows = [...data].sort((a, b) => Math.abs(b.d) - Math.abs(a.d));
  const maxD = Math.max(...rows.map((r) => Math.abs(r.d)));
  const top = Math.ceil(maxD * 2) / 2;

  const draw = () => {
    width = Math.round(el.clientWidth) || 320;
    el.replaceChildren();
    el.setAttribute("role", "group");

    const keys = document.createElement("p");
    keys.className = "chart-keys";
    keys.innerHTML =
      '<span class="ck ck-large"><i></i>Large effect (0.8 or more)</span><span class="ck ck-small"><i></i>Smaller effect</span>';
    el.appendChild(keys);

    const W = Math.max(280, width);
    const x = linear([0, top], [0, W - PAD_R]);
    const axisY = TOP + rows.length * ROW_H;
    const H = axisY + AXIS_H;
    const largeCount = rows.filter((r) => Math.abs(r.d) >= LARGE_EFFECT).length;
    const svg = svgEl("svg", {
      viewBox: `0 0 ${W} ${H}`,
      width: W,
      height: H,
      class: "bar-svg bio-svg",
      role: "img",
      "aria-label": `Bar chart of effect size (Cohen's d) for ${int(rows.length)} voice measurements. ${rows[0].plainName} is largest at ${dp2(Math.abs(rows[0].d))}. ${int(largeCount)} reach the conventional large-effect line of ${dp2(LARGE_EFFECT)}.`,
    });
    el.appendChild(svg);

    for (const t of ticks(0, top, W < 480 ? 1 : 0.5)) {
      svgEl("line", { x1: x(t), x2: x(t), y1: TOP - 4, y2: axisY, class: "grid" }, svg);
      svgText(svg, dp2(t), { x: x(t), y: axisY + 16, class: "tick", "text-anchor": x(t) < 12 ? "start" : "middle" });
    }
    svgText(svg, "Effect size: group gap in standard deviations (Cohen's d)", {
      x: 0,
      y: axisY + 36,
      class: "axis-title",
    });
    svgEl("line", { x1: x(LARGE_EFFECT), x2: x(LARGE_EFFECT), y1: TOP - 14, y2: axisY, class: "chance" }, svg);
    svgText(svg, `${dp2(LARGE_EFFECT)}, a large effect`, {
      x: x(LARGE_EFFECT) + 6,
      y: TOP - 12,
      class: "chance-label",
    });

    rows.forEach((r, i) => {
      const y0 = TOP + i * ROW_H;
      const ad = Math.abs(r.d);
      const g = svgEl("g", { class: `bar-row${ad >= LARGE_EFFECT ? " is-large" : ""}`, "data-feature": r.feature }, svg);
      svgTitle(
        g,
        `${r.plainName} (${r.feature}): ${r.meaning}. ${r.direction}. Effect size ${dp2(ad)}; score alone ${dp3(r.auc)}.`,
      );
      const name = svgText(g, r.plainName, { x: 0, y: y0 + 14, class: "name" });
      svgEl("tspan", { class: "code" }, name).textContent = ` ${r.feature}`;
      svgText(g, `alone ${dp3(r.auc)}`, { x: W, y: y0 + 14, class: "aux", "text-anchor": "end" });
      svgEl("rect", { x: 0, y: y0 + 22, width: Math.max(2, x(ad)), height: 11, rx: 2, class: "bar" }, g);
      svgText(g, dp2(ad), { x: x(ad) + 8, y: y0 + 32, class: "val" });
      svgText(g, r.direction.replace(/ PD$/, " Parkinson's"), { x: x(ad) + 42, y: y0 + 32, class: "dir" });
    });

    const note = document.createElement("p");
    note.className = "chart-note";
    note.textContent =
      '"alone" is the ROC-AUC per person when that single measurement is used as the score, with no model, scored in whichever direction separates better. Rows are sorted by effect size.';
    el.appendChild(note);
  };

  draw();
  new ResizeObserver(() => {
    const w = Math.round(el.clientWidth);
    if (w > 0 && w !== width) draw();
  }).observe(el);
}
