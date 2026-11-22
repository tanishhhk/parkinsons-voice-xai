import { dp2, dp3, int } from "../format";
import { linear, svgEl, svgText, svgTitle, ticks } from "./svg";

export interface SingleFeatureRow {
  model: string;
  features: number;
  recordingAuc: number;
  personAuc: number;
  personSd: number;
}

const ROW_H = 46;
const TOP = 6;
const PAD_L = 4;
const PAD_R = 22;
const AXIS_H = 44;

function label(r: SingleFeatureRow): string {
  return r.features === 1 ? `${r.model} alone` : `All ${int(r.features)} features`;
}

/**
 * Person-level ROC-AUC per model with a whisker for the SD between runs. The takeaway line
 * compares the spread of the best single measurement with the all-feature model, computed here.
 */
export function renderSingleFeature(el: HTMLElement, data: SingleFeatureRow[]): void {
  let width = 0;
  const rows = [...data].sort((a, b) => b.personAuc - a.personAuc);
  const all = data.reduce((a, b) => (b.features > a.features ? b : a));
  const best = data.filter((d) => d.features === 1).reduce((a, b) => (b.personAuc > a.personAuc ? b : a));
  const ratio = Math.round(all.personSd / best.personSd);

  const draw = () => {
    width = Math.round(el.clientWidth) || 320;
    el.replaceChildren();
    el.setAttribute("role", "group");

    const callout = document.createElement("p");
    callout.className = "chart-callout";
    const strong = document.createElement("strong");
    strong.textContent = `${int(ratio)}× less variation`;
    callout.append(
      `${best.model} alone scores ${dp3(best.personAuc)} against ${dp3(all.personAuc)} for all ${int(all.features)} features, with `,
      strong,
      ` between runs (SD ${dp3(best.personSd)} against ${dp3(all.personSd)}).`,
    );
    el.appendChild(callout);

    const keys = document.createElement("p");
    keys.className = "chart-keys";
    keys.innerHTML =
      '<span class="ck ck-single"><i></i>One measurement</span><span class="ck ck-all"><i></i>All features</span><span class="ck ck-sd"><i></i>± SD between runs</span>';
    el.appendChild(keys);

    const W = Math.max(280, width);
    const lo = 0.5;
    const x = linear([lo, 1], [PAD_L, W - PAD_R]);
    const axisY = TOP + rows.length * ROW_H;
    const H = axisY + AXIS_H;
    const svg = svgEl("svg", {
      viewBox: `0 0 ${W} ${H}`,
      width: W,
      height: H,
      class: "bar-svg single-svg",
      role: "img",
      "aria-label": `Bar chart of ROC-AUC per person. ${rows.map((r) => `${label(r)} ${dp3(r.personAuc)}`).join(", ")}.`,
    });
    el.appendChild(svg);

    for (const t of ticks(lo, 1, 0.1)) {
      svgEl("line", { x1: x(t), x2: x(t), y1: TOP, y2: axisY, class: "grid" }, svg);
      svgText(svg, dp2(t), { x: x(t), y: axisY + 16, class: "tick", "text-anchor": x(t) < 12 ? "start" : "middle" });
    }
    svgText(svg, "ROC-AUC per person (0.50 is a coin flip)", { x: 0, y: axisY + 36, class: "axis-title" });

    rows.forEach((r, i) => {
      const y0 = TOP + i * ROW_H;
      const many = r.features > 1;
      const g = svgEl("g", { class: `bar-row${many ? " is-all" : ""}`, "data-model": r.model }, svg);
      svgTitle(
        g,
        `${label(r)}: ROC-AUC per person ${dp3(r.personAuc)}, SD between runs ${dp3(r.personSd)}. Per recording ${dp3(r.recordingAuc)}.`,
      );
      svgText(g, label(r), { x: 0, y: y0 + 15, class: "name" });
      svgText(g, `${dp3(r.personAuc)} ± ${dp3(r.personSd)}`, {
        x: W - PAD_R,
        y: y0 + 15,
        class: "val",
        "text-anchor": "end",
      });
      svgEl("rect", { x: x(lo), y: y0 + 22, width: x(r.personAuc) - x(lo), height: 12, rx: 2, class: "bar" }, g);
      const cy = y0 + 28;
      const a = x(r.personAuc - r.personSd);
      const b = x(r.personAuc + r.personSd);
      svgEl("line", { x1: a, x2: b, y1: cy, y2: cy, class: "whisker" }, g);
      svgEl("line", { x1: a, x2: a, y1: cy - 6, y2: cy + 6, class: "whisker" }, g);
      svgEl("line", { x1: b, x2: b, y1: cy - 6, y2: cy + 6, class: "whisker" }, g);
    });
  };

  draw();
  new ResizeObserver(() => {
    const w = Math.round(el.clientWidth);
    if (w > 0 && w !== width) draw();
  }).observe(el);
}
