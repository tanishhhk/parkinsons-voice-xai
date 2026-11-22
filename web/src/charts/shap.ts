import { dp1, dp3, int } from "../format";
import { linear, svgEl, svgText, svgTitle } from "./svg";

export interface ShapRow {
  feature: string;
  plainName: string;
  meanAbs: number;
  rankMean: number;
  rankSd: number;
}

const ROW_H = 44;
const ROW_H_NARROW = 58;
const NARROW = 480;
const TOP = 6;
const PAD_R = 138;

/**
 * Mean |SHAP| per measurement as horizontal bars. Each row also states its average rank and the
 * spread of that rank across the stability splits; the stable pair is tagged in text as well as colour.
 */
export function renderShap(el: HTMLElement, rows: ShapRow[], stableTop: string[], splits: number): void {
  let width = 0;
  const sorted = [...rows].sort((a, b) => b.meanAbs - a.meanAbs);

  const draw = () => {
    width = Math.round(el.clientWidth) || 320;
    el.replaceChildren();
    el.setAttribute("role", "group");

    const keys = document.createElement("p");
    keys.className = "chart-keys";
    keys.innerHTML =
      '<span class="ck ck-stable"><i></i>Stable top two</span><span class="ck ck-other"><i></i>Other measurements</span>';
    el.appendChild(keys);

    const W = Math.max(280, width);
    const narrow = W < NARROW;
    const rh = narrow ? ROW_H_NARROW : ROW_H;
    const H = TOP + sorted.length * rh + 8;
    const x = linear([0, sorted[0].meanAbs], [0, W - PAD_R]);
    const svg = svgEl("svg", {
      viewBox: `0 0 ${W} ${H}`,
      width: W,
      height: H,
      class: "bar-svg shap-svg",
      role: "img",
      "aria-label": `Bar chart of mean absolute SHAP value for ${int(sorted.length)} measurements. ${sorted[0].plainName} (${sorted[0].feature}) is highest at ${dp3(sorted[0].meanAbs)}. ${stableTop.join(" and ")} keep their top places across ${int(splits)} splits.`,
    });
    el.appendChild(svg);

    sorted.forEach((r, i) => {
      const y0 = TOP + i * rh;
      const stable = stableTop.includes(r.feature);
      const g = svgEl("g", { class: `bar-row${stable ? " is-stable" : ""}`, "data-feature": r.feature }, svg);
      svgTitle(
        g,
        `${r.plainName} (${r.feature}): mean |SHAP| ${dp3(r.meanAbs)}. Average rank ${dp1(r.rankMean)}, spread ${dp1(r.rankSd)} across ${int(splits)} splits.`,
      );
      const name = svgText(g, r.plainName, { x: 0, y: y0 + 14, class: "name" });
      svgEl("tspan", { class: "code" }, name).textContent = ` ${r.feature}`;
      const by = narrow ? 14 : 0;
      svgText(g, `rank ${dp1(r.rankMean)} ± ${dp1(r.rankSd)}`, {
        x: narrow ? 0 : W,
        y: y0 + 14 + by,
        class: "aux",
        "text-anchor": narrow ? "start" : "end",
      });
      svgEl("rect", { x: 0, y: y0 + 22 + by, width: Math.max(2, x(r.meanAbs)), height: 11, rx: 2, class: "bar" }, g);
      svgText(g, dp3(r.meanAbs), { x: x(r.meanAbs) + 8, y: y0 + 32 + by, class: "val" });
      if (stable) svgText(g, "stable top two", { x: x(r.meanAbs) + 46, y: y0 + 32 + by, class: "tag" });
    });

    const note = document.createElement("p");
    note.className = "chart-note";
    note.textContent = `Bar length is the average size of each measurement's push on a prediction. "rank" is its place by that size, averaged over ${int(splits)} different splits, ± how far it moved. Only ${stableTop.join(" and ")} hold their place.`;
    el.appendChild(note);
  };

  draw();
  new ResizeObserver(() => {
    const w = Math.round(el.clientWidth);
    if (w > 0 && w !== width) draw();
  }).observe(el);
}
