import { dp2, dp3, int } from "../format";
import { linear, svgEl, svgText, svgTitle, ticks } from "./svg";

export interface BenchmarkModel {
  model: string;
  runs: number[];
  mean: number;
  totalRuns: number;
  undefinedRuns: number;
}

const ROW_H = 52;
const TOP = 26;
const PAD_L = 22;
const PAD_R = 22;
const AXIS_H = 44;

/** Deterministic vertical jitter in [-1, 1] from the run index (golden-ratio sequence). */
function jitter(i: number): number {
  return ((i * 0.618033988749895) % 1) * 2 - 1;
}

/**
 * Strip chart: one row per model with a dot for every test run, a bar at the mean, and a
 * dashed line at 0.5 (chance). The legend and note state how many runs each average covers.
 */
export function renderBenchmark(el: HTMLElement, data: BenchmarkModel[]): void {
  let width = 0;
  const first = data[0];
  const ran = first.runs.length;

  const draw = () => {
    width = Math.round(el.clientWidth) || 320;
    el.replaceChildren();
    el.setAttribute("role", "group");

    const best = data[0];
    el.setAttribute(
      "aria-label",
      `ROC-AUC of every test run for ${int(data.length)} models. ${best.model} is highest, averaging ${dp3(best.mean)} over ${int(ran)} of ${int(first.totalRuns)} runs.`,
    );

    const keys = document.createElement("p");
    keys.className = "chart-keys";
    keys.innerHTML =
      '<span class="ck ck-run"><i></i>One test run</span>' +
      '<span class="ck ck-mean"><i></i>Average</span>' +
      '<span class="ck ck-chance"><i></i>0.50, a coin flip</span>';
    el.appendChild(keys);

    const W = Math.max(280, width);
    const lo = Math.floor(Math.min(...data.flatMap((d) => d.runs)) * 10) / 10;
    const x = linear([lo, 1], [PAD_L, W - PAD_R]);
    const H = TOP + data.length * ROW_H + AXIS_H;
    const svg = svgEl("svg", {
      viewBox: `0 0 ${W} ${H}`,
      width: W,
      height: H,
      class: "bench-svg",
      role: "img",
      "aria-label": `Strip chart of ROC-AUC per test run. ${data
        .map((d) => `${d.model} ${dp3(d.mean)}`)
        .join(", ")}.`,
    });
    el.appendChild(svg);

    const axisY = TOP + data.length * ROW_H;
    for (const t of ticks(lo, 1, 0.1)) {
      svgEl("line", { x1: x(t), x2: x(t), y1: TOP - 4, y2: axisY, class: "grid" }, svg);
      svgText(svg, dp2(t), { x: x(t), y: axisY + 16, class: "tick", "text-anchor": "middle" });
    }
    svgText(svg, "ROC-AUC per recording, one dot per test run", { x: 0, y: axisY + 36, class: "axis-title" });
    svgEl("line", { x1: x(0.5), x2: x(0.5), y1: TOP - 14, y2: axisY, class: "chance" }, svg);
    svgText(svg, "0.50", { x: x(0.5), y: TOP - 18, class: "chance-label", "text-anchor": "middle" });

    data.forEach((d, i) => {
      const y0 = TOP + i * ROW_H;
      const g = svgEl("g", { class: "bench-row", "data-model": d.model }, svg);
      svgTitle(
        g,
        `${d.model}: average ${dp3(d.mean)} over ${int(d.runs.length)} of ${int(d.totalRuns)} runs (${int(d.undefinedRuns)} undefined).`,
      );
      svgText(g, d.model, { x: 0, y: y0 + 14, class: "name" });
      svgText(g, dp3(d.mean), { x: W - PAD_R, y: y0 + 14, class: "mean-val", "text-anchor": "end" });
      const cy = y0 + 34;
      d.runs.forEach((v, k) => {
        svgEl("circle", { cx: x(v), cy: cy + jitter(k) * 8, r: 3, class: "run" }, g);
      });
      svgEl("rect", { x: x(d.mean) - 1.5, y: cy - 13, width: 3, height: 26, rx: 1.5, class: "mean" }, g);
    });

    const note = document.createElement("p");
    note.className = "chart-note";
    note.textContent = `Each model averages ${int(ran)} of ${int(first.totalRuns)} runs: in ${int(first.undefinedRuns)} runs the test group held no healthy speakers, so ROC-AUC is undefined.`;
    el.appendChild(note);
  };

  draw();
  new ResizeObserver(() => {
    const w = Math.round(el.clientWidth);
    if (w > 0 && w !== width) draw();
  }).observe(el);
}
