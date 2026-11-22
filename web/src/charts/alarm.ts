import { dp1, dp2, dp3, int, pct1 } from "../format";
import { operatingPoint, type PersonRow } from "../screening";
import { linear, svgEl, svgText, svgTitle, ticks } from "./svg";

export interface ScreeningData {
  persons: { repeat: number; subject: string; truth: number; prob: number; threshold: number }[];
  thresholdMean: number;
}
export interface CohortCounts {
  healthy: number;
  pd: number;
}

type Mode = "frozen" | number;

const PAD = { l: 52, r: 14, t: 12, b: 50 };

/** A rate as a percentage; the exact ends read "0%" and "100%"; undefined reads a dash. */
function rate(v: number | null): string {
  if (v === null || !Number.isFinite(v)) return "–";
  if (v === 0) return "0%";
  if (v === 1) return "100%";
  return pct1(v);
}
const count = (v: number): string => (Number.isFinite(v) ? dp1(v) : "–");

/** One ROC curve per repeat: (false-alarm rate, sensitivity) swept over every distinct score. */
function rocCurve(rows: PersonRow[]): [number, number][] {
  const pos = rows.filter((r) => r.truth === 1).length;
  const neg = rows.length - pos;
  if (!pos || !neg) return [];
  const sorted = [...rows].sort((a, b) => b.prob - a.prob);
  const pts: [number, number][] = [[0, 0]];
  let tp = 0;
  let fp = 0;
  for (let i = 0; i < sorted.length; i++) {
    if (sorted[i].truth === 1) tp++;
    else fp++;
    if (i === sorted.length - 1 || sorted[i + 1].prob !== sorted[i].prob) pts.push([fp / neg, tp / pos]);
  }
  return pts;
}

function curves(persons: PersonRow[]): [number, number][][] {
  const groups = new Map<number, PersonRow[]>();
  for (const p of persons) {
    const g = groups.get(p.repeat);
    if (g) g.push(p);
    else groups.set(p.repeat, [p]);
  }
  return [...groups.values()].map(rocCurve).filter((c) => c.length > 0);
}

interface State {
  mode: Mode;
  width: number;
  dot: SVGGElement | null;
  scales: { x: (v: number) => number; y: (v: number) => number } | null;
}
const states = new WeakMap<HTMLElement, State>();

/**
 * Interactive alarm level: a slider moves the threshold; the readout and a marked point on the
 * per-round ROC curves follow. The frozen thresholds as reported are the default and a reset.
 */
export function renderAlarm(el: HTMLElement, screening: ScreeningData, cohort: CohortCounts): void {
  const persons = screening.persons as PersonRow[];
  if (states.has(el)) return;
  const st: State = { mode: "frozen", width: 0, dot: null, scales: null };
  states.set(el, st);

  const frozen = operatingPoint(persons, "frozen");
  const frozenT = screening.thresholdMean;
  const frozenStep = String(Math.round(frozenT * 100) / 100);
  const rocCurves = curves(persons);
  const aucText = Number.isFinite(frozen.auc) ? dp3(frozen.auc) : "–";

  el.replaceChildren();
  el.setAttribute("role", "group");
  el.setAttribute("aria-label", "Interactive alarm threshold");

  const layout = document.createElement("div");
  layout.className = "alarm";
  el.appendChild(layout);

  const panel = document.createElement("div");
  panel.className = "alarm-panel";
  layout.appendChild(panel);

  // Slider
  const sliderBlock = document.createElement("div");
  sliderBlock.className = "alarm-slider";
  const label = document.createElement("label");
  label.htmlFor = "alarm-threshold";
  label.className = "alarm-label";
  label.innerHTML =
    '<span>Alarm level: a risk score at or above this raises the alarm</span><output class="alarm-value" for="alarm-threshold"></output>';
  const value = label.querySelector<HTMLOutputElement>("output")!;
  const track = document.createElement("div");
  track.className = "alarm-track";
  const input = document.createElement("input");
  input.type = "range";
  input.id = "alarm-threshold";
  input.min = "0";
  input.max = "1";
  input.step = "0.01";
  input.value = frozenStep;
  const mark = document.createElement("span");
  mark.className = "alarm-mark";
  mark.style.setProperty("--f", String(frozenT));
  mark.setAttribute("aria-hidden", "true");
  mark.innerHTML = `<i></i><b>frozen, per run (avg ${dp2(frozenT)})</b>`;
  track.append(input, mark);
  const scale = document.createElement("div");
  scale.className = "alarm-scale";
  scale.setAttribute("aria-hidden", "true");
  scale.innerHTML = "<span>0.00 flags everyone</span><span>1.00 flags no one</span>";
  const explain = document.createElement("p");
  explain.className = "chart-note alarm-explain";
  explain.textContent = `The reported result fixes the alarm level in advance for each test run, about ${dp2(frozenT)} on average. The slider applies one level to everyone, so even at ${dp2(frozenT)} the numbers can differ slightly.`;
  sliderBlock.append(label, track, scale, explain);
  panel.appendChild(sliderBlock);

  const reset = document.createElement("button");
  reset.type = "button";
  reset.className = "alarm-reset";
  reset.textContent = "Reported result (per-run thresholds)";
  panel.appendChild(reset);

  // Readout: visible numbers update on every tick; the live copy waits for the slider to settle.
  const readout = document.createElement("dl");
  readout.className = "alarm-readout";
  readout.innerHTML =
    '<div><dt>Sensitivity, cases caught</dt><dd class="alarm-sens"></dd></div>' +
    '<div><dt>Specificity, healthy cleared</dt><dd class="alarm-spec"></dd></div>';
  panel.appendChild(readout);
  const sensEl = readout.querySelector<HTMLElement>(".alarm-sens")!;
  const specEl = readout.querySelector<HTMLElement>(".alarm-spec")!;

  const round = document.createElement("p");
  round.className = "alarm-round";
  panel.appendChild(round);

  const live = document.createElement("p");
  live.className = "alarm-live";
  panel.appendChild(live);

  // ROC
  const rocBox = document.createElement("div");
  rocBox.className = "alarm-roc";
  const keys = document.createElement("p");
  keys.className = "chart-keys";
  keys.innerHTML =
    '<span class="ck ck-run"><i></i>One round of testing</span>' +
    '<span class="ck ck-now"><i></i>Current alarm level</span>' +
    '<span class="ck ck-frozen"><i></i>Reported (per-run levels)</span>' +
    '<span class="ck ck-chance"><i></i>Chance</span>';
  const plot = document.createElement("div");
  plot.className = "plot";
  const auc = document.createElement("p");
  auc.className = "chart-note alarm-auc";
  auc.textContent = `ROC-AUC per person (mean of ${int(rocCurves.length)} rounds): ${aucText}`;
  rocBox.append(keys, plot, auc);
  layout.appendChild(rocBox);

  const current = () => operatingPoint(persons, st.mode);

  function paint(announce: boolean): void {
    const op = current();
    const c = op.counts;
    const roundText =
      `Of the ${int(cohort.healthy)} healthy people, on average ${count(c.cleared)} are cleared and ` +
      `${count(c.falseAlarms)} get a false alarm. Of the ${int(cohort.pd)} with Parkinson's, on average ` +
      `${count(c.caught)} are caught and ${count(c.missed)} are missed.`;
    const level = st.mode === "frozen" ? "Reported result, per-run thresholds" : `Alarm level ${dp2(st.mode)}`;
    sensEl.textContent = rate(op.sensitivity);
    specEl.textContent = rate(op.specificity);
    value.textContent =
      st.mode === "frozen" ? `per-run levels (avg ${dp2(frozenT)})` : `${dp2(st.mode)}, one level for all runs`;
    input.setAttribute(
      "aria-valuetext",
      st.mode === "frozen" ? `per-run levels, average ${dp2(frozenT)}, as reported` : `${dp2(st.mode)}, one level for all runs`,
    );
    round.textContent = roundText;
    if (announce) {
      live.textContent = `${level}. Sensitivity ${rate(op.sensitivity)}, specificity ${rate(op.specificity)}. ${roundText}`;
    }
    moveDot(op);
  }

  function moveDot(op = current()): void {
    if (!st.dot || !st.scales) return;
    if (op.specificity === null || op.sensitivity === null) {
      st.dot.setAttribute("visibility", "hidden");
      return;
    }
    st.dot.setAttribute("visibility", "visible");
    st.dot.setAttribute("transform", `translate(${st.scales.x(1 - op.specificity)} ${st.scales.y(op.sensitivity)})`);
  }

  function draw(): void {
    const w = Math.round(plot.clientWidth) || 320;
    st.width = w;
    const side = Math.min(Math.max(260, w), 520);
    const H = Math.round(side * 0.92);
    const x = linear([0, 1], [PAD.l, side - PAD.r]);
    const y = linear([0, 1], [H - PAD.b, PAD.t]);
    st.scales = { x, y };
    plot.replaceChildren();
    const svg = svgEl("svg", {
      viewBox: `0 0 ${side} ${H}`,
      width: side,
      height: H,
      class: "alarm-svg",
      role: "img",
      "aria-label": `ROC curves for ${int(rocCurves.length)} rounds of testing: the share of healthy people falsely alarmed against the share of cases caught, with a dot for the current alarm level. ROC-AUC per person is ${aucText}, the mean of ${int(rocCurves.length)} rounds.`,
    });
    plot.appendChild(svg);
    for (const t of ticks(0, 1, 0.25)) {
      svgEl("line", { x1: x(t), x2: x(t), y1: y(0), y2: y(1), class: "grid" }, svg);
      svgEl("line", { x1: x(0), x2: x(1), y1: y(t), y2: y(t), class: "grid" }, svg);
      svgText(svg, dp2(t), { x: x(t), y: y(0) + 16, class: "tick", "text-anchor": "middle" });
      svgText(svg, dp2(t), { x: x(0) - 8, y: y(t) + 4, class: "tick", "text-anchor": "end" });
    }
    svgText(svg, "False alarms among healthy people", {
      x: (x(0) + x(1)) / 2,
      y: y(0) + 40,
      class: "axis-title",
      "text-anchor": "middle",
    });
    const yt = svgText(svg, "Cases caught", { x: 0, y: 0, class: "axis-title", "text-anchor": "middle" });
    yt.setAttribute("transform", `translate(12 ${(y(0) + y(1)) / 2}) rotate(-90)`);
    svgEl("line", { x1: x(0), y1: y(0), x2: x(1), y2: y(1), class: "chance" }, svg);
    for (const c of rocCurves) {
      const d = c.map(([a, b], i) => `${i ? "L" : "M"}${x(a).toFixed(1)} ${y(b).toFixed(1)}`).join("");
      svgEl("path", { d, class: "roc-run" }, svg);
    }
    if (frozen.specificity !== null && frozen.sensitivity !== null) {
      const f = svgEl(
        "circle",
        { cx: x(1 - frozen.specificity), cy: y(frozen.sensitivity), r: 9, class: "roc-frozen" },
        svg,
      );
      svgTitle(f, "Reported result, per-run thresholds");
    }
    const dot = svgEl("g", { class: "roc-now" }, svg);
    svgEl("circle", { r: 6, class: "roc-now-dot" }, dot);
    st.dot = dot;
    moveDot();
  }

  input.addEventListener("input", () => {
    st.mode = Number(input.value);
    paint(false);
  });
  input.addEventListener("change", () => {
    st.mode = Number(input.value);
    paint(true);
  });
  reset.addEventListener("click", () => {
    st.mode = "frozen";
    input.value = frozenStep;
    paint(true);
  });

  draw();
  paint(true);
  // Attach the live region only after its text is set, so nothing is announced on load.
  live.setAttribute("aria-live", "polite");
  live.setAttribute("aria-atomic", "true");
  new ResizeObserver(() => {
    const w = Math.round(plot.clientWidth);
    if (w > 0 && w !== st.width) draw();
  }).observe(plot);
}
