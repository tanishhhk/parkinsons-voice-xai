import { dp2, dp3, int } from "../format";
import { linear, svgEl, svgText, svgTitle, ticks } from "./svg";

export type SplitMode = "recording" | "person";

export interface SplitModel {
  model: string;
  recording: number;
  person: number;
  gap: number;
  ciLow: number;
  ciHigh: number;
  p: number;
}
export interface SplitData {
  models: SplitModel[];
  meanGap: number;
  significant: number;
  total: number;
  bestLeaky: { model: string; recording: number };
  bestHonest: { model: string; person: number };
  rankCorr: number;
}

const ROW_H = 54;
const TOP = 8;
const PAD_L = 22;
const PAD_R = 46;
const AXIS_H = 44;

const MODES: { mode: SplitMode; label: string }[] = [
  { mode: "recording", label: "Random split by recording" },
  { mode: "person", label: "Split by person" },
];

interface State {
  mode: SplitMode;
  rows: Map<string, SVGGElement>;
  svg: SVGSVGElement | null;
  buttons: HTMLButtonElement[];
  width: number;
}
const states = new WeakMap<HTMLElement, State>();

/**
 * Dumbbell chart: one row per model, a tulip dot for ROC-AUC under the random split by
 * recording and a slate dot for the split by person, joined by a line (the inflation).
 * The toggle chooses which split orders the rows and is emphasised.
 */
export function renderSplit(el: HTMLElement, split: SplitData, mode: SplitMode = "recording"): void {
  let st = states.get(el);
  if (!st) {
    st = { mode, rows: new Map(), svg: null, buttons: [], width: 0 };
    states.set(el, st);
    build(el, split, st);
    const state = st;
    new ResizeObserver(() => {
      const w = Math.round(el.clientWidth);
      if (w > 0 && w !== state.width) drawPlot(el, split, state);
    }).observe(el);
  }
  setMode(el, split, st, mode, false);
}

function build(el: HTMLElement, split: SplitData, st: State): void {
  el.replaceChildren();
  el.setAttribute("role", "group");
  el.setAttribute(
    "aria-label",
    `ROC-AUC for ${int(split.total)} models under a random split by recording and a split by person`,
  );

  const bar = document.createElement("div");
  bar.className = "toggle";
  bar.setAttribute("role", "radiogroup");
  bar.setAttribute("aria-label", "Highlight and order by split");
  for (const { mode, label } of MODES) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "toggle-btn";
    b.setAttribute("role", "radio");
    b.dataset.mode = mode;
    b.textContent = label;
    b.addEventListener("click", () => setMode(el, split, st, mode, true));
    b.addEventListener("keydown", (e) => {
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) return;
      e.preventDefault();
      const next = MODES.find((m) => m.mode !== st.mode)!.mode;
      setMode(el, split, st, next, true, true);
    });
    bar.appendChild(b);
    st.buttons.push(b);
  }
  el.appendChild(bar);

  const summary = document.createElement("p");
  summary.className = "chart-summary";
  summary.setAttribute("aria-live", "polite");
  el.appendChild(summary);

  const keys = document.createElement("p");
  keys.className = "chart-keys";
  keys.innerHTML =
    '<span class="ck ck-recording"><i></i>Random split by recording</span>' +
    '<span class="ck ck-person"><i></i>Split by person</span>';
  el.appendChild(keys);

  const holder = document.createElement("div");
  holder.className = "plot";
  el.appendChild(holder);

  const note = document.createElement("p");
  note.className = "chart-note";
  note.textContent = `* The gap is statistically significant for ${int(split.significant)} of ${int(split.total)} models. Hover a row for its 95% interval.`;
  el.appendChild(note);
}

function setMode(
  el: HTMLElement,
  split: SplitData,
  st: State,
  mode: SplitMode,
  animate: boolean,
  focus = false,
): void {
  st.mode = mode;
  for (const b of st.buttons) {
    const on = b.dataset.mode === mode;
    b.setAttribute("aria-checked", String(on));
    b.tabIndex = on ? 0 : -1;
    if (on && focus) b.focus();
  }
  const best = mode === "recording" ? split.bestLeaky.model : split.bestHonest.model;
  const val = mode === "recording" ? split.bestLeaky.recording : split.bestHonest.person;
  const summary = el.querySelector(".chart-summary");
  if (summary) {
    summary.textContent = `${mode === "recording" ? "Random split" : "Split by person"}: ${best} ranks first at ${dp3(val)}.`;
  }
  if (!st.svg) drawPlot(el, split, st);
  else place(split, st, animate);
}

function place(split: SplitData, st: State, animate: boolean): void {
  const svg = st.svg!;
  svg.dataset.mode = st.mode;
  svg.classList.toggle("no-anim", !animate);
  const sorted = [...split.models].sort((a, b) => b[st.mode] - a[st.mode]);
  sorted.forEach((m, i) => {
    const g = st.rows.get(m.model)!;
    g.style.transform = `translateY(${TOP + i * ROW_H}px)`;
    g.classList.toggle("best", i === 0);
  });
}

function drawPlot(el: HTMLElement, split: SplitData, st: State): void {
  const holder = el.querySelector<HTMLElement>(".plot")!;
  const W = Math.max(280, Math.round(el.clientWidth) || 320);
  st.width = W;
  const n = split.models.length;
  const H = TOP + n * ROW_H + AXIS_H;
  const x = linear([0.5, 1], [PAD_L, W - PAD_R]);

  holder.replaceChildren();
  st.rows.clear();
  const svg = svgEl("svg", {
    viewBox: `0 0 ${W} ${H}`,
    width: W,
    height: H,
    class: "split-svg",
    role: "img",
    "aria-label": `Dumbbell chart. All ${int(split.total)} models score lower under the split by person than under the random split by recording, by ${dp3(split.meanGap)} ROC-AUC on average. The top model changes from ${split.bestLeaky.model} at ${dp3(split.bestLeaky.recording)} to ${split.bestHonest.model} at ${dp3(split.bestHonest.person)}.`,
  });
  holder.appendChild(svg);
  st.svg = svg;

  const axisY = TOP + n * ROW_H;
  for (const t of ticks(0.5, 1, 0.1)) {
    svgEl("line", { x1: x(t), x2: x(t), y1: TOP, y2: axisY, class: "grid" }, svg);
    svgText(svg, dp2(t), { x: x(t), y: axisY + 16, class: "tick", "text-anchor": "middle" });
  }
  svgText(svg, "ROC-AUC per recording (0.50 is a coin flip)", { x: 0, y: axisY + 36, class: "axis-title" });

  for (const m of split.models) {
    const g = svgEl("g", { class: "row", "data-model": m.model }, svg);
    svgTitle(
      g,
      `${m.model}: ${dp3(m.recording)} random split, ${dp3(m.person)} split by person. Gap ${dp3(m.gap)} (95% interval ${dp3(m.ciLow)} to ${dp3(m.ciHigh)}).`,
    );
    svgText(g, m.model, { x: 0, y: 14, class: "name" });
    svgText(g, `gap ${dp3(m.gap)}${m.p < 0.05 ? "*" : ""}`, {
      x: W - PAD_R + 20,
      y: 14,
      class: "gap",
      "text-anchor": "end",
    });
    svgEl("line", { x1: x(m.person), x2: x(m.recording), y1: 36, y2: 36, class: "conn" }, g);
    svgEl("circle", { cx: x(m.person), cy: 36, r: 6, class: "dot dot-person" }, g);
    svgEl("circle", { cx: x(m.recording), cy: 36, r: 6, class: "dot dot-recording" }, g);
    svgText(g, dp3(m.person), { x: x(m.person) - 11, y: 40, class: "val val-person", "text-anchor": "end" });
    svgText(g, dp3(m.recording), { x: x(m.recording) + 11, y: 40, class: "val val-recording" });
    st.rows.set(m.model, g);
  }
  place(split, st, false);
}
