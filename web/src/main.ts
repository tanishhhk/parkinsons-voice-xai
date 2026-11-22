import "./style.css";
import data from "./data.json";
import { renderBenchmark } from "./charts/benchmark";
import { renderSplit } from "./charts/split";
import { drawCohort, drawTrace, trackSections } from "./page";

// Chart modules mount into the [data-chart] elements in index.html (added by later tasks).

const trace = document.querySelector<SVGSVGElement>("#trace");
if (trace) {
  drawTrace(trace);
  let width = trace.clientWidth;
  new ResizeObserver(() => {
    if (Math.abs(trace.clientWidth - width) < 40) return;
    width = trace.clientWidth;
    drawTrace(trace, false);
  }).observe(trace);
}

const cohort = document.querySelector<HTMLElement>("#cohort");
if (cohort) drawCohort(cohort, data.dataset.pd, data.dataset.healthy);

const nav = document.querySelector<HTMLElement>(".toc");
if (nav) trackSections(nav);

const splitEl = document.querySelector<HTMLElement>('[data-chart="split"]');
if (splitEl) renderSplit(splitEl, data.split, "recording");

const benchEl = document.querySelector<HTMLElement>('[data-chart="benchmark"]');
if (benchEl) renderBenchmark(benchEl, data.benchmark);
