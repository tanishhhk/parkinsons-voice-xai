import "./style.css";
import data from "./data.json";
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
