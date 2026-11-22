import "./style.css";
import data from "./data.json";
import { renderBenchmark } from "./charts/benchmark";
import { renderBiomarkers } from "./charts/biomarkers";
import { renderPeople } from "./charts/people";
import { renderShap } from "./charts/shap";
import { renderSingleFeature } from "./charts/singleFeature";
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

const shapEl = document.querySelector<HTMLElement>('[data-chart="shap"]');
if (shapEl) renderShap(shapEl, data.shap, data.explain.stableTop, data.explain.stabilitySplits);

const singleEl = document.querySelector<HTMLElement>('[data-chart="single-feature"]');
if (singleEl) renderSingleFeature(singleEl, data.singleFeature);

const bioEl = document.querySelector<HTMLElement>('[data-chart="biomarkers"]');
if (bioEl) renderBiomarkers(bioEl, data.biomarkers);

const peopleEl = document.querySelector<HTMLElement>('[data-chart="people"]');
if (peopleEl) renderPeople(peopleEl, data.people, data.screening.thresholdMean);
