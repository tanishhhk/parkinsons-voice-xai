// Reads ../results and writes src/data.json. Every number on the site comes from here.
import { readFileSync, writeFileSync } from "node:fs";

const RESULTS = new URL("../../results/", import.meta.url);
const OUT = new URL("../src/data.json", import.meta.url);
const TOTAL_RUNS = 40;

const text = (path) => readFileSync(new URL(path, RESULTS), "utf8");
const json = (path) => JSON.parse(text(path));

// Split on commas that are not inside {...} (cv_folds.csv best_params).
function splitLine(line) {
  const cells = [];
  let depth = 0;
  let cell = "";
  for (const ch of line) {
    if (ch === "{") depth++;
    if (ch === "}") depth--;
    if (ch === "," && depth === 0) {
      cells.push(cell);
      cell = "";
    } else cell += ch;
  }
  cells.push(cell);
  return cells;
}

const toValue = (s) => (s.trim() !== "" && Number.isFinite(Number(s)) ? Number(s) : s);

function parseCsv(path) {
  const [head, ...lines] = text(path).split(/\r?\n/).filter((l) => l.trim() !== "");
  const cols = splitLine(head);
  return lines.map((line) => {
    const cells = splitLine(line);
    return Object.fromEntries(cols.map((c, i) => [c, toValue(cells[i] ?? "")]));
  });
}

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

// --- dataset
const meta = json("dataset_meta.json");
const dataset = {
  records: meta.n_records,
  people: meta.n_subjects,
  pd: meta.subject_class_balance["1"],
  healthy: meta.subject_class_balance["0"],
  recordsPerPerson: [meta.records_per_subject.min, meta.records_per_subject.max],
  featuresBefore: meta.n_features + Object.keys(meta.dropped_features).length,
  featuresAfter: meta.n_features,
  dropped: Object.entries(meta.dropped_features).map(([feature, d]) => ({
    feature,
    twin: d.correlated_with,
    r: d.r,
  })),
};
const prevalence = meta.record_class_balance["1"] / meta.n_records;

// --- leaky vs honest split
const analysis = json("analysis_report.json");
const splitModels = parseCsv("tables/leakage_optimism_roc_auc.csv")
  .filter((r) => r.model !== "Baseline (majority)")
  .map((r) => ({
    model: r.model,
    recording: r.roc_auc_record,
    person: r.roc_auc_subject,
    gap: r.optimism,
    ciLow: r.ci95_low,
    ciHigh: r.ci95_high,
    p: r.p_value,
  }));
const byModel = (name) => splitModels.find((m) => m.model === name);
const bestLeaky = analysis.ranking.best_under_leaky_protocol;
const bestHonest = analysis.ranking.best_under_honest_protocol;
const split = {
  models: splitModels,
  meanGap: analysis.roc_auc.mean_optimism_excl_baseline,
  significant: splitModels.filter((m) => m.p < 0.05).length,
  total: splitModels.length,
  bestLeaky: { model: bestLeaky, recording: byModel(bestLeaky).recording },
  bestHonest: { model: bestHonest, person: byModel(bestHonest).person },
  rankCorr: analysis.ranking.spearman_rank_correlation_between_protocols,
};

// --- benchmark: person-split ROC-AUC per repeat/fold
const folds = parseCsv("tables/cv_folds.csv").filter((r) => r.protocol === "subject");
const benchmark = [...new Set(folds.map((r) => r.model))]
  .map((model) => {
    const runs = folds
      .filter((r) => r.model === model && typeof r.roc_auc === "number")
      .map((r) => r.roc_auc);
    return { model, runs, mean: mean(runs), totalRuns: TOTAL_RUNS, undefinedRuns: TOTAL_RUNS - runs.length };
  })
  .sort((a, b) => b.mean - a.mean);

// --- single features
const singleFeature = parseCsv("tables/single_feature_cv.csv").map((r) => ({
  model: r.model,
  features: r.n_features,
  recordingAuc: r.recording_auc,
  personAuc: r.speaker_auc,
  personSd: r.speaker_sd,
}));

// --- SHAP stability joined to plain names
const plainName = Object.fromEntries(
  parseCsv("tables/biomarker_model_attribution.csv").map((r) => [r.feature, r.plain_name]),
);
const shap = parseCsv("tables/shap_stability.csv").map((r) => ({
  feature: r[""],
  plainName: plainName[r[""]],
  meanAbs: r.mean_abs_shap,
  rankMean: r.mean_rank,
  rankSd: r.rank_sd,
}));

// --- biomarkers
const biomarkers = parseCsv("tables/biomarker_univariate.csv").map((r) => ({
  feature: r.feature,
  plainName: r.plain_name,
  meaning: r.meaning,
  direction: r.direction,
  d: r.cohens_d,
  auc: r.auc_single_feature,
}));

// --- people
const people = parseCsv("tables/patient_cards.csv").map((r) => ({
  subject: r.subject,
  truth: r.y_true,
  prob: r.y_prob,
  outcome: r.outcome,
}));

// --- screening (speaker-aggregated like _agg_speaker in src/screening.py)
const screeningReport = json("screening_report.json");
const groups = new Map();
for (const r of parseCsv("tables/screening_oof.csv")) {
  const key = `${r.repeat}|${r.subject}`;
  const g = groups.get(key) ?? { repeat: r.repeat, subject: r.subject, truth: r.y_true, threshold: r.threshold, probs: [] };
  g.probs.push(r.y_prob);
  groups.set(key, g);
}
const persons = [...groups.values()].map(({ probs, ...g }) => ({ ...g, prob: mean(probs) }));

// --- explainability
const expl = json("explainability_report.json");
const explain = {
  shapAveraged: expl.mean_abs_shap_classmean,
  shapCorrect: expl.mean_abs_shap_positive_class,
  stableTop: shap.filter((s) => s.rankMean < 4).map((s) => s.feature),
};

const data = {
  dataset,
  split,
  benchmark,
  singleFeature,
  shap,
  biomarkers,
  people,
  screening: {
    report: screeningReport.speaker_level,
    recordingLevel: screeningReport.recording_level,
    thresholdMean: screeningReport.threshold_mean,
    persons,
  },
  explain,
  prevalence,
};

writeFileSync(OUT, JSON.stringify(data, null, 2) + "\n");
console.log(`wrote ${OUT.pathname}`);
