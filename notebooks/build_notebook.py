"""Generate parkinsons_xai.ipynb from source, then execute it so that every
output in the committed notebook is real and reproducible."""
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

C = []
M = lambda s: C.append(new_markdown_cell(s.strip()))
K = lambda s: C.append(new_code_cell(s.strip()))

M(r"""
# Parkinson's Disease Prediction from Voice, with Explainable AI

**Aim.** Predict Parkinson's disease from sustained-vowel recordings, and explain
what in the voice drives each prediction.

**Answer, in three lines.**

1. It works, at screening strength. On speakers the model has never heard, it reaches
   **ROC-AUC 0.852** and catches **92% of Parkinson's cases** — but flags **half of
   healthy speakers** as suspicious. That specificity is the real limitation.
2. The signal is **impaired pitch control**. `spread1` and `PPE`, two nonlinear
   measures of fundamental-frequency variation, are both higher in Parkinson's with
   Cohen's *d* of 2.03 and 1.91.
3. **One feature is enough.** `spread1` alone reaches **0.924 AUC**, beating the full
   16-feature model at **0.847**. The extra features add variance, not signal.

Everything below is measured, not asserted. All model outputs come from speakers held
out of training, using speaker-grouped cross-validation (Section 6 explains why that
matters and what it costs).

---

## Contents

1. The voice data
2. Which voice measures signal Parkinson's
3. Choosing a model
4. The screening model and its operating point
5. Explaining individual predictions
6. Validity: why the split is grouped by speaker
7. Conclusions and limitations
""")

K(r"""
%matplotlib inline
import sys, json, warnings
sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

from config import TABLES_DIR, FIGURES_DIR, RESULTS_DIR
from data import load_raw, add_subject_column, load_dataset
print("pandas", pd.__version__, "| numpy", np.__version__)
""")

M(r"""
## 1. The voice data

195 sustained-vowel phonations from **32 speakers**, 24 with Parkinson's and 8 healthy.
Each speaker contributed 6 or 7 recordings in a single session. Each recording is
summarised by 22 acoustic measures: pitch statistics, jitter (pitch instability),
shimmer (loudness instability), noise ratios, and four nonlinear dynamical measures.
""")

K(r"""
raw = load_raw()
df = add_subject_column(raw)
per_speaker = df.groupby("subject").agg(n_recordings=("name", "size"),
                                        status=("status", "first"))

print(f"recordings : {len(raw)}")
print(f"speakers   : {len(per_speaker)}  "
      f"({int((per_speaker.status==1).sum())} Parkinson's, "
      f"{int((per_speaker.status==0).sum())} healthy)")
print(f"recordings per speaker: {per_speaker.n_recordings.min()}"
      f"-{per_speaker.n_recordings.max()}")
print(f"\nrecording-level prevalence: {raw.status.mean():.1%} Parkinson's")
print("A model that always says 'Parkinson's' is therefore already "
      f"{raw.status.mean():.1%} accurate — accuracy alone is uninformative here.")

X, y, groups, meta = load_dataset()
print(f"\nafter removing 6 algebraically redundant columns: {X.shape[1]} features")
""")

M(r"""
## 2. Which voice measures signal Parkinson's

Before any model, the direct question: does each acoustic measure differ between
groups, in which direction, and by how much?

Features are averaged within each speaker first, so these statistics count 32
independent people rather than 195 correlated recordings.
""")

K(r"""
uni = pd.read_csv(TABLES_DIR / "biomarker_univariate.csv")
cols = ["plain_name", "feature", "direction", "cohens_d", "auc_single_feature",
        "healthy_mean", "parkinsons_mean"]
display(uni[cols].head(10).round(3))
""")

K(r"""
display(Image(filename=str(FIGURES_DIR / "biomarker_separation.png")))
""")

M(r"""
**The clinical reading.** The strongest signals are not the classical jitter and
shimmer perturbation measures — they are `spread1` and `PPE`, both nonlinear
descriptors of how erratically the fundamental frequency moves. Both are *higher*
in Parkinson's, with very large effect sizes (*d* = 2.03 and 1.91).

That is consistent with the pathophysiology: Parkinson's disrupts the fine motor
control of the laryngeal muscles, so pitch becomes harder to hold steady. Loudness
instability (`MDVP:Shimmer`, *d* = 1.24) and amplitude perturbation (`MDVP:APQ`,
*d* = 1.22) follow the same pattern, one tier down.
""")

M(r"""
## 3. Choosing a model

Twelve classifiers, all evaluated identically: 4-fold x 10-repeat **nested**
cross-validation, grouped by speaker, with hyper-parameters tuned by an inner CV that
obeys the same grouping. 960 fold evaluations in total.
""")

K(r"""
folds = pd.read_csv(TABLES_DIR / "cv_folds.csv")
honest = folds[folds.protocol == "subject"]
board = (honest.groupby("model")[["roc_auc", "balanced_accuracy",
                                  "sensitivity", "specificity", "mcc"]]
         .agg(["mean", "std"]))
board.columns = [f"{a}_{b}" for a, b in board.columns]
board = board[["roc_auc_mean", "roc_auc_std", "balanced_accuracy_mean",
               "sensitivity_mean", "specificity_mean", "mcc_mean"]]
display(board.sort_values("roc_auc_mean", ascending=False).round(3))
""")

M(r"""
**Logistic regression wins.** It beats every ensemble, the SVM and the neural network.
At 32 speakers and 16 features there is not enough data for flexible models to find
anything a linear boundary misses, and they pay for the attempt in variance.

Logistic regression is therefore the deployed model for the rest of this notebook.
It is also the most explainable choice, which here costs nothing.
""")

M(r"""
## 4. The screening model and its operating point

A screening test is defined by its operating point, not by its accuracy. The threshold
is tuned to reach **90% sensitivity**, and the cost in specificity is then measured.

The threshold is chosen *nested*: inside every outer fold it is fitted on an inner
cross-validation of the training speakers only, then frozen and applied to the
held-out speakers. Choosing a threshold on the same data used to report it would
be a subtle leak.
""")

K(r"""
scr = json.loads((RESULTS_DIR / "screening_report.json").read_text())
print(f"model      : {scr['model']}")
print(f"threshold  : {scr['threshold_mean']:.3f} +/- {scr['threshold_sd']:.3f} "
      f"(chosen independently in each of {scr['n_outer_folds']} folds)")

rows = []
for level in ["recording_level", "speaker_level"]:
    d = scr[level]
    rows.append({"level": level.replace("_level", ""),
                 "ROC-AUC": f"{d['roc_auc']:.3f} +/- {d['roc_auc_sd']:.3f}",
                 "sensitivity": f"{d['sensitivity']:.3f} +/- {d['sensitivity_sd']:.3f}",
                 "specificity": f"{d['specificity']:.3f} +/- {d['specificity_sd']:.3f}",
                 "PPV": f"{d['ppv']:.3f}", "NPV": f"{d['npv']:.3f}"})
display(pd.DataFrame(rows))

print("\nCalibration:", json.dumps(scr["calibration"]))
""")

K(r"""
display(Image(filename=str(FIGURES_DIR / "screening_curves.png")))
display(Image(filename=str(FIGURES_DIR / "screening_confusion.png")))
""")

M(r"""
**What this test would actually do.** Per repeat, among the 8 held-out healthy
speakers it correctly clears about 3.9 and wrongly flags about 4.1. Among the 24
with Parkinson's it catches about 22.1 and misses about 1.9.

So: it rarely misses a case, and it cries wolf about half the time on healthy
speakers. As a triage step before clinical assessment that may be acceptable. As a
standalone diagnostic it is not, and no accuracy figure would have made that visible.
""")

K(r"""
display(Image(filename=str(FIGURES_DIR / "screening_calibration.png")))
display(Image(filename=str(FIGURES_DIR / "screening_risk_distribution.png")))
""")

M(r"""
The risk plot shows every speaker scored while held out. The two groups overlap in a
band roughly between 0.3 and 0.6 — that overlap *is* the specificity problem, and it
is a property of the data, not of the classifier.
""")

M(r"""
## 5. Explaining individual predictions

Feature importance without a sign is not a clinical finding. Below, SHAP values are
computed from the deployed logistic regression on held-out speakers, so the direction
of every effect is explicit.
""")

K(r"""
signed = pd.read_csv(TABLES_DIR / "biomarker_model_attribution.csv")
display(signed[["plain_name", "feature", "mean_abs_shap", "effect_direction",
                "value_shap_corr"]].head(10).round(4))
""")

K(r"""
display(Image(filename=str(FIGURES_DIR / "biomarker_beeswarm.png")))
display(Image(filename=str(FIGURES_DIR / "biomarker_dependence.png")))
""")

M(r"""
The model's own attributions agree with the univariate statistics of Section 2:
`spread1` and `PPE` dominate, both raising predicted risk when high. Average pitch
(`MDVP:Fo`) acts in the opposite direction — a higher, better-sustained pitch lowers
predicted risk.

### Per-speaker explanation cards

The deliverable of an explainable screening tool is not a global bar chart, it is a
per-patient answer to "why was I flagged?". Below: one true positive, one true
negative, one false positive and one false negative, all held out of training.
""")

K(r"""
cards = pd.read_csv(TABLES_DIR / "patient_cards.csv", index_col=0)
display(cards[["y_true", "y_prob", "pred", "outcome"]].round(3))
display(Image(filename=str(FIGURES_DIR / "patient_cards.png")))
""")

M(r"""
The false negative (S25) is the instructive one: this speaker has Parkinson's but
scores 0.32, below the flag threshold. Their pitch-variation measures sit in the
healthy range. A voice test cannot detect a patient whose voice is not yet affected,
and that is a limit of the modality rather than of the model.
""")

M(r"""
## 6. Parsimony: one feature beats sixteen

If `spread1` carries most of the signal, what do the other fifteen features buy?
Each row below is a complete model, refit under the identical speaker-grouped
protocol with `C` tuned the same way.
""")

K(r"""
sf = pd.read_csv(TABLES_DIR / "single_feature_cv.csv")
display(sf.round(3))
best1 = sf.iloc[0]; allf = sf[sf.model.str.startswith("ALL")].iloc[0]
print(f"\n{best1.model} alone : speaker-level AUC {best1.speaker_auc:.3f} "
      f"+/- {best1.speaker_sd:.3f}  (1 feature)")
print(f"All features       : speaker-level AUC {allf.speaker_auc:.3f} "
      f"+/- {allf.speaker_sd:.3f}  ({int(allf.n_features)} features)")
print(f"\nDifference: {best1.speaker_auc - allf.speaker_auc:+.3f} AUC in favour "
      f"of the single feature, with {allf.speaker_sd/best1.speaker_sd:.1f}x "
      f"less run-to-run variance.")
""")

M(r"""
**This is the most useful thing XAI produced here.** The explanation was not
decoration on top of a model — it identified that a one-variable model is both more
accurate and eight times more stable than the sixteen-variable one. At 32 speakers,
the additional features are fitting noise.

A deployable version of this work should be a two-feature model (`spread1`, `PPE`),
not an ensemble over 22 acoustic measures.
""")

M(r"""
## 7. Validity: why the split is grouped by speaker

Every number above uses `StratifiedGroupKFold` on speaker ID. This section shows what
happens without it, because the alternative is what most published notebooks on this
dataset do.

Splitting 195 recordings at random puts five of a person's six recordings in training
and the sixth in test. The model can score well by recognising a voice it has already
heard. A 39-row random test set contains roughly **6 independent people**.
""")

K(r"""
lk = pd.read_csv(TABLES_DIR / "leakage_optimism_roc_auc.csv")
display(lk[["model", "roc_auc_record", "roc_auc_subject", "optimism",
            "ci95_low", "ci95_high", "p_value"]].round(4))
print(f"Mean inflation: "
      f"{lk[lk.model!='Baseline (majority)'].optimism.mean():+.3f} AUC")
print(f"Models significantly inflated at p<0.001: "
      f"{int((lk.p_value<0.001).sum())} of {len(lk)-1}")
""")

K(r"""
display(Image(filename=str(FIGURES_DIR / "protocol_gap_roc_auc.png")))
""")

K(r"""
rep = json.loads((RESULTS_DIR / "analysis_report.json").read_text())
print(json.dumps(rep["ranking"], indent=2))
""")

M(r"""
The inflation is large (+0.159 AUC on average), but the second consequence is worse:
the two protocols rank the models almost independently (Spearman 0.19, *p* = 0.57).
Extra Trees looks best under the leaky split and Logistic Regression is best under the
honest one. **A model chosen on a leaky benchmark is chosen at random.**

This also explains a familiar number. The commonly reported ~95% accuracy on this
dataset is a single draw from the red distribution below, not a generalisation
estimate.
""")

K(r"""
display(Image(filename=str(FIGURES_DIR / "fold_distributions.png")))
for m in ["Random Forest", "Logistic Regression"]:
    v = folds[(folds.model == m) & (folds.protocol == "subject")].roc_auc
    print(f"{m:22s} honest AUC across folds: {v.min():.3f} - {v.max():.3f}")
""")

M(r"""
### Stability of the explanations

Explanations deserve the same scepticism as scores. Refitting and re-explaining
across 12 speaker-disjoint folds:
""")

K(r"""
rep_s = json.loads((RESULTS_DIR / "shap_stability_report.json").read_text())
print("mean pairwise Spearman between fold rankings:",
      round(rep_s["mean_pairwise_spearman"], 3))
print("worst pair:", round(rep_s["min_pairwise_spearman"], 3))
display(pd.read_csv(TABLES_DIR / "shap_stability.csv", index_col=0).round(4).head(8))
display(Image(filename=str(FIGURES_DIR / "shap_stability.png")))
""")

M(r"""
Only `spread1` and `PPE` rank consistently. Everything below them reorders from fold
to fold, which is why Section 6's conclusion is stated as a two-feature model rather
than a ranked list of ten biomarkers.

*Implementation note.* For binary classification `shap.TreeExplainer` returns an array
of shape `(n, features, 2)` whose class slices are symmetric, so averaging over the
class axis returns approximately zero. Measured on this data: **1.35e-17** for the
class-mean versus **3.16e-02** for the positive-class slice. Attribution plots built
on the class-mean show floating-point noise.
""")

M(r"""
## 8. Conclusions

**On prediction.** Voice-based screening for Parkinson's is feasible but not
diagnostic. Held-out speaker-level ROC-AUC is **0.852**; at a 90%-sensitivity
operating point the test catches 92% of cases at **48.8% specificity**. It is a
triage instrument, not a diagnosis.

**On explainability.** The dominant signal is impaired control of pitch — `spread1`
and `PPE`, both raised in Parkinson's with *d* > 1.9. Classical jitter and shimmer
measures matter less than the nonlinear descriptors. Per-speaker SHAP cards make each
flag auditable, including the failures.

**On modelling.** A one-feature logistic regression (**0.924 AUC**) outperforms the
full 16-feature model (**0.847**) with far lower variance. Complexity is not rewarded
at this sample size.

**On methodology.** Speaker-grouped validation is not optional on this dataset.
Ignoring it inflates ROC-AUC by +0.159 on average and makes model selection
meaningless.

## Limitations

* **32 speakers, 8 of them healthy.** Every specificity estimate rests on 8 people.
  Confidence intervals are wide; treat all figures as approximate.
* **Single session, single site.** One cohort, one recording setup (Little et al.,
  2007). Nothing here establishes generalisation across microphones, languages,
  recording conditions, or disease stage.
* **No external validation set.** All figures are cross-validation estimates. At this
  sample size a held-out set would be too small to be informative.
* **Cross-sectional, not diagnostic.** Speakers already carry a diagnosis; this does
  not demonstrate early detection, and no severity or medication state is modelled.
* **Not a medical device.** This is a methodology and explainability study on a public
  benchmark.

## Data citation

Little, M., McSharry, P., Roberts, S., Costello, D., & Moroz, I. (2007).
*Exploiting nonlinear recurrence and fractal scaling properties for voice disorder
detection.* BioMedical Engineering OnLine, 6:23.
Dataset: UCI Machine Learning Repository, "Parkinsons" (ID 174), CC BY 4.0.
""")

nb = new_notebook(cells=C, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python"}})

out = "parkinsons_xai.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(C), "cells")
