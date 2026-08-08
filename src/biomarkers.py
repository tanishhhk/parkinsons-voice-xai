"""What in the voice signals Parkinson's, and in which direction.

Feature importance without a sign is not a clinical finding. This module reports,
for every acoustic measure: whether it is higher or lower in Parkinson's speakers,
how well it separates them on its own, and how the deployed model uses it.

All model-based quantities are computed on speakers held out of training.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from config import FIGURES_DIR, N_SPLITS, RESULTS_DIR, SEED, TABLES_DIR
from data import load_dataset
from screening import _fit
from models import get_model_zoo

# Plain-language description of each acoustic measure and what it captures.
GLOSSARY = {
    "MDVP:Fo(Hz)":      ("Average pitch", "Mean fundamental frequency of the sustained vowel"),
    "MDVP:Fhi(Hz)":     ("Highest pitch", "Maximum fundamental frequency reached"),
    "MDVP:Flo(Hz)":     ("Lowest pitch", "Minimum fundamental frequency reached"),
    "MDVP:Jitter(%)":   ("Pitch instability", "Cycle-to-cycle variation in pitch period"),
    "MDVP:Jitter(Abs)": ("Pitch instability (abs)", "Absolute cycle-to-cycle pitch variation"),
    "MDVP:PPQ":         ("Pitch perturbation", "Five-point period perturbation quotient"),
    "MDVP:Shimmer":     ("Loudness instability", "Cycle-to-cycle variation in amplitude"),
    "MDVP:APQ":         ("Amplitude perturbation", "Eleven-point amplitude perturbation quotient"),
    "NHR":              ("Noise-to-harmonics", "Ratio of noise to tonal components"),
    "HNR":              ("Harmonics-to-noise", "Tonal clarity; low values mean a breathy, hoarse voice"),
    "RPDE":             ("Signal unpredictability", "Recurrence period density entropy of the waveform"),
    "DFA":              ("Breath-noise scaling", "Detrended fluctuation analysis; turbulent-noise self-similarity"),
    "spread1":          ("Pitch-variation measure 1", "Nonlinear measure of fundamental-frequency variation"),
    "spread2":          ("Pitch-variation measure 2", "Nonlinear measure of fundamental-frequency variation"),
    "D2":               ("Signal complexity", "Correlation dimension of the reconstructed attractor"),
    "PPE":              ("Pitch-period entropy", "Impaired control of pitch on a logarithmic scale"),
}


def univariate_separation(X, y, groups):
    """Per-feature discriminative power and direction, at the speaker level.

    Averaging each feature within a speaker first means the statistics count
    32 independent people rather than 195 correlated recordings.
    """
    df = X.copy()
    df["_y"] = y.values
    df["_g"] = groups.values
    per_speaker = df.groupby("_g").mean()
    ys = per_speaker.pop("_y").astype(int)

    rows = []
    for col in X.columns:
        v = per_speaker[col]
        a = roc_auc_score(ys, v)
        pd_mean, hc_mean = v[ys == 1].mean(), v[ys == 0].mean()
        pooled_sd = np.sqrt((v[ys == 1].var(ddof=1) + v[ys == 0].var(ddof=1)) / 2)
        rows.append({
            "feature": col,
            "plain_name": GLOSSARY.get(col, ("", ""))[0],
            "meaning": GLOSSARY.get(col, ("", ""))[1],
            "healthy_mean": hc_mean,
            "parkinsons_mean": pd_mean,
            "direction": "higher in PD" if pd_mean > hc_mean else "lower in PD",
            "cohens_d": (pd_mean - hc_mean) / pooled_sd if pooled_sd else np.nan,
            # AUC folded above 0.5 so it reads as "separation strength"
            "auc_single_feature": max(a, 1 - a),
        })
    out = (pd.DataFrame(rows)
           .sort_values("auc_single_feature", ascending=False)
           .reset_index(drop=True))
    out.to_csv(TABLES_DIR / "biomarker_univariate.csv", index=False)
    return out


def model_attributions(model_name="Logistic Regression", seed=SEED):
    """Signed SHAP values from the deployed model, on held-out speakers."""
    import shap

    X, y, groups, _ = load_dataset()
    Xv, yv, gv = X.values, y.values, groups.values
    pipe, grid = get_model_zoo()[model_name]

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    phis, vals, probs, truths, subs = [], [], [], [], []

    for tr, te in cv.split(X, y, groups):
        model = _fit(pipe, grid, Xv, yv, gv, tr, seed)
        scaler, clf = model.named_steps["scaler"], model.named_steps["clf"]
        bg = scaler.transform(Xv[tr])
        expl = shap.LinearExplainer(clf, bg)
        phis.append(expl.shap_values(scaler.transform(Xv[te])))
        vals.append(Xv[te])
        probs.append(model.predict_proba(Xv[te])[:, 1])
        truths.append(yv[te])
        subs.append(gv[te])

    phi = np.vstack(phis)
    raw = np.vstack(vals)
    meta = pd.DataFrame({"subject": np.concatenate(subs),
                         "y_true": np.concatenate(truths),
                         "y_prob": np.concatenate(probs)})

    signed = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(phi).mean(axis=0),
        # sign of the value-attribution relationship = direction of effect
        "effect_direction": [
            "raises PD risk when high" if np.corrcoef(raw[:, j], phi[:, j])[0, 1] > 0
            else "lowers PD risk when high" for j in range(phi.shape[1])],
        "value_shap_corr": [np.corrcoef(raw[:, j], phi[:, j])[0, 1]
                            for j in range(phi.shape[1])],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    signed["plain_name"] = signed.feature.map(lambda f: GLOSSARY.get(f, ("", ""))[0])
    signed.to_csv(TABLES_DIR / "biomarker_model_attribution.csv", index=False)

    return phi, raw, meta, signed, X.columns


# ------------------------------------------------------------------ figures
def fig_biomarker_table(uni):
    top = uni.head(10).iloc[::-1]
    colors = ["#c0504d" if d == "higher in PD" else "#3b6ea5"
              for d in top.direction]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.barh(top.plain_name + "  (" + top.feature + ")",
            top.auc_single_feature, color=colors)
    ax.axvline(0.5, color="k", ls=":", lw=1)
    for yi, (a, d) in enumerate(zip(top.auc_single_feature, top.cohens_d)):
        ax.text(a + .008, yi, f"d={d:+.2f}", va="center", fontsize=8.5)
    ax.set(xlim=(0.4, 1.02), xlabel="single-feature AUC across 32 speakers",
           title="Which voice measures separate Parkinson's speakers on their own")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#c0504d", "#3b6ea5"]]
    ax.legend(handles, ["higher in Parkinson's", "lower in Parkinson's"],
              frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "biomarker_separation.png", dpi=160)
    plt.close(fig)


def fig_beeswarm(phi, raw, cols):
    import shap
    plt.figure()
    shap.summary_plot(phi, pd.DataFrame(raw, columns=cols), show=False,
                      max_display=16)
    plt.title("How the screening model uses each voice measure\n"
              "(red = high value; right = pushes toward Parkinson's)", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "biomarker_beeswarm.png", dpi=160)
    plt.close()


def fig_dependence(phi, raw, cols, signed, n=4):
    top = signed.head(n).feature.tolist()
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4))
    for ax, f in zip(np.atleast_1d(axes), top):
        j = list(cols).index(f)
        ax.scatter(raw[:, j], phi[:, j], s=14, alpha=.65, color="#3b6ea5")
        ax.axhline(0, color="k", lw=.8, ls=":")
        ax.set_xlabel(f)
        ax.set_title(GLOSSARY.get(f, ("", ""))[0], fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("SHAP value\n(+ = toward Parkinson's)")
    fig.suptitle("Dose-response: measured value vs its contribution", y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "biomarker_dependence.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def fig_patient_cards(phi, raw, meta, cols, threshold, k=6):
    """One explanation card per speaker: risk score plus the reasons for it."""
    sp = (meta.assign(idx=np.arange(len(meta)))
          .groupby("subject")
          .agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"),
               idx=("idx", "first")))
    sp["pred"] = (sp.y_prob >= threshold).astype(int)
    sp["outcome"] = np.select(
        [(sp.y_true == 1) & (sp.pred == 1), (sp.y_true == 0) & (sp.pred == 0),
         (sp.y_true == 0) & (sp.pred == 1)],
        ["true positive", "true negative", "false positive"], "false negative")

    picks = []
    for kind in ["true positive", "true negative", "false positive", "false negative"]:
        sub = sp[sp.outcome == kind]
        if len(sub):
            picks.append(sub.assign(kind=kind).iloc[[0]])
    cards = pd.concat(picks).head(k)

    fig, axes = plt.subplots(1, len(cards), figsize=(4.3 * len(cards), 4.6))
    for ax, (subject, row) in zip(np.atleast_1d(axes), cards.iterrows()):
        i = int(row.idx)
        contrib = pd.Series(phi[i], index=cols).sort_values(key=abs,
                                                            ascending=False).head(7)[::-1]
        ax.barh(contrib.index, contrib.values,
                color=["#c0504d" if v > 0 else "#3b6ea5" for v in contrib.values])
        ax.axvline(0, color="k", lw=.8)
        flag = "FLAG" if row.pred else "clear"
        ax.set_title(f"{subject} — {row.kind}\n"
                     f"risk {row.y_prob:.2f} vs threshold {threshold:.2f} -> {flag}",
                     fontsize=10)
        ax.tick_params(labelsize=8)
    np.atleast_1d(axes)[0].set_xlabel("contribution to log-odds of Parkinson's")
    fig.suptitle("Per-speaker explanations (all speakers held out of training)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "patient_cards.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    cards.to_csv(TABLES_DIR / "patient_cards.csv")
    return cards


def single_feature_cv(features=("spread1", "PPE", "spread2", "MDVP:APQ"),
                      n_repeats=10):
    """Cross-validated AUC of a one-feature model, for a fair comparison.

    The univariate AUC above is descriptive and computed in-sample, so it is not
    comparable to the cross-validated figures for the full model. This refits a
    single-feature logistic regression under the identical speaker-grouped
    protocol so the two numbers can sit in the same table.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, groups, _ = load_dataset()
    rows = []
    for feat in list(features) + ["ALL FEATURES"]:
        cols = list(X.columns) if feat == "ALL FEATURES" else [feat]
        Xf = X[cols].values
        recs, specs = [], []
        for rep in range(n_repeats):
            cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                      random_state=SEED + rep)
            probs, truth, subj = [], [], []
            for tr, te in cv.split(Xf, y, groups):
                base = Pipeline([("s", StandardScaler()),
                                 ("c", LogisticRegression(max_iter=5000,
                                                          class_weight="balanced"))])
                # tune C with the same grouped inner CV used everywhere else,
                # so single-feature and full models get identical treatment
                m = GridSearchCV(base, {"c__C": [0.01, 0.1, 1.0, 10.0]},
                                 scoring="roc_auc",
                                 cv=StratifiedGroupKFold(n_splits=3, shuffle=True,
                                                         random_state=SEED),
                                 n_jobs=-1)
                m.fit(Xf[tr], y.values[tr], groups=groups.values[tr])
                probs.append(m.predict_proba(Xf[te])[:, 1])
                truth.append(y.values[te])
                subj.append(groups.values[te])
            d = pd.DataFrame({"p": np.concatenate(probs),
                              "y": np.concatenate(truth),
                              "s": np.concatenate(subj)})
            recs.append(roc_auc_score(d.y, d.p))
            sp = d.groupby("s").agg(y=("y", "first"), p=("p", "mean"))
            specs.append(roc_auc_score(sp.y, sp.p))
        rows.append({"model": feat if feat != "ALL FEATURES" else "ALL 16 FEATURES", "n_features": len(cols),
                     "recording_auc": np.mean(recs), "recording_sd": np.std(recs),
                     "speaker_auc": np.mean(specs), "speaker_sd": np.std(specs)})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "single_feature_cv.csv", index=False)
    print(out.round(3).to_string(index=False))
    return out


def main(model_name="Logistic Regression"):
    X, y, groups, _ = load_dataset()
    uni = univariate_separation(X, y, groups)
    print(uni[["feature", "plain_name", "direction", "cohens_d",
               "auc_single_feature"]].head(10).round(3).to_string(index=False))

    phi, raw, meta, signed, cols = model_attributions(model_name)
    print("\nModel attribution (signed):")
    print(signed[["feature", "plain_name", "mean_abs_shap",
                  "effect_direction"]].head(8).round(4).to_string(index=False))

    thr = json.loads((RESULTS_DIR / "screening_report.json").read_text())["threshold_mean"]
    fig_biomarker_table(uni)
    fig_beeswarm(phi, raw, cols)
    fig_dependence(phi, raw, cols, signed)
    cards = fig_patient_cards(phi, raw, meta, cols, thr)
    print("\nExplanation cards:")
    print(cards[["y_true", "y_prob", "pred", "outcome"]].round(3).to_string())

    agree = uni.set_index("feature").loc[signed.feature]
    report = {
        "model": model_name,
        "top_biomarkers_univariate": uni.head(5)[
            ["feature", "plain_name", "direction", "cohens_d",
             "auc_single_feature"]].round(3).to_dict("records"),
        "top_biomarkers_model": signed.head(5)[
            ["feature", "plain_name", "mean_abs_shap",
             "effect_direction"]].round(4).to_dict("records"),
        "spearman_univariate_vs_model": round(float(
            pd.Series(agree.auc_single_feature.values).corr(
                pd.Series(signed.mean_abs_shap.values), method="spearman")), 3),
    }
    (RESULTS_DIR / "biomarker_report.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report["top_biomarkers_univariate"], indent=2))
    return report


if __name__ == "__main__":
    main()
