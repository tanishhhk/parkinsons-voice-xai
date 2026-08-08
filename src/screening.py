"""Build and evaluate the deployable screening model.

A screening test is defined by its operating point, not by its accuracy. The
threshold here is chosen to reach a target sensitivity, and the specificity that
this costs is then measured on held-out speakers.

Threshold selection is nested: inside every outer fold the threshold is picked
from an inner cross-validation of the *training* speakers only, then applied
unchanged to the held-out speakers. Choosing a threshold on the same data used
to report it is a subtle form of leakage and is avoided here.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, brier_score_loss, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from config import (FIGURES_DIR, INNER_SPLITS, N_REPEATS, N_SPLITS,
                    RESULTS_DIR, SEED, TABLES_DIR)
from data import load_dataset
from models import get_model_zoo

TARGET_SENSITIVITY = 0.90


def threshold_for_sensitivity(y_true, y_prob, target=TARGET_SENSITIVITY):
    """Lowest-cost threshold whose sensitivity is at least `target`."""
    order = np.argsort(-y_prob)
    probs = y_prob[order]
    pos = y_true[order] == 1
    n_pos = pos.sum()
    if n_pos == 0:
        return 0.5
    tpr = np.cumsum(pos) / n_pos
    idx = np.argmax(tpr >= target)
    if tpr[idx] < target:
        return float(probs.min())
    return float(probs[idx])


def _fit(pipe, grid, X, y, g, idx, seed):
    est = clone(pipe)
    if grid:
        s = GridSearchCV(est, grid, scoring="roc_auc",
                         cv=StratifiedGroupKFold(n_splits=INNER_SPLITS,
                                                 shuffle=True,
                                                 random_state=seed),
                         n_jobs=-1, refit=True)
        s.fit(X[idx], y[idx], groups=g[idx])
        return s.best_estimator_
    est.fit(X[idx], y[idx])
    return est


def run(model_name="Logistic Regression", target=TARGET_SENSITIVITY,
        n_repeats=N_REPEATS):
    """Nested evaluation with a nested-chosen screening threshold."""
    X, y, groups, _ = load_dataset()
    Xv, yv, gv = X.values, y.values, groups.values
    pipe, grid = get_model_zoo()[model_name]

    rows, thresholds = [], []
    for rep in range(n_repeats):
        seed = SEED + rep
        outer = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                     random_state=seed)
        for fold, (tr, te) in enumerate(outer.split(X, y, groups)):
            # --- inner CV on the training speakers, purely to set a threshold
            inner = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True,
                                         random_state=seed + 100)
            oof = np.zeros(len(tr))
            for itr, ite in inner.split(Xv[tr], yv[tr], gv[tr]):
                m = _fit(pipe, grid, Xv[tr], yv[tr], gv[tr], itr, seed)
                oof[ite] = m.predict_proba(Xv[tr][ite])[:, 1]
            thr = threshold_for_sensitivity(yv[tr], oof, target)
            thresholds.append(thr)

            # --- refit on all training speakers, apply the frozen threshold
            model = _fit(pipe, grid, Xv, yv, gv, tr, seed)
            prob = model.predict_proba(Xv[te])[:, 1]
            rows.append(pd.DataFrame({
                "repeat": rep, "fold": fold, "threshold": thr,
                "subject": gv[te], "y_true": yv[te], "y_prob": prob}))

    oof_df = pd.concat(rows, ignore_index=True)
    oof_df["model"] = model_name
    oof_df.to_csv(TABLES_DIR / "screening_oof.csv", index=False)

    report = {"model": model_name, "target_sensitivity": target,
              "threshold_mean": float(np.mean(thresholds)),
              "threshold_sd": float(np.std(thresholds)),
              "n_outer_folds": len(thresholds)}
    report["recording_level"] = _operating_point(oof_df, level="recording")
    report["speaker_level"] = _operating_point(oof_df, level="speaker")
    report["calibration"] = _calibration(oof_df)

    (RESULTS_DIR / "screening_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return oof_df, report


def _agg_speaker(df):
    return (df.groupby(["repeat", "subject"])
            .agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"),
                 threshold=("threshold", "first")).reset_index())


def _operating_point(oof, level="recording"):
    d = _agg_speaker(oof) if level == "speaker" else oof
    per_rep = []
    for rep, chunk in d.groupby("repeat"):
        pred = (chunk.y_prob >= chunk.threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(chunk.y_true, pred,
                                          labels=[0, 1]).ravel()
        per_rep.append({
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "ppv": tp / (tp + fp) if tp + fp else np.nan,
            "npv": tn / (tn + fn) if tn + fn else np.nan,
            "roc_auc": roc_auc_score(chunk.y_true, chunk.y_prob),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn})
    m = pd.DataFrame(per_rep)
    out = {k: round(float(m[k].mean()), 4)
           for k in ["sensitivity", "specificity", "ppv", "npv", "roc_auc"]}
    out.update({f"{k}_sd": round(float(m[k].std()), 4)
                for k in ["sensitivity", "specificity", "roc_auc"]})
    out["mean_confusion"] = {k: round(float(m[k].mean()), 2)
                             for k in ["tn", "fp", "fn", "tp"]}
    return out


def _calibration(oof):
    return {"brier_score": round(float(brier_score_loss(oof.y_true, oof.y_prob)), 4),
            "mean_predicted": round(float(oof.y_prob.mean()), 4),
            "observed_prevalence": round(float(oof.y_true.mean()), 4)}


# ------------------------------------------------------------------ figures
def figures(oof):
    _fig_curves(oof)
    _fig_confusion(oof)
    _fig_calibration(oof)
    _fig_risk_distribution(oof)


def _fig_curves(oof):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    grid = np.linspace(0, 1, 101)
    for level, color, ls in [("recording", "#3b6ea5", "-"),
                             ("speaker", "#c0504d", "-")]:
        d = _agg_speaker(oof) if level == "speaker" else oof
        tprs, aucs, precs = [], [], []
        for _, chunk in d.groupby("repeat"):
            fpr, tpr, _ = roc_curve(chunk.y_true, chunk.y_prob)
            tprs.append(np.interp(grid, fpr, tpr))
            aucs.append(auc(fpr, tpr))
            p, r, _ = precision_recall_curve(chunk.y_true, chunk.y_prob)
            precs.append(np.interp(grid, r[::-1], p[::-1]))
        tm, ts = np.mean(tprs, 0), np.std(tprs, 0)
        axes[0].plot(grid, tm, ls, color=color,
                     label=f"{level}: AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
        axes[0].fill_between(grid, tm - ts, tm + ts, color=color, alpha=.15)
        pm = np.mean(precs, 0)
        axes[1].plot(grid, pm, ls, color=color, label=level)
    axes[0].plot([0, 1], [0, 1], "k:", lw=1)
    axes[0].set(xlabel="1 - specificity", ylabel="sensitivity", title="ROC")
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    axes[1].axhline(oof.y_true.mean(), color="k", ls=":", lw=1)
    axes[1].set(xlabel="recall", ylabel="precision",
                title="Precision-recall", ylim=(0, 1.02))
    axes[1].legend(loc="lower left", frameon=False, fontsize=9)
    fig.suptitle("Held-out speakers, 10 repeats (shaded = +/- 1 SD)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "screening_curves.png", dpi=160)
    plt.close(fig)


def _fig_confusion(oof):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, level in zip(axes, ["recording", "speaker"]):
        d = _agg_speaker(oof) if level == "speaker" else oof
        mats = []
        for _, chunk in d.groupby("repeat"):
            pred = (chunk.y_prob >= chunk.threshold).astype(int)
            mats.append(confusion_matrix(chunk.y_true, pred, labels=[0, 1]))
        cm = np.mean(mats, 0)
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center",
                        color="white" if cm[i, j] / cm[i].sum() > .5 else "black",
                        fontsize=13)
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=["pred healthy", "pred PD"],
               yticklabels=["healthy", "PD"],
               title=f"{level} level (mean count / repeat)")
    fig.suptitle("Confusion at the 90%-sensitivity screening threshold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "screening_confusion.png", dpi=160)
    plt.close(fig)


def _fig_calibration(oof):
    fig, ax = plt.subplots(figsize=(5, 4.6))
    frac, mean_pred = calibration_curve(oof.y_true, oof.y_prob, n_bins=8,
                                        strategy="quantile")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect")
    ax.plot(mean_pred, frac, "o-", color="#3b6ea5",
            label=f"model (Brier {brier_score_loss(oof.y_true, oof.y_prob):.3f})")
    ax.set(xlabel="predicted probability", ylabel="observed fraction with PD",
           title="Calibration on held-out speakers")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "screening_calibration.png", dpi=160)
    plt.close(fig)


def _fig_risk_distribution(oof):
    sp = _agg_speaker(oof)
    med = sp.groupby(["subject"]).agg(y_true=("y_true", "first"),
                                      y_prob=("y_prob", "median"))
    thr = oof.threshold.mean()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for lab, color, name in [(0, "#6a9c78", "healthy"), (1, "#c0504d", "Parkinson's")]:
        d = med[med.y_true == lab].sort_values("y_prob")
        ax.scatter(d.y_prob, np.arange(len(d)) + (0 if lab == 0 else 0),
                   color=color, label=name, s=42, zorder=3)
    ax.axvline(thr, color="k", ls="--", lw=1.2,
               label=f"screening threshold {thr:.2f}")
    ax.set(xlabel="predicted probability of Parkinson's (median over repeats)",
           ylabel="speakers (ranked within group)",
           title="Every speaker, scored while held out")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "screening_risk_distribution.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "Logistic Regression"
    oof, rep = run(name)
    figures(oof)
    print("\nfigures ->", FIGURES_DIR)
