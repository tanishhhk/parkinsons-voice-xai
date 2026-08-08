"""Quantify the leakage effect and produce the report figures."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import FIGURES_DIR, RESULTS_DIR, TABLES_DIR

PALETTE = {"record": "#c0504d", "subject": "#3b6ea5"}


def load():
    folds = pd.read_csv(TABLES_DIR / "cv_folds.csv")
    subj = pd.read_csv(TABLES_DIR / "subject_level_folds.csv")
    return folds, subj


def leakage_table(folds, metric="roc_auc"):
    """Per-model optimism, paired by repeat index, with a t-test and CI."""
    rows = []
    for model, chunk in folds.groupby("model"):
        rec = chunk[chunk.protocol == "record"].groupby("repeat")[metric].mean()
        sub = chunk[chunk.protocol == "subject"].groupby("repeat")[metric].mean()
        common = rec.index.intersection(sub.index)
        diff = (rec[common] - sub[common]).values
        if np.allclose(diff, 0):
            t, p = np.nan, np.nan
        else:
            t, p = stats.ttest_rel(rec[common], sub[common])
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
        rows.append({
            "model": model,
            f"{metric}_record": rec.mean(),
            f"{metric}_subject": sub.mean(),
            "optimism": diff.mean(),
            "ci95_low": diff.mean() - 1.96 * se,
            "ci95_high": diff.mean() + 1.96 * se,
            "t_stat": t, "p_value": p, "n_repeats": len(diff)})
    out = pd.DataFrame(rows).sort_values("optimism", ascending=False)
    out.to_csv(TABLES_DIR / f"leakage_optimism_{metric}.csv", index=False)
    return out


def fig_protocol_gap(folds, metric="roc_auc"):
    agg = (folds.groupby(["model", "protocol"])[metric]
           .agg(["mean", "std"]).reset_index())
    order = (agg[agg.protocol == "subject"].sort_values("mean")["model"].tolist())
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, proto in enumerate(["record", "subject"]):
        d = agg[agg.protocol == proto].set_index("model").loc[order]
        ax.barh(y + (0.2 if i == 0 else -0.2), d["mean"], height=0.38,
                xerr=d["std"], color=PALETTE[proto], alpha=.9,
                label={"record": "Random split by recording (leaky)",
                       "subject": "Grouped split by speaker (honest)"}[proto],
                error_kw=dict(lw=.8, alpha=.6))
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.text(0.505, len(order) - .4, "chance", fontsize=8)
    ax.set_yticks(y, order)
    ax.set_xlabel(f"{metric.upper().replace('_',' ')} (mean +/- SD over 40 folds)")
    ax.set_xlim(0.4, 1.0)
    ax.set_title("Subject leakage inflates every non-trivial model")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"protocol_gap_{metric}.png", dpi=160)
    plt.close(fig)


def fig_optimism(lk, metric="roc_auc"):
    d = lk[lk.model != "Baseline (majority)"].sort_values("optimism")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(d.model, d.optimism,
            xerr=[d.optimism - d.ci95_low, d.ci95_high - d.optimism],
            color="#c0504d", alpha=.9, error_kw=dict(lw=1))
    ax.set_xlabel(f"Optimism: leaky {metric.upper()} - honest {metric.upper()}")
    ax.set_title("How much accuracy is borrowed from speaker identity")
    ax.axvline(0, color="k", lw=.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"optimism_{metric}.png", dpi=160)
    plt.close(fig)


def fig_subject_level(subj):
    agg = (subj.groupby("model")[["roc_auc", "balanced_accuracy",
                                  "sensitivity", "specificity"]]
           .mean().sort_values("roc_auc"))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(agg))
    w = 0.2
    for i, (col, c) in enumerate(zip(
            ["roc_auc", "balanced_accuracy", "sensitivity", "specificity"],
            ["#3b6ea5", "#6a9c78", "#d99a3a", "#a05195"])):
        ax.bar(x + (i - 1.5) * w, agg[col], width=w, label=col, color=c)
    ax.set_xticks(x, agg.index, rotation=35, ha="right")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Speaker-level diagnosis (probabilities averaged per speaker)")
    ax.legend(ncol=4, frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "subject_level_scores.png", dpi=160)
    plt.close(fig)


def fig_distribution(folds, metric="roc_auc"):
    """Fold-level spread: shows how unstable a single split would have been."""
    models = ["Logistic Regression", "SVM (RBF)", "Random Forest", "XGBoost"]
    fig, axes = plt.subplots(1, len(models), figsize=(13, 3.6), sharey=True)
    for ax, m in zip(axes, models):
        for proto in ["record", "subject"]:
            v = folds[(folds.model == m) & (folds.protocol == proto)][metric]
            ax.hist(v, bins=14, alpha=.65, color=PALETTE[proto],
                    label=proto, range=(0.3, 1.0))
        ax.set_title(m, fontsize=10)
        ax.set_xlabel(metric)
    axes[0].set_ylabel("folds")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Per-fold score distribution (40 folds each)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fold_distributions.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def main():
    folds, subj = load()
    report = {}
    for metric in ["roc_auc", "accuracy", "balanced_accuracy"]:
        lk = leakage_table(folds, metric)
        report[metric] = {
            "mean_optimism_excl_baseline": float(
                lk[lk.model != "Baseline (majority)"].optimism.mean()),
            "max_optimism": {"model": lk.iloc[0].model,
                             "value": float(lk.iloc[0].optimism)},
        }
        fig_protocol_gap(folds, metric)
        if metric == "roc_auc":
            fig_optimism(lk, metric)
            print(lk.round(4).to_string(index=False))

    fig_subject_level(subj)
    fig_distribution(folds)

    best_honest = (folds[folds.protocol == "subject"]
                   .groupby("model").roc_auc.mean().drop("Baseline (majority)")
                   .sort_values(ascending=False))
    best_leaky = (folds[folds.protocol == "record"]
                  .groupby("model").roc_auc.mean().drop("Baseline (majority)")
                  .sort_values(ascending=False))
    rho = stats.spearmanr(best_leaky.loc[best_honest.index], best_honest)
    report["ranking"] = {
        "best_under_leaky_protocol": best_leaky.index[0],
        "best_under_honest_protocol": best_honest.index[0],
        "spearman_rank_correlation_between_protocols": round(rho.statistic, 4),
        "spearman_p": round(rho.pvalue, 4),
    }
    (RESULTS_DIR / "analysis_report.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
