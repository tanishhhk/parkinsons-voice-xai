"""Explainability: SHAP, permutation importance, and LIME.

Two correctness rules are enforced here, both of which are routinely broken
in notebooks on this dataset:

1. Explanations are computed on speakers the model has NOT been trained on.
   Explaining training data describes memorisation, not behaviour.

2. For binary classification, SHAP returns values per class. Averaging over
   the class axis gives approximately zero, because the two classes are
   symmetric (phi_class0 = -phi_class1). The positive-class slice must be
   taken explicitly.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from config import FIGURES_DIR, INNER_SPLITS, SEED, TABLES_DIR
from data import load_dataset
from models import get_model_zoo


def holdout_by_subject(X, y, groups, seed=SEED):
    """One subject-disjoint train/test split (first fold of 4)."""
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    tr, te = next(cv.split(X, y, groups))
    return tr, te


def fit_best(name, X, y, groups, idx):
    pipe, grid = get_model_zoo()[name]
    est = clone(pipe)
    if grid:
        search = GridSearchCV(
            est, grid, scoring="roc_auc",
            cv=StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True,
                                    random_state=SEED),
            n_jobs=-1, refit=True)
        search.fit(X.values[idx], y.values[idx], groups=groups.values[idx])
        return search.best_estimator_, search.best_params_
    est.fit(X.values[idx], y.values[idx])
    return est, {}


def positive_class_shap(explanation):
    """Return an (n_samples, n_features) array of positive-class SHAP values."""
    vals = explanation.values if hasattr(explanation, "values") else explanation
    vals = np.asarray(vals)
    if vals.ndim == 3:          # (n, features, classes)
        return vals[:, :, 1]
    return vals                 # already (n, features)


def run(model_name="Random Forest"):
    import shap

    X, y, groups, _ = load_dataset()
    tr, te = holdout_by_subject(X, y, groups)
    model, best = fit_best(model_name, X, y, groups, tr)

    X_tr = pd.DataFrame(X.values[tr], columns=X.columns)
    X_te = pd.DataFrame(X.values[te], columns=X.columns)
    y_te = y.values[te]

    report = {"model": model_name, "best_params": {k: str(v) for k, v in best.items()},
              "n_train_subjects": int(pd.unique(groups.values[tr]).size),
              "n_test_subjects": int(pd.unique(groups.values[te]).size)}

    # ---------------------------------------------------------------- SHAP
    tree = model.named_steps["clf"]
    explainer = shap.TreeExplainer(tree)
    sv = explainer(X_te)
    phi = positive_class_shap(sv)

    raw = np.asarray(sv.values)
    report["shap_raw_shape"] = list(raw.shape)
    if raw.ndim == 3:
        # Demonstrate why the class-mean is wrong.
        report["mean_abs_shap_classmean"] = float(
            np.abs(raw.mean(axis=2)).mean())
        report["mean_abs_shap_positive_class"] = float(np.abs(phi).mean())

    shap_rank = (pd.Series(np.abs(phi).mean(axis=0), index=X.columns)
                 .sort_values(ascending=False))
    shap_rank.to_csv(TABLES_DIR / "shap_importance.csv", header=["mean_abs_shap"])

    plt.figure()
    shap.summary_plot(phi, X_te, show=False, max_display=16)
    plt.title(f"SHAP (positive class) - {model_name}, unseen speakers")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=160)
    plt.close()

    plt.figure()
    shap.summary_plot(phi, X_te, plot_type="bar", show=False, max_display=16)
    plt.title(f"Mean |SHAP| - {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar.png", dpi=160)
    plt.close()

    # ------------------------------------------- Permutation importance
    perm = permutation_importance(model, X.values[te], y_te, n_repeats=50,
                                  random_state=SEED, scoring="roc_auc",
                                  n_jobs=-1)
    perm_df = (pd.DataFrame({"feature": X.columns,
                             "importance": perm.importances_mean,
                             "std": perm.importances_std})
               .sort_values("importance", ascending=False))
    perm_df.to_csv(TABLES_DIR / "permutation_importance.csv", index=False)

    plt.figure(figsize=(7, 6))
    top = perm_df.head(16).iloc[::-1]
    plt.barh(top.feature, top.importance, xerr=top["std"], color="#3b6ea5")
    plt.xlabel("Drop in ROC-AUC when permuted")
    plt.title(f"Permutation importance - {model_name}, unseen speakers")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "permutation_importance.png", dpi=160)
    plt.close()

    # Agreement between the two attribution methods
    common = perm_df.set_index("feature")["importance"]
    report["spearman_shap_vs_permutation"] = float(
        pd.Series(shap_rank).corr(common, method="spearman"))
    report["top5_shap"] = shap_rank.head(5).round(4).to_dict()
    report["top5_permutation"] = (perm_df.set_index("feature")["importance"]
                                  .head(5).round(4).to_dict())

    # ---------------------------------------------------------------- LIME
    try:
        from lime.lime_tabular import LimeTabularExplainer
        lime_exp = LimeTabularExplainer(
            X_tr.values, feature_names=list(X.columns),
            class_names=["Healthy", "Parkinson's"], mode="classification",
            discretize_continuous=True, random_state=SEED)

        y_prob = model.predict_proba(X.values[te])[:, 1]
        picks = {"most_confident_PD": int(np.argmax(y_prob)),
                 "most_confident_healthy": int(np.argmin(y_prob)),
                 "most_uncertain": int(np.argmin(np.abs(y_prob - 0.5)))}
        lime_rows = []
        for label, i in picks.items():
            exp = lime_exp.explain_instance(
                X.values[te][i], model.predict_proba, num_features=8)
            for feat, weight in exp.as_list():
                lime_rows.append({"case": label, "true_label": int(y_te[i]),
                                  "p_pd": round(float(y_prob[i]), 4),
                                  "rule": feat, "weight": round(weight, 4)})
        pd.DataFrame(lime_rows).to_csv(TABLES_DIR / "lime_cases.csv",
                                       index=False)
        report["lime_cases"] = picks
    except Exception as exc:  # pragma: no cover
        report["lime_error"] = str(exc)

    (TABLES_DIR.parent / "explainability_report.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def stability(model_name="Random Forest", n_splits=4, n_repeats=3):
    """How reproducible is the feature ranking across speaker-disjoint folds?

    A single explanation computed on one fold of 8 speakers is an anecdote.
    This refits and re-explains across many folds and reports the pairwise
    Spearman agreement between the resulting rankings.
    """
    import shap
    from itertools import combinations

    X, y, groups, _ = load_dataset()
    rankings = []
    for rep in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                  random_state=SEED + rep)
        for tr, te in cv.split(X, y, groups):
            model, _ = fit_best(model_name, X, y, groups, tr)
            phi = positive_class_shap(
                shap.TreeExplainer(model.named_steps["clf"])(
                    pd.DataFrame(X.values[te], columns=X.columns)))
            rankings.append(pd.Series(np.abs(phi).mean(axis=0),
                                      index=X.columns))

    R = pd.concat(rankings, axis=1)
    R.columns = [f"fold{i}" for i in range(R.shape[1])]
    rhos = [R[a].corr(R[b], method="spearman") for a, b in combinations(R.columns, 2)]

    summary = pd.DataFrame({
        "mean_abs_shap": R.mean(axis=1),
        "sd_across_folds": R.std(axis=1),
        "mean_rank": R.rank(ascending=False).mean(axis=1),
        "rank_sd": R.rank(ascending=False).std(axis=1),
    }).sort_values("mean_rank")
    summary.to_csv(TABLES_DIR / "shap_stability.csv")

    plt.figure(figsize=(8, 6))
    top = summary.head(16).iloc[::-1]
    plt.barh(top.index, top.mean_abs_shap, xerr=top.sd_across_folds,
             color="#6a9c78")
    plt.xlabel("mean |SHAP| (positive class), averaged over folds")
    plt.title(f"Feature importance with fold-to-fold uncertainty\n"
              f"{model_name}, {len(rankings)} speaker-disjoint folds")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_stability.png", dpi=160)
    plt.close()

    out = {"n_folds": len(rankings),
           "mean_pairwise_spearman": float(np.mean(rhos)),
           "min_pairwise_spearman": float(np.min(rhos)),
           "top5_stable": summary.head(5).round(4).to_dict("index")}
    print(json.dumps(out, indent=2))
    (TABLES_DIR.parent / "shap_stability_report.json").write_text(
        json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import sys
    if "--stability" in sys.argv:
        stability()
    else:
        run()
