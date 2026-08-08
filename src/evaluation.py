"""Evaluation protocols.

Two protocols are implemented so that the cost of the methodological error
can be measured rather than merely asserted:

  * `record`  - StratifiedKFold over individual recordings. Recordings from
                the same speaker appear in both train and test. This is the
                protocol used in most published notebooks on this dataset,
                and it is optimistically biased.
  * `subject` - StratifiedGroupKFold over speakers. No speaker is ever split
                across the boundary. This estimates what everyone actually
                cares about: performance on a person the model has never heard.

Hyper-parameters are selected by an inner CV that uses the *same* grouping
rule as the outer loop, so the protocol is honest end to end.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, roc_auc_score)

from config import INNER_SPLITS, N_REPEATS, N_SPLITS, SEED

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def make_outer_splits(protocol, X, y, groups, n_splits=N_SPLITS,
                      n_repeats=N_REPEATS, seed=SEED):
    """Yield (repeat, fold, train_idx, test_idx) for the chosen protocol."""
    for rep in range(n_repeats):
        rs = seed + rep
        if protocol == "record":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                 random_state=rs)
            splitter = cv.split(X, y)
        elif protocol == "subject":
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                      random_state=rs)
            splitter = cv.split(X, y, groups)
        else:
            raise ValueError(protocol)
        for fold, (tr, te) in enumerate(splitter):
            yield rep, fold, tr, te


def _inner_cv(protocol, seed):
    if protocol == "record":
        return StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True,
                               random_state=seed)
    return StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True,
                                random_state=seed)


def _scores(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


def _predict_prob(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def nested_cv(name, pipeline, grid, X, y, groups, protocol,
              n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED,
              collect_predictions=False):
    """Run repeated nested CV and return a tidy per-fold DataFrame."""
    rows, preds = [], []
    Xv, yv, gv = X.values, y.values, groups.values

    for rep, fold, tr, te in make_outer_splits(protocol, X, y, groups,
                                               n_splits, n_repeats, seed):
        est = clone(pipeline)
        if grid:
            search = GridSearchCV(
                est, grid, scoring="roc_auc",
                cv=_inner_cv(protocol, seed + rep), n_jobs=-1,
                refit=True, error_score="raise")
            fit_kw = {"groups": gv[tr]} if protocol == "subject" else {}
            search.fit(Xv[tr], yv[tr], **fit_kw)
            model, best = search.best_estimator_, search.best_params_
        else:
            est.fit(Xv[tr], yv[tr])
            model, best = est, {}

        y_pred = model.predict(Xv[te])
        y_prob = _predict_prob(model, Xv[te])
        row = {"model": name, "protocol": protocol, "repeat": rep,
               "fold": fold, "n_test": len(te),
               "n_test_subjects": len(np.unique(gv[te])),
               "best_params": str(best)}
        row.update(_scores(yv[te], y_pred, y_prob))
        rows.append(row)

        if collect_predictions and y_prob is not None:
            preds.append(pd.DataFrame({
                "model": name, "protocol": protocol, "repeat": rep,
                "fold": fold, "subject": gv[te], "y_true": yv[te],
                "y_prob": y_prob, "y_pred": y_pred}))

    df = pd.DataFrame(rows)
    return (df, pd.concat(preds, ignore_index=True)) if collect_predictions else (df, None)


def subject_level_scores(pred_df):
    """Aggregate recording-level probabilities into one decision per speaker.

    Clinically the unit of interest is the person, not the utterance. Mean
    probability across a speaker's recordings is the natural aggregation.
    """
    rows = []
    for (model, protocol, rep), chunk in pred_df.groupby(
            ["model", "protocol", "repeat"]):
        agg = chunk.groupby("subject").agg(
            y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
        y_pred = (agg.y_prob >= 0.5).astype(int)
        s = _scores(agg.y_true.values, y_pred.values, agg.y_prob.values)
        s.update({"model": model, "protocol": protocol, "repeat": rep,
                  "n_subjects": len(agg)})
        rows.append(s)
    return pd.DataFrame(rows)


def summarise(df, metrics=None):
    """Mean +/- std and a 95% normal-approximation interval per model."""
    metrics = metrics or ["accuracy", "balanced_accuracy", "roc_auc",
                          "pr_auc", "f1", "mcc", "sensitivity", "specificity"]
    g = df.groupby(["protocol", "model"])[metrics]
    out = g.agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    n = df.groupby(["protocol", "model"]).size().rename("n_folds")
    return out.join(n).reset_index()
