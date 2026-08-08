"""Run the benchmark: every model x both evaluation protocols.

The work is checkpointed per (protocol, model) so that a long run can be
resumed or parallelised across sessions.

    python run_experiments.py --list
    python run_experiments.py --protocol subject --models "Random Forest"
    python run_experiments.py                       # everything not yet done
    python run_experiments.py --combine             # build summary tables
"""
from __future__ import annotations

import argparse
import json
import re
import time

import pandas as pd

from config import N_REPEATS, N_SPLITS, RESULTS_DIR, TABLES_DIR
from data import load_dataset
from evaluation import nested_cv, subject_level_scores, summarise
from models import get_model_zoo

PARTS_DIR = TABLES_DIR / "parts"
PARTS_DIR.mkdir(parents=True, exist_ok=True)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def run_one(name, protocol, X, y, groups, force=False):
    fold_path = PARTS_DIR / f"folds__{protocol}__{_slug(name)}.csv"
    pred_path = PARTS_DIR / f"preds__{protocol}__{_slug(name)}.csv"
    if fold_path.exists() and not force:
        return pd.read_csv(fold_path), False

    pipe, grid = get_model_zoo()[name]
    t0 = time.time()
    folds, preds = nested_cv(name, pipe, grid, X, y, groups, protocol,
                             collect_predictions=True)
    folds.to_csv(fold_path, index=False)
    if preds is not None:
        preds.to_csv(pred_path, index=False)
    print(f"{protocol:8s} {name:22s} acc {folds.accuracy.mean():.3f}"
          f"  bal-acc {folds.balanced_accuracy.mean():.3f}"
          f"  auc {folds.roc_auc.mean():.3f}"
          f"  mcc {folds.mcc.mean():.3f}   [{time.time() - t0:6.1f}s]",
          flush=True)
    return folds, True


def combine():
    folds = pd.concat([pd.read_csv(p) for p in sorted(PARTS_DIR.glob("folds__*.csv"))],
                      ignore_index=True)
    preds = pd.concat([pd.read_csv(p) for p in sorted(PARTS_DIR.glob("preds__*.csv"))],
                      ignore_index=True)
    folds.to_csv(TABLES_DIR / "cv_folds.csv", index=False)
    preds.to_csv(TABLES_DIR / "cv_predictions.csv", index=False)

    summarise(folds).to_csv(TABLES_DIR / "summary_recording_level.csv", index=False)

    subj = subject_level_scores(preds[preds.protocol == "subject"])
    subj.to_csv(TABLES_DIR / "subject_level_folds.csv", index=False)
    summarise(subj).to_csv(TABLES_DIR / "summary_subject_level.csv", index=False)

    print(f"combined {folds.model.nunique()} models x "
          f"{folds.protocol.nunique()} protocols "
          f"({len(folds)} folds) -> {TABLES_DIR}")
    return folds, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["record", "subject"], default=None)
    ap.add_argument("--models", default=None,
                    help="comma-separated model names")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    zoo = get_model_zoo()
    if args.list:
        for k in zoo:
            print(k)
        return
    if args.combine:
        combine()
        return

    X, y, groups, meta = load_dataset()
    (RESULTS_DIR / "dataset_meta.json").write_text(
        json.dumps(meta, indent=2, default=str))

    names = ([m.strip() for m in args.models.split(",")]
             if args.models else list(zoo))
    protocols = [args.protocol] if args.protocol else ["record", "subject"]
    print(f"{N_SPLITS}-fold x {N_REPEATS} repeats nested CV", flush=True)
    for protocol in protocols:
        for name in names:
            run_one(name, protocol, X, y, groups, force=args.force)


if __name__ == "__main__":
    main()
