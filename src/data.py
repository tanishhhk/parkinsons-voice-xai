"""Data loading, subject-ID recovery, and redundancy pruning.

The single most important function here is `add_subject_column`. The UCI
Parkinson's dataset contains multiple sustained-phonation recordings per
person, encoded in the `name` field as `phon_R01_S<subject>_<recording>`.
Any evaluation that ignores this grouping leaks subject identity between
train and test folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CORR_THRESHOLD, DATA_FILE, ID_COL, SUBJECT_COL, TARGET

SUBJECT_PATTERN = r"phon_R01_S(\d+)_\d+"


def load_raw(path=DATA_FILE) -> pd.DataFrame:
    """Load the raw CSV with the `name` column preserved."""
    df = pd.read_csv(path)
    if df[ID_COL].duplicated().any():
        raise ValueError("Duplicate recording IDs found.")
    return df


def add_subject_column(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the subject ID out of the recording name."""
    df = df.copy()
    subj = df[ID_COL].str.extract(SUBJECT_PATTERN)[0]
    if subj.isna().any():
        bad = df.loc[subj.isna(), ID_COL].tolist()
        raise ValueError(f"Could not parse subject ID from: {bad[:5]}")
    df[SUBJECT_COL] = "S" + subj
    return df


def check_label_consistency(df: pd.DataFrame) -> None:
    """A subject must carry exactly one diagnosis across their recordings."""
    per_subject = df.groupby(SUBJECT_COL)[TARGET].nunique()
    if (per_subject > 1).any():
        raise ValueError("Inconsistent labels within a subject.")


def find_redundant_features(X: pd.DataFrame, threshold=CORR_THRESHOLD):
    """Greedily drop one feature from every near-perfectly correlated pair.

    Several columns are deterministic multiples of others (Jitter:DDP is
    3 x MDVP:RAP; Shimmer:DDA is 3 x Shimmer:APQ3), which splits importance
    between duplicates and destabilises attribution methods.
    """
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    dropped = {}
    for col in upper.columns:
        partners = upper.index[upper[col] > threshold].tolist()
        if partners:
            keep = partners[0]
            if col not in dropped:
                dropped[col] = (keep, float(upper.loc[keep, col]))
    return dropped


def load_dataset(drop_redundant: bool = True):
    """Return (X, y, groups, meta) ready for grouped cross-validation."""
    df = add_subject_column(load_raw())
    check_label_consistency(df)

    y = df[TARGET].astype(int)
    groups = df[SUBJECT_COL]
    X = df.drop(columns=[ID_COL, TARGET, SUBJECT_COL])

    dropped = find_redundant_features(X)
    if drop_redundant and dropped:
        X = X.drop(columns=list(dropped))

    meta = {
        "n_records": int(len(df)),
        "n_subjects": int(groups.nunique()),
        "n_features": int(X.shape[1]),
        "records_per_subject": groups.value_counts().describe().to_dict(),
        "record_class_balance": y.value_counts().to_dict(),
        "subject_class_balance": df.groupby(SUBJECT_COL)[TARGET]
        .first()
        .value_counts()
        .to_dict(),
        "dropped_features": {k: {"correlated_with": v[0], "r": round(v[1], 6)}
                             for k, v in dropped.items()},
    }
    return X, y, groups, meta


if __name__ == "__main__":
    import json
    X, y, g, meta = load_dataset()
    print(json.dumps(meta, indent=2, default=str))
