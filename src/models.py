"""Model zoo: every estimator is wrapped in a Pipeline so that scaling is
fitted inside each CV fold, never on the full dataset."""
from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import SEED

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False


def _pipe(estimator, scale=True):
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def get_model_zoo():
    """Return {name: (pipeline, param_grid)}.

    `class_weight='balanced'` is used wherever supported: the record-level
    prior is 75% positive, so an unweighted model is rewarded for guessing.
    """
    zoo = {}

    zoo["Baseline (majority)"] = (
        _pipe(DummyClassifier(strategy="most_frequent"), scale=False), {})

    zoo["Logistic Regression"] = (
        _pipe(LogisticRegression(max_iter=5000, class_weight="balanced",
                                 random_state=SEED)),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]})

    zoo["LDA"] = (
        _pipe(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")), {})

    zoo["Gaussian NB"] = (
        _pipe(GaussianNB()),
        {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]})

    zoo["k-NN"] = (
        _pipe(KNeighborsClassifier()),
        {"clf__n_neighbors": [3, 5, 9, 15],
         "clf__weights": ["uniform", "distance"]})

    zoo["SVM (linear)"] = (
        _pipe(SVC(kernel="linear", probability=True,
                  class_weight="balanced", random_state=SEED)),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]})

    zoo["SVM (RBF)"] = (
        _pipe(SVC(kernel="rbf", probability=True,
                  class_weight="balanced", random_state=SEED)),
        {"clf__C": [0.1, 1.0, 10.0, 100.0],
         "clf__gamma": ["scale", 0.01, 0.1]})

    zoo["Random Forest"] = (
        _pipe(RandomForestClassifier(n_estimators=300, n_jobs=1,
                                     class_weight="balanced_subsample",
                                     random_state=SEED), scale=False),
        {"clf__max_depth": [None, 6],
         "clf__min_samples_leaf": [1, 3]})

    zoo["Extra Trees"] = (
        _pipe(ExtraTreesClassifier(n_estimators=300, n_jobs=1,
                                   class_weight="balanced",
                                   random_state=SEED), scale=False),
        {"clf__max_depth": [None, 6],
         "clf__min_samples_leaf": [1, 3]})

    zoo["Gradient Boosting"] = (
        _pipe(GradientBoostingClassifier(random_state=SEED), scale=False),
        {"clf__n_estimators": [200],
         "clf__learning_rate": [0.05, 0.1],
         "clf__max_depth": [2, 3]})

    if HAS_XGB:
        zoo["XGBoost"] = (
            _pipe(XGBClassifier(eval_metric="logloss", n_jobs=1,
                                tree_method="hist", random_state=SEED),
                  scale=False),
            {"clf__n_estimators": [300],
             "clf__learning_rate": [0.05, 0.1],
             "clf__max_depth": [2, 4],
             "clf__subsample": [0.8]})

    zoo["MLP"] = (
        _pipe(MLPClassifier(max_iter=1500, random_state=SEED)),
        {"clf__hidden_layer_sizes": [(32,), (64, 32)],
         "clf__alpha": [1e-2]})

    return zoo
