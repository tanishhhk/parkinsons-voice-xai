"""Central configuration: paths, seeds, and experiment constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

for _d in (RESULTS_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "parkinsons.data"

SEED = 42
# 4 outer folds keeps >=2 healthy speakers in every subject-grouped test fold
# (there are only 8 healthy speakers in total).
N_SPLITS = 4          # outer CV folds
N_REPEATS = 10        # outer CV repeats (different shuffles)
INNER_SPLITS = 3      # inner CV folds for hyper-parameter search

TARGET = "status"
ID_COL = "name"
SUBJECT_COL = "subject"

# Correlation threshold above which one member of a feature pair is dropped.
CORR_THRESHOLD = 0.98
