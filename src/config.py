"""
Central configuration for the Iran Airspace Crisis Impact Analysis project.
All paths, constants, and tunable settings live here.
"""

from pathlib import Path

# ── Repo root ─────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]

# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR          = ROOT_DIR / "data"
RAW_DIR           = DATA_DIR / "raw"
PROCESSED_DIR     = DATA_DIR / "processed"
FEATURES_DIR      = DATA_DIR / "features"

# ── Artefact directories ──────────────────────────────────────────────────────
MODELS_DIR        = ROOT_DIR / "models"
REPORTS_DIR       = ROOT_DIR / "reports"
FIGURES_DIR       = REPORTS_DIR / "figures"

# ── Raw file names ─────────────────────────────────────────────────────────────
RAW_FILES = {
    "conflict_events":        RAW_DIR / "conflict_events.csv",
    "airspace_closures":      RAW_DIR / "airspace_closures.csv",
    "airport_disruptions":    RAW_DIR / "airport_disruptions.csv",
    "flight_cancellations":   RAW_DIR / "flight_cancellations.csv",
    "flight_reroutes":        RAW_DIR / "flight_reroutes.csv",
    "airline_losses":         RAW_DIR / "airline_losses_estimate.csv",
}

# ── Processed file names ───────────────────────────────────────────────────────
PROCESSED_FILES = {
    "conflict_events":        PROCESSED_DIR / "conflict_events_clean.parquet",
    "airspace_closures":      PROCESSED_DIR / "airspace_closures_clean.parquet",
    "airport_disruptions":    PROCESSED_DIR / "airport_disruptions_clean.parquet",
    "flight_cancellations":   PROCESSED_DIR / "flight_cancellations_clean.parquet",
    "flight_reroutes":        PROCESSED_DIR / "flight_reroutes_clean.parquet",
    "airline_losses":         PROCESSED_DIR / "airline_losses_clean.parquet",
    "master":                 PROCESSED_DIR / "master_dataset.parquet",
}

FEATURE_MATRIX_PATH = FEATURES_DIR / "feature_matrix.parquet"

# ── Model artefact names ───────────────────────────────────────────────────────
MODEL_PATH        = MODELS_DIR / "airline_loss_regressor.pkl"
SCALER_PATH       = MODELS_DIR / "scaler.pkl"
FEATURE_LIST_PATH = MODELS_DIR / "feature_list.json"

# ── Modelling constants ────────────────────────────────────────────────────────
RANDOM_STATE      = 42
TEST_SIZE         = 0.20
CV_FOLDS          = 5

TARGET_COLUMN     = "estimated_daily_loss_usd"

# ── API settings ──────────────────────────────────────────────────────────────
API_HOST          = "0.0.0.0"
API_PORT          = 8000
API_VERSION       = "v1"
