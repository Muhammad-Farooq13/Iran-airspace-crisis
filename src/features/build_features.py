"""
Feature engineering pipeline.

Reads the master_dataset.parquet produced by src.data.pipeline and
engineers the final feature matrix used by the models.

Usage:
    python -m src.features.build_features
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_MATRIX_PATH,
    FEATURE_LIST_PATH,
    FEATURES_DIR,
    MODELS_DIR,
    PROCESSED_FILES,
    SCALER_PATH,
    TARGET_COLUMN,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Feature definitions
# ──────────────────────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    # Direct airline operation metrics
    "cancelled_flights",
    "rerouted_flights",
    "additional_fuel_cost_usd",
    "passengers_impacted",
    # Derived airline ratios (from cleaning)
    "fuel_cost_ratio",
    "reroute_ratio",
    "loss_per_passenger",
    # Reroute characteristics
    "avg_extra_km",
    "avg_delay_min",
    "avg_cost_per_km",
    "total_reroutes",
    # Cancellation characteristics
    "total_recorded_cancellations",
    "wide_body_cancellations",
    # Crisis-wide context
    "n_primary_closed_firs",
    "n_total_closures",
    "avg_closure_hours",
    "precautionary_pct",
    "avg_airport_runway_sev",
    "total_airport_disrupted",
    "early_critical_events",
    "total_conflict_events",
]

CATEGORICAL_FEATURES = ["country"]


def build_features(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns (X, y) where X is the scaled feature matrix and y is the target.
    """
    df = master.copy()

    # ── Interaction features ──────────────────────────────────────────────────
    df["cancelled_x_widebody"] = df["cancelled_flights"] * df["wide_body_cancellations"].fillna(0)
    df["disrupted_per_route"]  = (
        df["total_airport_disrupted"] / (df["rerouted_flights"] + 1)
    ).round(4)
    df["fuel_x_distance"] = df["additional_fuel_cost_usd"] * df["avg_extra_km"].fillna(0)

    # ── Log-transforms for right-skewed features ──────────────────────────────
    log_cols = [
        "additional_fuel_cost_usd",
        "passengers_impacted",
        "fuel_x_distance",
    ]
    for col in log_cols:
        df[f"log_{col}"] = np.log1p(df[col].fillna(0))

    # ── One-hot encode country ────────────────────────────────────────────────
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=int)

    # ── Assemble feature list ──────────────────────────────────────────────────
    extra_features = (
        ["cancelled_x_widebody", "disrupted_per_route", "fuel_x_distance"]
        + [f"log_{c}" for c in log_cols]
        + [c for c in df.columns if c.startswith("country_")]
    )
    all_features = NUMERIC_FEATURES + extra_features

    # Keep only columns that exist (fill missing with 0)
    available = [f for f in all_features if f in df.columns]
    X = df[available].fillna(0)
    y = df[TARGET_COLUMN]

    log.info("Feature matrix: %d samples × %d features", *X.shape)
    return X, y


def scale_and_save(X: pd.DataFrame) -> pd.DataFrame:
    """Fit a StandardScaler on X, persist it, return scaled DataFrame."""
    import pickle

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    with open(SCALER_PATH, "wb") as fh:
        pickle.dump(scaler, fh)

    # Persist feature list
    with open(FEATURE_LIST_PATH, "w") as fh:
        json.dump(list(X.columns), fh, indent=2)

    log.info("Scaler saved → %s", SCALER_PATH)
    log.info("Feature list saved → %s", FEATURE_LIST_PATH)
    return X_scaled


def run_feature_pipeline() -> tuple[pd.DataFrame, pd.Series]:
    master = pd.read_parquet(PROCESSED_FILES["master"])
    X, y = build_features(master)
    X_scaled = scale_and_save(X)
    X_scaled[TARGET_COLUMN] = y.values
    X_scaled.to_parquet(FEATURE_MATRIX_PATH, index=False)
    log.info("Feature matrix saved → %s", FEATURE_MATRIX_PATH)
    # Return unscaled X for interpretability + y
    return X, y


if __name__ == "__main__":
    run_feature_pipeline()
