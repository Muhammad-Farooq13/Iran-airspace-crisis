"""
Model training, cross-validation, and evaluation pipeline.

Trains a suite of regression models to predict estimated daily airline
losses (USD) from operational and crisis-context features, then selects
and persists the best model.

Usage:
    python -m src.models.train
"""

from __future__ import annotations

import json
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    CV_FOLDS,
    FEATURE_LIST_PATH,
    FEATURE_MATRIX_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    SCALER_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────

CANDIDATE_MODELS: dict[str, object] = {
    "ridge": Ridge(alpha=1.0),
    "elastic_net": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
    "random_forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=RANDOM_STATE,
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse":  round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "mae":   round(float(mean_absolute_error(y_true, y_pred)), 2),
        "mape":  round(float(mean_absolute_percentage_error(y_true, y_pred)) * 100, 4),
        "r2":    round(float(r2_score(y_true, y_pred)), 6),
    }


def _cross_validate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = CV_FOLDS,
) -> dict[str, float]:
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
    return {
        "cv_r2_mean": round(float(scores.mean()), 6),
        "cv_r2_std":  round(float(scores.std()),  6),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Feature importance
# ──────────────────────────────────────────────────────────────────────────────

def _feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        return pd.DataFrame()

    return (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────────────────────────────────────

def train(
    feature_matrix_path: Path = FEATURE_MATRIX_PATH,
) -> tuple[object, dict]:
    """
    Load feature matrix, train all candidate models, select the best by
    CV R², persist it to disk, and return (best_model, results_dict).
    """
    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_parquet(feature_matrix_path)
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    log.info("Training on %d samples × %d features", *X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ── Fit scaler on training set only ──────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Train and evaluate all candidates ─────────────────────────────────────
    results: dict[str, dict] = {}
    trained_models: dict[str, object] = {}

    for name, model in CANDIDATE_MODELS.items():
        log.info("Training: %s", name)
        model.fit(X_train_s, y_train)
        y_pred_train = model.predict(X_train_s)
        y_pred_test  = model.predict(X_test_s)

        cv_metrics = _cross_validate(model, X_train_s, y_train)
        train_metrics = _evaluate(y_train.values, y_pred_train)
        test_metrics  = _evaluate(y_test.values,  y_pred_test)

        results[name] = {
            "train": train_metrics,
            "test":  test_metrics,
            "cv":    cv_metrics,
        }
        trained_models[name] = model

        log.info(
            "  Train R²=%.4f | Test R²=%.4f | CV R²=%.4f±%.4f",
            train_metrics["r2"],
            test_metrics["r2"],
            cv_metrics["cv_r2_mean"],
            cv_metrics["cv_r2_std"],
        )

    # ── Select best model by CV R² ────────────────────────────────────────────
    best_name = max(results, key=lambda n: results[n]["cv"]["cv_r2_mean"])
    best_model = trained_models[best_name]
    log.info("Best model: %s (CV R²=%.4f)", best_name, results[best_name]["cv"]["cv_r2_mean"])

    # ── Feature importances ───────────────────────────────────────────────────
    fi_df = _feature_importance(best_model, list(X.columns))
    fi_path = MODELS_DIR / "feature_importances.csv"
    fi_df.to_csv(fi_path, index=False)
    log.info("Feature importances saved → %s", fi_path)

    # ── Persist artefacts ────────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(best_model, fh)
    with open(SCALER_PATH, "wb") as fh:
        pickle.dump(scaler, fh)
    with open(FEATURE_LIST_PATH, "w") as fh:
        json.dump(list(X.columns), fh, indent=2)

    # Save results summary
    results_path = MODELS_DIR / "training_results.json"
    with open(results_path, "w") as fh:
        json.dump({"best_model": best_name, "results": results}, fh, indent=2)

    log.info("Model saved → %s", MODEL_PATH)
    log.info("Results saved → %s", results_path)

    return best_model, {"best_model": best_name, "results": results}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
