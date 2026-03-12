"""
train_demo.py — lightweight demo model for the Streamlit dashboard.

Trains a GradientBoostingRegressor directly from the raw
airline_losses_estimate.csv file (no full pipeline required).
Saves models/iran_demo.pkl which the Streamlit app loads at startup.

Usage:
    python train_demo.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

RAW_LOSSES = Path("data/raw/airline_losses_estimate.csv")
DEMO_PKL   = Path("models/iran_demo.pkl")
RANDOM_STATE = 42


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Derive features from raw airline losses CSV."""
    X = df.copy()

    # Derived ratios
    total_ops = X["cancelled_flights"] + X["rerouted_flights"] + 1
    X["fuel_cost_ratio"]   = X["additional_fuel_cost_usd"] / total_ops
    X["reroute_ratio"]     = X["rerouted_flights"] / total_ops
    X["loss_per_passenger"] = (
        X["estimated_daily_loss_usd"] / X["passengers_impacted"].clip(lower=1)
    )
    X["log_fuel_cost"] = np.log1p(X["additional_fuel_cost_usd"])

    # Country one-hot
    country_dummies = pd.get_dummies(X["country"], prefix="country")

    numeric_cols = [
        "cancelled_flights",
        "rerouted_flights",
        "additional_fuel_cost_usd",
        "passengers_impacted",
        "fuel_cost_ratio",
        "reroute_ratio",
        "log_fuel_cost",
    ]

    y = X["estimated_daily_loss_usd"]
    X_out = pd.concat([X[numeric_cols], country_dummies], axis=1)
    return X_out, y


def main() -> None:
    print("Loading raw airline loss data...")
    df = pd.read_csv(RAW_LOSSES)
    print(f"  {len(df)} rows loaded")

    X, y = build_features(df)
    feature_names = X.columns.tolist()
    print(f"  {len(feature_names)} features: {feature_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_sc, y_train)

    # Metrics
    y_pred  = model.predict(X_test_sc)
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    X_all_sc = scaler.transform(X)
    cv_scores = cross_val_score(model, X_all_sc, y, cv=5, scoring="r2")

    metrics = {
        "test_r2":   round(float(test_r2), 4),
        "test_mae":  round(float(test_mae), 2),
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std":  round(float(cv_scores.std()),  4),
    }
    print(f"  Test R²={metrics['test_r2']:.4f} | MAE=${metrics['test_mae']:,.0f}")
    print(f"  CV R²={metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")

    payload = {
        "model":         model,
        "scaler":        scaler,
        "features":      feature_names,
        "country_cols":  [c for c in feature_names if c.startswith("country_")],
        "metrics":       metrics,
    }

    DEMO_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(DEMO_PKL, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nDemo model saved → {DEMO_PKL}")

    # Also persist country list as JSON for Streamlit dropdown
    countries = sorted(
        c.replace("country_", "") for c in feature_names if c.startswith("country_")
    )
    with open("models/demo_countries.json", "w") as f:
        json.dump(countries, f, indent=2)
    print(f"Countries saved → models/demo_countries.json ({len(countries)} entries)")


if __name__ == "__main__":
    main()
