"""
Model prediction utilities — used by the API and notebooks.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_LIST_PATH, MODEL_PATH, SCALER_PATH

log = logging.getLogger(__name__)


def load_artefacts(
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    feature_list_path: Path = FEATURE_LIST_PATH,
) -> tuple[object, object, list[str]]:
    """Load and return (model, scaler, feature_names)."""
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)
    with open(feature_list_path) as fh:
        feature_names = json.load(fh)
    return model, scaler, feature_names


def predict(
    input_data: dict | pd.DataFrame,
    model=None,
    scaler=None,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """
    Predict daily airline loss (USD) for one or more records.

    Parameters
    ----------
    input_data : dict or DataFrame
        Feature values. Missing features are filled with 0.
    model, scaler, feature_names:
        If None, loaded from disk on every call (development convenience).
        Pass pre-loaded artefacts for production / API use.

    Returns
    -------
    np.ndarray of predicted values in USD.
    """
    if model is None or scaler is None or feature_names is None:
        model, scaler, feature_names = load_artefacts()

    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # Align to expected feature list
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_names].fillna(0)

    X_scaled = scaler.transform(df)
    return model.predict(X_scaled)
