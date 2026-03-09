"""
Tests for feature engineering and model predict utilities.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import run_full_pipeline
from src.features.build_features import build_features


@pytest.fixture(scope="module")
def master_df():
    return run_full_pipeline()


class TestBuildFeatures:
    def test_returns_dataframe_and_series(self, master_df):
        X, y = build_features(master_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_no_missing_in_X(self, master_df):
        X, _ = build_features(master_df)
        assert X.isna().sum().sum() == 0

    def test_y_positive(self, master_df):
        _, y = build_features(master_df)
        assert (y > 0).all()

    def test_log_features_present(self, master_df):
        X, _ = build_features(master_df)
        log_cols = [c for c in X.columns if c.startswith("log_")]
        assert len(log_cols) > 0

    def test_interaction_features_present(self, master_df):
        X, _ = build_features(master_df)
        assert "cancelled_x_widebody" in X.columns
