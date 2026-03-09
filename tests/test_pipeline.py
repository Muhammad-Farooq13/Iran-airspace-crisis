"""
Unit and integration tests for the data pipeline.
Run with: pytest tests/ -v
"""

import pandas as pd
import pytest
from pathlib import Path

from src.config import RAW_FILES
from src.data.pipeline import (
    _clean_airline_losses,
    _clean_airspace_closures,
    _clean_airport_disruptions,
    _clean_conflict_events,
    _clean_flight_cancellations,
    _clean_flight_reroutes,
)


@pytest.fixture
def airline_losses_df():
    return _clean_airline_losses(RAW_FILES["airline_losses"])


@pytest.fixture
def conflict_events_df():
    return _clean_conflict_events(RAW_FILES["conflict_events"])


@pytest.fixture
def airspace_closures_df():
    return _clean_airspace_closures(RAW_FILES["airspace_closures"])


@pytest.fixture
def airport_disruptions_df():
    return _clean_airport_disruptions(RAW_FILES["airport_disruptions"])


@pytest.fixture
def flight_reroutes_df():
    return _clean_flight_reroutes(RAW_FILES["flight_reroutes"])


@pytest.fixture
def flight_cancellations_df():
    return _clean_flight_cancellations(RAW_FILES["flight_cancellations"])


# ── airline losses ─────────────────────────────────────────────────────────────

class TestAirlineLossesCleaner:
    def test_no_missing_loss(self, airline_losses_df):
        assert airline_losses_df["estimated_daily_loss_usd"].isna().sum() == 0

    def test_fuel_cost_ratio_between_0_and_1(self, airline_losses_df):
        assert (airline_losses_df["fuel_cost_ratio"].dropna() <= 1).all()

    def test_reroute_ratio_between_0_and_1(self, airline_losses_df):
        r = airline_losses_df["reroute_ratio"]
        assert (r >= 0).all() and (r <= 1).all()

    def test_loss_per_passenger_positive(self, airline_losses_df):
        assert (airline_losses_df["loss_per_passenger"] >= 0).all()


# ── conflict events ────────────────────────────────────────────────────────────

class TestConflictEventsCleaner:
    def test_datetime_utc_parsed(self, conflict_events_df):
        assert pd.api.types.is_datetime64_any_dtype(conflict_events_df["datetime_utc"])

    def test_severity_score_range(self, conflict_events_df):
        scores = conflict_events_df["severity_score"]
        assert scores.between(1, 4).all()

    def test_crisis_day_positive(self, conflict_events_df):
        assert (conflict_events_df["crisis_day"] >= 1).all()

    def test_fir_impact_binary(self, conflict_events_df):
        assert set(conflict_events_df["fir_impact"].unique()).issubset({0, 1})


# ── airspace closures ──────────────────────────────────────────────────────────

class TestAirspaceClosuresCleaner:
    def test_duration_positive(self, airspace_closures_df):
        assert (airspace_closures_df["closure_duration_hours"] > 0).all()

    def test_closure_type_valid(self, airspace_closures_df):
        valid = {"active_conflict", "precautionary", "spillover", "other"}
        assert set(airspace_closures_df["closure_type"].unique()).issubset(valid)

    def test_primary_conflict_flag_binary(self, airspace_closures_df):
        assert set(airspace_closures_df["is_primary_conflict_country"].unique()).issubset({0, 1})


# ── airport disruptions ────────────────────────────────────────────────────────

class TestAirportDisruptionsCleaner:
    def test_total_disrupted_sum(self, airport_disruptions_df):
        expected = (
            airport_disruptions_df["flights_cancelled"]
            + airport_disruptions_df["flights_delayed"]
            + airport_disruptions_df["flights_diverted"]
        )
        pd.testing.assert_series_equal(
            airport_disruptions_df["total_disrupted"],
            expected,
            check_names=False,
        )

    def test_runway_severity_range(self, airport_disruptions_df):
        assert airport_disruptions_df["runway_severity"].between(0, 5).all()


# ── flight reroutes ────────────────────────────────────────────────────────────

class TestFlightReroutesCleaner:
    def test_cost_per_km_positive(self, flight_reroutes_df):
        assert (flight_reroutes_df["cost_per_km"].dropna() > 0).all()

    def test_delay_category_valid(self, flight_reroutes_df):
        assert set(flight_reroutes_df["delay_category"].unique()).issubset({"low", "medium", "high"})

    def test_crisis_day_positive(self, flight_reroutes_df):
        assert (flight_reroutes_df["crisis_day"].dropna() >= 1).all()


# ── flight cancellations ───────────────────────────────────────────────────────

class TestFlightCancellationsCleaner:
    def test_date_parsed(self, flight_cancellations_df):
        assert pd.api.types.is_datetime64_any_dtype(flight_cancellations_df["date"])

    def test_cancellation_category_valid(self, flight_cancellations_df):
        valid = {"airspace_closed", "precautionary", "destination_closed", "other"}
        assert set(flight_cancellations_df["cancellation_category"].unique()).issubset(valid)

    def test_wide_body_flag_binary(self, flight_cancellations_df):
        assert set(flight_cancellations_df["is_wide_body"].unique()).issubset({0, 1})
