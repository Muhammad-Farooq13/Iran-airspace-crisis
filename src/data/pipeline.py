"""
Data ingestion, cleaning, and preprocessing pipeline.

Reads every raw CSV, applies dataset-specific cleaning rules, and writes
Parquet files to data/processed/.  A final step joins them into a
master_dataset.parquet that drives the modelling stage.

Usage (from project root):
    python -m src.data.pipeline          # full run
    python -m src.data.pipeline --step clean   # only cleaning
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_DIR,
    PROCESSED_FILES,
    RAW_FILES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Individual cleaners
# ──────────────────────────────────────────────────────────────────────────────

def _clean_conflict_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Combine date + time into a single UTC timestamp
    df["datetime_utc"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time_utc"].astype(str),
        utc=True,
        errors="coerce",
    )
    df = df.drop(columns=["date", "time_utc"])

    # Ordinal encoding for severity
    severity_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    df["severity_score"] = df["severity"].map(severity_map)

    # Boolean flag: does event directly close/restrict an FIR?
    df["fir_impact"] = df["aviation_impact"].str.lower().str.contains(
        r"clos|restrict|shutdown|suspend", regex=True
    ).astype(int)

    # Days elapsed since crisis onset (2026-02-28)
    crisis_start = pd.Timestamp("2026-02-28", tz="UTC")
    df["crisis_day"] = (df["datetime_utc"] - crisis_start).dt.days + 1

    log.info("conflict_events: %d rows, %d cols", *df.shape)
    return df


def _clean_airspace_closures(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["closure_start"] = pd.to_datetime(df["closure_start_time"], utc=True, errors="coerce")
    df["closure_end"]   = pd.to_datetime(df["closure_end_time"],   utc=True, errors="coerce")
    df = df.drop(columns=["closure_start_time", "closure_end_time"])

    # Duration in hours
    df["closure_duration_hours"] = (
        (df["closure_end"] - df["closure_start"]).dt.total_seconds() / 3600
    ).round(2)

    # Classify closure type
    def _classify_reason(r: str) -> str:
        r = r.lower()
        if "active conflict" in r or "military" in r:
            return "active_conflict"
        if "precautionary" in r:
            return "precautionary"
        if "spillover" in r:
            return "spillover"
        return "other"

    df["closure_type"] = df["closure_reason"].apply(_classify_reason)

    # Flag: directly over conflict country (Iran, Israel, Yemen)
    conflict_countries = {"iran", "israel", "yemen"}
    df["is_primary_conflict_country"] = (
        df["country"].str.lower().isin(conflict_countries)
    ).astype(int)

    log.info("airspace_closures: %d rows, %d cols", *df.shape)
    return df


def _clean_airport_disruptions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Derive total disrupted flights
    df["total_disrupted"] = (
        df["flights_cancelled"] + df["flights_delayed"] + df["flights_diverted"]
    )

    # Encode runway status
    status_map = {
        "FULLY OPERATIONAL":        0,
        "ADVISORY ACTIVE":          1,
        "PARTIALLY RESTRICTED":     2,
        "RESTRICTED - NOTAM Active": 3,
        "RESTRICTED - Conflict Proximity": 4,
        "RESTRICTED - Airspace Buffer": 4,
        "RESTRICTED - Military Sector": 4,
        "RESTRICTED - Security Alert":  4,
        "CLOSED - Conflict Zone":   5,
    }
    df["runway_severity"] = df["runway_status"].map(status_map).fillna(3).astype(int)

    # Flag: airport inside primary conflict zone
    conflict_countries = {"iran", "israel", "yemen"}
    df["in_conflict_country"] = (
        df["country"].str.lower().isin(conflict_countries)
    ).astype(int)

    log.info("airport_disruptions: %d rows, %d cols", *df.shape)
    return df


def _clean_flight_cancellations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Crisis day
    crisis_start = pd.Timestamp("2026-02-28")
    df["crisis_day"] = (df["date"] - crisis_start).dt.days + 1

    # Reason category
    def _cat_reason(r: str) -> str:
        r = r.lower()
        if "airspace closed" in r or "fir" in r:
            return "airspace_closed"
        if "precautionary" in r:
            return "precautionary"
        if "destination" in r:
            return "destination_closed"
        return "other"

    df["cancellation_category"] = df["cancellation_reason"].apply(_cat_reason)

    # Wide-body flag (proxy for revenue impact)
    wide_body_types = {"boeing 777", "boeing 787", "airbus a350", "airbus a330", "airbus a340"}
    df["is_wide_body"] = df["aircraft_type"].str.lower().apply(
        lambda x: any(wb in x for wb in wide_body_types)
    ).astype(int)

    log.info("flight_cancellations: %d rows, %d cols", *df.shape)
    return df


def _clean_flight_reroutes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Ensure numeric
    for col in ["additional_distance_km", "additional_fuel_cost_usd", "delay_minutes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cost per km (fuel efficiency ratio)
    df["cost_per_km"] = (df["additional_fuel_cost_usd"] / df["additional_distance_km"]).round(4)

    # Delay category
    def _delay_bucket(mins: float) -> str:
        if mins < 45:
            return "low"
        if mins < 90:
            return "medium"
        return "high"

    df["delay_category"] = df["delay_minutes"].apply(_delay_bucket)

    # Extract date from flight_id suffix (YYYYMMDD)
    df["date"] = pd.to_datetime(
        df["flight_id"].str.extract(r"(\d{8})$")[0], format="%Y%m%d", errors="coerce"
    )

    crisis_start = pd.Timestamp("2026-02-28")
    df["crisis_day"] = (df["date"] - crisis_start).dt.days + 1

    log.info("flight_reroutes: %d rows, %d cols", *df.shape)
    return df


def _clean_airline_losses(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    for col in [
        "estimated_daily_loss_usd",
        "cancelled_flights",
        "rerouted_flights",
        "additional_fuel_cost_usd",
        "passengers_impacted",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Revenue efficiency: fuel surcharge as % of daily loss
    df["fuel_cost_ratio"] = (
        df["additional_fuel_cost_usd"] / df["estimated_daily_loss_usd"]
    ).round(4)

    # Reroute ratio: proportion of affected flights that were rerouted vs cancelled
    df["reroute_ratio"] = (
        df["rerouted_flights"] / (df["cancelled_flights"] + df["rerouted_flights"] + 1e-9)
    ).round(4)

    # Revenue lost per passenger
    df["loss_per_passenger"] = (
        df["estimated_daily_loss_usd"] / (df["passengers_impacted"] + 1)
    ).round(2)

    log.info("airline_losses: %d rows, %d cols", *df.shape)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Master dataset builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_master(processed: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join airline-level aggregates to create one row per airline, enriched
    with crisis-level context features computed from other tables.
    """
    base = processed["airline_losses"].copy()

    # ── Reroute stats per airline ────────────────────────────────────────────
    reroutes = processed["flight_reroutes"]
    reroute_agg = (
        reroutes.groupby("airline")
        .agg(
            avg_extra_km   =("additional_distance_km", "mean"),
            avg_delay_min  =("delay_minutes",          "mean"),
            total_reroutes =("flight_id",               "count"),
            avg_cost_per_km=("cost_per_km",            "mean"),
        )
        .reset_index()
    )
    base = base.merge(reroute_agg, on="airline", how="left")

    # ── Cancellation stats per airline ──────────────────────────────────────
    cancellations = processed["flight_cancellations"]
    cancel_agg = (
        cancellations.groupby("airline")
        .agg(
            total_recorded_cancellations=("flight_number", "count"),
            wide_body_cancellations     =("is_wide_body",   "sum"),
        )
        .reset_index()
    )
    base = base.merge(cancel_agg, on="airline", how="left")

    # ── Global crisis context (scalar signals) ───────────────────────────────
    closures = processed["airspace_closures"]
    n_primary_closed   = closures["is_primary_conflict_country"].sum()
    n_total_closures   = len(closures)
    avg_closure_hours  = closures["closure_duration_hours"].mean()
    precautionary_pct  = (
        closures["closure_type"].eq("precautionary").sum() / n_total_closures
    )

    disruptions = processed["airport_disruptions"]
    avg_runway_severity = disruptions["runway_severity"].mean()
    total_disrupted     = disruptions["total_disrupted"].sum()

    base["n_primary_closed_firs"]  = int(n_primary_closed)
    base["n_total_closures"]       = int(n_total_closures)
    base["avg_closure_hours"]      = round(float(avg_closure_hours), 2)
    base["precautionary_pct"]      = round(float(precautionary_pct), 4)
    base["avg_airport_runway_sev"] = round(float(avg_runway_severity), 4)
    base["total_airport_disrupted"]= int(total_disrupted)

    # ── Conflict intensity (events from the first 3 days) ───────────────────
    events = processed["conflict_events"]
    early_events = events[events["crisis_day"] <= 3]
    base["early_critical_events"] = int(
        early_events["severity_score"].ge(4).sum()
    )
    base["total_conflict_events"]  = int(len(events))

    log.info("master_dataset: %d rows, %d cols", *base.shape)
    return base


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_cleaning() -> dict[str, pd.DataFrame]:
    """Clean all raw datasets and save to data/processed/ as Parquet."""
    cleaners = {
        "conflict_events":      _clean_conflict_events,
        "airspace_closures":    _clean_airspace_closures,
        "airport_disruptions":  _clean_airport_disruptions,
        "flight_cancellations": _clean_flight_cancellations,
        "flight_reroutes":      _clean_flight_reroutes,
        "airline_losses":       _clean_airline_losses,
    }
    processed = {}
    for name, cleaner in cleaners.items():
        log.info("Cleaning: %s", name)
        df = cleaner(RAW_FILES[name])
        out_path = PROCESSED_FILES[name]
        df.to_parquet(out_path, index=False)
        log.info("  Saved → %s", out_path)
        processed[name] = df

    return processed


def run_master_build(processed: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Build and save the master dataset."""
    if processed is None:
        processed = {
            name: pd.read_parquet(path)
            for name, path in PROCESSED_FILES.items()
            if name != "master"
        }
    master = _build_master(processed)
    master.to_parquet(PROCESSED_FILES["master"], index=False)
    log.info("Master dataset saved → %s", PROCESSED_FILES["master"])
    return master


def run_full_pipeline() -> pd.DataFrame:
    processed = run_cleaning()
    return run_master_build(processed)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data pipeline for Iran Airspace Crisis project")
    parser.add_argument(
        "--step",
        choices=["all", "clean", "master"],
        default="all",
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    if args.step in ("all", "clean"):
        processed = run_cleaning()
    if args.step == "master":
        run_master_build()
    elif args.step == "all":
        run_master_build(processed)
