"""
FastAPI application — Airline Loss Prediction Service.

Endpoints:
  GET  /              → health check
  GET  /v1/info       → model metadata
  POST /v1/predict    → single prediction
  POST /v1/predict/batch → batch predictions

Run locally:
    uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import API_VERSION, FEATURE_LIST_PATH, MODEL_PATH, SCALER_PATH
from src.models.predict import load_artefacts, predict

log = logging.getLogger(__name__)

app = FastAPI(
    title="Iran Airspace Crisis — Airline Loss Predictor",
    description=(
        "Predicts the estimated daily revenue loss (USD) for an airline "
        "based on crisis-context and operational features derived from the "
        "Iran Airspace Crisis dataset (February–March 2026)."
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ──────────────────────────────────────────────────────────────────────────────
# Artefact cache — loaded once at startup
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_artefacts():
    return load_artefacts(MODEL_PATH, SCALER_PATH, FEATURE_LIST_PATH)


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    cancelled_flights:           int   = Field(..., ge=0, description="Number of cancelled flights")
    rerouted_flights:            int   = Field(..., ge=0, description="Number of rerouted flights")
    additional_fuel_cost_usd:    float = Field(..., ge=0, description="Additional fuel cost (USD)")
    passengers_impacted:         int   = Field(..., ge=0, description="Total passengers impacted")
    avg_extra_km:                float = Field(0.0,  ge=0, description="Average extra distance per reroute (km)")
    avg_delay_min:               float = Field(0.0,  ge=0, description="Average delay per reroute (minutes)")
    avg_cost_per_km:             float = Field(0.0,  ge=0)
    total_reroutes:              int   = Field(0,    ge=0)
    total_recorded_cancellations:int   = Field(0,    ge=0)
    wide_body_cancellations:     int   = Field(0,    ge=0)
    n_primary_closed_firs:       int   = Field(3,    ge=0, description="# primary conflict-country FIRs closed")
    n_total_closures:            int   = Field(25,   ge=0)
    avg_closure_hours:           float = Field(100.0,ge=0)
    precautionary_pct:           float = Field(0.6,  ge=0, le=1)
    avg_airport_runway_sev:      float = Field(2.5,  ge=0)
    total_airport_disrupted:     int   = Field(1800, ge=0)
    early_critical_events:       int   = Field(5,    ge=0)
    total_conflict_events:       int   = Field(28,   ge=0)

    model_config = {"json_schema_extra": {
        "example": {
            "cancelled_flights": 18,
            "rerouted_flights": 62,
            "additional_fuel_cost_usd": 2835200,
            "passengers_impacted": 9180,
            "avg_extra_km": 740.0,
            "avg_delay_min": 67.0,
            "avg_cost_per_km": 78.0,
            "total_reroutes": 8,
            "total_recorded_cancellations": 7,
            "wide_body_cancellations": 5,
            "n_primary_closed_firs": 3,
            "n_total_closures": 25,
            "avg_closure_hours": 112.4,
            "precautionary_pct": 0.6,
            "avg_airport_runway_sev": 2.5,
            "total_airport_disrupted": 1856,
            "early_critical_events": 7,
            "total_conflict_events": 28
        }
    }}


class PredictionResponse(BaseModel):
    predicted_daily_loss_usd: float
    predicted_daily_loss_millions: float


class BatchPredictionRequest(BaseModel):
    records: list[PredictionRequest] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class ModelInfo(BaseModel):
    model_path: str
    feature_count: int
    feature_names: list[str]


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "Airline Loss Predictor"}


@app.get(f"/{API_VERSION}/info", response_model=ModelInfo, tags=["Model"])
def model_info():
    try:
        _, _, feature_names = _get_artefacts()
        return ModelInfo(
            model_path=str(MODEL_PATH),
            feature_count=len(feature_names),
            feature_names=feature_names,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model artefacts not found. Run the training pipeline first. ({exc})",
        ) from exc


@app.post(f"/{API_VERSION}/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(request: PredictionRequest):
    try:
        model, scaler, feature_names = _get_artefacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    result = predict(
        request.model_dump(),
        model=model,
        scaler=scaler,
        feature_names=feature_names,
    )
    loss = float(result[0])
    return PredictionResponse(
        predicted_daily_loss_usd=round(loss, 2),
        predicted_daily_loss_millions=round(loss / 1e6, 4),
    )


@app.post(f"/{API_VERSION}/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    try:
        model, scaler, feature_names = _get_artefacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    import pandas as pd

    df = pd.DataFrame([r.model_dump() for r in request.records])
    results = predict(df, model=model, scaler=scaler, feature_names=feature_names)

    predictions = [
        PredictionResponse(
            predicted_daily_loss_usd=round(float(v), 2),
            predicted_daily_loss_millions=round(float(v) / 1e6, 4),
        )
        for v in results
    ]
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))
