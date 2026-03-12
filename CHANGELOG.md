# Changelog

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-03-12

### Added
- **End-to-end ML pipeline** across 6 raw CSV datasets (conflict events, airspace
  closures, airport disruptions, flight cancellations, reroutes, airline losses)
- `src/data/pipeline.py` — ETL with date parsing, derived ratios, multi-source join
  → `master_dataset.parquet`
- `src/features/build_features.py` — 55-feature matrix (22 numeric + 29 country
  one-hot dummies + 4 interaction terms), StandardScaler, feature list JSON
- `src/models/train.py` — 4 candidate regressors (Ridge, ElasticNet, Random Forest,
  Gradient Boosting) with 5-fold CV; best model serialised to `models/airline_loss_regressor.pkl`
- `src/models/predict.py` — artefact loader and inference helper
- `src/api/app.py` — FastAPI service with `/v1/predict` and `/v1/predict/batch` endpoints
- `src/visualization/plots.py` — reporting-quality matplotlib/seaborn charts
- **24 pytest tests** covering pipeline, feature engineering, model I/O, and API
- `streamlit_app.py` — interactive four-tab dashboard:
  - 🗺️ Crisis Overview (conflict map, airline loss ranking, event timeline)
  - ✈️ Airline Loss Predictor (live Gradient Boosting inference, gauge chart)
  - 🔍 Data Explorer (CSV browser, histograms, scatter, correlation heatmap)
  - ⚙️ Pipeline & Models (architecture, full model comparison, API reference)
- `train_demo.py` — lightweight demo trainer (no full pipeline dependency);
  produces `models/iran_demo.pkl` (Test R² = 0.96, 35 samples × 36 features)
- `.streamlit/config.toml` — dark theme with crisis-red accent (`#e74c3c`)
- `runtime.txt` — `python-3.11` for Streamlit Cloud
- `requirements-ci.txt` — minimal CI dependency set (lint + tests only)
- Plotly (`>=5.15`) and Streamlit (`>=1.36`) added to `requirements.txt`

### Changed
- CI upgraded: `codecov/codecov-action@v4` → `@v5`
- CI now uses `requirements-ci.txt` (faster installs, no full heavy stack)
- CI no longer runs the full data pipeline/train in CI (data-dependent steps
  removed from automated workflow)
- Docker job in CI now uses `docker/setup-buildx-action@v3`
- `.gitignore` updated: `models/iran_demo.pkl` and `models/demo_countries.json`
  excluded from blanket ignore so the demo model is versioned in the repo

### Fixed
- N/A — all 24 tests passed from initial commit

---

## [Unreleased]

- Temporal train/test split (by date) for more realistic evaluation
- SHAP feature importance visualisation in Streamlit
- Airline-level loss breakdown by route disruption category