# Iran Airspace Crisis — Impact Analysis

[![CI](https://github.com/Muhammad-Farooq13/Iran-airspace-crisis/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Farooq13/Iran-airspace-crisis/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **End-to-end ML pipeline** analysing the cascading aviation impact of the February 2026 Iran Airspace Crisis, triggered by US airstrikes on Iranian nuclear facilities at Natanz, Fordow, and Arak.

## Background

On **28 February 2026**, US strikes on Iranian nuclear infrastructure triggered the immediate closure of the **Iranian FIR (OIIX)** and a cascade of emergency NOTAMs across the region. Within 72 hours, 30+ airlines had cancelled hundreds of flights, thousands of passengers were stranded or rerouted, and estimated daily airline losses exceeded **$35 million USD**.

This project provides:
- A cleaned, feature-engineered dataset of the 12-day acute crisis period
- A regression model predicting **daily airline revenue loss** during crisis events
- An interactive **Streamlit dashboard** for exploration and live inference
- A **FastAPI** REST service for programmatic predictions

---

## Live Dashboard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://iran-airspace-crisis.streamlit.app)

**4 tabs:**
| Tab | Content |
|-----|---------|
| 🗺️ Crisis Overview | Conflict event map, airline loss ranking, event timeline |
| ✈️ Airline Loss Predictor | Interactive ML inference (Gradient Boosting, Test R²=0.96) |
| 🔍 Data Explorer | Raw CSV browser, histograms, scatter plots, correlation heatmap |
| ⚙️ Pipeline & Models | Architecture diagram, full model comparison, API reference |

---

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `conflict_events.csv` | 28 | Geolocation, severity, and aviation impact of each conflict event |
| `airspace_closures.csv` | 25 | FIR closures — type, duration, affected regions |
| `airport_disruptions.csv` | 35 | Airport-level disruption metrics and runway severity |
| `flight_cancellations.csv` | 50 | Per-flight cancellation records with airline and aircraft type |
| `flight_reroutes.csv` | 45 | Reroute details — extra km, delay, fuel cost |
| `airline_losses_estimate.csv` | 35 | **Target variable**: estimated daily loss (USD) per airline |

---

## Model Results

| Model | Test R² | CV R² | Test MAE |
|-------|---------|-------|----------|
| **Ridge** ✅ | 0.7826 | 0.892 ± 0.122 | $85,936 |
| ElasticNet | 0.7694 | 0.891 ± 0.122 | $91,199 |
| Random Forest | 0.6259 | 0.741 ± 0.331 | $149,790 |
| Gradient Boosting | 0.6398 | 0.775 ± 0.275 | $115,149 |

Best model: **Ridge Regression** (selected by 5-fold CV R²).  
Feature set: 55 features — 22 numeric operational metrics + 29 country one-hot dummies + 4 derived interaction terms.

---

## Project Structure

```
iranair/
├── data/
│   ├── raw/               # Original CSV datasets
│   └── processed/         # Cleaned Parquet files (git-ignored)
├── src/
│   ├── config.py          # Central path & constant registry
│   ├── data/pipeline.py   # ETL — clean, derive, join → master_dataset.parquet
│   ├── features/
│   │   └── build_features.py  # Feature engineering → 55-column matrix
│   ├── models/
│   │   ├── train.py       # Train 4 candidates, CV-select best, save .pkl
│   │   └── predict.py     # Load artefacts + inference helper
│   ├── api/app.py         # FastAPI REST service
│   └── visualization/plots.py
├── tests/                 # 24 unit tests (pytest)
├── streamlit_app.py       # Interactive dashboard (4 tabs)
├── train_demo.py          # Lightweight demo model trainer
├── models/
│   ├── iran_demo.pkl      # Pre-trained Streamlit demo model (versioned)
│   └── demo_countries.json
├── .github/workflows/ci.yml
├── requirements.txt       # Full dependencies (includes streamlit, plotly)
├── requirements-ci.txt    # Minimal CI dependencies
├── runtime.txt            # python-3.11 (Streamlit Cloud)
└── pyproject.toml
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Muhammad-Farooq13/Iran-airspace-crisis
cd Iran-airspace-crisis

# 2. Install
pip install -r requirements.txt

# 3. Run full pipeline
python -m src.data.pipeline --step all
python -m src.features.build_features
python -m src.models.train

# 4. Launch Streamlit dashboard
streamlit run streamlit_app.py

# 5. (Optional) Start FastAPI service
uvicorn src.api.app:app --reload --port 8000
# → http://localhost:8000/docs
```

### Tests

```bash
pytest tests/ -v --cov=src
```

---

## REST API

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Response:
```json
{"predicted_daily_loss_usd": 4183520.0, "predicted_daily_loss_millions": 4.1835}
```

---

## License

MIT © Muhammad Farooq