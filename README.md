# Iran Airspace Crisis — Aviation Impact Analysis & Loss Prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI](https://github.com/Muhammad-Farooq-13/iran-airspace-crisis/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Farooq-13/iran-airspace-crisis/actions)

> **A complete, production-grade data science project** analysing the global aviation impact of the Iran Airspace Crisis (February–March 2026) and predicting airline financial losses using machine learning.

---

## Problem Statement

On 28 February 2026, US airstrikes on Iranian nuclear facilities triggered immediate closure of Iranian, Israeli, and Yemeni airspace — cascading into 25 FIR closures across the Middle East and Central Asia. The crisis grounded hundreds of flights, forced thousands of kilometres of reroutes, and inflicted tens of millions of dollars in daily losses on the global aviation industry.

**Objective:** Build a reproducible ML pipeline to *predict estimated daily airline revenue losses (USD)* from operational disruption and crisis-context features, enabling rapid financial impact assessment during future geopolitical aviation crises.

**Success Criteria:**
- R² ≥ 0.85 on hold-out test set
- MAPE ≤ 15% on unseen airlines
- REST API prediction latency < 100 ms
- Fully reproducible via a single `make all` command

---

## Dataset

Six CSV datasets covering Feb 28 – Mar 9, 2026:

| File | Description | Rows |
|------|-------------|------|
| `conflict_events.csv` | Geo-timestamped military/conflict events with severity ratings | 29 |
| `airspace_closures.csv` | FIR closure records with duration and authority | 25 |
| `airport_disruptions.csv` | Per-airport cancelled/delayed/diverted counts and runway status | 29 |
| `flight_cancellations.csv` | Individual cancelled flights with aircraft type and reason | 50 |
| `flight_reroutes.csv` | Rerouted flights with extra distance, fuel cost, delay | 46 |
| `airline_losses_estimate.csv` | Airline-level daily loss estimates and operational stats | 29 |

---

## Project Structure

```
iran-airspace-crisis/
│
├── data/
│   ├── raw/                  # Original CSV files (immutable)
│   ├── processed/            # Cleaned Parquet files
│   └── features/             # Final feature matrix
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   └── 02_modelling.ipynb    # Model training & evaluation
│
├── src/
│   ├── config.py             # Central paths & constants
│   ├── data/
│   │   └── pipeline.py       # ETL: clean → process → master
│   ├── features/
│   │   └── build_features.py # Feature engineering
│   ├── models/
│   │   ├── train.py          # Multi-model training & selection
│   │   └── predict.py        # Inference utilities
│   ├── visualization/
│   │   └── plots.py          # Reusable plotting functions
│   └── api/
│       └── app.py            # FastAPI prediction service
│
├── tests/
│   ├── test_pipeline.py      # Data cleaning unit tests
│   └── test_features.py      # Feature engineering tests
│
├── models/                   # Serialised model artefacts (.pkl, .json)
├── reports/figures/          # Auto-generated plots
├── .github/workflows/ci.yml  # GitHub Actions CI pipeline
├── Dockerfile
├── Makefile
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Quick Start

### 1. Clone and install
```bash
git clone https://github.com/Muhammad-Farooq-13/iran-airspace-crisis.git
cd iran-airspace-crisis
python -m venv .venv
# Windows: .venv\Scripts\activate  |  Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
make all
# Equivalent to: data → features → train → test → serve
```

Or step by step:
```bash
make data       # Clean raw CSVs → data/processed/
make features   # Build feature matrix → data/features/
make train      # Train models → models/
make test       # Run pytest test suite
make serve      # Start FastAPI server on :8000
```

### 3. Explore notebooks
```bash
jupyter lab notebooks/
```

### 4. Use the prediction API
```bash
# Health check
curl http://localhost:8000/

# Single prediction
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cancelled_flights": 18,
    "rerouted_flights": 62,
    "additional_fuel_cost_usd": 2835200,
    "passengers_impacted": 9180,
    "avg_extra_km": 740,
    "avg_delay_min": 67,
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

Interactive API docs: http://localhost:8000/docs

---

## Methodology

### Pipeline Overview
```
Raw CSVs  →  [ETL / Clean]  →  Processed Parquet
                                      ↓
                             [Feature Engineering]
                                      ↓
                              Feature Matrix (scaled)
                                      ↓
                      ┌──────────────┼──────────────┐
                    Ridge       Random Forest    Gradient Boosting
                      └──────────────┼──────────────┘
                               [5-fold CV]
                                      ↓
                              Best Model Selected
                                      ↓
                            FastAPI REST Service
```

### Models Evaluated
| Model | Rationale |
|-------|-----------|
| Ridge Regression | Interpretable baseline; handles multicollinearity |
| ElasticNet | L1+L2 regularisation for sparse features |
| Random Forest | Non-linear, robust, low variance |
| Gradient Boosting | Best for tabular data; handles skewed targets |

### Feature Groups
1. **Direct operational** — cancelled/rerouted flights, fuel costs, passengers
2. **Derived ratios** — fuel cost ratio, reroute ratio, loss per passenger
3. **Reroute characteristics** — avg extra km, delay, cost/km
4. **Crisis context** — FIR closures, closure duration, conflict events
5. **Interaction terms** — cancelled × wide-body, fuel × distance
6. **Log transforms** — right-skewed monetary features

---

## Results

| Model | CV R² | Test RMSE | Test MAE | MAPE % |
|-------|-------|-----------|----------|--------|
| Gradient Boosting | **0.96** | $142K | $98K | 8.2% |
| Random Forest | 0.94 | $168K | $118K | 10.1% |
| Ridge Regression | 0.89 | $234K | $176K | 14.7% |
| ElasticNet | 0.87 | $251K | $192K | 16.3% |

*Results are illustrative — run `make train` for exact figures on your machine.*

---

## Docker Deployment

```bash
docker build -t airline-loss-predictor .
docker run -p 8000:8000 airline-loss-predictor
```

---

## Development

```bash
# Lint
make lint

# Format
make format

# Type check
make typecheck
```

---

## Key Skills Demonstrated

- **End-to-end ML pipeline**: raw data → cleaned features → trained model → REST API
- **Modular Python packaging**: `src/` layout with clear separation of concerns
- **Reproducible ETL**: Parquet-based data versioning, no manual steps
- **Statistical rigour**: K-fold CV, learning curves, residual analysis
- **Production patterns**: Pydantic validation, LRU-cached model loading, structured logging
- **DevOps**: Docker, GitHub Actions CI, Makefile automation
- **Testing**: pytest with fixtures, parametrized checks, coverage reporting

---

## License

MIT © 2026 [Muhammad Farooq](https://github.com/Muhammad-Farooq-13)

---

## Author

**Muhammad Farooq**  
📧 [mfarooqshafee333@gmail.com](mailto:mfarooqshafee333@gmail.com)  
🐙 [github.com/Muhammad-Farooq-13](https://github.com/Muhammad-Farooq-13)
