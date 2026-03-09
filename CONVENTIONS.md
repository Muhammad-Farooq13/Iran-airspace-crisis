# Iran Airspace Crisis — Codebase Conventions

## Build & run commands
- Full pipeline: `python -m src.data.pipeline --step all` → `python -m src.features.build_features` → `python -m src.models.train`
- Tests: `pytest tests/ -v`
- API: `uvicorn src.api.app:app --reload --port 8000`
- Or use Makefile: `make all`, `make serve`, `make test`

## Data flow
raw CSVs (data/raw/) → Parquet (data/processed/) → feature matrix (data/features/) → models/ → API

## Config
All paths and constants are in src/config.py — never hardcode paths elsewhere.

## Parquet format
All intermediate datasets stored as Parquet (pyarrow). Never commit processed data.

## Model artefacts
- airline_loss_regressor.pkl — best sklearn model
- scaler.pkl — StandardScaler fitted on train set only
- feature_list.json — ordered list of feature names for alignment

## API
FastAPI app at src/api/app.py. Pydantic v2 models for request validation.
Model loaded once via @lru_cache — do not reload per request.

## Testing
pytest fixtures in tests/test_pipeline.py load real raw CSVs.
tests/test_features.py runs full pipeline (module-scoped fixture for speed).

## Code style
- black, line-length=100
- ruff select=E,F,I,N,W,UP
- Type hints required in src/
