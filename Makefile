.PHONY: all data features train test lint format typecheck serve clean

PYTHON   := python
PYTEST   := pytest
UVICORN  := uvicorn

# ── Full pipeline ──────────────────────────────────────────────────────────────
all: data features train test
	@echo "✓ Full pipeline complete."

# ── Individual steps ───────────────────────────────────────────────────────────
data:
	$(PYTHON) -m src.data.pipeline --step all

features:
	$(PYTHON) -m src.features.build_features

train:
	$(PYTHON) -m src.models.train

test:
	$(PYTEST) tests/ -v --tb=short

# ── API ────────────────────────────────────────────────────────────────────────
serve:
	$(UVICORN) src.api.app:app --reload --host 0.0.0.0 --port 8000

# ── Code quality ───────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/

format:
	black src/ tests/ notebooks/

typecheck:
	mypy src/

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	@echo "Cleaned."
