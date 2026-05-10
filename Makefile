# ──────────────────────────────────────────────────────────────────────────────
#  BOI CASA Deposit Forecasting — Makefile  (v2.0 · post-fix)
#  Usage: make help
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: install pipeline pipeline-fast pipeline-all visualize dashboard \
        test test-fast test-advanced lint format clean report report-pdf \
        report-html report-md all help

PYTHON     := python3
PIP        := pip3
DATA       := data/raw/boi_casa_deposits.csv
TRAIN_FRAC := 0.80
CV_SPLITS  := 5
MODELS_ALL := ARIMA SARIMA SARIMAX AutoARIMA HoltWinters Prophet

# ── Help ──────────────────────────────────────────────────────────────────────
help:              ## Show all available commands
	@echo ""
	@echo "  BOI CASA Forecasting — available make targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Install ───────────────────────────────────────────────────────────────────
install:           ## Install all Python + Node dependencies
	$(PIP) install -r requirements.txt
	npm install -g docx 2>/dev/null || true
	@echo "  All dependencies installed."

# ── Pipeline ──────────────────────────────────────────────────────────────────
pipeline:          ## Run full forecasting pipeline (all 6 models, 5-fold CV)
	$(PYTHON) run_pipeline.py \
	  --data $(DATA) \
	  --train-frac $(TRAIN_FRAC) \
	  --cv-splits $(CV_SPLITS)

pipeline-fast:     ## Quick run — ARIMA + HoltWinters only, no CV
	$(PYTHON) run_pipeline.py \
	  --data $(DATA) \
	  --models ARIMA HoltWinters \
	  --train-frac $(TRAIN_FRAC) \
	  --no-cv

pipeline-all:      ## Run all 6 models with 5-fold CV (same as pipeline)
	$(PYTHON) run_pipeline.py \
	  --data $(DATA) \
	  --models $(MODELS_ALL) \
	  --train-frac $(TRAIN_FRAC) \
	  --cv-splits $(CV_SPLITS)

pipeline-sarimax:  ## Run SARIMAX + HoltWinters + AutoARIMA with real macro exog
	$(PYTHON) run_pipeline.py \
	  --data $(DATA) \
	  --models SARIMAX HoltWinters AutoARIMA \
	  --train-frac $(TRAIN_FRAC) \
	  --no-cv

train: pipeline    ## Alias for pipeline

# ── Visualizations ────────────────────────────────────────────────────────────
visualize:         ## Generate all 13 charts (9 PNG + 4 interactive HTML)
	$(PYTHON) run_visualizations.py --data $(DATA)

# ── Dashboard ─────────────────────────────────────────────────────────────────
dashboard:         ## Launch Bloomberg-style Streamlit dashboard (port 8501)
	streamlit run dashboard/app.py \
	  --server.port 8501 \
	  --theme.base dark

# ── Reports ───────────────────────────────────────────────────────────────────
report:            ## Generate PDF + HTML + Markdown reports
	$(PYTHON) run_reports.py --formats pdf html markdown --report-dir reports

report-pdf:        ## Generate PDF report only
	$(PYTHON) run_reports.py --formats pdf --report-dir reports

report-html:       ## Generate HTML report only
	$(PYTHON) run_reports.py --formats html --report-dir reports

report-md:         ## Generate Markdown report only
	$(PYTHON) run_reports.py --formats markdown --report-dir reports

# ── Tests ─────────────────────────────────────────────────────────────────────
test:              ## Run all 57 unit tests
	$(PYTHON) -m pytest tests/ -v --tb=short

test-fast:         ## Run core tests only (metrics + diagnostics + data_loader)
	$(PYTHON) -m pytest \
	  tests/test_metrics.py \
	  tests/test_diagnostics.py \
	  tests/test_data_loader.py \
	  -v --tb=short

test-advanced:     ## Run new advanced model tests (HoltWinters, SARIMAX exog, CV)
	$(PYTHON) -m pytest tests/test_advanced_models.py -v --tb=short

test-cov:          ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=term-missing --tb=short

# ── Code quality ──────────────────────────────────────────────────────────────
lint:              ## Run flake8 linter (max line 120)
	flake8 src/ models/ visualizations/ dashboard/ \
	  --max-line-length=120 \
	  --exclude=__pycache__

format:            ## Auto-format with black (line length 100)
	black src/ models/ visualizations/ dashboard/ tests/ \
	  --line-length 100

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:             ## Remove generated outputs and Python/Node caches
	@rm -rf visualizations/output/*.png \
	        visualizations/output/*.html 2>/dev/null; true
	@rm -rf reports/*.log 2>/dev/null; true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	@find . -name "*.pyc" -delete 2>/dev/null; true
	@echo "  Cleaned output artefacts and caches."

clean-all: clean   ## Also remove processed data (keeps raw data + macro CSV)
	@rm -rf data/processed/*.csv 2>/dev/null; true
	@echo "  Cleaned processed data."

# ── Full run ──────────────────────────────────────────────────────────────────
all:               ## Full run: pipeline + visualize + test
	$(MAKE) pipeline
	$(MAKE) visualize
	$(MAKE) test
	@echo ""
	@echo "  All steps complete."

ci:                ## CI sequence: install + lint + test (used in GitHub Actions)
	$(MAKE) install
	$(MAKE) lint
	$(MAKE) test
