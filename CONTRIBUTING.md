# Contributing Guide

Thank you for your interest in contributing to the BOI CASA Forecasting Platform!

## Getting Started

```bash
git clone https://github.com/yourusername/boi-casa-forecasting.git
cd boi-casa-forecasting
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Adding a New Model

1. Create `models/your_model.py` inheriting `BaseForecastModel`
2. Implement `fit()`, `forecast()`, `fitted_values()`
3. Register a fitter function in `src/pipelines/forecasting_pipeline.py → FITTERS`
4. Register in `dashboard/app.py → FITTERS`
5. Add tests in `tests/`

## Code Standards

- **PEP 8** formatting (use `black .` to auto-format)
- **Type hints** on all function signatures
- **Docstrings** on all public classes and functions
- Run `make lint` before committing

## Pull Request Checklist

- [ ] All existing tests pass (`make test`)
- [ ] New code has tests
- [ ] Code is PEP 8 compliant
- [ ] Docstrings added
- [ ] README updated if new feature added

## Commit Message Format

```
feat: add LSTM forecasting model
fix: handle NaN in residual diagnostics
docs: update architecture diagram
test: add CV edge case for short series
refactor: extract metric helpers into utils
```
