# fraud-alert-explainer

LightGBM card-fraud model on public simulated data (Sparkov), with SHAP-grounded,
LLM-written alert explanations and a deterministic check that they cite only SHAP's reasons.

## Tech stack

- Python 3.12+, managed by **uv** (no pip, no manual venvs)
- src layout: code in `src/fraud_alert_explainer/`, tests in `tests/`
- Ruff (lint + format), mypy strict, pytest + pytest-cov
- LightGBM, scikit-learn, shap; Ollama (local) for explanations

## Commands

- `make check`: lint + typecheck + tests. **Run before saying a task is done.**
- `make train`: features, both models, `results/` and `docs/img/` (needs `data/raw/`)
- `make explain`: LLM explanation eval (needs Ollama with the models pulled)
- `make format`, `make test`, `uv run pytest tests/test_x.py::test_name -x`

## Architecture

`data.py` (read raw CSVs) → `features.py` (past-only card history features) →
`model.py` (time split, LightGBM, logistic baseline, metrics) → `alerts.py`
(alerts/day vs fraud caught) → `explain.py` (SHAP reasons, prompt) + `llm.py` (Ollama) →
`grounding.py` (citation and number check). `pipeline.py` wires them; `charts.py` draws the README.

## Rules

- Raw data never goes in git (`data/` is ignored). Only public data; never real bank data.
- Every number in the README must come from `results/`, produced by `make train` / `make explain`.
- Thresholds are chosen on the validation period, then applied unchanged to the test period.
- Conventional Commits (`feat:`, `fix:`, `docs:` ...).
