"""End-to-end runs: `train` (features, models, metrics, charts) and `explain` (LLM eval)."""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud_alert_explainer import alerts, charts, grounding
from fraud_alert_explainer.data import load_raw
from fraud_alert_explainer.explain import SYSTEM_PROMPT, ShapExplainer, build_prompt
from fraud_alert_explainer.features import FEATURE_NAMES, TARGET, build_features
from fraud_alert_explainer.llm import DEFAULT_BASE_URL, OllamaClient
from fraud_alert_explainer.model import (
    fraud_scores,
    metrics,
    time_split,
    train_baseline,
    train_lightgbm,
)

log = logging.getLogger(__name__)

ROOT = Path.cwd()
RAW_DIR = ROOT / "data" / "raw"
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "lightgbm.txt"
RESULTS = ROOT / "results"
IMG = ROOT / "docs" / "img"

VALID_START = "2020-04-01"
TEST_START = "2020-07-01"
BUDGETS_PER_DAY = (10, 25, 50, 100)
EXPLAIN_BUDGET = 50  # the operating point whose alerts get explained
SHAP_SAMPLE = 20_000
SEED = 42


def _n_days(frame: pd.DataFrame) -> float:
    return float(frame["ts"].dt.normalize().nunique())


def load_features() -> pd.DataFrame:
    if FEATURES_PATH.exists():
        return pd.read_parquet(FEATURES_PATH)
    log.info("building features from %s", RAW_DIR)
    features = build_features(load_raw(RAW_DIR))
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PATH, index=False)
    return features


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def train() -> dict[str, Any]:
    split = time_split(load_features(), VALID_START, TEST_START)
    for name, part in (("train", split.train), ("valid", split.valid), ("test", split.test)):
        log.info(
            "%s: %d rows, %d fraud, %s to %s",
            name,
            len(part),
            part[TARGET].sum(),
            part["ts"].min(),
            part["ts"].max(),
        )

    models = {
        "LightGBM": train_lightgbm(split.train, split.valid),
        "Logistic regression": train_baseline(split.train),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    models_lgb = models["LightGBM"]
    assert isinstance(models_lgb, lgb.LGBMClassifier)  # noqa: S101 - narrows the type for mypy
    models_lgb.booster_.save_model(MODEL_PATH)

    report: dict[str, Any] = {
        "split": {
            name: {
                "rows": len(part),
                "fraud": int(part[TARGET].sum()),
                "start": str(part["ts"].min()),
                "end": str(part["ts"].max()),
                "days": _n_days(part),
            }
            for name, part in (("train", split.train), ("valid", split.valid), ("test", split.test))
        },
        "lightgbm_best_iteration": int(models_lgb.best_iteration_),
        "models": {},
    }
    test_scores, curves, rows = {}, {}, []
    y_valid, y_test = split.valid[TARGET].to_numpy(), split.test[TARGET].to_numpy()
    for name, model in models.items():
        valid_s, test_s = fraud_scores(model, split.valid), fraud_scores(model, split.test)
        test_scores[name] = test_s
        valid_curve = alerts.budget_curve(
            y_valid, valid_s, split.valid["amt"], _n_days(split.valid)
        )
        curves[name] = alerts.budget_curve(y_test, test_s, split.test["amt"], _n_days(split.test))
        # Thresholds are picked on validation, then applied unchanged to the test period.
        points = [
            alerts.operating_point(
                y_test,
                test_s,
                split.test["amt"],
                _n_days(split.test),
                alerts.threshold_for_budget(valid_curve, budget),
                budget,
            )
            for budget in BUDGETS_PER_DAY
        ]
        rows += alerts.budget_table(points, name)
        report["models"][name] = {
            "valid": asdict(metrics(y_valid, valid_s)),
            "test": asdict(metrics(y_test, test_s)),
            "operating_points": [asdict(p) for p in points],
        }

    _write(RESULTS / "metrics.json", json.dumps(report, indent=2) + "\n")
    header = (
        "| Model | Budget (alerts/day) | Alerts/day on test | Precision | Fraud caught | "
        "Fraud $ caught |\n|---|---:|---:|---:|---:|---:|\n"
    )
    _write(RESULTS / "operating_points.md", header + "\n".join(rows) + "\n")

    pr = {name: report["models"][name]["test"]["pr_auc"] for name in models}
    charts.pr_curves(y_test, test_scores, pr, IMG / "pr_curve.png")
    charts.budget_tradeoff(curves, EXPLAIN_BUDGET, IMG / "alerts_vs_fraud_caught.png")
    sample = split.test.sample(min(SHAP_SAMPLE, len(split.test)), random_state=SEED)
    shap_values = ShapExplainer(lgb.Booster(model_file=MODEL_PATH)).shap_values(sample)
    charts.shap_importance(shap_values, FEATURE_NAMES, IMG / "shap_importance.png")
    return report


def explain(models: list[str], n_alerts: int, base_url: str = DEFAULT_BASE_URL) -> str:
    split = time_split(load_features(), VALID_START, TEST_START)
    report = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    point = next(
        p
        for p in report["models"]["LightGBM"]["operating_points"]
        if p["budget_per_day"] == EXPLAIN_BUDGET
    )
    booster = lgb.Booster(model_file=MODEL_PATH)
    test = split.test.copy()
    test["score"] = booster.predict(test[FEATURE_NAMES])
    flagged = test[test["score"] >= point["threshold"]]
    sample = flagged.sample(min(n_alerts, len(flagged)), random_state=SEED).sort_values("ts")
    reasons = ShapExplainer(booster).reasons(sample)

    summaries = []
    for model in models:
        with httpx.Client(base_url=base_url, timeout=120) as http:
            client = OllamaClient(http, model)
            client.chat(SYSTEM_PROMPT, "Warm-up: reply OK.", max_tokens=5)
            results, latencies, rows = [], [], []
            for (_, row), alert_reasons in zip(sample.iterrows(), reasons, strict=True):
                facts, prompt = build_prompt(float(row["score"]), alert_reasons)
                start = time.perf_counter()
                text = client.chat(SYSTEM_PROMPT, prompt)
                latencies.append(time.perf_counter() - start)
                result = grounding.check(text, [r.feature for r in alert_reasons], facts)
                results.append(result)
                rows.append(
                    {
                        "ts": str(row["ts"]),
                        "is_fraud": int(row[TARGET]),
                        "score": round(float(row["score"]), 4),
                        "facts": facts,
                        "given": [r.feature for r in alert_reasons],
                        "explanation": text,
                        "grounded": result.grounded,
                        "cited": list(result.cited),
                        "off_list": list(result.uncited_by_shap),
                        "unsupported_numbers": list(result.unsupported_numbers),
                        "coverage": round(result.coverage, 3),
                        "latency_s": round(latencies[-1], 3),
                    }
                )
                log.info("%s %s grounded=%s", model, row["ts"], result.grounded)
        summary = grounding.summarize(model, results, latencies)
        summaries.append(summary)
        name = model.replace(":", "-").replace("/", "-")
        _write(
            RESULTS / "explanations" / f"{name}.json",
            json.dumps({"summary": asdict(summary), "alerts": rows}, indent=2, ensure_ascii=False)
            + "\n",
        )

    table = grounding.markdown_table(summaries)
    precision = float(np.mean(sample[TARGET]))
    _write(
        RESULTS / "explanations" / "summary.md",
        f"{table}\n\n{len(sample)} alerts sampled from the test period at the "
        f"{EXPLAIN_BUDGET}/day threshold ({precision:.0%} of them are labelled fraud).\n",
    )
    return table
