import json
import math

import httpx
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from fraud_alert_explainer.explain import (
    Reason,
    ShapExplainer,
    build_prompt,
    format_value,
    top_reasons,
)
from fraud_alert_explainer.features import FEATURE_NAMES
from fraud_alert_explainer.grounding import check
from fraud_alert_explainer.llm import OllamaClient
from fraud_alert_explainer.model import time_split, train_lightgbm


@pytest.mark.parametrize(
    ("feature", "value", "expected"),
    [
        ("amt", 1077.694, "$1,077.69"),
        ("amt_vs_card_avg", 6.25, "6.2x"),
        ("category", "shopping_net", "online shopping"),
        ("hour", 22, "22:00 (10 PM)"),
        ("hour", 0, "00:00 (12 AM)"),
        ("day_of_week", 6, "Sunday"),
        ("distance_km", 87.4, "87 km"),
        ("age", 73.6, "74 years"),
        ("mins_since_prev_txn", 2.04, "2.0 minutes"),
        ("txns_prev_24h", 9, "9"),
        ("mins_since_prev_txn", math.nan, "no earlier transaction on this card"),
    ],
)
def test_format_value_reads_like_an_analyst_note(
    feature: str, value: object, expected: str
) -> None:
    assert format_value(feature, value) == expected


def test_top_reasons_keeps_only_positive_contributions_in_order() -> None:
    shap_row = np.zeros(len(FEATURE_NAMES))
    shap_row[FEATURE_NAMES.index("hour")] = 0.5
    shap_row[FEATURE_NAMES.index("amt")] = 2.0
    shap_row[FEATURE_NAMES.index("age")] = -1.0
    x_row = pd.Series(dict.fromkeys(FEATURE_NAMES, 1.0))

    reasons = top_reasons(shap_row, x_row, k=4)

    assert [r.feature for r in reasons] == ["amt", "hour"]


def test_build_prompt_lists_only_the_given_reasons() -> None:
    reasons = [Reason("amt", 1077.69, 2.0), Reason("hour", 22, 0.5)]

    facts, prompt = build_prompt(0.973, reasons)

    assert facts == [
        "model fraud score: 0.97",
        "transaction amount: $1,077.69",
        "hour of day: 22:00 (10 PM)",
    ]
    assert "- transaction amount: $1,077.69\n- hour of day: 22:00 (10 PM)" in prompt
    assert "age" not in prompt


def test_shap_explainer_explains_a_trained_model(features: pd.DataFrame) -> None:
    split = time_split(features, "2020-03-01", "2020-07-01")
    model = train_lightgbm(split.train, split.valid, n_estimators=30)
    alerts = split.test[split.test["is_fraud"] == 1].head(3)

    reasons = ShapExplainer(model).reasons(alerts, k=3)

    assert len(reasons) == 3
    assert all(0 < len(r) <= 3 for r in reasons)
    assert all(reason.shap_value > 0 for r in reasons for reason in r)


def test_shap_explainer_accepts_a_booster(features: pd.DataFrame) -> None:
    split = time_split(features, "2020-03-01", "2020-07-01")
    booster = train_lightgbm(split.train, split.valid, n_estimators=30).booster_
    assert isinstance(booster, lgb.Booster)

    values = ShapExplainer(booster).shap_values(split.test.head(5))

    assert values.shape == (5, len(FEATURE_NAMES))


def test_ollama_client_sends_a_deterministic_chat_request() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "  Flagged for amount. "}})

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ollama") as http:
        reply = OllamaClient(http, "qwen2.5:1.5b").chat("system", "user")

    assert reply == "Flagged for amount."
    assert seen["model"] == "qwen2.5:1.5b"
    assert seen["stream"] is False
    assert seen["options"] == {"temperature": 0, "num_predict": 200, "seed": 0}


def test_prompt_facts_ground_an_explanation_that_repeats_them() -> None:
    reasons = [Reason("amt", 1077.69, 2.0), Reason("category", "shopping_net", 1.0)]
    facts, _ = build_prompt(0.97, reasons)

    result = check(
        "Flagged because the transaction amount of $1,077.69 is high for online shopping.",
        [r.feature for r in reasons],
        facts,
    )

    assert result.grounded
