"""Per-alert SHAP reasons, and the prompt that turns them into a plain-language explanation."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
import shap

from fraud_alert_explainer.features import FEATURE_NAMES, FEATURES_BY_NAME

CATEGORY_NAMES = {
    "entertainment": "entertainment",
    "food_dining": "dining",
    "gas_transport": "gas and transport",
    "grocery_net": "online grocery",
    "grocery_pos": "in-store grocery",
    "health_fitness": "health and fitness",
    "home": "home goods",
    "kids_pets": "kids and pets",
    "misc_net": "miscellaneous online",
    "misc_pos": "miscellaneous in-store",
    "personal_care": "personal care",
    "shopping_net": "online shopping",
    "shopping_pos": "in-store shopping",
    "travel": "travel",
}
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

SYSTEM_PROMPT = (
    "You explain card-fraud alerts to a bank analyst. "
    "Use ONLY the reasons listed. Do not mention any other attribute of the transaction or "
    "the cardholder, and do not write any number that is not in the list. "
    "Write 2 or 3 plain sentences saying why the transaction was flagged. "
    "A reason can be an unusually low value as well as a high one, so do not call a value "
    "high or low unless the list says so. "
    "Do not say it is fraud; the analyst decides."
)


@dataclass(frozen=True)
class Reason:
    feature: str
    value: Any
    shap_value: float  # contribution to the fraud log-odds


def format_value(feature: str, value: Any) -> str:
    """Render a feature value the way an analyst would read it."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "no earlier transaction on this card"
    if feature == "category":
        return CATEGORY_NAMES.get(str(value), str(value))
    if feature == "day_of_week":
        return DAYS[int(value)]
    if feature == "hour":
        hour = int(value)
        return f"{hour:02d}:00 ({hour % 12 or 12} {'AM' if hour < 12 else 'PM'})"
    number = float(value)
    unit = FEATURES_BY_NAME[feature].unit
    if unit == "$":
        return f"${number:,.2f}"
    if unit == "x":
        return f"{number:.1f}x"
    if unit == "km":
        return f"{number:.0f} km"
    if feature == "age":
        return f"{number:.0f} years"
    if feature == "mins_since_prev_txn":
        return f"{number:,.1f} minutes"
    return f"{number:,.0f}"


def fact_line(reason: Reason) -> str:
    return f"{FEATURES_BY_NAME[reason.feature].label}: {format_value(reason.feature, reason.value)}"


def top_reasons(
    shap_row: npt.NDArray[np.float64], x_row: pd.Series[Any], k: int = 4
) -> list[Reason]:
    """The k features that pushed this score up the most (positive SHAP only)."""
    order = np.argsort(-shap_row, kind="stable")
    return [
        Reason(FEATURE_NAMES[i], x_row[FEATURE_NAMES[i]], float(shap_row[i]))
        for i in order[:k]
        if shap_row[i] > 0
    ]


def build_prompt(score: float, reasons: list[Reason]) -> tuple[list[str], str]:
    """Fact lines (also used by the grounding check) and the user message."""
    facts = [f"model fraud probability: {score:.1%}", *(fact_line(r) for r in reasons)]
    bullets = "\n".join(f"- {fact}" for fact in facts[1:])
    user = (
        f"{facts[0].capitalize()}\n"
        f"Reasons the model flagged this transaction, strongest first:\n{bullets}\n\n"
        "Write the explanation."
    )
    return facts, user


class ShapExplainer:
    """TreeSHAP over the LightGBM model, in log-odds."""

    def __init__(self, model: lgb.LGBMClassifier | lgb.Booster) -> None:
        self._explainer = shap.TreeExplainer(model)

    def shap_values(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        with warnings.catch_warnings():
            # shap warns about a list output for binary LightGBM, then returns one array.
            warnings.filterwarnings("ignore", message="LightGBM binary classifier")
            values = self._explainer.shap_values(frame[FEATURE_NAMES])
        # Older shap versions return [negative_class, positive_class] for binary LightGBM.
        if isinstance(values, list):
            values = values[1]
        return np.asarray(values, dtype=np.float64)

    def reasons(self, frame: pd.DataFrame, k: int = 4) -> list[list[Reason]]:
        values = self.shap_values(frame)
        return [top_reasons(values[i], frame.iloc[i], k) for i in range(len(frame))]
