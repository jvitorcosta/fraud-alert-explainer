"""Precision/recall vs threshold, as the analyst trade-off: alerts per day vs fraud caught."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


def budget_curve(
    y_true: npt.ArrayLike, scores: npt.ArrayLike, amounts: npt.ArrayLike, n_days: float
) -> pd.DataFrame:
    """One row per distinct score threshold, highest threshold first.

    Alerting on every transaction scoring >= `threshold` yields `alerts_per_day`, catches
    `recall` of fraud cases and `amount_recall` of fraud dollars, at `precision`.
    """
    if n_days <= 0:
        raise ValueError("n_days must be positive")
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    amt = np.asarray(amounts, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    s, y, amt = s[order], y[order], amt[order]

    # Last index of each run of equal scores: alerting is all-or-nothing within a tie.
    last = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
    alerts = last + 1
    caught = np.cumsum(y)[last]
    caught_amt = np.cumsum(y * amt)[last]
    return pd.DataFrame(
        {
            "threshold": s[last],
            "alerts": alerts,
            "alerts_per_day": alerts / n_days,
            "fraud_caught": caught.astype(np.int64),
            "precision": caught / alerts,
            "recall": caught / max(y.sum(), 1.0),
            "amount_recall": caught_amt / max(float((y * amt).sum()), 1e-9),
        }
    )


def threshold_for_budget(curve: pd.DataFrame, alerts_per_day: float) -> float:
    """Lowest threshold whose alert volume stays within the daily budget."""
    within = curve[curve["alerts_per_day"] <= alerts_per_day]
    if within.empty:
        raise ValueError(f"no threshold keeps alerts under {alerts_per_day}/day")
    return float(within["threshold"].iloc[-1])


@dataclass(frozen=True)
class OperatingPoint:
    budget_per_day: float
    threshold: float
    alerts_per_day: float
    precision: float
    recall: float
    amount_recall: float


def operating_point(
    y_true: npt.ArrayLike,
    scores: npt.ArrayLike,
    amounts: npt.ArrayLike,
    n_days: float,
    threshold: float,
    budget_per_day: float,
) -> OperatingPoint:
    """What a threshold (picked on another period) actually does on this one."""
    y = np.asarray(y_true, dtype=np.float64)
    amt = np.asarray(amounts, dtype=np.float64)
    flagged = np.asarray(scores) >= threshold
    n_alerts = int(flagged.sum())
    caught = float(y[flagged].sum())
    return OperatingPoint(
        budget_per_day=budget_per_day,
        threshold=threshold,
        alerts_per_day=n_alerts / n_days,
        precision=caught / n_alerts if n_alerts else 0.0,
        recall=caught / max(y.sum(), 1.0),
        amount_recall=float((y * amt)[flagged].sum()) / max(float((y * amt).sum()), 1e-9),
    )


def budget_table(points: Sequence[OperatingPoint], model: str) -> list[str]:
    """Markdown rows for the README."""
    return [
        f"| {model} | {p.budget_per_day:g} | {p.alerts_per_day:.1f} | {p.precision:.0%} | "
        f"{p.recall:.0%} | {p.amount_recall:.0%} |"
        for p in points
    ]
