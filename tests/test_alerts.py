import numpy as np
import pytest

from fraud_alert_explainer.alerts import (
    budget_curve,
    budget_table,
    operating_point,
    threshold_for_budget,
)

Y = np.array([1, 0, 1, 0, 0, 1])
SCORES = np.array([0.9, 0.8, 0.7, 0.7, 0.2, 0.1])
AMOUNTS = np.array([100.0, 5.0, 50.0, 5.0, 5.0, 50.0])


def test_budget_curve_accumulates_alerts_and_catches() -> None:
    curve = budget_curve(Y, SCORES, AMOUNTS, n_days=2)

    assert curve["alerts"].tolist() == [1, 2, 4, 5, 6]
    assert curve["fraud_caught"].tolist() == [1, 1, 2, 2, 3]
    assert curve["alerts_per_day"].tolist() == [0.5, 1.0, 2.0, 2.5, 3.0]
    assert curve["recall"].iloc[-1] == pytest.approx(1.0)
    assert curve["amount_recall"].iloc[0] == pytest.approx(0.5)


def test_budget_curve_treats_tied_scores_as_one_threshold() -> None:
    curve = budget_curve(Y, SCORES, AMOUNTS, n_days=1)

    assert curve["threshold"].tolist().count(0.7) == 1


def test_budget_curve_rejects_empty_period() -> None:
    with pytest.raises(ValueError, match="positive"):
        budget_curve(Y, SCORES, AMOUNTS, n_days=0)


def test_threshold_for_budget_stays_within_budget() -> None:
    curve = budget_curve(Y, SCORES, AMOUNTS, n_days=2)

    assert threshold_for_budget(curve, alerts_per_day=2.2) == pytest.approx(0.7)


def test_threshold_for_budget_raises_when_nothing_fits() -> None:
    curve = budget_curve(Y, SCORES, AMOUNTS, n_days=1)

    with pytest.raises(ValueError, match="no threshold"):
        threshold_for_budget(curve, alerts_per_day=0.5)


def test_operating_point_applies_a_fixed_threshold() -> None:
    point = operating_point(Y, SCORES, AMOUNTS, n_days=2, threshold=0.7, budget_per_day=2)

    assert point.alerts_per_day == pytest.approx(2.0)
    assert point.precision == pytest.approx(0.5)
    assert point.recall == pytest.approx(2 / 3)
    assert point.amount_recall == pytest.approx(0.75)


def test_operating_point_with_no_alerts_has_zero_precision() -> None:
    point = operating_point(Y, SCORES, AMOUNTS, n_days=1, threshold=2.0, budget_per_day=1)

    assert point.precision == 0.0
    assert point.recall == 0.0


def test_budget_table_renders_one_row_per_point() -> None:
    point = operating_point(Y, SCORES, AMOUNTS, n_days=2, threshold=0.7, budget_per_day=2)

    rows = budget_table([point], "LightGBM")

    assert rows == ["| LightGBM | 2 | 2.0 | 50% | 67% | 75% |"]
