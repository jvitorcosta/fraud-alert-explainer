import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from fraud_alert_explainer.model import (
    Split,
    fraud_scores,
    metrics,
    time_split,
    train_baseline,
    train_lightgbm,
)


@pytest.fixture(scope="module")
def split(features: pd.DataFrame) -> Split:
    return time_split(features, "2020-03-01", "2020-07-01")


@pytest.fixture(scope="module")
def lightgbm_model(split: Split) -> lgb.LGBMClassifier:
    return train_lightgbm(split.train, split.valid, n_estimators=50)


def test_time_split_keeps_periods_apart(split: Split) -> None:
    assert split.train["ts"].max() < split.valid["ts"].min()
    assert split.valid["ts"].max() < split.test["ts"].min()


def test_time_split_rejects_inverted_dates(features: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="before"):
        time_split(features, "2020-07-01", "2020-03-01")


def test_lightgbm_beats_chance_on_the_test_period(
    split: Split, lightgbm_model: lgb.LGBMClassifier
) -> None:
    result = metrics(split.test["is_fraud"], fraud_scores(lightgbm_model, split.test))

    assert result.pr_auc > 5 * result.fraud_rate


def test_baseline_scores_are_probabilities(split: Split) -> None:
    scores = fraud_scores(train_baseline(split.train), split.test)

    assert scores.shape == (len(split.test),)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_metrics_perfect_ranking_has_pr_auc_one() -> None:
    result = metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))

    assert result.pr_auc == pytest.approx(1.0)
    assert result.n_fraud == 2
    assert result.fraud_rate == pytest.approx(0.5)
