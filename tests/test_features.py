import numpy as np
import pandas as pd
import pytest

from fraud_alert_explainer.features import (
    FEATURE_NAMES,
    build_features,
    haversine_km,
    prior_window_stats,
)


def test_haversine_km_matches_known_distance() -> None:
    # New York to Los Angeles is about 3,936 km.
    distance = haversine_km(40.7128, -74.0060, 34.0522, -118.2437)

    assert distance == pytest.approx(3_936, rel=0.01)


def test_prior_window_stats_counts_only_earlier_rows_in_window() -> None:
    cards = np.array([0, 0, 0, 0, 1], dtype=np.int64)
    seconds = np.array([0, 100, 3_000, 4_000, 3_000], dtype=np.int64)
    amounts = np.array([10.0, 20.0, 30.0, 40.0, 99.0])

    count, total = prior_window_stats(cards, seconds, amounts, window_s=3_600)

    assert count.tolist() == [0, 1, 2, 1, 0]
    assert total.tolist() == [0.0, 10.0, 30.0, 30.0, 0.0]


def test_prior_window_stats_rejects_unsorted_input() -> None:
    cards = np.array([0, 0], dtype=np.int64)
    seconds = np.array([10, 5], dtype=np.int64)

    with pytest.raises(ValueError, match="sorted"):
        prior_window_stats(cards, seconds, np.array([1.0, 1.0]), window_s=60)


def test_build_features_has_every_model_feature(features: pd.DataFrame) -> None:
    assert set(FEATURE_NAMES) <= set(features.columns)
    assert features["ts"].is_monotonic_increasing


def test_build_features_does_not_look_ahead(raw: pd.DataFrame) -> None:
    cutoff = raw["ts"].iloc[len(raw) // 2]
    full = build_features(raw)
    past_only = build_features(raw[raw["ts"] <= cutoff])

    shared = full[full["ts"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(shared, past_only.reset_index(drop=True))


def test_build_features_first_card_transaction_has_no_history(features: pd.DataFrame) -> None:
    first = features.sort_values("ts").groupby("card_id").head(1)

    assert (first["txns_prev_24h"] == 0).all()
    assert first["mins_since_prev_txn"].isna().all()
    assert first["amt_vs_card_avg"].isna().all()
