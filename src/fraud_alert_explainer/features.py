"""Turn raw transactions into model features.

Every card-history feature uses only that card's earlier transactions, so a row never sees
its own label or the future.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

HOUR_S = 3_600
DAY_S = 86_400
EARTH_RADIUS_KM = 6_371.0
# Spacing between cards in the sort key; far larger than the dataset's span in seconds.
_CARD_STRIDE_S = 10**10


@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    unit: str = ""


FEATURES = (
    Feature("amt", "transaction amount", "$"),
    Feature("amt_vs_card_avg", "amount compared with this card's average", "x"),
    Feature("category", "merchant category"),
    Feature("hour", "hour of day"),
    Feature("day_of_week", "day of week"),
    Feature("txns_prev_1h", "card transactions in the previous hour"),
    Feature("txns_prev_24h", "card transactions in the previous 24 hours"),
    Feature("amt_prev_24h", "card spend in the previous 24 hours", "$"),
    Feature("mins_since_prev_txn", "minutes since the card's previous transaction"),
    Feature("age", "cardholder age"),
    Feature("distance_km", "distance from the cardholder's home to the merchant", "km"),
    Feature("city_pop", "population of the cardholder's city"),
)
FEATURE_NAMES = [f.name for f in FEATURES]
FEATURES_BY_NAME = {f.name: f for f in FEATURES}
CATEGORICAL = ["category"]
TARGET = "is_fraud"


def haversine_km(
    lat1: npt.ArrayLike, lon1: npt.ArrayLike, lat2: npt.ArrayLike, lon2: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Great-circle distance in kilometres."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return np.asarray(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)), dtype=np.float64)


def prior_window_stats(
    card_codes: npt.NDArray[np.int64],
    seconds: npt.NDArray[np.int64],
    amounts: npt.NDArray[np.float64],
    window_s: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Count and sum of each card's earlier transactions within `window_s` seconds.

    Inputs must be sorted by card, then time. The current row is excluded; earlier rows with
    the same timestamp are included.
    """
    key = card_codes * _CARD_STRIDE_S + seconds
    if np.any(np.diff(key) < 0):
        raise ValueError("rows must be sorted by card, then time")
    position = np.arange(len(key))
    start = np.searchsorted(key, key - window_s, side="left")
    cumulative = np.concatenate([[0.0], np.cumsum(amounts)])
    count = (position - start).astype(np.int64)
    total = cumulative[position] - cumulative[start]
    return count, total


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one row per transaction: timestamp, features, label and amount, in time order."""
    df = raw.sort_values(["card_id", "ts"], kind="stable").reset_index(drop=True)
    card_codes = np.asarray(df["card_id"].astype("category").cat.codes, dtype=np.int64)
    seconds = np.asarray((df["ts"] - pd.Timestamp("1970-01-01")).dt.total_seconds(), dtype=np.int64)
    amounts = np.asarray(df["amt"], dtype=np.float64)

    out = pd.DataFrame({"ts": df["ts"], "card_id": df["card_id"], "amt": df["amt"]})
    out["category"] = df["category"].astype("category")
    out["hour"] = df["ts"].dt.hour.astype("int8")
    out["day_of_week"] = df["ts"].dt.day_of_week.astype("int8")
    out["age"] = ((df["ts"] - df["dob"]).dt.days / 365.25).round(1)
    out["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    out["city_pop"] = df["city_pop"]

    count_1h, _ = prior_window_stats(card_codes, seconds, amounts, HOUR_S)
    count_24h, sum_24h = prior_window_stats(card_codes, seconds, amounts, DAY_S)
    out["txns_prev_1h"] = count_1h
    out["txns_prev_24h"] = count_24h
    out["amt_prev_24h"] = sum_24h

    by_card = df.groupby("card_id", sort=False)
    out["mins_since_prev_txn"] = by_card["ts"].diff().dt.total_seconds() / 60
    prior_sum = by_card["amt"].cumsum() - df["amt"]
    prior_count = by_card.cumcount()
    prior_avg = prior_sum / prior_count.where(prior_count > 0)
    out["amt_vs_card_avg"] = df["amt"] / prior_avg

    out[TARGET] = df[TARGET]
    return out.sort_values("ts", kind="stable").reset_index(drop=True)
