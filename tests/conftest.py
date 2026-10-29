from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_alert_explainer.features import build_features

CATEGORIES = ["grocery_pos", "shopping_net", "gas_transport", "misc_net"]


def make_raw(n: int = 4_000, n_cards: int = 40, seed: int = 0) -> pd.DataFrame:
    """Raw-schema transactions where fraud is large, online and at night."""
    rng = np.random.default_rng(seed)
    ts = pd.Timestamp("2019-01-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 730 * 86_400, n)), unit="s"
    )
    category = rng.choice(CATEGORIES, n)
    amt = rng.gamma(2.0, 30.0, n).round(2)
    night = (ts.hour >= 22) | (ts.hour < 4)
    is_fraud = (night & (category == "shopping_net") & (rng.random(n) < 0.6)) | (
        rng.random(n) < 0.005
    )
    amt = np.where(is_fraud, amt * 8, amt)
    return pd.DataFrame(
        {
            "ts": ts,
            "card_id": rng.integers(0, n_cards, n).astype("int64") + 4_000_000_000_000,
            "category": pd.Categorical(category),
            "amt": amt,
            "gender": pd.Categorical(rng.choice(["F", "M"], n)),
            "lat": rng.uniform(30, 45, n),
            "long": rng.uniform(-110, -75, n),
            "city_pop": rng.integers(500, 2_000_000, n),
            "dob": pd.Timestamp("1960-01-01")
            + pd.to_timedelta(rng.integers(0, 40 * 365, n), unit="D"),
            "merch_lat": rng.uniform(30, 45, n),
            "merch_long": rng.uniform(-110, -75, n),
            "is_fraud": is_fraud.astype("int8"),
        }
    )


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture(scope="session")
def features(raw: pd.DataFrame) -> pd.DataFrame:
    return build_features(raw)


@pytest.fixture
def raw_csv_dir(tmp_path: Path, raw: pd.DataFrame) -> Path:
    """The raw frame written out in Kaggle's CSV layout, split across the two files."""
    kaggle = raw.rename(columns={"ts": "trans_date_trans_time", "card_id": "cc_num"}).assign(
        first="Jane", last="Doe", merchant="fraud_Shop", trans_num="abc"
    )
    half = len(kaggle) // 2
    kaggle.iloc[:half].to_csv(tmp_path / "fraudTrain.csv")
    kaggle.iloc[half:].to_csv(tmp_path / "fraudTest.csv")
    return tmp_path
