from pathlib import Path

import pandas as pd
import pytest

from fraud_alert_explainer.data import MissingDataError, load_raw


def test_load_raw_combines_files_in_time_order(raw_csv_dir: Path, raw: pd.DataFrame) -> None:
    loaded = load_raw(raw_csv_dir)

    assert len(loaded) == len(raw)
    assert loaded["ts"].is_monotonic_increasing


def test_load_raw_drops_identifying_columns(raw_csv_dir: Path) -> None:
    loaded = load_raw(raw_csv_dir)

    assert not {"first", "last", "merchant", "trans_num"} & set(loaded.columns)
    assert {"ts", "card_id", "amt", "is_fraud"} <= set(loaded.columns)


def test_load_raw_names_the_download_when_files_are_missing(tmp_path: Path) -> None:
    with pytest.raises(MissingDataError, match="kartik2112/fraud-detection"):
        load_raw(tmp_path)
