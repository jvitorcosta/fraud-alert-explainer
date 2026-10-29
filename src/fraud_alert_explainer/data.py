"""Load the Sparkov simulated card-transaction dataset (Kaggle: kartik2112/fraud-detection)."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

RAW_FILES = ("fraudTrain.csv", "fraudTest.csv")

# Only what the features need. Names, street, job, merchant name and transaction id are
# dropped at read time: the data is simulated, but an alert model should not learn from them.
_DTYPES = {
    "cc_num": "int64",
    "category": "category",
    "amt": "float64",
    "gender": "category",
    "lat": "float64",
    "long": "float64",
    "city_pop": "int64",
    "merch_lat": "float64",
    "merch_long": "float64",
    "is_fraud": "int8",
}
_DATES = ("trans_date_trans_time", "dob")


class MissingDataError(FileNotFoundError):
    """Raised when the raw CSVs have not been downloaded yet."""


def read_csv(path: Path) -> pd.DataFrame:
    """Read one raw CSV, keeping only the columns the pipeline uses."""
    frame = pd.read_csv(
        path,
        usecols=[*_DTYPES, *_DATES],
        dtype=_DTYPES,
        parse_dates=list(_DATES),
    )
    return frame.rename(columns={"trans_date_trans_time": "ts", "cc_num": "card_id"})


def load_raw(raw_dir: Path, files: Sequence[str] = RAW_FILES) -> pd.DataFrame:
    """Load and concatenate the raw files, sorted by time.

    Kaggle ships the data pre-split; both files are combined so the split is ours and by time.
    """
    paths = [raw_dir / name for name in files]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise MissingDataError(
            f"Missing {', '.join(missing)}. Download https://www.kaggle.com/datasets/"
            f"kartik2112/fraud-detection and unzip it into {raw_dir}."
        )
    frames = [read_csv(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    for column in ("category", "gender"):
        combined[column] = combined[column].astype("category")
    return combined.sort_values("ts", kind="stable").reset_index(drop=True)
