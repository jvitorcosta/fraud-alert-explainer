"""Time-aware split, the two models, and their headline metrics."""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_alert_explainer.features import CATEGORICAL, FEATURE_NAMES, TARGET

SEED = 42


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def time_split(features: pd.DataFrame, valid_start: str, test_start: str) -> Split:
    """Split by timestamp: train < valid_start <= valid < test_start <= test."""
    valid_from, test_from = pd.Timestamp(valid_start), pd.Timestamp(test_start)
    if valid_from >= test_from:
        raise ValueError("valid_start must be before test_start")
    ts = features["ts"]
    return Split(
        train=features[ts < valid_from],
        valid=features[(ts >= valid_from) & (ts < test_from)],
        test=features[ts >= test_from],
    )


def xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, npt.NDArray[np.int8]]:
    """Model inputs and labels."""
    return frame[FEATURE_NAMES], frame[TARGET].to_numpy(np.int8)


def train_lightgbm(
    train: pd.DataFrame, valid: pd.DataFrame, *, n_estimators: int = 2_000
) -> lgb.LGBMClassifier:
    """Gradient-boosted trees, early-stopped on validation average precision."""
    x_train, y_train = xy(train)
    x_valid, y_valid = xy(valid)
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_X=(x_valid,),
        eval_y=(y_valid,),
        eval_metric="average_precision",
        categorical_feature=CATEGORICAL,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def train_baseline(train: pd.DataFrame) -> Pipeline:
    """Logistic regression on the same features: one-hot category, scaled numerics."""
    numeric = [name for name in FEATURE_NAMES if name not in CATEGORICAL]
    preprocess = ColumnTransformer(
        [
            ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ]
    )
    model = make_pipeline(preprocess, LogisticRegression(max_iter=2_000))
    x_train, y_train = xy(train)
    model.fit(x_train, y_train)
    return model


def fraud_scores(
    model: lgb.LGBMClassifier | Pipeline, frame: pd.DataFrame
) -> npt.NDArray[np.float64]:
    """Predicted fraud probability per row."""
    x, _ = xy(frame)
    proba = np.asarray(model.predict_proba(x), dtype=np.float64)
    return proba[:, 1]


@dataclass(frozen=True)
class Metrics:
    pr_auc: float
    roc_auc: float
    fraud_rate: float
    n_rows: int
    n_fraud: int


def metrics(y_true: npt.ArrayLike, scores: npt.ArrayLike) -> Metrics:
    """PR-AUC (average precision) is the headline; ROC-AUC flatters rare-event models."""
    y = np.asarray(y_true)
    return Metrics(
        pr_auc=float(average_precision_score(y, scores)),
        roc_auc=float(roc_auc_score(y, scores)),
        fraud_rate=float(y.mean()),
        n_rows=len(y),
        n_fraud=int(y.sum()),
    )
