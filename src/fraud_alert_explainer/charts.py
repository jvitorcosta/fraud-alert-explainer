"""README charts (static PNGs): PR curves, the alert-budget trade-off, global SHAP importance."""

from collections.abc import Mapping
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve

from fraud_alert_explainer.features import FEATURES_BY_NAME

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834")  # categorical slots 1-2: LightGBM, baseline


def _style(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, pad=12)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def pr_curves(
    y_true: npt.ArrayLike,
    scores: Mapping[str, npt.ArrayLike],
    pr_aucs: Mapping[str, float],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    for color, (name, s) in zip(SERIES, scores.items(), strict=False):
        precision, recall, _ = precision_recall_curve(y_true, s)
        ax.plot(
            recall,
            precision,
            color=color,
            linewidth=2,
            label=f"{name} (PR-AUC {pr_aucs[name]:.3f})",
        )
    _style(
        ax,
        "Precision vs recall, test period (Jul-Dec 2020)",
        "Recall (share of fraud caught)",
        "Precision (share of alerts that are fraud)",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, labelcolor=TEXT, loc="lower left")
    _save(fig, path)


def budget_tradeoff(
    curves: Mapping[str, pd.DataFrame], budget_per_day: float, path: Path, max_per_day: float = 200
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    for color, (name, curve) in zip(SERIES, curves.items(), strict=False):
        shown = curve[curve["alerts_per_day"] <= max_per_day]
        ax.plot(shown["alerts_per_day"], shown["recall"], color=color, linewidth=2, label=name)
    ax.axvline(budget_per_day, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(
        f"{budget_per_day:g} alerts/day",
        (budget_per_day, 0.04),
        xytext=(6, 0),
        textcoords="offset points",
        color=TEXT_SECONDARY,
        fontsize=9,
    )
    _style(
        ax,
        "Fraud caught vs analyst workload, test period",
        "Alerts per day",
        "Share of fraud cases caught",
    )
    ax.set_xlim(0, max_per_day)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, labelcolor=TEXT, loc="lower right")
    _save(fig, path)


def shap_importance(shap_values: npt.NDArray[np.float64], names: list[str], path: Path) -> None:
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)
    labels = [FEATURES_BY_NAME[names[i]].label for i in order]
    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=SURFACE)
    ax.barh(labels, mean_abs[order], color=SERIES[0], height=0.6)
    _style(
        ax,
        "Which features drive the score (mean |SHAP|, test alerts)",
        "Mean |SHAP value| (log-odds)",
        "",
    )
    ax.grid(axis="y", visible=False)
    _save(fig, path)
