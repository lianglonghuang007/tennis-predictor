"""Evaluation metrics for win-probability models: Brier score, log-loss, calibration."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


def evaluate_probabilities(y_true: pd.Series, y_prob: pd.Series) -> dict:
    """Return Brier score and log-loss for a set of predicted probabilities.

    Both are "lower is better," and both specifically reward well-calibrated
    probabilities, not just correct yes/no guesses — that matters more here
    than accuracy, since this project's actual output is a probability
    (e.g. "62% to win"), not a bet on who wins.
    """
    return {
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "n": len(y_true),
    }


def calibration_table(y_true: pd.Series, y_prob: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Bucket predictions into n_bins groups and compare predicted vs. actual win rate per bucket.

    strategy="quantile" sizes the buckets so each has roughly equal match
    counts, rather than equal-width probability ranges — since most
    predictions cluster near 0.5, equal-width bins would leave the extreme
    (near-0 or near-1) buckets almost empty and statistically meaningless.

    A perfectly calibrated model's two columns sit right on top of each
    other: among all matches predicted at ~70%, about 70% of them should
    have actually been won.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"predicted_win_rate": prob_pred, "actual_win_rate": prob_true})


def plot_calibration_curve(
    y_true: pd.Series, y_prob: pd.Series, n_bins: int = 10, label: str = "model", ax=None
):
    """Plot predicted vs. actual win rate, with a diagonal reference line for perfect calibration."""
    table = calibration_table(y_true, y_prob, n_bins=n_bins)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(table["predicted_win_rate"], table["actual_win_rate"], marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Actual win rate")
    ax.legend()
    return ax
