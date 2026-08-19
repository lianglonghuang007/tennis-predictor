"""LightGBM primary model: gradient-boosted trees over the full engineered feature
set, tuned via time-based cross-validation with Optuna, explained via SHAP.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

from src.features.build_features import PLAYER_STAT_SUFFIXES

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Identifiers, free text, or information only known AFTER the match is
# played — none of these are legitimate pre-match features.
EXCLUDED_COLUMNS = [
    "tourney_id", "tourney_name", "source_file", "score", "round", "tourney_date",
    "player_1_id", "player_1_name", "player_1_entry",
    "player_2_id", "player_2_name", "player_2_entry",
    "player_1_won",
    "minutes", "is_retirement", "is_walkover", "is_default", "is_incomplete_match",
    "has_serve_stats", "has_match_num",
]

# player_1_ace, player_2_bpFaced, etc: these describe what happened DURING
# this specific match (the box score) — only the *_lastN rolling versions
# (a player's form BEFORE this match) are safe to train on.
RAW_MATCH_STAT_COLUMNS = [f"player_{p}_{suffix}" for p in (1, 2) for suffix in PLAYER_STAT_SUFFIXES]

CATEGORICAL_COLUMNS = [
    "surface", "indoor", "tourney_level",
    "player_1_hand", "player_2_hand", "player_1_ioc", "player_2_ioc",
]


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (X, feature_cols): every column except identifiers, leakage, and raw match-outcome stats.

    Categorical columns are cast to pandas' "category" dtype — LightGBM's
    sklearn API detects category-dtype columns automatically and splits on
    them directly, without needing one-hot encoding the way logistic
    regression would.
    """
    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDED_COLUMNS and c not in RAW_MATCH_STAT_COLUMNS
    ]
    X = df[feature_cols].copy()
    for col in CATEGORICAL_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X, feature_cols


def time_series_cv_folds(df: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return n_splits (train_idx, val_idx) folds, each validating only on matches that
    come strictly AFTER everything in that fold's training portion.

    df must already be sorted chronologically. Unlike random K-fold, this
    never validates on a match older than one it trained on — the same
    point-in-time principle used throughout this project, now applied to
    model evaluation instead of feature construction.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    return list(splitter.split(df))


def tune_hyperparameters(train_df: pd.DataFrame, n_trials: int = 20, n_splits: int = 5) -> dict:
    """Run an Optuna search for LightGBM hyperparameters, scored by mean Brier score across time-based CV folds."""
    X, _ = select_features(train_df)
    y = train_df["player_1_won"]
    folds = time_series_cv_folds(train_df, n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        }
        fold_scores = []
        for train_idx, val_idx in folds:
            model = lgb.LGBMClassifier(**params, verbosity=-1)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            val_probs = model.predict_proba(X.iloc[val_idx])[:, 1]
            fold_scores.append(brier_score_loss(y.iloc[val_idx], val_probs))
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def train_lightgbm(train_df: pd.DataFrame, params: dict) -> lgb.LGBMClassifier:
    """Fit a final LightGBM model on the full training set using tuned hyperparameters."""
    X, _ = select_features(train_df)
    y = train_df["player_1_won"]
    model = lgb.LGBMClassifier(**params, verbosity=-1)
    model.fit(X, y)
    return model


def predict_proba(model: lgb.LGBMClassifier, df: pd.DataFrame) -> pd.Series:
    """Return P(player_1 wins) for every row in df."""
    X, _ = select_features(df)
    probs = model.predict_proba(X)[:, 1]
    return pd.Series(probs, index=X.index, name="player_1_win_prob")


def compute_shap_importance(model: lgb.LGBMClassifier, df: pd.DataFrame) -> pd.DataFrame:
    """Return features ranked by mean |SHAP value|: how much each feature moved predictions, on average.

    Unlike LightGBM's built-in "feature importance" (which just counts how
    often a feature was split on), SHAP attributes each individual
    prediction's deviation from the average back to specific features — a
    feature can be split on rarely but still matter a lot whenever it IS
    used, and SHAP captures that; a raw split-count doesn't.
    """
    X, feature_cols = select_features(df)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
