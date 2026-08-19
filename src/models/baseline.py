"""Bradley-Terry baseline: logistic regression on the Elo/rank/age gaps between the two players."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURE_COLUMNS = ["elo_pre_diff", "rank_diff", "age_diff"]


def prepare_xy(
    df: pd.DataFrame, feature_cols: list[str] = DEFAULT_FEATURE_COLUMNS
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract (X, y) for the baseline model, dropping rows with missing feature values.

    Logistic regression can't handle NaN inputs directly. rank_diff/age_diff
    can be NaN when a player's rank or age is missing in the source data (a
    known, documented gap) — those rows are dropped for THIS model only;
    LightGBM (which natively handles missing values) won't need to drop them.
    """
    clean = df.dropna(subset=feature_cols)
    X = clean[feature_cols]
    y = clean["player_1_won"]
    return X, y


def train_baseline(
    train_df: pd.DataFrame, feature_cols: list[str] = DEFAULT_FEATURE_COLUMNS
) -> Pipeline:
    """Fit a Bradley-Terry-style baseline: standardize the diff features, then logistic regression.

    Standardizing matters specifically because the default features live on
    very different scales (elo_pre_diff spans ~hundreds, rank_diff spans
    ~hundreds with a very different distribution, age_diff spans single-digit
    years) — without scaling, L2 regularization penalizes each coefficient
    unevenly just because of its raw units, not how informative it actually is.
    """
    X, y = prepare_xy(train_df, feature_cols)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression()),
    ])
    model.fit(X, y)
    return model


def predict_proba(
    model: Pipeline, df: pd.DataFrame, feature_cols: list[str] = DEFAULT_FEATURE_COLUMNS
) -> pd.Series:
    """Return P(player_1 wins) for every row in df with complete feature data."""
    X, _ = prepare_xy(df, feature_cols)
    probs = model.predict_proba(X)[:, 1]
    return pd.Series(probs, index=X.index, name="player_1_win_prob")
