"""Compute surface-specific Elo ratings with time decay for inactive players."""

from __future__ import annotations

import pandas as pd

# Surfaces can be missing (NaN) in a small number of rows. NaN can't be used
# safely as a dict key (two NaNs are never equal to each other), so every
# missing surface is bucketed under this shared label instead.
UNKNOWN_SURFACE = "Unknown"


def _decayed_rating(
    ratings: dict, key: tuple, current_date: pd.Timestamp, initial_rating: float, half_life_days: float
) -> float:
    """Look up a player-surface's current rating, decayed toward the mean for time since their last match."""
    if key not in ratings:
        return initial_rating
    rating, last_date = ratings[key]
    days_inactive = (current_date - last_date).days
    if days_inactive <= 0:
        return rating
    # Half-life decay: the rating moves halfway back toward the mean every
    # `half_life_days` days of inactivity, so a long absence pulls a stale
    # rating back toward "average" instead of carrying full weight forever.
    decay_factor = 0.5 ** (days_inactive / half_life_days)
    return initial_rating + (rating - initial_rating) * decay_factor


def _run_elo_walk(
    df: pd.DataFrame, initial_rating: float, k: float, half_life_days: float
) -> tuple[list[float], list[float], dict[tuple, tuple[float, pd.Timestamp]]]:
    """Walk matches chronologically, updating Elo ratings; return each row's pre-match ratings
    AND the final ratings dict.

    Factored out of compute_elo_ratings so the training-time per-row output
    and a "current state" snapshot (used for live predictions, not present
    yet in this file) can share one implementation of the actual Elo math,
    instead of two copies that could quietly drift out of sync.
    """
    ratings: dict[tuple, tuple[float, pd.Timestamp]] = {}
    winner_elo_pre: list[float] = []
    loser_elo_pre: list[float] = []

    for row in df.itertuples():
        surface = row.surface if pd.notna(row.surface) else UNKNOWN_SURFACE
        date = row.tourney_date

        winner_key = (row.winner_id, surface)
        loser_key = (row.loser_id, surface)

        winner_rating = _decayed_rating(ratings, winner_key, date, initial_rating, half_life_days)
        loser_rating = _decayed_rating(ratings, loser_key, date, initial_rating, half_life_days)

        winner_elo_pre.append(winner_rating)
        loser_elo_pre.append(loser_rating)

        if row.is_walkover:
            continue

        # Standard Elo expected-score formula: probability the winner was
        # expected to win, given the rating gap. A 400-point gap means the
        # higher-rated player was expected to win about 10x more often.
        expected_winner = 1.0 / (1.0 + 10.0 ** ((loser_rating - winner_rating) / 400.0))
        expected_loser = 1.0 - expected_winner

        new_winner_rating = winner_rating + k * (1.0 - expected_winner)
        new_loser_rating = loser_rating + k * (0.0 - expected_loser)

        ratings[winner_key] = (new_winner_rating, date)
        ratings[loser_key] = (new_loser_rating, date)

    return winner_elo_pre, loser_elo_pre, ratings


def compute_elo_ratings(
    df: pd.DataFrame,
    initial_rating: float = 1500.0,
    k: float = 32.0,
    half_life_days: float = 180.0,
) -> pd.DataFrame:
    """Add winner_elo_pre / loser_elo_pre columns: each player's surface Elo rating going INTO the match.

    df must already be sorted chronologically (clean_matches guarantees this).
    Ratings are updated match-by-match in date order, so a match's "_pre"
    rating always reflects only information available before that match was
    played — this is what makes the feature safe to train on without leaking
    the future.

    Walkover matches (df["is_walkover"]) are skipped for rating UPDATES — no
    tennis was actually played — but a pre-match rating is still recorded for
    that row so every row has a value.
    """
    df = df.copy()
    winner_elo_pre, loser_elo_pre, _ = _run_elo_walk(df, initial_rating, k, half_life_days)
    df["winner_elo_pre"] = winner_elo_pre
    df["loser_elo_pre"] = loser_elo_pre
    return df


def current_elo_ratings(
    df: pd.DataFrame,
    initial_rating: float = 1500.0,
    k: float = 32.0,
    half_life_days: float = 180.0,
) -> pd.DataFrame:
    """Each player's surface Elo rating AS OF RIGHT NOW — after their last known match, not before it.

    A live prediction for a hypothetical upcoming match needs "what is this
    player's rating today," which is the FINAL state of the same walk
    compute_elo_ratings already does, not any row's pre-match value.
    """
    _, _, ratings = _run_elo_walk(df, initial_rating, k, half_life_days)
    rows = [
        {"player_id": player_id, "surface": surface, "elo": rating}
        for (player_id, surface), (rating, _last_date) in ratings.items()
    ]
    return pd.DataFrame(rows)
