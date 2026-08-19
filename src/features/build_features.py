"""Build a model-ready feature table from the cleaned, winner/loser-oriented match data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.elo import UNKNOWN_SURFACE, compute_elo_ratings
from src.features.rolling_stats import ROLLING_STAT_NAMES, add_rolling_stats

# Ordinal encoding of tournament round, earliest to latest. RR (round-robin,
# used at season-ending events) is mapped alongside R128 as "early stage" —
# an imperfect simplification, since round-robin matches at an elite
# invitational aren't really the same competitive tier as a first-round match
# at a random 250-level event, but ordinal stage-of-event is what this
# encodes, not field strength (Elo/rank already capture that separately).
# Labels not in this mapping (there's exactly 1 NaN round in the dataset)
# become NaN rather than crashing.
ROUND_ORDER = {
    "RR": 1, "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "BR": 6, "3rd/4th": 6,
    "F": 7,
}

# Match-level columns describe the match itself, not either player, so they're
# copied over unchanged.
MATCH_LEVEL_COLUMNS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "indoor", "tourney_date", "match_num", "score", "best_of", "round", "round_number",
    "minutes", "source_file", "is_retirement", "is_walkover", "is_default",
    "is_incomplete_match", "has_serve_stats", "has_match_num",
]

# For each of these, the source data has a "winner_<suffix>" and a
# "loser_<suffix>" column that need to become "player_1_<suffix>" / "player_2_<suffix>".
PLAYER_ATTRIBUTE_SUFFIXES = [
    "id", "seed", "entry", "name", "hand", "ht", "ioc", "age", "rank", "rank_points",
]

# Columns added by compute_elo_ratings, add_rolling_stats, add_days_rest, and
# add_h2h_stats all use the same winner_*/loser_* naming as the raw
# attributes above, so they're reoriented the exact same way.
DERIVED_FEATURE_SUFFIXES = (
    ["elo_pre", "days_rest", "h2h_wins_pre", "h2h_losses_pre"]
    + [f"{stat}_last{w}" for stat in ROLLING_STAT_NAMES for w in (10, 20, 50)]
)

# Per-match serve/return stats mirror data_loader.STAT_COLUMNS but with
# "w_"/"l_" prefixes instead of "winner_"/"loser_".
PLAYER_STAT_SUFFIXES = [
    "ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced",
]


def add_days_rest(df: pd.DataFrame) -> pd.DataFrame:
    """Add winner_days_rest / loser_days_rest: days since each player's previous match (any surface).

    Same sequential, chronological-walk pattern as compute_elo_ratings — "days
    since this player's own last match" only makes sense computed in date order.
    """
    df = df.copy()
    last_played: dict[str, pd.Timestamp] = {}
    winner_days_rest = []
    loser_days_rest = []

    for row in df.itertuples():
        date = row.tourney_date

        winner_days_rest.append(
            (date - last_played[row.winner_id]).days if row.winner_id in last_played else np.nan
        )
        loser_days_rest.append(
            (date - last_played[row.loser_id]).days if row.loser_id in last_played else np.nan
        )

        last_played[row.winner_id] = date
        last_played[row.loser_id] = date

    df["winner_days_rest"] = winner_days_rest
    df["loser_days_rest"] = loser_days_rest
    return df


def add_h2h_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add each side's surface-specific head-to-head record against THIS SPECIFIC opponent, before this match.

    Keyed by (unordered pair of player_ids, surface) so "Player A vs Player B
    on Clay" accumulates its own running win count, separate from their Hard
    or Grass record.
    """
    df = df.copy()
    # key -> {player_id: wins against the other player in this pair, on this surface}
    h2h: dict[tuple, dict[str, int]] = {}

    winner_wins_pre, winner_losses_pre = [], []
    loser_wins_pre, loser_losses_pre = [], []

    for row in df.itertuples():
        surface = row.surface if pd.notna(row.surface) else UNKNOWN_SURFACE
        # frozenset() makes the key order-independent — "A vs B" and "B vs A"
        # must look up the same record.
        key = (frozenset((row.winner_id, row.loser_id)), surface)
        record = h2h.get(key, {})

        winner_wins_pre.append(record.get(row.winner_id, 0))
        winner_losses_pre.append(record.get(row.loser_id, 0))
        loser_wins_pre.append(record.get(row.loser_id, 0))
        loser_losses_pre.append(record.get(row.winner_id, 0))

        record[row.winner_id] = record.get(row.winner_id, 0) + 1
        h2h[key] = record

    df["winner_h2h_wins_pre"] = winner_wins_pre
    df["winner_h2h_losses_pre"] = winner_losses_pre
    df["loser_h2h_wins_pre"] = loser_wins_pre
    df["loser_h2h_losses_pre"] = loser_losses_pre
    return df


def add_round_number(df: pd.DataFrame) -> pd.DataFrame:
    """Add round_number: an ordinal encoding of df["round"] (see ROUND_ORDER above)."""
    df = df.copy()
    df["round_number"] = df["round"].map(ROUND_ORDER)
    return df


def orient_players(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Reshape winner/loser rows into symmetric player_1/player_2 rows.

    The source data always puts the match winner in "winner_*" columns and the
    loser in "loser_*" columns. Fed straight into a model, that would let it
    cheat by learning "player_1 always wins" instead of anything about tennis
    — the label would be 100% correlated with column assignment. To prevent
    that, each match's winner and loser are randomly (but reproducibly, via
    `seed`) assigned to player_1 or player_2, and `player_1_won` becomes the
    actual label the model has to predict.
    """
    # Seeded RNG so re-running this produces the identical "random" assignment
    # every time — reproducibility matters for debugging and fair comparisons.
    rng = np.random.default_rng(seed)
    winner_is_player_1 = rng.random(len(df)) < 0.5

    oriented = pd.DataFrame(index=df.index)
    for col in MATCH_LEVEL_COLUMNS:
        oriented[col] = df[col]

    # winner_*/loser_* prefixed columns: raw attributes plus every derived
    # feature (Elo, rolling stats, H2H, days rest) share this prefix convention.
    for suffix in PLAYER_ATTRIBUTE_SUFFIXES + DERIVED_FEATURE_SUFFIXES:
        winner_col = f"winner_{suffix}"
        loser_col = f"loser_{suffix}"
        oriented[f"player_1_{suffix}"] = np.where(winner_is_player_1, df[winner_col], df[loser_col])
        oriented[f"player_2_{suffix}"] = np.where(winner_is_player_1, df[loser_col], df[winner_col])

    # w_*/l_* prefixed columns: raw per-match serve/return stats.
    for suffix in PLAYER_STAT_SUFFIXES:
        winner_col = f"w_{suffix}"
        loser_col = f"l_{suffix}"
        oriented[f"player_1_{suffix}"] = np.where(winner_is_player_1, df[winner_col], df[loser_col])
        oriented[f"player_2_{suffix}"] = np.where(winner_is_player_1, df[loser_col], df[winner_col])

    oriented["player_1_won"] = winner_is_player_1.astype(int)

    return oriented


def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pairwise comparison features on an already-oriented (player_1/player_2) table.

    Raw player_1_x / player_2_x values are useful on their own, but the
    RELATIVE gap between the two players is what a linear model (the
    Bradley-Terry baseline) needs as an explicit input — it can't discover
    "subtract these two columns" on its own the way LightGBM can.
    """
    df = df.copy()
    df["rank_diff"] = df["player_1_rank"] - df["player_2_rank"]
    df["age_diff"] = df["player_1_age"] - df["player_2_age"]
    df["elo_pre_diff"] = df["player_1_elo_pre"] - df["player_2_elo_pre"]

    # Opponent-adjusted serve "edge": how much better is this player's own
    # recent serve form than THIS SPECIFIC opponent's recent return form —
    # not just their serve form in a vacuum, averaged over whoever they
    # happened to face recently.
    for window in (10, 20, 50):
        df[f"player_1_serve_edge_last{window}"] = (
            df[f"player_1_serve_win_pct_last{window}"] - df[f"player_2_return_win_pct_last{window}"]
        )
        df[f"player_2_serve_edge_last{window}"] = (
            df[f"player_2_serve_win_pct_last{window}"] - df[f"player_1_return_win_pct_last{window}"]
        )
    return df


def time_based_split(
    df: pd.DataFrame, val_start: str = "2024-07-01", test_start: str = "2025-07-01"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test by tourney_date cutoffs, not randomly.

    A random split would let the model train on matches from AFTER some of
    its validation/test matches — no real deployment could ever do that; on a
    live app you only ever have the past to predict the future from.
    """
    val_start = pd.Timestamp(val_start)
    test_start = pd.Timestamp(test_start)
    train = df[df["tourney_date"] < val_start]
    val = df[(df["tourney_date"] >= val_start) & (df["tourney_date"] < test_start)]
    test = df[df["tourney_date"] >= test_start]
    return train, val, test


def build_feature_table(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Run the full Week 1 feature pipeline: Elo, rolling stats, H2H, days rest,
    round encoding, player re-orientation, and pairwise diff features.
    """
    df = compute_elo_ratings(df)
    df = add_rolling_stats(df)
    df = add_days_rest(df)
    df = add_h2h_stats(df)
    df = add_round_number(df)

    oriented = orient_players(df, seed=seed)
    oriented = add_diff_features(oriented)
    return oriented
