"""Build a model-ready feature table from the cleaned, winner/loser-oriented match data."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Match-level columns describe the match itself, not either player, so they're
# copied over unchanged.
MATCH_LEVEL_COLUMNS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "indoor", "tourney_date", "match_num", "score", "best_of", "round",
    "minutes", "source_file", "is_retirement", "is_walkover", "is_default",
    "is_incomplete_match", "has_serve_stats", "has_match_num",
]

# For each of these, the source data has a "winner_<suffix>" and a
# "loser_<suffix>" column that need to become "player_1_<suffix>" / "player_2_<suffix>".
PLAYER_ATTRIBUTE_SUFFIXES = [
    "id", "seed", "entry", "name", "hand", "ht", "ioc", "age", "rank", "rank_points",
]

# Per-match serve/return stats mirror data_loader.STAT_COLUMNS but with
# "w_"/"l_" prefixes instead of "winner_"/"loser_".
PLAYER_STAT_SUFFIXES = [
    "ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced",
]


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

    for suffix in PLAYER_ATTRIBUTE_SUFFIXES:
        winner_col = f"winner_{suffix}"
        loser_col = f"loser_{suffix}"
        oriented[f"player_1_{suffix}"] = np.where(winner_is_player_1, df[winner_col], df[loser_col])
        oriented[f"player_2_{suffix}"] = np.where(winner_is_player_1, df[loser_col], df[winner_col])

    for suffix in PLAYER_STAT_SUFFIXES:
        winner_col = f"w_{suffix}"
        loser_col = f"l_{suffix}"
        oriented[f"player_1_{suffix}"] = np.where(winner_is_player_1, df[winner_col], df[loser_col])
        oriented[f"player_2_{suffix}"] = np.where(winner_is_player_1, df[loser_col], df[winner_col])

    oriented["player_1_won"] = winner_is_player_1.astype(int)

    return oriented
