"""Build each player's CURRENT state — as of right now, not as of some historical match — for
serving live predictions. Distinct from the training-time functions in elo.py/rolling_stats.py,
which deliberately compute PRE-match values to avoid leakage; a live prediction needs the opposite:
a player's full known history, including their most recent completed match.
"""

from __future__ import annotations

import pandas as pd

from src.features.elo import current_elo_ratings
from src.features.rolling_stats import ROLLING_STAT_NAMES, _add_per_match_stats, _build_player_match_log

# How many of a player's trailing matches to average over, matching the
# windows used at training time so live features are on the same scale the
# model was trained on.
ROLLING_WINDOWS = (10, 20, 50)


def current_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Each player's current serve/return/BP form per surface, averaged over their most recent matches.

    Reuses _build_player_match_log and _add_per_match_stats from
    rolling_stats.py (the reshape and per-match rate calculations are
    identical either way) but skips the shift(1) those apply — shifting
    exists specifically to exclude a match's own result from ITS OWN rolling
    average during training; a live snapshot has no "own match" to exclude,
    it just wants the plain trailing average including the most recent one.
    """
    log = _build_player_match_log(df)
    log = _add_per_match_stats(log)
    log = log.sort_values(["player_id", "surface", "date"], kind="stable")

    rows = []
    for (player_id, surface), group in log.groupby(["player_id", "surface"], sort=False):
        row = {"player_id": player_id, "surface": surface}
        for stat in ROLLING_STAT_NAMES:
            for window in ROLLING_WINDOWS:
                row[f"{stat}_last{window}"] = group[stat].tail(window).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def _latest_player_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Each player's most recently known name/hand/height/country/rank/age, plus their last match date."""
    winner_rows = pd.DataFrame({
        "player_id": df["winner_id"], "date": df["tourney_date"],
        "name": df["winner_name"], "hand": df["winner_hand"], "ht": df["winner_ht"],
        "ioc": df["winner_ioc"], "age": df["winner_age"],
        "rank": df["winner_rank"], "rank_points": df["winner_rank_points"],
    })
    loser_rows = pd.DataFrame({
        "player_id": df["loser_id"], "date": df["tourney_date"],
        "name": df["loser_name"], "hand": df["loser_hand"], "ht": df["loser_ht"],
        "ioc": df["loser_ioc"], "age": df["loser_age"],
        "rank": df["loser_rank"], "rank_points": df["loser_rank_points"],
    })
    combined = pd.concat([winner_rows, loser_rows], ignore_index=True)
    combined = combined.sort_values("date", kind="stable")
    latest = combined.groupby("player_id", sort=False).tail(1).reset_index(drop=True)
    return latest.rename(columns={"date": "last_match_date"})


def build_player_snapshots(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the two tables a live prediction needs: one row per player, and one row per (player, surface).

    Returns (players, player_surface_stats) — matching the eventual SQLite
    schema directly: player-level attributes (name, rank, age, ...) don't
    repeat per surface, but Elo and rolling form genuinely differ by surface.
    """
    players = _latest_player_attributes(df)

    elo = current_elo_ratings(df)
    rolling = current_rolling_stats(df)
    player_surface_stats = elo.merge(rolling, on=["player_id", "surface"], how="outer")

    # A player whose ONLY match on a surface was a walkover never gets
    # written into the Elo ratings dict (rating updates are skipped for
    # walkovers — no tennis was actually played), but still appears in the
    # rolling-stats log (which isn't walkover-aware) — a real edge case, not
    # a bug. Falls back to the same neutral starting rating used for any
    # other player with no rated history, rather than leaving it undefined.
    player_surface_stats["elo"] = player_surface_stats["elo"].fillna(1500.0)

    return players, player_surface_stats
