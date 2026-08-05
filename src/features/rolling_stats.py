"""Compute trailing (last-N-match), surface-specific serve/return form features."""

from __future__ import annotations

import pandas as pd

# Reused from elo.py so a missing surface is bucketed the same way everywhere.
from src.features.elo import UNKNOWN_SURFACE

# The four per-match stats we track a rolling trailing average of. Each is a
# rate (0 to 1), so averaging them across matches is meaningful.
ROLLING_STAT_NAMES = ["serve_win_pct", "return_win_pct", "bp_conversion", "bp_saved_pct"]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Elementwise division that returns NaN (not inf or an error) when the denominator is 0."""
    safe_denominator = denominator.where(denominator != 0)
    return numerator / safe_denominator


def _build_player_match_log(df: pd.DataFrame) -> pd.DataFrame:
    """Turn one row per match into two rows per match, tagged with "own" vs "opponent" stats.

    Rolling averages are inherently per-PLAYER (e.g. "Djokovic's last 10
    matches"), but the source data is organized per-MATCH with winner/loser
    columns. This reshape puts each player's own match history in its own
    rows, sortable by that player's own timeline, which a rolling window needs.
    """
    winner_rows = pd.DataFrame({
        "match_id": df.index,
        "player_id": df["winner_id"],
        "surface": df["surface"],
        "date": df["tourney_date"],
        "is_winner_row": True,
        "own_svpt": df["w_svpt"], "own_1stWon": df["w_1stWon"], "own_2ndWon": df["w_2ndWon"],
        "own_bpSaved": df["w_bpSaved"], "own_bpFaced": df["w_bpFaced"],
        "opp_svpt": df["l_svpt"], "opp_1stWon": df["l_1stWon"], "opp_2ndWon": df["l_2ndWon"],
        "opp_bpSaved": df["l_bpSaved"], "opp_bpFaced": df["l_bpFaced"],
    })
    loser_rows = pd.DataFrame({
        "match_id": df.index,
        "player_id": df["loser_id"],
        "surface": df["surface"],
        "date": df["tourney_date"],
        "is_winner_row": False,
        "own_svpt": df["l_svpt"], "own_1stWon": df["l_1stWon"], "own_2ndWon": df["l_2ndWon"],
        "own_bpSaved": df["l_bpSaved"], "own_bpFaced": df["l_bpFaced"],
        "opp_svpt": df["w_svpt"], "opp_1stWon": df["w_1stWon"], "opp_2ndWon": df["w_2ndWon"],
        "opp_bpSaved": df["w_bpSaved"], "opp_bpFaced": df["w_bpFaced"],
    })
    log = pd.concat([winner_rows, loser_rows], ignore_index=True)
    log["surface"] = log["surface"].fillna(UNKNOWN_SURFACE)
    return log


def _add_per_match_stats(log: pd.DataFrame) -> pd.DataFrame:
    """Add serve_win_pct, return_win_pct, bp_conversion, bp_saved_pct for each (match, player) row.

    serve_win_pct: fraction of this player's own service points they won.
    return_win_pct: fraction of the opponent's service points this player won
        (this player's return performance).
    bp_conversion: fraction of the opponent's faced break points that this
        player (the returner) converted — offensive break-point performance.
    bp_saved_pct: fraction of this player's own faced break points that they
        saved — defensive break-point performance.
    """
    log = log.copy()
    log["serve_win_pct"] = _safe_divide(log["own_1stWon"] + log["own_2ndWon"], log["own_svpt"])
    log["return_win_pct"] = _safe_divide(
        log["opp_svpt"] - log["opp_1stWon"] - log["opp_2ndWon"], log["opp_svpt"]
    )
    log["bp_conversion"] = _safe_divide(log["opp_bpFaced"] - log["opp_bpSaved"], log["opp_bpFaced"])
    log["bp_saved_pct"] = _safe_divide(log["own_bpSaved"], log["own_bpFaced"])
    return log


def _add_rolling_features(log: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    """Add "<stat>_last<N>" columns: each player's own trailing average on that surface.

    The critical trick for point-in-time correctness: .shift(1) BEFORE
    .rolling(...) — this excludes the current match's own result from its own
    rolling average, so a match's feature only reflects matches strictly
    before it, never the match's own outcome.
    """
    # kind="stable" keeps same-day matches in their original (match_id) order.
    log = log.sort_values(["player_id", "surface", "date", "match_id"], kind="stable").reset_index(drop=True)

    for stat in ROLLING_STAT_NAMES:
        shifted = log.groupby(["player_id", "surface"], sort=False)[stat].shift(1)
        for window in windows:
            col_name = f"{stat}_last{window}"
            # min_periods=1 lets a player's 2nd-ever match get a 1-match
            # average instead of waiting for a full window, while their
            # genuine 1st-ever match (0 prior matches) still correctly gets NaN.
            log[col_name] = (
                shifted.groupby([log["player_id"], log["surface"]], sort=False)
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
    return log


def add_rolling_stats(df: pd.DataFrame, windows: tuple[int, ...] = (10, 20, 50)) -> pd.DataFrame:
    """Attach winner_/loser_ trailing serve/return/BP form columns to a match DataFrame.

    Runs the full rolling-stats pipeline: reshape to a per-player log, compute
    per-match rate stats, compute trailing rolling averages, then merge the
    results back onto the original match-level df as winner_*/loser_* columns
    (mirroring elo.py's winner_elo_pre/loser_elo_pre naming, so both get
    reoriented into player_1_*/player_2_* together in the assembly step).
    """
    df = df.copy()
    log = _build_player_match_log(df)
    log = _add_per_match_stats(log)
    log = _add_rolling_features(log, windows)

    rolling_cols = [f"{stat}_last{w}" for stat in ROLLING_STAT_NAMES for w in windows]

    # Split back into winner/loser halves, indexed by match_id so each aligns
    # exactly with df's own row index.
    winner_log = log[log["is_winner_row"]].set_index("match_id")
    loser_log = log[~log["is_winner_row"]].set_index("match_id")

    for col in rolling_cols:
        df[f"winner_{col}"] = winner_log[col]
        df[f"loser_{col}"] = loser_log[col]

    return df
